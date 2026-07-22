#!/usr/bin/env python3
"""Run the frozen semantic-answer and cross-mechanism steering confirmation.

Unlike v2, this study never maps answers onto A/B proxy tokens.  It measures
the actual next-token contrast between the two scientific outcome words and
applies all three mechanism directions at each target family's fixed layer.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_vendor_jlens"))
sys.path.insert(0, str(ROOT / "scripts"))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from run_lens import _DTYPES, load_model  # noqa: E402
from run_jacobian_steering_pilot import (  # noqa: E402
    add_residual,
    atomic_json,
    capture_clean,
    format_chat,
    patched_logits,
    sha256,
    single_token,
)
from run_ensemble_steering_v2 import (  # noqa: E402
    direction_fit_hidden,
    family_token_ids,
    orthogonal_random,
    semantic_direct_direction,
    semantic_jacobian_direction,
)


def prompt_rows(family: dict) -> list[dict]:
    """Expand each physical condition into two answer-word presentation orders."""
    output = []
    positive = family["outcome_positive"]
    negative = family["outcome_negative"]
    for prompt in family["confirmation_prompts"]:
        for order, words in [
            ("positive-first", [positive, negative]),
            ("negative-first", [negative, positive]),
        ]:
            output.append({
                "family_id": family["family_id"],
                "condition_id": prompt["condition_id"],
                "prompt_id": (
                    f"{family['family_id']}--confirm--{prompt['condition_id']}--{order}"
                ),
                "presentation_order": order,
                "presented_words": words,
                "expected_outcome": prompt["expected_outcome"],
                "regime": prompt["regime"],
                "user": (
                    f"{prompt['stem']} Answer exactly one lowercase word from this "
                    f"ordered pair: {words[0]}, {words[1]}."
                ),
            })
    return output


def continuation_tokens(tokenizer, word: str) -> list[int]:
    """Tokenize the exact lowercase continuation without a leading space."""
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"empty continuation tokenization for {word!r}")
    return [int(token_id) for token_id in token_ids]


@torch.no_grad()
def answer_sequence_log_probability(
    model,
    input_ids: torch.Tensor,
    next_logits: torch.Tensor,
    layer: int,
    delta: torch.Tensor | None,
    answer_tokens: list[int],
) -> float:
    """Teacher-force one exact answer while patching only the final prompt token."""
    if len(answer_tokens) == 1:
        return float(torch.log_softmax(next_logits, dim=-1)[answer_tokens[0]])
    prefix = torch.tensor(
        answer_tokens[:-1], device=input_ids.device, dtype=input_ids.dtype
    ).unsqueeze(0)
    extended = torch.cat([input_ids, prefix], dim=1)
    prompt_final_position = int(input_ids.shape[1] - 1)
    final_layer = model.n_layers - 1
    if delta is None:
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(extended)
    else:
        with add_residual(model, layer, prompt_final_position, delta):
            with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
                model.forward(extended)
    start = prompt_final_position
    stop = start + len(answer_tokens)
    residual = recorder.activations[final_layer][0, start:stop].detach()
    logits = model.unembed(residual).float().cpu()
    logp = torch.log_softmax(logits, dim=-1)
    token_index = torch.tensor(answer_tokens, dtype=torch.long)
    positions = torch.arange(len(answer_tokens), dtype=torch.long)
    return float(logp[positions, token_index].sum())


def semantic_distribution_summary(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    positive_tokens: list[int],
    negative_tokens: list[int],
    positive_sequence_logp: float,
    negative_sequence_logp: float,
    positive_word: str,
    negative_word: str,
    tokenizer,
) -> dict:
    """Summarize the exact scientific answer-token contrast and output integrity."""
    logp = torch.log_softmax(logits, dim=-1)
    clean_logp = torch.log_softmax(clean_logits, dim=-1)
    clean_p = clean_logp.exp()
    kl = float((clean_p * (clean_logp - logp)).sum())
    top_id = int(logits.argmax())
    first_ids = {positive_tokens[0], negative_tokens[0]}
    choice = (
        positive_word
        if positive_sequence_logp >= negative_sequence_logp
        else negative_word
    )
    return {
        "positive_log_odds": float(
            positive_sequence_logp - negative_sequence_logp
        ),
        "positive_sequence_log_probability": positive_sequence_logp,
        "negative_sequence_log_probability": negative_sequence_logp,
        "sequence_probability_positive": float(math.exp(positive_sequence_logp)),
        "sequence_probability_negative": float(math.exp(negative_sequence_logp)),
        "choice": choice,
        "global_top_token": tokenizer.decode(
            [top_id], clean_up_tokenization_spaces=False
        ).strip(),
        "global_top_token_id": top_id,
        "global_top_is_valid_choice": top_id in first_ids,
        "probability_positive": float(logp[positive_tokens[0]].exp()),
        "probability_negative": float(logp[negative_tokens[0]].exp()),
        "valid_choice_probability": float(
            sum(logp[token_id].exp() for token_id in first_ids)
        ),
        "kl_clean_to_intervened": kl,
    }


def initialize_payload(
    manifest: dict,
    manifest_path: Path,
    output_path: Path,
    model,
    args,
) -> dict:
    if args.resume and output_path.exists():
        payload = json.loads(output_path.read_text())
        if payload["provenance"]["manifest_sha256"] != sha256(manifest_path):
            raise SystemExit("resume manifest hash does not match existing output")
        return payload
    runner_path = Path(__file__).resolve()
    helper_paths = [
        ROOT / "scripts/run_ensemble_steering_v2.py",
        ROOT / "scripts/run_jacobian_steering_pilot.py",
    ]
    return {
        "study_id": manifest["study_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256(manifest_path),
            "runner": str(runner_path.relative_to(ROOT)),
            "runner_sha256": sha256(runner_path),
            "helper_sha256": {
                str(path.relative_to(ROOT)): sha256(path) for path in helper_paths
            },
            "model": manifest["model"],
            "model_revision": manifest["model_revision"],
            "dtype": args.dtype,
            "device": str(model.input_device),
            "torch": torch.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "lenses": [
                {"seed": seed, "path": path, "sha256": sha256(ROOT / path)}
                for seed, path in enumerate(manifest["lens_paths"])
            ],
        },
        "token_inventory": {},
        "direction_diagnostics": {},
        "preflight": [],
        "intervention_rows": [],
    }


def cosine_matrix(vectors: list[torch.Tensor]) -> list[list[float]]:
    matrix = np.eye(len(vectors), dtype=float)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            value = float(torch.dot(vectors[i], vectors[j]))
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="experiments/semantic-steering-v3-preregistration.json",
    )
    parser.add_argument(
        "--output", default="experiments/semantic-steering-v3_raw.json"
    )
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--prompt-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_path = (ROOT / args.manifest).resolve()
    output_path = (ROOT / args.output).resolve()
    manifest = json.loads(manifest_path.read_text())
    families = manifest["families"]
    family_by_id = {family["family_id"]: family for family in families}
    if args.family:
        unknown = set(args.family) - set(family_by_id)
        if unknown:
            raise SystemExit(f"unknown families: {sorted(unknown)}")
        target_families = [family for family in families if family["family_id"] in args.family]
    else:
        target_families = families

    model = load_model(
        manifest["model"],
        dtype=_DTYPES[args.dtype],
        device=args.device,
        revision=manifest["model_revision"],
    )
    lenses = [jlens.JacobianLens.load(ROOT / path) for path in manifest["lens_paths"]]
    payload = initialize_payload(manifest, manifest_path, output_path, model, args)

    # Validate every answer and direction token before constructing any direction.
    token_inventory = {}
    answer_ids = {}
    concept_ids = {}
    for family in families:
        family_id = family["family_id"]
        positive_concepts, negative_concepts = family_token_ids(model.tokenizer, family)
        positive_answer = continuation_tokens(
            model.tokenizer, family["outcome_positive"]
        )
        negative_answer = continuation_tokens(
            model.tokenizer, family["outcome_negative"]
        )
        concept_ids[family_id] = (positive_concepts, negative_concepts)
        answer_ids[family_id] = (positive_answer, negative_answer)
        token_inventory[family_id] = {
            "concept_positive": dict(zip(family["concept_positive"], positive_concepts)),
            "concept_negative": dict(zip(family["concept_negative"], negative_concepts)),
            "outcome_positive": {
                "word": family["outcome_positive"], "token_ids": positive_answer,
                "decoded": model.tokenizer.decode(
                    positive_answer, clean_up_tokenization_spaces=False
                ),
            },
            "outcome_negative": {
                "word": family["outcome_negative"], "token_ids": negative_answer,
                "decoded": model.tokenizer.decode(
                    negative_answer, clean_up_tokenization_spaces=False
                ),
            },
        }
    payload["token_inventory"] = token_inventory
    atomic_json(output_path, payload)

    # Build every source mechanism at each target layer.  This is the frozen,
    # layer-matched cross-mechanism design; no result-dependent selection occurs.
    target_layers = sorted({int(family["fixed_layer"]) for family in target_families})
    jacobian_directions: dict[tuple[int, str, int], torch.Tensor] = {}
    direct_directions: dict[tuple[int, str], torch.Tensor] = {}
    for layer in target_layers:
        layer_diag = payload["direction_diagnostics"].setdefault(str(layer), {})
        seed0_sources = []
        for source_family in families:
            source_id = source_family["family_id"]
            fit_hidden = direction_fit_hidden(model, source_family, [layer])[layer]
            positive_ids, negative_ids = concept_ids[source_id]
            source_vectors = []
            fit_cosines = {}
            for lens_seed, lens in enumerate(lenses):
                direction, cosines = semantic_jacobian_direction(
                    model, lens, layer, fit_hidden, positive_ids, negative_ids
                )
                jacobian_directions[(layer, source_id, lens_seed)] = direction
                source_vectors.append(direction)
                fit_cosines[str(lens_seed)] = cosines
            direct, direct_cosines = semantic_direct_direction(
                model, fit_hidden, positive_ids, negative_ids
            )
            direct_directions[(layer, source_id)] = direct
            layer_diag[source_id] = {
                "fit_prompt_cosines_to_mean_by_lens_seed": fit_cosines,
                "lens_seed_direction_cosine_matrix": cosine_matrix(source_vectors),
                "direct_fit_prompt_cosines_to_mean": direct_cosines,
                "seed0_jacobian_cosine_with_direct": float(
                    torch.dot(source_vectors[0], direct)
                ),
            }
            seed0_sources.append(source_vectors[0])
        layer_diag["source_family_order"] = [family["family_id"] for family in families]
        layer_diag["seed0_cross_mechanism_cosine_matrix"] = cosine_matrix(seed0_sources)
        atomic_json(output_path, payload)

    completed = {
        (
            row["prompt_id"], row["method"], row.get("source_family_id"),
            row.get("lens_seed"), row.get("random_seed"),
            float(row["dose_percent"]),
        )
        for row in payload["intervention_rows"]
    }
    preflight_done = {row["prompt_id"] for row in payload["preflight"]}
    doses = [float(value) for value in manifest["design"]["doses_percent_residual_norm"]]
    random_seeds = [int(value) for value in manifest["design"]["random_seeds"]]
    requested_prompt_ids = set(args.prompt_id)
    known_prompt_ids = {
        row["prompt_id"] for family in target_families for row in prompt_rows(family)
    }
    if requested_prompt_ids:
        unknown = requested_prompt_ids - known_prompt_ids
        if unknown:
            raise SystemExit(f"unknown prompt ids: {sorted(unknown)}")

    checkpoint_counter = 0
    for target_index, family in enumerate(target_families):
        target_id = family["family_id"]
        layer = int(family["fixed_layer"])
        positive_tokens, negative_tokens = answer_ids[target_id]
        own_seed0 = jacobian_directions[(layer, target_id, 0)]
        own_direct = direct_directions[(layer, target_id)]
        randoms = {
            random_seed: orthogonal_random(
                own_seed0,
                own_direct,
                20260716 + 10000 * list(family_by_id).index(target_id) + random_seed,
            )
            for random_seed in random_seeds
        }
        random_diag = {
            str(seed): {
                "cosine_with_own_seed0": float(torch.dot(direction, own_seed0)),
                "cosine_with_own_direct": float(torch.dot(direction, own_direct)),
            }
            for seed, direction in randoms.items()
        }
        payload["direction_diagnostics"][str(layer)].setdefault(
            "random_controls_by_target", {}
        )[target_id] = random_diag
        atomic_json(output_path, payload)

        directions: list[tuple[str, str | None, int | None, int | None, str, torch.Tensor]] = []
        for source_family in families:
            source_id = source_family["family_id"]
            control_type = "own" if source_id == target_id else "wrong-mechanism"
            for lens_seed in range(len(lenses)):
                directions.append((
                    "semantic_jacobian", source_id, lens_seed, None, control_type,
                    jacobian_directions[(layer, source_id, lens_seed)],
                ))
        directions.append((
            "semantic_direct", target_id, None, None, "direct",
            direct_directions[(layer, target_id)],
        ))
        directions.extend(
            ("random", None, None, seed, "random", direction)
            for seed, direction in randoms.items()
        )

        print(f"[target={target_id}] layer={layer} directions={len(directions)}", flush=True)
        rows = prompt_rows(family)
        if requested_prompt_ids:
            rows = [row for row in rows if row["prompt_id"] in requested_prompt_ids]
        for prompt_index, prompt in enumerate(rows, start=1):
            text = format_chat(model.tokenizer, prompt["user"])
            input_ids = model.encode(text, max_length=512)
            hidden_by_layer, clean_logits = capture_clean(model, text, [layer])
            hidden = hidden_by_layer[layer]
            residual_norm = float(hidden.norm())
            clean_positive_logp = answer_sequence_log_probability(
                model, input_ids, clean_logits, layer, None, positive_tokens
            )
            clean_negative_logp = answer_sequence_log_probability(
                model, input_ids, clean_logits, layer, None, negative_tokens
            )
            clean_summary = semantic_distribution_summary(
                clean_logits, clean_logits, positive_tokens, negative_tokens,
                clean_positive_logp, clean_negative_logp,
                family["outcome_positive"], family["outcome_negative"], model.tokenizer,
            )
            if prompt["prompt_id"] not in preflight_done:
                payload["preflight"].append({
                    **prompt,
                    "fixed_layer": layer,
                    "outcome_positive": family["outcome_positive"],
                    "outcome_negative": family["outcome_negative"],
                    "clean_expected_correct": (
                        None if prompt["expected_outcome"] is None
                        else clean_summary["choice"] == prompt["expected_outcome"]
                    ),
                    "clean": clean_summary,
                })
                preflight_done.add(prompt["prompt_id"])
                atomic_json(output_path, payload)

            for method, source_id, lens_seed, random_seed, control_type, direction in directions:
                for dose in doses:
                    key = (
                        prompt["prompt_id"], method, source_id, lens_seed,
                        random_seed, dose,
                    )
                    if key in completed:
                        continue
                    delta = direction * dose / 100.0 * residual_norm
                    logits = clean_logits if dose == 0.0 else patched_logits(
                        model, text, layer, delta
                    )
                    if dose == 0.0:
                        positive_logp = clean_positive_logp
                        negative_logp = clean_negative_logp
                    else:
                        positive_logp = answer_sequence_log_probability(
                            model, input_ids, logits, layer, delta, positive_tokens
                        )
                        negative_logp = answer_sequence_log_probability(
                            model, input_ids, logits, layer, delta, negative_tokens
                        )
                    summary = semantic_distribution_summary(
                        logits, clean_logits, positive_tokens, negative_tokens,
                        positive_logp, negative_logp,
                        family["outcome_positive"], family["outcome_negative"],
                        model.tokenizer,
                    )
                    payload["intervention_rows"].append({
                        **{field: prompt[field] for field in [
                            "family_id", "condition_id", "prompt_id",
                            "presentation_order", "presented_words",
                            "expected_outcome", "regime",
                        ]},
                        "target_family_id": target_id,
                        "source_family_id": source_id,
                        "control_type": control_type,
                        "method": method,
                        "lens_seed": lens_seed,
                        "random_seed": random_seed,
                        "layer": layer,
                        "dose_percent": dose,
                        "residual_norm": residual_norm,
                        "delta_norm": float(delta.norm()),
                        **summary,
                    })
                    completed.add(key)
                    checkpoint_counter += 1
                    if checkpoint_counter % 25 == 0:
                        atomic_json(output_path, payload)
            atomic_json(output_path, payload)
            print(
                f"  {prompt_index:02d}/{len(rows):02d} {prompt['condition_id']} "
                f"{prompt['presentation_order']}: clean={clean_summary['choice']} "
                f"pair_mass={clean_summary['valid_choice_probability']:.3f}",
                flush=True,
            )

    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output_path, payload)
    print(
        f"complete: {len(payload['preflight'])} prompts, "
        f"{len(payload['intervention_rows'])} rows -> {output_path.relative_to(ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
