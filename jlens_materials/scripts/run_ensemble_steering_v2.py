#!/usr/bin/env python3
"""Calibrate and confirm transferable semantic-ensemble Jacobian steering."""

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
from run_lens import _DTYPES, load_model  # noqa: E402
from run_jacobian_steering_pilot import (  # noqa: E402
    atomic_json,
    capture_clean,
    distribution_summary,
    format_chat,
    patched_logits,
    random_direction,
    sha256,
    single_token,
    token_score,
)


def ensemble_score(model, normalized: torch.Tensor,
                   positive_ids: list[int], negative_ids: list[int]) -> torch.Tensor:
    positive = torch.stack([token_score(model, normalized, token_id)
                            for token_id in positive_ids])
    negative = torch.stack([token_score(model, normalized, token_id)
                            for token_id in negative_ids])
    return torch.logsumexp(positive, dim=0) - torch.logsumexp(negative, dim=0)


def semantic_jacobian_direction(model, lens, layer: int,
                                hidden_states: list[torch.Tensor],
                                positive_ids: list[int],
                                negative_ids: list[int]) -> tuple[torch.Tensor, list[float]]:
    """Average prompt-normalized J^T semantic gradients into one family direction."""
    J = lens.jacobians[layer].float().cpu()
    directions = []
    for hidden in hidden_states:
        transported = (J @ hidden.float().cpu()).detach().requires_grad_(True)
        normed = model._final_norm(
            transported.to(model._lm_head.weight.device, model._lm_head.weight.dtype)
        )
        score = ensemble_score(model, normed, positive_ids, negative_ids)
        grad_transport = torch.autograd.grad(score, transported)[0].float().cpu()
        direction = J.T @ grad_transport
        directions.append(direction / direction.norm().clamp_min(1e-12))
    stacked = torch.stack(directions)
    mean = stacked.mean(dim=0)
    mean = mean / mean.norm().clamp_min(1e-12)
    cosines = [float(torch.dot(direction, mean)) for direction in directions]
    return mean, cosines


def semantic_direct_direction(model, hidden_states: list[torch.Tensor],
                              positive_ids: list[int],
                              negative_ids: list[int]) -> tuple[torch.Tensor, list[float]]:
    directions = []
    for hidden in hidden_states:
        state = hidden.detach().float().to(model._lm_head.weight.device).requires_grad_(True)
        normed = model._final_norm(state.to(model._lm_head.weight.dtype))
        score = ensemble_score(model, normed, positive_ids, negative_ids)
        direction = torch.autograd.grad(score, state)[0].float().cpu()
        directions.append(direction / direction.norm().clamp_min(1e-12))
    stacked = torch.stack(directions)
    mean = stacked.mean(dim=0)
    mean = mean / mean.norm().clamp_min(1e-12)
    cosines = [float(torch.dot(direction, mean)) for direction in directions]
    return mean, cosines


def orthogonal_random(reference: torch.Tensor, direct: torch.Tensor, seed: int) -> torch.Tensor:
    direction = random_direction(reference, seed)
    direct_orthogonal = direct - torch.dot(direct, reference) * reference
    if direct_orthogonal.norm() > 1e-8:
        direct_orthogonal = direct_orthogonal / direct_orthogonal.norm()
        direction = direction - torch.dot(direction, direct_orthogonal) * direct_orthogonal
    return direction / direction.norm().clamp_min(1e-12)


def prompt_rows(family: dict, split: str) -> list[dict]:
    output = []
    for prompt in family[f"{split}_prompts"]:
        for order in ["ab", "ba"]:
            if order == "ab":
                option_a = family["outcome_positive"]
                option_b = family["outcome_negative"]
            else:
                option_a = family["outcome_negative"]
                option_b = family["outcome_positive"]
            output.append({
                "family_id": family["family_id"],
                "condition_id": prompt["condition_id"],
                "prompt_id": f"{family['family_id']}--{split}--{prompt['condition_id']}--{order}",
                "order": order,
                "option_a": option_a,
                "option_b": option_b,
                "expected_outcome": prompt["expected_outcome"],
                "user": (
                    f"{prompt['stem']} A: {option_a}. B: {option_b}. "
                    "Answer exactly A or B."
                ),
            })
    return output


def answer_mapping(family: dict, prompt: dict) -> tuple[str, str | None]:
    positive_answer = "A" if prompt["option_a"] == family["outcome_positive"] else "B"
    expected_answer = None
    if prompt["expected_outcome"] is not None:
        expected_answer = "A" if prompt["option_a"] == prompt["expected_outcome"] else "B"
    return positive_answer, expected_answer


def direction_fit_hidden(model, family: dict, layers: list[int]) -> dict[int, list[torch.Tensor]]:
    result = {layer: [] for layer in layers}
    for user in family["direction_fit_prompts"]:
        text = format_chat(model.tokenizer, user)
        hidden, _ = capture_clean(model, text, layers)
        for layer in layers:
            result[layer].append(hidden[layer])
    return result


def family_token_ids(tokenizer, family: dict) -> tuple[list[int], list[int]]:
    positive = [single_token(tokenizer, word) for word in family["concept_positive"]]
    negative = [single_token(tokenizer, word) for word in family["concept_negative"]]
    return positive, negative


def summarize_clean(model, clean_logits: torch.Tensor, answer_a_id: int,
                    answer_b_id: int, positive_answer: str) -> dict:
    return distribution_summary(
        clean_logits, clean_logits, answer_a_id, answer_b_id,
        positive_answer, model.tokenizer,
    )


def calibration_complete(payload: dict, family_id: str, layers: list[int],
                         n_prompts: int) -> bool:
    expected = len(layers) * n_prompts * 3
    observed = sum(row["family_id"] == family_id for row in payload["calibration_rows"])
    return observed == expected


def select_layer(payload: dict, family_id: str, layers: list[int]) -> tuple[int, list[dict]]:
    rows = [row for row in payload["calibration_rows"] if row["family_id"] == family_id]
    endpoints = []
    for layer in layers:
        layer_rows = [row for row in rows if row["layer"] == layer]
        grouped: dict[str, dict[float, float]] = {}
        for row in layer_rows:
            grouped.setdefault(row["prompt_id"], {})[float(row["dose_percent"])] = float(
                row["positive_log_odds"]
            )
        values = [doses[4.0] - doses[-4.0] for doses in grouped.values()]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        endpoints.append({
            "layer": layer,
            "mean_endpoint": mean,
            "population_std": std,
            "selection_score": mean - std,
            "positive_prompt_orders": int(sum(value > 0 for value in values)),
            "n_prompt_orders": len(values),
            "prompt_order_endpoints": values,
        })
    selected = sorted(endpoints, key=lambda row: (-row["selection_score"], row["layer"]))[0]
    return int(selected["layer"]), endpoints


def initialize_payload(manifest: dict, manifest_path: Path, output_path: Path,
                       model, lenses: list, args) -> dict:
    if args.resume and output_path.exists():
        payload = json.loads(output_path.read_text())
        if payload["provenance"]["manifest_sha256"] != sha256(manifest_path):
            raise SystemExit("resume manifest hash does not match output")
        return payload
    return {
        "study_id": manifest["study_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256(manifest_path),
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
        "calibration_rows": [],
        "layer_selection": {},
        "direction_diagnostics": {},
        "confirmation_preflight": [],
        "confirmation_rows": [],
    }


def run_calibration(model, lens, manifest: dict, families: list[dict], payload: dict,
                    output_path: Path, answer_a_id: int, answer_b_id: int) -> None:
    layers = [int(layer) for layer in manifest["candidate_layers"]]
    doses = [float(dose) for dose in manifest["calibration"]["doses_percent_residual_norm"]]
    completed = {
        (row["prompt_id"], int(row["layer"]), float(row["dose_percent"]))
        for row in payload["calibration_rows"]
    }
    for family in families:
        family_id = family["family_id"]
        print(f"[calibration] {family_id}", flush=True)
        positive_ids, negative_ids = family_token_ids(model.tokenizer, family)
        fit_hidden = direction_fit_hidden(model, family, layers)
        directions = {}
        diagnostics = {}
        for layer in layers:
            direction, cosines = semantic_jacobian_direction(
                model, lens, layer, fit_hidden[layer], positive_ids, negative_ids
            )
            directions[layer] = direction
            diagnostics[str(layer)] = {"fit_prompt_cosines_to_mean": cosines}
        payload["direction_diagnostics"].setdefault(family_id, {})["calibration_seed0"] = diagnostics
        atomic_json(output_path, payload)

        prompts = prompt_rows(family, "calibration")
        for prompt in prompts:
            text = format_chat(model.tokenizer, prompt["user"])
            hidden_by_layer, clean_logits = capture_clean(model, text, layers)
            positive_answer, expected_answer = answer_mapping(family, prompt)
            clean_summary = summarize_clean(
                model, clean_logits, answer_a_id, answer_b_id, positive_answer
            )
            for layer in layers:
                residual_norm = float(hidden_by_layer[layer].norm())
                for dose in doses:
                    key = (prompt["prompt_id"], layer, dose)
                    if key in completed:
                        continue
                    delta = directions[layer] * dose / 100.0 * residual_norm
                    logits = clean_logits if dose == 0.0 else patched_logits(
                        model, text, layer, delta
                    )
                    summary = distribution_summary(
                        logits, clean_logits, answer_a_id, answer_b_id,
                        positive_answer, model.tokenizer,
                    )
                    payload["calibration_rows"].append({
                        **{key: prompt[key] for key in [
                            "family_id", "condition_id", "prompt_id", "order",
                            "option_a", "option_b", "expected_outcome",
                        ]},
                        "expected_answer": expected_answer,
                        "layer": layer,
                        "method": "ensemble_jacobian",
                        "lens_seed": 0,
                        "dose_percent": dose,
                        "residual_norm": residual_norm,
                        "delta_norm": float(delta.norm()),
                        "clean": clean_summary,
                        **summary,
                    })
                    completed.add(key)
                    atomic_json(output_path, payload)

        if not calibration_complete(payload, family_id, layers, len(prompts)):
            raise RuntimeError(f"incomplete calibration rows for {family_id}")
        selected, layer_rows = select_layer(payload, family_id, layers)
        payload["layer_selection"][family_id] = {
            "selected_layer": selected,
            "candidate_results": layer_rows,
        }
        atomic_json(output_path, payload)
        print(f"  selected layer {selected}", flush=True)


def run_confirmation(model, lenses: list, manifest: dict, families: list[dict],
                     payload: dict, output_path: Path, answer_a_id: int,
                     answer_b_id: int) -> None:
    doses = [float(dose) for dose in manifest["confirmation"]["doses_percent_residual_norm"]]
    random_seeds = [int(seed) for seed in manifest["confirmation"]["random_seeds"]]
    completed = {
        (row["prompt_id"], row["method"], row.get("lens_seed"),
         row.get("random_seed"), float(row["dose_percent"]))
        for row in payload["confirmation_rows"]
    }
    preflight_done = {row["prompt_id"] for row in payload["confirmation_preflight"]}

    for family_index, family in enumerate(families):
        family_id = family["family_id"]
        if family_id not in payload["layer_selection"]:
            raise SystemExit(f"calibration selection missing for {family_id}")
        layer = int(payload["layer_selection"][family_id]["selected_layer"])
        print(f"[confirmation] {family_id} layer={layer}", flush=True)
        positive_ids, negative_ids = family_token_ids(model.tokenizer, family)
        fit_hidden = direction_fit_hidden(model, family, [layer])[layer]

        jacobian_directions = []
        lens_diagnostics = {}
        for lens_seed, lens in enumerate(lenses):
            direction, cosines = semantic_jacobian_direction(
                model, lens, layer, fit_hidden, positive_ids, negative_ids
            )
            jacobian_directions.append(direction)
            lens_diagnostics[str(lens_seed)] = {"fit_prompt_cosines_to_mean": cosines}
        direct, direct_cosines = semantic_direct_direction(
            model, fit_hidden, positive_ids, negative_ids
        )
        randoms = {
            seed: orthogonal_random(
                jacobian_directions[0], direct,
                20260716 + 10000 * family_index + seed,
            )
            for seed in random_seeds
        }
        lens_cosines = np.eye(len(jacobian_directions)).tolist()
        for i in range(len(jacobian_directions)):
            for j in range(i + 1, len(jacobian_directions)):
                value = float(torch.dot(jacobian_directions[i], jacobian_directions[j]))
                lens_cosines[i][j] = value
                lens_cosines[j][i] = value
        payload["direction_diagnostics"].setdefault(family_id, {})["confirmation"] = {
            "layer": layer,
            "lens_fit_prompt_cosines": lens_diagnostics,
            "lens_direction_cosine_matrix": lens_cosines,
            "direct_fit_prompt_cosines_to_mean": direct_cosines,
            "random_orthogonal_to": ["lens_seed0", "direct_component_orthogonal_to_lens_seed0"],
        }
        atomic_json(output_path, payload)

        directions = [
            ("ensemble_jacobian", seed, None, direction)
            for seed, direction in enumerate(jacobian_directions)
        ]
        directions.append(("ensemble_direct", None, None, direct))
        directions.extend(
            ("random", None, seed, direction) for seed, direction in randoms.items()
        )

        for prompt in prompt_rows(family, "confirmation"):
            prompt_id = prompt["prompt_id"]
            text = format_chat(model.tokenizer, prompt["user"])
            hidden_by_layer, clean_logits = capture_clean(model, text, [layer])
            hidden = hidden_by_layer[layer]
            residual_norm = float(hidden.norm())
            positive_answer, expected_answer = answer_mapping(family, prompt)
            clean_summary = summarize_clean(
                model, clean_logits, answer_a_id, answer_b_id, positive_answer
            )
            accepted = bool(clean_summary["global_top_is_valid_choice"])
            if prompt_id not in preflight_done:
                payload["confirmation_preflight"].append({
                    **{key: prompt[key] for key in [
                        "family_id", "condition_id", "prompt_id", "order",
                        "option_a", "option_b", "expected_outcome", "user",
                    ]},
                    "positive_answer": positive_answer,
                    "expected_answer": expected_answer,
                    "accepted": accepted,
                    "clean": clean_summary,
                })
                preflight_done.add(prompt_id)
                atomic_json(output_path, payload)
            if not accepted:
                print(f"  excluded {prompt_id}: top={clean_summary['global_top_token']}", flush=True)
                continue

            for method, lens_seed, random_seed, direction in directions:
                for dose in doses:
                    key = (prompt_id, method, lens_seed, random_seed, dose)
                    if key in completed:
                        continue
                    delta = direction * dose / 100.0 * residual_norm
                    logits = clean_logits if dose == 0.0 else patched_logits(
                        model, text, layer, delta
                    )
                    summary = distribution_summary(
                        logits, clean_logits, answer_a_id, answer_b_id,
                        positive_answer, model.tokenizer,
                    )
                    payload["confirmation_rows"].append({
                        **{field: prompt[field] for field in [
                            "family_id", "condition_id", "prompt_id", "order",
                            "option_a", "option_b", "expected_outcome",
                        ]},
                        "expected_answer": expected_answer,
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
                    atomic_json(output_path, payload)
            print(f"  complete {prompt_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="experiments/ensemble-steering-v2-preregistration.json"
    )
    parser.add_argument("--output", default="experiments/ensemble-steering-v2_raw.json")
    parser.add_argument("--stage", choices=["calibrate", "confirm", "all"], default="all")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_path = (ROOT / args.manifest).resolve()
    output_path = (ROOT / args.output).resolve()
    manifest = json.loads(manifest_path.read_text())
    families = manifest["families"]
    if args.family:
        wanted = set(args.family)
        families = [family for family in families if family["family_id"] in wanted]
        missing = wanted - {family["family_id"] for family in families}
        if missing:
            raise SystemExit(f"unknown families: {sorted(missing)}")

    model = load_model(
        manifest["model"], dtype=_DTYPES[args.dtype], device=args.device,
        revision=manifest["model_revision"],
    )
    lenses = [jlens.JacobianLens.load(ROOT / path) for path in manifest["lens_paths"]]
    payload = initialize_payload(manifest, manifest_path, output_path, model, lenses, args)
    answer_a_id = single_token(model.tokenizer, "A", prefer_plain=True)
    answer_b_id = single_token(model.tokenizer, "B", prefer_plain=True)

    token_inventory = {}
    for family in families:
        positive, negative = family_token_ids(model.tokenizer, family)
        token_inventory[family["family_id"]] = {
            "positive": dict(zip(family["concept_positive"], positive)),
            "negative": dict(zip(family["concept_negative"], negative)),
        }
    payload["token_inventory"] = token_inventory
    atomic_json(output_path, payload)

    if args.stage in {"calibrate", "all"}:
        run_calibration(
            model, lenses[int(manifest["calibration"]["lens_seed"])], manifest,
            families, payload, output_path, answer_a_id, answer_b_id,
        )
    if args.stage in {"confirm", "all"}:
        run_confirmation(
            model, lenses, manifest, families, payload, output_path,
            answer_a_id, answer_b_id,
        )
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output_path, payload)
    print(f"complete -> {output_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
