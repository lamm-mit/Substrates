#!/usr/bin/env python3
"""Run localized, contrastive Jacobian steering without touching the paper.

The intervention is intentionally narrower than the legacy global coordinate
swap: one residual position, one source layer, and a dose expressed as a
fraction of the clean residual norm.  The primary direction differentiates a
positive-minus-negative concept score through Gemma's exact final norm and a
fitted Jacobian map.  Results are checkpointed after every prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_vendor_jlens"))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from run_lens import _DTYPES, load_model  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def single_token(tokenizer, text: str, *, prefer_plain: bool = False) -> int:
    forms = [text, " " + text] if prefer_plain else [" " + text, text]
    forms += [f.capitalize() for f in forms]
    for form in forms:
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    raise ValueError(f"{text!r} has no usable one-token form")


def format_chat(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def capture_clean(model, text: str, layers: list[int]) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    input_ids = model.encode(text, max_length=512)
    capture = sorted(set(layers + [model.n_layers - 1]))
    with ActivationRecorder(model.layers, at=capture) as recorder:
        model.forward(input_ids)
    hidden = {
        layer: recorder.activations[layer][0, -1].detach().float().cpu()
        for layer in layers
    }
    final = recorder.activations[model.n_layers - 1][0, -1:].detach()
    logits = model.unembed(final).float()[0].cpu()
    return hidden, logits


def token_score(model, normalized: torch.Tensor, token_id: int) -> torch.Tensor:
    weight = model._lm_head.weight[token_id].to(normalized.device, normalized.dtype)
    raw = (normalized * weight).sum()
    softcap = getattr(model, "_logit_softcap", None)
    if softcap is not None:
        raw = softcap * torch.tanh(raw / softcap)
    return raw


def exact_jacobian_direction(model, lens, layer: int, hidden: torch.Tensor,
                             positive_id: int, negative_id: int) -> torch.Tensor:
    """Gradient of the J-lens token contrast, including exact final norm."""
    J = lens.jacobians[layer].float().cpu()
    transported = (J @ hidden.float().cpu()).detach().requires_grad_(True)
    normed = model._final_norm(
        transported.to(model._lm_head.weight.device, model._lm_head.weight.dtype)
    )
    score = token_score(model, normed, positive_id) - token_score(model, normed, negative_id)
    grad_transport = torch.autograd.grad(score, transported)[0].float().cpu()
    direction = J.T @ grad_transport
    return direction / direction.norm().clamp_min(1e-12)


def direct_direction(model, hidden: torch.Tensor, positive_id: int,
                     negative_id: int) -> torch.Tensor:
    """Matched direct-unembedding concept direction with exact final norm."""
    state = hidden.detach().float().to(
        model._lm_head.weight.device
    ).requires_grad_(True)
    normed = model._final_norm(state.to(model._lm_head.weight.dtype))
    score = token_score(model, normed, positive_id) - token_score(model, normed, negative_id)
    direction = torch.autograd.grad(score, state)[0].float().cpu()
    return direction / direction.norm().clamp_min(1e-12)


def random_direction(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(reference.shape, generator=generator)
    direction -= torch.dot(direction, reference) * reference
    return direction / direction.norm().clamp_min(1e-12)


@contextmanager
def add_residual(model, layer: int, position: int, delta: torch.Tensor):
    def hook(_module, _inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        patched = tensor.clone()
        patched[:, position, :] += delta.to(patched.device, patched.dtype)
        if torch.is_tensor(output):
            return patched
        return (patched, *output[1:])

    handle = model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def patched_logits(model, text: str, layer: int, delta: torch.Tensor) -> torch.Tensor:
    input_ids = model.encode(text, max_length=512)
    final = model.n_layers - 1
    with add_residual(model, layer, -1, delta):
        with ActivationRecorder(model.layers, at=[final]) as recorder:
            model.forward(input_ids)
    residual = recorder.activations[final][0, -1:].detach()
    return model.unembed(residual).float()[0].cpu()


def distribution_summary(logits: torch.Tensor, clean_logits: torch.Tensor,
                         answer_a_id: int, answer_b_id: int,
                         positive_answer: str, tokenizer) -> dict:
    logp = torch.log_softmax(logits, dim=-1)
    clean_logp = torch.log_softmax(clean_logits, dim=-1)
    clean_p = clean_logp.exp()
    kl = float((clean_p * (clean_logp - logp)).sum())
    top_id = int(logits.argmax())
    answer_ids = {"A": answer_a_id, "B": answer_b_id}
    negative_answer = "B" if positive_answer == "A" else "A"
    choice = "A" if logits[answer_a_id] >= logits[answer_b_id] else "B"
    return {
        "positive_log_odds": float(
            logits[answer_ids[positive_answer]] - logits[answer_ids[negative_answer]]
        ),
        "choice": choice,
        "global_top_token": tokenizer.decode(
            [top_id], clean_up_tokenization_spaces=False
        ).strip(),
        "global_top_token_id": top_id,
        "global_top_is_valid_choice": top_id in answer_ids.values(),
        "probability_a": float(logp[answer_a_id].exp()),
        "probability_b": float(logp[answer_b_id].exp()),
        "valid_choice_probability": float(
            logp[answer_a_id].exp() + logp[answer_b_id].exp()
        ),
        "kl_clean_to_intervened": kl,
    }


def flatten_cases(manifest: dict) -> list[tuple[dict, dict]]:
    return [
        (case, prompt)
        for case in manifest["cases"]
        for prompt in case["prompts"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="experiments/jacobian-steering-pilot-v1-preregistration.json",
    )
    parser.add_argument(
        "--output",
        default="experiments/jacobian-steering-pilot-v1_raw.json",
    )
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--methods",
        default="jacobian_exact",
        help="comma list: jacobian_exact,direct,random",
    )
    parser.add_argument("--prompt-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_path = (ROOT / args.manifest).resolve()
    output_path = (ROOT / args.output).resolve()
    manifest = json.loads(manifest_path.read_text())
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    unsupported = set(methods) - {"jacobian_exact", "direct", "random"}
    if unsupported:
        raise SystemExit(f"unsupported methods: {sorted(unsupported)}")

    selected = flatten_cases(manifest)
    if args.prompt_id:
        wanted = set(args.prompt_id)
        selected = [(case, prompt) for case, prompt in selected if prompt["prompt_id"] in wanted]
        missing = wanted - {prompt["prompt_id"] for _, prompt in selected}
        if missing:
            raise SystemExit(f"unknown prompt ids: {sorted(missing)}")

    model = load_model(
        manifest["model"],
        dtype=_DTYPES[args.dtype],
        device=args.device,
        revision=manifest["model_revision"],
    )
    lenses = [jlens.JacobianLens.load(ROOT / path) for path in manifest["lens_paths"]]
    lens_meta = [
        {
            "seed": seed,
            "path": path,
            "sha256": sha256(ROOT / path),
        }
        for seed, path in enumerate(manifest["lens_paths"])
    ]

    if args.resume and output_path.exists():
        payload = json.loads(output_path.read_text())
        if payload["provenance"]["manifest_sha256"] != sha256(manifest_path):
            raise SystemExit("resume manifest hash does not match existing output")
        payload["methods"] = list(dict.fromkeys(payload.get("methods", []) + methods))
    else:
        payload = {
            "study_id": manifest["study_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "manifest": str(manifest_path.relative_to(ROOT)),
                "manifest_sha256": sha256(manifest_path),
                "model": manifest["model"],
                "model_revision": manifest["model_revision"],
                "dtype": args.dtype,
                "device": str(model.input_device),
                "lenses": lens_meta,
                "torch": torch.__version__,
                "python": sys.version,
                "platform": platform.platform(),
            },
            "methods": methods,
            "doses_percent_residual_norm": manifest["intervention"][
                "doses_percent_residual_norm"
            ],
            "preflight": [],
            "interventions": [],
        }

    completed = {
        (row["prompt_id"], row["method"], row.get("lens_seed"), row.get("random_seed"), row["dose_percent"])
        for row in payload["interventions"]
    }
    preflight_done = {row["prompt_id"] for row in payload["preflight"]}
    answer_a_id = single_token(model.tokenizer, "A", prefer_plain=True)
    answer_b_id = single_token(model.tokenizer, "B", prefer_plain=True)
    doses = manifest["intervention"]["doses_percent_residual_norm"]

    for case, prompt in selected:
        prompt_id = prompt["prompt_id"]
        print(f"[{prompt_id}] clean preflight and steering")
        text = format_chat(model.tokenizer, prompt["user"])
        layer = int(case["primary_layer"])
        hidden_by_layer, clean_logits = capture_clean(model, text, [layer])
        hidden = hidden_by_layer[layer]
        positive_answer = (
            "A" if prompt["option_a"] == case["outcome_positive"] else "B"
        )
        expected_answer = None
        if prompt["expected_outcome"] is not None:
            expected_answer = (
                "A" if prompt["option_a"] == prompt["expected_outcome"] else "B"
            )
        clean_summary = distribution_summary(
            clean_logits,
            clean_logits,
            answer_a_id,
            answer_b_id,
            positive_answer,
            model.tokenizer,
        )
        clean_accepted = clean_summary["global_top_is_valid_choice"] and (
            expected_answer is None or clean_summary["choice"] == expected_answer
        )
        if prompt_id not in preflight_done:
            payload["preflight"].append({
                "case_id": case["case_id"],
                "prompt_id": prompt_id,
                "condition": prompt["condition"],
                "inferential": prompt["inferential"],
                "option_a": prompt["option_a"],
                "option_b": prompt["option_b"],
                "positive_answer": positive_answer,
                "expected_answer": expected_answer,
                "accepted": bool(clean_accepted),
                "clean": clean_summary,
                "prompt": prompt["user"],
            })
            preflight_done.add(prompt_id)
            atomic_json(output_path, payload)
        if not clean_accepted:
            print(f"  excluded: clean choice={clean_summary['choice']} expected={expected_answer}")
            continue

        positive_id = single_token(model.tokenizer, case["concept_positive"])
        negative_id = single_token(model.tokenizer, case["concept_negative"])
        directions: list[tuple[str, int | None, int | None, torch.Tensor]] = []
        if "jacobian_exact" in methods:
            for lens_seed, lens in enumerate(lenses):
                directions.append((
                    "jacobian_exact",
                    lens_seed,
                    None,
                    exact_jacobian_direction(
                        model, lens, layer, hidden, positive_id, negative_id
                    ),
                ))
        if "direct" in methods:
            directions.append((
                "direct", None, None,
                direct_direction(model, hidden, positive_id, negative_id),
            ))
        if "random" in methods:
            reference = directions[0][3] if directions else direct_direction(
                model, hidden, positive_id, negative_id
            )
            for random_seed in range(3):
                directions.append((
                    "random", None, random_seed,
                    random_direction(reference, 20260716 + 1000 * random_seed + len(prompt_id)),
                ))

        residual_norm = float(hidden.norm())
        for method, lens_seed, random_seed, direction in directions:
            for dose in doses:
                key = (prompt_id, method, lens_seed, random_seed, float(dose))
                if key in completed:
                    continue
                delta = direction * (float(dose) / 100.0) * residual_norm
                logits = clean_logits if float(dose) == 0.0 else patched_logits(
                    model, text, layer, delta
                )
                summary = distribution_summary(
                    logits,
                    clean_logits,
                    answer_a_id,
                    answer_b_id,
                    positive_answer,
                    model.tokenizer,
                )
                payload["interventions"].append({
                    "case_id": case["case_id"],
                    "prompt_id": prompt_id,
                    "condition": prompt["condition"],
                    "inferential": prompt["inferential"],
                    "method": method,
                    "lens_seed": lens_seed,
                    "random_seed": random_seed,
                    "layer": layer,
                    "dose_percent": float(dose),
                    "residual_norm": residual_norm,
                    "delta_norm": float(delta.norm()),
                    "concept_positive": case["concept_positive"],
                    "concept_negative": case["concept_negative"],
                    "positive_answer": positive_answer,
                    "expected_answer": expected_answer,
                    **summary,
                })
                completed.add(key)
                atomic_json(output_path, payload)
                print(
                    f"  {method} seed={lens_seed if lens_seed is not None else random_seed} "
                    f"dose={dose:+.2f}% logodds={summary['positive_log_odds']:+.3f} "
                    f"choice={summary['choice']} valid={summary['valid_choice_probability']:.3f} "
                    f"KL={summary['kl_clean_to_intervened']:.3g}"
                )

    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output_path, payload)
    print(f"complete -> {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
