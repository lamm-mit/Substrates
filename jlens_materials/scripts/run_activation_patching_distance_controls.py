#!/usr/bin/env python3
"""Run frozen state-distance-matched falsification controls for activation patching."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_vendor_jlens"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_lens import _DTYPES, load_model  # noqa: E402
from run_jacobian_steering_pilot import atomic_json, format_chat, sha256  # noqa: E402
from run_semantic_steering_v3 import continuation_tokens  # noqa: E402
from run_counterfactual_activation_patching import (  # noqa: E402
    capture_prompt,
    donor_ids,
    expanded_prompts,
    patched_log_odds,
)


DEFAULT_PROTOCOL = (
    "experiments/candidate-activation-patching-2026-07-16/falsification_protocol.json"
)
DEFAULT_OUTPUT = (
    "experiments/candidate-activation-patching-2026-07-16/distance_controls_raw.json"
)


def validate(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise RuntimeError(f"fingerprint mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    false_path = (ROOT / args.protocol).resolve()
    false = json.loads(false_path.read_text())
    primary_path = ROOT / false["primary_protocol"]
    raw_path = ROOT / false["primary_raw"]
    validate(primary_path, false["primary_protocol_sha256"])
    validate(raw_path, false["primary_raw_sha256"])
    primary = json.loads(primary_path.read_text())
    source_path = ROOT / primary["inputs"]["prompt_manifest"]
    validate(source_path, primary["inputs"]["prompt_manifest_sha256"])
    source = json.loads(source_path.read_text())
    grain = next(
        row for row in source["families"] if row["family_id"] == "grain-size-strengthening"
    )
    prompts = expanded_prompts(grain)
    prompt_by_id = {row["prompt_id"]: row for row in prompts}
    index = {
        (row["pair_id"], row["relation"], row["presentation_order"]): row
        for row in prompts
    }
    pair_order = primary["cyclic_pair_order"]
    next_pair = {
        pair_order[position]: pair_order[(position + 1) % len(pair_order)]
        for position in range(len(pair_order))
    }
    layers = [int(layer) for layer in false["layers"]]

    model = load_model(
        primary["model"], dtype=_DTYPES[args.dtype], device=args.device,
        revision=primary["model_revision"],
    )
    higher = continuation_tokens(model.tokenizer, "higher")
    lower = continuation_tokens(model.tokenizer, "lower")
    if len(higher) != 1 or len(lower) != 1:
        raise RuntimeError("higher/lower must remain single tokens")
    answer_ids = (higher[0], lower[0])

    output_path = (ROOT / args.output).resolve()
    if args.resume and output_path.exists():
        payload = json.loads(output_path.read_text())
        if payload["provenance"]["protocol_sha256"] != sha256(false_path):
            raise RuntimeError("resume protocol mismatch")
    else:
        payload = {
            "study_id": false["study_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "protocol": str(false_path.relative_to(ROOT)),
                "protocol_sha256": sha256(false_path),
                "runner": str(Path(__file__).resolve().relative_to(ROOT)),
                "runner_sha256": sha256(Path(__file__).resolve()),
                "primary_raw_sha256": sha256(raw_path),
                "model": primary["model"],
                "model_revision": primary["model_revision"],
                "device": str(model.input_device),
                "dtype": args.dtype,
                "torch": torch.__version__,
                "python": sys.version,
                "platform": platform.platform(),
            },
            "rows": [],
        }
        atomic_json(output_path, payload)

    print(f"capturing {len(prompts)} prompts", flush=True)
    captured = {}
    for number, prompt in enumerate(prompts, start=1):
        text = format_chat(model.tokenizer, prompt["user"])
        captured[prompt["prompt_id"]] = capture_prompt(
            model, text, layers, answer_ids
        )
        print(f"  clean {number:02d}/{len(prompts):02d}", flush=True)

    control_sources = {
        "distance_matched_cross_material_same": "cross_material_same",
        "distance_matched_order_only": "order_only",
    }
    complete = {
        (row["receiver_prompt_id"], row["control"], int(row["layer"]))
        for row in payload["rows"]
    }
    total = len(prompts) * len(control_sources) * len(layers)
    done = len(complete)
    print(f"running {total} distance-matched patches", flush=True)
    for receiver_number, receiver in enumerate(prompts, start=1):
        receiver_id = receiver["prompt_id"]
        receiver_data = captured[receiver_id]
        choices = donor_ids(receiver, index, next_pair)
        reverse_id = choices["matched_reverse"]
        reverse_data = captured[reverse_id]
        receiver_sign = 1.0 if receiver["relation"] == "refinement" else -1.0
        for control, source_control in control_sources.items():
            donor_id = choices[source_control]
            donor = prompt_by_id[donor_id]
            donor_data = captured[donor_id]
            for layer in layers:
                key = (receiver_id, control, layer)
                if key in complete:
                    continue
                receiver_state = receiver_data["states"][layer]
                reverse_difference = reverse_data["states"][layer] - receiver_state
                control_difference = donor_data["states"][layer] - receiver_state
                target_norm = reverse_difference.norm()
                source_norm = control_difference.norm()
                if float(source_norm) <= 1e-12:
                    raise RuntimeError(f"zero control difference: {key}")
                scale = target_norm / source_norm
                replacement = receiver_state + scale * control_difference
                achieved = (replacement - receiver_state).norm()
                patched = patched_log_odds(
                    model, receiver_data["input_ids"], layer, replacement, answer_ids
                )
                clean = float(receiver_data["clean_log_odds"])
                shift = patched - clean
                payload["rows"].append({
                    "receiver_prompt_id": receiver_id,
                    "receiver_condition_id": receiver["condition_id"],
                    "receiver_pair_id": receiver["pair_id"],
                    "receiver_relation": receiver["relation"],
                    "receiver_presentation_order": receiver["presentation_order"],
                    "source_donor_prompt_id": donor_id,
                    "source_donor_pair_id": donor["pair_id"],
                    "source_donor_relation": donor["relation"],
                    "matched_reverse_donor_prompt_id": reverse_id,
                    "control": control,
                    "layer": layer,
                    "depth_percent": 100.0 * layer / (model.n_layers - 1),
                    "clean_higher_minus_lower": clean,
                    "patched_higher_minus_lower": patched,
                    "raw_shift": shift,
                    "counterfactual_aligned_shift": -receiver_sign * shift,
                    "source_difference_norm": float(source_norm),
                    "target_reverse_difference_norm": float(target_norm),
                    "achieved_difference_norm": float(achieved),
                    "scale": float(scale),
                })
                complete.add(key)
                done += 1
                if done % 25 == 0:
                    atomic_json(output_path, payload)
        atomic_json(output_path, payload)
        print(f"  receiver {receiver_number:02d}/{len(prompts):02d}", flush=True)

    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output_path, payload)
    print(f"complete: {len(payload['rows'])} rows", flush=True)


if __name__ == "__main__":
    main()
