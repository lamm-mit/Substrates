#!/usr/bin/env python3
"""Capture all-layer states for the scaffolded abstraction development run."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "_vendor_jlens"), str(ROOT / "scripts")]

from jlens.hooks import ActivationRecorder  # noqa: E402
from run_jacobian_steering_pilot import atomic_json, format_chat, sha256  # noqa: E402
from run_lens import _DTYPES, load_model  # noqa: E402
from run_lexical_adversarial_representation import pair_logits  # noqa: E402
from run_semantic_steering_v3 import continuation_tokens  # noqa: E402


DEFAULT_DIR = "experiments/elicited-physics-abstraction-2026-07-18"


def marker_positions(
    input_ids: torch.Tensor,
    marker_ids: dict[str, int],
    prompt_id: str,
) -> dict[str, int]:
    """Locate the three registered marker tokens and final prompt position."""
    output = {}
    for name, token_id in marker_ids.items():
        matches = (
            (input_ids[0] == token_id)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"{prompt_id}: expected one {name} marker, found {len(matches)}"
            )
        output[name] = int(matches[0])
    if not output["law"] < output["comparison"] < output["decision"]:
        raise RuntimeError(f"{prompt_id}: marker order is invalid: {output}")
    output["final_prompt"] = int(input_ids.shape[1] - 1)
    return output


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default=DEFAULT_DIR)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    out = (ROOT / args.experiment_dir).resolve()
    protocol_path = out / "protocol.json"
    manifest_path = out / "prompt_manifest.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != protocol["inputs"]["prompt_manifest"]["sha256"]:
        raise RuntimeError("manifest fingerprint mismatch")
    if sha256(Path(__file__).resolve()) != protocol["inputs"]["runner"]["sha256"]:
        raise RuntimeError("runner fingerprint mismatch")

    model = load_model(
        protocol["model"],
        dtype=_DTYPES[args.dtype],
        device=args.device,
        revision=protocol["model_revision"],
    )
    marker_ids = {}
    for name, marker in manifest["markers"].items():
        ids = model.tokenizer.encode(" " + marker, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{name} marker is not one token: {ids}")
        marker_ids[name] = int(ids[0])
    higher = continuation_tokens(model.tokenizer, "higher")
    lower = continuation_tokens(model.tokenizer, "lower")
    if len(higher) != 1 or len(lower) != 1:
        raise RuntimeError(f"answers are not single tokens: {higher=}, {lower=}")
    answer_ids = (int(higher[0]), int(lower[0]))

    prompts = manifest["prompts"]
    positions = protocol["positions"]
    layers = list(range(model.n_layers))
    states = np.empty(
        (len(positions), len(prompts), len(layers), model.d_model),
        dtype=np.float16,
    )
    rows = []
    for prompt_index, prompt in enumerate(prompts):
        text = format_chat(model.tokenizer, prompt["user"])
        input_ids = model.encode(text, max_length=512)
        locations = marker_positions(input_ids, marker_ids, prompt["prompt_id"])
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)
        for layer_index, layer in enumerate(layers):
            activation = recorder.activations[layer][0]
            for position_index, position in enumerate(positions):
                states[position_index, prompt_index, layer_index] = (
                    activation[locations[position]]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
        final_state = recorder.activations[layers[-1]][
            0, locations["final_prompt"] : locations["final_prompt"] + 1
        ]
        logits = pair_logits(
            model,
            model._final_norm(
                final_state.to(
                    device=model._lm_head.weight.device,
                    dtype=model._lm_head.weight.dtype,
                )
            ).float(),
            answer_ids,
        )[0].float().cpu()
        predicted = "higher" if float(logits[0]) > float(logits[1]) else "lower"
        rows.append(
            {
                **{
                    key: prompt[key]
                    for key in (
                        "prompt_id",
                        "law_id",
                        "split",
                        "surface",
                        "case_index",
                        "law_sign",
                        "numerical_sign",
                        "physical_sign",
                        "expected_answer",
                    )
                },
                "predicted_answer": predicted,
                "correct": predicted == prompt["expected_answer"],
                "higher_minus_lower_log_odds": float(logits[0] - logits[1]),
                "n_tokens": int(input_ids.shape[1]),
                "positions": locations,
            }
        )
        print(f"{prompt_index + 1:03d}/{len(prompts):03d} {prompt['prompt_id']}")

    state_path = out / "representations.npz"
    np.savez_compressed(
        state_path,
        prompt_ids=np.asarray([x["prompt_id"] for x in prompts]),
        positions=np.asarray(positions),
        layers=np.asarray(layers, dtype=np.int16),
        raw_states=states,
    )
    atomic_json(
        out / "raw.json",
        {
            "study_id": protocol["study_id"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "protocol_sha256": sha256(protocol_path),
                "manifest_sha256": sha256(manifest_path),
                "runner_sha256": sha256(Path(__file__).resolve()),
                "states_sha256": sha256(state_path),
                "model": protocol["model"],
                "model_revision": protocol["model_revision"],
                "dtype": args.dtype,
                "device": str(model.input_device),
                "torch": torch.__version__,
                "python": sys.version,
                "platform": platform.platform(),
            },
            "answer_tokens": {"higher": answer_ids[0], "lower": answer_ids[1]},
            "clean_rows": rows,
        },
    )
    print(f"saved {states.shape} to {state_path}")


if __name__ == "__main__":
    main()
