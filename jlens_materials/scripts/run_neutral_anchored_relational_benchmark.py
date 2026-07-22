#!/usr/bin/env python3
"""Capture the frozen layer-34 and output states for the 60-law benchmark."""

from __future__ import annotations

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
from run_lens import load_model  # noqa: E402
from run_semantic_steering_v3 import continuation_tokens  # noqa: E402


OUT = ROOT / "experiments" / "neutral-anchored-relational-physics-2026-07-18"


@torch.inference_mode()
def main() -> None:
    protocol_path = OUT / "protocol.json"
    manifest_path = OUT / "prompt_manifest.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != protocol["inputs"]["prompt_manifest"]["sha256"]:
        raise RuntimeError("prompt manifest fingerprint mismatch")
    if sha256(Path(__file__).resolve()) != protocol["inputs"]["runner"]["sha256"]:
        raise RuntimeError("runner fingerprint mismatch")
    model = load_model(
        protocol["model"],
        dtype=torch.bfloat16,
        device="mps",
        revision=protocol["model_revision"],
    )
    probe_layer = int(protocol["frozen_lens"]["layer"])
    final_layer = model.n_layers - 1
    answer_tokens = {}
    for word in ("higher", "lower"):
        tokens = continuation_tokens(model.tokenizer, word)
        if len(tokens) != 1:
            raise RuntimeError(f"{word!r} is not a one-token continuation: {tokens}")
        answer_tokens[word] = int(tokens[0])

    prompts = manifest["prompts"]
    states = np.empty((len(prompts), model.d_model), dtype=np.float16)
    rows = []
    for index, prompt in enumerate(prompts):
        ids = model.encode(
            format_chat(model.tokenizer, prompt["user"]), max_length=512
        )
        with ActivationRecorder(
            model.layers, at=[probe_layer, final_layer]
        ) as recorder:
            model.forward(ids)
        states[index] = (
            recorder.activations[probe_layer][0, -1]
            .float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        final_state = recorder.activations[final_layer][0, -1:]
        logits = model.unembed(final_state).float()[0].cpu()
        selected = {
            word: float(logits[token_id])
            for word, token_id in answer_tokens.items()
        }
        predicted = max(selected, key=selected.get)
        behavior_evaluable = prompt["expected_answer"] != "unchanged"
        rows.append(
            {
                "prompt_id": prompt["prompt_id"],
                "law_id": prompt["law_id"],
                "category": prompt["category"],
                "domain": prompt["domain"],
                "surface": prompt["surface"],
                "case_index": prompt["case_index"],
                "numerical_sign": prompt["numerical_sign"],
                "physical_sign": prompt["physical_sign"],
                "answer_order": prompt["answer_order"],
                "expected_answer": prompt["expected_answer"],
                "predicted_answer": predicted,
                "behavior_evaluable": behavior_evaluable,
                "correct": (
                    predicted == prompt["expected_answer"]
                    if behavior_evaluable
                    else None
                ),
                "answer_logits": selected,
                "higher_minus_lower_logit": selected["higher"] - selected["lower"],
                "n_tokens": int(ids.shape[1]),
            }
        )
        if (index + 1) % 20 == 0 or index == 0:
            print(f"{index + 1:04d}/{len(prompts):04d} {prompt['prompt_id']}")

    state_path = OUT / "representations.npz"
    np.savez_compressed(
        state_path,
        prompt_ids=np.asarray([x["prompt_id"] for x in prompts]),
        layer=np.asarray([probe_layer], dtype=np.int16),
        raw_states=states,
    )
    atomic_json(
        OUT / "raw.json",
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
                "probe_layer": probe_layer,
                "device": str(model.input_device),
                "dtype": "bfloat16",
                "torch": torch.__version__,
                "python": sys.version,
                "platform": platform.platform(),
            },
            "answer_tokens": answer_tokens,
            "clean_rows": rows,
        },
    )
    print(f"saved {states.shape} layer-{probe_layer} states to {state_path}")


if __name__ == "__main__":
    main()
