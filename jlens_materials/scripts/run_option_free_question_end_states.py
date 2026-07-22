#!/usr/bin/env python3
"""Capture Gemma states after scientific questions with no answer scaffold.

The runner is protocol-bound.  It formats each archived scientific stem as a
complete chat user turn, records the final prompt-position residual at all 25
registered layers, applies the same three frozen Jacobian lenses, and retains
the model's clean next-token behavior.  It does not append answer choices,
answer words, an arbitrary code, or the earlier checkpoint marker.
"""

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_vendor_jlens"))
sys.path.insert(0, str(ROOT / "scripts"))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from run_jacobian_steering_pilot import atomic_json, format_chat, sha256  # noqa: E402
from run_lens import _DTYPES, load_model  # noqa: E402
from run_lexical_adversarial_representation import (  # noqa: E402
    decoder_basis_representations,
)


DEFAULT_PROTOCOL = (
    "experiments/option-free-question-end-2026-07-18/protocol.json"
)
DEFAULT_OUTPUT = (
    "experiments/option-free-question-end-2026-07-18/raw.json"
)
DEFAULT_STATES = (
    "experiments/option-free-question-end-2026-07-18/representations.npz"
)


def validate_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"fingerprint mismatch for {path}: {actual} != {expected}")


@torch.inference_mode()
def capture_states(
    model,
    prompts: list[dict],
    layers: list[int],
) -> tuple[np.ndarray, list[dict]]:
    states = np.empty(
        (len(prompts), len(layers), model.d_model), dtype=np.float16
    )
    clean_rows = []
    final_layer = model.n_layers - 1
    capture = sorted(set(layers + [final_layer]))
    for prompt_index, prompt in enumerate(prompts):
        user = str(prompt["stem"])
        if any(
            marker in user.lower()
            for marker in (
                "answer exactly",
                "output a",
                "output b",
                "internal checkpoint",
                "if the scientific answer",
            )
        ):
            raise RuntimeError(
                f"answer scaffold leaked into option-free stem: {prompt['prompt_id']}"
            )
        text = format_chat(model.tokenizer, user)
        input_ids = model.encode(text, max_length=512)
        final_position = int(input_ids.shape[1] - 1)
        with ActivationRecorder(model.layers, at=capture) as recorder:
            model.forward(input_ids)
        for layer_index, layer in enumerate(layers):
            states[prompt_index, layer_index] = (
                recorder.activations[layer][0, final_position]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
        final_state = recorder.activations[final_layer][
            0, final_position : final_position + 1
        ].detach()
        full_logits = model.unembed(final_state).float()[0]
        top_values, top_ids = torch.topk(full_logits, k=10)
        clean_rows.append(
            {
                "prompt_id": prompt["prompt_id"],
                "family_id": prompt["family_id"],
                "triplet_id": prompt["triplet_id"],
                "variant": prompt["variant"],
                "expected_outcome": prompt["expected_outcome"],
                "n_tokens": int(input_ids.shape[1]),
                "final_position": final_position,
                "top_tokens": [
                    {
                        "rank": rank + 1,
                        "token_id": int(token_id),
                        "token": model.tokenizer.decode(
                            [int(token_id)],
                            clean_up_tokenization_spaces=False,
                        ),
                        "logit": float(value),
                    }
                    for rank, (value, token_id) in enumerate(
                        zip(
                            top_values.detach().cpu().tolist(),
                            top_ids.detach().cpu().tolist(),
                        )
                    )
                ],
            }
        )
        print(
            f"  captured {prompt_index + 1:02d}/{len(prompts):02d}: "
            f"{prompt['prompt_id']}",
            flush=True,
        )
    return states, clean_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--states-output", default=DEFAULT_STATES)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-size", type=int, default=12)
    args = parser.parse_args()

    protocol_path = (ROOT / args.protocol).resolve()
    output_path = (ROOT / args.output).resolve()
    states_path = (ROOT / args.states_output).resolve()
    protocol = json.loads(protocol_path.read_text())

    manifest_path = ROOT / protocol["inputs"]["prompt_manifest"]["path"]
    runner_path = Path(__file__).resolve()
    validate_hash(
        manifest_path, protocol["inputs"]["prompt_manifest"]["sha256"]
    )
    validate_hash(runner_path, protocol["inputs"]["runner"]["sha256"])
    for lens_row in protocol["lenses"]:
        validate_hash(ROOT / lens_row["path"], lens_row["sha256"])

    manifest = json.loads(manifest_path.read_text())
    prompts = manifest["prompts"]
    if len(prompts) != 72:
        raise RuntimeError(f"expected 72 prompts, found {len(prompts)}")
    layers = [int(layer) for layer in protocol["source_layers"]]

    model = load_model(
        protocol["model"],
        dtype=_DTYPES[args.dtype],
        device=args.device,
        revision=protocol["model_revision"],
    )
    lenses = [
        jlens.JacobianLens.load(ROOT / row["path"])
        for row in protocol["lenses"]
    ]
    if any(lens.source_layers != layers for lens in lenses):
        raise RuntimeError("lens source layers do not match protocol")

    raw, clean_rows = capture_states(model, prompts, layers)
    print("transporting option-free question-end states", flush=True)
    direct, jacobian = decoder_basis_representations(
        model, lenses, raw, layers, args.chunk_size
    )

    states_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        states_path,
        prompt_ids=np.asarray([row["prompt_id"] for row in prompts]),
        positions=np.asarray(["question_end"]),
        layers=np.asarray(layers, dtype=np.int16),
        raw_states=raw[None, ...],
        direct_decoder_basis=direct[None, ...],
        jacobian_decoder_basis=jacobian[:, None, ...],
    )
    payload = {
        "study_id": protocol["study_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "protocol": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": sha256(protocol_path),
            "prompt_manifest": str(manifest_path.relative_to(ROOT)),
            "prompt_manifest_sha256": sha256(manifest_path),
            "runner": str(runner_path.relative_to(ROOT)),
            "runner_sha256": sha256(runner_path),
            "states": str(states_path.relative_to(ROOT)),
            "model": protocol["model"],
            "model_revision": protocol["model_revision"],
            "dtype": args.dtype,
            "device": str(model.input_device),
            "torch": torch.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "lenses": protocol["lenses"],
        },
        "dimensions": {
            "n_prompts": len(prompts),
            "n_triplets": len(manifest["triplets"]),
            "n_families": len(manifest["families"]),
            "n_layers": len(layers),
            "n_positions": 1,
            "d_model": model.d_model,
        },
        "position": {
            "name": "question_end",
            "definition": (
                "final chat-template prompt token after the complete scientific "
                "stem, with no answer choices, answer words, arbitrary code, "
                "response-format instruction, or checkpoint marker"
            ),
        },
        "clean_rows": clean_rows,
    }
    atomic_json(output_path, payload)
    print(
        f"complete: {len(clean_rows)} prompts, "
        f"raw={raw.shape}, direct={direct.shape}, jacobian={jacobian.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
