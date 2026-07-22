#!/usr/bin/env python3
"""Run the frozen arbitrary-answer-code binding falsification study."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
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
    pair_logits,
)
from run_semantic_steering_v3 import continuation_tokens  # noqa: E402

DEFAULT_PROTOCOL = (
    "experiments/answer-code-binding-2026-07-17/protocol-amendment-v1.json"
)
DEFAULT_OUTPUT = "experiments/answer-code-binding-2026-07-17/raw.json"
DEFAULT_STATES = "experiments/answer-code-binding-2026-07-17/representations.npz"


def validate_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"fingerprint mismatch for {path}: {actual} != {expected}")


@torch.inference_mode()
def capture_two_positions(
    model,
    prompts: list[dict],
    layers: list[int],
    checkpoint_token_id: int,
    code_ids: tuple[int, int],
) -> tuple[np.ndarray, list[dict]]:
    states = np.empty(
        (2, len(prompts), len(layers), model.d_model), dtype=np.float16
    )
    clean_rows = []
    for prompt_index, prompt in enumerate(prompts):
        text = format_chat(model.tokenizer, prompt["user"])
        input_ids = model.encode(text, max_length=512)
        matches = (
            input_ids[0] == checkpoint_token_id
        ).nonzero(as_tuple=False).flatten().tolist()
        if len(matches) != 1:
            decoded = model.tokenizer.decode(
                input_ids[0].detach().cpu().tolist(),
                clean_up_tokenization_spaces=False,
            )
            raise RuntimeError(
                f"{prompt['prompt_id']} has {len(matches)} checkpoint tokens: {decoded}"
            )
        checkpoint_position = int(matches[0])
        final_position = int(input_ids.shape[1] - 1)
        final_layer = model.n_layers - 1
        capture = sorted(set(layers + [final_layer]))
        with ActivationRecorder(model.layers, at=capture) as recorder:
            model.forward(input_ids)
        for layer_index, layer in enumerate(layers):
            activation = recorder.activations[layer][0]
            states[0, prompt_index, layer_index] = (
                activation[checkpoint_position]
                .detach().float().cpu().numpy().astype(np.float16)
            )
            states[1, prompt_index, layer_index] = (
                activation[final_position]
                .detach().float().cpu().numpy().astype(np.float16)
            )
        final_state = recorder.activations[final_layer][
            0, final_position:final_position + 1
        ].detach()
        normalized = model._final_norm(
            final_state.to(
                device=model._lm_head.weight.device,
                dtype=model._lm_head.weight.dtype,
            )
        ).float()
        logits = pair_logits(model, normalized, code_ids)[0].detach().cpu()
        full_logits = model.unembed(final_state).float()[0]
        top_id = int(full_logits.argmax())
        predicted = "A" if float(logits[0]) > float(logits[1]) else "B"
        clean_rows.append({
            "prompt_id": prompt["prompt_id"],
            "triplet_id": prompt["triplet_id"],
            "family_id": prompt["family_id"],
            "variant": prompt["variant"],
            "expected_physical_outcome": prompt["expected_physical_outcome"],
            "expected_code": prompt["expected_code"],
            "predicted_code": predicted,
            "registered_code_correct": predicted == prompt["expected_code"],
            "A_minus_B_log_odds": float(logits[0] - logits[1]),
            "global_top_token_id": top_id,
            "global_top_token": model.tokenizer.decode(
                [top_id], clean_up_tokenization_spaces=False
            ).strip(),
            "global_top_is_code": top_id in code_ids,
            "checkpoint_position": checkpoint_position,
            "final_position": final_position,
            "n_tokens": int(input_ids.shape[1]),
        })
        print(
            f"  captured {prompt_index + 1:02d}/{len(prompts):02d}: "
            f"{prompt['prompt_id']}",
            flush=True,
        )
    return states, clean_rows


def score_readouts(
    model,
    prompts: list[dict],
    layers: list[int],
    positions: list[str],
    direct: np.ndarray,
    jacobian: np.ndarray,
    physical_ids: dict[str, tuple[int, int]],
    code_ids: tuple[int, int],
) -> list[dict]:
    methods = {
        "direct": direct,
        "jacobian_seed0": jacobian[0],
        "jacobian_seed1": jacobian[1],
        "jacobian_seed2": jacobian[2],
        "jacobian_ensemble": jacobian.astype(np.float32).mean(axis=0),
    }
    output = []
    device = model._lm_head.weight.device
    for method, values in methods.items():
        for position_index, position in enumerate(positions):
            for layer_index, layer in enumerate(layers):
                states = torch.from_numpy(
                    values[position_index, :, layer_index].astype(np.float32)
                ).to(device=device)
                code = pair_logits(model, states, code_ids).detach().cpu()
                by_family: dict[str, torch.Tensor] = {}
                for family_id, ids in physical_ids.items():
                    family_indices = [
                        index for index, prompt in enumerate(prompts)
                        if prompt["family_id"] == family_id
                    ]
                    family_states = states[family_indices]
                    by_family[family_id] = pair_logits(
                        model, family_states, ids
                    ).detach().cpu()
                family_offsets = defaultdict(int)
                for prompt_index, prompt in enumerate(prompts):
                    family_id = prompt["family_id"]
                    within_family = family_offsets[family_id]
                    physical = by_family[family_id][within_family]
                    family_offsets[family_id] += 1
                    output.append({
                        "prompt_id": prompt["prompt_id"],
                        "triplet_id": prompt["triplet_id"],
                        "family_id": family_id,
                        "variant": prompt["variant"],
                        "method": method,
                        "position": position,
                        "layer": int(layer),
                        "depth_percent": 100.0 * int(layer) / (model.n_layers - 1),
                        "physical_positive_minus_negative": float(
                            physical[0] - physical[1]
                        ),
                        "code_A_minus_B": float(code[prompt_index, 0] - code[prompt_index, 1]),
                    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
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
    manifest_path = ROOT / protocol["inputs"]["prompt_manifest"]
    validate_hash(manifest_path, protocol["inputs"]["prompt_manifest_sha256"])
    validate_hash(
        ROOT / protocol["inputs"]["runner"], protocol["inputs"]["runner_sha256"]
    )
    validate_hash(
        ROOT / protocol["inputs"]["source_statistics"],
        protocol["inputs"]["source_statistics_sha256"],
    )
    for lens_row in protocol["lenses"]:
        validate_hash(ROOT / lens_row["path"], lens_row["sha256"])
    manifest = json.loads(manifest_path.read_text())
    prompts = manifest["prompts"]
    layers = [int(layer) for layer in protocol["source_layers"]]

    model = load_model(
        protocol["model"],
        dtype=_DTYPES[args.dtype],
        device=args.device,
        revision=protocol["model_revision"],
    )
    checkpoint = continuation_tokens(
        model.tokenizer,
        protocol.get("checkpoint_token_text", protocol["checkpoint_word"]),
    )
    code_a = continuation_tokens(model.tokenizer, "A")
    code_b = continuation_tokens(model.tokenizer, "B")
    if len(checkpoint) != 1 or len(code_a) != 1 or len(code_b) != 1:
        raise RuntimeError(
            f"registered marker/code tokenization failed: "
            f"{checkpoint=} {code_a=} {code_b=}"
        )
    code_ids = (code_a[0], code_b[0])
    physical_ids = {}
    for family in manifest["families"]:
        positive = continuation_tokens(
            model.tokenizer, family["outcome_positive"]
        )
        negative = continuation_tokens(
            model.tokenizer, family["outcome_negative"]
        )
        if len(positive) != 1 or len(negative) != 1:
            raise RuntimeError(f"physical answer tokenization failed: {family}")
        physical_ids[family["family_id"]] = (positive[0], negative[0])
    lenses = [
        jlens.JacobianLens.load(ROOT / row["path"]) for row in protocol["lenses"]
    ]
    if any(lens.source_layers != layers for lens in lenses):
        raise RuntimeError("lens source layers do not match protocol")

    raw, clean_rows = capture_two_positions(
        model, prompts, layers, checkpoint[0], code_ids
    )
    direct_positions = []
    jacobian_positions = []
    for position_index, position in enumerate(["checkpoint", "final_prompt"]):
        print(f"transporting {position} states", flush=True)
        direct, jacobian = decoder_basis_representations(
            model, lenses, raw[position_index], layers, args.chunk_size
        )
        direct_positions.append(direct)
        jacobian_positions.append(jacobian)
    direct_array = np.stack(direct_positions, axis=0)
    jacobian_array = np.stack(jacobian_positions, axis=1)
    readout_rows = score_readouts(
        model,
        prompts,
        layers,
        ["checkpoint", "final_prompt"],
        direct_array,
        jacobian_array,
        physical_ids,
        code_ids,
    )

    states_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        states_path,
        prompt_ids=np.asarray([row["prompt_id"] for row in prompts]),
        positions=np.asarray(["checkpoint", "final_prompt"]),
        layers=np.asarray(layers, dtype=np.int16),
        raw_states=raw,
        direct_decoder_basis=direct_array,
        jacobian_decoder_basis=jacobian_array,
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
            "runner": str(Path(__file__).resolve().relative_to(ROOT)),
            "runner_sha256": sha256(Path(__file__).resolve()),
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
            "n_positions": 2,
            "d_model": model.d_model,
        },
        "token_ids": {
            "checkpoint": checkpoint,
            "A": code_a,
            "B": code_b,
            "physical_by_family": {
                family: list(ids) for family, ids in physical_ids.items()
            },
        },
        "clean_rows": clean_rows,
        "readout_rows": readout_rows,
    }
    atomic_json(output_path, payload)
    print(
        f"complete: {len(clean_rows)} clean rows, "
        f"{len(readout_rows)} readout rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
