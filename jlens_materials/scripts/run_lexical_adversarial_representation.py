#!/usr/bin/env python3
"""Execute the frozen lexical-adversarial materials representation study."""

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
from run_semantic_steering_v3 import continuation_tokens  # noqa: E402


DEFAULT_PROTOCOL = (
    "experiments/lexical-adversarial-representation-2026-07-17/protocol.json"
)
DEFAULT_OUTPUT = (
    "experiments/lexical-adversarial-representation-2026-07-17/raw.json"
)
DEFAULT_STATES = (
    "experiments/lexical-adversarial-representation-2026-07-17/"
    "representations.npz"
)


def validate_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"fingerprint mismatch for {path}: {actual} != {expected}")


def normalized_decoder_basis(model, states: torch.Tensor) -> torch.Tensor:
    device = model._lm_head.weight.device
    dtype = model._lm_head.weight.dtype
    return model._final_norm(states.to(device=device, dtype=dtype)).float()


def pair_logits(
    model, normalized: torch.Tensor, token_ids: tuple[int, int]
) -> torch.Tensor:
    device = model._lm_head.weight.device
    weight = model._lm_head.weight.index_select(
        0, torch.tensor(token_ids, device=device, dtype=torch.long)
    ).float()
    logits = normalized.float() @ weight.T
    bias = getattr(model._lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias.index_select(
            0, torch.tensor(token_ids, device=device, dtype=torch.long)
        ).float()
    softcap = getattr(model, "_logit_softcap", None)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


@torch.inference_mode()
def capture_states(
    model,
    prompts: list[dict],
    layers: list[int],
    answer_ids: dict[str, tuple[int, int]],
) -> tuple[np.ndarray, list[dict]]:
    states = np.empty(
        (len(prompts), len(layers), model.d_model),
        dtype=np.float16,
    )
    clean_rows: list[dict] = []
    for prompt_index, prompt in enumerate(prompts):
        text = format_chat(model.tokenizer, prompt["user"])
        input_ids = model.encode(text, max_length=512)
        final = model.n_layers - 1
        capture = sorted(set(layers + [final]))
        with ActivationRecorder(model.layers, at=capture) as recorder:
            model.forward(input_ids)
        for layer_index, layer in enumerate(layers):
            states[prompt_index, layer_index] = (
                recorder.activations[layer][0, -1]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
        final_state = recorder.activations[final][0, -1:].detach()
        normalized = normalized_decoder_basis(model, final_state)
        ids = answer_ids[prompt["family_id"]]
        logits = pair_logits(model, normalized, ids)[0].detach().cpu()
        full_logits = model.unembed(final_state).float()[0]
        top_id = int(full_logits.argmax())
        predicted = (
            prompt["outcome_positive"] if float(logits[0]) > float(logits[1])
            else prompt["outcome_negative"]
        )
        clean_rows.append({
            "prompt_id": prompt["prompt_id"],
            "triplet_id": prompt["triplet_id"],
            "family_id": prompt["family_id"],
            "variant": prompt["variant"],
            "expected_outcome": prompt["expected_outcome"],
            "predicted_registered_outcome": predicted,
            "registered_pair_correct": predicted == prompt["expected_outcome"],
            "positive_minus_negative_log_odds": float(logits[0] - logits[1]),
            "global_top_token_id": top_id,
            "global_top_token": model.tokenizer.decode(
                [top_id], clean_up_tokenization_spaces=False
            ).strip(),
            "global_top_is_registered_answer": top_id in ids,
            "n_tokens": int(input_ids.shape[1]),
        })
        print(
            f"  captured {prompt_index + 1:02d}/{len(prompts):02d}: "
            f"{prompt['prompt_id']}",
            flush=True,
        )
    return states, clean_rows


@torch.inference_mode()
def decoder_basis_representations(
    model,
    lenses: list[jlens.JacobianLens],
    raw_states: np.ndarray,
    layers: list[int],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_prompts, n_layers, d_model = raw_states.shape
    direct = np.empty((n_prompts, n_layers, d_model), dtype=np.float16)
    transported = np.empty(
        (len(lenses), n_prompts, n_layers, d_model), dtype=np.float16
    )
    device = model.input_device

    for layer_index, layer in enumerate(layers):
        source_cpu = torch.from_numpy(raw_states[:, layer_index].astype(np.float32))
        for start in range(0, n_prompts, chunk_size):
            stop = min(start + chunk_size, n_prompts)
            source = source_cpu[start:stop].to(device=device)
            direct[start:stop, layer_index] = (
                normalized_decoder_basis(model, source)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
        for lens_index, lens in enumerate(lenses):
            matrix = lens.jacobians[layer].to(device=device)
            for start in range(0, n_prompts, chunk_size):
                stop = min(start + chunk_size, n_prompts)
                source = source_cpu[start:stop].to(device=device)
                target = source @ matrix.T
                transported[lens_index, start:stop, layer_index] = (
                    normalized_decoder_basis(model, target)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
            del matrix
        if device.type == "mps":
            torch.mps.empty_cache()
        print(
            f"  transported layer {layer_index + 1:02d}/{len(layers):02d}: {layer}",
            flush=True,
        )
    return direct, transported


@torch.inference_mode()
def top_token_rows(
    model,
    direct: np.ndarray,
    jacobian: np.ndarray,
    prompts: list[dict],
    layers: list[int],
    selected_layers: list[int],
    top_k: int,
    chunk_size: int,
) -> list[dict]:
    rows: list[dict] = []
    device = model.input_device
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    jacobian_mean = jacobian.astype(np.float32).mean(axis=0)
    decoded_cache: dict[int, str] = {}
    for method, array in [
        ("direct", direct),
        ("jacobian_ensemble", jacobian_mean),
    ]:
        for layer in selected_layers:
            layer_index = layer_to_index[layer]
            for start in range(0, len(prompts), chunk_size):
                stop = min(start + chunk_size, len(prompts))
                states = torch.from_numpy(
                    array[start:stop, layer_index].astype(np.float32)
                ).to(device=device, dtype=model._lm_head.weight.dtype)
                logits = model.unembed(states)
                values, indices = torch.topk(logits, k=top_k, dim=-1)
                values = values.float().detach().cpu().numpy()
                indices = indices.detach().cpu().numpy()
                for local_index, prompt_index in enumerate(range(start, stop)):
                    token_ids = [int(value) for value in indices[local_index]]
                    for token_id in token_ids:
                        if token_id not in decoded_cache:
                            decoded_cache[token_id] = model.tokenizer.decode(
                                [token_id], clean_up_tokenization_spaces=False
                            ).strip()
                    rows.append({
                        "prompt_id": prompts[prompt_index]["prompt_id"],
                        "triplet_id": prompts[prompt_index]["triplet_id"],
                        "family_id": prompts[prompt_index]["family_id"],
                        "variant": prompts[prompt_index]["variant"],
                        "method": method,
                        "layer": layer,
                        "depth_percent": 100.0 * layer / (model.n_layers - 1),
                        "token_ids": token_ids,
                        "tokens": [decoded_cache[token_id] for token_id in token_ids],
                        "scores": [float(value) for value in values[local_index]],
                    })
                del logits, values, indices
            if device.type == "mps":
                torch.mps.empty_cache()
            print(f"  decoded {method} layer {layer}", flush=True)
    return rows


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
        ROOT / protocol["inputs"]["runner"],
        protocol["inputs"]["runner_sha256"],
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
    answer_ids: dict[str, tuple[int, int]] = {}
    answer_token_record: dict[str, dict] = {}
    for family in manifest["families"]:
        positive = continuation_tokens(model.tokenizer, family["outcome_positive"])
        negative = continuation_tokens(model.tokenizer, family["outcome_negative"])
        if len(positive) != 1 or len(negative) != 1:
            raise RuntimeError(
                f"registered answers are not single tokens for {family['family_id']}: "
                f"{positive=} {negative=}"
            )
        answer_ids[family["family_id"]] = (positive[0], negative[0])
        answer_token_record[family["family_id"]] = {
            family["outcome_positive"]: positive,
            family["outcome_negative"]: negative,
        }

    lenses = [
        jlens.JacobianLens.load(ROOT / row["path"]) for row in protocol["lenses"]
    ]
    if any(lens.source_layers != layers for lens in lenses):
        raise RuntimeError("lens source layers do not match frozen layers")

    print(
        f"capturing {len(prompts)} prompts at {len(layers)} layers on "
        f"{model.input_device}",
        flush=True,
    )
    raw_states, clean_rows = capture_states(model, prompts, layers, answer_ids)
    print("transporting representations", flush=True)
    direct, jacobian = decoder_basis_representations(
        model, lenses, raw_states, layers, args.chunk_size
    )
    print("decoding target-free top-token neighborhoods", flush=True)
    top_rows = top_token_rows(
        model,
        direct,
        jacobian,
        prompts,
        layers,
        [int(layer) for layer in protocol["target_free_layers"]],
        int(protocol["target_free_top_k_retained"]),
        args.chunk_size,
    )

    states_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        states_path,
        prompt_ids=np.asarray([row["prompt_id"] for row in prompts]),
        layers=np.asarray(layers, dtype=np.int16),
        raw_states=raw_states,
        direct_decoder_basis=direct,
        jacobian_decoder_basis=jacobian,
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
            "d_model": model.d_model,
        },
        "answer_tokens": answer_token_record,
        "clean_rows": clean_rows,
        "target_free_top_tokens": top_rows,
    }
    atomic_json(output_path, payload)
    print(
        f"complete: {len(clean_rows)} clean rows, "
        f"{len(top_rows)} target-free rows, states={states_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
