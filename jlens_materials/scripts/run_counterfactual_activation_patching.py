#!/usr/bin/env python3
"""Run the frozen materials counterfactual activation-patching candidate study."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_vendor_jlens"))
sys.path.insert(0, str(ROOT / "scripts"))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from run_lens import _DTYPES, load_model  # noqa: E402
from run_jacobian_steering_pilot import atomic_json, format_chat, sha256  # noqa: E402
from run_semantic_steering_v3 import continuation_tokens, prompt_rows  # noqa: E402


DEFAULT_PROTOCOL = (
    "experiments/candidate-activation-patching-2026-07-16/protocol.json"
)
DEFAULT_OUTPUT = (
    "experiments/candidate-activation-patching-2026-07-16/raw.json"
)


def validate_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"fingerprint mismatch for {path}: {actual} != {expected}")


def expanded_prompts(family: dict) -> list[dict]:
    base = {row["condition_id"]: row for row in family["confirmation_prompts"]}
    rows = []
    for row in prompt_rows(family):
        source = base[row["condition_id"]]
        rows.append({
            **row,
            "pair_id": source["pair_id"],
            "relation": source["relation"],
        })
    return rows


@contextmanager
def replace_residual(model, layer: int, position: int, replacement: torch.Tensor):
    """Replace one post-block residual with an exact donor residual."""

    def hook(_module, _inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        patched = tensor.clone()
        patched[:, position, :] = replacement.to(patched.device, patched.dtype)
        if torch.is_tensor(output):
            return patched
        return (patched, *output[1:])

    handle = model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def two_token_logits(
    model, states: torch.Tensor, token_ids: tuple[int, int]
) -> torch.Tensor:
    """Exact LM-head logits for two vocabulary entries without a full-vocab GEMM."""
    device = model._lm_head.weight.device
    dtype = model._lm_head.weight.dtype
    normalized = model._final_norm(states.to(device=device, dtype=dtype))
    ids = torch.tensor(token_ids, device=device, dtype=torch.long)
    weight = model._lm_head.weight.index_select(0, ids)
    logits = normalized @ weight.T
    bias = getattr(model._lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias.index_select(0, ids)
    softcap = getattr(model, "_logit_softcap", None)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits.float()


@torch.inference_mode()
def capture_prompt(model, text: str, layers: list[int], token_ids: tuple[int, int]) -> dict:
    input_ids = model.encode(text, max_length=512)
    final = model.n_layers - 1
    capture = sorted(set(layers + [final]))
    with ActivationRecorder(model.layers, at=capture) as recorder:
        model.forward(input_ids)
    states = {
        layer: recorder.activations[layer][0, -1].detach().float().cpu()
        for layer in layers
    }
    final_state = recorder.activations[final][0, -1:].detach()
    pair_logits = two_token_logits(model, final_state, token_ids)[0].cpu()
    full_logits = model.unembed(final_state).float()[0]
    logp = torch.log_softmax(full_logits, dim=-1)
    top_id = int(full_logits.argmax())
    return {
        "input_ids": input_ids.detach(),
        "states": states,
        "clean_log_odds": float(pair_logits[0] - pair_logits[1]),
        "clean_pair_logits": [float(value) for value in pair_logits],
        "clean_pair_probability": float(
            logp[token_ids[0]].exp() + logp[token_ids[1]].exp()
        ),
        "clean_top_token_id": top_id,
        "clean_top_token": model.tokenizer.decode(
            [top_id], clean_up_tokenization_spaces=False
        ).strip(),
        "clean_top_is_registered_answer": top_id in token_ids,
    }


@torch.inference_mode()
def patched_log_odds(
    model,
    input_ids: torch.Tensor,
    layer: int,
    donor_state: torch.Tensor,
    token_ids: tuple[int, int],
) -> float:
    final = model.n_layers - 1
    with replace_residual(model, layer, -1, donor_state):
        with ActivationRecorder(model.layers, at=[final]) as recorder:
            model.forward(input_ids)
    final_state = recorder.activations[final][0, -1:].detach()
    logits = two_token_logits(model, final_state, token_ids)[0].cpu()
    return float(logits[0] - logits[1])


@torch.inference_mode()
def readout_curves(
    model,
    lenses: list,
    prompts: list[dict],
    captured: dict[str, dict],
    layers: list[int],
    token_ids: tuple[int, int],
) -> list[dict]:
    output = []
    hidden_by_layer = {
        layer: torch.stack([captured[row["prompt_id"]]["states"][layer] for row in prompts])
        for layer in layers
    }
    for layer in layers:
        hidden = hidden_by_layer[layer]
        direct = two_token_logits(model, hidden, token_ids).cpu()
        for prompt, logits in zip(prompts, direct):
            output.append({
                "prompt_id": prompt["prompt_id"],
                "pair_id": prompt["pair_id"],
                "condition_id": prompt["condition_id"],
                "relation": prompt["relation"],
                "presentation_order": prompt["presentation_order"],
                "method": "direct",
                "lens_seed": None,
                "layer": layer,
                "depth_percent": 100.0 * layer / (model.n_layers - 1),
                "higher_minus_lower": float(logits[0] - logits[1]),
            })
        for lens_seed, lens in enumerate(lenses):
            transported = lens.transport(hidden, layer)
            logits_batch = two_token_logits(model, transported, token_ids).cpu()
            for prompt, logits in zip(prompts, logits_batch):
                output.append({
                    "prompt_id": prompt["prompt_id"],
                    "pair_id": prompt["pair_id"],
                    "condition_id": prompt["condition_id"],
                    "relation": prompt["relation"],
                    "presentation_order": prompt["presentation_order"],
                    "method": "jacobian",
                    "lens_seed": lens_seed,
                    "layer": layer,
                    "depth_percent": 100.0 * layer / (model.n_layers - 1),
                    "higher_minus_lower": float(logits[0] - logits[1]),
                })
    return output


def donor_ids(prompt: dict, index: dict[tuple[str, str, str], dict], next_pair: dict[str, str]) -> dict[str, str]:
    relation = prompt["relation"]
    reverse = "coarsening" if relation == "refinement" else "refinement"
    order = prompt["presentation_order"]
    opposite_order = "negative-first" if order == "positive-first" else "positive-first"
    pair_id = prompt["pair_id"]
    cross = next_pair[pair_id]
    return {
        "matched_reverse": index[(pair_id, reverse, order)]["prompt_id"],
        "cross_material_reverse": index[(cross, reverse, order)]["prompt_id"],
        "cross_material_same": index[(cross, relation, order)]["prompt_id"],
        "order_only": index[(pair_id, relation, opposite_order)]["prompt_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol_path = (ROOT / args.protocol).resolve()
    output_path = (ROOT / args.output).resolve()
    protocol = json.loads(protocol_path.read_text())
    inputs = protocol["inputs"]
    prompt_path = ROOT / inputs["prompt_manifest"]
    validate_hash(prompt_path, inputs["prompt_manifest_sha256"])
    validate_hash(
        ROOT / inputs["previously_inspected_raw"],
        inputs["previously_inspected_raw_sha256"],
    )
    validate_hash(ROOT / inputs["latent_metadata"], inputs["latent_metadata_sha256"])
    for lens_row in protocol["lenses"]:
        validate_hash(ROOT / lens_row["path"], lens_row["sha256"])

    source = json.loads(prompt_path.read_text())
    grain = next(
        row for row in source["families"] if row["family_id"] == "grain-size-strengthening"
    )
    prompts = expanded_prompts(grain)
    prompt_by_id = {row["prompt_id"]: row for row in prompts}
    index = {
        (row["pair_id"], row["relation"], row["presentation_order"]): row
        for row in prompts
    }
    pair_order = protocol["cyclic_pair_order"]
    next_pair = {
        pair_order[index]: pair_order[(index + 1) % len(pair_order)]
        for index in range(len(pair_order))
    }
    layers = [int(layer) for layer in protocol["source_layers"]]

    model = load_model(
        protocol["model"],
        dtype=_DTYPES[args.dtype],
        device=args.device,
        revision=protocol["model_revision"],
    )
    higher = continuation_tokens(model.tokenizer, "higher")
    lower = continuation_tokens(model.tokenizer, "lower")
    if len(higher) != 1 or len(lower) != 1:
        raise RuntimeError(f"registered answers are not single tokens: {higher=} {lower=}")
    token_ids = (higher[0], lower[0])
    lenses = [jlens.JacobianLens.load(ROOT / row["path"]) for row in protocol["lenses"]]
    if any(lens.source_layers != layers for lens in lenses):
        raise RuntimeError("lens source layers do not match frozen patching layers")

    if args.resume and output_path.exists():
        payload = json.loads(output_path.read_text())
        if payload["provenance"]["protocol_sha256"] != sha256(protocol_path):
            raise RuntimeError("resume output has a different frozen protocol")
    else:
        payload = {
            "study_id": protocol["study_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "protocol": str(protocol_path.relative_to(ROOT)),
                "protocol_sha256": sha256(protocol_path),
                "runner": str(Path(__file__).resolve().relative_to(ROOT)),
                "runner_sha256": sha256(Path(__file__).resolve()),
                "model": protocol["model"],
                "model_revision": protocol["model_revision"],
                "dtype": args.dtype,
                "device": str(model.input_device),
                "torch": torch.__version__,
                "python": sys.version,
                "platform": platform.platform(),
                "lenses": protocol["lenses"],
            },
            "answer_tokens": {
                "higher": higher,
                "lower": lower,
                "decoded_higher": model.tokenizer.decode(higher),
                "decoded_lower": model.tokenizer.decode(lower),
            },
            "clean_prompts": [],
            "readout_rows": [],
            "patch_rows": [],
        }
        atomic_json(output_path, payload)

    print(f"capturing {len(prompts)} clean prompts at {len(layers)} layers", flush=True)
    captured = {}
    clean_done = {row["prompt_id"] for row in payload["clean_prompts"]}
    for prompt_index, prompt in enumerate(prompts, start=1):
        text = format_chat(model.tokenizer, prompt["user"])
        result = capture_prompt(model, text, layers, token_ids)
        captured[prompt["prompt_id"]] = result
        if prompt["prompt_id"] not in clean_done:
            payload["clean_prompts"].append({
                **{key: prompt[key] for key in [
                    "prompt_id", "condition_id", "pair_id", "relation",
                    "presentation_order", "presented_words", "expected_outcome", "user",
                ]},
                "n_tokens": int(result["input_ids"].shape[1]),
                **{key: value for key, value in result.items() if key not in {"input_ids", "states"}},
            })
            clean_done.add(prompt["prompt_id"])
            atomic_json(output_path, payload)
        print(f"  clean {prompt_index:02d}/{len(prompts):02d}: {prompt['prompt_id']}", flush=True)

    if not payload["readout_rows"]:
        print("computing Jacobian and direct readout curves", flush=True)
        payload["readout_rows"] = readout_curves(
            model, lenses, prompts, captured, layers, token_ids
        )
        atomic_json(output_path, payload)

    completed = {
        (row["receiver_prompt_id"], row["control"], int(row["layer"]))
        for row in payload["patch_rows"]
    }
    total = len(prompts) * len(protocol["donor_controls"]) * len(layers)
    done = len(completed)
    print(f"patching {total} receiver/control/layer combinations", flush=True)
    for receiver_index, receiver in enumerate(prompts, start=1):
        receiver_id = receiver["prompt_id"]
        receiver_data = captured[receiver_id]
        receiver_sign = 1.0 if receiver["relation"] == "refinement" else -1.0
        donors = donor_ids(receiver, index, next_pair)
        for control in protocol["donor_controls"]:
            donor_id = donors[control]
            donor = prompt_by_id[donor_id]
            donor_data = captured[donor_id]
            for layer in layers:
                key = (receiver_id, control, layer)
                if key in completed:
                    continue
                patched = patched_log_odds(
                    model,
                    receiver_data["input_ids"],
                    layer,
                    donor_data["states"][layer],
                    token_ids,
                )
                clean = float(receiver_data["clean_log_odds"])
                shift = patched - clean
                receiver_state = receiver_data["states"][layer]
                donor_state = donor_data["states"][layer]
                cosine = float(torch.nn.functional.cosine_similarity(
                    receiver_state.unsqueeze(0), donor_state.unsqueeze(0)
                )[0])
                payload["patch_rows"].append({
                    "receiver_prompt_id": receiver_id,
                    "receiver_condition_id": receiver["condition_id"],
                    "receiver_pair_id": receiver["pair_id"],
                    "receiver_relation": receiver["relation"],
                    "receiver_presentation_order": receiver["presentation_order"],
                    "donor_prompt_id": donor_id,
                    "donor_condition_id": donor["condition_id"],
                    "donor_pair_id": donor["pair_id"],
                    "donor_relation": donor["relation"],
                    "donor_presentation_order": donor["presentation_order"],
                    "control": control,
                    "layer": layer,
                    "depth_percent": 100.0 * layer / (model.n_layers - 1),
                    "clean_higher_minus_lower": clean,
                    "donor_clean_higher_minus_lower": float(donor_data["clean_log_odds"]),
                    "patched_higher_minus_lower": patched,
                    "raw_shift": shift,
                    "counterfactual_aligned_shift": -receiver_sign * shift,
                    "receiver_donor_state_cosine": cosine,
                    "receiver_state_norm": float(receiver_state.norm()),
                    "donor_state_norm": float(donor_state.norm()),
                    "state_difference_norm": float((donor_state - receiver_state).norm()),
                })
                completed.add(key)
                done += 1
                if done % 25 == 0:
                    atomic_json(output_path, payload)
        atomic_json(output_path, payload)
        print(f"  patched receiver {receiver_index:02d}/{len(prompts):02d}", flush=True)

    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output_path, payload)
    print(
        f"complete: {len(payload['clean_prompts'])} clean prompts, "
        f"{len(payload['readout_rows'])} readouts, {len(payload['patch_rows'])} patches",
        flush=True,
    )


if __name__ == "__main__":
    main()
