#!/usr/bin/env python3
"""Run frozen option-free cross-mechanism activation patching.

Donor states come from the archived natural question-end representation
array. The receiver is the corresponding option-free scientific question.
Only the final prompt-position residual is replaced, and receiver-specific
answer logits are measured after the remaining frozen Gemma layers.
"""

from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT / "scripts"))

from jlens.hooks import ActivationRecorder  # noqa: E402
from run_jacobian_steering_pilot import atomic_json, format_chat, sha256  # noqa: E402
from run_lens import _DTYPES, load_model  # noqa: E402
from run_semantic_steering_v3 import continuation_tokens  # noqa: E402


DEFAULT_PROTOCOL = (
    "experiments/cross-mechanism-activation-patching-2026-07-18/"
    "protocol.json"
)
DEFAULT_OUTPUT = (
    "experiments/cross-mechanism-activation-patching-2026-07-18/raw.json"
)


def validate_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"fingerprint mismatch for {path}: {actual} != {expected}"
        )


@contextmanager
def replace_residual(
    model,
    layer: int,
    position: int,
    replacement: torch.Tensor,
):
    """Replace one post-block residual with one archived donor residual."""

    def hook(_module, _inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        patched = tensor.clone()
        patched[:, position, :] = replacement.to(
            patched.device, patched.dtype
        )
        if torch.is_tensor(output):
            return patched
        return (patched, *output[1:])

    handle = model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def two_token_logits(
    model,
    states: torch.Tensor,
    token_ids: tuple[int, int],
) -> torch.Tensor:
    """Calculate exact LM-head logits for two vocabulary entries."""

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
def clean_receiver(
    model,
    input_ids: torch.Tensor,
    token_ids: tuple[int, int],
) -> dict:
    final_layer = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(input_ids)
    final_state = recorder.activations[final_layer][0, -1:].detach()
    logits = two_token_logits(model, final_state, token_ids)[0].cpu()
    return {
        "positive_logit": float(logits[0]),
        "negative_logit": float(logits[1]),
        "positive_minus_negative": float(logits[0] - logits[1]),
    }


@torch.inference_mode()
def patched_receiver_margin(
    model,
    input_ids: torch.Tensor,
    layer: int,
    donor_state: torch.Tensor,
    token_ids: tuple[int, int],
) -> float:
    final_layer = model.n_layers - 1
    with replace_residual(model, layer, -1, donor_state):
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(input_ids)
    final_state = recorder.activations[final_layer][0, -1:].detach()
    logits = two_token_logits(model, final_state, token_ids)[0].cpu()
    return float(logits[0] - logits[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--local-model-snapshot",
        type=Path,
        default=None,
        help=(
            "optional immutable local Hugging Face snapshot; scientific "
            "revision remains fixed by the protocol"
        ),
    )
    args = parser.parse_args()

    protocol_path = (ROOT / args.protocol).resolve()
    output_path = (ROOT / args.output).resolve()
    protocol = json.loads(protocol_path.read_text())
    validate_hash(
        Path(__file__).resolve(),
        protocol["inputs"]["runner"]["sha256"],
    )
    manifest_path = ROOT / protocol["inputs"]["prompt_manifest"]["path"]
    states_path = ROOT / protocol["inputs"]["representations"]["path"]
    validate_hash(
        manifest_path,
        protocol["inputs"]["prompt_manifest"]["sha256"],
    )
    validate_hash(
        states_path,
        protocol["inputs"]["representations"]["sha256"],
    )

    manifest = json.loads(manifest_path.read_text())
    family_rows = {row["family_id"]: row for row in manifest["families"]}
    prompts = [
        row for row in manifest["prompts"] if row["variant"] == "anchor"
    ]
    if len(prompts) != 24:
        raise RuntimeError(f"expected 24 anchor prompts, found {len(prompts)}")
    prompt_ids = [str(row["prompt_id"]) for row in prompts]

    with np.load(states_path) as arrays:
        archived_prompt_ids = arrays["prompt_ids"].astype(str).tolist()
        archived_layers = arrays["layers"].astype(int).tolist()
        positions = arrays["positions"].astype(str).tolist()
        if positions != ["question_end"]:
            raise RuntimeError(f"unexpected archived positions: {positions}")
        selected_layers = [int(layer) for layer in protocol["source_layers"]]
        if not set(selected_layers).issubset(archived_layers):
            raise RuntimeError("protocol layer absent from archived states")
        prompt_index = {
            prompt_id: index
            for index, prompt_id in enumerate(archived_prompt_ids)
        }
        layer_index = {
            layer: index for index, layer in enumerate(archived_layers)
        }
        donor_states = {
            (prompt_id, layer): torch.from_numpy(
                arrays["raw_states"][
                    0,
                    prompt_index[prompt_id],
                    layer_index[layer],
                ].astype(np.float32)
            )
            for prompt_id in prompt_ids
            for layer in selected_layers
        }

    model_source = (
        str(args.local_model_snapshot.expanduser().resolve())
        if args.local_model_snapshot is not None
        else protocol["model"]
    )
    revision = None if args.local_model_snapshot is not None else protocol[
        "model_revision"
    ]
    model = load_model(
        model_source,
        dtype=_DTYPES[args.dtype],
        device=args.device,
        revision=revision,
    )

    answer_tokens: dict[str, tuple[int, int]] = {}
    for family_id, family in family_rows.items():
        positive = continuation_tokens(
            model.tokenizer, family["outcome_positive"]
        )
        negative = continuation_tokens(
            model.tokenizer, family["outcome_negative"]
        )
        if len(positive) != 1 or len(negative) != 1:
            raise RuntimeError(
                f"answer pair is not single-token for {family_id}: "
                f"{positive=} {negative=}"
            )
        answer_tokens[family_id] = (positive[0], negative[0])

    encoded: dict[str, torch.Tensor] = {}
    clean: dict[str, dict] = {}
    for index, prompt in enumerate(prompts, start=1):
        prompt_id = str(prompt["prompt_id"])
        text = format_chat(model.tokenizer, str(prompt["stem"]))
        input_ids = model.encode(text, max_length=512)
        encoded[prompt_id] = input_ids.detach()
        clean[prompt_id] = clean_receiver(
            model,
            input_ids,
            answer_tokens[str(prompt["family_id"])],
        )
        clean[prompt_id].update(
            {
                "prompt_id": prompt_id,
                "family_id": str(prompt["family_id"]),
                "triplet_id": str(prompt["triplet_id"]),
                "variant": str(prompt["variant"]),
                "expected_outcome": str(prompt["expected_outcome"]),
                "numeric_direction": str(prompt["numeric_direction"]),
                "n_tokens": int(input_ids.shape[1]),
            }
        )
        print(f"  clean receiver {index:02d}/{len(prompts):02d}", flush=True)

    if args.resume and output_path.exists():
        payload = json.loads(output_path.read_text())
        if payload["protocol_sha256"] != sha256(protocol_path):
            raise RuntimeError("resume output uses another protocol")
    else:
        payload = {
            "study_id": protocol["study_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": sha256(protocol_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "model": protocol["model"],
            "model_revision": protocol["model_revision"],
            "execution_model_source": model_source,
            "device": str(model.input_device),
            "dtype": args.dtype,
            "torch": torch.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "answer_token_ids": {
                family_id: list(token_ids)
                for family_id, token_ids in answer_tokens.items()
            },
            "clean_receivers": list(clean.values()),
            "patch_rows": [],
        }

    completed = {
        (
            row["receiver_prompt_id"],
            row["donor_prompt_id"],
            int(row["layer"]),
        )
        for row in payload["patch_rows"]
    }
    total = (
        len(prompts)
        * (len(family_rows) - 1)
        * 4
        * len(protocol["source_layers"])
    )
    print(f"running {total} frozen cross-mechanism patches", flush=True)
    for receiver_number, receiver in enumerate(prompts, start=1):
        receiver_id = str(receiver["prompt_id"])
        receiver_family = str(receiver["family_id"])
        receiver_family_row = family_rows[receiver_family]
        receiver_vocab = (
            receiver_family_row["outcome_positive"],
            receiver_family_row["outcome_negative"],
        )
        eligible_donors = [
            donor
            for donor in prompts
            if donor["family_id"] != receiver_family
        ]
        if len(eligible_donors) != 20:
            raise RuntimeError("unexpected cross-mechanism donor count")
        for donor in eligible_donors:
            donor_id = str(donor["prompt_id"])
            donor_family = str(donor["family_id"])
            donor_family_row = family_rows[donor_family]
            donor_vocab = (
                donor_family_row["outcome_positive"],
                donor_family_row["outcome_negative"],
            )
            donor_outcome_sign = (
                1.0
                if donor["expected_outcome"]
                == donor_family_row["outcome_positive"]
                else -1.0
            )
            donor_numeric_sign = (
                1.0 if donor["numeric_direction"] == "increase" else -1.0
            )
            receiver_orientation = (
                1
                if receiver_family_row["positive_numeric_direction"]
                == "increase"
                else -1
            )
            donor_orientation = (
                1
                if donor_family_row["positive_numeric_direction"]
                == "increase"
                else -1
            )
            for layer in protocol["source_layers"]:
                key = (receiver_id, donor_id, int(layer))
                if key in completed:
                    continue
                patched_margin = patched_receiver_margin(
                    model,
                    encoded[receiver_id],
                    int(layer),
                    donor_states[(donor_id, int(layer))],
                    answer_tokens[receiver_family],
                )
                clean_margin = float(
                    clean[receiver_id]["positive_minus_negative"]
                )
                shift = patched_margin - clean_margin
                receiver_state = donor_states[(receiver_id, int(layer))]
                donor_state = donor_states[(donor_id, int(layer))]
                payload["patch_rows"].append(
                    {
                        "receiver_prompt_id": receiver_id,
                        "receiver_family": receiver_family,
                        "receiver_case": str(receiver["triplet_id"]),
                        "receiver_expected_outcome": str(
                            receiver["expected_outcome"]
                        ),
                        "receiver_numeric_direction": str(
                            receiver["numeric_direction"]
                        ),
                        "receiver_answer_vocabulary": list(receiver_vocab),
                        "donor_prompt_id": donor_id,
                        "donor_family": donor_family,
                        "donor_case": str(donor["triplet_id"]),
                        "donor_expected_outcome": str(
                            donor["expected_outcome"]
                        ),
                        "donor_numeric_direction": str(
                            donor["numeric_direction"]
                        ),
                        "donor_answer_vocabulary": list(donor_vocab),
                        "cross_vocabulary": donor_vocab != receiver_vocab,
                        "opposite_response_orientation": (
                            donor_orientation != receiver_orientation
                        ),
                        "layer": int(layer),
                        "depth_percent": (
                            100.0
                            * int(layer)
                            / (model.n_layers - 1)
                        ),
                        "receiver_clean_positive_minus_negative": clean_margin,
                        "patched_positive_minus_negative": patched_margin,
                        "patch_shift": shift,
                        "donor_outcome_sign": donor_outcome_sign,
                        "donor_numeric_sign": donor_numeric_sign,
                        "donor_outcome_aligned_shift": (
                            donor_outcome_sign * shift
                        ),
                        "donor_numeric_aligned_shift": (
                            donor_numeric_sign * shift
                        ),
                        "receiver_donor_state_distance": float(
                            (donor_state - receiver_state).norm()
                        ),
                    }
                )
        payload["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output_path, payload)
        print(
            f"  patched receiver {receiver_number:02d}/{len(prompts):02d}; "
            f"{len(payload['patch_rows'])}/{total}",
            flush=True,
        )

    if len(payload["patch_rows"]) != total:
        raise RuntimeError(
            f"incomplete patch output: {len(payload['patch_rows'])}/{total}"
        )
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output_path, payload)
    print(f"wrote {output_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
