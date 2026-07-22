#!/usr/bin/env python3
"""Extract held-out residual and Jacobian-transported vectors for Figure 5.

This is a forward-only extraction. It does not refit a lens or repeat the
Jacobian backward passes used during fitting.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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

import domain_prompts as dp  # noqa: E402
import paper_protocol as pp  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from run_lens import load_model  # noqa: E402


MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
PROMPT_PATH = ROOT / "prompts" / "materials-heldout-v1-preregistered.json"
STATS_PATH = ROOT / "experiments" / "materials-heldout-v1_statistics.json"
PROTOCOL_PATH = ROOT / "experiments" / "materials-heldout-v1-latent-geometry-protocol.md"
LENS_PATHS = [
    ROOT / "lenses" / "hub" / "gemma4-e4b-it" / "paper" / f"seed{seed}.pt"
    for seed in range(3)
]
META_PATHS = [path.with_suffix(path.suffix + ".meta.json") for path in LENS_PATHS]
DEFAULT_NPZ = ROOT / "experiments" / "materials-heldout-v1_latent_vectors.npz"
DEFAULT_META = ROOT / "experiments" / "materials-heldout-v1_latent_vectors.meta.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def normalized_final_state(model, tensor: torch.Tensor) -> torch.Tensor:
    device = model.input_device
    target_dtype = model._lm_head.weight.dtype
    normalized = model._final_norm(tensor.to(device=device, dtype=target_dtype)).float()
    return l2_normalize(normalized)


def candidate_rows(stats: dict) -> list[dict]:
    rows = []
    for method in ("jacobian", "logit"):
        families = stats["open_vocabulary"]["methods"][method]["families"]
        for family in sorted(families):
            for rank, candidate in enumerate(families[family]["candidates"][:5], start=1):
                rows.append(
                    {
                        "method": method,
                        "family": family,
                        "family_rank": rank,
                        "token": candidate["token"],
                        "score": candidate["family_specificity_score"],
                        "prompt_support": candidate["prompt_support"],
                    }
                )
    return rows


def encode_candidate(tokenizer, word: str) -> tuple[list[int], str]:
    variants = [("space_prefixed", " " + word), ("bare", word)]
    for rule, text in variants:
        ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        if ids:
            return ids, rule
    raise ValueError(f"could not encode candidate {word!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--output", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    lens_meta = [json.loads(path.read_text()) for path in META_PATHS]
    source_layers = lens_meta[0]["source_layers"]
    target_layer = int(lens_meta[0]["target_layer"])
    for seed, meta in enumerate(lens_meta):
        if meta["source_layers"] != source_layers or int(meta["target_layer"]) != target_layer:
            raise ValueError(f"lens seed {seed} disagrees on registered layers")
        if meta["model"]["model_revision"] != MODEL_REVISION:
            raise ValueError(f"lens seed {seed} model revision mismatch")

    prompts = dp.load_prompts(str(PROMPT_PATH))
    if len(prompts) != 50:
        raise ValueError(f"expected 50 prompts, found {len(prompts)}")
    stats = json.loads(STATS_PATH.read_text())
    candidates = candidate_rows(stats)

    model = load_model(
        MODEL_ID,
        dtype=dtype,
        device=args.device,
        revision=MODEL_REVISION,
    )
    layers_to_record = sorted(set(source_layers + [target_layer]))
    n_prompts = len(prompts)
    d_model = model.d_model
    raw_full = np.empty((n_prompts, len(source_layers), d_model), dtype=np.float32)
    target = np.empty((n_prompts, d_model), dtype=np.float16)
    lexical = np.empty((n_prompts, d_model), dtype=np.float16)
    prompt_meta = []

    for prompt_index, prompt in enumerate(prompts):
        text = dp.resolve_text(prompt, model.tokenizer)
        ids = model.encode(text, max_length=512)
        positions = pp.resolve_score_positions(model.tokenizer, text, prompt, strict=True)
        position = positions[-1]
        with torch.inference_mode(), ActivationRecorder(model.layers, at=layers_to_record) as recorder:
            model.forward(ids)
            for layer_index, layer in enumerate(source_layers):
                vector = recorder.activations[layer][0, position].detach().float()
                raw_full[prompt_index, layer_index] = vector.cpu().numpy()
            target_vector = recorder.activations[target_layer][0, position].detach().float().unsqueeze(0)
            target[prompt_index] = normalized_final_state(model, target_vector)[0].cpu().numpy().astype(np.float16)
            token_vectors = model._embed_tokens(ids)[0].detach().float()
            lexical_vector = l2_normalize(token_vectors.mean(dim=0, keepdim=True))[0]
            lexical[prompt_index] = lexical_vector.cpu().numpy().astype(np.float16)
        prompt_meta.append(
            {
                "slug": prompt.slug,
                "family": prompt.target_family,
                "phrasing_id": prompt.phrasing_id,
                "score_position": position,
                "n_tokens": int(ids.shape[1]),
                "prompt_text": text,
            }
        )
        if (prompt_index + 1) % 5 == 0 or prompt_index == 0:
            print(f"captured residuals: {prompt_index + 1}/{n_prompts}", flush=True)
        del ids

    raw_norm = raw_full / np.maximum(
        np.linalg.norm(raw_full, axis=-1, keepdims=True), 1e-12
    )
    raw = raw_norm.astype(np.float16)
    transported = np.empty((3, n_prompts, len(source_layers), d_model), dtype=np.float16)
    raw_float = torch.from_numpy(raw_full)
    for seed, path in enumerate(LENS_PATHS):
        print(f"loading lens seed {seed}: {path}", flush=True)
        lens = JacobianLens.load(str(path))
        for layer_index, layer in enumerate(source_layers):
            h = raw_float[:, layer_index].to(model.input_device)
            matrix = lens.jacobians[layer].to(model.input_device)
            with torch.inference_mode():
                mapped = h @ matrix.T
                mapped = normalized_final_state(model, mapped)
            transported[seed, :, layer_index] = mapped.cpu().numpy().astype(np.float16)
            del h, matrix, mapped
            if args.device == "mps" and hasattr(torch, "mps") and (layer_index + 1) % 5 == 0:
                torch.mps.empty_cache()
        print(f"transported seed {seed}", flush=True)
        del lens
        gc.collect()
        if args.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    word_vectors = np.empty((len(candidates), d_model), dtype=np.float16)
    for row_index, row in enumerate(candidates):
        token_ids, rule = encode_candidate(model.tokenizer, row["token"])
        row["token_ids"] = token_ids
        row["encoding_rule"] = rule
        vector = model._lm_head.weight[token_ids].detach().float().mean(dim=0, keepdim=True)
        word_vectors[row_index] = l2_normalize(vector)[0].cpu().numpy().astype(np.float16)

    depths = np.asarray([100.0 * layer / (model.n_layers - 1) for layer in source_layers], dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        raw_states=raw,
        transported_states=transported,
        target_states=target,
        lexical_states=lexical,
        word_vectors=word_vectors,
        source_layers=np.asarray(source_layers, dtype=np.int16),
        depths=depths,
        slugs=np.asarray([row["slug"] for row in prompt_meta]),
        families=np.asarray([row["family"] for row in prompt_meta]),
        phrasing_ids=np.asarray([row["phrasing_id"] for row in prompt_meta]),
        word_tokens=np.asarray([row["token"] for row in candidates]),
        word_families=np.asarray([row["family"] for row in candidates]),
        word_methods=np.asarray([row["method"] for row in candidates]),
        word_ranks=np.asarray([row["family_rank"] for row in candidates], dtype=np.int8),
    )
    metadata = {
        "analysis_status": "post hoc exploratory geometry; protocol frozen before vector extraction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": args.device,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "source_layers": source_layers,
        "target_layer": target_layer,
        "d_model": d_model,
        "prompt_manifest": str(PROMPT_PATH.relative_to(ROOT)),
        "prompt_manifest_sha256": sha256(PROMPT_PATH),
        "statistics_source_sha256": sha256(STATS_PATH),
        "lenses": [
            {
                "seed": seed,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "metadata_sha256": sha256(META_PATHS[seed]),
                "corpus_sha256": lens_meta[seed]["corpus"]["sha256"],
            }
            for seed, path in enumerate(LENS_PATHS)
        ],
        "prompts": prompt_meta,
        "words": candidates,
        "arrays": {
            "raw_states": list(raw.shape),
            "transported_states": list(transported.shape),
            "target_states": list(target.shape),
            "lexical_states": list(lexical.shape),
            "word_vectors": list(word_vectors.shape),
        },
        "output_npz": str(args.output.relative_to(ROOT)),
        "output_npz_sha256": sha256(args.output),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {args.metadata}")


if __name__ == "__main__":
    main()
