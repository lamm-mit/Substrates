#!/usr/bin/env python3
"""Score exact multi-token materials terms under direct and Jacobian readouts.

The Jacobian lens is a next-token readout, so a multi-token word cannot be
ranked by pretending that every token piece is predicted from the same state.
This runner uses the lens only for the first continuation token.  It then
teacher-forces the remaining pieces through the unchanged Gemma model.  For
equal-token-length target and contrast words, the resulting target-minus-
contrast score is an exact restricted sequence log-odds ratio: the shared
first-token softmax denominator cancels.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_vendor_jlens"))
sys.path.insert(0, str(ROOT / "scripts"))

import jlens  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from run_jacobian_steering_pilot import atomic_json, sha256  # noqa: E402
from run_lens import _DTYPES, load_model  # noqa: E402
from run_lexical_adversarial_representation import (  # noqa: E402
    decoder_basis_representations,
)


DEFAULT_PROTOCOL = (
    "experiments/multitoken-sequence-robustness-2026-07-18/protocol.json"
)
DEFAULT_RAW = (
    "experiments/multitoken-sequence-robustness-2026-07-18/raw.json"
)
OUT = ROOT / "experiments" / "multitoken-sequence-robustness-2026-07-18"
FIG = OUT / "figures"


def validate_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"fingerprint mismatch for {path}: {actual} != {expected}")


def continuation_tokens(tokenizer, word: str) -> list[int]:
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError(f"empty continuation tokenization for {word!r}")
    return [int(token_id) for token_id in token_ids]


@torch.inference_mode()
def capture_prompt_states(
    model,
    prompt_rows: list[dict],
    layers: list[int],
) -> tuple[np.ndarray, dict[str, torch.Tensor], list[dict]]:
    raw = np.empty(
        (len(prompt_rows), len(layers), model.d_model), dtype=np.float16
    )
    encoded = {}
    clean = []
    final_layer = model.n_layers - 1
    capture = sorted(set(layers + [final_layer]))
    for prompt_index, row in enumerate(prompt_rows):
        # Association prompts use their frozen plain-text field, exactly as in
        # the held-out run; no chat template or suffix is added.
        input_ids = model.encode(str(row["text"]), max_length=512)
        final_position = int(input_ids.shape[1] - 1)
        with ActivationRecorder(model.layers, at=capture) as recorder:
            model.forward(input_ids)
        for layer_index, layer in enumerate(layers):
            raw[prompt_index, layer_index] = (
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
        logits = model.unembed(final_state).float()[0]
        top_values, top_ids = torch.topk(logits, k=10)
        encoded[str(row["slug"])] = input_ids.detach().clone()
        clean.append(
            {
                "slug": str(row["slug"]),
                "family": str(row["target_family"]),
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
            f"  captured {prompt_index + 1:02d}/{len(prompt_rows):02d}: "
            f"{row['slug']}",
            flush=True,
        )
    return raw, encoded, clean


@torch.inference_mode()
def remaining_piece_log_probability(
    model,
    input_ids: torch.Tensor,
    token_ids: list[int],
) -> float:
    if len(token_ids) == 1:
        return 0.0
    prefix = torch.tensor(
        token_ids[:-1],
        device=input_ids.device,
        dtype=input_ids.dtype,
    ).unsqueeze(0)
    extended = torch.cat([input_ids, prefix], dim=1)
    prompt_final = int(input_ids.shape[1] - 1)
    final_layer = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(extended)
    # Position prompt_final predicts piece 0. Position prompt_final + j
    # predicts piece j, so exclude the first prediction here.
    states = recorder.activations[final_layer][
        0,
        prompt_final + 1 : prompt_final + len(token_ids),
    ].detach()
    if states.shape[0] != len(token_ids) - 1:
        raise RuntimeError("teacher-forced continuation position mismatch")
    logits = model.unembed(states).float()
    logp = torch.log_softmax(logits, dim=-1)
    targets = torch.tensor(
        token_ids[1:],
        device=logp.device,
        dtype=torch.long,
    )
    positions = torch.arange(len(targets), device=logp.device)
    return float(logp[positions, targets].sum().detach().cpu())


def first_piece_logits(
    model,
    states: np.ndarray,
    token_ids: tuple[int, int],
    chunk_size: int,
) -> np.ndarray:
    output = np.empty(
        (states.shape[0], states.shape[1], 2), dtype=np.float32
    )
    weight = model._lm_head.weight[
        torch.tensor(token_ids, device=model._lm_head.weight.device)
    ].detach()
    for layer_index in range(states.shape[1]):
        for start in range(0, states.shape[0], chunk_size):
            stop = min(start + chunk_size, states.shape[0])
            source = torch.from_numpy(
                states[start:stop, layer_index].astype(np.float32)
            ).to(
                device=weight.device,
                dtype=weight.dtype,
            )
            output[start:stop, layer_index] = (
                (source @ weight.T)
                .float()
                .detach()
                .cpu()
                .numpy()
            )
    return output


def make_figure(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    layers: np.ndarray,
    depths: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = {"jacobian_ensemble": "#16697A", "direct": "#B56576"}
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.5))

    primary = summary[
        summary["method"].isin(["jacobian_ensemble", "direct"])
    ]
    slugs = sorted(primary["slug"].unique())
    x = np.arange(len(slugs))
    for offset, (method, label) in zip(
        (-0.13, 0.13),
        (("jacobian_ensemble", "Jacobian"), ("direct", "Direct")),
    ):
        values = (
            primary[primary["method"] == method]
            .set_index("slug")
            .loc[slugs, "band_sequence_margin"]
        )
        axes[0, 0].scatter(
            x + offset,
            values,
            color=colors[method],
            s=25,
            label=label,
        )
    axes[0, 0].axhline(0, color="#8B8E91", linestyle="--", linewidth=0.9)
    axes[0, 0].set_ylabel("target minus contrast\nsequence score")
    axes[0, 0].set_xticks(
        x,
        [
            f"{'C' if 'cleavage' in slug else 'M'}{index + 1}"
            for index, slug in enumerate(slugs)
        ],
        rotation=0,
    )
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0, 0].text(
        -0.15,
        1.04,
        "A",
        transform=axes[0, 0].transAxes,
        fontweight="bold",
        fontsize=10,
    )

    for family, linestyle in (
        ("cleavage", "-"),
        ("rapid-transformation", "--"),
    ):
        for method, label in (
            ("jacobian_ensemble", "Jacobian"),
            ("direct", "Direct"),
        ):
            curve = (
                rows[
                    (rows["family"] == family)
                    & (rows["method"] == method)
                ]
                .groupby("layer", as_index=False)["sequence_margin"]
                .mean()
                .set_index("layer")
                .loc[layers]
            )
            axes[0, 1].plot(
                depths,
                curve["sequence_margin"],
                color=colors[method],
                linestyle=linestyle,
                linewidth=1.5,
                label=(
                    f"{label}, "
                    f"{'transgranular' if family == 'cleavage' else 'martensite'}"
                ),
            )
    axes[0, 1].axvspan(38, 92, color="#D9E7EA", alpha=0.45)
    axes[0, 1].axhline(0, color="#8B8E91", linestyle=":", linewidth=0.9)
    axes[0, 1].set_xlabel("layer depth (%)")
    axes[0, 1].set_ylabel("target minus contrast\nsequence score")
    axes[0, 1].legend(frameon=False, fontsize=6.5)
    axes[0, 1].text(
        -0.15,
        1.04,
        "B",
        transform=axes[0, 1].transAxes,
        fontweight="bold",
        fontsize=10,
    )

    j = summary[summary["method"] == "jacobian_ensemble"]
    axes[1, 0].scatter(
        j["band_first_piece_margin"],
        j["band_sequence_margin"],
        c=[
            "#D17C38" if family == "cleavage" else "#5B8C5A"
            for family in j["family"]
        ],
        s=34,
        edgecolor="white",
        linewidth=0.5,
    )
    low = min(
        float(j["band_first_piece_margin"].min()),
        float(j["band_sequence_margin"].min()),
    )
    high = max(
        float(j["band_first_piece_margin"].max()),
        float(j["band_sequence_margin"].max()),
    )
    axes[1, 0].plot([low, high], [low, high], color="#8B8E91", linestyle="--")
    axes[1, 0].set_xlabel("first-piece margin")
    axes[1, 0].set_ylabel("full sequence margin")
    axes[1, 0].text(
        -0.15,
        1.04,
        "C",
        transform=axes[1, 0].transAxes,
        fontweight="bold",
        fontsize=10,
    )

    seeds = summary[summary["method"].str.startswith("jacobian_seed")]
    for family, marker, color in (
        ("cleavage", "o", "#D17C38"),
        ("rapid-transformation", "s", "#5B8C5A"),
    ):
        subset = (
            seeds[seeds["family"] == family]
            .groupby("method", as_index=False)["band_sequence_margin"]
            .mean()
        )
        axes[1, 1].scatter(
            [0, 1, 2],
            subset.set_index("method").loc[
                ["jacobian_seed0", "jacobian_seed1", "jacobian_seed2"],
                "band_sequence_margin",
            ],
            color=color,
            marker=marker,
            s=34,
            label=(
                "transgranular vs intergranular"
                if family == "cleavage"
                else "martensite vs bainite"
            ),
        )
    axes[1, 1].axhline(0, color="#8B8E91", linestyle="--", linewidth=0.9)
    axes[1, 1].set_xticks([0, 1, 2], ["fit 0", "fit 1", "fit 2"])
    axes[1, 1].set_ylabel("mean band sequence margin")
    axes[1, 1].legend(frameon=False, fontsize=6.5)
    axes[1, 1].text(
        -0.15,
        1.04,
        "D",
        transform=axes[1, 1].transAxes,
        fontweight="bold",
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=0.11,
        top=0.97,
        wspace=0.34,
        hspace=0.38,
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(
            FIG / f"multitoken-sequence-robustness.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--amendment",
        default=None,
        help="optional implementation-only amendment bound to the base protocol",
    )
    parser.add_argument("--output", default=DEFAULT_RAW)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument(
        "--local-model-snapshot",
        type=Path,
        default=None,
        help=(
            "optional local Hugging Face snapshot for offline execution; "
            "the frozen model id and revision remain the scientific provenance"
        ),
    )
    args = parser.parse_args()

    protocol_path = (ROOT / args.protocol).resolve()
    output_path = (ROOT / args.output).resolve()
    protocol = json.loads(protocol_path.read_text())
    amendment_path = None
    amendment = None
    if args.amendment is not None:
        amendment_path = (ROOT / args.amendment).resolve()
        amendment = json.loads(amendment_path.read_text())
        if amendment["base_protocol_sha256"] != sha256(protocol_path):
            raise RuntimeError("amendment is not bound to the base protocol")
        protocol["inputs"]["runner"]["sha256"] = amendment[
            "amended_runner_sha256"
        ]
    runner_path = Path(__file__).resolve()
    validate_hash(runner_path, protocol["inputs"]["runner"]["sha256"])
    manifest_path = ROOT / protocol["inputs"]["prompt_manifest"]["path"]
    validate_hash(
        manifest_path, protocol["inputs"]["prompt_manifest"]["sha256"]
    )
    for lens_row in protocol["lenses"]:
        validate_hash(ROOT / lens_row["path"], lens_row["sha256"])

    manifest = json.loads(manifest_path.read_text())
    selected_families = set(protocol["contrasts"])
    prompts = [
        row
        for row in manifest["prompts"]
        if str(row["target_family"]) in selected_families
    ]
    if len(prompts) != 10:
        raise RuntimeError(f"expected 10 selected prompts, found {len(prompts)}")
    for row in prompts:
        if any(
            term.lower() in str(row["text"]).lower()
            for term in (
                protocol["contrasts"][str(row["target_family"])]["target"],
                protocol["contrasts"][str(row["target_family"])]["contrast"],
            )
        ):
            raise RuntimeError(f"target/contrast leaks into prompt {row['slug']}")

    model_source = (
        str(args.local_model_snapshot.resolve())
        if args.local_model_snapshot is not None
        else protocol["model"]
    )
    model = load_model(
        model_source,
        dtype=_DTYPES[args.dtype],
        device=args.device,
        revision=None if args.local_model_snapshot is not None else protocol["model_revision"],
    )
    layers = [int(layer) for layer in protocol["source_layers"]]
    tokenization = {}
    for family, contrast in protocol["contrasts"].items():
        target = continuation_tokens(model.tokenizer, contrast["target"])
        comparator = continuation_tokens(model.tokenizer, contrast["contrast"])
        if len(target) != len(comparator) or len(target) < 2:
            raise RuntimeError(
                f"contrast is not equal-length multi-token: {family}"
            )
        tokenization[family] = {
            "target": target,
            "contrast": comparator,
            "target_pieces": [
                model.tokenizer.decode(
                    [token_id], clean_up_tokenization_spaces=False
                )
                for token_id in target
            ],
            "contrast_pieces": [
                model.tokenizer.decode(
                    [token_id], clean_up_tokenization_spaces=False
                )
                for token_id in comparator
            ],
        }
    expected_tokenization = protocol["tokenization_preflight"]
    for family, resolved in tokenization.items():
        for key in ("target", "contrast"):
            if resolved[key] != expected_tokenization[family][key]:
                raise RuntimeError(f"tokenization changed for {family} {key}")

    lenses = [
        jlens.JacobianLens.load(ROOT / row["path"])
        for row in protocol["lenses"]
    ]
    if any(lens.source_layers != layers for lens in lenses):
        raise RuntimeError("lens source layers do not match protocol")

    raw, encoded, clean = capture_prompt_states(model, prompts, layers)
    direct, jacobian = decoder_basis_representations(
        model, lenses, raw, layers, args.chunk_size
    )

    continuation = {}
    for row in prompts:
        family = str(row["target_family"])
        continuation[str(row["slug"])] = {}
        for label in ("target", "contrast"):
            continuation[str(row["slug"])][label] = (
                remaining_piece_log_probability(
                    model,
                    encoded[str(row["slug"])],
                    tokenization[family][label],
                )
            )
        print(f"  sequence continuation: {row['slug']}", flush=True)

    method_arrays = {
        "direct": direct,
        "jacobian_seed0": jacobian[0],
        "jacobian_seed1": jacobian[1],
        "jacobian_seed2": jacobian[2],
        "jacobian_ensemble": jacobian.astype(np.float32).mean(axis=0),
    }
    result_rows = []
    for method, states in method_arrays.items():
        for prompt_index, row in enumerate(prompts):
            family = str(row["target_family"])
            target_tokens = tokenization[family]["target"]
            contrast_tokens = tokenization[family]["contrast"]
            logits = first_piece_logits(
                model,
                states[prompt_index : prompt_index + 1],
                (target_tokens[0], contrast_tokens[0]),
                args.chunk_size,
            )[0]
            target_continuation = continuation[str(row["slug"])]["target"]
            contrast_continuation = continuation[str(row["slug"])]["contrast"]
            for layer_index, layer in enumerate(layers):
                first_margin = float(
                    logits[layer_index, 0] - logits[layer_index, 1]
                )
                sequence_margin = float(
                    first_margin
                    + target_continuation
                    - contrast_continuation
                )
                result_rows.append(
                    {
                        "slug": str(row["slug"]),
                        "family": family,
                        "method": method,
                        "layer": int(layer),
                        "depth_percent": 100.0
                        * int(layer)
                        / (model.n_layers - 1),
                        "target": protocol["contrasts"][family]["target"],
                        "contrast": protocol["contrasts"][family]["contrast"],
                        "n_pieces": len(target_tokens),
                        "target_first_piece_logit": float(
                            logits[layer_index, 0]
                        ),
                        "contrast_first_piece_logit": float(
                            logits[layer_index, 1]
                        ),
                        "first_piece_margin": first_margin,
                        "target_remaining_log_probability": float(
                            target_continuation
                        ),
                        "contrast_remaining_log_probability": float(
                            contrast_continuation
                        ),
                        "sequence_margin": sequence_margin,
                        "target_wins": bool(sequence_margin > 0),
                    }
                )
    frame = pd.DataFrame(result_rows)
    band_low, band_high = protocol["registered_band_percent"]
    band = frame[
        (frame["depth_percent"] >= band_low)
        & (frame["depth_percent"] <= band_high)
    ]
    summary = (
        band.groupby(
            ["slug", "family", "method", "target", "contrast"],
            as_index=False,
        )
        .agg(
            band_first_piece_margin=("first_piece_margin", "mean"),
            band_sequence_margin=("sequence_margin", "mean"),
            positive_layer_fraction=("target_wins", "mean"),
        )
    )
    family_summary = (
        summary.groupby(
            ["family", "method", "target", "contrast"],
            as_index=False,
        )
        .agg(
            mean_band_first_piece_margin=("band_first_piece_margin", "mean"),
            mean_band_sequence_margin=("band_sequence_margin", "mean"),
            positive_prompts=("band_sequence_margin", lambda x: int((x > 0).sum())),
            n_prompts=("slug", "size"),
        )
    )

    j = summary[summary["method"] == "jacobian_ensemble"].set_index("slug")
    d = summary[summary["method"] == "direct"].set_index("slug")
    prompt_delta = pd.DataFrame(
        {
            "slug": j.index,
            "family": j["family"],
            "jacobian_band_sequence_margin": j["band_sequence_margin"],
            "direct_band_sequence_margin": d.loc[
                j.index, "band_sequence_margin"
            ],
        }
    ).reset_index(drop=True)
    prompt_delta["jacobian_minus_direct"] = (
        prompt_delta["jacobian_band_sequence_margin"]
        - prompt_delta["direct_band_sequence_margin"]
    )
    family_delta = (
        prompt_delta.groupby("family", as_index=False)
        .agg(
            jacobian_band_sequence_margin=(
                "jacobian_band_sequence_margin",
                "mean",
            ),
            direct_band_sequence_margin=(
                "direct_band_sequence_margin",
                "mean",
            ),
            jacobian_minus_direct=("jacobian_minus_direct", "mean"),
            positive_jacobian_prompts=(
                "jacobian_band_sequence_margin",
                lambda x: int((x > 0).sum()),
            ),
            positive_delta_prompts=(
                "jacobian_minus_direct",
                lambda x: int((x > 0).sum()),
            ),
        )
    )

    both_families_positive = bool(
        np.all(family_delta["jacobian_band_sequence_margin"] > 0)
    )
    positive_prompts = int(
        np.sum(prompt_delta["jacobian_band_sequence_margin"] > 0)
    )
    all_seed_family_signs = True
    for family in selected_families:
        seed_means = family_summary[
            (family_summary["family"] == family)
            & family_summary["method"].str.startswith("jacobian_seed")
        ]["mean_band_sequence_margin"]
        all_seed_family_signs &= bool(np.all(seed_means > 0))
    robust = bool(
        both_families_positive
        and positive_prompts >= 8
        and all_seed_family_signs
    )

    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "layer_sequence_scores.csv", index=False)
    summary.to_csv(OUT / "prompt_band_summary.csv", index=False)
    family_summary.to_csv(OUT / "family_method_summary.csv", index=False)
    prompt_delta.to_csv(OUT / "prompt_jacobian_direct_contrasts.csv", index=False)
    family_delta.to_csv(OUT / "family_jacobian_direct_contrasts.csv", index=False)
    make_figure(
        frame,
        summary,
        np.asarray(layers),
        np.asarray([100.0 * layer / (model.n_layers - 1) for layer in layers]),
    )

    payload = {
        "study_id": protocol["study_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path),
        "protocol_amendment": (
            str(amendment_path.relative_to(ROOT))
            if amendment_path is not None
            else None
        ),
        "protocol_amendment_sha256": (
            sha256(amendment_path) if amendment_path is not None else None
        ),
        "runner_sha256": sha256(runner_path),
        "model": protocol["model"],
        "model_revision": protocol["model_revision"],
        "execution_model_source": model_source,
        "device": str(model.input_device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "tokenization": tokenization,
        "clean_rows": clean,
        "continuation_log_probabilities": continuation,
        "frozen_verdict": {
            "sequence_robustness": "pass" if robust else "fail",
            "both_family_jacobian_margins_positive": both_families_positive,
            "positive_jacobian_prompts": positive_prompts,
            "required_positive_prompts": 8,
            "all_three_seed_family_margins_positive": all_seed_family_signs,
            "jacobian_specific": bool(
                np.all(family_delta["jacobian_minus_direct"] > 0)
            ),
        },
        "family_results": family_delta.to_dict(orient="records"),
        "guardrail": (
            "This post-hoc robustness test evaluates two originally excluded "
            "multi-token terms against equal-piece scientific contrasts. It "
            "does not replace the preregistered single-token endpoint."
        ),
    }
    atomic_json(output_path, payload)
    (OUT / "statistics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "RESULTS.md").write_text(
        "\n".join(
            [
                "# Multi-token sequence robustness",
                "",
                (
                    f"Frozen verdict: **{'PASS' if robust else 'FAIL'}** "
                    "for the descriptive sequence-robustness rule."
                ),
                "",
                "## Exact contrasts",
                "",
                "- `transgranular` versus `intergranular` (two pieces each).",
                "- `martensite` versus `bainite` (three pieces each).",
                "",
                "The lens scores only the first piece from an intermediate",
                "state. Remaining pieces are teacher-forced through unchanged",
                "Gemma. Equal token lengths make the target-minus-contrast score",
                "a restricted sequence log-odds ratio.",
                "",
                "## Family results",
                "",
            ]
            + [
                (
                    f"- {row.family}: Jacobian "
                    f"{row.jacobian_band_sequence_margin:+.3f}, direct "
                    f"{row.direct_band_sequence_margin:+.3f}, difference "
                    f"{row.jacobian_minus_direct:+.3f}; "
                    f"{int(row.positive_jacobian_prompts)}/5 prompts positive."
                )
                for row in family_delta.itertuples(index=False)
            ]
            + [
                "",
                (
                    f"Across both families, {positive_prompts}/10 prompt-level "
                    "Jacobian band margins are positive."
                ),
                "",
                "## Reproduction",
                "",
                "```bash",
                "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\",
                "  python scripts/run_multitoken_sequence_robustness.py --device cpu",
                "```",
                "",
            ]
        )
    )
    print(json.dumps(payload["frozen_verdict"], indent=2))


if __name__ == "__main__":
    main()
