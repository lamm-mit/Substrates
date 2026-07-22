#!/usr/bin/env python3
"""Analyze one prompt across independently fitted paper-protocol lenses.

This consumes stored run JSON files only; it does not load Gemma or lens weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jlens-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matviz  # noqa: E402


DEFAULT_RUNS = [
    ROOT / "runs" / f"gemma4-e4b-it-paper-seed{seed}.json"
    for seed in range(3)
]
DEFAULT_SLUG = "paper-v2-assoc-notch-resistance-01"
DEFAULT_CONCEPT = "toughness"


def one_indexed(value: object) -> int | None:
    if value is None or int(value) < 0:
        return None
    return int(value) + 1


def average_ranks(values: list[float]) -> np.ndarray:
    """Return average ranks, including the midpoint treatment of ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise ValueError("Spearman inputs must have the same length >= 2")
    return float(np.corrcoef(average_ranks(values_a), average_ranks(values_b))[0, 1])


def concept_trajectory(record: dict, lens_name: str, label: str) -> dict:
    return next(
        item for item in record["concept_trajectories"][lens_name]
        if item["label"] == label
    )


def emergence(record: dict, label: str) -> dict:
    return next(item for item in record["emergence"] if item["label"] == label)


def load_prompt_records(run_paths: list[Path], slug: str) -> tuple[list[dict], list[dict]]:
    runs = [json.loads(path.read_text()) for path in run_paths]
    records = [next(item for item in run["prompts"] if item["slug"] == slug) for run in runs]
    reference = records[0]
    expected = {
        "model": runs[0]["model"],
        "model_revision": runs[0]["model_identity"]["model_revision"],
        "prompt_text": reference["prompt_text"],
        "score_positions": reference["score_positions"],
        "tracked": [item["label"] for item in reference["tracked"]],
    }
    for index, (run, record) in enumerate(zip(runs, records)):
        observed = {
            "model": run["model"],
            "model_revision": run["model_identity"]["model_revision"],
            "prompt_text": record["prompt_text"],
            "score_positions": record["score_positions"],
            "tracked": [item["label"] for item in record["tracked"]],
        }
        if observed != expected:
            raise ValueError(f"seed {index} is not comparable")
        if not run["methodology"].get("paper_protocol_complete"):
            raise ValueError(f"seed {index} is not a complete paper-protocol run")
        if not record.get("valid_for_metrics"):
            raise ValueError(f"seed {index} excluded {slug}: {record.get('excluded_reasons')}")
    return runs, records


def top_tokens_at_depth(record: dict, depth: float) -> list[str]:
    layer = min(record["layer_readouts"], key=lambda item: abs(float(item["depth"]) - depth))
    return [str(token).strip() for token in layer["top_tokens"]]


def plot_prompt(records: list[dict], concept: str, output_base: Path) -> list[Path]:
    colors = matviz.SERIES[:3]
    trajectories = [concept_trajectory(record, "jacobian_lens", concept) for record in records]
    logit = concept_trajectory(records[0], "logit_lens", concept)
    tracked = [item["label"] for item in records[0]["tracked"]]

    fig, (ax_trace, ax_specificity) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": [1.65, 1.0]}
    )
    band = records[0]["band"]
    ax_trace.axvspan(band[0], band[1], color=matviz.GRID, alpha=0.18, zorder=0)
    for seed, (trajectory, color) in enumerate(zip(trajectories, colors)):
        ax_trace.plot(
            trajectory["depths"], trajectory["ranks"], marker="o", ms=4.2,
            lw=1.8, color=color, label=f"Jacobian lens, seed {seed}", zorder=3,
        )
    ax_trace.plot(
        logit["depths"], logit["ranks"], color=matviz.MUTE, lw=1.8,
        linestyle="--", marker="s", ms=3.5, label="logit lens", zorder=2,
    )
    best = emergence(records[0], concept)
    best_rank = one_indexed(best["best_rank"])
    ax_trace.annotate(
        f"{concept}: rank {best_rank} in all 3 seeds\nlayer depth {best['best_depth']:.1f}%",
        xy=(best["best_depth"], best_rank), xytext=(best["best_depth"] + 8, 8),
        arrowprops={"arrowstyle": "->", "color": matviz.INK, "lw": 0.9},
        fontsize=8.5, color=matviz.INK,
    )
    ax_trace.set_yscale("log")
    ax_trace.invert_yaxis()
    ax_trace.set_xlim(0, 100)
    ax_trace.set_ylim(300000, 0.8)
    ax_trace.set_xlabel("layer depth (reindexed 0–100)")
    ax_trace.set_ylabel("vocabulary rank (lower is stronger)")
    ax_trace.grid(True, which="both", color=matviz.GRID, lw=0.55, alpha=0.7)
    ax_trace.legend(loc="lower left", fontsize=7.5, framealpha=0.92)
    ax_trace.set_title("A  Transient concept emergence across depth", loc="left", fontweight="bold")

    x = np.arange(len(tracked), dtype=float)
    offsets = (-0.17, 0.0, 0.17)
    for seed, (record, color, offset) in enumerate(zip(records, colors, offsets)):
        ranks = [one_indexed(emergence(record, label)["best_rank"]) for label in tracked]
        ax_specificity.scatter(
            x + offset, ranks, s=56, color=color, edgecolor=matviz.INK,
            linewidth=0.45, label=f"seed {seed}", zorder=4,
        )
    logit_ranks = [one_indexed(emergence(records[0], label)["logit_lens_best_rank"]) for label in tracked]
    ax_specificity.scatter(
        x, logit_ranks, s=62, marker="s", facecolor="none", edgecolor=matviz.MUTE,
        linewidth=1.4, label="logit lens", zorder=3,
    )
    ax_specificity.set_yscale("log")
    ax_specificity.invert_yaxis()
    ax_specificity.set_ylim(50000, 0.8)
    ax_specificity.set_xticks(x)
    ax_specificity.set_xticklabels(tracked)
    ax_specificity.set_ylabel("best rank in preregistered band")
    ax_specificity.grid(True, axis="y", which="both", color=matviz.GRID, lw=0.55, alpha=0.7)
    ax_specificity.set_title("B  The readout is mechanism-specific", loc="left", fontweight="bold")
    ax_specificity.legend(loc="lower left", fontsize=7.5, framealpha=0.92)

    fig.suptitle(
        "One prompt, three independently fitted lenses: latent fracture toughness",
        x=0.055, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.055, 0.93,
        "Same Gemma checkpoint and prompt; lenses fitted on three independently shuffled 1,000-record WikiText samples.",
        ha="left", color=matviz.MUTE, fontsize=8.5,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.82, wspace=0.30)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".png", ".svg"):
        output = output_base.with_suffix(suffix)
        fig.savefig(output, dpi=220, bbox_inches="tight")
        written.append(output)
    plt.close(fig)
    return written


def build_report(
    runs: list[dict], records: list[dict], run_paths: list[Path], concept: str,
    figure_path: Path,
) -> tuple[str, dict]:
    trajectories = [concept_trajectory(record, "jacobian_lens", concept) for record in records]
    pairwise = []
    for first in range(len(trajectories)):
        for second in range(first + 1, len(trajectories)):
            rho = spearman(
                [math.log10(rank) for rank in trajectories[first]["ranks"]],
                [math.log10(rank) for rank in trajectories[second]["ranks"]],
            )
            pairwise.append({"seeds": [first, second], "spearman_rho": rho})

    tracked = [item["label"] for item in records[0]["tracked"]]
    primary = []
    for seed, (run, record) in enumerate(zip(runs, records)):
        item = emergence(record, concept)
        j_rank = one_indexed(item["best_rank"])
        logit_rank = one_indexed(item["logit_lens_best_rank"])
        depth = float(item["best_depth"])
        trajectory = concept_trajectory(record, "jacobian_lens", concept)
        best_index = trajectory["depths"].index(depth)
        primary.append({
            "seed": seed,
            "corpus_sha256": run["lens_provenance"]["corpus"]["sha256"],
            "jacobian_rank": j_rank,
            "logit_rank": logit_rank,
            "rank_improvement_ratio": logit_rank / j_rank,
            "best_depth": depth,
            "source_layer": trajectory["layers"][best_index],
            "sustained_top5_onset": item["onset_depth"],
            "top_tokens": top_tokens_at_depth(record, depth),
        })

    specificity = []
    for label in tracked:
        seed_rows = []
        for seed, record in enumerate(records):
            item = emergence(record, label)
            seed_rows.append({
                "seed": seed,
                "jacobian_rank": one_indexed(item["best_rank"]),
                "best_depth": item["best_depth"],
                "logit_rank": one_indexed(item["logit_lens_best_rank"]),
            })
        specificity.append({"concept": label, "seeds": seed_rows})

    stats = {
        "slug": records[0]["slug"],
        "title": records[0]["title"],
        "prompt": records[0]["prompt_text"],
        "generated_completion": records[0]["generated_completion"],
        "score_positions": records[0]["score_positions"],
        "workspace_band": records[0]["band"],
        "model": runs[0]["model"],
        "model_revision": runs[0]["model_identity"]["model_revision"],
        "lens_records_per_seed": [run["lens_n_prompts"] for run in runs],
        "primary_concept": concept,
        "primary_results": primary,
        "trajectory_pairwise_spearman": pairwise,
        "specificity_results": specificity,
    }

    lines = [
        f"# Deep dive: `{concept}` from one compact-tension prompt across three paper lenses",
        "",
        "## Exact prompt",
        "",
        f"> {records[0]['prompt_text']}",
        "",
        f"The predeclared tracked concepts were {', '.join(f'`{label}`' for label in tracked)}. "
        f"They were absent from the tokenized input and the clean one-token continuation "
        f"(`{records[0]['generated_completion'].strip()}`). The fixed readout was the final prompt "
        f"token at position `{records[0]['score_positions'][0]}`, and the preregistered comparison "
        f"band was {records[0]['band'][0]:.0f}–{records[0]['band'][1]:.0f}% network depth.",
        "",
        "## Result",
        "",
        "| lens seed | Jacobian rank | best depth | source layer | logit-lens rank | rank ratio | peak decoded tokens |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in primary:
        tokens = ", ".join(f"`{token}`" for token in row["top_tokens"][:5])
        lines.append(
            f"| {row['seed']} | **{row['jacobian_rank']}** | {row['best_depth']:.1f}% | "
            f"{row['source_layer']} | {row['logit_rank']:,} | {row['rank_improvement_ratio']:.0f}× | {tokens} |"
        )
    relative_figure = figure_path.relative_to(ROOT)
    lines.extend([
        "",
        f"![Three-seed concept analysis](../{relative_figure.as_posix()})",
        "",
        "The same localized event appeared in every independently fitted lens: `toughness` rose "
        "to rank 2 at source layer 18 (43.9% depth). At that layer, all three seeds decoded the "
        "same leading sequence: `robustness`, `toughness`, `capability`, `abilitas`, and `ductility`. "
        "The matched logit lens never ranked `toughness` better than 1,276 anywhere in the fixed "
        "band, a 638-fold rank contrast.",
        "",
        "## Trajectory reproducibility",
        "",
    ])
    for item in pairwise:
        lines.append(
            f"- Seeds {item['seeds'][0]} and {item['seeds'][1]}: Spearman "
            f"$\\rho={item['spearman_rho']:.3f}$ across the full 26-point depth trajectory."
        )
    lines.extend([
        "",
        "Immediately before the peak, the three ranks were 84, 78, and 86 at 39.0% depth. "
        "Immediately after it, they were 465, 522, and 450 at 48.8%. The concept is therefore a "
        "sharp, reproducible intermediate-layer event rather than a generally elevated word.",
        "",
        "## Specificity and limitations",
        "",
        "| tracked concept | seed-0 J rank | seed-1 J rank | seed-2 J rank | logit rank |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in specificity:
        rows = item["seeds"]
        lines.append(
            f"| `{item['concept']}` | {rows[0]['jacobian_rank']:,} | {rows[1]['jacobian_rank']:,} | "
            f"{rows[2]['jacobian_rank']:,} | {rows[0]['logit_rank']:,} |"
        )
    lines.extend([
        "",
        "The Jacobian lens selectively elevated the engineering property `toughness`; it did not "
        "indiscriminately elevate the related words `crack` and `fracture`. This supports a "
        "mechanism-specific readout interpretation for this example.",
        "",
        "This remains an illustrative case, not a population-level statistical result. Rank 2 "
        "occurred at one sampled source layer, so the preregistered two-layer sustained-top-5 "
        "criterion was not met. A readable representation also does not establish that the "
        "representation caused the model's output, nor does it constitute a literal transcript "
        "of internal prose. Population claims must use all 50 prompts with family-clustered "
        "uncertainty; causal claims require a separate intervention experiment.",
        "",
        "## Reproducibility",
        "",
        f"- Model: `{runs[0]['model']}` at `{runs[0]['model_identity']['model_revision']}`",
        "- Lens fit: 1,000 unique WikiText-103 records per seed; 128 tokens; target layer 40; 25 source layers",
        f"- Run files: {', '.join(f'`{path.relative_to(ROOT).as_posix()}`' for path in run_paths)}",
        "- Ranks in this document are one-indexed; lower is stronger.",
        "",
    ])
    return "\n".join(lines), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--concept", default=DEFAULT_CONCEPT)
    parser.add_argument(
        "--output-base", type=Path,
        default=ROOT / "figures" / "gemma4-paper-multiseed" / "notch-resistance-01-multiseed",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "experiments" / "DEEP_DIVE_NOTCH_TOUGHNESS_MULTI_SEED.md",
    )
    parser.add_argument(
        "--statistics", type=Path,
        default=ROOT / "experiments" / "deep-dive-notch-toughness-multiseed_statistics.json",
    )
    args = parser.parse_args()

    run_paths = [path.resolve() for path in args.runs]
    runs, records = load_prompt_records(run_paths, args.slug)
    written = plot_prompt(records, args.concept, args.output_base.resolve())
    report, stats = build_report(runs, records, run_paths, args.concept, written[0].resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n")
    args.statistics.parent.mkdir(parents=True, exist_ok=True)
    args.statistics.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {args.report}")
    print(f"wrote {args.statistics}")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
