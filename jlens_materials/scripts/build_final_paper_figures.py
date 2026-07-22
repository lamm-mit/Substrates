#!/usr/bin/env python3
"""Build paper Figures 4 and 7 from frozen archived artifacts.

Figure 1 is authored directly in TikZ. Figure 4 compares unrestricted
single-fit streams with filtered three-fit-consensus streams for five exact
held-out prompts. Figure 7 magnifies one development case and one held-out
case without adding a new inferential endpoint.
Population Figures 2, 3, and 6 are built by the held-out analysis scripts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_materials_heldout_v1 import FUNCTION_WORDS  # noqa: E402


RUN_PATHS = [
    ROOT / "runs" / f"gemma4-e4b-it-heldout-v1-seed{seed}.json"
    for seed in range(3)
]
DEV_RUN_PATHS = [
    ROOT / "runs" / f"gemma4-e4b-it-paper-seed{seed}.json"
    for seed in range(3)
]
STATS_PATH = ROOT / "experiments" / "materials-heldout-v1_statistics.json"
DEEP_DIVE_PATH = (
    ROOT / "experiments" / "deep-dive-notch-toughness-multiseed_statistics.json"
)
OUT_DIR = ROOT / "figures" / "materials-heldout-v1"
STREAM_STATS_PATH = ROOT / "experiments" / "materials-heldout-v1_semantic_streams.json"
FIG7_STATS_PATH = ROOT / "experiments" / "figure7-layer-resolved-cases.json"
SUP_STREAM_STEM = OUT_DIR / "figure-s2-development-semantic-stream"
FIG7_STEM = OUT_DIR / "figure7-layer-resolved-cases"

J_COLOR = "#087f8c"
DIRECT_COLOR = "#66727a"
PURPLE = "#6f5aa8"
GOLD = "#d59c32"
GRAY = "#6d747b"
LIGHT_GRAY = "#e8eaed"
FAMILY_COLORS = {
    "boundary-attack": "#b33c49",
    "cleavage": "#557a95",
}
TOKEN_COLORS = ["#176B87", "#2A9D8F", "#6C8EAD", "#7A6AAE", "#79A15A", "#4F86A6"]
STREAM_CASES = [
    ("A", "Boundary attack", "heldout-v1-assoc-boundary-attack-05"),
    ("B", "Notch resistance", "heldout-v1-assoc-notch-resistance-01"),
    ("C", "Line-defect motion", "heldout-v1-assoc-line-defect-motion-04"),
    ("D", "Ductile failure", "heldout-v1-assoc-ductile-03"),
    ("E", "Cleavage", "heldout-v1-assoc-cleavage-03"),
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"{label}  {title}", loc="left", fontweight="bold", pad=8)


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str = "#f7f8fa",
    edge: str = "#a8adb4",
    fontsize: float = 10,
    weight: str = "normal",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        wrap=True,
    )
    return patch


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color=GRAY) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def build_design_figure() -> None:
    """Legacy raster design draft; the manuscript Figure 1 is the TikZ source."""
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(
        ax,
        (0.025, 0.665),
        0.245,
        0.245,
        "UNSEEN MATERIALS DESCRIPTION\n\n"
        "A stainless weld remained near 675 degC.\n"
        "Nitric acid exposed narrow trenches along\n"
        "grain edges beside chromium-rich carbides.\n\n"
        "Omitted before execution:\n"
        "boundary  |  corrosion  |  sensitization",
        face="#f5f1ea",
        edge="#c8ad7f",
        fontsize=9.2,
    )

    box(ax, (0.33, 0.745), 0.235, 0.095, "Gemma-4-E4B-it\n42 transformer layers", face="#edf2f7", edge="#7991a8", weight="bold")
    for i in range(11):
        x = 0.345 + i * 0.018
        color = mpl.colors.to_hex(plt.cm.Blues(0.25 + 0.055 * i))
        ax.add_patch(FancyBboxPatch((x, 0.68), 0.012, 0.045, boxstyle="round,pad=0.002", facecolor=color, edgecolor="none"))
    ax.text(0.447, 0.655, "fixed final-prompt position; 25 registered depths", ha="center", fontsize=8.7, color=GRAY)
    arrow(ax, (0.27, 0.79), (0.33, 0.79))

    box(ax, (0.64, 0.805), 0.31, 0.105, "DIRECT UNEMBEDDING\nintermediate state\n-> vocabulary decoder", face="#edf2f3", edge=DIRECT_COLOR, weight="bold", fontsize=9.5)
    box(ax, (0.64, 0.655), 0.31, 0.105, "JACOBIAN READOUT\nintermediate state -> downstream map\n-> vocabulary decoder", face="#e8f5f4", edge=J_COLOR, weight="bold", fontsize=9.5)
    arrow(ax, (0.565, 0.805), (0.64, 0.857), DIRECT_COLOR)
    arrow(ax, (0.565, 0.745), (0.64, 0.707), J_COLOR)

    box(ax, (0.04, 0.37), 0.19, 0.115, "Three lens fits\n1,000 WikiText records each\nindependent corpus seeds", face="#f1edfa", edge=PURPLE, weight="bold")
    box(ax, (0.285, 0.37), 0.19, 0.115, "Held-out suite\n10 physical families\n5 phrasings each", face="#edf5ee", edge="#5b9465", weight="bold")
    box(ax, (0.53, 0.37), 0.19, 0.115, "Controlled branch\nrank declared absent concepts\nin the full vocabulary", face="#e8f5f4", edge=J_COLOR, weight="bold")
    box(ax, (0.775, 0.37), 0.19, 0.115, "Target-free branch\ndiscover stable words\nwithout the answer list", face="#f4f1e8", edge=GOLD, weight="bold")
    arrow(ax, (0.23, 0.428), (0.285, 0.428))
    arrow(ax, (0.475, 0.445), (0.53, 0.445))
    ax.add_patch(
        FancyArrowPatch(
            (0.475, 0.465),
            (0.775, 0.465),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=GRAY,
            connectionstyle="arc3,rad=-0.28",
        )
    )

    box(ax, (0.105, 0.135), 0.25, 0.105, "Reproducibility\nDo independent fits preserve\nconcept ranks?", face="#f7f7f7", fontsize=8.8)
    box(ax, (0.375, 0.135), 0.25, 0.105, "Mechanism selectivity\nWhich physical families\nimprove or fail?", face="#f7f7f7", fontsize=8.8)
    box(ax, (0.645, 0.135), 0.25, 0.105, "Blinded interpretation\nCan discovered words identify\nthe physical family?", face="#f7f7f7", fontsize=8.8)
    for x in (0.23, 0.5, 0.77):
        arrow(ax, (x, 0.37), (x, 0.24))
    save_all(fig, OUT_DIR / "figure1-study-design")


def normalize_token(token: str) -> str | None:
    value = token.strip().lower()
    if len(value) < 3 or not value.isascii() or not value.isalpha():
        return None
    return value


def record_words(record: dict) -> set[str]:
    text = f"{record.get('prompt_text', '')} {record.get('generated_completion', '')}"
    return set(re.findall(r"[a-z]+", text.lower()))


def filtered_ranked_tokens(record: dict, layer_index: int, scaffold: set[str]) -> list[str]:
    forbidden = record_words(record) | FUNCTION_WORDS | scaffold
    values = []
    for raw in record["layer_readouts"][layer_index]["top_tokens"]:
        token = normalize_token(raw)
        if token is not None and token not in forbidden and token not in values:
            values.append(token)
    return values


def family_stream(
    indexes: list[dict[str, dict]], family: str, scaffold: set[str],
    display_tokens: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, dict]:
    slugs = [slug for slug, record in indexes[0].items() if record["target_family"] == family]
    first = indexes[0][slugs[0]]
    depths = np.asarray([row["depth"] for row in first["layer_readouts"]], dtype=float)
    scores: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(depths), dtype=float))
    prompt_support: dict[str, set[str]] = defaultdict(set)
    agreement = np.zeros(len(depths), dtype=float)

    for slug in slugs:
        records = [index[slug] for index in indexes]
        for li in range(len(depths)):
            ranked = [filtered_ranked_tokens(record, li, scaffold) for record in records]
            sets = [set(row) for row in ranked]
            pairwise = []
            for a, b in ((0, 1), (0, 2), (1, 2)):
                union = sets[a] | sets[b]
                pairwise.append(len(sets[a] & sets[b]) / len(union) if union else 1.0)
            agreement[li] += float(np.mean(pairwise)) / len(slugs)
            consensus = sets[0] & sets[1] & sets[2]
            for token in consensus:
                reciprocal = np.mean([1.0 / (ranked[seed].index(token) + 1) for seed in range(3)])
                scores[token][li] += float(reciprocal) / len(slugs)
                prompt_support[token].add(slug)

    band = (depths >= 38.0) & (depths <= 92.0)
    integrated = {
        token: float(np.trapezoid(values[band], depths[band]))
        for token, values in scores.items()
    }
    selected = [token.lower() for token in display_tokens[:6]]
    matrix = np.asarray([scores[token] for token in selected], dtype=float)
    meta = {
        "family": family,
        "slugs": slugs,
        "selected_tokens": [
            {
                "token": token,
                "integrated_prominence": integrated.get(token, 0.0),
                "phrasing_support": len(prompt_support.get(token, set())),
            }
            for token in selected
        ],
    }
    return depths, selected, matrix, agreement, meta


def _select_stream_tokens(
    depths: np.ndarray,
    scores: dict[str, np.ndarray],
    *,
    limit: int,
    band_only: bool,
) -> tuple[list[str], np.ndarray, list[dict]]:
    keep = (depths >= 38.0) & (depths <= 92.0) if band_only else np.ones(len(depths), dtype=bool)
    integrated = {
        token: float(np.trapezoid(values[keep], depths[keep]))
        for token, values in scores.items()
        if np.any(values[keep] > 0)
    }
    selected = [token for token, _ in sorted(integrated.items(), key=lambda item: item[1], reverse=True)[:limit]]
    first_layer = {
        token: next((index for index, value in enumerate(scores[token]) if value > 0), len(depths))
        for token in selected
    }
    selected.sort(key=lambda token: (first_layer[token], -integrated[token]))
    matrix = np.asarray([scores[token] for token in selected], dtype=float)
    metadata = [
        {
            "token": token,
            "integrated_prominence": integrated[token],
            "first_nonzero_depth": float(depths[first_layer[token]]),
        }
        for token in selected
    ]
    return selected, matrix, metadata


def unfiltered_prompt_stream(record: dict) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    """Original-style stream: leading words from one fit, with no lexical filters."""
    depths = np.asarray([row["depth"] for row in record["layer_readouts"]], dtype=float)
    width = max(len(row["top_tokens"]) for row in record["layer_readouts"])
    scores: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(depths), dtype=float))
    for layer_index, row in enumerate(record["layer_readouts"]):
        seen: set[str] = set()
        for rank, raw in enumerate(row["top_tokens"]):
            token = normalize_token(raw)
            if token is None or token in seen:
                continue
            seen.add(token)
            scores[token][layer_index] = max(
                scores[token][layer_index],
                (width - rank) / width,
            )
    selected, matrix, metadata = _select_stream_tokens(
        depths, scores, limit=7, band_only=False,
    )
    return depths, selected, matrix, {
        "protocol": "seed-0 leading decoded words; no input, output, function-word, or scaffold filter",
        "selected_tokens": metadata,
    }


def filtered_prompt_stream(
    records: list[dict], scaffold: set[str],
) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    """Strict target-free stream: input/output removal plus three-fit consensus."""
    depths = np.asarray([row["depth"] for row in records[0]["layer_readouts"]], dtype=float)
    scores: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(depths), dtype=float))
    layer_consensus_sizes: list[int] = []
    for layer_index in range(len(depths)):
        ranked = [filtered_ranked_tokens(record, layer_index, scaffold) for record in records]
        consensus = set(ranked[0]) & set(ranked[1]) & set(ranked[2])
        layer_consensus_sizes.append(len(consensus))
        for token in consensus:
            scores[token][layer_index] = float(
                np.mean([1.0 / (values.index(token) + 1) for values in ranked])
            )
    selected, matrix, metadata = _select_stream_tokens(
        depths, scores, limit=7, band_only=True,
    )
    return depths, selected, matrix, {
        "protocol": "input/output, frozen function-word, and global-scaffold filters; intersection across all three fits",
        "selected_tokens": metadata,
        "consensus_words_per_layer": layer_consensus_sizes,
    }


def stream_panel(
    ax: plt.Axes,
    depths: np.ndarray,
    tokens: list[str],
    matrix: np.ndarray,
    title: str = "",
    letter: str = "",
    *,
    show_xlabel: bool = True,
) -> None:
    if letter or title:
        panel_label(ax, letter, title)
    ax.axvspan(38, 92, color="#eceff3", alpha=0.6, zorder=0)
    if not tokens or matrix.size == 0 or not np.any(matrix):
        ax.text(0.5, 0.5, "No stable target-free words", transform=ax.transAxes, ha="center", va="center")
    else:
        # Choose label depths independently of ribbon order.  The token set and
        # ribbon values are unchanged; this only prevents two long labels from
        # being printed at the same horizontal position in the paper rendering.
        label_indices: dict[int, int] = {}
        occupied: list[tuple[float, str]] = []
        strength_order = sorted(
            range(len(tokens)), key=lambda index: float(np.max(matrix[index])), reverse=True,
        )
        for index in strength_order:
            candidates = [
                int(candidate)
                for candidate in np.argsort(matrix[index])[::-1]
                if matrix[index, candidate] > 0 and 6 <= depths[candidate] <= 94
            ]
            if not candidates:
                continue

            def clears_existing(candidate: int) -> bool:
                x_value = float(depths[candidate])
                return all(
                    abs(x_value - used_x) >= 4.5 + 0.48 * (len(tokens[index]) + len(used_token))
                    for used_x, used_token in occupied
                )

            chosen = next((candidate for candidate in candidates if clears_existing(candidate)), None)
            if chosen is None:
                # A dense panel may make complete separation impossible.  Use
                # the strongest position with the largest clearance instead of
                # silently dropping or manually selecting a scientific term.
                chosen = max(
                    candidates,
                    key=lambda candidate: (
                        min(
                            (abs(float(depths[candidate]) - used_x) for used_x, _ in occupied),
                            default=100.0,
                        ),
                        float(matrix[index, candidate]),
                    ),
                )
            label_indices[index] = chosen
            occupied.append((float(depths[chosen]), tokens[index]))

        total = matrix.sum(axis=0)
        lower = -total / 2.0
        for index, token in enumerate(tokens):
            upper = lower + matrix[index]
            ax.fill_between(depths, lower, upper, color=TOKEN_COLORS[index % len(TOKEN_COLORS)],
                            alpha=0.93, linewidth=0.7, edgecolor="white")
            label_index = label_indices.get(index)
            if label_index is not None:
                label_x = float(depths[label_index])
                ax.text(
                    label_x, (lower[label_index] + upper[label_index]) / 2.0, token,
                    ha="center", va="center", fontsize=7.6, color="#17222A",
                    path_effects=[path_effects.withStroke(linewidth=1.35, foreground="white")],
                )
            lower = upper
    ax.set_xlim(0, 100)
    ax.set_xlabel("normalized layer depth" if show_xlabel else "")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", alpha=0.12)
    ax.margins(y=0.08)


def build_stream_figure() -> None:
    runs = [json.loads(path.read_text()) for path in RUN_PATHS]
    indexes = [{record["slug"]: record for record in run["prompts"]} for run in runs]
    stats = json.loads(STATS_PATH.read_text())
    scaffold = {
        row["token"]
        for row in stats["open_vocabulary"]["methods"]["jacobian"]["global_scaffold"]
    }
    paired_cases = []
    fig, axes = plt.subplots(
        len(STREAM_CASES), 2, figsize=(13.5, 12.8), sharex=True,
        constrained_layout=True,
    )
    for row_index, (letter, family_label, slug) in enumerate(STREAM_CASES):
        records = [index[slug] for index in indexes]
        raw = unfiltered_prompt_stream(records[0])
        filtered = filtered_prompt_stream(records, scaffold)
        stream_panel(
            axes[row_index, 0], raw[0], raw[1], raw[2],
            show_xlabel=row_index == len(STREAM_CASES) - 1,
        )
        stream_panel(
            axes[row_index, 1], filtered[0], filtered[1], filtered[2],
            show_xlabel=row_index == len(STREAM_CASES) - 1,
        )
        axes[row_index, 0].text(
            0.0, 1.04, f"{letter}  {family_label}",
            transform=axes[row_index, 0].transAxes,
            ha="left", va="bottom", fontsize=10, fontweight="bold",
        )
        paired_cases.append({
            "panel": letter,
            "family_label": family_label,
            "slug": slug,
            "prompt": records[0]["prompt_text"],
            "generated_completion": records[0].get("generated_completion", ""),
            "unfiltered_single_fit": raw[3],
            "filtered_three_fit_consensus": filtered[3],
        })
    axes[0, 0].text(
        0.5, 1.28, "Unfiltered single fit", transform=axes[0, 0].transAxes,
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )
    axes[0, 1].text(
        0.5, 1.28, "Filtered three-fit consensus", transform=axes[0, 1].transAxes,
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )
    fig.supylabel("rank-derived display prominence", x=0.005, fontsize=10)
    save_all(fig, OUT_DIR / "figure4-semantic-streams")

    payload = {
        "analysis_status": "retrospective exploratory semantic-stream visualization",
        "protocol": "experiments/materials-heldout-v1-semantic-stream-protocol.md",
        "raw_run_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in RUN_PATHS},
        "function_word_count": len(FUNCTION_WORDS),
        "global_scaffold": sorted(scaffold),
        "paired_prompt_cases": paired_cases,
    }
    STREAM_STATS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def build_development_stream_supplement() -> None:
    """Clean supplementary rendering of the original coalescence ThemeRiver."""
    run = json.loads((ROOT / "runs" / "gemma4-materials-paper-v2.json").read_text())
    record = next(row for row in run["prompts"] if row["slug"] == "paper-v2-assoc-ductile-02")
    depths = np.asarray([row["depth"] for row in record["layer_readouts"]], dtype=float)
    width = max(len(row["top_tokens"]) for row in record["layer_readouts"])
    series: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(depths), dtype=float))
    for layer_index, row in enumerate(record["layer_readouts"]):
        for rank, raw in enumerate(row["top_tokens"]):
            token = normalize_token(raw)
            if token is not None:
                series[token][layer_index] = float(width - rank)
    selected = [token for token, _ in sorted(series.items(), key=lambda item: float(np.sum(item[1])), reverse=True)[:12]]
    matrix = np.asarray([series[token] for token in selected], dtype=float)
    first = lambda token: next((index for index, value in enumerate(series[token]) if value > 0), len(depths))
    selected.sort(key=first)
    matrix = np.asarray([series[token] for token in selected], dtype=float)
    fig, ax = plt.subplots(figsize=(12.8, 5.0), constrained_layout=True)
    stream_panel(ax, depths, selected, matrix, "", "")
    save_all(fig, SUP_STREAM_STEM)


def trajectory(record: dict, concept: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    row = next(item for item in record["concept_trajectories"][method] if item["label"] == concept)
    return np.asarray(row["depths"], dtype=float), np.asarray(row["ranks"], dtype=float)


def case_payload(
    *,
    title: str,
    slug: str,
    run_paths: list[Path],
    primary: str,
    alternatives: list[str],
    selection_status: str,
    selection_rationale: str,
) -> tuple[dict, list[dict]]:
    runs = [json.loads(path.read_text()) for path in run_paths]
    records = [next(record for record in run["prompts"] if record["slug"] == slug) for run in runs]
    concepts = [primary, *alternatives]
    concept_rows = []
    for concept in concepts:
        seeds = []
        for seed, record in enumerate(records):
            depths, ranks = trajectory(record, concept, "jacobian_lens")
            in_band = (depths >= 38) & (depths <= 92)
            band_indexes = np.flatnonzero(in_band)
            best_index = int(band_indexes[np.argmin(ranks[in_band])])
            seeds.append(
                {
                    "seed": seed,
                    "best_rank": int(ranks[best_index]),
                    "best_depth": float(depths[best_index]),
                    "depths": depths.tolist(),
                    "ranks": ranks.astype(int).tolist(),
                }
            )
        direct_depths, direct_ranks = trajectory(records[0], concept, "logit_lens")
        direct_in_band = (direct_depths >= 38) & (direct_depths <= 92)
        direct_indexes = np.flatnonzero(direct_in_band)
        direct_best_index = int(direct_indexes[np.argmin(direct_ranks[direct_in_band])])
        concept_rows.append(
            {
                "concept": concept,
                "jacobian": seeds,
                "direct": {
                    "best_rank": int(direct_ranks[direct_best_index]),
                    "best_depth": float(direct_depths[direct_best_index]),
                    "depths": direct_depths.tolist(),
                    "ranks": direct_ranks.astype(int).tolist(),
                },
            }
        )

    primary_row = concept_rows[0]
    peak_depth = primary_row["jacobian"][0]["best_depth"]
    peak_tokens = next(
        row["top_tokens"]
        for row in records[0]["layer_readouts"]
        if math.isclose(float(row["depth"]), peak_depth)
    )
    payload = {
        "title": title,
        "slug": slug,
        "prompt": records[0]["prompt_text"],
        "generated_completion": records[0]["generated_completion"].strip(),
        "primary_concept": primary,
        "predeclared_concepts": concepts,
        "selection_status": selection_status,
        "selection_rationale": selection_rationale,
        "concepts_fixed_before_execution": True,
        "concepts_absent_from_input_and_one_token_continuation": True,
        "leading_seed0_tokens_at_primary_peak": peak_tokens[:12],
        "concept_results": concept_rows,
        "raw_runs": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in run_paths
        ],
    }
    return payload, records


def prompt_card(ax: plt.Axes, label: str, case: dict, accent: str) -> None:
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.005, 0.04), 0.99, 0.90,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor="#f7f8fa",
            edgecolor=accent,
            linewidth=1.2,
        )
    )
    ax.text(0.018, 0.82, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.050, 0.82, case["title"], transform=ax.transAxes, fontsize=9.8, fontweight="bold", va="top")
    ax.text(
        0.985, 0.82, case["selection_status"], transform=ax.transAxes,
        fontsize=8.1, fontweight="bold", color=accent, ha="right", va="top",
    )
    ax.text(
        0.018, 0.57, textwrap.fill(case["prompt"], width=132),
        transform=ax.transAxes, fontsize=8.8, va="top", linespacing=1.08,
    )
    declared = "  |  ".join(case["predeclared_concepts"])
    ax.text(
        0.018, 0.13,
        f"Predeclared and absent:  {declared}     one-token continuation: {case['generated_completion']!r}",
        transform=ax.transAxes, fontsize=8.0, color=GRAY, va="bottom",
    )


def rank_panels(
    ax_traj: plt.Axes,
    ax_spec: plt.Axes,
    case: dict,
    records: list[dict],
    trajectory_label: str,
    specificity_label: str,
    *,
    show_legend: bool,
) -> None:
    primary = case["primary_concept"]
    seed_colors = ["#c4553d", "#2c7f91", "#6f58a6"]

    panel_label(ax_traj, trajectory_label, "")
    ax_traj.axvspan(38, 92, color="#eceff3", alpha=0.65, label="registered band")
    for seed, record in enumerate(records):
        x, y = trajectory(record, primary, "jacobian_lens")
        ax_traj.plot(
            x, y, marker="o", markersize=3.2, linewidth=1.65,
            color=seed_colors[seed], label=f"Jacobian fit {seed}",
        )
    x_logit, y_logit = trajectory(records[0], primary, "logit_lens")
    ax_traj.plot(
        x_logit, y_logit, marker="s", markersize=3.1, linewidth=1.5,
        linestyle="--", color="#777777", label="direct unembedding",
    )
    peak_depth = case["concept_results"][0]["jacobian"][0]["best_depth"]
    ax_traj.axvline(peak_depth, color=J_COLOR, linewidth=0.9, linestyle=":", alpha=0.8)
    ax_traj.set_yscale("log")
    ax_traj.invert_yaxis()
    ax_traj.set_xlim(0, 100)
    # Rank 1 is the physical ceiling, but placing it exactly on the axes limit
    # clips the stroke.  A small amount of logarithmic headroom keeps sustained
    # rank-1 traces fully visible without changing the rank scale.
    ax_traj.set_ylim(3e5, 0.7)
    ax_traj.set_xlabel("normalized layer depth")
    ax_traj.set_ylabel(f"rank of {primary!r} (lower is stronger)")
    best_ranks = [row["best_rank"] for row in case["concept_results"][0]["jacobian"]]
    rank_summary = best_ranks[0] if len(set(best_ranks)) == 1 else f"{min(best_ranks)}-{max(best_ranks)}"
    ax_traj.text(
        0.985, 0.975, f"best Jacobian rank: {rank_summary} in all 3 fits",
        transform=ax_traj.transAxes, ha="right", va="top", fontsize=8.4,
        color=J_COLOR, fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.2},
    )
    if show_legend:
        ax_traj.legend(loc="lower left", frameon=True, framealpha=0.92, ncol=2, fontsize=8.0)

    panel_label(ax_spec, specificity_label, "")
    concepts = case["predeclared_concepts"]
    y_positions = np.arange(len(concepts))
    for seed, color in enumerate(seed_colors):
        ranks = [row["jacobian"][seed]["best_rank"] for row in case["concept_results"]]
        ax_spec.scatter(
            ranks, y_positions + (seed - 1) * 0.08, s=50,
            color=color, edgecolor="white", linewidth=0.5,
        )
    direct = [row["direct"]["best_rank"] for row in case["concept_results"]]
    ax_spec.scatter(
        direct, y_positions, s=60, facecolor="white", edgecolor="#777777",
        marker="s", linewidth=1.5,
    )
    ax_spec.set_xscale("log")
    ax_spec.invert_xaxis()
    ax_spec.set_yticks(y_positions, concepts)
    ax_spec.invert_yaxis()
    ax_spec.set_xlabel("best rank in registered band (right is stronger)")
    ax_spec.grid(axis="x", alpha=0.22)


def build_layer_resolved_cases_figure() -> None:
    dive = json.loads(DEEP_DIVE_PATH.read_text())
    toughness, toughness_records = case_payload(
        title="Notch resistance development example",
        slug=dive["slug"],
        run_paths=DEV_RUN_PATHS,
        primary="toughness",
        alternatives=["fracture", "crack"],
        selection_status="development only - excluded from population inference",
        selection_rationale=("Earlier development case retained to make the controlled-rank "
                             "measurement tangible; it is not one of the 50 held-out prompts."),
    )
    corrosion, corrosion_records = case_payload(
        title="Boundary attack held-out example",
        slug="heldout-v1-assoc-boundary-attack-05",
        run_paths=RUN_PATHS,
        primary="corrosion",
        alternatives=["sensitization", "boundary"],
        selection_status="held-out - already counted in Figure 2",
        selection_rationale=("Post hoc magnification of the strongest held-out boundary-attack "
                             "controlled-rank event; it adds no new endpoint or observation."),
    )

    artifact = {
        "analysis_status": "post hoc explanatory display; no new inferential endpoint",
        "relationship_to_population_analysis": (
            "Figure 2 aggregates all held-out prompt-concept pairs. Figure 7 preserves the "
            "layer and lens-fit dimensions for two examples; the held-out example was already "
            "counted in Figure 2 and the development example is excluded from its statistics."
        ),
        "cases": [toughness, corrosion],
    }
    FIG7_STATS_PATH.write_text(json.dumps(artifact, indent=2) + "\n")

    fig = plt.figure(figsize=(13.7, 10.1))
    gs = fig.add_gridspec(
        4, 2,
        width_ratios=[1.65, 1.0],
        height_ratios=[0.46, 1.0, 0.46, 1.0],
        hspace=0.36,
        wspace=0.27,
    )
    prompt_card(fig.add_subplot(gs[0, :]), "A", toughness, PURPLE)
    rank_panels(
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
        toughness, toughness_records, "B", "C", show_legend=True,
    )
    prompt_card(fig.add_subplot(gs[2, :]), "D", corrosion, J_COLOR)
    rank_panels(
        fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]),
        corrosion, corrosion_records, "E", "F", show_legend=False,
    )
    save_all(fig, FIG7_STEM)


def main() -> None:
    configure_style()
    build_stream_figure()
    build_development_stream_supplement()
    build_layer_resolved_cases_figure()
    print("wrote Figures 4, 7, and Supplementary Figure S2 to", OUT_DIR)


if __name__ == "__main__":
    main()
