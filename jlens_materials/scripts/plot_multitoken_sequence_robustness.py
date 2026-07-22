#!/usr/bin/env python3
"""Rebuild the multi-token robustness figure from archived CSV outputs.

This plotting-only script is deliberately separate from the protocol-bound
model runner. It changes no scores, endpoints, or verdicts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "multitoken-sequence-robustness-2026-07-18"
FIG = OUT / "figures"


def main() -> None:
    rows = pd.read_csv(OUT / "layer_sequence_scores.csv")
    summary = pd.read_csv(OUT / "prompt_band_summary.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = {"jacobian_ensemble": "#16697A", "direct": "#B56576"}
    family_colors = {
        "cleavage": "#D17C38",
        "rapid-transformation": "#5B8C5A",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.55))

    primary = summary[
        summary["method"].isin(["jacobian_ensemble", "direct"])
    ].copy()
    family_order = ["cleavage", "rapid-transformation"]
    slug_order: list[str] = []
    tick_labels: list[str] = []
    for family, prefix in (("cleavage", "C"), ("rapid-transformation", "M")):
        family_slugs = sorted(
            primary.loc[primary["family"] == family, "slug"].unique()
        )
        slug_order.extend(family_slugs)
        tick_labels.extend(
            [f"{prefix}{index}" for index in range(1, len(family_slugs) + 1)]
        )
    x = np.arange(len(slug_order))
    for offset, method, label in (
        (-0.13, "jacobian_ensemble", "Jacobian"),
        (0.13, "direct", "Direct"),
    ):
        values = (
            primary[primary["method"] == method]
            .set_index("slug")
            .loc[slug_order, "band_sequence_margin"]
        )
        axes[0, 0].scatter(
            x + offset,
            values,
            color=colors[method],
            s=25,
            label=label,
            zorder=3,
        )
    axes[0, 0].axhline(0, color="#8B8E91", linestyle="--", linewidth=0.9)
    axes[0, 0].axvline(4.5, color="#D5D7D9", linewidth=0.8)
    axes[0, 0].set_ylabel("target minus contrast\nsequence score")
    axes[0, 0].set_xticks(x, tick_labels)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0, 0].text(
        -0.15,
        1.04,
        "A",
        transform=axes[0, 0].transAxes,
        fontweight="bold",
        fontsize=10,
    )

    layers = np.sort(rows["layer"].unique())
    depths = (
        rows[["layer", "depth_percent"]]
        .drop_duplicates()
        .set_index("layer")
        .loc[layers, "depth_percent"]
        .to_numpy()
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
    axes[0, 1].legend(frameon=False, fontsize=6.3, loc="best")
    axes[0, 1].text(
        -0.15,
        1.04,
        "B",
        transform=axes[0, 1].transAxes,
        fontweight="bold",
        fontsize=10,
    )

    jacobian = summary[summary["method"] == "jacobian_ensemble"]
    for family in family_order:
        subset = jacobian[jacobian["family"] == family]
        axes[1, 0].scatter(
            subset["band_first_piece_margin"],
            subset["band_sequence_margin"],
            color=family_colors[family],
            s=34,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    low = min(
        float(jacobian["band_first_piece_margin"].min()),
        float(jacobian["band_sequence_margin"].min()),
    )
    high = max(
        float(jacobian["band_first_piece_margin"].max()),
        float(jacobian["band_sequence_margin"].max()),
    )
    axes[1, 0].plot(
        [low, high],
        [low, high],
        color="#8B8E91",
        linestyle="--",
        linewidth=0.9,
    )
    axes[1, 0].set_xlabel("first-piece margin")
    axes[1, 0].set_ylabel("full-sequence margin")
    axes[1, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=family_colors["cleavage"],
                markeredgecolor="white",
                label="transgranular vs intergranular",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=family_colors["rapid-transformation"],
                markeredgecolor="white",
                label="martensite vs bainite",
            ),
        ],
        frameon=False,
        loc="upper left",
        fontsize=6.3,
    )
    axes[1, 0].text(
        -0.15,
        1.04,
        "C",
        transform=axes[1, 0].transAxes,
        fontweight="bold",
        fontsize=10,
    )

    seeds = summary[summary["method"].str.startswith("jacobian_seed")]
    for family, marker in (
        ("cleavage", "o"),
        ("rapid-transformation", "s"),
    ):
        subset = (
            seeds[seeds["family"] == family]
            .groupby("method", as_index=False)["band_sequence_margin"]
            .mean()
            .set_index("method")
        )
        axes[1, 1].scatter(
            [0, 1, 2],
            subset.loc[
                ["jacobian_seed0", "jacobian_seed1", "jacobian_seed2"],
                "band_sequence_margin",
            ],
            color=family_colors[family],
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
    axes[1, 1].legend(frameon=False, fontsize=6.3, loc="best")
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


if __name__ == "__main__":
    main()
