#!/usr/bin/env python3
"""Rebuild the cross-mechanism patching figure from archived CSV outputs.

This plotting-only script is separate from the protocol-bound analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "experiments"
    / "cross-mechanism-activation-patching-2026-07-18"
)
FIG = OUT / "figures"
FAMILY_SHORT = {
    "obstacle-spacing-orowan": "Orowan",
    "porosity-modulus": "Porosity",
    "pearlite-spacing-strength": "Pearlite",
    "dislocation-density-strength": "Dislocation",
    "particle-fraction-modulus": "Particles",
    "crosslink-density-modulus": "Crosslinks",
}


def main() -> None:
    unordered = pd.read_csv(
        OUT / "unordered_mechanism_pair_effects.csv"
    )
    ordered = pd.read_csv(OUT / "ordered_mechanism_pair_effects.csv")
    receiver_layer = pd.read_csv(OUT / "receiver_layer_contrasts.csv")
    subset = pd.read_csv(OUT / "subset_statistics.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.1,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    teal = "#176B7A"
    orange = "#D17C38"
    purple = "#7B6BA8"
    green = "#5B8C5A"
    gray = "#8B8E91"
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.6))

    pair_table = unordered.sort_values(
        "physical_outcome_contrast", ascending=True
    )
    y = np.arange(len(pair_table))
    axes[0, 0].hlines(
        y,
        pair_table["numeric_direction_contrast"],
        pair_table["physical_outcome_contrast"],
        color="#C9CBCC",
        linewidth=0.9,
    )
    axes[0, 0].scatter(
        pair_table["physical_outcome_contrast"],
        y,
        color=teal,
        s=23,
        label="physical outcome",
        zorder=3,
    )
    axes[0, 0].scatter(
        pair_table["numeric_direction_contrast"],
        y,
        color=orange,
        s=23,
        label="numerical direction",
        zorder=3,
    )
    labels = []
    for pair in pair_table["family_pair"]:
        left, right = pair.split(" | ")
        labels.append(f"{FAMILY_SHORT[left]}–{FAMILY_SHORT[right]}")
    axes[0, 0].set_yticks(y, labels)
    axes[0, 0].axvline(0, color=gray, linestyle="--", linewidth=0.8)
    axes[0, 0].set_xlabel("patched answer-margin contrast")
    axes[0, 0].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
    )

    names = {
        "all": "All pairs",
        "cross_vocabulary": "Different answer words",
        "opposite_orientation": "Reversed numeric trend",
        "both": "Both controls",
    }
    order = [
        "all",
        "cross_vocabulary",
        "opposite_orientation",
        "both",
    ]
    summary = subset.set_index("subset").loc[order]
    x = np.arange(len(summary))
    means = summary["mean"].to_numpy()
    axes[0, 1].errorbar(
        x,
        means,
        yerr=np.vstack(
            [
                means - summary["ci_low"].to_numpy(),
                summary["ci_high"].to_numpy() - means,
            ]
        ),
        fmt="o",
        color=teal,
        ecolor=teal,
        capsize=3,
        markersize=5,
    )
    axes[0, 1].axhline(0, color=gray, linestyle="--", linewidth=0.8)
    axes[0, 1].set_xticks(
        x,
        [names[name] for name in order],
        rotation=18,
        ha="right",
    )
    axes[0, 1].set_ylabel("physical-outcome transfer")

    curve = (
        receiver_layer.groupby(["layer", "depth_percent"], as_index=False)
        .agg(
            physical=("physical_outcome_contrast", "mean"),
            numeric=("numeric_direction_contrast", "mean"),
        )
        .sort_values("depth_percent")
    )
    axes[1, 0].plot(
        curve["depth_percent"],
        curve["physical"],
        color=teal,
        marker="o",
        linewidth=1.5,
        label="physical outcome",
    )
    axes[1, 0].plot(
        curve["depth_percent"],
        curve["numeric"],
        color=orange,
        marker="s",
        linewidth=1.5,
        label="numerical direction",
    )
    axes[1, 0].axhline(0, color=gray, linestyle="--", linewidth=0.8)
    axes[1, 0].set_xlabel("layer depth (%)")
    axes[1, 0].set_ylabel("patched answer-margin contrast")
    axes[1, 0].legend(frameon=False, loc="upper left")

    family = (
        ordered.groupby("donor_family", as_index=False)
        .agg(physical=("physical_outcome_contrast", "mean"))
        .sort_values("physical")
    )
    orientation = {
        "obstacle-spacing-orowan": "inverse",
        "porosity-modulus": "inverse",
        "pearlite-spacing-strength": "inverse",
        "dislocation-density-strength": "direct",
        "particle-fraction-modulus": "direct",
        "crosslink-density-modulus": "direct",
    }
    colors = [
        purple if orientation[row] == "direct" else green
        for row in family["donor_family"]
    ]
    y_family = np.arange(len(family))
    axes[1, 1].barh(
        y_family,
        family["physical"],
        color=colors,
        alpha=0.9,
    )
    axes[1, 1].axvline(0, color=gray, linestyle="--", linewidth=0.8)
    axes[1, 1].set_yticks(
        y_family,
        [FAMILY_SHORT[row] for row in family["donor_family"]],
    )
    axes[1, 1].set_xlabel("physical-outcome transfer")
    axes[1, 1].legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                color=purple,
                label="property rises with input",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                color=green,
                label="property falls with input",
            ),
        ],
        frameon=False,
        loc="lower right",
    )

    for label, axis in zip("ABCD", axes.flat):
        axis.text(
            -0.17,
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    fig.subplots_adjust(
        left=0.19,
        right=0.985,
        bottom=0.12,
        top=0.97,
        wspace=0.37,
        hspace=0.43,
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(
            FIG / f"cross-mechanism-activation-patching.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
