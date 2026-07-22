#!/usr/bin/env python3
"""Plot the descriptive layer sweep for the disjoint 12-law confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "experiments"
    / "relational-contrast-confirmation-2026-07-18"
    / "descriptive_layer_profile.csv"
)
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "relational-contrast-layer-sweep"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    profile = pd.read_csv(args.input)
    required = {"layer", "orientation_auc"}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    max_layer = int(profile["layer"].max())
    profile["depth"] = 100.0 * profile["layer"] / max_layer
    selected = profile.loc[profile["layer"] == 34]
    if len(selected) != 1:
        raise ValueError("Expected exactly one row for frozen layer 34")
    selected_depth = float(selected.iloc[0]["depth"])
    selected_auc = float(selected.iloc[0]["orientation_auc"])

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(6.7, 2.65))
    ax.plot(
        profile["depth"],
        profile["orientation_auc"],
        color="#276b9a",
        linewidth=1.9,
        marker="o",
        markersize=2.8,
        markeredgewidth=0,
    )
    ax.axhline(0.5, color="0.45", linestyle="--", linewidth=1.0)
    ax.axvline(selected_depth, color="#7b5ea7", linestyle=":", linewidth=1.3)
    ax.scatter(
        [selected_depth],
        [selected_auc],
        color="#7b5ea7",
        edgecolor="white",
        linewidth=0.7,
        s=42,
        zorder=4,
    )
    ax.annotate(
        f"frozen layer 34\nAUC = {selected_auc:.3f}",
        xy=(selected_depth, selected_auc),
        xytext=(selected_depth - 4.0, 0.76),
        ha="right",
        va="top",
        color="#5f4486",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": "#7b5ea7", "lw": 0.9},
    )
    ax.text(1.5, 0.515, "chance", color="0.4", fontsize=7.5, va="bottom")
    ax.set(
        xlabel="normalized layer depth (%)",
        ylabel="direct-versus-inverse ROC-AUC",
        xlim=(0, 100),
        ylim=(0.35, 1.025),
    )
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    fig.tight_layout()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
