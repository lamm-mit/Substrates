#!/usr/bin/env python3
"""Create a plain-language diagram comparing the two readout methods."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "engineering"


def box(ax, xy, width, height, text, *, face, edge="#444444", fontsize=10):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, *, color="#555555", style="-|>", lw=1.6):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                mutation_scale=13, color=color, linewidth=lw))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.03, 0.39), 0.18, 0.22,
        "Engineering prompt\n\n'Large flaw, modest load,\nsudden brittle failure'",
        face="#f2f2f2", fontsize=9.5)
    arrow(ax, (0.21, 0.50), (0.28, 0.50))

    layer_x = [0.29, 0.35, 0.41, 0.47, 0.53]
    for index, x in enumerate(layer_x):
        box(ax, (x, 0.36), 0.045, 0.28, f"L{index+1}",
            face="#dceaf2" if index != 2 else "#f6d8cc", fontsize=9)
        if index < len(layer_x) - 1:
            arrow(ax, (x + 0.045, 0.50), (layer_x[index + 1], 0.50), lw=1.0)
    ax.text(0.42, 0.75, "Gemma hidden state changes across layers",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(0.432, 0.30, "read the hidden vector here",
            ha="center", fontsize=9, color="#8f2f27")

    arrow(ax, (0.432, 0.36), (0.67, 0.72), color="#2b7a8c")
    arrow(ax, (0.432, 0.36), (0.67, 0.28), color="#c1553b")

    box(ax, (0.67, 0.62), 0.18, 0.21,
        "LOGIT LENS\n\nApply final word-output\ncoordinates directly",
        face="#d7ebef", edge="#2b7a8c", fontsize=9.5)
    box(ax, (0.67, 0.17), 0.18, 0.25,
        "JACOBIAN LENS\n\nFirst map through the\naverage downstream effect,\nthen read words",
        face="#f5d8ce", edge="#c1553b", fontsize=8.8)

    arrow(ax, (0.85, 0.725), (0.91, 0.725), color="#2b7a8c")
    arrow(ax, (0.85, 0.295), (0.91, 0.295), color="#c1553b")
    box(ax, (0.90, 0.61), 0.105, 0.23,
        "Ranked words\n\n1: This\n...\n6261: toughness",
        face="#f7f7f7", edge="#2b7a8c", fontsize=7.9)
    box(ax, (0.90, 0.18), 0.105, 0.23,
        "Ranked words\n\n1. ...\n2. toughness\n3. ...",
        face="#f7f7f7", edge="#c1553b", fontsize=8.0)

    ax.text(0.50, 0.03,
            "Both are measurement tools. A high-ranked concept is readable; it is not automatically the cause of the answer.",
            ha="center", fontsize=9.5, color="#444444")
    fig.suptitle("Two ways to translate an intermediate model state into engineering words",
                 fontsize=14, fontweight="bold", y=0.97)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"lens-comparison-explainer.{suffix}", dpi=240,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'lens-comparison-explainer.pdf'}")


if __name__ == "__main__":
    main()
