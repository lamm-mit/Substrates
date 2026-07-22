#!/usr/bin/env python3
"""Create engineering-facing concept traces for fixed positive/negative cases."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "gemma4-materials-key-concept-traces-v1.json"
OUT = ROOT / "figures" / "gemma4-materials-key-concept-traces-v1"

PANELS = [
    ("paper-v2-assoc-notch-resistance-04", "toughness", "Notch resistance", "strong positive"),
    ("paper-v2-assoc-ductile-02", "coalescence", "Ductile failure", "strong positive"),
    ("paper-v2-assoc-line-defect-motion-01", "dislocation", "Line-defect motion", "strong positive"),
    ("paper-v2-assoc-rapid-transformation-04", "tetragonal", "Diffusionless transformation", "strong positive"),
    ("paper-v2-assoc-cleavage-03", "cleavage", "Low-temperature cleavage", "prespecified weak case"),
    ("paper-v2-assoc-hot-air-surface-layer-01", "oxidation", "Hot-air surface reaction", "prespecified weak case"),
]


def find_curve(record: dict, lens: str, label: str) -> dict:
    return next(
        item for item in record["concept_trajectories"][lens]
        if item["label"] == label
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run = json.loads(RUN.read_text())
    by_slug = {record["slug"]: record for record in run["prompts"]}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, (slug, label, panel_title, status) in zip(axes.flat, PANELS):
        record = by_slug[slug]
        j = find_curve(record, "jacobian_lens", label)
        l = find_curve(record, "logit_lens", label)
        jx = np.asarray(j["depths"])
        jy = np.asarray(j["ranks"], dtype=float)
        keep = (jx >= 38) & (jx <= 92) & (jy > 0)
        lx = np.asarray(l["depths"])
        ly = np.asarray(l["ranks"], dtype=float)
        lkeep = ly > 0
        ax.plot(jx[keep], jy[keep], "-o", ms=3.5, lw=2.3, color="#c1553b",
                label="Jacobian lens")
        ax.plot(lx[lkeep], ly[lkeep], "--", lw=2.0, color="#2b7a8c",
                label="Logit lens")
        emergence = next(item for item in record["emergence"] if item["label"] == label)
        jbest = int(emergence["best_rank"]) + 1
        lbest = int(emergence["logit_lens_best_rank"]) + 1
        ax.axhspan(1, 100, color="#f3e7c6", alpha=0.28, zorder=-2)
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_ylim(50000, 0.8)
        ax.set_xlim(37, 93)
        ax.set_yticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1k", "10k"])
        ax.grid(True, which="major", alpha=0.18)
        ax.set_title(f"{panel_title}: '{label}'", loc="left", fontsize=11,
                     fontweight="bold", color="#333333", pad=8)
        ax.text(0.0, 0.97, status, transform=ax.transAxes,
                fontsize=8.0, color="#666666", va="top")
        ax.text(0.98, 0.96, f"best rank\nJ: {jbest:,}\nlogit: {lbest:,}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                      "edgecolor": "#cccccc", "alpha": 0.92})
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("vocabulary rank (higher on page = easier to read)")
    for ax in axes[-1, :]:
        ax.set_xlabel("network depth (%)")
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        "Readable scientific concepts across Gemma-4 E4B-it's layers\n"
        "Tracked words were absent from both prompt and one-token continuation; traces are readouts, not literal private prose",
        fontsize=14, fontweight="bold"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"key-concept-traces.{suffix}", dpi=240,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'key-concept-traces.pdf'}")


if __name__ == "__main__":
    main()
