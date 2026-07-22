#!/usr/bin/env python3
"""Analyze and visualize the frozen mechanism-versus-eponym experiment."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "gemma4-materials-mechanism-name-v1.json"
OUT_JSON = ROOT / "experiments" / "gemma4-materials-mechanism-name-v1_statistics.json"
OUT_MD = ROOT / "experiments" / "gemma4-materials-mechanism-name-v1_results.md"
APPENDIX_MD = ROOT / "experiments" / "gemma4-materials-mechanism-name-v1_appendix.md"
FIG_DIR = ROOT / "figures" / "gemma4-materials-mechanism-name-v1"
SEED = 20260710
N_BOOT = 20_000


PAIRS = {
    "grain-size-strength": ("boundary", "Hall"),
    "flaw-controlled-failure": ("crack", "Griffith"),
    "resolved-slip-threshold": ("dislocation", "Schmid"),
    "lattice-diffusion-creep": ("vacancy", "Herring"),
    "diffusionless-lattice-change": ("tetragonal", "Bain"),
}

FAMILY_LABELS = {
    "grain-size-strength": "Grain-size strengthening",
    "flaw-controlled-failure": "Flaw-controlled failure",
    "resolved-slip-threshold": "Resolved slip threshold",
    "lattice-diffusion-creep": "Lattice-diffusion creep",
    "diffusionless-lattice-change": "Diffusionless lattice change",
}


def bootstrap(values: list[float], seed: int = SEED) -> dict:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def directional_test(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    try:
        statistic, pvalue = wilcoxon(x, alternative="less", zero_method="wilcox")
        statistic, pvalue = float(statistic), float(pvalue)
    except ValueError:
        statistic, pvalue = None, None
    return {
        "effect": bootstrap(values),
        "mechanism_wins": int(np.sum(x < 0)),
        "ties": int(np.sum(np.abs(x) < 1e-12)),
        "name_wins": int(np.sum(x > 0)),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_one_sided": pvalue,
    }


def analyze(run: dict) -> tuple[dict, list[dict]]:
    rows = []
    for record in run["prompts"]:
        family = record.get("category")
        if not record.get("valid_for_metrics") or family not in PAIRS:
            continue
        mechanism, name = PAIRS[family]
        emergence = {item["label"]: item for item in record["emergence"]}
        mech = emergence[mechanism]
        eponym = emergence[name]
        row = {
            "slug": record["slug"],
            "family": family,
            "family_label": FAMILY_LABELS[family],
            "prompt": record["prompt_text"],
            "completion": record.get("generated_completion", ""),
            "mechanism": mechanism,
            "name": name,
            "j_mechanism_rank": int(mech["best_rank"]) + 1,
            "j_name_rank": int(eponym["best_rank"]) + 1,
            "logit_mechanism_rank": int(mech["logit_lens_best_rank"]) + 1,
            "logit_name_rank": int(eponym["logit_lens_best_rank"]) + 1,
            "j_contrast": math.log10(int(mech["best_rank"]) + 1)
                          - math.log10(int(eponym["best_rank"]) + 1),
            "logit_contrast": math.log10(int(mech["logit_lens_best_rank"]) + 1)
                              - math.log10(int(eponym["logit_lens_best_rank"]) + 1),
        }
        row["j_minus_logit_contrast"] = row["j_contrast"] - row["logit_contrast"]
        rows.append(row)

    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    stats = {
        "n_valid": len(rows),
        "n_total": len(run["prompts"]),
        "contrast_definition": "log10(mechanism rank) - log10(eponym rank); negative favors mechanism",
        "jacobian_lens": directional_test([row["j_contrast"] for row in rows]),
        "logit_lens": directional_test([row["logit_contrast"] for row in rows]),
        "j_minus_logit": directional_test([row["j_minus_logit_contrast"] for row in rows]),
        "families": {},
    }
    for family, family_rows in sorted(by_family.items()):
        stats["families"][family] = {
            "label": FAMILY_LABELS[family],
            "n": len(family_rows),
            "mechanism": PAIRS[family][0],
            "name": PAIRS[family][1],
            "jacobian_lens": directional_test([row["j_contrast"] for row in family_rows]),
            "logit_lens": directional_test([row["logit_contrast"] for row in family_rows]),
        }
    return stats, rows


def plot_summary(stats: dict, rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)

    ax = axes[0]
    j = np.asarray([row["j_contrast"] for row in rows])
    l = np.asarray([row["logit_contrast"] for row in rows])
    for index, (a, b) in enumerate(zip(j, l)):
        ax.plot([0, 1], [a, b], color="#b7b7b7", alpha=0.38, lw=0.7)
    ax.scatter(np.zeros(len(j)), j, color="#c1553b", s=24, alpha=0.75, label="J-lens items")
    ax.scatter(np.ones(len(l)), l, color="#2b7a8c", s=24, alpha=0.75, label="logit-lens items")
    ax.scatter([0, 1], [j.mean(), l.mean()], marker="D", s=80,
               color=["#8f2f27", "#185c6b"], edgecolor="white", zorder=5)
    ax.axhline(0, color="#555555", lw=1)
    ax.set_xticks([0, 1], ["Jacobian lens", "Logit lens"])
    ax.set_ylabel("log10 rank(mechanism) - log10 rank(name)")
    ax.set_title("A  Physical mechanism versus textbook name", loc="left", fontweight="bold")
    ax.text(0.02, 0.02, "below zero = physical carrier is easier to read",
            transform=ax.transAxes, fontsize=9, color="#555555")
    ax.grid(True, axis="y", alpha=0.18)

    ax = axes[1]
    families = list(stats["families"])
    y = np.arange(len(families))
    j_mean = [stats["families"][f]["jacobian_lens"]["effect"]["mean"] for f in families]
    j_low = [stats["families"][f]["jacobian_lens"]["effect"]["low"] for f in families]
    j_high = [stats["families"][f]["jacobian_lens"]["effect"]["high"] for f in families]
    l_mean = [stats["families"][f]["logit_lens"]["effect"]["mean"] for f in families]
    ax.errorbar(j_mean, y - 0.12,
                xerr=[np.asarray(j_mean)-np.asarray(j_low), np.asarray(j_high)-np.asarray(j_mean)],
                fmt="o", color="#c1553b", capsize=3, label="Jacobian lens")
    ax.scatter(l_mean, y + 0.12, marker="s", color="#2b7a8c", label="Logit lens")
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(y, [FAMILY_LABELS[f] for f in families])
    ax.set_xlabel("mean mechanism-minus-name log-rank")
    ax.set_title("B  Effect by engineering principle", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(True, axis="x", alpha=0.18)

    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mechanism-name-summary.{suffix}", dpi=240,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _trajectory_map(record: dict, lens: str) -> dict[str, dict]:
    return {
        item["label"]: item
        for item in record["concept_trajectories"][lens]
    }


def plot_concept_traces(run: dict) -> None:
    representatives = [
        next(record for record in run["prompts"]
             if record["slug"] == f"mechanism-name-{family}-01")
        for family in PAIRS
    ]
    fig, axes = plt.subplots(5, 1, figsize=(10.5, 12.2), sharex=True,
                             constrained_layout=True)
    for ax, record in zip(axes, representatives):
        family = record["category"]
        mechanism, name = PAIRS[family]
        j = _trajectory_map(record, "jacobian_lens")
        l = _trajectory_map(record, "logit_lens")
        for label, color in ((mechanism, "#c1553b"), (name, "#6a51a3")):
            jx = np.asarray(j[label]["depths"])
            jy = np.asarray(j[label]["ranks"], dtype=float)
            keep = (jx >= 38) & (jx <= 92) & (jy > 0)
            ax.plot(jx[keep], jy[keep], "-o", ms=2.8, lw=2, color=color,
                    label=f"{label}: Jacobian")
            lx = np.asarray(l[label]["depths"])
            ly = np.asarray(l[label]["ranks"], dtype=float)
            keep = ly > 0
            ax.plot(lx[keep], ly[keep], "--", lw=1.5, color=color, alpha=0.72,
                    label=f"{label}: logit")
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_ylim(50000, 0.8)
        ax.set_yticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1k", "10k"])
        ax.grid(True, which="major", alpha=0.18)
        ax.set_ylabel("rank")
        ax.set_title(
            f"{FAMILY_LABELS[family]}: physical carrier '{mechanism}' vs name '{name}'",
            loc="left", fontsize=10, fontweight="bold"
        )
        ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="lower left")
    axes[-1].set_xlabel("network depth (%) - moving upward means easier to read")
    fig.suptitle(
        "Predeclared concept traces through Gemma-4 E4B-it\n"
        "Solid = Jacobian lens; dashed = ordinary logit lens; these are readouts, not literal chain of thought",
        fontsize=13, fontweight="bold"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mechanism-name-concept-traces.{suffix}", dpi=240,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_outputs(stats: dict, rows: list[dict]) -> None:
    OUT_JSON.write_text(json.dumps(stats, indent=2) + "\n")
    j = stats["jacobian_lens"]
    l = stats["logit_lens"]
    delta = stats["j_minus_logit"]
    lines = [
        "# Mechanism-versus-eponym v1: frozen results",
        "",
        f"Valid items: {stats['n_valid']}/{stats['n_total']}.",
        "",
        "Negative contrasts mean that the physical carrier is ranked ahead of the textbook surname.",
        "",
        "| lens | mean contrast | bootstrap 95% CI | carrier wins | ties | name wins | one-sided p |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Jacobian | {j['effect']['mean']:.3f} | [{j['effect']['low']:.3f}, {j['effect']['high']:.3f}] | {j['mechanism_wins']} | {j['ties']} | {j['name_wins']} | {j['wilcoxon_p_one_sided']:.4g} |",
        f"| Logit | {l['effect']['mean']:.3f} | [{l['effect']['low']:.3f}, {l['effect']['high']:.3f}] | {l['mechanism_wins']} | {l['ties']} | {l['name_wins']} | {l['wilcoxon_p_one_sided']:.4g} |",
        "",
        f"The J-minus-logit contrast is {delta['effect']['mean']:.3f} "
        f"(95% CI [{delta['effect']['low']:.3f}, {delta['effect']['high']:.3f}], "
        f"one-sided p={delta['wilcoxon_p_one_sided']:.4g}).",
        "",
        "## Families",
        "",
        "| family | pair | J mean | logit mean |",
        "|---|---|---:|---:|",
    ]
    for family, item in stats["families"].items():
        lines.append(
            f"| {item['label']} | {item['mechanism']} vs {item['name']} | "
            f"{item['jacobian_lens']['effect']['mean']:.3f} | "
            f"{item['logit_lens']['effect']['mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This experiment distinguishes readable physical carriers from readable eponyms. It does not demonstrate a complete derivation or a human-like hidden chain of thought. The reused lens remains exploratory because its fitting corpus provenance is incomplete.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n")

    appendix = [
        "# Mechanism-versus-eponym v1: complete prompt and result appendix",
        "",
        "Ranks are one-indexed; rank 1 is strongest. Every prompt and one-token completion is shown.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        appendix.extend([
            f"## {index}. {row['slug']}",
            "",
            f"**Prompt:** {row['prompt']}",
            "",
            f"**Generated continuation:** `{row['completion']}`",
            "",
            f"**Tracked pair:** `{row['mechanism']}` vs `{row['name']}`",
            "",
            "| readout | mechanism rank | name rank | log10 contrast |",
            "|---|---:|---:|---:|",
            f"| Jacobian lens | {row['j_mechanism_rank']} | {row['j_name_rank']} | {row['j_contrast']:.3f} |",
            f"| Logit lens | {row['logit_mechanism_rank']} | {row['logit_name_rank']} | {row['logit_contrast']:.3f} |",
            "",
        ])
    APPENDIX_MD.write_text("\n".join(appendix) + "\n")


def main() -> None:
    run = json.loads(RUN.read_text())
    stats, rows = analyze(run)
    write_outputs(stats, rows)
    plot_summary(stats, rows)
    plot_concept_traces(run)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
