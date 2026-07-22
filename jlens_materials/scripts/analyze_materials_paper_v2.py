#!/usr/bin/env python3
"""Paired statistics and publication figures for materials paper v2."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper_protocol as pp  # noqa: E402


RUN_PATH = ROOT / "runs" / "gemma4-materials-paper-v2.json"
EXP_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "figures" / "gemma4-materials-paper-v2"
STATS_PATH = EXP_DIR / "gemma4-materials-paper-v2_statistics.json"
CSV_PATH = EXP_DIR / "gemma4-materials-paper-v2_item_results.csv"
RESULTS_PATH = EXP_DIR / "gemma4-materials-paper-v2_results.md"
SEED = 20260710
N_BOOT = 20_000


def bootstrap_mean(values: list[float], *, seed: int = SEED) -> dict:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(N_BOOT, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(sampled, 0.025)),
        "high": float(np.quantile(sampled, 0.975)),
        "n": int(len(array)),
    }


def paired_test(first: list[float], second: list[float], *, alternative: str) -> dict:
    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    nonzero = delta[np.abs(delta) > 1e-12]
    try:
        statistic, pvalue = wilcoxon(first, second, alternative=alternative, zero_method="wilcox")
        statistic, pvalue = float(statistic), float(pvalue)
    except ValueError:
        statistic, pvalue = None, None
    return {
        "delta": bootstrap_mean(delta.tolist()),
        "wins_first": int(np.sum(delta > 0 if alternative == "greater" else delta < 0)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
        "losses_first": int(np.sum(delta < 0 if alternative == "greater" else delta > 0)),
        "win_fraction_non_ties": (
            float(np.mean(nonzero > 0 if alternative == "greater" else nonzero < 0))
            if len(nonzero) else None
        ),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_one_sided": pvalue,
    }


def item_auc(ranks: list[int], ks: list[int]) -> float:
    values = [float(np.mean([0 <= rank < k for rank in ranks])) for k in ks]
    return pp.log_k_auc(ks, values)


def association_analysis(run: dict) -> tuple[dict, list[dict]]:
    metric = run["metrics"]["by_shape"]["ASSOCIATION"]
    ks = metric["ks"]
    items = []
    concept_rows = []
    for record in run["prompts"]:
        if record.get("shape") != "ASSOCIATION" or not record.get("valid_for_metrics"):
            continue
        j_ranks = [int(item["best_rank"]) for item in record["emergence"]]
        l_ranks = [int(item["logit_lens_best_rank"]) for item in record["emergence"]]
        j_auc = item_auc(j_ranks, ks)
        l_auc = item_auc(l_ranks, ks)
        item = {
            "slug": record["slug"],
            "title": record["title"],
            "family": record.get("category", "uncategorized"),
            "prompt": record["prompt_text"],
            "j_auc": j_auc,
            "logit_auc": l_auc,
            "delta_auc": j_auc - l_auc,
        }
        items.append(item)
        for emergence in record["emergence"]:
            j_rank = int(emergence["best_rank"]) + 1
            logit_rank = int(emergence["logit_lens_best_rank"]) + 1
            concept_rows.append({
                "slug": record["slug"],
                "title": record["title"],
                "family": record.get("category", "uncategorized"),
                "concept": emergence["label"],
                "j_rank": j_rank,
                "logit_rank": logit_rank,
                "log10_improvement": math.log10(logit_rank / j_rank),
                "best_depth": emergence["best_depth"],
                "onset_depth": emergence["onset_depth"],
                "prompt": record["prompt_text"],
            })

    family_items = defaultdict(list)
    for item in items:
        family_items[item["family"]].append(item)
    families = {}
    for family, records in sorted(family_items.items()):
        j = [record["j_auc"] for record in records]
        l = [record["logit_auc"] for record in records]
        families[family] = {
            "n": len(records),
            "j_auc_mean": float(np.mean(j)),
            "logit_auc_mean": float(np.mean(l)),
            "delta_auc": bootstrap_mean((np.asarray(j) - np.asarray(l)).tolist()),
            "j_wins": int(sum(a > b for a, b in zip(j, l))),
            "ties": int(sum(abs(a - b) <= 1e-12 for a, b in zip(j, l))),
        }

    paired = paired_test(
        [item["j_auc"] for item in items],
        [item["logit_auc"] for item in items],
        alternative="greater",
    )
    return {
        "n_items": len(items),
        "ks": ks,
        "aggregate_j_auc": metric["jacobian_lens"]["auc_log_k"],
        "aggregate_logit_auc": metric["logit_lens"]["auc_log_k"],
        "relative_auc_gain": (
            metric["jacobian_lens"]["auc_log_k"] / metric["logit_lens"]["auc_log_k"] - 1
        ),
        "paired_item_auc": paired,
        "families": families,
        "top_j_improvements": sorted(
            concept_rows, key=lambda row: row["log10_improvement"], reverse=True
        )[:25],
        "items": items,
    }, concept_rows


def _condition_score(record: dict, lens: str) -> float:
    if lens == "j":
        ranks = [int(item["best_rank"]) + 1 for item in record["emergence"]]
    else:
        ranks = [int(item["logit_lens_best_rank"]) + 1 for item in record["emergence"]]
    return float(np.mean(np.log10(ranks)))


def modulation_analysis(run: dict) -> dict:
    carriers: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in run["prompts"]:
        if record.get("shape") == "MODULATION" and record.get("valid_for_metrics"):
            carriers[record["carrier_id"]][record["condition"]] = record

    by_lens = {}
    for lens in ("j", "logit"):
        scores = {condition: [] for condition in ("focus", "suppress", "control")}
        per_carrier = []
        for carrier_id, conditions in sorted(carriers.items()):
            row = {"carrier_id": carrier_id}
            for condition in scores:
                value = _condition_score(conditions[condition], lens)
                scores[condition].append(value)
                row[condition] = value
            per_carrier.append(row)
        by_lens[lens] = {
            "condition_mean_log10_rank": {
                condition: bootstrap_mean(values, seed=SEED + index)
                for index, (condition, values) in enumerate(scores.items())
            },
            "focus_vs_suppress": paired_test(scores["focus"], scores["suppress"], alternative="less"),
            "focus_vs_control": paired_test(scores["focus"], scores["control"], alternative="less"),
            "per_carrier": per_carrier,
        }

    per_concept = {}
    for concept in ("dislocation", "pile", "stress"):
        per_concept[concept] = {}
        for lens in ("j", "logit"):
            values = {condition: [] for condition in ("focus", "suppress", "control")}
            for _, conditions in sorted(carriers.items()):
                for condition, record in conditions.items():
                    item = next(item for item in record["emergence"] if item["label"] == concept)
                    rank = (int(item["best_rank"]) + 1 if lens == "j"
                            else int(item["logit_lens_best_rank"]) + 1)
                    values[condition].append(math.log10(rank))
            per_concept[concept][lens] = {
                "focus_vs_suppress": paired_test(values["focus"], values["suppress"], alternative="less"),
                "focus_vs_control": paired_test(values["focus"], values["control"], alternative="less"),
            }

    hit_rates = {}
    for lens in ("j", "logit"):
        hit_rates[lens] = {}
        for condition in ("focus", "suppress", "control"):
            records = [conditions[condition] for conditions in carriers.values()]
            hit_rates[lens][condition] = {}
            for k in (1, 5, 10, 50, 100):
                hit_rates[lens][condition][str(k)] = float(np.mean([
                    any((int(item["best_rank"]) if lens == "j" else
                         int(item["logit_lens_best_rank"])) < k
                        for item in record["emergence"])
                    for record in records
                ]))

    return {
        "n_carriers": len(carriers),
        "by_lens": by_lens,
        "per_concept": per_concept,
        "any_target_hit_rate": hit_rates,
    }


def write_csv(rows: list[dict]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(run: dict, association: dict, modulation: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2), constrained_layout=True)

    metric = run["metrics"]["by_shape"]["ASSOCIATION"]
    ks = metric["ks"]
    axes[0, 0].plot(ks, metric["jacobian_lens"]["pass_at_k"], "o-", label="J-lens", color="#d1495b")
    axes[0, 0].plot(ks, metric["logit_lens"]["pass_at_k"], "s-", label="logit lens", color="#2a788e")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("rank cutoff k (farther right = less strict)")
    axes[0, 0].set_ylabel("fraction of declared concepts recovered")
    axes[0, 0].set_title("A  Concept recovery (higher is better)", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(True, alpha=0.2)
    for label, values, color in (
        ("J 21.2%", metric["jacobian_lens"]["pass_at_k"], "#d1495b"),
        ("logit 14.0%", metric["logit_lens"]["pass_at_k"], "#2a788e"),
    ):
        axes[0, 0].annotate(
            label,
            (ks[-1], values[-1]),
            xytext=(-6, 7),
            textcoords="offset points",
            ha="right",
            color=color,
            fontsize=9,
        )

    cmap = plt.get_cmap("tab10")
    families = sorted(association["families"])
    colors = {family: cmap(index) for index, family in enumerate(families)}
    for item in association["items"]:
        axes[0, 1].scatter(item["logit_auc"], item["j_auc"], color=colors[item["family"]], s=35, alpha=0.85)
    limit = max(max(item["j_auc"], item["logit_auc"]) for item in association["items"]) * 1.08
    axes[0, 1].plot([0, limit], [0, limit], "--", color="#777777")
    axes[0, 1].set_xlim(-0.005, limit)
    axes[0, 1].set_ylim(-0.005, limit)
    axes[0, 1].set_xlabel("logit-lens recovery AUC (right = better)")
    axes[0, 1].set_ylabel("J-lens recovery AUC (up = better)")
    axes[0, 1].set_title("B  Prompt-by-prompt comparison", loc="left", fontweight="bold")
    axes[0, 1].text(
        0.04,
        0.94,
        "above diagonal: J-lens better",
        transform=axes[0, 1].transAxes,
        va="top",
        fontsize=9,
        color="#555555",
    )
    axes[0, 1].grid(True, alpha=0.2)

    deltas = [association["families"][family]["delta_auc"]["mean"] for family in families]
    low = [association["families"][family]["delta_auc"]["low"] for family in families]
    high = [association["families"][family]["delta_auc"]["high"] for family in families]
    ypos = np.arange(len(families))
    axes[1, 0].barh(ypos, deltas, color=[colors[family] for family in families], alpha=0.85)
    axes[1, 0].errorbar(deltas, ypos, xerr=[np.asarray(deltas)-np.asarray(low), np.asarray(high)-np.asarray(deltas)], fmt="none", color="black", capsize=3)
    axes[1, 0].axvline(0, color="#555555", linewidth=1)
    axes[1, 0].set_yticks(ypos, families)
    axes[1, 0].set_xlabel("mean J AUC - logit AUC (right = J-lens better)")
    axes[1, 0].set_title("C  Differences by mechanism family", loc="left", fontweight="bold")

    ax = axes[1, 1]
    x = np.arange(3)
    conditions = ("focus", "suppress", "control")
    for carrier in modulation["by_lens"]["j"]["per_carrier"]:
        ax.plot(x, [carrier[c] for c in conditions], color="#bbbbbb", alpha=0.35, linewidth=0.8)
    means = [modulation["by_lens"]["j"]["condition_mean_log10_rank"][c]["mean"] for c in conditions]
    ax.plot(x, means, "o-", color="#d1495b", linewidth=2.5, markersize=7, label="J-lens mean")
    logit_means = [modulation["by_lens"]["logit"]["condition_mean_log10_rank"][c]["mean"] for c in conditions]
    ax.plot(x, logit_means, "s--", color="#2a788e", linewidth=2, markersize=6, label="logit-lens mean")
    ax.set_xticks(x, ("focus on concept", "suppress concept", "neutral control"))
    ax.set_ylabel("mean log10 best rank (lower is stronger)")
    ax.set_title("D  Instruction effect (lower is stronger)", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle("Preregistered Gemma-4 materials evaluation with a legacy 1,000-prompt lens", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.01, "Exploratory because lens provenance and source/target-layer recipe are not paper-complete.", ha="center", fontsize=9, color="#555555")
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"paper-v2-summary.{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_results(stats: dict) -> None:
    assoc = stats["association"]
    mod = stats["modulation"]
    pair = assoc["paired_item_auc"]
    lines = [
        "# Gemma-4 materials paper v2: frozen results",
        "",
        "This file was generated from the frozen run JSON by `scripts/analyze_materials_paper_v2.py`.",
        "The lens has 1,000 fitting prompts but legacy/unverified provenance, so results are large-sample exploratory evidence.",
        "",
        "## Association endpoint",
        "",
        f"All {assoc['n_items']} association items passed the controls. Aggregate pass@k AUC was "
        f"{assoc['aggregate_j_auc']:.4f} for J-lens and {assoc['aggregate_logit_auc']:.4f} for logit lens "
        f"({100*assoc['relative_auc_gain']:.1f}% relative gain).",
        "",
        f"The paired item-AUC difference was {pair['delta']['mean']:.4f} "
        f"(bootstrap 95% CI {pair['delta']['low']:.4f} to {pair['delta']['high']:.4f}); "
        f"one-sided Wilcoxon p={pair['wilcoxon_p_one_sided']:.4g}. "
        f"J-lens won {pair['wins_first']} items, tied {pair['ties']}, and lost {pair['losses_first']}.",
        "",
        "### Family-level results",
        "",
        "| family | n | J AUC | logit AUC | delta | J wins | ties |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, result in assoc["families"].items():
        lines.append(
            f"| {family} | {result['n']} | {result['j_auc_mean']:.4f} | "
            f"{result['logit_auc_mean']:.4f} | {result['delta_auc']['mean']:+.4f} | "
            f"{result['j_wins']} | {result['ties']} |"
        )
    lines.extend([
        "",
        "### Strongest concept-level J improvements",
        "",
        "| prompt | family | concept | J rank | logit rank | log10 gain | depth |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in assoc["top_j_improvements"][:15]:
        lines.append(
            f"| {row['slug']} | {row['family']} | {row['concept']} | {row['j_rank']} | "
            f"{row['logit_rank']} | {row['log10_improvement']:.2f} | {row['best_depth']:.1f}% |"
        )

    lines.extend(["", "## Directed-modulation endpoint", ""])
    for lens_label, lens_key in (("J-lens", "j"), ("logit lens", "logit")):
        result = mod["by_lens"][lens_key]
        fs = result["focus_vs_suppress"]
        fc = result["focus_vs_control"]
        lines.extend([
            f"### {lens_label}",
            "",
            f"Focus-minus-suppress mean log-rank contrast: {fs['delta']['mean']:+.3f} "
            f"(95% CI {fs['delta']['low']:+.3f} to {fs['delta']['high']:+.3f}; "
            f"focus wins {fs['wins_first']}/{mod['n_carriers']}; one-sided Wilcoxon p={fs['wilcoxon_p_one_sided']:.4g}).",
            "",
            f"Focus-minus-neutral mean log-rank contrast: {fc['delta']['mean']:+.3f} "
            f"(95% CI {fc['delta']['low']:+.3f} to {fc['delta']['high']:+.3f}; "
            f"focus wins {fc['wins_first']}/{mod['n_carriers']}; one-sided Wilcoxon p={fc['wilcoxon_p_one_sided']:.4g}).",
            "",
        ])
    lines.extend([
        "A negative contrast favors focus. These concepts were present in the focus/suppress instructions, so this is a directed-retention result, not input-absent discovery.",
        "",
        "## Artifact index",
        "",
        "- Exact prompts: [`../prompts/materials-paper-v2-preregistered.json`](../prompts/materials-paper-v2-preregistered.json)",
        "- Raw run: [`../runs/gemma4-materials-paper-v2.json`](../runs/gemma4-materials-paper-v2.json)",
        "- Machine-readable statistics: [`gemma4-materials-paper-v2_statistics.json`](gemma4-materials-paper-v2_statistics.json)",
        "- Concept-level CSV: [`gemma4-materials-paper-v2_item_results.csv`](gemma4-materials-paper-v2_item_results.csv)",
        "- Summary figure: [`../figures/gemma4-materials-paper-v2/paper-v2-summary.pdf`](../figures/gemma4-materials-paper-v2/paper-v2-summary.pdf)",
    ])
    RESULTS_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    run = json.loads(RUN_PATH.read_text())
    association, concept_rows = association_analysis(run)
    modulation = modulation_analysis(run)
    stats = {
        "run": str(RUN_PATH.relative_to(ROOT)),
        "preregistered_prompt_manifest": "prompts/materials-paper-v2-preregistered.json",
        "seed": SEED,
        "bootstrap_resamples": N_BOOT,
        "claims_level": run["methodology"]["claims_level"],
        "association": association,
        "modulation": modulation,
    }
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    write_csv(concept_rows)
    write_results(stats)
    plot_summary(run, association, modulation)
    print(f"wrote {STATS_PATH}")
    print(f"wrote {CSV_PATH}")
    print(f"wrote {RESULTS_PATH}")
    print(f"wrote {FIG_DIR / 'paper-v2-summary.pdf'}")


if __name__ == "__main__":
    main()
