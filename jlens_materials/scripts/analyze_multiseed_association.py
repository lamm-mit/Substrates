#!/usr/bin/env python3
"""Integrated three-seed analysis of the frozen 50-item association suite.

The lens-fit seed is treated as a repeated measurement, not as an independent
prompt. Population uncertainty is estimated by resampling mechanism families
and then phrasings within families.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper_protocol as pp  # noqa: E402


RUN_PATHS = [
    ROOT / "runs" / f"gemma4-e4b-it-paper-seed{seed}.json"
    for seed in range(3)
]
EXP_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "figures" / "gemma4-paper-multiseed"
STATS_PATH = EXP_DIR / "gemma4-paper-multiseed-integrated_statistics.json"
CSV_PATH = EXP_DIR / "gemma4-paper-multiseed-integrated_prompt_results.csv"
RESULTS_PATH = EXP_DIR / "GEMMA4_PAPER_MULTI_SEED_INTEGRATED_ANALYSIS.md"
FIGURE_PATH = FIG_DIR / "integrated-association-analysis"
RNG_SEED = 20260714
N_BOOT = 20_000
N_CORR_BOOT = 5_000


def item_auc(ranks: list[int], ks: list[int]) -> float:
    pass_at_k = [float(np.mean([0 <= rank < k for rank in ranks])) for k in ks]
    return float(pp.log_k_auc(ks, pass_at_k))


def indexed_association_records(run: dict) -> dict[str, dict]:
    return {
        record["slug"]: record
        for record in run["prompts"]
        if record.get("shape") == "ASSOCIATION"
    }


def validate_runs(runs: list[dict]) -> list[str]:
    if len(runs) != 3:
        raise ValueError("This analysis requires exactly three lens-fit seeds")
    slugs = list(indexed_association_records(runs[0]))
    if len(slugs) != 50:
        raise ValueError(f"Expected 50 association prompts, found {len(slugs)}")
    for seed, run in enumerate(runs):
        if run.get("errors"):
            raise ValueError(f"Seed {seed} contains run errors")
        if not run.get("methodology", {}).get("paper_protocol_complete"):
            raise ValueError(f"Seed {seed} is not paper-protocol complete")
        records = indexed_association_records(run)
        if list(records) != slugs:
            raise ValueError(f"Seed {seed} prompt order/set differs")
        if not all(record.get("valid_for_metrics") for record in records.values()):
            raise ValueError(f"Seed {seed} contains excluded association prompts")
    return slugs


def percentile_summary(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "n_resamples": int(len(values)),
    }


def hierarchical_effect_bootstrap(
    prompt_rows: list[dict], *, seed: int = RNG_SEED
) -> dict:
    """Resample families, then phrasings, after averaging lens seeds per prompt."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in prompt_rows:
        grouped[row["family"]].append(float(row["delta_auc_mean_seed"]))
    families = sorted(grouped)
    arrays = [np.asarray(grouped[family], dtype=float) for family in families]
    rng = np.random.default_rng(seed)
    estimates = np.empty(N_BOOT, dtype=float)
    for index in range(N_BOOT):
        chosen = rng.integers(0, len(arrays), size=len(arrays))
        family_estimates = []
        for family_index in chosen:
            values = arrays[int(family_index)]
            within = rng.integers(0, len(values), size=len(values))
            family_estimates.append(float(np.mean(values[within])))
        estimates[index] = float(np.mean(family_estimates))
    result = percentile_summary(estimates)
    result["observed_mean"] = float(np.mean([
        row["delta_auc_mean_seed"] for row in prompt_rows
    ]))
    result["unit"] = "10 mechanism families; five phrasings resampled within family"
    return result


def exact_family_sign_flip(family_effects: dict[str, float]) -> dict:
    values = np.asarray([family_effects[name] for name in sorted(family_effects)])
    observed = float(np.mean(values))
    null = np.asarray([
        float(np.mean(values * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    return {
        "observed_mean": observed,
        "p_one_sided": float(np.mean(null >= observed - 1e-15)),
        "n_permutations": int(len(null)),
        "unit": "mechanism-family mean effect",
    }


def spearman_value(first: np.ndarray, second: np.ndarray) -> float:
    if len(np.unique(first)) < 2 or len(np.unique(second)) < 2:
        return float("nan")
    result = spearmanr(first, second)
    return float(result.statistic)


def family_clustered_spearman(
    rows: list[dict], first_key: str, second_key: str, *, seed: int
) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    families = sorted(grouped)
    observed = spearman_value(
        np.asarray([row[first_key] for row in rows], dtype=float),
        np.asarray([row[second_key] for row in rows], dtype=float),
    )
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(N_CORR_BOOT):
        chosen = rng.integers(0, len(families), size=len(families))
        sampled = [row for index in chosen for row in grouped[families[int(index)]]]
        estimate = spearman_value(
            np.asarray([row[first_key] for row in sampled], dtype=float),
            np.asarray([row[second_key] for row in sampled], dtype=float),
        )
        if math.isfinite(estimate):
            estimates.append(estimate)
    estimates_array = np.asarray(estimates, dtype=float)
    return {
        "rho": observed,
        "family_clustered_low": float(np.quantile(estimates_array, 0.025)),
        "family_clustered_high": float(np.quantile(estimates_array, 0.975)),
        "n_rows": len(rows),
        "n_resamples": len(estimates),
    }


def build_rows(runs: list[dict], slugs: list[str]) -> tuple[list[dict], list[dict], list[int]]:
    indexed = [indexed_association_records(run) for run in runs]
    ks = list(runs[0]["metrics"]["by_shape"]["ASSOCIATION"]["ks"])
    prompt_rows = []
    concept_rows = []
    for slug in slugs:
        records = [records_by_slug[slug] for records_by_slug in indexed]
        reference = records[0]
        labels = [[item["label"] for item in record["emergence"]] for record in records]
        if labels[1:] != labels[:1] * 2:
            raise ValueError(f"Tracked concepts differ across seeds for {slug}")
        j_aucs = [
            item_auc([int(item["best_rank"]) for item in record["emergence"]], ks)
            for record in records
        ]
        logit_auc = item_auc(
            [int(item["logit_lens_best_rank"]) for item in reference["emergence"]],
            ks,
        )
        prompt_rows.append({
            "slug": slug,
            "family": reference["category"],
            "phrasing_id": reference.get("phrasing_id"),
            "prompt": reference["prompt_text"],
            "n_concepts": len(reference["emergence"]),
            "j_auc_seed0": j_aucs[0],
            "j_auc_seed1": j_aucs[1],
            "j_auc_seed2": j_aucs[2],
            "j_auc_mean_seed": float(np.mean(j_aucs)),
            "logit_auc": logit_auc,
            "delta_auc_mean_seed": float(np.mean(j_aucs) - logit_auc),
        })
        for concept_index, label in enumerate(labels[0]):
            emergence = [record["emergence"][concept_index] for record in records]
            j_ranks = [int(item["best_rank"]) + 1 for item in emergence]
            logit_rank = int(emergence[0]["logit_lens_best_rank"]) + 1
            j_log_ranks = [math.log10(rank) for rank in j_ranks]
            concept_rows.append({
                "slug": slug,
                "family": reference["category"],
                "concept": label,
                "j_rank_seed0": j_ranks[0],
                "j_rank_seed1": j_ranks[1],
                "j_rank_seed2": j_ranks[2],
                "j_log_rank_seed0": j_log_ranks[0],
                "j_log_rank_seed1": j_log_ranks[1],
                "j_log_rank_seed2": j_log_ranks[2],
                "j_log_rank_mean_seed": float(np.mean(j_log_ranks)),
                "logit_rank": logit_rank,
                "logit_log_rank": math.log10(logit_rank),
                "log10_rank_advantage": float(math.log10(logit_rank) - np.mean(j_log_ranks)),
            })
    return prompt_rows, concept_rows, ks


def build_statistics(runs: list[dict], prompt_rows: list[dict], concept_rows: list[dict], ks: list[int]) -> dict:
    per_seed = []
    for seed in range(3):
        j = np.asarray([row[f"j_auc_seed{seed}"] for row in prompt_rows])
        logit = np.asarray([row["logit_auc"] for row in prompt_rows])
        per_seed.append({
            "seed": seed,
            "j_auc_mean": float(np.mean(j)),
            "logit_auc_mean": float(np.mean(logit)),
            "delta_auc": float(np.mean(j - logit)),
            "j_wins": int(np.sum(j > logit + 1e-12)),
            "ties": int(np.sum(np.abs(j - logit) <= 1e-12)),
            "j_losses": int(np.sum(j < logit - 1e-12)),
        })

    family_groups: dict[str, list[dict]] = defaultdict(list)
    for row in prompt_rows:
        family_groups[row["family"]].append(row)
    families = {}
    for family, rows in sorted(family_groups.items()):
        deltas = np.asarray([row["delta_auc_mean_seed"] for row in rows])
        families[family] = {
            "n_phrasings": len(rows),
            "j_auc_mean_seed_and_prompt": float(np.mean([
                row["j_auc_mean_seed"] for row in rows
            ])),
            "logit_auc_mean_prompt": float(np.mean([row["logit_auc"] for row in rows])),
            "delta_auc_mean": float(np.mean(deltas)),
            "delta_auc_min": float(np.min(deltas)),
            "delta_auc_max": float(np.max(deltas)),
        }

    mean_j = np.asarray([row["j_auc_mean_seed"] for row in prompt_rows])
    logit = np.asarray([row["logit_auc"] for row in prompt_rows])
    overall = {
        "n_prompts": len(prompt_rows),
        "n_families": len(families),
        "n_declared_concepts": len(concept_rows),
        "j_auc_mean_across_seeds_and_prompts": float(np.mean(mean_j)),
        "logit_auc_mean_across_prompts": float(np.mean(logit)),
        "delta_auc": float(np.mean(mean_j - logit)),
        "relative_auc_gain": float(np.mean(mean_j) / np.mean(logit) - 1),
        "j_wins": int(np.sum(mean_j > logit + 1e-12)),
        "ties": int(np.sum(np.abs(mean_j - logit) <= 1e-12)),
        "j_losses": int(np.sum(mean_j < logit - 1e-12)),
        "hierarchical_family_bootstrap": hierarchical_effect_bootstrap(prompt_rows),
        "exact_family_sign_flip": exact_family_sign_flip({
            family: result["delta_auc_mean"] for family, result in families.items()
        }),
    }

    prompt_correlations = {}
    concept_correlations = {}
    for first in range(3):
        for second in range(first + 1, 3):
            label = f"seed{first}_vs_seed{second}"
            prompt_correlations[label] = family_clustered_spearman(
                prompt_rows,
                f"j_auc_seed{first}",
                f"j_auc_seed{second}",
                seed=RNG_SEED + 10 * first + second,
            )
            concept_correlations[label] = family_clustered_spearman(
                concept_rows,
                f"j_log_rank_seed{first}",
                f"j_log_rank_seed{second}",
                seed=RNG_SEED + 100 + 10 * first + second,
            )
    prompt_correlations["mean_j_vs_logit"] = family_clustered_spearman(
        prompt_rows,
        "j_auc_mean_seed",
        "logit_auc",
        seed=RNG_SEED + 200,
    )
    concept_correlations["mean_j_vs_logit"] = family_clustered_spearman(
        concept_rows,
        "j_log_rank_mean_seed",
        "logit_log_rank",
        seed=RNG_SEED + 201,
    )

    pass_at_k = {
        "ks": ks,
        "j_by_seed": [
            runs[seed]["metrics"]["by_shape"]["ASSOCIATION"]["jacobian_lens"]["pass_at_k"]
            for seed in range(3)
        ],
        "j_mean": np.mean(np.asarray([
            runs[seed]["metrics"]["by_shape"]["ASSOCIATION"]["jacobian_lens"]["pass_at_k"]
            for seed in range(3)
        ]), axis=0).tolist(),
        "logit": runs[0]["metrics"]["by_shape"]["ASSOCIATION"]["logit_lens"]["pass_at_k"],
    }
    return {
        "analysis_status": "retrospective integrated robustness analysis",
        "analysis_seed": RNG_SEED,
        "bootstrap_resamples": N_BOOT,
        "correlation_bootstrap_resamples": N_CORR_BOOT,
        "runs": [str(path.relative_to(ROOT)) for path in RUN_PATHS],
        "overall": overall,
        "per_seed": per_seed,
        "families": families,
        "correlations": {
            "prompt_auc": prompt_correlations,
            "concept_log10_rank": concept_correlations,
        },
        "pass_at_k": pass_at_k,
    }


def write_prompt_csv(rows: list[dict]) -> None:
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(stats: dict, prompt_rows: list[dict], concept_rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    j_color = "#C33C54"
    logit_color = "#287C8E"
    neutral = "#6E7781"
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.2), constrained_layout=True)

    pass_data = stats["pass_at_k"]
    ks = pass_data["ks"]
    ax = axes[0, 0]
    for seed, values in enumerate(pass_data["j_by_seed"]):
        ax.plot(ks, values, color=j_color, alpha=0.25, linewidth=1.2)
    ax.plot(ks, pass_data["j_mean"], "o-", color=j_color, linewidth=2.4,
            label="J-lens (mean of 3 fits)")
    ax.plot(ks, pass_data["logit"], "s-", color=logit_color, linewidth=2.0,
            label="logit lens")
    ax.set_xscale("log")
    ax.set_xlabel("Allowed vocabulary rank, k")
    ax.set_ylabel("Fraction of declared concepts recovered")
    ax.set_title("A  Recovery of predeclared concepts", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    x = np.asarray([row["logit_auc"] for row in prompt_rows])
    y = np.asarray([row["j_auc_mean_seed"] for row in prompt_rows])
    limit = max(float(np.max(x)), float(np.max(y))) * 1.08
    ax.scatter(x, y, s=42, color=neutral, alpha=0.75, edgecolor="none")
    ax.plot([0, limit], [0, limit], linestyle="--", color=neutral, linewidth=1)
    ax.set_xlim(-0.006, limit)
    ax.set_ylim(-0.006, limit)
    ax.set_xlabel("Logit-lens prompt AUC")
    ax.set_ylabel("Mean three-seed J-lens prompt AUC")
    ax.set_title("B  Prompt-level comparison", loc="left", fontweight="bold")
    overall = stats["overall"]
    ci = overall["hierarchical_family_bootstrap"]
    ax.text(
        0.04, 0.95,
        f"mean difference = {overall['delta_auc']:+.4f}\n"
        f"family-hierarchical 95% CI\n{ci['low']:+.4f} to {ci['high']:+.4f}",
        transform=ax.transAxes, va="top",
    )
    ax.grid(True, alpha=0.18)

    ax = axes[1, 0]
    families = sorted(stats["families"], key=lambda name: stats["families"][name]["delta_auc_mean"])
    y_positions = np.arange(len(families))
    for y_position, family in zip(y_positions, families):
        raw = [row["delta_auc_mean_seed"] for row in prompt_rows if row["family"] == family]
        ax.scatter(raw, np.full(len(raw), y_position), s=24, color=neutral, alpha=0.35)
        ax.scatter(stats["families"][family]["delta_auc_mean"], y_position,
                   marker="D", s=48, color=j_color, zorder=3)
    ax.axvline(0, color=neutral, linewidth=1)
    ax.set_yticks(y_positions, [name.replace("-", " ") for name in families])
    ax.set_xlabel("J-lens minus logit-lens AUC (right favors J-lens)")
    ax.set_title("C  Heterogeneity across mechanism families", loc="left", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.18)

    ax = axes[1, 1]
    variables = ["J seed 0", "J seed 1", "J seed 2", "logit"]
    arrays = [
        np.asarray([row[f"j_log_rank_seed{seed}"] for row in concept_rows])
        for seed in range(3)
    ] + [np.asarray([row["logit_log_rank"] for row in concept_rows])]
    matrix = np.asarray([
        [spearman_value(first, second) for second in arrays]
        for first in arrays
    ])
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(4), variables, rotation=25, ha="right")
    ax.set_yticks(range(4), variables)
    for row in range(4):
        for column in range(4):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center",
                    color="white" if value < 0.72 else "black")
    ax.set_title("D  Rank reproducibility across 150 concepts", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman rank correlation")

    fig.suptitle(
        "Three lens fits agree closely; the average J-lens advantage is small and family-dependent",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5, -0.01,
        "Lens-fit seeds are repeated measurements. Population uncertainty resamples 10 mechanism families and five phrasings within each family.",
        ha="center", fontsize=9, color=neutral,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURE_PATH.with_suffix(f".{suffix}"), dpi=240,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fmt_ci(result: dict) -> str:
    return f"{result['rho']:.3f} ({result['family_clustered_low']:.3f} to {result['family_clustered_high']:.3f})"


def write_results(stats: dict) -> None:
    overall = stats["overall"]
    bootstrap = overall["hierarchical_family_bootstrap"]
    sign_flip = overall["exact_family_sign_flip"]
    concept_corr = stats["correlations"]["concept_log10_rank"]
    prompt_corr = stats["correlations"]["prompt_auc"]
    lines = [
        "# Integrated three-seed analysis of the 50 materials association prompts",
        "",
        "## Bottom line",
        "",
        f"Across the 50 prompts, mean J-lens recovery AUC was {overall['j_auc_mean_across_seeds_and_prompts']:.4f}; "
        f"logit-lens AUC was {overall['logit_auc_mean_across_prompts']:.4f}. The absolute difference was "
        f"{overall['delta_auc']:+.4f} ({100 * overall['relative_auc_gain']:.1f}% relative), with a "
        f"family-hierarchical 95% interval of {bootstrap['low']:+.4f} to {bootstrap['high']:+.4f}. "
        f"The exact one-sided family sign-flip p-value was {sign_flip['p_one_sided']:.3f}. The interval crosses zero: "
        "these paper-protocol lenses do **not** support a universal aggregate J-lens advantage on this suite.",
        "",
        f"The scientifically stronger positive result is reproducibility. Across all {overall['n_declared_concepts']} declared "
        "prompt-concept pairs, independently fitted J-lenses produced almost identical rank orderings:",
        "",
        f"- seed 0 versus 1: rho = {fmt_ci(concept_corr['seed0_vs_seed1'])};",
        f"- seed 0 versus 2: rho = {fmt_ci(concept_corr['seed0_vs_seed2'])};",
        f"- seed 1 versus 2: rho = {fmt_ci(concept_corr['seed1_vs_seed2'])}.",
        "",
        "Intervals above are family-clustered bootstrap intervals. They describe reproducibility, not evidence that a decoded word is causal or that the model understands the mechanism.",
        "",
        "![Integrated three-seed analysis](../figures/gemma4-paper-multiseed/integrated-association-analysis.png)",
        "",
        "## What is correlated?",
        "",
        "The seed-to-seed correlations compare the full-vocabulary ranks assigned to the same declared scientific concept in the same prompt. They answer: *if the lens is refitted on a different WikiText sample, do the same prompt-concept pairs remain easy or hard to read?* The answer is yes.",
        "",
        f"The mean J-lens and logit-lens concept ranks were only moderately associated: rho = {fmt_ci(concept_corr['mean_j_vs_logit'])}. "
        "Thus the methods share some notion of which concepts are intrinsically easy or hard, but the J-lens is not merely a noisy copy of direct unembedding. This correlation does not itself show that either lens is correct.",
        "",
        "At the coarser prompt-AUC level, correlations are inflated by many prompts with zero recovery for both methods and should be treated as descriptive:",
        "",
        f"- seed 0 versus 1: rho = {fmt_ci(prompt_corr['seed0_vs_seed1'])};",
        f"- seed 0 versus 2: rho = {fmt_ci(prompt_corr['seed0_vs_seed2'])};",
        f"- seed 1 versus 2: rho = {fmt_ci(prompt_corr['seed1_vs_seed2'])};",
        f"- mean J-lens versus logit lens: rho = {fmt_ci(prompt_corr['mean_j_vs_logit'])}.",
        "",
        "## Per-seed aggregate results",
        "",
        "| lens-fit seed | J AUC | logit AUC | difference | J wins | ties | J losses |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['j_auc_mean']:.4f} | {row['logit_auc_mean']:.4f} | "
            f"{row['delta_auc']:+.4f} | {row['j_wins']} | {row['ties']} | {row['j_losses']} |"
        )
    lines.extend([
        "",
        "The logit-lens value is identical across rows because no fitted lens is involved in direct unembedding.",
        "",
        "## Mechanism-family heterogeneity",
        "",
        "| family | mean J AUC | logit AUC | difference |",
        "|---|---:|---:|---:|",
    ])
    for family, row in sorted(stats["families"].items()):
        lines.append(
            f"| {family} | {row['j_auc_mean_seed_and_prompt']:.4f} | "
            f"{row['logit_auc_mean_prompt']:.4f} | {row['delta_auc_mean']:+.4f} |"
        )
    lines.extend([
        "",
        "These five-phrasing family estimates are descriptive. Notch resistance and ductile failure favor the J-lens, whereas cyclic loading and particle strengthening favor direct unembedding. This heterogeneity is why a single pooled correlation or p-value would tell an incomplete story.",
        "",
        "## Integrated analysis to add after the new experiments",
        "",
        "The final paper should connect three independently defined prompt-level scores on genuinely held-out prompts:",
        "",
        "1. **Open-vocabulary semantic agreement:** whether a frozen, unrestricted decoder and blinded ontology recover the correct mechanism family.",
        "2. **Counterfactual sensitivity:** whether the readout moves in the physically correct direction when one causal feature is changed.",
        "3. **Behavioral performance:** whether the model answers a separate scientific question correctly.",
        "",
        "We can then estimate family-blocked Spearman correlations among these three scores and the predeclared readout score. The most meaningful paper result would be a positive association between readable mechanism evidence and correct counterfactual/behavioral performance. A correlation only between two lenses is a reproducibility or shared-difficulty result, not evidence of understanding.",
        "",
        "## Statistical guardrails",
        "",
        "- Average the three lens-fit seeds within each prompt before population inference.",
        "- Resample or permute at the 10-family level; do not count 150 seed-prompt rows as independent.",
        "- Treat the three seed values as robustness measurements, not as n=3 biological-style replicates.",
        "- Label this analysis retrospective; reserve confirmatory claims for the new held-out prompt suite.",
        "- Report effect sizes and clustered intervals even when showing a correlation coefficient.",
        "",
        "## Artifacts",
        "",
        "- Exact prompts and per-concept values: [`PAPER_ASSOCIATION_PROMPTS_MULTI_SEED_SI.md`](PAPER_ASSOCIATION_PROMPTS_MULTI_SEED_SI.md)",
        "- Prompt-level CSV: [`gemma4-paper-multiseed-integrated_prompt_results.csv`](gemma4-paper-multiseed-integrated_prompt_results.csv)",
        "- Machine-readable statistics: [`gemma4-paper-multiseed-integrated_statistics.json`](gemma4-paper-multiseed-integrated_statistics.json)",
        "- Figure (vector): [`../figures/gemma4-paper-multiseed/integrated-association-analysis.pdf`](../figures/gemma4-paper-multiseed/integrated-association-analysis.pdf)",
        "- Reproduce: `python scripts/analyze_multiseed_association.py`",
        "",
    ])
    RESULTS_PATH.write_text("\n".join(lines))


def main() -> None:
    runs = [json.loads(path.read_text()) for path in RUN_PATHS]
    slugs = validate_runs(runs)
    prompt_rows, concept_rows, ks = build_rows(runs, slugs)
    stats = build_statistics(runs, prompt_rows, concept_rows, ks)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    write_prompt_csv(prompt_rows)
    plot_results(stats, prompt_rows, concept_rows)
    write_results(stats)
    print(f"wrote {STATS_PATH}")
    print(f"wrote {CSV_PATH}")
    print(f"wrote {RESULTS_PATH}")
    print(f"wrote {FIGURE_PATH.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
