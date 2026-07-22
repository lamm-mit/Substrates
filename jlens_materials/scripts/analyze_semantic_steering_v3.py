#!/usr/bin/env python3
"""Analyze the frozen v3 semantic-answer steering confirmation.

The physical condition is always the independent statistical unit.  Lens
seeds, random seeds, and answer-word presentation orders are technical
replicates that are averaged before family-level inference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments/semantic-steering-v3_raw.json"
MANIFEST = ROOT / "experiments/semantic-steering-v3-preregistration.json"
OUT_MD = ROOT / "experiments/semantic-steering-v3_results.md"
OUT_INVENTORY = ROOT / "experiments/semantic-steering-v3_prompt_inventory.md"
OUT_PROMPT = ROOT / "experiments/semantic-steering-v3_prompt_effects.csv"
OUT_CONDITION = ROOT / "experiments/semantic-steering-v3_condition_effects.csv"
OUT_CURVES = ROOT / "experiments/semantic-steering-v3_curves.csv"
OUT_SPECIFICITY = ROOT / "experiments/semantic-steering-v3_specificity_matrix.csv"
OUT_ALIGNED = ROOT / "experiments/semantic-steering-v3_expected_aligned_secondary.csv"
OUT_STATS = ROOT / "experiments/semantic-steering-v3_statistics.json"
FIG_DIR = ROOT / "figures/semantic-steering-v3"

FAMILY_LABELS = {
    "intergranular-corrosion": "Intergranular corrosion",
    "martensitic-transformation": "Martensitic transformation",
    "grain-size-strengthening": "Grain-size strengthening",
}
FAMILY_SHORT = {
    "intergranular-corrosion": "Corrosion",
    "martensitic-transformation": "Transformation",
    "grain-size-strengthening": "Grain size",
}
CONDITION_SHORT = {
    "long-sensitizing-service": "long sensitizing service",
    "stress-relief-carbide-network": "stress-relief carbides",
    "aged-high-carbon-sheet": "aged high-carbon sheet",
    "failed-stabilization-cycle": "failed stabilization",
    "proper-titanium-stabilization": "proper Ti stabilization",
    "ultralow-carbon-transient-cycle": "ultralow-C transient",
    "desensitized-by-remelting": "desensitized remelt",
    "carbon-scavenged-niobium": "Nb carbon scavenging",
    "brief-ambiguous-dwell": "brief ambiguous dwell",
    "unknown-carbon-short-stress-relief": "unknown-C stress relief",
    "laser-self-quenched-track": "laser self-quench",
    "induction-hardened-gear-tooth": "induction-hardened gear",
    "high-hardenability-gas-quench": "high-hardenability gas quench",
    "cryogenic-retained-austenite": "cryogenic conversion",
    "isothermal-pearlite-completion": "isothermal pearlite",
    "austempered-bainitic-state": "austempered bainite",
    "normalized-thick-plain-carbon": "normalized thick section",
    "spheroidized-carbide-aggregate": "spheroidized carbides",
    "polymer-quenched-shaft-center": "polymer-quenched center",
    "compressed-air-thin-plate": "compressed-air plate",
    "ecap-refined-aluminum": "ECAP-refined aluminum",
    "recrystallized-fine-brass": "fine recrystallized brass",
    "pinned-ferritic-steel-grains": "pinned ferritic grains",
    "fine-grained-nickel-foil": "fine nickel foil",
    "annealed-copper-grain-growth": "annealed copper growth",
    "inhibitor-loss-electrical-steel": "inhibitor-loss steel",
    "overannealed-alpha-brass": "overannealed brass",
    "coarsened-ferritic-plate": "coarsened ferritic plate",
    "nearly-equal-grain-sizes": "nearly equal sizes",
    "nanocrystalline-crossover": "nanocrystalline crossover",
}
CONTROL_LABELS = {
    "own": "Matched Jacobian",
    "wrong-mechanism": "Other mechanisms",
    "direct": "Direct decoder",
    "random": "Random directions",
}
COLORS = {
    "own": "#147A8A",
    "wrong-mechanism": "#7562A8",
    "direct": "#4F8A5B",
    "random": "#777777",
}
MARKERS = {"own": "o", "wrong-mechanism": "^", "direct": "s", "random": "D"}
PRESENTATION_EXAMPLES = {
    "intergranular-corrosion": "long-sensitizing-service",
    "martensitic-transformation": "laser-self-quenched-track",
    "grain-size-strengthening": "ecap-refined-aluminum",
}


def bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, n_boot: int = 30000
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), *np.quantile(samples, [0.025, 0.975]).tolist()


def fmt_ci(values: tuple[float, float, float]) -> str:
    return f"{values[0]:+.3f} [{values[1]:+.3f}, {values[2]:+.3f}]"


def safe_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.allclose(values, 0):
        return 1.0
    return float(stats.wilcoxon(values, alternative="two-sided").pvalue)


def endpoint_tables(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    rows["direction_id"] = np.select(
        [
            rows["method"].eq("semantic_jacobian"),
            rows["method"].eq("semantic_direct"),
            rows["method"].eq("random"),
        ],
        [
            "jac-" + rows["source_family_id"].fillna("")
            + "-lens-" + rows["lens_seed"].fillna(-1).astype(int).astype(str),
            "direct",
            "random-" + rows["random_seed"].fillna(-1).astype(int).astype(str),
        ],
        default="unknown",
    )
    clean = rows[rows["dose_percent"].eq(0)][
        ["prompt_id", "direction_id", "positive_log_odds"]
    ].rename(columns={"positive_log_odds": "direction_clean_log_odds"})
    rows = rows.merge(clean, on=["prompt_id", "direction_id"], how="left")
    rows["delta_log_odds"] = rows["positive_log_odds"] - rows["direction_clean_log_odds"]

    endpoint_records = []
    group_cols = [
        "target_family_id", "condition_id", "prompt_id", "presentation_order",
        "regime", "method", "control_type", "source_family_id", "direction_id",
    ]
    for keys, group in rows.groupby(group_cols, dropna=False):
        values = group.set_index("dose_percent")["positive_log_odds"]
        minus = group.loc[group["dose_percent"].eq(-4.0)].iloc[0]
        plus = group.loc[group["dose_percent"].eq(4.0)].iloc[0]
        endpoint_records.append({
            **dict(zip(group_cols, keys)),
            "endpoint_change_log_odds": float(values.loc[4.0] - values.loc[-4.0]),
            "linear_slope_per_percent": float(
                np.polyfit(group["dose_percent"], group["positive_log_odds"], 1)[0]
            ),
            "endpoint_choice_flip": bool(minus["choice"] != plus["choice"]),
            "minus_choice": minus["choice"],
            "plus_choice": plus["choice"],
        })
    endpoint_rep = pd.DataFrame(endpoint_records)

    prompt_effects = (
        endpoint_rep.groupby(
            [
                "target_family_id", "condition_id", "prompt_id",
                "presentation_order", "regime", "method", "control_type",
                "source_family_id",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            endpoint_change_log_odds=("endpoint_change_log_odds", "mean"),
            endpoint_replicate_sd=("endpoint_change_log_odds", "std"),
            linear_slope_per_percent=("linear_slope_per_percent", "mean"),
            replicate_flip_fraction=("endpoint_choice_flip", "mean"),
            n_replicates=("direction_id", "nunique"),
        )
    )
    condition_effects_long = (
        prompt_effects.groupby(
            [
                "target_family_id", "condition_id", "regime", "method",
                "control_type", "source_family_id",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            endpoint_change_log_odds=("endpoint_change_log_odds", "mean"),
            presentation_order_sd=("endpoint_change_log_odds", "std"),
            linear_slope_per_percent=("linear_slope_per_percent", "mean"),
            mean_replicate_flip_fraction=("replicate_flip_fraction", "mean"),
        )
    )
    return rows, endpoint_rep, prompt_effects, condition_effects_long


def control_condition_table(condition_long: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (family_id, condition_id, regime), group in condition_long.groupby(
        ["target_family_id", "condition_id", "regime"]
    ):
        own = group[group["control_type"].eq("own")]["endpoint_change_log_odds"]
        wrong = group[group["control_type"].eq("wrong-mechanism")][
            "endpoint_change_log_odds"
        ]
        direct = group[group["control_type"].eq("direct")]["endpoint_change_log_odds"]
        random = group[group["control_type"].eq("random")]["endpoint_change_log_odds"]
        records.append({
            "target_family_id": family_id,
            "condition_id": condition_id,
            "regime": regime,
            "own_jacobian": float(own.iloc[0]),
            "mean_wrong_mechanisms": float(wrong.mean()),
            "direct_decoder": float(direct.iloc[0]),
            "random_directions": float(random.iloc[0]),
        })
    return pd.DataFrame(records)


def analyze() -> None:
    raw = json.loads(RAW.read_text())
    manifest = json.loads(MANIFEST.read_text())
    rows_raw = pd.DataFrame(raw["intervention_rows"])
    preflight = pd.DataFrame(raw["preflight"])
    rows, endpoint_rep, prompt_effects, condition_long = endpoint_tables(rows_raw)
    prompt_effects.to_csv(OUT_PROMPT, index=False)
    condition = control_condition_table(condition_long)
    condition.to_csv(OUT_CONDITION, index=False)

    # Dose curves average only technical replicates, retaining prompt orders.
    curves = (
        rows.groupby(
            [
                "target_family_id", "condition_id", "prompt_id",
                "presentation_order", "regime", "control_type", "dose_percent",
            ],
            as_index=False,
        )
        .agg(
            delta_log_odds=("delta_log_odds", "mean"),
            technical_replicate_sd=("delta_log_odds", "std"),
            valid_choice_probability=("valid_choice_probability", "mean"),
            global_top_is_valid_choice=("global_top_is_valid_choice", "mean"),
            kl_clean_to_intervened=("kl_clean_to_intervened", "mean"),
        )
    )
    curves.to_csv(OUT_CURVES, index=False)

    rng = np.random.default_rng(20260716)
    criteria = manifest["design"]["family_success_criteria"]
    family_summaries = []
    for family_id in FAMILY_LABELS:
        subset = condition[condition["target_family_id"].eq(family_id)]
        own = subset["own_jacobian"].to_numpy()
        wrong = subset["mean_wrong_mechanisms"].to_numpy()
        direct = subset["direct_decoder"].to_numpy()
        random = subset["random_directions"].to_numpy()
        own_prompt = prompt_effects[
            prompt_effects["target_family_id"].eq(family_id)
            & prompt_effects["control_type"].eq("own")
        ]
        agreements = []
        for _, group in own_prompt.groupby("condition_id"):
            ordered = group.set_index("presentation_order")["endpoint_change_log_odds"]
            required = {"positive-first", "negative-first"}
            if required.issubset(ordered.index):
                a = float(ordered["positive-first"])
                b = float(ordered["negative-first"])
                agreements.append(a != 0 and b != 0 and np.sign(a) == np.sign(b))
        integrity_rows = rows[
            rows["target_family_id"].eq(family_id) & rows["control_type"].eq("own")
        ]
        own_minus_random = bootstrap_mean(own - random, rng)
        own_minus_wrong = bootstrap_mean(own - wrong, rng)
        positive_fraction = float(np.mean(own > 0))
        order_fraction = float(np.mean(agreements))
        top_valid_fraction = float(integrity_rows["global_top_is_valid_choice"].mean())
        passed = bool(
            positive_fraction >= criteria["positive_condition_fraction_at_least"]
            and order_fraction
            >= criteria["presentation_orders_same_nonzero_sign_fraction_at_least"]
            and own_minus_random[1] > 0
            and own_minus_wrong[1] > 0
            and top_valid_fraction
            >= criteria[
                "registered_answer_first_token_is_global_top_fraction_at_least"
            ]
        )
        family_summaries.append({
            "family_id": family_id,
            "own_jacobian": bootstrap_mean(own, rng),
            "mean_wrong_mechanisms": bootstrap_mean(wrong, rng),
            "direct_decoder": bootstrap_mean(direct, rng),
            "random_directions": bootstrap_mean(random, rng),
            "own_minus_random": own_minus_random,
            "own_minus_mean_wrong": own_minus_wrong,
            "positive_conditions": int((own > 0).sum()),
            "n_conditions": int(len(own)),
            "positive_condition_fraction": positive_fraction,
            "presentation_order_agreement_count": int(sum(agreements)),
            "presentation_order_agreement_n": int(len(agreements)),
            "presentation_order_agreement_fraction": order_fraction,
            "global_top_valid_fraction_own_rows": top_valid_fraction,
            "minimum_pair_probability_own_rows": float(
                integrity_rows["valid_choice_probability"].min()
            ),
            "median_pair_probability_own_rows": float(
                integrity_rows["valid_choice_probability"].median()
            ),
            "maximum_kl_own_rows": float(integrity_rows["kl_clean_to_intervened"].max()),
            "wilcoxon_own_vs_random_p": safe_wilcoxon(own - random),
            "wilcoxon_own_vs_wrong_p": safe_wilcoxon(own - wrong),
            "sign_test_positive_p": float(
                stats.binomtest(int((own > 0).sum()), len(own), 0.5).pvalue
            ),
            "registered_pass": passed,
        })

    # Full layer-matched target-by-source specificity matrix.
    specificity_records = []
    semantic_conditions = condition_long[
        condition_long["method"].eq("semantic_jacobian")
    ]
    for target_id in FAMILY_LABELS:
        for source_id in FAMILY_LABELS:
            values = semantic_conditions[
                semantic_conditions["target_family_id"].eq(target_id)
                & semantic_conditions["source_family_id"].eq(source_id)
            ]["endpoint_change_log_odds"].to_numpy()
            estimate = bootstrap_mean(values, rng)
            specificity_records.append({
                "target_family_id": target_id,
                "source_family_id": source_id,
                "matched": target_id == source_id,
                "mean_endpoint_change_log_odds": estimate[0],
                "ci95_low": estimate[1],
                "ci95_high": estimate[2],
                "n_physical_conditions": len(values),
            })
    specificity = pd.DataFrame(specificity_records)
    specificity.to_csv(OUT_SPECIFICITY, index=False)

    # Aggregate inference weights each physical condition once.
    own_all = condition["own_jacobian"].to_numpy()
    wrong_all = condition["mean_wrong_mechanisms"].to_numpy()
    direct_all = condition["direct_decoder"].to_numpy()
    random_all = condition["random_directions"].to_numpy()
    aggregate = {
        "own_jacobian": bootstrap_mean(own_all, rng),
        "mean_wrong_mechanisms": bootstrap_mean(wrong_all, rng),
        "direct_decoder": bootstrap_mean(direct_all, rng),
        "random_directions": bootstrap_mean(random_all, rng),
        "own_minus_random": bootstrap_mean(own_all - random_all, rng),
        "own_minus_mean_wrong": bootstrap_mean(own_all - wrong_all, rng),
        "own_minus_direct": bootstrap_mean(own_all - direct_all, rng),
        "positive_conditions": int((own_all > 0).sum()),
        "n_conditions": int(len(own_all)),
        "sign_test_positive_p": float(
            stats.binomtest(int((own_all > 0).sum()), len(own_all), 0.5).pvalue
        ),
        "wilcoxon_own_vs_random_p": safe_wilcoxon(own_all - random_all),
        "wilcoxon_own_vs_wrong_p": safe_wilcoxon(own_all - wrong_all),
        "wilcoxon_own_vs_direct_p": safe_wilcoxon(own_all - direct_all),
    }

    # Secondary, explicitly non-preregistered analysis: does the same positive
    # mechanism direction improve the frozen expected answer in both positive
    # and negative physical contexts?  Threshold cases have no expected label.
    aligned = condition[~condition["regime"].eq("near-threshold")].copy()
    aligned["expected_sign"] = aligned["regime"].map(
        {"positive": 1.0, "negative": -1.0}
    )
    aligned_columns = {
        "own_jacobian": "own_expected_aligned",
        "mean_wrong_mechanisms": "wrong_expected_aligned",
        "direct_decoder": "direct_expected_aligned",
        "random_directions": "random_expected_aligned",
    }
    for source, destination in aligned_columns.items():
        aligned[destination] = aligned[source] * aligned["expected_sign"]
    aligned.to_csv(OUT_ALIGNED, index=False)
    secondary_families = []
    for family_id in FAMILY_LABELS:
        subset = aligned[aligned["target_family_id"].eq(family_id)]
        own = subset["own_expected_aligned"].to_numpy()
        wrong = subset["wrong_expected_aligned"].to_numpy()
        direct = subset["direct_expected_aligned"].to_numpy()
        random = subset["random_expected_aligned"].to_numpy()
        positive_context = subset[subset["regime"].eq("positive")]["own_jacobian"]
        negative_context = subset[subset["regime"].eq("negative")]["own_jacobian"]
        secondary_families.append({
            "family_id": family_id,
            "own_expected_aligned": bootstrap_mean(own, rng),
            "wrong_expected_aligned": bootstrap_mean(wrong, rng),
            "direct_expected_aligned": bootstrap_mean(direct, rng),
            "random_expected_aligned": bootstrap_mean(random, rng),
            "own_minus_random_expected_aligned": bootstrap_mean(own - random, rng),
            "own_minus_wrong_expected_aligned": bootstrap_mean(own - wrong, rng),
            "own_minus_direct_expected_aligned": bootstrap_mean(own - direct, rng),
            "aligned_positive_conditions": int((own > 0).sum()),
            "n_determinate_conditions": int(len(own)),
            "positive_context_raw_positive": int((positive_context > 0).sum()),
            "n_positive_contexts": int(len(positive_context)),
            "negative_context_raw_negative": int((negative_context < 0).sum()),
            "n_negative_contexts": int(len(negative_context)),
            "sign_test_aligned_p": float(
                stats.binomtest(int((own > 0).sum()), len(own), 0.5).pvalue
            ),
            "wilcoxon_own_vs_random_p": safe_wilcoxon(own - random),
            "wilcoxon_own_vs_wrong_p": safe_wilcoxon(own - wrong),
            "wilcoxon_own_vs_direct_p": safe_wilcoxon(own - direct),
        })
    aligned_own = aligned["own_expected_aligned"].to_numpy()
    aligned_wrong = aligned["wrong_expected_aligned"].to_numpy()
    aligned_direct = aligned["direct_expected_aligned"].to_numpy()
    aligned_random = aligned["random_expected_aligned"].to_numpy()
    secondary_aggregate = {
        "own_expected_aligned": bootstrap_mean(aligned_own, rng),
        "wrong_expected_aligned": bootstrap_mean(aligned_wrong, rng),
        "direct_expected_aligned": bootstrap_mean(aligned_direct, rng),
        "random_expected_aligned": bootstrap_mean(aligned_random, rng),
        "own_minus_random_expected_aligned": bootstrap_mean(
            aligned_own - aligned_random, rng
        ),
        "own_minus_wrong_expected_aligned": bootstrap_mean(
            aligned_own - aligned_wrong, rng
        ),
        "own_minus_direct_expected_aligned": bootstrap_mean(
            aligned_own - aligned_direct, rng
        ),
        "aligned_positive_conditions": int((aligned_own > 0).sum()),
        "n_determinate_conditions": int(len(aligned_own)),
        "sign_test_aligned_p": float(
            stats.binomtest(
                int((aligned_own > 0).sum()), len(aligned_own), 0.5
            ).pvalue
        ),
        "wilcoxon_own_vs_random_p": safe_wilcoxon(aligned_own - aligned_random),
        "wilcoxon_own_vs_wrong_p": safe_wilcoxon(aligned_own - aligned_wrong),
        "wilcoxon_own_vs_direct_p": safe_wilcoxon(aligned_own - aligned_direct),
    }

    determinate = preflight[preflight["expected_outcome"].notna()]
    clean_accuracy = float(determinate["clean_expected_correct"].mean())
    clean_top_valid = float(preflight["clean"].map(
        lambda value: value["global_top_is_valid_choice"]
    ).mean())
    clean_pair_mass = preflight["clean"].map(
        lambda value: value["valid_choice_probability"]
    )
    own_endpoint_rep = endpoint_rep[endpoint_rep["control_type"].eq("own")]
    seed_pivot = own_endpoint_rep.pivot(
        index="prompt_id", columns="direction_id", values="endpoint_change_log_odds"
    )
    correlation = seed_pivot.corr()
    correlation_values = correlation.to_numpy()
    seed_values = correlation_values[
        np.triu_indices_from(correlation_values, k=1)
    ]
    seed_values = seed_values[np.isfinite(seed_values)]
    median_seed_correlation = float(np.median(seed_values))
    own_flips = int(own_endpoint_rep["endpoint_choice_flip"].sum())
    own_endpoint_count = int(len(own_endpoint_rep))

    integrity = {
        "clean_expected_accuracy_determinate_prompt_orders": clean_accuracy,
        "clean_global_top_valid_fraction_all_prompt_orders": clean_top_valid,
        "clean_pair_probability_minimum": float(clean_pair_mass.min()),
        "clean_pair_probability_median": float(clean_pair_mass.median()),
        "own_jacobian_endpoint_choice_flips": own_flips,
        "own_jacobian_endpoint_replicates": own_endpoint_count,
        "median_lens_seed_endpoint_correlation": median_seed_correlation,
    }

    statistics = {
        "study_id": raw["study_id"],
        "design_accounting": {
            "physical_conditions": int(condition.shape[0]),
            "prompt_orders": int(preflight.shape[0]),
            "raw_intervention_rows": int(rows_raw.shape[0]),
            "lens_seeds": 3,
            "random_directions": 10,
            "doses": [-4, -2, 0, 2, 4],
        },
        "aggregate": aggregate,
        "families": family_summaries,
        "integrity": integrity,
        "secondary_expected_aligned": {
            "status": "post hoc secondary analysis; requires new-data confirmation",
            "aggregate": secondary_aggregate,
            "families": secondary_families,
        },
    }
    OUT_STATS.write_text(json.dumps(statistics, indent=2) + "\n")

    make_figures(curves, condition, specificity, aligned)
    write_inventory(preflight, manifest)
    write_report(
        raw, manifest, statistics, condition, specificity, prompt_effects,
        endpoint_rep,
    )
    print(OUT_MD.read_text())


def make_figures(
    curves: pd.DataFrame,
    condition: pd.DataFrame,
    specificity: pd.DataFrame,
    aligned: pd.DataFrame,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    doses = [-4, -2, 0, 2, 4]

    # Equal-weight family dose responses across physical conditions.
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), sharex=True)
    for index, family_id in enumerate(FAMILY_LABELS):
        ax = axes[index]
        subset = curves[curves["target_family_id"].eq(family_id)]
        for control_type in CONTROL_LABELS:
            control = subset[subset["control_type"].eq(control_type)]
            condition_curve = (
                control.groupby(["condition_id", "dose_percent"], as_index=False)
                .agg(delta_log_odds=("delta_log_odds", "mean"))
            )
            wide = condition_curve.pivot(
                index="condition_id", columns="dose_percent", values="delta_log_odds"
            ).reindex(columns=doses)
            mean = wide.mean(axis=0)
            sem = wide.sem(axis=0)
            ax.plot(
                doses, mean, marker=MARKERS[control_type], markersize=4.5,
                linewidth=2.0 if control_type == "own" else 1.45,
                color=COLORS[control_type], label=CONTROL_LABELS[control_type],
            )
            ax.fill_between(
                doses, mean - sem, mean + sem, color=COLORS[control_type],
                alpha=0.10, linewidth=0,
            )
        ax.axhline(0, color="#9A9A9A", linewidth=0.8)
        ax.axvline(0, color="#D5D5D5", linewidth=0.8)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.65)
        ax.set_title(FAMILY_LABELS[family_id], fontsize=10.5, pad=8)
        ax.set_xlabel("Intervention dose (% residual norm)")
        if index == 0:
            ax.set_ylabel("Change in positive-answer log odds")
        ax.text(
            -0.13, 1.04, chr(ord("A") + index), transform=ax.transAxes,
            fontweight="bold", va="bottom",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, ncol=4, loc="upper center",
        bbox_to_anchor=(0.5, 0.995), fontsize=8.2,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.80, wspace=0.27)
    fig.savefig(FIG_DIR / "family_dose_response.png", dpi=300)
    fig.savefig(FIG_DIR / "family_dose_response.pdf")
    plt.close(fig)

    # Target answer by source mechanism specificity matrix.
    family_ids = list(FAMILY_LABELS)
    matrix = specificity.pivot(
        index="target_family_id", columns="source_family_id",
        values="mean_endpoint_change_log_odds",
    ).reindex(index=family_ids, columns=family_ids)
    limit = max(0.25, float(np.abs(matrix.to_numpy()).max()))
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    image = ax.imshow(matrix.to_numpy(), cmap="PRGn", vmin=-limit, vmax=limit)
    ax.set_xticks(range(3), [FAMILY_SHORT[value] for value in family_ids], rotation=25,
                  ha="right")
    ax.set_yticks(range(3), [FAMILY_SHORT[value] for value in family_ids])
    ax.set_xlabel("Source steering direction")
    ax.set_ylabel("Target scientific answer")
    for row_index, target_id in enumerate(family_ids):
        for col_index, source_id in enumerate(family_ids):
            value = float(matrix.loc[target_id, source_id])
            text_color = "white" if abs(value) > 0.62 * limit else "black"
            ax.text(
                col_index, row_index, f"{value:+.2f}", ha="center", va="center",
                color=text_color, fontsize=10, fontweight="bold" if row_index == col_index else "normal",
            )
            if row_index == col_index:
                ax.add_patch(plt.Rectangle(
                    (col_index - 0.48, row_index - 0.48), 0.96, 0.96,
                    fill=False, edgecolor="black", linewidth=1.8,
                ))
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.05)
    colorbar.set_label("Endpoint change in target-answer log odds")
    ax.spines[:].set_visible(False)
    fig.subplots_adjust(left=0.21, right=0.88, bottom=0.22, top=0.97)
    fig.savefig(FIG_DIR / "cross_mechanism_specificity.png", dpi=300)
    fig.savefig(FIG_DIR / "cross_mechanism_specificity.pdf")
    plt.close(fig)

    # All physical conditions, exposing heterogeneity rather than only averages.
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 7.1), sharex=True)
    all_values = condition[
        ["own_jacobian", "mean_wrong_mechanisms", "direct_decoder", "random_directions"]
    ].to_numpy()
    xlimit = max(0.5, float(np.nanmax(np.abs(all_values))) * 1.18)
    offsets = {"own": -0.24, "wrong-mechanism": -0.08, "direct": 0.08, "random": 0.24}
    columns = {
        "own": "own_jacobian",
        "wrong-mechanism": "mean_wrong_mechanisms",
        "direct": "direct_decoder",
        "random": "random_directions",
    }
    for index, family_id in enumerate(FAMILY_LABELS):
        ax = axes[index]
        subset = condition[condition["target_family_id"].eq(family_id)].copy()
        subset["regime_order"] = subset["regime"].map(
            {"positive": 0, "negative": 1, "near-threshold": 2}
        )
        subset = subset.sort_values(["regime_order", "condition_id"])
        y = np.arange(len(subset))
        for control_type, column in columns.items():
            ax.scatter(
                subset[column], y + offsets[control_type],
                marker=MARKERS[control_type], s=34 if control_type == "own" else 25,
                color=COLORS[control_type], label=CONTROL_LABELS[control_type], zorder=3,
            )
        ax.axvline(0, color="#8E8E8E", linewidth=0.85)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.65)
        ax.set_xlim(-xlimit, xlimit)
        ax.set_yticks(y, [CONDITION_SHORT[value] for value in subset["condition_id"]],
                      fontsize=7.7)
        ax.invert_yaxis()
        ax.set_title(FAMILY_LABELS[family_id], fontsize=10.5, pad=8)
        ax.set_xlabel("Endpoint change in positive-answer log odds")
        ax.text(
            -0.15, 1.035, chr(ord("A") + index), transform=ax.transAxes,
            fontweight="bold", va="bottom",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, ncol=4, loc="upper center",
        bbox_to_anchor=(0.5, 0.995), fontsize=8.2,
    )
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.11, top=0.86, wspace=0.48)
    fig.savefig(FIG_DIR / "condition_endpoints.png", dpi=300)
    fig.savefig(FIG_DIR / "condition_endpoints.pdf")
    plt.close(fig)

    # Three predeclared, physically transparent examples.
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.35), sharex=True)
    for index, (family_id, condition_id) in enumerate(PRESENTATION_EXAMPLES.items()):
        ax = axes[index]
        subset = curves[
            curves["target_family_id"].eq(family_id)
            & curves["condition_id"].eq(condition_id)
        ]
        for control_type in CONTROL_LABELS:
            line = (
                subset[subset["control_type"].eq(control_type)]
                .groupby("dose_percent", as_index=False)
                .agg(delta_log_odds=("delta_log_odds", "mean"))
                .sort_values("dose_percent")
            )
            ax.plot(
                line["dose_percent"], line["delta_log_odds"],
                marker=MARKERS[control_type], markersize=4.5,
                linewidth=2.0 if control_type == "own" else 1.45,
                color=COLORS[control_type], label=CONTROL_LABELS[control_type],
            )
        ax.axhline(0, color="#9A9A9A", linewidth=0.8)
        ax.axvline(0, color="#D5D5D5", linewidth=0.8)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.65)
        ax.set_title(
            f"{FAMILY_LABELS[family_id]}\n{CONDITION_SHORT[condition_id]}",
            fontsize=10.2, pad=8,
        )
        ax.set_xlabel("Intervention dose (% residual norm)")
        if index == 0:
            ax.set_ylabel("Change in positive-answer log odds")
        ax.text(
            -0.13, 1.04, chr(ord("A") + index), transform=ax.transAxes,
            fontweight="bold", va="bottom",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, ncol=4, loc="upper center",
        bbox_to_anchor=(0.5, 0.995), fontsize=8.2,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.75, wspace=0.27)
    fig.savefig(FIG_DIR / "three_predeclared_examples.png", dpi=300)
    fig.savefig(FIG_DIR / "three_predeclared_examples.pdf")
    plt.close(fig)

    # Explicitly secondary: frozen expected-answer alignment in determinate cases.
    aligned_plot_columns = {
        "own": "own_expected_aligned",
        "wrong-mechanism": "wrong_expected_aligned",
        "direct": "direct_expected_aligned",
        "random": "random_expected_aligned",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharex=True, sharey=True)
    plot_rng = np.random.default_rng(20260716)
    for index, family_id in enumerate(FAMILY_LABELS):
        ax = axes[index]
        subset = aligned[aligned["target_family_id"].eq(family_id)]
        for y, (control_type, column) in enumerate(aligned_plot_columns.items()):
            values = subset[column].to_numpy()
            jitter = plot_rng.uniform(-0.09, 0.09, size=len(values))
            ax.scatter(
                values, y + jitter, s=20, marker=MARKERS[control_type],
                color=COLORS[control_type], alpha=0.48, zorder=2,
            )
            estimate = bootstrap_mean(values, plot_rng)
            ax.errorbar(
                estimate[0], y, xerr=[[estimate[0] - estimate[1]],
                                      [estimate[2] - estimate[0]]],
                marker=MARKERS[control_type], markersize=7,
                color=COLORS[control_type], linewidth=1.8, capsize=3,
                markeredgecolor="white", markeredgewidth=0.7, zorder=4,
            )
        ax.axvline(0, color="#888888", linewidth=0.85)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.65)
        ax.set_title(FAMILY_LABELS[family_id], fontsize=10.5, pad=8)
        ax.set_xlabel("Change toward frozen correct answer")
        ax.text(
            -0.13, 1.04, chr(ord("A") + index), transform=ax.transAxes,
            fontweight="bold", va="bottom",
        )
    axes[0].set_yticks(
        range(4), [CONTROL_LABELS[value] for value in aligned_plot_columns], fontsize=8.2
    )
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.15, top=0.87, wspace=0.18)
    fig.savefig(FIG_DIR / "expected_answer_alignment_secondary.png", dpi=300)
    fig.savefig(FIG_DIR / "expected_answer_alignment_secondary.pdf")
    plt.close(fig)


def write_inventory(preflight: pd.DataFrame, manifest: dict) -> None:
    lines = [
        "# Semantic-steering v3 exact prompt inventory",
        "",
        "This file records every exact model prompt and its clean semantic-token answer. "
        "Each physical condition was asked twice; only the order of the two answer words changed.",
        "",
    ]
    for family in manifest["families"]:
        family_id = family["family_id"]
        lines.extend([
            f"## {FAMILY_LABELS[family_id]}",
            "",
            f"Fixed layer: {family['fixed_layer']}. Positive answer: "
            f"`{family['outcome_positive']}`. Negative answer: "
            f"`{family['outcome_negative']}`.",
            "",
        ])
        subset = preflight[preflight["family_id"].eq(family_id)]
        for condition_id, group in subset.groupby("condition_id", sort=False):
            lines.extend([
                f"### {condition_id}",
                "",
                f"Regime: `{group.iloc[0]['regime']}`. Expected outcome: "
                f"`{group.iloc[0]['expected_outcome']}`.",
                "",
            ])
            for row in group.itertuples():
                clean = row.clean
                lines.extend([
                    f"- `{row.presentation_order}` — clean choice `{clean['choice']}`, "
                    f"positive log odds {clean['positive_log_odds']:+.3f}, "
                    f"pair probability {clean['valid_choice_probability']:.3%}.",
                    f"  - {row.user}",
                ])
            lines.append("")
    OUT_INVENTORY.write_text("\n".join(lines) + "\n")


def write_report(
    raw: dict,
    manifest: dict,
    statistics: dict,
    condition: pd.DataFrame,
    specificity: pd.DataFrame,
    prompt_effects: pd.DataFrame,
    endpoint_rep: pd.DataFrame,
) -> None:
    aggregate = statistics["aggregate"]
    integrity = statistics["integrity"]
    family_lines = [
        "| Materials mechanism | Matched Jacobian | Matched − random | Matched − other mechanisms | Positive cases | Order agreement | Valid top word | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in statistics["families"]:
        family_lines.append(
            f"| {FAMILY_LABELS[summary['family_id']]} | "
            f"{fmt_ci(summary['own_jacobian'])} | {fmt_ci(summary['own_minus_random'])} | "
            f"{fmt_ci(summary['own_minus_mean_wrong'])} | "
            f"{summary['positive_conditions']}/{summary['n_conditions']} | "
            f"{summary['presentation_order_agreement_count']}/"
            f"{summary['presentation_order_agreement_n']} | "
            f"{summary['global_top_valid_fraction_own_rows']:.1%} | "
            f"{'PASS' if summary['registered_pass'] else 'FAIL'} |"
        )

    secondary = statistics["secondary_expected_aligned"]
    aligned_lines = [
        "| Materials mechanism | Matched direction toward correct answer | Matched − random | Matched − other mechanisms | Correct-direction cases | Positive contexts ↑ | Negative contexts ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in secondary["families"]:
        aligned_lines.append(
            f"| {FAMILY_LABELS[summary['family_id']]} | "
            f"{fmt_ci(summary['own_expected_aligned'])} | "
            f"{fmt_ci(summary['own_minus_random_expected_aligned'])} | "
            f"{fmt_ci(summary['own_minus_wrong_expected_aligned'])} | "
            f"{summary['aligned_positive_conditions']}/"
            f"{summary['n_determinate_conditions']} | "
            f"{summary['positive_context_raw_positive']}/"
            f"{summary['n_positive_contexts']} | "
            f"{summary['negative_context_raw_negative']}/"
            f"{summary['n_negative_contexts']} |"
        )

    matrix = specificity.pivot(
        index="target_family_id", columns="source_family_id",
        values="mean_endpoint_change_log_odds",
    ).reindex(index=FAMILY_LABELS, columns=FAMILY_LABELS)
    matrix_lines = [
        "| Target answer \\ source direction | Corrosion | Transformation | Grain size |",
        "|---|---:|---:|---:|",
    ]
    for target_id in FAMILY_LABELS:
        matrix_lines.append(
            f"| {FAMILY_SHORT[target_id]} | "
            + " | ".join(f"{matrix.loc[target_id, source_id]:+.3f}" for source_id in FAMILY_LABELS)
            + " |"
        )

    condition_lines = [
        "| Family and new physical condition | Regime | Matched J | Other mechanisms | Direct | Random | Odds multiplier |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in condition.sort_values(["target_family_id", "regime", "condition_id"]).itertuples():
        condition_lines.append(
            f"| {FAMILY_SHORT[row.target_family_id]} — {row.condition_id} | {row.regime} | "
            f"{row.own_jacobian:+.3f} | {row.mean_wrong_mechanisms:+.3f} | "
            f"{row.direct_decoder:+.3f} | {row.random_directions:+.3f} | "
            f"{math.exp(row.own_jacobian):.2f}× |"
        )

    # Post-hoc examples are explicitly labeled and do not affect inference.
    strongest_lines = []
    for family_id in FAMILY_LABELS:
        subset = condition[
            condition["target_family_id"].eq(family_id)
            & ~condition["regime"].eq("near-threshold")
        ].copy()
        subset["selectivity"] = subset["own_jacobian"] - subset["mean_wrong_mechanisms"]
        row = subset.sort_values(["selectivity", "own_jacobian"], ascending=False).iloc[0]
        strongest_lines.append(
            f"- {FAMILY_LABELS[family_id]}: `{row['condition_id']}` moved by "
            f"{row['own_jacobian']:+.3f} log odds ({math.exp(row['own_jacobian']):.2f}× odds); "
            f"the mean wrong-mechanism effect was {row['mean_wrong_mechanisms']:+.3f}."
        )

    report = f"""# Semantic-answer steering v3 confirmatory results

This frozen study is reported in the paper and Supplementary Information.

## What was tested

The experiment asked whether a Jacobian direction derived from a materials mechanism can *causally move the model's scientific answer*, and whether it does so more strongly than unrelated mechanism directions. It used the actual answer words—`grooves` versus `clean`, `hard` versus `soft`, and `higher` versus `lower`—instead of hiding those outcomes behind A/B labels.

- 30 entirely new physical conditions: 10 per family, each with four clear positive, four clear negative, and two near-threshold cases.
- 60 exact prompts because each condition was asked with both answer-word presentation orders.
- Fixed v2 layers only: corrosion layer 16, transformation layer 24, grain size layer 16. No v3 tuning or layer selection.
- At every prompt: three matched Jacobian directions (lens seeds), six wrong-mechanism directions (two mechanisms times three seeds), one matched direct-decoder direction, ten random directions, and five symmetric doses.
- {statistics['design_accounting']['raw_intervention_rows']:,} raw registered rows; the physical condition—not a row, seed, or wording order—is the statistical unit.

## Preregistered family results

{chr(10).join(family_lines)}

The registered family verdict requires all five conditions simultaneously: at least 8/10 positive physical conditions, at least 8/10 same-sign presentation orders, a 95% bootstrap interval above zero for matched minus random, a 95% interval above zero for matched minus wrong mechanisms, and the next-token global top beginning one of the two registered scientific answers in at least 90% of matched-Jacobian rows.

## Integrated result

Across all 30 physical conditions, the matched Jacobian endpoint was {fmt_ci(aggregate['own_jacobian'])}. Random directions produced {fmt_ci(aggregate['random_directions'])}; other material mechanisms produced {fmt_ci(aggregate['mean_wrong_mechanisms'])}; and the direct-decoder control produced {fmt_ci(aggregate['direct_decoder'])}.

The paired matched-minus-random effect was {fmt_ci(aggregate['own_minus_random'])} (Wilcoxon p={aggregate['wilcoxon_own_vs_random_p']:.4g}); matched minus other mechanisms was {fmt_ci(aggregate['own_minus_mean_wrong'])} (p={aggregate['wilcoxon_own_vs_wrong_p']:.4g}); and matched minus direct was {fmt_ci(aggregate['own_minus_direct'])} (p={aggregate['wilcoxon_own_vs_direct_p']:.4g}). The matched direction moved the intended scientific contrast positively in {aggregate['positive_conditions']}/{aggregate['n_conditions']} physical conditions (exact sign-test p={aggregate['sign_test_positive_p']:.4g}).

## Secondary context-aligned result

This analysis was **not preregistered** and is therefore hypothesis-generating. The physical answers and positive/negative regimes were frozen before execution, but the decision to orient each endpoint toward the frozen correct answer was made after observing that the grain-size direction changed sign between refinement and coarsening. Positive values below mean that the same positive mechanism intervention moved toward `higher` after refinement and toward `lower` after coarsening (and analogously toward the frozen correct word in the other families).

{chr(10).join(aligned_lines)}

Across all 24 determinate conditions, the matched expected-answer-aligned effect was {fmt_ci(secondary['aggregate']['own_expected_aligned'])}. It was positive in {secondary['aggregate']['aligned_positive_conditions']}/{secondary['aggregate']['n_determinate_conditions']} conditions (exact sign-test p={secondary['aggregate']['sign_test_aligned_p']:.4g}). Matched minus random was {fmt_ci(secondary['aggregate']['own_minus_random_expected_aligned'])} (Wilcoxon p={secondary['aggregate']['wilcoxon_own_vs_random_p']:.4g}); matched minus other mechanisms was {fmt_ci(secondary['aggregate']['own_minus_wrong_expected_aligned'])} (p={secondary['aggregate']['wilcoxon_own_vs_wrong_p']:.4g}); and matched minus direct was {fmt_ci(secondary['aggregate']['own_minus_direct_expected_aligned'])} (p={secondary['aggregate']['wilcoxon_own_vs_direct_p']:.4g}). Grain-size strengthening showed the cleanest relational pattern: all four refinement cases moved toward `higher`, and all four coarsening cases moved toward `lower`. A new, disjoint matched-pair experiment is required before treating this as confirmatory evidence.

## Cross-mechanism specificity

Each row below is the target scientific answer; each column is the source mechanism used to build the Jacobian direction. The diagonal is the scientifically matched intervention. All directions were reconstructed at the target row's fixed layer, so this comparison is not confounded by layer or dose.

{chr(10).join(matrix_lines)}

## Every physical condition

Endpoint values are changes in positive-answer log odds from −4% to +4% residual-norm dose. `Odds multiplier` is `exp(endpoint)` and expresses how much the positive-versus-negative answer odds changed across that dose range.

{chr(10).join(condition_lines)}

## Strong descriptive examples

These are selected post hoc by matched-minus-wrong-mechanism effect and are for illustration only; they do not enter a separate hypothesis test.

{chr(10).join(strongest_lines)}

## Output integrity and reproducibility

- Clean accuracy on determinate prompt orders: {integrity['clean_expected_accuracy_determinate_prompt_orders']:.1%}.
- Either scientific answer was the clean global top token in {integrity['clean_global_top_valid_fraction_all_prompt_orders']:.1%} of prompt orders.
- Clean probability mass on the two registered answer beginnings: minimum {integrity['clean_pair_probability_minimum']:.3%}, median {integrity['clean_pair_probability_median']:.3%}.
- Matched-Jacobian endpoint choice flips: {integrity['own_jacobian_endpoint_choice_flips']}/{integrity['own_jacobian_endpoint_replicates']} lens-by-prompt replicates.
- Median endpoint correlation among the three independently fitted lens seeds: {integrity['median_lens_seed_endpoint_correlation']:.3f}.
- The frozen protocol, exact prompt text, clean outputs, every dose row, analysis tables, and figures are all retained below.

## Interpretation

A selective positive effect means the frozen mechanism direction is not merely changing generic confidence: moving along that internal direction changes the odds of the corresponding scientific answer more than random vectors or unrelated materials mechanisms. This is causal evidence about a localized representation-to-output pathway. It is not evidence that the plotted/read-out words are a literal private chain of thought, nor by itself proof of complete physical understanding.

## Files

- Frozen protocol: `experiments/semantic-steering-v3-preregistration.json`
- Exact prompts and clean answers: `experiments/semantic-steering-v3_prompt_inventory.md`
- Raw rows: `experiments/semantic-steering-v3_raw.json`
- Machine-readable statistics: `experiments/semantic-steering-v3_statistics.json`
- Prompt-order effects: `experiments/semantic-steering-v3_prompt_effects.csv`
- Physical-condition effects: `experiments/semantic-steering-v3_condition_effects.csv`
- Cross-mechanism matrix: `experiments/semantic-steering-v3_specificity_matrix.csv`
- Expected-answer-aligned secondary table: `experiments/semantic-steering-v3_expected_aligned_secondary.csv`
- Dose curves: `experiments/semantic-steering-v3_curves.csv`
- Figures: `figures/semantic-steering-v3/`
"""
    OUT_MD.write_text(report)


if __name__ == "__main__":
    analyze()
