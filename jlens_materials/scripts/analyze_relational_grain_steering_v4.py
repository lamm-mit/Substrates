#!/usr/bin/env python3
"""Analyze the preregistered matched-pair grain relation confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from analyze_semantic_steering_v3 import (  # noqa: E402
    COLORS,
    CONTROL_LABELS,
    MARKERS,
    bootstrap_mean,
    control_condition_table,
    endpoint_tables,
    fmt_ci,
    safe_wilcoxon,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments/relational-grain-steering-v4_raw.json"
MANIFEST = ROOT / "experiments/relational-grain-steering-v4-preregistration.json"
OUT_MD = ROOT / "experiments/relational-grain-steering-v4_results.md"
OUT_INVENTORY = ROOT / "experiments/relational-grain-steering-v4_prompt_inventory.md"
OUT_PROMPT = ROOT / "experiments/relational-grain-steering-v4_prompt_effects.csv"
OUT_CONDITION = ROOT / "experiments/relational-grain-steering-v4_condition_effects.csv"
OUT_PAIR = ROOT / "experiments/relational-grain-steering-v4_pair_effects.csv"
OUT_CURVES = ROOT / "experiments/relational-grain-steering-v4_curves.csv"
OUT_STATS = ROOT / "experiments/relational-grain-steering-v4_statistics.json"
FIG_DIR = ROOT / "figures/relational-grain-steering-v4"

PAIR_LABELS = {
    "titanium-64-8": "titanium alloy",
    "magnesium-48-12": "magnesium alloy",
    "low-carbon-steel-72-18": "low-carbon steel",
    "silver-40-10": "silver",
    "cobalt-alloy-56-14": "cobalt alloy",
    "bronze-36-9": "bronze",
}


def main() -> None:
    raw = json.loads(RAW.read_text())
    manifest = json.loads(MANIFEST.read_text())
    rows_raw = pd.DataFrame(raw["intervention_rows"])
    preflight = pd.DataFrame(raw["preflight"])
    rows, endpoint_rep, prompt_effects, condition_long = endpoint_tables(rows_raw)
    prompt_effects["pair_id"] = prompt_effects["condition_id"].str.rsplit("--", n=1).str[0]
    prompt_effects["relation"] = prompt_effects["condition_id"].str.rsplit("--", n=1).str[1]
    prompt_effects.to_csv(OUT_PROMPT, index=False)

    condition = control_condition_table(condition_long)
    condition["pair_id"] = condition["condition_id"].str.rsplit("--", n=1).str[0]
    condition["relation"] = condition["condition_id"].str.rsplit("--", n=1).str[1]
    condition["expected_sign"] = condition["relation"].map(
        {"refinement": 1.0, "coarsening": -1.0}
    )
    columns = {
        "own_jacobian": "own_expected_aligned",
        "mean_wrong_mechanisms": "wrong_expected_aligned",
        "direct_decoder": "direct_expected_aligned",
        "random_directions": "random_expected_aligned",
    }
    for source, destination in columns.items():
        condition[destination] = condition[source] * condition["expected_sign"]
    condition.to_csv(OUT_CONDITION, index=False)

    pair = (
        condition.groupby("pair_id", as_index=False)
        .agg(
            own_expected_aligned=("own_expected_aligned", "mean"),
            wrong_expected_aligned=("wrong_expected_aligned", "mean"),
            direct_expected_aligned=("direct_expected_aligned", "mean"),
            random_expected_aligned=("random_expected_aligned", "mean"),
        )
    )
    pair.to_csv(OUT_PAIR, index=False)

    # Technical-replicate dose curves with frozen relation alignment.
    curves = (
        rows.groupby(
            [
                "condition_id", "prompt_id", "presentation_order", "regime",
                "control_type", "dose_percent",
            ],
            as_index=False,
        )
        .agg(
            delta_log_odds=("delta_log_odds", "mean"),
            valid_choice_probability=("valid_choice_probability", "mean"),
            global_top_is_valid_choice=("global_top_is_valid_choice", "mean"),
            kl_clean_to_intervened=("kl_clean_to_intervened", "mean"),
        )
    )
    curves["pair_id"] = curves["condition_id"].str.rsplit("--", n=1).str[0]
    curves["relation"] = curves["condition_id"].str.rsplit("--", n=1).str[1]
    curves["expected_sign"] = curves["relation"].map(
        {"refinement": 1.0, "coarsening": -1.0}
    )
    curves["expected_aligned_delta_log_odds"] = (
        curves["delta_log_odds"] * curves["expected_sign"]
    )
    curves.to_csv(OUT_CURVES, index=False)

    rng = np.random.default_rng(20260716)
    own = pair["own_expected_aligned"].to_numpy()
    wrong = pair["wrong_expected_aligned"].to_numpy()
    direct = pair["direct_expected_aligned"].to_numpy()
    random = pair["random_expected_aligned"].to_numpy()
    own_ci = bootstrap_mean(own, rng)
    own_random_ci = bootstrap_mean(own - random, rng)
    own_wrong_ci = bootstrap_mean(own - wrong, rng)
    own_direct_ci = bootstrap_mean(own - direct, rng)

    own_conditions = condition["own_expected_aligned"].to_numpy()
    both_correct = []
    for _, group in condition.groupby("pair_id"):
        both_correct.append(bool((group["own_expected_aligned"] > 0).all()))
    own_prompt = prompt_effects[prompt_effects["control_type"].eq("own")]
    order_agreements = []
    for _, group in own_prompt.groupby("condition_id"):
        ordered = group.set_index("presentation_order")["endpoint_change_log_odds"]
        a = float(ordered["positive-first"])
        b = float(ordered["negative-first"])
        order_agreements.append(a != 0 and b != 0 and np.sign(a) == np.sign(b))

    own_rows = rows[rows["control_type"].eq("own")]
    top_valid_fraction = float(own_rows["global_top_is_valid_choice"].mean())
    criteria = manifest["design"]["success_criteria"]
    condition_fraction = float(np.mean(own_conditions > 0))
    pair_fraction = float(np.mean(both_correct))
    order_fraction = float(np.mean(order_agreements))
    passed = bool(
        condition_fraction
        >= criteria["individual_conditions_expected_aligned_positive_fraction_at_least"]
        and pair_fraction
        >= criteria["matched_pairs_both_relations_correct_fraction_at_least"]
        and order_fraction
        >= criteria["presentation_orders_same_nonzero_raw_sign_fraction_at_least"]
        and own_random_ci[1] > 0
        and own_wrong_ci[1] > 0
        and own_direct_ci[1] > 0
        and top_valid_fraction
        >= criteria["registered_answer_first_token_is_global_top_fraction_at_least"]
    )

    refinement_raw = condition[condition["relation"].eq("refinement")]["own_jacobian"]
    coarsening_raw = condition[condition["relation"].eq("coarsening")]["own_jacobian"]
    determinate = preflight[preflight["expected_outcome"].notna()]
    clean_accuracy = float(determinate["clean_expected_correct"].mean())
    clean_pair_mass = preflight["clean"].map(
        lambda value: value["valid_choice_probability"]
    )
    seed_endpoint = endpoint_rep[endpoint_rep["control_type"].eq("own")].pivot(
        index="prompt_id", columns="direction_id", values="endpoint_change_log_odds"
    )
    seed_correlation = seed_endpoint.corr()
    seed_correlation_values = seed_correlation.to_numpy()
    upper = seed_correlation_values[
        np.triu_indices_from(seed_correlation_values, k=1)
    ]
    upper = upper[np.isfinite(upper)]

    stats_payload = {
        "study_id": raw["study_id"],
        "registered_pass": passed,
        "design_accounting": {
            "matched_material_pairs": int(pair.shape[0]),
            "physical_conditions": int(condition.shape[0]),
            "prompt_orders": int(preflight.shape[0]),
            "raw_intervention_rows": int(rows_raw.shape[0]),
        },
        "primary": {
            "matched_expected_aligned": own_ci,
            "matched_minus_random": own_random_ci,
            "matched_minus_wrong_mechanisms": own_wrong_ci,
            "matched_minus_direct": own_direct_ci,
            "random_expected_aligned": bootstrap_mean(random, rng),
            "wrong_expected_aligned": bootstrap_mean(wrong, rng),
            "direct_expected_aligned": bootstrap_mean(direct, rng),
            "individual_correct_direction_conditions": int((own_conditions > 0).sum()),
            "n_conditions": int(len(own_conditions)),
            "matched_pairs_both_relations_correct": int(sum(both_correct)),
            "n_pairs": int(len(both_correct)),
            "presentation_order_agreements": int(sum(order_agreements)),
            "n_conditions_with_orders": int(len(order_agreements)),
            "refinement_raw_positive": int((refinement_raw > 0).sum()),
            "n_refinement": int(len(refinement_raw)),
            "coarsening_raw_negative": int((coarsening_raw < 0).sum()),
            "n_coarsening": int(len(coarsening_raw)),
            "sign_test_conditions_p": float(
                stats.binomtest(int((own_conditions > 0).sum()), len(own_conditions), 0.5).pvalue
            ),
            "wilcoxon_matched_vs_random_p": safe_wilcoxon(own - random),
            "wilcoxon_matched_vs_wrong_p": safe_wilcoxon(own - wrong),
            "wilcoxon_matched_vs_direct_p": safe_wilcoxon(own - direct),
        },
        "integrity": {
            "clean_expected_accuracy_prompt_orders": clean_accuracy,
            "own_rows_global_top_valid_fraction": top_valid_fraction,
            "clean_pair_probability_minimum": float(clean_pair_mass.min()),
            "clean_pair_probability_median": float(clean_pair_mass.median()),
            "own_rows_minimum_pair_probability": float(
                own_rows["valid_choice_probability"].min()
            ),
            "own_rows_maximum_kl": float(own_rows["kl_clean_to_intervened"].max()),
            "median_lens_seed_endpoint_correlation": float(np.median(upper)),
        },
    }
    OUT_STATS.write_text(json.dumps(stats_payload, indent=2) + "\n")
    make_figure(condition, pair, curves)
    write_inventory(preflight)
    write_report(stats_payload, condition, pair)
    print(OUT_MD.read_text())


def make_figure(condition: pd.DataFrame, pair: pd.DataFrame, curves: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))

    # A: raw sign reversal for every matched material pair.
    ax = axes[0]
    for pair_id, group in condition.groupby("pair_id"):
        ordered = group.set_index("relation")["own_jacobian"]
        ax.plot(
            [0, 1], [ordered["refinement"], ordered["coarsening"]],
            color=COLORS["own"], alpha=0.35, linewidth=1.2, marker="o",
            markersize=4,
        )
    means = condition.groupby("relation")["own_jacobian"].mean()
    ax.plot(
        [0, 1], [means["refinement"], means["coarsening"]],
        color=COLORS["own"], linewidth=3.0, marker="o", markersize=7,
        markeredgecolor="white", markeredgewidth=0.8,
    )
    ax.axhline(0, color="#888888", linewidth=0.85)
    ax.set_xticks([0, 1], ["Refinement", "Coarsening"])
    ax.set_ylabel("Raw endpoint: higher − lower log odds")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.65)
    ax.text(-0.14, 1.03, "A", transform=ax.transAxes, fontweight="bold")

    # B: pair-level expected-answer-aligned controls.
    ax = axes[1]
    plot_columns = {
        "own": "own_expected_aligned",
        "wrong-mechanism": "wrong_expected_aligned",
        "direct": "direct_expected_aligned",
        "random": "random_expected_aligned",
    }
    rng = np.random.default_rng(20260716)
    for y, (control, column) in enumerate(plot_columns.items()):
        values = pair[column].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            values, y + jitter, color=COLORS[control], marker=MARKERS[control],
            s=24, alpha=0.52, zorder=2,
        )
        estimate = bootstrap_mean(values, rng)
        ax.errorbar(
            estimate[0], y,
            xerr=[[estimate[0] - estimate[1]], [estimate[2] - estimate[0]]],
            color=COLORS[control], marker=MARKERS[control], markersize=7,
            linewidth=1.8, capsize=3, markeredgecolor="white",
            markeredgewidth=0.7, zorder=4,
        )
    ax.axvline(0, color="#888888", linewidth=0.85)
    ax.set_yticks(range(4), [CONTROL_LABELS[value] for value in plot_columns], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pair-level change toward correct answer")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.65)
    ax.text(-0.14, 1.03, "B", transform=ax.transAxes, fontweight="bold")

    # C: expected-answer-aligned dose response, equal weight per pair.
    ax = axes[2]
    doses = [-4, -2, 0, 2, 4]
    for control in plot_columns:
        subset = curves[curves["control_type"].eq(control)]
        pair_curve = (
            subset.groupby(["pair_id", "dose_percent"], as_index=False)
            .agg(value=("expected_aligned_delta_log_odds", "mean"))
        )
        wide = pair_curve.pivot(
            index="pair_id", columns="dose_percent", values="value"
        ).reindex(columns=doses)
        mean = wide.mean(axis=0)
        sem = wide.sem(axis=0)
        ax.plot(
            doses, mean, color=COLORS[control], marker=MARKERS[control],
            markersize=4.5, linewidth=2.0 if control == "own" else 1.4,
            label=CONTROL_LABELS[control],
        )
        ax.fill_between(
            doses, mean - sem, mean + sem, color=COLORS[control],
            alpha=0.10, linewidth=0,
        )
    ax.axhline(0, color="#888888", linewidth=0.85)
    ax.axvline(0, color="#D5D5D5", linewidth=0.8)
    ax.set_xlabel("Intervention dose (% residual norm)")
    ax.set_ylabel("Change toward correct answer")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.65)
    ax.legend(frameon=False, fontsize=7.7, loc="upper left")
    ax.text(-0.14, 1.03, "C", transform=ax.transAxes, fontweight="bold")

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.94, wspace=0.38)
    fig.savefig(FIG_DIR / "relational_confirmation.png", dpi=300)
    fig.savefig(FIG_DIR / "relational_confirmation.pdf")
    plt.close(fig)


def write_inventory(preflight: pd.DataFrame) -> None:
    lines = [
        "# Relational grain steering v4 exact prompt inventory",
        "",
        "Every prompt below was frozen before v4 execution. Each physical condition "
        "appears in two answer-word presentation orders.",
        "",
    ]
    for condition_id, group in preflight.groupby("condition_id", sort=False):
        pair_id, relation = condition_id.rsplit("--", 1)
        lines.extend([
            f"## {PAIR_LABELS[pair_id]} — {relation}",
            "",
            f"Pair id: `{pair_id}`. Expected answer: `{group.iloc[0]['expected_outcome']}`.",
            "",
        ])
        for row in group.itertuples():
            clean = row.clean
            lines.extend([
                f"- `{row.presentation_order}` — clean `{clean['choice']}`, "
                f"higher-minus-lower log odds {clean['positive_log_odds']:+.3f}, "
                f"pair mass {clean['valid_choice_probability']:.3%}.",
                f"  - {row.user}",
            ])
        lines.append("")
    OUT_INVENTORY.write_text("\n".join(lines) + "\n")


def write_report(stats_payload: dict, condition: pd.DataFrame, pair: pd.DataFrame) -> None:
    primary = stats_payload["primary"]
    integrity = stats_payload["integrity"]
    pair_lines = [
        "| Matched material pair | Matched J | Other mechanisms | Direct | Random | Both relations correct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pair.itertuples():
        group = condition[condition["pair_id"].eq(row.pair_id)]
        both = bool((group["own_expected_aligned"] > 0).all())
        pair_lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {row.own_expected_aligned:+.3f} | "
            f"{row.wrong_expected_aligned:+.3f} | {row.direct_expected_aligned:+.3f} | "
            f"{row.random_expected_aligned:+.3f} | {'yes' if both else 'no'} |"
        )
    condition_lines = [
        "| Material | Relation | Raw matched endpoint | Expected-aligned endpoint |",
        "|---|---|---:|---:|",
    ]
    for row in condition.sort_values(["pair_id", "relation"]).itertuples():
        condition_lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {row.relation} | "
            f"{row.own_jacobian:+.3f} | {row.own_expected_aligned:+.3f} |"
        )

    report = f"""# Relational grain steering v4 confirmatory results

This study prospectively tested the context-sensitive grain-size result first noticed post hoc in v3. It was frozen before any v4 model output and is reported in the paper and Supplementary Information.

## What was tested

Six entirely new materials formed matched pairs. Within each pair, alloy identity, the two grain-size values, all controlled covariates, and the answer words were held fixed; only the direction of change was reversed. The same frozen layer-16 grain-size Jacobian direction was then applied to both prompts. A genuine relation-sensitive result should increase `higher` after refinement but increase `lower` after coarsening.

- 6 matched material pairs, 12 physical conditions, and 24 exact answer-order prompts.
- 2,400 registered dose rows: three lens seeds, two wrong-mechanism directions across three seeds, direct decoder, ten random controls, and five symmetric doses.
- Primary statistical unit: matched material pair.
- No v4 layer, direction, concept, dose, prompt, or threshold tuning.

## Preregistered result

**Verdict: {'PASS' if stats_payload['registered_pass'] else 'FAIL'}.**

The matched direction moved toward the frozen correct answer by {fmt_ci(primary['matched_expected_aligned'])}. Matched minus random was {fmt_ci(primary['matched_minus_random'])} (Wilcoxon p={primary['wilcoxon_matched_vs_random_p']:.4g}); matched minus the two other materials mechanisms was {fmt_ci(primary['matched_minus_wrong_mechanisms'])} (p={primary['wilcoxon_matched_vs_wrong_p']:.4g}); and matched minus direct decoder was {fmt_ci(primary['matched_minus_direct'])} (p={primary['wilcoxon_matched_vs_direct_p']:.4g}).

The effect moved in the correct direction in {primary['individual_correct_direction_conditions']}/{primary['n_conditions']} individual conditions and in both relations for {primary['matched_pairs_both_relations_correct']}/{primary['n_pairs']} material pairs. It was raw-positive for {primary['refinement_raw_positive']}/{primary['n_refinement']} refinement conditions and raw-negative for {primary['coarsening_raw_negative']}/{primary['n_coarsening']} coarsening conditions. The two answer-word orders had the same nonzero raw sign in {primary['presentation_order_agreements']}/{primary['n_conditions_with_orders']} conditions.

{chr(10).join(pair_lines)}

## Every physical condition

`Raw matched endpoint` is the change in higher-minus-lower log odds from −4% to +4%. `Expected-aligned` reverses the sign for coarsening, so positive always means movement toward the frozen physically correct answer.

{chr(10).join(condition_lines)}

## Integrity

- Clean expected-answer accuracy across prompt orders: {integrity['clean_expected_accuracy_prompt_orders']:.1%}.
- A registered answer beginning remained global top-1 in {integrity['own_rows_global_top_valid_fraction']:.1%} of matched-direction rows.
- Clean answer-pair probability: minimum {integrity['clean_pair_probability_minimum']:.3%}, median {integrity['clean_pair_probability_median']:.3%}.
- Minimum pair mass under matched interventions: {integrity['own_rows_minimum_pair_probability']:.3%}; maximum clean-to-intervened KL: {integrity['own_rows_maximum_kl']:.4f}.
- Median endpoint correlation across the three independent lens seeds: {integrity['median_lens_seed_endpoint_correlation']:.3f}.

## Interpretation

If the registered test passes, the direction is not acting as a fixed `higher`-token bias: its effect reverses when the physical relation reverses, while unrelated materials directions and the direct decoder are controlled at the same layer and dose. This is evidence for a context-dependent causal pathway linking a grain-boundary mechanism representation to a materials decision. It remains a localized intervention result, not a literal chain of thought or a claim of unrestricted understanding.

## Files

- Frozen protocol: `experiments/relational-grain-steering-v4-preregistration.json`
- Exact prompts and clean answers: `experiments/relational-grain-steering-v4_prompt_inventory.md`
- Raw rows: `experiments/relational-grain-steering-v4_raw.json`
- Statistics: `experiments/relational-grain-steering-v4_statistics.json`
- Prompt, condition, pair, and curve CSV files: `experiments/relational-grain-steering-v4_*.csv`
- Figure: `figures/relational-grain-steering-v4/relational_confirmation.pdf`
"""
    OUT_MD.write_text(report)


if __name__ == "__main__":
    main()
