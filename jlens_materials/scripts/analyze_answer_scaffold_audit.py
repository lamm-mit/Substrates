#!/usr/bin/env python3
"""Post-hoc audit of pre-option versus ordinary answer-scaffold readout."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_lexical_adversarial_representation as shared
from analyze_answer_code_binding import build_endpoint_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "experiments"
    / "late-physics-representation-replication-2026-07-17"
)
CODE = ROOT / "experiments" / "answer-code-binding-2026-07-17"
OUT = ROOT / "experiments" / "answer-scaffold-audit-2026-07-17"
FIG = OUT / "figures"


def final_relation_rows(raw: dict, manifest: dict) -> pd.DataFrame:
    clean = pd.DataFrame(raw["clean_rows"])
    triplets = pd.DataFrame(manifest["triplets"])[
        ["triplet_id", "anchor_expected_outcome"]
    ]
    families = pd.DataFrame(manifest["families"])[
        ["family_id", "outcome_positive"]
    ]
    clean = clean.merge(triplets, on="triplet_id", validate="many_to_one")
    clean = clean.merge(families, on="family_id", validate="many_to_one")
    clean["anchor_sign"] = np.where(
        clean["anchor_expected_outcome"] == clean["outcome_positive"],
        1.0,
        -1.0,
    )
    clean["anchor_signed_physical"] = (
        clean["positive_minus_negative_log_odds"] * clean["anchor_sign"]
    )
    pivot = clean.pivot(
        index=["triplet_id", "family_id"],
        columns="variant",
        values="anchor_signed_physical",
    ).reset_index()
    pivot["physics_separation"] = (
        0.5 * (pivot["anchor"] + pivot["physics_paraphrase"])
        - pivot["lexical_counterfactual"]
    )
    pivot["method"] = "ordinary_answer_scaffold"
    pivot["position"] = "after ordinary answer choices"
    return pivot[[
        "method",
        "position",
        "triplet_id",
        "family_id",
        "physics_separation",
    ]]


def bootstrap(frame: pd.DataFrame, value: str, seed: int) -> dict:
    return shared.two_stage_bootstrap(
        frame,
        value,
        n_resamples=30000,
        seed=seed,
    )


def figure(
    comparison: pd.DataFrame,
    family: pd.DataFrame,
    source_clean: pd.DataFrame,
    code_clean: pd.DataFrame,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
    })
    teal = "#167D8D"
    purple = "#6F5AA8"
    rust = "#C75B39"
    gray = "#777777"
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), constrained_layout=True)
    ax_a, ax_b, ax_c = axes
    stages = [
        ("preoption_direct", "before choices\ndirect", purple),
        ("preoption_jacobian", "before choices\nJacobian", teal),
        ("ordinary_answer_scaffold", "after ordinary\nhigher/lower", rust),
    ]
    x = np.arange(len(stages))
    for _, row in comparison.iterrows():
        values = [float(row[key]) for key, _, _ in stages]
        ax_a.plot(x, values, color="#C8C8C8", linewidth=0.65, alpha=0.65)
    for index, (key, _, color) in enumerate(stages):
        values = comparison[key].to_numpy(dtype=float)
        ax_a.scatter(
            np.full(len(values), index),
            values,
            color=color,
            s=15,
            alpha=0.75,
            zorder=2,
        )
        ax_a.scatter(
            index,
            values.mean(),
            color="white",
            edgecolor=color,
            linewidth=1.5,
            s=75,
            zorder=3,
        )
    ax_a.axhline(0, color=gray, linewidth=0.8)
    ax_a.set_xticks(x, [label for _, label, _ in stages])
    ax_a.set_ylabel("Physics-relation separation\n(logit-difference units)")
    ax_a.text(0.02, 0.97, "A", transform=ax_a.transAxes, va="top", fontweight="bold")

    short = {
        "crosslink-density-modulus": "crosslink",
        "dislocation-density-strength": "dislocation",
        "obstacle-spacing-orowan": "obstacle",
        "particle-fraction-modulus": "particles",
        "pearlite-spacing-strength": "pearlite",
        "porosity-modulus": "porosity",
    }
    order = list(short)
    y = np.arange(len(order))
    final_minus_pre = family.set_index("family_id").loc[order]
    ax_b.barh(
        y,
        final_minus_pre["ordinary_minus_preoption_direct"],
        color=rust,
        height=0.62,
    )
    ax_b.axvline(0, color=gray, linewidth=0.8)
    ax_b.set_yticks(y, [short[value] for value in order])
    ax_b.set_xlabel("Gain after ordinary answer choices")
    ax_b.text(0.02, 0.97, "B", transform=ax_b.transAxes, va="top", fontweight="bold")

    source_accuracy = (
        source_clean.groupby("family_id")["registered_pair_correct"]
        .mean()
        .loc[order]
    )
    code_accuracy = (
        code_clean.groupby("family_id")["registered_code_correct"]
        .mean()
        .loc[order]
    )
    height = 0.34
    ax_c.barh(
        y - height / 2,
        source_accuracy,
        height=height,
        color=teal,
        label="ordinary words",
    )
    ax_c.barh(
        y + height / 2,
        code_accuracy,
        height=height,
        color="#A9A9A9",
        label="arbitrary A/B",
    )
    ax_c.axvline(0.5, color=gray, linestyle="--", linewidth=0.8)
    ax_c.set_xlim(0, 1)
    ax_c.set_yticks(y, [short[value] for value in order])
    ax_c.set_xlabel("Clean forced-pair accuracy")
    ax_c.text(0.02, 0.97, "C", transform=ax_c.transAxes, va="top", fontweight="bold")
    handles, labels = ax_c.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )

    for suffix in ["png", "pdf"]:
        fig.savefig(
            FIG / f"answer-scaffold-audit.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_raw_path = SOURCE / "raw.json"
    source_manifest_path = SOURCE / "prompt_manifest.json"
    source_statistics_path = SOURCE / "statistics.json"
    code_raw_path = CODE / "raw.json"
    code_manifest_path = CODE / "prompt_manifest.json"
    code_protocol_path = CODE / "protocol-amendment-v1.json"
    source_raw = json.loads(source_raw_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    code_raw = json.loads(code_raw_path.read_text())
    code_manifest = json.loads(code_manifest_path.read_text())

    endpoint = build_endpoint_rows(code_raw, code_manifest)
    pre = endpoint[
        (endpoint["position"] == "checkpoint")
        & (endpoint["layer"] == 39)
        & endpoint["method"].isin(["direct", "jacobian_ensemble"])
    ][[
        "method",
        "triplet_id",
        "family_id",
        "physics_separation",
    ]].copy()
    pre["method"] = pre["method"].map({
        "direct": "preoption_direct",
        "jacobian_ensemble": "preoption_jacobian",
    })
    ordinary = final_relation_rows(source_raw, source_manifest)
    long = pd.concat([pre, ordinary], ignore_index=True)
    long.to_csv(OUT / "triplet_relation_separation.csv", index=False)
    comparison = long.pivot(
        index=["triplet_id", "family_id"],
        columns="method",
        values="physics_separation",
    ).reset_index()
    comparison["ordinary_minus_preoption_direct"] = (
        comparison["ordinary_answer_scaffold"]
        - comparison["preoption_direct"]
    )
    comparison["ordinary_minus_preoption_jacobian"] = (
        comparison["ordinary_answer_scaffold"]
        - comparison["preoption_jacobian"]
    )
    comparison.to_csv(OUT / "paired_scaffold_contrasts.csv", index=False)
    family = comparison.groupby("family_id", as_index=False).mean(numeric_only=True)
    family.to_csv(OUT / "family_scaffold_contrasts.csv", index=False)

    statistics = {
        "study_id": "answer-scaffold-audit-2026-07-17",
        "status": (
            "post-hoc paired audit motivated after inspecting the prospective "
            "late-transition and answer-code results; no new model pass"
        ),
        "question": (
            "How much does registered physical-relation separation change "
            "between a state immediately before any answer choices and the "
            "ordinary final state after the semantic higher/lower choices?"
        ),
        "provenance": {
            "source_raw_sha256": shared.sha256(source_raw_path),
            "source_manifest_sha256": shared.sha256(source_manifest_path),
            "source_statistics_sha256": shared.sha256(source_statistics_path),
            "code_raw_sha256": shared.sha256(code_raw_path),
            "code_manifest_sha256": shared.sha256(code_manifest_path),
            "code_protocol_sha256": shared.sha256(code_protocol_path),
        },
        "endpoint_notes": {
            "preoption": (
                "Layer 39 (95.1% depth) at the contextual checkpoint, which "
                "has seen the complete scientific stem but no answer choices."
            ),
            "ordinary": (
                "Final model output state from the completed disjoint "
                "replication after the ordinary semantic answer pair."
            ),
            "comparability": (
                "Both use the same family-specific positive-minus-negative "
                "decoder contrast and the same triplet relation-separation "
                "formula. Positions and prompt suffixes differ by design."
            ),
        },
        "bootstrap": {
            "preoption_direct": bootstrap(
                long[long["method"] == "preoption_direct"],
                "physics_separation",
                20260751,
            ),
            "preoption_jacobian": bootstrap(
                long[long["method"] == "preoption_jacobian"],
                "physics_separation",
                20260752,
            ),
            "ordinary_answer_scaffold": bootstrap(
                long[long["method"] == "ordinary_answer_scaffold"],
                "physics_separation",
                20260753,
            ),
            "ordinary_minus_preoption_direct": bootstrap(
                comparison,
                "ordinary_minus_preoption_direct",
                20260754,
            ),
            "ordinary_minus_preoption_jacobian": bootstrap(
                comparison,
                "ordinary_minus_preoption_jacobian",
                20260755,
            ),
        },
        "behavior": {
            "ordinary_registered_pair_accuracy": float(
                pd.DataFrame(source_raw["clean_rows"])[
                    "registered_pair_correct"
                ].mean()
            ),
            "arbitrary_code_registered_pair_accuracy": float(
                pd.DataFrame(code_raw["clean_rows"])[
                    "registered_code_correct"
                ].mean()
            ),
        },
        "guardrails": [
            "This audit was defined after inspecting both source experiments and is descriptive, not confirmatory.",
            "The large scaffold contrast may reflect answer-word exposure, final-position computation, instruction following, or all three.",
            "It does not show that earlier states contain no distributed physical information; it tests one registered decoder contrast.",
        ],
    }
    shared.dump_json(OUT / "statistics.json", statistics)
    source_clean = pd.DataFrame(source_raw["clean_rows"])
    code_clean = pd.DataFrame(code_raw["clean_rows"])
    figure(comparison, family, source_clean, code_clean)
    before = statistics["bootstrap"]["preoption_direct"]
    after = statistics["bootstrap"]["ordinary_answer_scaffold"]
    gain = statistics["bootstrap"]["ordinary_minus_preoption_direct"]
    (OUT / "RESULTS.md").write_text(
        "\n".join([
            "# Post-hoc answer-scaffold audit",
            "",
            "This analysis is outside the paper and is explicitly post-hoc.",
            "",
            "## Result",
            "",
            (
                f"At 95.1% depth before any answer choices, direct decoding "
                f"gave a physics-relation separation of **{before['mean']:+.3f}** "
                f"(two-stage 95% CI {before['ci95'][0]:+.3f} to "
                f"{before['ci95'][1]:+.3f}). After the ordinary semantic "
                f"`higher/lower` scaffold, the final-state separation was "
                f"**{after['mean']:+.3f}** ({after['ci95'][0]:+.3f} to "
                f"{after['ci95'][1]:+.3f}). The paired gain was "
                f"**{gain['mean']:+.3f}** ({gain['ci95'][0]:+.3f} to "
                f"{gain['ci95'][1]:+.3f})."
            ),
            "",
            (
                f"Ordinary answer accuracy was "
                f"{statistics['behavior']['ordinary_registered_pair_accuracy']:.1%}, "
                f"compared with "
                f"{statistics['behavior']['arbitrary_code_registered_pair_accuracy']:.1%} "
                "under the A/B manipulation."
            ),
            "",
            "## Meaning",
            "",
            (
                "The registered physical-answer contrast becomes dramatically "
                "more separated after the prompt explicitly supplies the answer "
                "words. This makes answer-scaffold preparation a plausible "
                "contributor to the robust late transition. It does not erase "
                "the transition result, but it prevents interpreting that result "
                "alone as proof of an option-free internal physical variable."
            ),
            "",
            "## Files",
            "",
            "- `triplet_relation_separation.csv`: all stages and triplets.",
            "- `paired_scaffold_contrasts.csv`: paired independent units.",
            "- `family_scaffold_contrasts.csv`: six family summaries.",
            "- `statistics.json`: fingerprints, bootstraps, and guardrails.",
            "- `figures/answer-scaffold-audit.{png,pdf}`.",
            "",
        ]) + "\n"
    )
    (OUT / "README.md").write_text(
        "\n".join([
            "# Answer-scaffold audit",
            "",
            "Post-hoc, no-new-forward-pass audit reported in the Supplementary Information.",
            "",
            "```bash",
            "python scripts/analyze_answer_scaffold_audit.py",
            "```",
            "",
            "Read `RESULTS.md` and `statistics.json` together.",
            "",
        ]) + "\n"
    )
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
