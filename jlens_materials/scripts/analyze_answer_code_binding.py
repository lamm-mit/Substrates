#!/usr/bin/env python3
"""Analyze the frozen arbitrary answer-code binding falsification study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import analyze_lexical_adversarial_representation as shared

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "answer-code-binding-2026-07-17"
PROTOCOL_PATH = OUT / "protocol-amendment-v1.json"
MANIFEST_PATH = OUT / "prompt_manifest.json"
RAW_PATH = OUT / "raw.json"
STATES_PATH = OUT / "representations.npz"
FIG = OUT / "figures"

VARIANTS = ["anchor", "physics_paraphrase", "lexical_counterfactual"]
METHODS = [
    "direct",
    "jacobian_seed0",
    "jacobian_seed1",
    "jacobian_seed2",
    "jacobian_ensemble",
]


def build_endpoint_rows(raw: dict, manifest: dict) -> pd.DataFrame:
    frame = pd.DataFrame(raw["readout_rows"])
    triplets = pd.DataFrame(manifest["triplets"])[
        ["triplet_id", "anchor_expected_outcome"]
    ]
    families = pd.DataFrame(manifest["families"])[
        ["family_id", "outcome_positive"]
    ]
    frame = frame.merge(triplets, on="triplet_id", validate="many_to_one")
    frame = frame.merge(families, on="family_id", validate="many_to_one")
    frame["anchor_sign"] = np.where(
        frame["anchor_expected_outcome"] == frame["outcome_positive"],
        1.0,
        -1.0,
    )
    frame["anchor_signed_physical"] = (
        frame["physical_positive_minus_negative"] * frame["anchor_sign"]
    )
    keys = [
        "method",
        "position",
        "layer",
        "depth_percent",
        "triplet_id",
        "family_id",
    ]
    pivot = frame.pivot(
        index=keys,
        columns="variant",
        values=["anchor_signed_physical", "code_A_minus_B"],
    ).reset_index()
    pivot.columns = [
        "_".join(value).rstrip("_") if isinstance(value, tuple) else value
        for value in pivot.columns
    ]
    pivot["physics_separation"] = (
        0.5 * (
            pivot["anchor_signed_physical_anchor"]
            + pivot["anchor_signed_physical_physics_paraphrase"]
        )
        - pivot["anchor_signed_physical_lexical_counterfactual"]
    )
    pivot["code_separation"] = (
        0.5 * (
            pivot["code_A_minus_B_anchor"]
            + pivot["code_A_minus_B_lexical_counterfactual"]
        )
        - pivot["code_A_minus_B_physics_paraphrase"]
    )
    return pivot


def band_rows(
    frame: pd.DataFrame,
    position: str,
    value: str,
    low: float,
    high: float,
) -> pd.DataFrame:
    return (
        frame[
            (frame["position"] == position)
            & frame["depth_percent"].between(low, high)
        ]
        .groupby(["method", "triplet_id", "family_id"], as_index=False)[value]
        .mean()
    )


def bootstrap_all(frame: pd.DataFrame, value: str, seed: int) -> dict:
    return {
        method: shared.two_stage_bootstrap(
            frame[frame["method"] == method],
            value,
            n_resamples=30000,
            seed=seed + index,
        )
        for index, method in enumerate(METHODS)
    }


def paired_contrast(
    frame: pd.DataFrame,
    value: str,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    pivot = frame.pivot(
        index=["triplet_id", "family_id"],
        columns="method",
        values=value,
    ).reset_index()
    pivot["jacobian_minus_direct"] = (
        pivot["jacobian_ensemble"] - pivot["direct"]
    )
    result = shared.two_stage_bootstrap(
        pivot,
        "jacobian_minus_direct",
        n_resamples=30000,
        seed=seed,
    )
    family = pivot.groupby("family_id")["jacobian_minus_direct"].mean().to_numpy()
    result["exact_family_sign_flip_p"] = shared.exact_family_sign_flip(family)
    return pivot, result


def breadth(frame: pd.DataFrame, value: str) -> dict:
    primary = frame[frame["method"] == "jacobian_ensemble"]
    family = primary.groupby("family_id", as_index=False)[value].mean()
    n_positive = int((primary[value] > 0).sum())
    n_family_positive = int((family[value] > 0).sum())
    return {
        "positive_triplets": n_positive,
        "n_triplets": len(primary),
        "exact_two_sided_triplet_sign_p": float(
            stats.binomtest(
                n_positive, len(primary), p=0.5, alternative="two-sided"
            ).pvalue
        ),
        "positive_families": n_family_positive,
        "n_families": len(family),
        "triplet_breadth_pass": n_positive >= 18,
        "family_breadth_pass": n_family_positive >= 5,
    }


def geometry_rows(
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
) -> pd.DataFrame:
    prompt_ids = [str(value) for value in arrays["prompt_ids"]]
    expected = [row["prompt_id"] for row in manifest["prompts"]]
    if prompt_ids != expected:
        raise RuntimeError("Representation prompt order does not match manifest.")
    prompt_index = {
        prompt_id: index for index, prompt_id in enumerate(prompt_ids)
    }
    layers = arrays["layers"].astype(int)
    jacobian = arrays["jacobian_decoder_basis"].astype(np.float32)
    output = []
    for position_index, position in enumerate(
        [str(value) for value in arrays["positions"]]
    ):
        methods = {
            "raw_residual": arrays["raw_states"][
                position_index
            ].astype(np.float32),
            "direct": arrays["direct_decoder_basis"][
                position_index
            ].astype(np.float32),
            "jacobian_seed0": jacobian[0, position_index],
            "jacobian_seed1": jacobian[1, position_index],
            "jacobian_seed2": jacobian[2, position_index],
            "jacobian_ensemble": jacobian[:, position_index].mean(axis=0),
        }
        rows = shared.similarity_rows(
            methods,
            layers,
            manifest["triplets"],
            prompt_index,
            centered=True,
            accumulator_dtype=np.float64,
        )
        rows.insert(1, "position", position)
        output.append(rows)
    return pd.concat(output, ignore_index=True)


def clean_summary(raw: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    clean = pd.DataFrame(raw["clean_rows"])
    triplets = []
    for triplet_id, group in clean.groupby("triplet_id", sort=True):
        group = group.set_index("variant")
        triplets.append({
            "triplet_id": triplet_id,
            "family_id": str(group.iloc[0]["family_id"]),
            "all_three_code_correct": bool(
                group.loc[VARIANTS, "registered_code_correct"].all()
            ),
            "predicted_registered_pattern": "".join(
                str(group.loc[variant, "predicted_code"])
                for variant in VARIANTS
            ),
            "expected_registered_pattern": "ABA",
        })
    triplet_frame = pd.DataFrame(triplets)
    summary = {
        "registered_code_pair_accuracy": float(
            clean["registered_code_correct"].mean()
        ),
        "global_top_token_is_A_or_B_rate": float(
            clean["global_top_is_code"].mean()
        ),
        "triplets_all_three_code_correct": int(
            triplet_frame["all_three_code_correct"].sum()
        ),
        "n_triplets": len(triplet_frame),
        "by_variant_accuracy": {
            variant: float(group["registered_code_correct"].mean())
            for variant, group in clean.groupby("variant")
        },
        "by_family_accuracy": {
            family: float(group["registered_code_correct"].mean())
            for family, group in clean.groupby("family_id")
        },
    }
    return clean, triplet_frame, summary


def layer_ci(
    frame: pd.DataFrame,
    value: str,
    methods: list[str],
) -> pd.DataFrame:
    return shared.layer_summary_with_ci(
        frame,
        value,
        methods,
        n_resamples=5000,
    )


def make_figure(
    endpoints: pd.DataFrame,
    checkpoint_physics: pd.DataFrame,
    final_code: pd.DataFrame,
    geometry: pd.DataFrame,
    clean: pd.DataFrame,
    band: tuple[float, float],
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })
    teal = "#167D8D"
    purple = "#6F5AA8"
    rust = "#C75B39"
    gray = "#777777"
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    physics_curve = layer_ci(
        endpoints[endpoints["position"] == "checkpoint"],
        "physics_separation",
        ["jacobian_ensemble", "direct"],
    )
    for method, color, label in [
        ("jacobian_ensemble", teal, "Jacobian"),
        ("direct", purple, "direct"),
    ]:
        subset = physics_curve[
            physics_curve["method"] == method
        ].sort_values("layer")
        x = subset["depth_percent"].to_numpy()
        mean = subset["mean"].to_numpy()
        low = np.asarray([value[0] for value in subset["ci95"]])
        high = np.asarray([value[1] for value in subset["ci95"]])
        ax_a.plot(x, mean, color=color, linewidth=1.7, label=label)
        ax_a.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
    ax_a.axhline(0, color=gray, linewidth=0.8)
    ax_a.axvspan(*band, color=teal, alpha=0.08, linewidth=0)
    ax_a.set_xlabel("Layer depth (%)")
    ax_a.set_ylabel("Physics separation before code\n(logit-difference units)")
    ax_a.text(0.02, 0.97, "A", transform=ax_a.transAxes, va="top", fontweight="bold")

    for position, color, label in [
        ("checkpoint", gray, "before mapping"),
        ("final_prompt", rust, "after mapping"),
    ]:
        curve = layer_ci(
            endpoints[
                (endpoints["position"] == position)
                & (endpoints["method"] == "jacobian_ensemble")
            ],
            "code_separation",
            ["jacobian_ensemble"],
        ).sort_values("layer")
        x = curve["depth_percent"].to_numpy()
        mean = curve["mean"].to_numpy()
        low = np.asarray([value[0] for value in curve["ci95"]])
        high = np.asarray([value[1] for value in curve["ci95"]])
        ax_b.plot(x, mean, color=color, linewidth=1.8, label=label)
        ax_b.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
    ax_b.axhline(0, color=gray, linewidth=0.8)
    ax_b.axvspan(*band, color=teal, alpha=0.08, linewidth=0)
    ax_b.set_xlabel("Layer depth (%)")
    ax_b.set_ylabel("Arbitrary A/B separation\n(Jacobian logit-difference units)")
    ax_b.text(0.02, 0.97, "B", transform=ax_b.transAxes, va="top", fontweight="bold")

    short = {
        "crosslink-density-modulus": "crosslink",
        "dislocation-density-strength": "dislocation",
        "obstacle-spacing-orowan": "obstacle",
        "particle-fraction-modulus": "particles",
        "pearlite-spacing-strength": "pearlite",
        "porosity-modulus": "porosity",
    }
    family_order = list(short)
    phys_family = (
        checkpoint_physics[
            checkpoint_physics["method"] == "jacobian_ensemble"
        ]
        .groupby("family_id")["physics_separation"]
        .mean()
    )
    code_family = (
        final_code[final_code["method"] == "jacobian_ensemble"]
        .groupby("family_id")["code_separation"]
        .mean()
    )
    y = np.arange(len(family_order))
    for index, family in enumerate(family_order):
        ax_c.plot(
            [phys_family[family], code_family[family]],
            [index, index],
            color="#B8B8B8",
            linewidth=1.0,
            zorder=1,
        )
    ax_c.scatter(
        phys_family.loc[family_order],
        y,
        color=teal,
        marker="o",
        s=38,
        label="physics before code",
        zorder=2,
    )
    ax_c.scatter(
        code_family.loc[family_order],
        y,
        color=rust,
        marker="s",
        s=34,
        label="A/B after mapping",
        zorder=2,
    )
    ax_c.axvline(0, color=gray, linewidth=0.8)
    ax_c.set_yticks(y, [short[value] for value in family_order])
    ax_c.set_xlabel("Frozen late-window separation")
    ax_c.text(0.02, 0.97, "C", transform=ax_c.transAxes, va="top", fontweight="bold")

    family_accuracy = (
        clean.groupby("family_id")["registered_code_correct"].mean()
        .loc[family_order]
    )
    bars = ax_d.barh(
        y,
        family_accuracy,
        color=np.where(family_accuracy >= 2 / 3, teal, "#B7B7B7"),
        height=0.62,
    )
    ax_d.axvline(0.5, color=gray, linestyle="--", linewidth=0.9)
    ax_d.set_xlim(0, 1)
    ax_d.set_yticks(y, [short[value] for value in family_order])
    ax_d.set_xlabel("Clean forced-pair A/B accuracy")
    for bar, value in zip(bars, family_accuracy):
        ax_d.text(
            min(float(value) + 0.025, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}",
            va="center",
            fontsize=8,
        )
    ax_d.text(0.02, 0.97, "D", transform=ax_d.transAxes, va="top", fontweight="bold")

    handles_a, labels_a = ax_a.get_legend_handles_labels()
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    handles_c, labels_c = ax_c.get_legend_handles_labels()
    fig.legend(
        handles_a + handles_b + handles_c,
        labels_a + labels_b + labels_c,
        loc="outside lower center",
        ncol=6,
        frameon=False,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(
            FIG / f"answer-code-binding.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    raw = json.loads(RAW_PATH.read_text())
    if raw["provenance"]["protocol_sha256"] != shared.sha256(PROTOCOL_PATH):
        raise RuntimeError("Raw output does not match the amended frozen protocol.")
    for key in ["prompt_manifest", "runner", "source_statistics"]:
        if protocol["inputs"][f"{key}_sha256"] != shared.sha256(
            ROOT / protocol["inputs"][key]
        ):
            raise RuntimeError(f"Frozen input fingerprint changed: {key}")
    endpoints = build_endpoint_rows(raw, manifest)
    endpoints.to_csv(OUT / "layer_readout_separations.csv", index=False)
    low, high = protocol["registered_band_percent"]
    checkpoint_physics = band_rows(
        endpoints, "checkpoint", "physics_separation", low, high
    )
    final_code = band_rows(
        endpoints, "final_prompt", "code_separation", low, high
    )
    checkpoint_code = band_rows(
        endpoints, "checkpoint", "code_separation", low, high
    )
    checkpoint_physics.to_csv(
        OUT / "checkpoint_physics_triplet_separation.csv", index=False
    )
    final_code.to_csv(OUT / "final_code_triplet_separation.csv", index=False)
    checkpoint_code.to_csv(
        OUT / "checkpoint_code_triplet_separation.csv", index=False
    )
    transition = final_code.merge(
        checkpoint_code,
        on=["method", "triplet_id", "family_id"],
        suffixes=("_final", "_checkpoint"),
        validate="one_to_one",
    )
    transition["final_minus_checkpoint_code_separation"] = (
        transition["code_separation_final"]
        - transition["code_separation_checkpoint"]
    )
    transition.to_csv(
        OUT / "code_binding_transition_triplet.csv", index=False
    )

    checkpoint_stats = bootstrap_all(
        checkpoint_physics, "physics_separation", 20260741
    )
    code_stats = bootstrap_all(final_code, "code_separation", 20260841)
    transition_stats = bootstrap_all(
        transition,
        "final_minus_checkpoint_code_separation",
        20260941,
    )
    _, checkpoint_contrast = paired_contrast(
        checkpoint_physics, "physics_separation", 20261041
    )
    _, code_contrast = paired_contrast(
        final_code, "code_separation", 20261141
    )
    _, transition_contrast = paired_contrast(
        transition, "final_minus_checkpoint_code_separation", 20261241
    )

    checkpoint_breadth = breadth(
        checkpoint_physics, "physics_separation"
    )
    code_breadth = breadth(final_code, "code_separation")
    transition_breadth = breadth(
        transition, "final_minus_checkpoint_code_separation"
    )
    j_checkpoint = checkpoint_stats["jacobian_ensemble"]
    j_code = code_stats["jacobian_ensemble"]
    j_transition = transition_stats["jacobian_ensemble"]
    decision = {
        "checkpoint_physics_ci_above_zero": j_checkpoint["ci95"][0] > 0,
        "final_code_ci_above_zero": j_code["ci95"][0] > 0,
        "code_transition_ci_above_zero": j_transition["ci95"][0] > 0,
        "checkpoint_physics_breadth": checkpoint_breadth,
        "final_code_breadth": code_breadth,
    }
    decision["frozen_success_rule_pass"] = bool(
        decision["checkpoint_physics_ci_above_zero"]
        and decision["final_code_ci_above_zero"]
        and decision["code_transition_ci_above_zero"]
        and checkpoint_breadth["triplet_breadth_pass"]
        and checkpoint_breadth["family_breadth_pass"]
        and code_breadth["triplet_breadth_pass"]
        and code_breadth["family_breadth_pass"]
    )

    clean, clean_triplets, behavior = clean_summary(raw)
    clean.to_csv(OUT / "clean_code_behavior.csv", index=False)
    clean_triplets.to_csv(OUT / "clean_code_triplet_consistency.csv", index=False)

    arrays = np.load(STATES_PATH)
    geometry = geometry_rows(arrays, manifest)
    geometry.to_csv(OUT / "layer_full_state_geometry.csv", index=False)
    geometry_band = (
        geometry[geometry["depth_percent"].between(low, high)]
        .groupby(
            ["method", "position", "triplet_id", "family_id"],
            as_index=False,
        )["physics_minus_lexical_margin"]
        .mean()
    )
    geometry_band.to_csv(OUT / "late_full_state_geometry.csv", index=False)
    geometry_stats = {}
    for position in ["checkpoint", "final_prompt"]:
        subset = geometry_band[geometry_band["position"] == position]
        geometry_stats[position] = {
            method: shared.two_stage_bootstrap(
                subset[subset["method"] == method],
                "physics_minus_lexical_margin",
                n_resamples=30000,
                seed=20261341
                + 100 * ["checkpoint", "final_prompt"].index(position)
                + index,
            )
            for index, method in enumerate(
                ["raw_residual", "direct", "jacobian_ensemble"]
            )
        }

    family_endpoints = {}
    for label, frame, value in [
        ("checkpoint_physics", checkpoint_physics, "physics_separation"),
        ("final_code", final_code, "code_separation"),
        (
            "code_transition",
            transition,
            "final_minus_checkpoint_code_separation",
        ),
    ]:
        family_endpoints[label] = {
            method: {
                family: float(group[value].mean())
                for family, group in frame[
                    frame["method"] == method
                ].groupby("family_id")
            }
            for method in ["jacobian_ensemble", "direct"]
        }

    statistics = {
        "study_id": protocol["study_id"],
        "status": protocol["status"],
        "provenance": {
            "protocol_sha256": shared.sha256(PROTOCOL_PATH),
            "original_protocol_sha256": shared.sha256(OUT / "protocol.json"),
            "manifest_sha256": shared.sha256(MANIFEST_PATH),
            "raw_sha256": shared.sha256(RAW_PATH),
            "representations_sha256": shared.sha256(STATES_PATH),
            "runner_sha256": shared.sha256(
                ROOT / protocol["inputs"]["runner"]
            ),
            "source_statistics_sha256": shared.sha256(
                ROOT / protocol["inputs"]["source_statistics"]
            ),
        },
        "execution_amendment": protocol["amendment"],
        "behavior": behavior,
        "registered_band_percent": [low, high],
        "checkpoint_physics_bootstrap": checkpoint_stats,
        "final_code_bootstrap": code_stats,
        "code_binding_transition_bootstrap": transition_stats,
        "jacobian_minus_direct": {
            "checkpoint_physics": checkpoint_contrast,
            "final_code": code_contrast,
            "code_binding_transition": transition_contrast,
        },
        "breadth": {
            "checkpoint_physics": checkpoint_breadth,
            "final_code": code_breadth,
            "code_binding_transition": transition_breadth,
        },
        "family_endpoints": family_endpoints,
        "full_state_geometry": geometry_stats,
        "decision": decision,
        "guardrails": protocol["guardrails"],
    }
    shared.dump_json(OUT / "statistics.json", statistics)
    make_figure(
        endpoints,
        checkpoint_physics,
        final_code,
        geometry,
        clean,
        (low, high),
    )

    lines = [
        "# Arbitrary answer-code binding falsification",
        "",
        "## Frozen question",
        "",
        protocol["scientific_question"],
        "",
        "## Decision",
        "",
        (
            f"Frozen success rule: **{'PASS' if decision['frozen_success_rule_pass'] else 'FAIL'}**. "
            "The result does not support the full registered two-stage binding claim."
        ),
        "",
        "## Scientific relation before the mapping",
        "",
        (
            f"The Jacobian checkpoint physics separation was "
            f"**{j_checkpoint['mean']:+.4f}** (two-stage 95% CI "
            f"{j_checkpoint['ci95'][0]:+.4f} to {j_checkpoint['ci95'][1]:+.4f}); "
            f"{checkpoint_breadth['positive_triplets']}/24 triplets and "
            f"{checkpoint_breadth['positive_families']}/6 families were positive. "
            f"Direct decoding was {checkpoint_stats['direct']['mean']:+.4f} "
            f"({checkpoint_stats['direct']['ci95'][0]:+.4f} to "
            f"{checkpoint_stats['direct']['ci95'][1]:+.4f})."
        ),
        "",
        "## Arbitrary code after the mapping",
        "",
        (
            f"The Jacobian final-position A/B separation was "
            f"**{j_code['mean']:+.4f}** (95% CI "
            f"{j_code['ci95'][0]:+.4f} to {j_code['ci95'][1]:+.4f}); "
            f"{code_breadth['positive_triplets']}/24 triplets and "
            f"{code_breadth['positive_families']}/6 families were positive. "
            f"The final-minus-checkpoint transition was "
            f"{j_transition['mean']:+.4f} ({j_transition['ci95'][0]:+.4f} to "
            f"{j_transition['ci95'][1]:+.4f})."
        ),
        "",
        "## Behavioral manipulation check",
        "",
        (
            f"Forced-pair A/B accuracy was {behavior['registered_code_pair_accuracy']:.1%}; "
            f"only {behavior['triplets_all_three_code_correct']}/24 triplets "
            "produced the complete registered ABA pattern. A or B was the global "
            f"top next token for {behavior['global_top_token_is_A_or_B_rate']:.1%} "
            "of prompts. This weak compliance is a major limitation of the "
            "falsification, not a detail to omit."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The experiment cleanly breaks the usual alignment between physical "
            "answer and response token, but Gemma did not reliably obey the "
            "arbitrary code. Therefore the null/heterogeneous endpoint cannot "
            "distinguish absence of staged binding from failure of the behavioral "
            "manipulation. The result is useful as a documented negative control "
            "and argues against claiming that the replicated late transition is "
            "already proven to be a physics-first computation."
        ),
        "",
        "## Complete record",
        "",
        "- `PROTOCOL.md`, `protocol.json`, and `protocol-amendment-v1.json`.",
        "- `execution-attempt-1-pre-forward.json`: retained tokenization abort.",
        "- `prompt_manifest.json`: all 72 exact prompts and code mappings.",
        "- `raw.json` and `representations.npz`: all outputs and states.",
        "- `layer_readout_separations.csv`: every registered layerwise contrast.",
        "- `checkpoint_physics_triplet_separation.csv`.",
        "- `final_code_triplet_separation.csv`.",
        "- `code_binding_transition_triplet.csv`.",
        "- `layer_full_state_geometry.csv` and `late_full_state_geometry.csv`.",
        "- `clean_code_behavior.csv` and `clean_code_triplet_consistency.csv`.",
        "- `statistics.json`: complete inference and decision record.",
        "- `figures/answer-code-binding.{png,pdf}`.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    (OUT / "README.md").write_text(
        "\n".join([
            "# Arbitrary answer-code binding falsification",
            "",
            "Prospectively frozen negative control reported in the Supplementary Information.",
            "",
            "```bash",
            (
                "python scripts/run_answer_code_binding.py --device mps "
                "--dtype bfloat16 --chunk-size 12"
            ),
            "python scripts/analyze_answer_code_binding.py",
            "```",
            "",
            (
                "Read `PROTOCOL.md`, the amendment and pre-forward failure "
                "record, then `RESULTS.md`. Every prompt and independent unit "
                "is retained."
            ),
            "",
        ]) + "\n"
    )
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
