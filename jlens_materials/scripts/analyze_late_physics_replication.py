#!/usr/bin/env python3
"""Analyze the frozen disjoint late-physics representation replication."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import analyze_lexical_adversarial_representation as shared

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "late-physics-representation-replication-2026-07-17"
DISCOVERY = (
    ROOT / "experiments" / "lexical-adversarial-representation-2026-07-17"
)
PROTOCOL_PATH = OUT / "protocol.json"
MANIFEST_PATH = OUT / "prompt_manifest.json"
RAW_PATH = OUT / "raw.json"
STATES_PATH = OUT / "representations.npz"
FIG = OUT / "figures"


def aggregate_band(
    frame: pd.DataFrame,
    low: float,
    high: float,
) -> pd.DataFrame:
    return (
        frame[frame["depth_percent"].between(low, high)]
        .groupby(["method", "triplet_id", "family_id"], as_index=False)[
            "physics_minus_lexical_margin"
        ]
        .mean()
    )


def bootstrap_methods(
    frame: pd.DataFrame,
    methods: list[str],
    seed: int,
) -> dict:
    return {
        method: shared.two_stage_bootstrap(
            frame[frame["method"] == method],
            "physics_minus_lexical_margin",
            n_resamples=30000,
            seed=seed + method_index,
        )
        for method_index, method in enumerate(methods)
    }


def transition_frame(late: pd.DataFrame, middle: pd.DataFrame) -> pd.DataFrame:
    merged = late.merge(
        middle,
        on=["method", "triplet_id", "family_id"],
        suffixes=("_late", "_middle"),
        validate="one_to_one",
    )
    merged["late_minus_middle"] = (
        merged["physics_minus_lexical_margin_late"]
        - merged["physics_minus_lexical_margin_middle"]
    )
    return merged


def make_figure(
    layer_frame: pd.DataFrame,
    primary_family: pd.DataFrame,
    discovery_late: pd.DataFrame,
    replication_late: pd.DataFrame,
    target_free: pd.DataFrame,
    late_band: tuple[float, float],
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
    colors = {"jacobian_ensemble": "#167D8D", "direct": "#6F5AA8"}
    labels = {"jacobian_ensemble": "Jacobian", "direct": "direct"}
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    layer_summary = shared.layer_summary_with_ci(
        layer_frame,
        "physics_minus_lexical_margin",
        ["jacobian_ensemble", "direct"],
        n_resamples=5000,
    )
    for method in ["jacobian_ensemble", "direct"]:
        subset = layer_summary[layer_summary["method"] == method].sort_values("layer")
        x = subset["depth_percent"].to_numpy()
        mean = subset["mean"].to_numpy()
        low = np.asarray([value[0] for value in subset["ci95"]])
        high = np.asarray([value[1] for value in subset["ci95"]])
        ax_a.plot(x, mean, color=colors[method], linewidth=1.7, label=labels[method])
        ax_a.fill_between(x, low, high, color=colors[method], alpha=0.15, linewidth=0)
    ax_a.axhline(0, color="#666666", linewidth=0.8)
    ax_a.axvspan(late_band[0], late_band[1], color="#167D8D", alpha=0.08, linewidth=0)
    ax_a.set_xlabel("Layer depth (%)")
    ax_a.set_ylabel("Physics-equivalence margin\n(centered cosine units)")
    ax_a.text(0.02, 0.97, "A", transform=ax_a.transAxes, va="top", fontweight="bold")

    cohort_rows = []
    for cohort, frame in [
        ("discovery", discovery_late),
        ("disjoint replication", replication_late),
    ]:
        for method in ["jacobian_ensemble", "direct"]:
            result = shared.two_stage_bootstrap(
                frame[frame["method"] == method],
                "physics_minus_lexical_margin",
                n_resamples=30000,
                seed=20260730 + 10 * (cohort == "disjoint replication")
                + (method == "direct"),
            )
            cohort_rows.append({"cohort": cohort, "method": method, **result})
    cohort_frame = pd.DataFrame(cohort_rows)
    x = np.arange(2)
    for method, offset, marker in [
        ("jacobian_ensemble", -0.12, "o"),
        ("direct", 0.12, "^"),
    ]:
        subset = cohort_frame[cohort_frame["method"] == method].set_index("cohort").loc[
            ["discovery", "disjoint replication"]
        ]
        means = subset["mean"].to_numpy()
        low = np.asarray([value[0] for value in subset["ci95"]])
        high = np.asarray([value[1] for value in subset["ci95"]])
        ax_b.errorbar(
            x + offset,
            means,
            yerr=[means - low, high - means],
            fmt=marker,
            color=colors[method],
            capsize=3,
            markersize=7,
            label=labels[method],
        )
    ax_b.axhline(0, color="#666666", linewidth=0.8)
    ax_b.set_xticks(x, ["discovery", "disjoint\nreplication"])
    ax_b.set_ylabel("Late-window physics-equivalence margin")
    ax_b.text(0.02, 0.97, "B", transform=ax_b.transAxes, va="top", fontweight="bold")

    family_order = list(dict.fromkeys(primary_family["family_id"]))
    short = {
        "obstacle-spacing-orowan": "obstacle\nspacing",
        "porosity-modulus": "porosity",
        "pearlite-spacing-strength": "pearlite\nspacing",
        "dislocation-density-strength": "dislocation\ndensity",
        "particle-fraction-modulus": "particle\nfraction",
        "crosslink-density-modulus": "crosslink\ndensity",
    }
    x = np.arange(len(family_order))
    for method, offset, marker in [
        ("jacobian_ensemble", -0.14, "o"),
        ("direct", 0.14, "^"),
    ]:
        subset = primary_family[
            primary_family["method"] == method
        ].set_index("family_id").loc[family_order]
        ax_c.scatter(
            x + offset,
            subset["physics_minus_lexical_margin"],
            color=colors[method],
            marker=marker,
            s=38,
        )
    ax_c.axhline(0, color="#666666", linewidth=0.8)
    ax_c.set_xticks(x, [short[value] for value in family_order])
    ax_c.set_ylabel("Replication late-window margin")
    ax_c.text(0.02, 0.97, "C", transform=ax_c.transAxes, va="top", fontweight="bold")

    target_summary = shared.layer_summary_with_ci(
        target_free.rename(columns={"physics_minus_lexical_jaccard": "margin"}),
        "margin",
        ["jacobian_ensemble", "direct"],
        n_resamples=5000,
    )
    for method, marker in [
        ("jacobian_ensemble", "o"),
        ("direct", "^"),
    ]:
        subset = target_summary[target_summary["method"] == method].sort_values("layer")
        x_values = subset["depth_percent"].to_numpy()
        means = subset["mean"].to_numpy()
        low = np.asarray([value[0] for value in subset["ci95"]])
        high = np.asarray([value[1] for value in subset["ci95"]])
        ax_d.errorbar(
            x_values,
            means,
            yerr=[means - low, high - means],
            color=colors[method],
            marker=marker,
            linewidth=1.4,
            capsize=2,
            label=labels[method],
        )
    ax_d.axhline(0, color="#666666", linewidth=0.8)
    ax_d.axvspan(late_band[0], late_band[1], color="#167D8D", alpha=0.08, linewidth=0)
    ax_d.set_xlabel("Layer depth (%)")
    ax_d.set_ylabel("Target-free word-set margin\n(Jaccard units)")
    ax_d.text(0.02, 0.97, "D", transform=ax_d.transAxes, va="top", fontweight="bold")

    handles, legend_labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(
            FIG / f"late-physics-representation-replication.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    raw = json.loads(RAW_PATH.read_text())
    if raw["provenance"]["protocol_sha256"] != shared.sha256(PROTOCOL_PATH):
        raise RuntimeError("raw output does not match frozen replication protocol")
    if shared.sha256(
        DISCOVERY / "statistics.json"
    ) != protocol["inputs"]["discovery_statistics_sha256"]:
        raise RuntimeError("motivating discovery statistics changed after freezing")
    arrays = np.load(STATES_PATH)
    prompt_ids = [str(value) for value in arrays["prompt_ids"]]
    expected = [row["prompt_id"] for row in manifest["prompts"]]
    if prompt_ids != expected:
        raise RuntimeError("state prompt order does not match manifest")
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    layers = arrays["layers"].astype(int)
    jacobian = arrays["jacobian_decoder_basis"].astype(np.float32)
    methods = {
        "raw_residual": arrays["raw_states"].astype(np.float32),
        "direct": arrays["direct_decoder_basis"].astype(np.float32),
        "jacobian_seed0": jacobian[0],
        "jacobian_seed1": jacobian[1],
        "jacobian_seed2": jacobian[2],
        "jacobian_ensemble": jacobian.mean(axis=0),
    }
    centered = shared.similarity_rows(
        methods,
        layers,
        manifest["triplets"],
        prompt_index,
        centered=True,
        accumulator_dtype=np.float64,
    )
    centered.to_csv(OUT / "layer_similarity_margins.csv", index=False)

    late_low, late_high = protocol["registered_band_percent"]
    full_low, full_high = protocol["secondary_full_band_percent"]
    late = aggregate_band(centered, late_low, late_high)
    full = aggregate_band(centered, full_low, full_high)
    middle = aggregate_band(centered, 38.0, 70.0)
    late.to_csv(OUT / "late_window_triplet_margins.csv", index=False)
    full.to_csv(OUT / "full_band_triplet_margins.csv", index=False)
    transition = transition_frame(late, middle)
    transition.to_csv(OUT / "late_minus_middle_triplet_margins.csv", index=False)

    family_late = (
        late.groupby(["method", "family_id"], as_index=False)[
            "physics_minus_lexical_margin"
        ].mean()
    )
    family_late.to_csv(OUT / "family_late_window_margins.csv", index=False)
    method_names = list(methods)
    late_stats = bootstrap_methods(late, method_names, 20260718)
    full_stats = bootstrap_methods(full, method_names, 20260818)
    transition_stats = {
        method: shared.two_stage_bootstrap(
            transition[transition["method"] == method].rename(
                columns={"late_minus_middle": "value"}
            ),
            "value",
            n_resamples=30000,
            seed=20260918 + method_index,
        )
        for method_index, method in enumerate(method_names)
    }

    pivot = late.pivot(
        index=["triplet_id", "family_id"],
        columns="method",
        values="physics_minus_lexical_margin",
    ).reset_index()
    pivot["jacobian_minus_direct"] = pivot["jacobian_ensemble"] - pivot["direct"]
    method_contrast = shared.two_stage_bootstrap(
        pivot,
        "jacobian_minus_direct",
        n_resamples=30000,
        seed=20261018,
    )
    family_contrast = (
        pivot.groupby("family_id")["jacobian_minus_direct"].mean().to_numpy()
    )
    method_contrast["exact_family_sign_flip_p"] = shared.exact_family_sign_flip(
        family_contrast
    )

    clean_frame, behavior = shared.behavior_summary(raw, manifest)
    clean_frame.to_csv(OUT / "clean_behavior.csv", index=False)
    pd.DataFrame(behavior["triplet_rows"]).to_csv(
        OUT / "clean_triplet_consistency.csv", index=False
    )
    target_free, _ = shared.target_free_analysis(raw, manifest, protocol)
    target_free.to_csv(OUT / "target_free_jaccard_margins.csv", index=False)
    target_band = (
        target_free.groupby(["method", "triplet_id", "family_id"], as_index=False)[
            "physics_minus_lexical_jaccard"
        ].mean()
    )
    target_stats = {
        method: shared.two_stage_bootstrap(
            target_band[target_band["method"] == method],
            "physics_minus_lexical_jaccard",
            n_resamples=30000,
            seed=20261118 + method_index,
        )
        for method_index, method in enumerate(["jacobian_ensemble", "direct"])
    }

    primary_rows = late[late["method"] == "jacobian_ensemble"]
    primary_family = family_late[
        family_late["method"] == "jacobian_ensemble"
    ]
    n_positive = int((primary_rows["physics_minus_lexical_margin"] > 0).sum())
    n_family_positive = int(
        (primary_family["physics_minus_lexical_margin"] > 0).sum()
    )
    decision = {
        "primary_ci_above_zero": late_stats["jacobian_ensemble"]["ci95"][0] > 0,
        "positive_triplets": n_positive,
        "triplet_breadth_pass": n_positive >= 18,
        "positive_families": n_family_positive,
        "family_breadth_pass": n_family_positive >= 5,
    }
    decision["frozen_success_rule_pass"] = all([
        decision["primary_ci_above_zero"],
        decision["triplet_breadth_pass"],
        decision["family_breadth_pass"],
    ])

    discovery_layer = pd.read_csv(DISCOVERY / "layer_similarity_margins.csv")
    discovery_centered = discovery_layer[
        discovery_layer["centered"].astype(str).str.lower().isin(["true", "1"])
    ]
    discovery_late = aggregate_band(discovery_centered, late_low, late_high)
    stats_payload = {
        "study_id": protocol["study_id"],
        "status": protocol["status"],
        "dimensions": raw["dimensions"],
        "provenance": {
            "protocol_sha256": shared.sha256(PROTOCOL_PATH),
            "manifest_sha256": shared.sha256(MANIFEST_PATH),
            "raw_sha256": shared.sha256(RAW_PATH),
            "representations_sha256": shared.sha256(STATES_PATH),
            "motivating_statistics_sha256": shared.sha256(
                DISCOVERY / "statistics.json"
            ),
        },
        "lexical_preflight": {
            key: {
                field: value for field, value in result.items() if field != "rows"
            }
            for key, result in protocol["lexical_adversarial_preflight"].items()
        },
        "behavior": {
            key: value for key, value in behavior.items() if key != "triplet_rows"
        },
        "late_window_percent": [late_low, late_high],
        "late_window_bootstrap": late_stats,
        "secondary_full_band_bootstrap": full_stats,
        "late_minus_middle_bootstrap": transition_stats,
        "jacobian_minus_direct_late": method_contrast,
        "primary_breadth": {
            "positive_triplets": n_positive,
            "n_triplets": len(primary_rows),
            "exact_two_sided_sign_p": float(stats.binomtest(
                n_positive,
                len(primary_rows),
                p=0.5,
                alternative="two-sided",
            ).pvalue),
            "positive_families": n_family_positive,
            "n_families": len(primary_family),
        },
        "family_late_window_margins": {
            method: {
                row["family_id"]: float(row["physics_minus_lexical_margin"])
                for _, row in family_late[
                    family_late["method"] == method
                ].iterrows()
            }
            for method in method_names
        },
        "target_free_jaccard_bootstrap": target_stats,
        "decision": decision,
        "guardrails": protocol["guardrails"],
    }
    shared.dump_json(OUT / "statistics.json", stats_payload)
    make_figure(
        centered[centered["method"].isin(["jacobian_ensemble", "direct"])],
        family_late[family_late["method"].isin(["jacobian_ensemble", "direct"])],
        discovery_late,
        late[late["method"].isin(["jacobian_ensemble", "direct"])],
        target_free,
        (late_low, late_high),
    )

    primary = late_stats["jacobian_ensemble"]
    direct = late_stats["direct"]
    transition_j = transition_stats["jacobian_ensemble"]
    target_j = target_stats["jacobian_ensemble"]
    target_d = target_stats["direct"]
    lines = [
        "# Disjoint late-physics representation replication",
        "",
        "## Frozen question",
        "",
        protocol["scientific_question"],
        "",
        "## Prospective primary result",
        "",
        (
            f"The frozen 80--96% Jacobian ensemble margin was "
            f"**{primary['mean']:+.4f}** (two-stage 95% CI "
            f"{primary['ci95'][0]:+.4f} to {primary['ci95'][1]:+.4f}). "
            f"{n_positive}/24 triplets and {n_family_positive}/6 family means "
            f"were positive. Frozen replication decision: "
            f"**{'PASS' if decision['frozen_success_rule_pass'] else 'FAIL'}**."
        ),
        "",
        (
            f"Direct decoding gave {direct['mean']:+.4f} "
            f"({direct['ci95'][0]:+.4f} to {direct['ci95'][1]:+.4f}). "
            f"The paired Jacobian-minus-direct contrast was "
            f"{method_contrast['mean']:+.4f} "
            f"({method_contrast['ci95'][0]:+.4f} to "
            f"{method_contrast['ci95'][1]:+.4f}; exact family sign-flip "
            f"p={method_contrast['exact_family_sign_flip_p']:.4f})."
        ),
        "",
        "## Transition and behavior",
        "",
        (
            f"The Jacobian late-minus-middle contrast was "
            f"{transition_j['mean']:+.4f} "
            f"({transition_j['ci95'][0]:+.4f} to "
            f"{transition_j['ci95'][1]:+.4f}). Registered-pair answer accuracy "
            f"was {behavior['registered_pair_accuracy']:.1%}, and "
            f"{behavior['triplets_full_scientific_consistency']}/24 triplets "
            "were fully scientifically consistent."
        ),
        "",
        "## Target-free vocabulary",
        "",
        (
            f"Late target-free word-set margins were "
            f"{target_j['mean']:+.4f} ({target_j['ci95'][0]:+.4f} to "
            f"{target_j['ci95'][1]:+.4f}) for Jacobian and "
            f"{target_d['mean']:+.4f} ({target_d['ci95'][0]:+.4f} to "
            f"{target_d['ci95'][1]:+.4f}) for direct decoding."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "The late window was selected from the completed discovery cohort "
            "and frozen before this disjoint run. A positive replication would "
            "support a late representational transition under this exact "
            "lexical-adversarial design. It would not establish causal use, a "
            "literal thought process, or unrestricted materials understanding."
        ),
        "",
        "## Files",
        "",
        "- `prompt_manifest.json`: all 72 disjoint exact prompts.",
        "- `protocol.json` and `PROTOCOL.md`: frozen late-window endpoint.",
        "- `raw.json` and `representations.npz`: complete model outputs.",
        "- `layer_similarity_margins.csv`: all layerwise triplet margins.",
        "- `late_window_triplet_margins.csv`: frozen primary independent units.",
        "- `late_minus_middle_triplet_margins.csv`: transition audit.",
        "- `target_free_jaccard_margins.csv`: open-vocabulary comparison.",
        "- `statistics.json`: complete decision record.",
        "- `figures/late-physics-representation-replication.{png,pdf}`.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    (OUT / "README.md").write_text(
        "\n".join([
            "# Disjoint replication of the late physical-equivalence transition",
            "",
            "Prospectively frozen disjoint replication reported in the paper and Supplementary Information.",
            "",
            "```bash",
            (
                "python scripts/run_lexical_adversarial_representation.py "
                "--protocol experiments/late-physics-representation-replication-2026-07-17/protocol.json "
                "--output experiments/late-physics-representation-replication-2026-07-17/raw.json "
                "--states-output experiments/late-physics-representation-replication-2026-07-17/representations.npz "
                "--device mps --dtype bfloat16"
            ),
            "python scripts/analyze_late_physics_replication.py",
            "```",
            "",
        ])
    )
    print(json.dumps(stats_payload, indent=2))


if __name__ == "__main__":
    main()
