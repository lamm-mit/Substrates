#!/usr/bin/env python3
"""Analyze frozen option-free cross-mechanism activation patching."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "experiments"
    / "cross-mechanism-activation-patching-2026-07-18"
)
RAW = OUT / "raw.json"
PROTOCOL = OUT / "protocol.json"
FIG = OUT / "figures"
SEED = 20260718


FAMILY_SHORT = {
    "obstacle-spacing-orowan": "Orowan",
    "porosity-modulus": "Porosity",
    "pearlite-spacing-strength": "Pearlite",
    "dislocation-density-strength": "Dislocation",
    "particle-fraction-modulus": "Particles",
    "crosslink-density-modulus": "Crosslinks",
}


def exact_sign_flip(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = (signs * values[None, :]).mean(axis=1)
    p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return {
        "mean": observed,
        "exact_two_sided_p": p,
        "n_units": int(len(values)),
        "n_assignments": int(len(null)),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
    }


def pair_bootstrap(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 30000,
) -> list[float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    draws = values[indices].mean(axis=1)
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def summarize_subset(
    pair_frame: pd.DataFrame,
    mask: pd.Series,
    column: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = pair_frame.loc[mask, column].to_numpy(dtype=float)
    result = exact_sign_flip(values)
    result["bootstrap_95"] = pair_bootstrap(values, rng)
    result["positive_units"] = int(np.sum(values > 0))
    return result


def structured_donor_label_nulls(
    patch: pd.DataFrame,
    observed: dict[str, float],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """Enumerate all balanced donor-outcome assignments within six families."""

    donor_mean = (
        patch.groupby(
            ["donor_family", "donor_prompt_id", "receiver_family"],
            as_index=False,
        )["patched_positive_minus_negative"]
        .mean()
    )
    pair_info = patch[
        [
            "donor_family",
            "receiver_family",
            "cross_vocabulary",
            "opposite_response_orientation",
        ]
    ].drop_duplicates()
    families = sorted(donor_mean["donor_family"].unique())
    if len(families) != 6:
        raise RuntimeError("structured null requires six donor families")
    assignments = list(itertools.combinations(range(4), 2))
    assignment_grid = np.asarray(
        list(itertools.product(range(len(assignments)), repeat=len(families))),
        dtype=np.int8,
    )
    subset_names = (
        "all",
        "cross_vocabulary",
        "opposite_orientation",
        "both",
    )
    family_sums = {
        name: np.zeros((len(families), len(assignments)), dtype=float)
        for name in subset_names
    }
    denominators = {name: 0 for name in subset_names}
    observed_assignment_indices: list[int] = []

    for family_index, donor_family in enumerate(families):
        donor_ids = sorted(
            donor_mean.loc[
                donor_mean["donor_family"] == donor_family,
                "donor_prompt_id",
            ].unique()
        )
        if len(donor_ids) != 4:
            raise RuntimeError(
                f"expected four donor cases for {donor_family}"
            )
        observed_signs = (
            patch.loc[
                patch["donor_family"] == donor_family,
                ["donor_prompt_id", "donor_outcome_sign"],
            ]
            .drop_duplicates()
            .set_index("donor_prompt_id")
            .loc[donor_ids, "donor_outcome_sign"]
            .to_numpy()
        )
        observed_positive = tuple(np.flatnonzero(observed_signs > 0))
        if observed_positive not in assignments:
            raise RuntimeError(
                f"unbalanced observed outcomes for {donor_family}"
            )
        observed_assignment_indices.append(
            assignments.index(observed_positive)
        )
        values = (
            donor_mean[donor_mean["donor_family"] == donor_family]
            .pivot(
                index="receiver_family",
                columns="donor_prompt_id",
                values="patched_positive_minus_negative",
            )
            .loc[:, donor_ids]
        )
        info = pair_info[
            pair_info["donor_family"] == donor_family
        ].set_index("receiver_family")
        for assignment_index, positive_indices in enumerate(assignments):
            sign = -np.ones(4, dtype=float)
            sign[list(positive_indices)] = 1.0
            effects = 2.0 * np.mean(values.to_numpy() * sign[None, :], axis=1)
            effect_by_receiver = pd.Series(effects, index=values.index)
            masks = {
                "all": np.ones(len(values), dtype=bool),
                "cross_vocabulary": info.loc[
                    values.index, "cross_vocabulary"
                ].astype(bool).to_numpy(),
                "opposite_orientation": info.loc[
                    values.index, "opposite_response_orientation"
                ].astype(bool).to_numpy(),
            }
            masks["both"] = (
                masks["cross_vocabulary"]
                & masks["opposite_orientation"]
            )
            for name, mask in masks.items():
                family_sums[name][family_index, assignment_index] = float(
                    effect_by_receiver.to_numpy()[mask].sum()
                )
        masks_for_count = {
            "all": np.ones(len(values), dtype=bool),
            "cross_vocabulary": info.loc[
                values.index, "cross_vocabulary"
            ].astype(bool).to_numpy(),
            "opposite_orientation": info.loc[
                values.index, "opposite_response_orientation"
            ].astype(bool).to_numpy(),
        }
        masks_for_count["both"] = (
            masks_for_count["cross_vocabulary"]
            & masks_for_count["opposite_orientation"]
        )
        for name, mask in masks_for_count.items():
            denominators[name] += int(mask.sum())

    null_arrays: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {}
    observed_rows = np.flatnonzero(
        np.all(
            assignment_grid
            == np.asarray(observed_assignment_indices, dtype=np.int8)[
                None, :
            ],
            axis=1,
        )
    )
    if len(observed_rows) != 1:
        raise RuntimeError("observed structured-null assignment is not unique")
    observed_row = int(observed_rows[0])
    for name in subset_names:
        null = np.zeros(len(assignment_grid), dtype=float)
        for family_index in range(len(families)):
            null += family_sums[name][
                family_index, assignment_grid[:, family_index]
            ]
        null /= denominators[name]
        null_arrays[name] = null
        observed_value = float(observed[name])
        if not np.isclose(
            null[observed_row], observed_value, atol=1e-10, rtol=1e-10
        ):
            raise RuntimeError(
                f"structured-null observed assignment mismatch for {name}: "
                f"{null[observed_row]} != {observed_value}"
            )
        results[name] = {
            "observed": observed_value,
            "exact_two_sided_p": float(
                np.mean(np.abs(null) >= abs(observed_value) - 1e-12)
            ),
            "n_assignments": int(len(null)),
            "null_mean": float(null.mean()),
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "n_ordered_pairs": int(denominators[name]),
            "observed_assignment_row": observed_row,
        }
    null_arrays["assignment_indices"] = assignment_grid
    null_arrays["family_order"] = np.asarray(families)
    null_arrays["observed_assignment_indices"] = np.asarray(
        observed_assignment_indices, dtype=np.int8
    )
    return results, null_arrays


def build_tables(
    patch: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = [
        "receiver_prompt_id",
        "receiver_family",
        "donor_family",
        "layer",
        "depth_percent",
        "cross_vocabulary",
        "opposite_response_orientation",
    ]
    receiver_layer = (
        patch.groupby(keys + ["donor_outcome_sign"], as_index=False)[
            "patched_positive_minus_negative"
        ]
        .mean()
        .pivot(
            index=keys,
            columns="donor_outcome_sign",
            values="patched_positive_minus_negative",
        )
        .reset_index()
        .rename_axis(columns=None)
        .rename(columns={-1.0: "negative_outcome", 1.0: "positive_outcome"})
    )
    receiver_layer["physical_outcome_contrast"] = (
        receiver_layer["positive_outcome"]
        - receiver_layer["negative_outcome"]
    )

    numeric = (
        patch.groupby(keys + ["donor_numeric_sign"], as_index=False)[
            "patched_positive_minus_negative"
        ]
        .mean()
        .pivot(
            index=keys,
            columns="donor_numeric_sign",
            values="patched_positive_minus_negative",
        )
        .reset_index()
        .rename_axis(columns=None)
        .rename(columns={-1.0: "numeric_decrease", 1.0: "numeric_increase"})
    )
    numeric["numeric_direction_contrast"] = (
        numeric["numeric_increase"] - numeric["numeric_decrease"]
    )
    receiver_layer = receiver_layer.merge(
        numeric[keys + ["numeric_direction_contrast"]],
        on=keys,
        validate="one_to_one",
    )

    ordered = (
        receiver_layer.groupby(
            [
                "donor_family",
                "receiver_family",
                "cross_vocabulary",
                "opposite_response_orientation",
            ],
            as_index=False,
        )
        .agg(
            physical_outcome_contrast=("physical_outcome_contrast", "mean"),
            numeric_direction_contrast=("numeric_direction_contrast", "mean"),
            n_receiver_layers=("physical_outcome_contrast", "size"),
        )
    )
    ordered["family_pair"] = ordered.apply(
        lambda row: " | ".join(
            sorted([row["donor_family"], row["receiver_family"]])
        ),
        axis=1,
    )
    unordered = (
        ordered.groupby(
            [
                "family_pair",
                "cross_vocabulary",
                "opposite_response_orientation",
            ],
            as_index=False,
        )
        .agg(
            physical_outcome_contrast=("physical_outcome_contrast", "mean"),
            numeric_direction_contrast=("numeric_direction_contrast", "mean"),
            n_directions=("donor_family", "size"),
        )
    )
    return receiver_layer, ordered, unordered


def make_figure(
    unordered: pd.DataFrame,
    ordered: pd.DataFrame,
    receiver_layer: pd.DataFrame,
    subset_summary: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.1,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    teal = "#176B7A"
    orange = "#D17C38"
    purple = "#7B6BA8"
    gray = "#8B8E91"
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.6))

    ordered_pairs = unordered.sort_values(
        "physical_outcome_contrast", ascending=True
    )
    y = np.arange(len(ordered_pairs))
    axes[0, 0].hlines(
        y,
        ordered_pairs["numeric_direction_contrast"],
        ordered_pairs["physical_outcome_contrast"],
        color="#C9CBCC",
        linewidth=0.9,
    )
    axes[0, 0].scatter(
        ordered_pairs["physical_outcome_contrast"],
        y,
        color=teal,
        s=23,
        label="physical outcome",
        zorder=3,
    )
    axes[0, 0].scatter(
        ordered_pairs["numeric_direction_contrast"],
        y,
        color=orange,
        s=23,
        label="numerical direction",
        zorder=3,
    )
    labels = []
    for pair in ordered_pairs["family_pair"]:
        left, right = pair.split(" | ")
        labels.append(f"{FAMILY_SHORT[left]}–{FAMILY_SHORT[right]}")
    axes[0, 0].set_yticks(y, labels)
    axes[0, 0].axvline(0, color=gray, linestyle="--", linewidth=0.8)
    axes[0, 0].set_xlabel("patched answer-margin contrast")
    axes[0, 0].legend(frameon=False, loc="lower right")

    display_names = {
        "all": "All pairs",
        "cross_vocabulary": "Different answer words",
        "opposite_orientation": "Reversed numeric trend",
        "both": "Both controls",
    }
    subset_order = [
        "all",
        "cross_vocabulary",
        "opposite_orientation",
        "both",
    ]
    summary = subset_summary.set_index("subset").loc[subset_order]
    x = np.arange(len(summary))
    means = summary["mean"].to_numpy()
    lower = means - summary["ci_low"].to_numpy()
    upper = summary["ci_high"].to_numpy() - means
    axes[0, 1].errorbar(
        x,
        means,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        color=teal,
        ecolor=teal,
        capsize=3,
        markersize=5,
    )
    axes[0, 1].axhline(0, color=gray, linestyle="--", linewidth=0.8)
    axes[0, 1].set_xticks(
        x,
        [display_names[name] for name in subset_order],
        rotation=18,
        ha="right",
    )
    axes[0, 1].set_ylabel("physical-outcome transfer")

    layer_curve = (
        receiver_layer.groupby(["layer", "depth_percent"], as_index=False)
        .agg(
            physical=("physical_outcome_contrast", "mean"),
            numeric=("numeric_direction_contrast", "mean"),
        )
        .sort_values("depth_percent")
    )
    axes[1, 0].plot(
        layer_curve["depth_percent"],
        layer_curve["physical"],
        color=teal,
        marker="o",
        linewidth=1.5,
        label="physical outcome",
    )
    axes[1, 0].plot(
        layer_curve["depth_percent"],
        layer_curve["numeric"],
        color=orange,
        marker="s",
        linewidth=1.5,
        label="numerical direction",
    )
    axes[1, 0].axhline(0, color=gray, linestyle="--", linewidth=0.8)
    axes[1, 0].set_xlabel("layer depth (%)")
    axes[1, 0].set_ylabel("patched answer-margin contrast")
    axes[1, 0].legend(frameon=False, loc="best")

    family = (
        ordered.groupby("donor_family", as_index=False)
        .agg(
            physical=("physical_outcome_contrast", "mean"),
            numeric=("numeric_direction_contrast", "mean"),
        )
        .sort_values("physical")
    )
    y_family = np.arange(len(family))
    orientation = {
        "obstacle-spacing-orowan": "inverse",
        "porosity-modulus": "inverse",
        "pearlite-spacing-strength": "inverse",
        "dislocation-density-strength": "direct",
        "particle-fraction-modulus": "direct",
        "crosslink-density-modulus": "direct",
    }
    family_colors = [
        purple if orientation[row] == "direct" else "#5B8C5A"
        for row in family["donor_family"]
    ]
    axes[1, 1].barh(
        y_family,
        family["physical"],
        color=family_colors,
        alpha=0.9,
    )
    axes[1, 1].axvline(0, color=gray, linestyle="--", linewidth=0.8)
    axes[1, 1].set_yticks(
        y_family,
        [FAMILY_SHORT[row] for row in family["donor_family"]],
    )
    axes[1, 1].set_xlabel("physical-outcome transfer")
    axes[1, 1].legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                color=purple,
                label="property rises with input",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                color="#5B8C5A",
                label="property falls with input",
            ),
        ],
        frameon=False,
        loc="lower right",
    )

    for label, axis in zip("ABCD", axes.flat):
        axis.text(
            -0.17,
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    fig.subplots_adjust(
        left=0.19,
        right=0.985,
        bottom=0.12,
        top=0.97,
        wspace=0.37,
        hspace=0.43,
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(
            FIG / f"cross-mechanism-activation-patching.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    raw = json.loads(RAW.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    patch = pd.DataFrame(raw["patch_rows"])
    if len(patch) != protocol["prompt_design"]["n_patches"]:
        raise RuntimeError(f"unexpected patch row count: {len(patch)}")
    if patch[
        ["receiver_prompt_id", "donor_prompt_id", "layer"]
    ].duplicated().any():
        raise RuntimeError("duplicate patch rows")
    if not np.isfinite(
        patch.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise RuntimeError("non-finite patch result")
    if sorted(patch["layer"].unique()) != protocol["source_layers"]:
        raise RuntimeError("patch layers differ from frozen protocol")

    receiver_layer, ordered, unordered = build_tables(patch)
    if len(ordered) != 30 or len(unordered) != 15:
        raise RuntimeError(
            f"unexpected mechanism-pair cardinality: {len(ordered)}, "
            f"{len(unordered)}"
        )
    if not np.all(unordered["n_directions"] == 2):
        raise RuntimeError("an unordered pair lacks both patching directions")

    rng = np.random.default_rng(SEED)
    masks = {
        "all": pd.Series(True, index=unordered.index),
        "cross_vocabulary": unordered["cross_vocabulary"].astype(bool),
        "opposite_orientation": unordered[
            "opposite_response_orientation"
        ].astype(bool),
        "both": (
            unordered["cross_vocabulary"].astype(bool)
            & unordered["opposite_response_orientation"].astype(bool)
        ),
    }
    summaries: dict[str, dict[str, Any]] = {}
    subset_rows = []
    for name, mask in masks.items():
        result = summarize_subset(
            unordered,
            mask,
            "physical_outcome_contrast",
            rng,
        )
        numeric = summarize_subset(
            unordered,
            mask,
            "numeric_direction_contrast",
            rng,
        )
        result["numeric_direction_control"] = numeric
        summaries[name] = result
        subset_rows.append(
            {
                "subset": name,
                "mean": result["mean"],
                "ci_low": result["bootstrap_95"][0],
                "ci_high": result["bootstrap_95"][1],
                "exact_two_sided_p": result["exact_two_sided_p"],
                "positive_pairs": result["positive_units"],
                "n_pairs": result["n_units"],
                "numeric_mean": numeric["mean"],
                "numeric_exact_two_sided_p": numeric[
                    "exact_two_sided_p"
                ],
            }
        )
    subset_summary = pd.DataFrame(subset_rows)
    structured_nulls, structured_null_arrays = structured_donor_label_nulls(
        patch,
        {name: summaries[name]["mean"] for name in summaries},
    )
    for row in subset_rows:
        row["structured_exact_two_sided_p"] = structured_nulls[
            row["subset"]
        ]["exact_two_sided_p"]
    subset_summary = pd.DataFrame(subset_rows)

    donor_family = (
        ordered.groupby("donor_family", as_index=False)
        .agg(
            physical_outcome_contrast=("physical_outcome_contrast", "mean"),
            numeric_direction_contrast=("numeric_direction_contrast", "mean"),
            positive_ordered_pairs=(
                "physical_outcome_contrast",
                lambda values: int(np.sum(np.asarray(values) > 0)),
            ),
            n_ordered_pairs=("receiver_family", "size"),
        )
    )
    breadth = int(np.sum(donor_family["physical_outcome_contrast"] > 0))
    gates = {
        "overall": bool(
            summaries["all"]["mean"] > 0
            and summaries["all"]["exact_two_sided_p"] <= 0.05
            and structured_nulls["all"]["exact_two_sided_p"] <= 0.05
        ),
        "cross_vocabulary": bool(
            summaries["cross_vocabulary"]["mean"] > 0
            and summaries["cross_vocabulary"]["exact_two_sided_p"] <= 0.05
            and structured_nulls["cross_vocabulary"][
                "exact_two_sided_p"
            ]
            <= 0.05
        ),
        "opposite_orientation": bool(
            summaries["opposite_orientation"]["mean"] > 0
            and summaries["opposite_orientation"]["exact_two_sided_p"]
            <= 0.05
            and structured_nulls["opposite_orientation"][
                "exact_two_sided_p"
            ]
            <= 0.05
        ),
        "breadth": breadth >= 5,
    }
    if all(gates.values()):
        verdict = "strong"
    elif gates["overall"] and (
        gates["cross_vocabulary"] or gates["opposite_orientation"]
    ):
        verdict = "partial"
    else:
        verdict = "none"

    clean = pd.DataFrame(raw["clean_receivers"])
    family_positive = {
        row["family_id"]: row["outcome_positive"]
        for row in json.loads(
            (
                ROOT
                / protocol["inputs"]["prompt_manifest"]["path"]
            ).read_text()
        )["families"]
    }
    clean["expected_sign"] = clean.apply(
        lambda row: (
            1
            if row["expected_outcome"]
            == family_positive[row["family_id"]]
            else -1
        ),
        axis=1,
    )
    clean["forced_pair_correct"] = (
        clean["expected_sign"] * clean["positive_minus_negative"] > 0
    )

    receiver_layer.to_csv(OUT / "receiver_layer_contrasts.csv", index=False)
    ordered.to_csv(OUT / "ordered_mechanism_pair_effects.csv", index=False)
    unordered.to_csv(OUT / "unordered_mechanism_pair_effects.csv", index=False)
    subset_summary.to_csv(OUT / "subset_statistics.csv", index=False)
    donor_family.to_csv(OUT / "donor_family_effects.csv", index=False)
    patch.to_csv(OUT / "all_patch_rows.csv", index=False)
    clean.to_csv(OUT / "clean_receiver_behavior.csv", index=False)
    np.savez_compressed(
        OUT / "primary_exact_donor_label_nulls.npz",
        **structured_null_arrays,
    )

    statistics = {
        "study_id": protocol["study_id"],
        "protocol_sha256": raw["protocol_sha256"],
        "runner_sha256": raw["runner_sha256"],
        "frozen_verdict": {
            "option_free_cross_mechanism_transfer": verdict,
            "gates": gates,
            "positive_donor_families": breadth,
            "required_positive_donor_families": 5,
        },
        "primary": summaries,
        "structured_donor_label_nulls": structured_nulls,
        "clean_forced_pair_accuracy": float(
            clean["forced_pair_correct"].mean()
        ),
        "donor_family_results": donor_family.to_dict(orient="records"),
        "dimensions": {
            "patch_rows": int(len(patch)),
            "receiver_prompts": int(patch["receiver_prompt_id"].nunique()),
            "donor_prompts": int(patch["donor_prompt_id"].nunique()),
            "ordered_mechanism_pairs": int(len(ordered)),
            "unordered_mechanism_pairs": int(len(unordered)),
            "layers": sorted(int(layer) for layer in patch["layer"].unique()),
        },
        "guardrail": (
            "Cross-vocabulary option-free state transfer rules out copying "
            "one particular answer token but remains compatible with a "
            "general answer-polarity or decision-state representation. "
            "This is causal sufficiency in a constrained forced-pair readout."
        ),
    }
    (OUT / "statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n"
    )
    (OUT / "RESULTS.md").write_text(
        "\n".join(
            [
                "# Option-free cross-mechanism activation patching",
                "",
                f"Frozen verdict: **{verdict.upper()}** evidence.",
                "",
                "## Primary physical-outcome transfer",
                "",
                (
                    f"- All pairs: {summaries['all']['mean']:+.3f}, "
                    f"pair-sign exact p="
                    f"{summaries['all']['exact_two_sided_p']:.6g}, "
                    f"structured donor-label exact p="
                    f"{structured_nulls['all']['exact_two_sided_p']:.6g}, "
                    f"95% pair-bootstrap interval "
                    f"[{summaries['all']['bootstrap_95'][0]:+.3f}, "
                    f"{summaries['all']['bootstrap_95'][1]:+.3f}]."
                ),
                (
                    f"- Across answer vocabularies: "
                    f"{summaries['cross_vocabulary']['mean']:+.3f}, "
                    f"pair-sign p="
                    f"{summaries['cross_vocabulary']['exact_two_sided_p']:.6g}, "
                    f"structured p="
                    f"{structured_nulls['cross_vocabulary']['exact_two_sided_p']:.6g}."
                ),
                (
                    f"- Across reversed numerical response orientations: "
                    f"{summaries['opposite_orientation']['mean']:+.3f}, "
                    f"pair-sign p="
                    f"{summaries['opposite_orientation']['exact_two_sided_p']:.6g}, "
                    f"structured p="
                    f"{structured_nulls['opposite_orientation']['exact_two_sided_p']:.6g}."
                ),
                (
                    f"- Both controls simultaneously: "
                    f"{summaries['both']['mean']:+.3f}, "
                    f"p={summaries['both']['exact_two_sided_p']:.6g}."
                ),
                (
                    f"- Donor-family breadth: {breadth}/6 positive; "
                    f"clean forced-pair accuracy "
                    f"{clean['forced_pair_correct'].mean():.1%}."
                ),
                "",
                "## Meaning",
                "",
                (
                    "A positive cross-vocabulary effect cannot be simple "
                    "copying of one answer token because donors and receivers "
                    "use different answer words. It can still reflect a "
                    "general positive-versus-negative answer decision rather "
                    "than a mechanism-specific constitutive law."
                ),
                "",
                "## Reproduction",
                "",
                "```bash",
                (
                    "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                    "python scripts/run_cross_mechanism_activation_patching.py "
                    "--device cpu --dtype bfloat16 "
                    "--local-model-snapshot /path/to/"
                    "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
                ),
                "python scripts/analyze_cross_mechanism_activation_patching.py",
                "```",
                "",
            ]
        )
    )
    make_figure(unordered, ordered, receiver_layer, subset_summary)
    print(json.dumps(statistics["frozen_verdict"], indent=2))


if __name__ == "__main__":
    main()
