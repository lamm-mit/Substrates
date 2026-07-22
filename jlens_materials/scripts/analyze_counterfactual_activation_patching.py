#!/usr/bin/env python3
"""Analyze the frozen candidate counterfactual activation-patching run."""

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
OUT = ROOT / "experiments" / "candidate-activation-patching-2026-07-16"
RAW = OUT / "raw.json"
DISTANCE_RAW = OUT / "distance_controls_raw.json"
PROTOCOL = OUT / "protocol.json"
FIG = OUT / "figures"
SEED = 20260716
N_BOOT = 30_000


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "n_pairs": int(len(values)),
        "positive_pairs": int(np.sum(values > 0)),
    }


def paired_bootstrap(
    left: pd.Series, right: pd.Series, rng: np.random.Generator
) -> dict:
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    return bootstrap_mean((aligned["left"] - aligned["right"]).to_numpy(), rng)


def circular_shift_test(x: np.ndarray, y: np.ndarray) -> dict:
    observed = float(stats.spearmanr(x, y).statistic)
    null = []
    for reverse in (False, True):
        candidate = y[::-1] if reverse else y
        for shift in range(len(y)):
            if not reverse and shift == 0:
                continue
            null.append(float(stats.spearmanr(x, np.roll(candidate, shift)).statistic))
    null_array = np.asarray(null)
    p_value = (1 + np.sum(np.abs(null_array) >= abs(observed))) / (1 + len(null_array))
    return {
        "spearman_rho": observed,
        "circular_shift_two_sided_p": float(p_value),
        "null_transformations": int(len(null_array)),
    }


def rowwise_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rank_x = stats.rankdata(x, axis=1)
    rank_y = stats.rankdata(y, axis=1)
    rank_x -= rank_x.mean(axis=1, keepdims=True)
    rank_y -= rank_y.mean(axis=1, keepdims=True)
    numerator = np.sum(rank_x * rank_y, axis=1)
    denominator = np.sqrt(np.sum(rank_x**2, axis=1) * np.sum(rank_y**2, axis=1))
    return numerator / np.maximum(denominator, 1e-12)


def correlation_difference_bootstrap(
    layer_pairs: pd.DataFrame,
    separation: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    patch = layer_pairs[layer_pairs["control"] == "matched_reverse"].pivot(
        index="pair_id", columns="layer", values="counterfactual_aligned_shift"
    )
    jacobian = separation[separation["method"] == "jacobian"].pivot(
        index="pair_id", columns="layer", values="relation_separation"
    )
    direct = separation[separation["method"] == "direct"].pivot(
        index="pair_id", columns="layer", values="relation_separation"
    )
    common_pairs = sorted(set(patch.index) & set(jacobian.index) & set(direct.index))
    common_layers = sorted(set(patch.columns) & set(jacobian.columns) & set(direct.columns))
    patch_values = patch.loc[common_pairs, common_layers].to_numpy()
    jacobian_values = jacobian.loc[common_pairs, common_layers].to_numpy()
    direct_values = direct.loc[common_pairs, common_layers].to_numpy()
    indices = rng.integers(0, len(common_pairs), size=(N_BOOT, len(common_pairs)))
    patch_boot = patch_values[indices].mean(axis=1)
    jacobian_boot = jacobian_values[indices].mean(axis=1)
    direct_boot = direct_values[indices].mean(axis=1)
    rho_j = rowwise_spearman(patch_boot, jacobian_boot)
    rho_d = rowwise_spearman(patch_boot, direct_boot)
    difference = rho_j - rho_d
    observed_j = float(stats.spearmanr(patch_values.mean(axis=0), jacobian_values.mean(axis=0)).statistic)
    observed_d = float(stats.spearmanr(patch_values.mean(axis=0), direct_values.mean(axis=0)).statistic)
    return {
        "observed_jacobian_minus_direct_rho": observed_j - observed_d,
        "pair_bootstrap_ci95": [float(value) for value in np.quantile(difference, [0.025, 0.975])],
        "pair_bootstrap_probability_difference_above_zero": float(np.mean(difference > 0)),
        "resamples": N_BOOT,
        "status": "post hoc falsification audit; not a frozen primary endpoint",
    }


def relative_onsets(curves: dict[str, np.ndarray], depths: np.ndarray) -> dict:
    output = {}
    for label, values in curves.items():
        peak = float(np.max(values))
        output[label] = {"peak": peak}
        for fraction in (0.1, 0.5):
            hits = np.where(values >= fraction * peak)[0]
            output[label][f"first_depth_at_{int(100 * fraction)}pct_peak"] = (
                None if len(hits) == 0 else float(depths[hits[0]])
            )
    return output


def validate(raw: dict, protocol: dict, patch: pd.DataFrame, readout: pd.DataFrame) -> None:
    if raw["provenance"]["protocol_sha256"] != "447327e8f277ca5cfc60944a5514e526aca89d168bf5435575390f512fe54258":
        raise RuntimeError("raw output is not tied to the frozen protocol")
    if len(raw["clean_prompts"]) != 24 or len(patch) != 2400 or len(readout) != 2400:
        raise RuntimeError("unexpected row count")
    if patch[["receiver_prompt_id", "control", "layer"]].duplicated().any():
        raise RuntimeError("duplicate patch rows")
    if readout[["prompt_id", "method", "lens_seed", "layer"]].fillna(-1).duplicated().any():
        raise RuntimeError("duplicate readout rows")
    if not np.isfinite(patch.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("non-finite patch value")
    readout_numeric = readout.select_dtypes(include=[np.number]).drop(
        columns=["lens_seed"], errors="ignore"
    )
    if not np.isfinite(readout_numeric.to_numpy()).all():
        raise RuntimeError("non-finite readout value")
    if sorted(patch["layer"].unique()) != protocol["source_layers"]:
        raise RuntimeError("patch layers do not match protocol")


def pair_band_effects(
    patch: pd.DataFrame, band: tuple[float, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    within = patch[
        patch["depth_percent"].between(band[0], band[1], inclusive="both")
    ].copy()
    prompt = (
        within.groupby(
            [
                "receiver_pair_id", "receiver_condition_id", "receiver_relation",
                "receiver_presentation_order", "control",
            ],
            as_index=False,
        )["counterfactual_aligned_shift"]
        .mean()
        .rename(columns={"counterfactual_aligned_shift": "band_mean_shift"})
    )
    pair = (
        prompt.groupby(["receiver_pair_id", "control"], as_index=False)["band_mean_shift"]
        .mean()
        .rename(columns={"receiver_pair_id": "pair_id"})
    )
    return prompt, pair


def layer_pair_effects(patch: pd.DataFrame) -> pd.DataFrame:
    return (
        patch.groupby(
            ["receiver_pair_id", "control", "layer", "depth_percent"], as_index=False
        )["counterfactual_aligned_shift"]
        .mean()
        .rename(columns={"receiver_pair_id": "pair_id"})
    )


def readout_separation(readout: pd.DataFrame) -> pd.DataFrame:
    collapsed = (
        readout.groupby(
            [
                "prompt_id", "pair_id", "relation", "presentation_order",
                "method", "layer", "depth_percent",
            ],
            as_index=False,
        )["higher_minus_lower"]
        .mean()
    )
    pivot = collapsed.pivot_table(
        index=["pair_id", "presentation_order", "method", "layer", "depth_percent"],
        columns="relation",
        values="higher_minus_lower",
    ).reset_index()
    pivot["relation_separation"] = 0.5 * (
        pivot["refinement"] - pivot["coarsening"]
    )
    return (
        pivot.groupby(
            ["pair_id", "method", "layer", "depth_percent"], as_index=False
        )["relation_separation"]
        .mean()
    )


def curve_with_ci(
    frame: pd.DataFrame, value: str, group: str, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    for keys, subset in frame.groupby([group, "layer", "depth_percent"], sort=True):
        group_value, layer, depth = keys
        values = subset.sort_values("pair_id")[value].to_numpy()
        summary = bootstrap_mean(values, rng)
        rows.append({
            group: group_value,
            "layer": int(layer),
            "depth_percent": float(depth),
            "mean": summary["mean"],
            "ci_low": summary["ci95"][0],
            "ci_high": summary["ci95"][1],
        })
    return pd.DataFrame(rows)


def make_figure(
    patch_curve: pd.DataFrame,
    layer_pairs: pd.DataFrame,
    readout_curve: pd.DataFrame,
    correlations: dict,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
    })
    # A wide, shallow canvas keeps the three scientific panels readable at
    # manuscript width. Explicit positions keep long material labels and the
    # two colorbars from creating oversized automatic-layout gutters.
    fig = plt.figure(figsize=(12.6, 3.5))
    ax_a = fig.add_axes([0.060, 0.245, 0.245, 0.700])
    ax_b = fig.add_axes([0.400, 0.245, 0.240, 0.700])
    colorbar_b_ax = fig.add_axes([0.647, 0.265, 0.010, 0.660])
    ax_c = fig.add_axes([0.725, 0.245, 0.220, 0.700])
    colorbar_c_ax = fig.add_axes([0.952, 0.265, 0.010, 0.660])
    legend_ax = fig.add_axes([0.060, 0.015, 0.902, 0.120])
    legend_ax.axis("off")

    palette = {
        "matched_reverse": "#167D8D",
        "cross_material_reverse": "#4C72B0",
        "distance_matched_cross_material_same": "#777777",
        "distance_matched_order_only": "#B8B8B8",
    }
    labels = {
        "matched_reverse": "same material, relation reversed",
        "cross_material_reverse": "new material, relation reversed",
        "distance_matched_cross_material_same": "same relation, distance matched",
        "distance_matched_order_only": "answer order, distance matched",
    }
    for control in [
        "matched_reverse", "cross_material_reverse",
        "distance_matched_cross_material_same", "distance_matched_order_only",
    ]:
        subset = patch_curve[patch_curve["control"] == control].sort_values("depth_percent")
        line_styles = {
            "matched_reverse": "-",
            "cross_material_reverse": "--",
            "distance_matched_cross_material_same": "-.",
            "distance_matched_order_only": ":",
        }
        ax_a.plot(
            subset["depth_percent"], subset["mean"], marker="o", markersize=2.8,
            linewidth=1.5, linestyle=line_styles[control], color=palette[control],
            label=labels[control],
        )
        if control == "matched_reverse":
            ax_a.fill_between(
                subset["depth_percent"].to_numpy(), subset["ci_low"].to_numpy(),
                subset["ci_high"].to_numpy(), color=palette[control], alpha=0.15,
                linewidth=0,
            )
    ax_a.axhline(0, color="#555555", linewidth=0.7)
    ax_a.axvspan(38, 92, color="#999999", alpha=0.08, linewidth=0)
    ax_a.set_xlabel("Layer depth (%)")
    ax_a.set_ylabel("Shift toward reversed answer\n(log-odds units)")
    ax_a.text(0.02, 0.98, "A", transform=ax_a.transAxes, ha="left", va="top", fontweight="bold")
    handles, legend_labels = ax_a.get_legend_handles_labels()
    legend_ax.legend(
        handles, legend_labels, frameon=False, loc="center", ncol=4,
        columnspacing=1.6, handlelength=2.6,
    )

    matched = layer_pairs[layer_pairs["control"] == "matched_reverse"]
    heat = matched.pivot(index="pair_id", columns="depth_percent", values="counterfactual_aligned_shift")
    heat = heat.loc[sorted(heat.index)]
    vmax = float(np.quantile(np.abs(heat.to_numpy()), 0.98))
    image = ax_b.imshow(
        heat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest", extent=[heat.columns.min(), heat.columns.max(), len(heat), 0],
    )
    short_names = {
        "bronze-36-9": "bronze",
        "cobalt-alloy-56-14": "cobalt",
        "low-carbon-steel-72-18": "low-carbon steel",
        "magnesium-48-12": "magnesium",
        "silver-40-10": "silver",
        "titanium-64-8": "titanium",
    }
    short = [short_names.get(name, name) for name in heat.index]
    ax_b.set_yticks(np.arange(len(short)) + 0.5, short)
    ax_b.set_xlabel("Layer depth (%)")
    colorbar = fig.colorbar(image, cax=colorbar_b_ax)
    colorbar.ax.set_title("Δ log odds", fontsize=7.5, pad=3)
    ax_b.text(0.02, 0.98, "B", transform=ax_b.transAxes, ha="left", va="top", fontweight="bold")

    patch_match = patch_curve[patch_curve["control"] == "matched_reverse"].sort_values("layer")
    for method, marker, color, label in [
        ("jacobian", "o", "#167D8D", "Jacobian readout"),
        ("direct", "^", "#6F5AA8", "direct readout"),
    ]:
        read = readout_curve[readout_curve["method"] == method].sort_values("layer")
        merged = patch_match.merge(read, on=["layer", "depth_percent"], suffixes=("_patch", "_read"))
        scatter = ax_c.scatter(
            merged["mean_read"], merged["mean_patch"], c=merged["depth_percent"],
            cmap="viridis", marker=marker, s=30, edgecolor="none", label=label,
        )
    ax_c.axhline(0, color="#777777", linewidth=0.7)
    ax_c.axvline(0, color="#777777", linewidth=0.7)
    ax_c.set_xlabel("Readable relation separation\n(log-odds units)")
    ax_c.set_ylabel("Reverse-patch shift")
    ax_c.yaxis.labelpad = 2
    ax_c.legend(frameon=False, loc="best")
    ax_c.text(
        0.97, 0.04,
        f"Jacobian ρ={correlations['jacobian']['spearman_rho']:+.2f}\n"
        f"direct ρ={correlations['direct']['spearman_rho']:+.2f}",
        transform=ax_c.transAxes, ha="right", va="bottom",
    )
    ax_c.text(0.02, 0.98, "C", transform=ax_c.transAxes, ha="left", va="top", fontweight="bold")
    depth_bar = fig.colorbar(scatter, cax=colorbar_c_ax)
    depth_bar.set_label("Depth (%)")

    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"counterfactual-activation-patching.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    raw = json.loads(RAW.read_text())
    distance_raw = json.loads(DISTANCE_RAW.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    patch = pd.DataFrame(raw["patch_rows"])
    distance_patch = pd.DataFrame(distance_raw["rows"])
    readout = pd.DataFrame(raw["readout_rows"])
    validate(raw, protocol, patch, readout)
    if len(distance_patch) != 1200 or distance_patch[
        ["receiver_prompt_id", "control", "layer"]
    ].duplicated().any():
        raise RuntimeError("invalid distance-matched falsification rows")
    norm_error = np.abs(
        distance_patch["achieved_difference_norm"]
        - distance_patch["target_reverse_difference_norm"]
    )
    if float(norm_error.max()) > 1e-4:
        raise RuntimeError("distance-matched control failed its norm constraint")
    rng = np.random.default_rng(SEED)
    band = tuple(float(value) for value in protocol["workspace_band_percent"])

    prompt_band, pair_band = pair_band_effects(patch, band)
    distance_prompt_band, distance_pair_band = pair_band_effects(distance_patch, band)
    band_state_diagnostics = (
        patch[patch["depth_percent"].between(band[0], band[1], inclusive="both")]
        .groupby("control", as_index=False)[
            [
                "receiver_donor_state_cosine", "state_difference_norm",
                "receiver_state_norm", "donor_state_norm",
            ]
        ]
        .mean()
    )
    layer_pairs = layer_pair_effects(patch)
    distance_layer_pairs = layer_pair_effects(distance_patch)
    separation = readout_separation(readout)
    patch_curve = curve_with_ci(
        layer_pairs, "counterfactual_aligned_shift", "control", rng
    )
    distance_patch_curve = curve_with_ci(
        distance_layer_pairs, "counterfactual_aligned_shift", "control", rng
    )
    display_patch_curve = pd.concat(
        [patch_curve, distance_patch_curve], ignore_index=True
    )
    readout_curve = curve_with_ci(
        separation, "relation_separation", "method", rng
    )

    pair_pivot = pair_band.pivot(index="pair_id", columns="control", values="band_mean_shift")
    endpoints = {
        control: bootstrap_mean(pair_pivot[control].to_numpy(), rng)
        for control in protocol["donor_controls"]
    }
    contrasts = {
        "matched_minus_order_only": paired_bootstrap(
            pair_pivot["matched_reverse"], pair_pivot["order_only"], rng
        ),
        "matched_minus_cross_material_same": paired_bootstrap(
            pair_pivot["matched_reverse"], pair_pivot["cross_material_same"], rng
        ),
        "matched_minus_cross_material_reverse": paired_bootstrap(
            pair_pivot["matched_reverse"], pair_pivot["cross_material_reverse"], rng
        ),
    }
    distance_pair_pivot = distance_pair_band.pivot(
        index="pair_id", columns="control", values="band_mean_shift"
    )
    distance_endpoints = {
        control: bootstrap_mean(distance_pair_pivot[control].to_numpy(), rng)
        for control in distance_pair_pivot.columns
    }
    distance_contrasts = {
        "matched_minus_distance_matched_cross_material_same": paired_bootstrap(
            pair_pivot["matched_reverse"],
            distance_pair_pivot["distance_matched_cross_material_same"], rng,
        ),
        "matched_minus_distance_matched_order_only": paired_bootstrap(
            pair_pivot["matched_reverse"],
            distance_pair_pivot["distance_matched_order_only"], rng,
        ),
    }

    condition = (
        prompt_band.groupby(
            ["receiver_condition_id", "receiver_pair_id", "receiver_relation", "control"],
            as_index=False,
        )["band_mean_shift"]
        .mean()
    )
    matched_condition = condition[condition["control"] == "matched_reverse"]
    positive_conditions = int(np.sum(matched_condition["band_mean_shift"] > 0))
    sign_test = float(stats.binomtest(positive_conditions, len(matched_condition), 0.5).pvalue)

    mean_matched = patch_curve[patch_curve["control"] == "matched_reverse"]
    peak = mean_matched.loc[mean_matched["mean"].idxmax()]
    peak_layer = int(peak["layer"])
    peak_rows = patch[(patch["control"] == "matched_reverse") & (patch["layer"] == peak_layer)].copy()
    receiver_sign = np.where(peak_rows["receiver_relation"] == "refinement", 1.0, -1.0)
    peak_rows["donor_answer_selected"] = (
        -receiver_sign * peak_rows["patched_higher_minus_lower"].to_numpy() > 0
    )
    peak_rows["receiver_clean_answer_selected"] = (
        receiver_sign * peak_rows["clean_higher_minus_lower"].to_numpy() > 0
    )

    patch_match = mean_matched.sort_values("layer")
    correlations = {}
    for method in ("jacobian", "direct"):
        method_curve = readout_curve[readout_curve["method"] == method].sort_values("layer")
        merged = patch_match.merge(method_curve, on=["layer", "depth_percent"], suffixes=("_patch", "_read"))
        correlations[method] = circular_shift_test(
            merged["mean_patch"].to_numpy(), merged["mean_read"].to_numpy()
        )

    aligned_curves = {
        "matched_patch": patch_match.sort_values("layer")["mean"].to_numpy(),
    }
    ordered_depths = patch_match.sort_values("layer")["depth_percent"].to_numpy()
    first_difference = {}
    for method in ("jacobian", "direct"):
        method_curve = readout_curve[readout_curve["method"] == method].sort_values("layer")
        aligned_curves[f"{method}_readout"] = method_curve["mean"].to_numpy()
        first_difference[method] = float(stats.spearmanr(
            np.diff(aligned_curves["matched_patch"]),
            np.diff(aligned_curves[f"{method}_readout"]),
        ).statistic)
    posthoc_correlation_audit = {
        "first_difference_spearman": first_difference,
        "relative_onsets": relative_onsets(aligned_curves, ordered_depths),
        "jacobian_minus_direct_pair_bootstrap": correlation_difference_bootstrap(
            layer_pairs, separation, rng
        ),
        "interpretation": (
            "The level correlation can be driven by a shared late-layer transition. "
            "First differences and relative onsets test whether the transitions coincide exactly."
        ),
        "status": "post hoc falsification audit added after inspecting the frozen outputs",
    }

    clean = pd.DataFrame(raw["clean_prompts"])
    clean_sign = np.where(clean["relation"] == "refinement", 1.0, -1.0)
    clean_correct = clean_sign * clean["clean_log_odds"].to_numpy() > 0

    decision = {
        "matched_reverse_ci_above_zero": endpoints["matched_reverse"]["ci95"][0] > 0,
        "matched_minus_order_ci_above_zero": contrasts["matched_minus_order_only"]["ci95"][0] > 0,
        "matched_minus_same_ci_above_zero": contrasts["matched_minus_cross_material_same"]["ci95"][0] > 0,
        "cross_material_reverse_ci_above_zero": endpoints["cross_material_reverse"]["ci95"][0] > 0,
    }
    decision["causal_localization_candidate_pass"] = all([
        decision["matched_reverse_ci_above_zero"],
        decision["matched_minus_order_ci_above_zero"],
        decision["matched_minus_same_ci_above_zero"],
    ])
    decision["distance_falsification_survives"] = all(
        summary["ci95"][0] > 0 for summary in distance_contrasts.values()
    )

    statistics = {
        "status": "exploratory causal follow-up under frozen protocol on previously inspected prompts",
        "band_percent": list(band),
        "n_material_pairs": 6,
        "n_conditions": 12,
        "n_prompt_orders": 24,
        "n_layers": 25,
        "clean": {
            "physically_correct_prompt_orders": int(clean_correct.sum()),
            "n_prompt_orders": int(len(clean)),
            "global_top_is_registered_answer": int(clean["clean_top_is_registered_answer"].sum()),
            "mean_registered_pair_probability": float(clean["clean_pair_probability"].mean()),
        },
        "band_endpoints": endpoints,
        "band_contrasts": contrasts,
        "distance_matched_falsification": {
            "status": "post-hoc falsification protocol frozen before control output",
            "maximum_absolute_norm_matching_error": float(norm_error.max()),
            "control_endpoints": distance_endpoints,
            "matched_reverse_contrasts": distance_contrasts,
        },
        "band_state_diagnostics_by_control": band_state_diagnostics.set_index(
            "control"
        ).to_dict(orient="index"),
        "matched_reverse_condition_sign_test": {
            "positive_conditions": positive_conditions,
            "n_conditions": int(len(matched_condition)),
            "exact_two_sided_p": sign_test,
        },
        "descriptive_peak": {
            "layer": peak_layer,
            "depth_percent": float(peak["depth_percent"]),
            "mean_shift": float(peak["mean"]),
            "ci95": [float(peak["ci_low"]), float(peak["ci_high"])],
            "donor_answer_selected_prompt_orders": int(peak_rows["donor_answer_selected"].sum()),
            "receiver_clean_answer_selected_prompt_orders": int(peak_rows["receiver_clean_answer_selected"].sum()),
            "n_prompt_orders": int(len(peak_rows)),
        },
        "readout_causality_correlations": correlations,
        "posthoc_readout_causality_audit": posthoc_correlation_audit,
        "decision": decision,
        "guardrail": (
            "Full-state activation patching tests causal sufficiency in a constrained matched task; "
            "it is not a literal chain of thought, a necessity result, or unrestricted materials understanding."
        ),
    }

    prompt_band.to_csv(OUT / "band_prompt_effects.csv", index=False)
    pair_band.to_csv(OUT / "band_pair_effects.csv", index=False)
    distance_prompt_band.to_csv(OUT / "distance_control_band_prompt_effects.csv", index=False)
    distance_pair_band.to_csv(OUT / "distance_control_band_pair_effects.csv", index=False)
    condition.to_csv(OUT / "band_condition_effects.csv", index=False)
    band_state_diagnostics.to_csv(OUT / "band_state_diagnostics.csv", index=False)
    layer_pairs.to_csv(OUT / "layer_pair_effects.csv", index=False)
    patch_curve.to_csv(OUT / "layer_curves.csv", index=False)
    distance_patch_curve.to_csv(OUT / "distance_control_layer_curves.csv", index=False)
    separation.to_csv(OUT / "readout_relation_separation_by_pair.csv", index=False)
    readout_curve.to_csv(OUT / "readout_relation_separation_curves.csv", index=False)
    dump_json(OUT / "statistics.json", statistics)
    make_figure(display_patch_curve, layer_pairs, readout_curve, correlations)

    matched = endpoints["matched_reverse"]
    reverse = endpoints["cross_material_reverse"]
    order_contrast = contrasts["matched_minus_order_only"]
    same_contrast = contrasts["matched_minus_cross_material_same"]
    distance_same = distance_endpoints["distance_matched_cross_material_same"]
    distance_order = distance_endpoints["distance_matched_order_only"]
    distance_same_contrast = distance_contrasts[
        "matched_minus_distance_matched_cross_material_same"
    ]
    distance_order_contrast = distance_contrasts[
        "matched_minus_distance_matched_order_only"
    ]
    lines = [
        "# Candidate counterfactual activation-patching results",
        "",
        "This is an exploratory causal follow-up under a frozen protocol on a previously inspected prompt cohort. It is reported in the Supplementary Information and is not an independent confirmation.",
        "",
        "## What was done",
        "",
        "For six matched materials, the final-prompt-token residual from a refinement prompt was transplanted into its coarsening counterpart and vice versa at each of 25 layers. Same-relation, answer-order-only, and cross-material donors were run as controls. Positive shifts below mean movement toward the answer appropriate to the donor's reversed physical relation.",
        "",
        "## Registered band result",
        "",
        f"The same-material reversed-relation patch produced a mean shift of **{matched['mean']:+.3f} log-odds units** (95% pair-bootstrap CI {matched['ci95'][0]:+.3f} to {matched['ci95'][1]:+.3f}) over the frozen 38--92% layer band. It was positive for {positive_conditions}/12 physical conditions (two-sided sign-test p={sign_test:.4g}).",
        f"Relative to the answer-order-only donor, the paired gain was **{order_contrast['mean']:+.3f}** (95% CI {order_contrast['ci95'][0]:+.3f} to {order_contrast['ci95'][1]:+.3f}); relative to a different-material donor with the same relation, it was **{same_contrast['mean']:+.3f}** (95% CI {same_contrast['ci95'][0]:+.3f} to {same_contrast['ci95'][1]:+.3f}).",
        f"The reversed relation also transferred across materials: **{reverse['mean']:+.3f}** (95% CI {reverse['ci95'][0]:+.3f} to {reverse['ci95'][1]:+.3f}).",
        "",
        "## State-distance falsification",
        "",
        f"Reverse donors were farther from the receiver than the original controls, so a second protocol scaled the relation-preserving and order-only donor differences to exactly the reverse donor's state distance. The distance-matched same-relation control was **{distance_same['mean']:+.3f}** (95% CI {distance_same['ci95'][0]:+.3f} to {distance_same['ci95'][1]:+.3f}) and the distance-matched order control was **{distance_order['mean']:+.3f}** (95% CI {distance_order['ci95'][0]:+.3f} to {distance_order['ci95'][1]:+.3f}). The matched reverse advantage remained **{distance_same_contrast['mean']:+.3f}** and **{distance_order_contrast['mean']:+.3f}**, with respective 95% intervals {distance_same_contrast['ci95'][0]:+.3f} to {distance_same_contrast['ci95'][1]:+.3f} and {distance_order_contrast['ci95'][0]:+.3f} to {distance_order_contrast['ci95'][1]:+.3f}.",
        "",
        "## Layer localization and readout bridge",
        "",
        f"The descriptive matched-patch curve peaked at layer {peak_layer} ({float(peak['depth_percent']):.1f}% depth) with a mean shift of {float(peak['mean']):+.3f}. Across the 25 registered layers, its Spearman association with readable relation separation was {correlations['jacobian']['spearman_rho']:+.3f} for the Jacobian lens and {correlations['direct']['spearman_rho']:+.3f} for direct unembedding. However, the post-hoc first-difference correlations were {first_difference['jacobian']:+.3f} and {first_difference['direct']:+.3f}; the strong level correlation therefore reflects a broadly shared late-layer transition, not exact layer-by-layer coincidence. Circular-shift and pair-bootstrap sensitivities are stored in `statistics.json` and remain descriptive.",
        "",
        "## Decision",
        "",
        f"Frozen causal-localization decision: **{'PASS' if decision['causal_localization_candidate_pass'] else 'FAIL'}**. Cross-material transfer: **{'PASS' if decision['cross_material_reverse_ci_above_zero'] else 'FAIL'}**. Post-hoc state-distance falsification: **{'SURVIVES' if decision['distance_falsification_survives'] else 'DOES NOT SURVIVE'}**.",
        "",
        "The passing result shows causal sufficiency of a matched internal state for this constrained relation, not a private verbal chain of thought or unrestricted physical understanding. A disjoint prompt cohort is required before treating it as confirmatory.",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
