#!/usr/bin/env python3
"""Analyze the frozen 60-law neutral-anchored relational benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "experiments" / "elicited-physics-abstraction-2026-07-18"
OUT = ROOT / "experiments" / "neutral-anchored-relational-physics-2026-07-18"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def fit_centroid(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive, negative = values[labels].mean(0), values[~labels].mean(0)
    direction = positive - negative
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return direction, 0.5 * (positive + negative)


def project(values: np.ndarray, direction: np.ndarray, midpoint: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ij,j->i", values - midpoint, direction, optimize=False
    )


def load_development() -> dict:
    manifest = json.loads((DEV / "prompt_manifest.json").read_text())
    with np.load(DEV / "representations.npz", allow_pickle=False) as arrays:
        ids = arrays["prompt_ids"].astype(str)
        positions = arrays["positions"].astype(str)
        layers = arrays["layers"].astype(int)
        states = arrays["raw_states"].astype(np.float64)
    by_id = {x["prompt_id"]: x for x in manifest["prompts"]}
    prompts = [by_id[x] for x in ids]
    return {
        "prompts": prompts,
        "positions": positions,
        "layers": layers,
        "states": states,
    }


def load_test() -> dict:
    manifest = json.loads((OUT / "prompt_manifest.json").read_text())
    raw = json.loads((OUT / "raw.json").read_text())
    with np.load(OUT / "representations.npz", allow_pickle=False) as arrays:
        ids = arrays["prompt_ids"].astype(str)
        layer = int(arrays["layer"][0])
        states = arrays["raw_states"].astype(np.float64)
    by_id = {x["prompt_id"]: x for x in manifest["prompts"]}
    clean_by_id = {x["prompt_id"]: x for x in raw["clean_rows"]}
    return {
        "manifest": manifest,
        "prompts": [by_id[x] for x in ids],
        "clean": [clean_by_id[x] for x in ids],
        "layer": layer,
        "states": states,
    }


def prompt_arrays(prompts: list[dict]) -> dict[str, np.ndarray]:
    return {
        "law": np.asarray([x["law_id"] for x in prompts]),
        "category": np.asarray([x.get("category", "") for x in prompts]),
        "domain": np.asarray([x.get("domain", "") for x in prompts]),
        "law_sign": np.asarray([x["law_sign"] for x in prompts], dtype=int),
        "neutral_role": np.asarray([x.get("neutral_role") or "" for x in prompts]),
        "physical": np.asarray([x["physical_sign"] > 0 for x in prompts]),
        "numeric": np.asarray([x["numerical_sign"] > 0 for x in prompts]),
        "surface": np.asarray([x["surface"] for x in prompts]),
        "case": np.asarray([x["case_index"] for x in prompts], dtype=int),
        "order": np.asarray([x["answer_order"] for x in prompts]),
    }


def matched_pairs(scores: np.ndarray, meta: dict) -> pd.DataFrame:
    rows = []
    for law_id in sorted(set(meta["law"])):
        for surface in ("a", "b"):
            for case in (0, 1):
                for order in sorted(set(meta["order"])):
                    base = (
                        (meta["law"] == law_id)
                        & (meta["surface"] == surface)
                        & (meta["case"] == case)
                        & (meta["order"] == order)
                    )
                    up, down = base & meta["numeric"], base & ~meta["numeric"]
                    if up.sum() != 1 or down.sum() != 1:
                        raise RuntimeError(
                            f"bad matched cell: {law_id}, {surface}, {case}, {order}"
                        )
                    rows.append(
                        {
                            "law_id": str(law_id),
                            "category": str(meta["category"][up][0]),
                            "domain": str(meta["domain"][up][0]),
                            "law_sign": int(meta["law_sign"][up][0]),
                            "neutral_role": str(meta["neutral_role"][up][0]),
                            "surface": surface,
                            "case_index": case,
                            "answer_order": order,
                            "contrast": float(scores[up][0] - scores[down][0]),
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_laws(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for law_id, group in pairs.groupby("law_id", sort=True):
        rows.append(
            {
                "law_id": law_id,
                "category": group["category"].iloc[0],
                "domain": group["domain"].iloc[0],
                "law_sign": int(group["law_sign"].iloc[0]),
                "neutral_role": group["neutral_role"].iloc[0],
                "mean_contrast": float(group["contrast"].mean()),
                "median_contrast": float(group["contrast"].median()),
                "contrast_sd": float(group["contrast"].std(ddof=1)),
                "surface_a_contrast": float(
                    group.loc[group["surface"] == "a", "contrast"].mean()
                ),
                "surface_b_contrast": float(
                    group.loc[group["surface"] == "b", "contrast"].mean()
                ),
                "n_pairs": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def pair_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    comparison = positive[:, None] - negative[None, :]
    return float(np.mean(comparison > 0) + 0.5 * np.mean(comparison == 0))


def endpoint_metrics(laws: pd.DataFrame, score_column: str) -> dict:
    direct = laws.loc[laws["category"] == "direct", score_column].to_numpy()
    inverse = laws.loc[laws["category"] == "inverse", score_column].to_numpy()
    neutral = laws.loc[
        (laws["category"] == "neutral")
        & (laws["neutral_role"] == "validation"),
        score_column,
    ].to_numpy()
    return {
        "direct_vs_inverse_auc": pair_auc(direct, inverse),
        "direct_vs_validation_neutral_auc": pair_auc(direct, neutral),
        "validation_neutral_vs_inverse_auc": pair_auc(neutral, inverse),
    }


def calibrate(laws: pd.DataFrame, column: str = "mean_contrast") -> tuple[float, float]:
    calibration = laws.loc[
        (laws["category"] == "neutral")
        & (laws["neutral_role"] == "calibration"),
        column,
    ].to_numpy()
    center = float(np.median(calibration))
    mad = float(np.median(np.abs(calibration - center)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        raise RuntimeError(f"neutral MAD is degenerate: {scale}")
    return center, scale


def stratified_auc_bootstrap(
    positive: np.ndarray,
    negative: np.ndarray,
    seed: int,
    n: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    output = np.empty(n)
    for start in range(0, n, 5000):
        count = min(5000, n - start)
        p = rng.choice(positive, size=(count, len(positive)), replace=True)
        q = rng.choice(negative, size=(count, len(negative)), replace=True)
        differences = p[:, :, None] - q[:, None, :]
        output[start : start + count] = (
            np.mean(differences > 0, axis=(1, 2))
            + 0.5 * np.mean(differences == 0, axis=(1, 2))
        )
    return [float(x) for x in np.quantile(output, [0.025, 0.975])]


def ordinal_permutation(
    signs: np.ndarray,
    scores: np.ndarray,
    seed: int,
    n: int,
) -> tuple[float, float]:
    observed = float(spearmanr(signs, scores).statistic)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n):
        permuted = rng.permutation(signs)
        statistic = float(spearmanr(permuted, scores).statistic)
        extreme += statistic >= observed - 1e-15
    return observed, float((extreme + 1) / (n + 1))


def tfidf_scores(dev: dict, test: dict, method: str) -> np.ndarray:
    if method == "word":
        vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True
        )
    else:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=30000,
            sublinear_tf=True,
        )
    dev_values = normalize(
        vectorizer.fit_transform([x["user"] for x in dev["prompts"]]).toarray()
    )
    test_values = normalize(
        vectorizer.transform([x["user"] for x in test["prompts"]]).toarray()
    )
    labels = np.asarray([x["physical_sign"] > 0 for x in dev["prompts"]])
    direction, midpoint = fit_centroid(dev_values, labels)
    return project(test_values, direction, midpoint)


def main() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name in ("prompt_manifest", "runner", "development_manifest", "development_states"):
        record = protocol["inputs"][name]
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen input changed: {name}")
    dev, test = load_development(), load_test()
    if test["layer"] != protocol["frozen_lens"]["layer"]:
        raise RuntimeError("captured layer differs from frozen layer")
    dev_position = int(
        np.flatnonzero(
            dev["positions"] == protocol["frozen_lens"]["position"]
        )[0]
    )
    dev_layer = int(
        np.flatnonzero(dev["layers"] == protocol["frozen_lens"]["layer"])[0]
    )
    dev_values = normalize(dev["states"][dev_position, :, dev_layer])
    dev_labels = np.asarray(
        [x["physical_sign"] > 0 for x in dev["prompts"]]
    )
    direction, midpoint = fit_centroid(dev_values, dev_labels)
    hidden_scores = project(normalize(test["states"]), direction, midpoint)
    meta = prompt_arrays(test["prompts"])
    pairs = matched_pairs(hidden_scores, meta)
    laws = aggregate_laws(pairs)
    center, scale = calibrate(laws)
    laws["neutral_centered_contrast"] = laws["mean_contrast"] - center
    laws["neutral_z"] = laws["neutral_centered_contrast"] / scale
    pairs["neutral_z"] = (pairs["contrast"] - center) / scale

    endpoints = endpoint_metrics(laws, "neutral_centered_contrast")
    directional = laws[laws["category"].isin(["direct", "inverse"])]
    calibrated_accuracy = float(
        np.mean(
            (directional["neutral_centered_contrast"] > 0)
            == (directional["category"] == "direct")
        )
    )
    ordinal = laws[
        (laws["category"] != "neutral")
        | (laws["neutral_role"] == "validation")
    ]
    rho, ordinal_p = ordinal_permutation(
        ordinal["law_sign"].to_numpy(),
        ordinal["neutral_z"].to_numpy(),
        protocol["inference"]["permutation_seed"],
        protocol["inference"]["permutations"],
    )
    surface_endpoints = {}
    for surface in ("a", "b"):
        column = f"surface_{surface}_contrast"
        local = laws.copy()
        local_center, _ = calibrate(local, column)
        local[f"{column}_centered"] = local[column] - local_center
        surface_endpoints[surface] = endpoint_metrics(
            local, f"{column}_centered"
        )["direct_vs_inverse_auc"]

    direct = laws.loc[
        laws["category"] == "direct", "neutral_centered_contrast"
    ].to_numpy()
    inverse = laws.loc[
        laws["category"] == "inverse", "neutral_centered_contrast"
    ].to_numpy()
    neutral_validation = laws.loc[
        (laws["category"] == "neutral")
        & (laws["neutral_role"] == "validation"),
        "neutral_centered_contrast",
    ].to_numpy()
    bootstrap = {
        "direct_vs_inverse": stratified_auc_bootstrap(
            direct,
            inverse,
            protocol["inference"]["bootstrap_seed"],
            protocol["inference"]["bootstrap_resamples"],
        ),
        "direct_vs_neutral": stratified_auc_bootstrap(
            direct,
            neutral_validation,
            protocol["inference"]["bootstrap_seed"] + 1,
            protocol["inference"]["bootstrap_resamples"],
        ),
        "neutral_vs_inverse": stratified_auc_bootstrap(
            neutral_validation,
            inverse,
            protocol["inference"]["bootstrap_seed"] + 2,
            protocol["inference"]["bootstrap_resamples"],
        ),
    }

    output_scores = np.asarray(
        [x["higher_minus_lower_logit"] for x in test["clean"]], dtype=float
    )
    controls = {}
    control_law_tables = {}
    for name, scores in {
        "output_head": output_scores,
        "word_tfidf": tfidf_scores(dev, test, "word"),
        "character_tfidf": tfidf_scores(dev, test, "character"),
    }.items():
        table = aggregate_laws(matched_pairs(scores, meta))
        calibration_values = table.loc[
            (table["category"] == "neutral")
            & (table["neutral_role"] == "calibration"),
            "mean_contrast",
        ].to_numpy()
        control_center = float(np.median(calibration_values))
        control_mad = float(
            np.median(np.abs(calibration_values - control_center))
        )
        control_scale = 1.4826 * control_mad
        # A text control can assign exactly the same score to both members of
        # every neutral pair, making its null MAD exactly zero. AUC is
        # threshold-free and remains well-defined; unit scale is used only to
        # keep the diagnostic z column finite.
        if control_scale <= 1e-12:
            control_scale = 1.0
        table["neutral_centered_contrast"] = table["mean_contrast"] - control_center
        table["neutral_z"] = table["neutral_centered_contrast"] / control_scale
        controls[name] = endpoint_metrics(table, "neutral_centered_contrast")
        control_law_tables[name] = table

    behavior = pd.DataFrame(test["clean"])
    behavior_summary = {
        category: float(group.loc[group["behavior_evaluable"], "correct"].mean())
        for category, group in behavior.groupby("category")
        if bool(group["behavior_evaluable"].any())
    }
    rule = protocol["success_rule"]
    passed = bool(
        endpoints["direct_vs_inverse_auc"]
        >= rule["direct_inverse_auc_minimum"]
        and endpoints["direct_vs_validation_neutral_auc"]
        >= rule["direct_neutral_auc_minimum"]
        and endpoints["validation_neutral_vs_inverse_auc"]
        >= rule["neutral_inverse_auc_minimum"]
        and calibrated_accuracy
        >= rule["calibrated_direct_inverse_accuracy_minimum"]
        and min(surface_endpoints.values())
        >= rule["both_surface_direct_inverse_auc_minimum"]
        and rho >= rule["ordinal_spearman_minimum"]
        and ordinal_p <= rule["ordinal_permutation_p_maximum"]
    )
    summary = {
        "study_id": protocol["study_id"],
        "analysis_sha256": sha256(Path(__file__).resolve()),
        "neutral_calibration": {
            "center": center,
            "robust_scale_1.4826_mad": scale,
            "n_calibration_laws": 10,
            "n_validation_laws": 10,
        },
        "primary": {
            **endpoints,
            "bootstrap_95": bootstrap,
            "calibrated_direct_inverse_accuracy": calibrated_accuracy,
            "correct_directional_laws": int(
                np.sum(
                    (directional["neutral_centered_contrast"] > 0)
                    == (directional["category"] == "direct")
                )
            ),
            "directional_laws_total": int(len(directional)),
            "surface_direct_inverse_auc": surface_endpoints,
            "ordinal_spearman": rho,
            "ordinal_permutation_p": ordinal_p,
            "category_median_z": {
                category: float(group["neutral_z"].median())
                for category, group in laws.groupby("category")
            },
        },
        "controls": controls,
        "behavior_accuracy": behavior_summary,
        "passed_all_frozen_criteria": passed,
    }
    pairs.to_csv(OUT / "matched_pair_contrasts.csv", index=False)
    laws.to_csv(OUT / "law_level_neutral_normalized.csv", index=False)
    for name, table in control_law_tables.items():
        table.to_csv(OUT / f"{name}_law_level.csv", index=False)
    (OUT / "statistics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    # Publication-quality audit figure.
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    category_order = ["inverse", "neutral", "direct"]
    colors = {"inverse": "#7662a8", "neutral": "#8c8c8c", "direct": "#2a9d8f"}
    rng = np.random.default_rng(20260725)
    for index, category in enumerate(category_order):
        selected = laws[laws["category"] == category]["neutral_z"].to_numpy()
        jitter = rng.uniform(-0.10, 0.10, len(selected))
        axes[0, 0].scatter(
            np.full(len(selected), index) + jitter,
            selected,
            s=22,
            alpha=0.8,
            color=colors[category],
            edgecolor="white",
            linewidth=0.4,
        )
        axes[0, 0].plot(
            [index - 0.22, index + 0.22],
            [np.median(selected)] * 2,
            color="black",
            lw=2,
        )
    axes[0, 0].axhline(0, color="0.35", ls="--", lw=1)
    axes[0, 0].set_xticks(range(3), ["Inverse", "Neutral", "Direct"])
    axes[0, 0].set_ylabel("Contrast relative to neutral median (robust z)")

    ordered = laws.sort_values("neutral_z")
    axes[0, 1].barh(
        range(len(ordered)),
        ordered["neutral_z"],
        color=ordered["category"].map(colors),
        height=0.82,
    )
    axes[0, 1].axvline(0, color="0.25", lw=1)
    axes[0, 1].set_yticks([])
    axes[0, 1].set_xlabel("Neutral-normalized matched contrast")

    endpoint_names = [
        "Direct vs\ninverse",
        "Direct vs\nneutral",
        "Neutral vs\ninverse",
    ]
    method_names = ["Hidden", "Output", "Word TF-IDF", "Char TF-IDF"]
    method_values = {
        "Hidden": list(endpoints.values()),
        "Output": list(controls["output_head"].values()),
        "Word TF-IDF": list(controls["word_tfidf"].values()),
        "Char TF-IDF": list(controls["character_tfidf"].values()),
    }
    x = np.arange(3)
    width = 0.19
    method_colors = ["#276b9a", "#d39b35", "#a7a7a7", "#666666"]
    for index, name in enumerate(method_names):
        axes[1, 0].bar(
            x + (index - 1.5) * width,
            method_values[name],
            width,
            label=name,
            color=method_colors[index],
        )
    axes[1, 0].axhline(0.5, color="0.45", ls="--", lw=1)
    axes[1, 0].set_xticks(x, endpoint_names)
    axes[1, 0].set_ylim(0, 1.03)
    axes[1, 0].set_ylabel("Pairwise AUC")
    legend_handles, legend_labels = axes[1, 0].get_legend_handles_labels()

    surface_values = [
        surface_endpoints["a"],
        surface_endpoints["b"],
        endpoints["direct_vs_inverse_auc"],
        calibrated_accuracy,
    ]
    axes[1, 1].bar(
        range(4),
        surface_values,
        color=["#5c8fb6", "#5c8fb6", "#276b9a", "#2a9d8f"],
    )
    axes[1, 1].axhline(0.5, color="0.45", ls="--", lw=1)
    axes[1, 1].set_xticks(
        range(4),
        ["Explicit\nsurface", "Rearranged\nsurface", "Threshold-free\nAUC", "Neutral-cut\naccuracy"],
        fontsize=7,
    )
    axes[1, 1].set_ylim(0, 1.03)
    axes[1, 1].set_ylabel("Performance")
    for axis, label in zip(axes.flat, "ABCD"):
        axis.text(
            -0.10,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        fontsize=7.5,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0.02, 0.075, 1.0, 0.98), h_pad=2.4, w_pad=2.0)
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(
        OUT / "figures" / "neutral-anchored-relational-physics.png",
        dpi=240,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "figures" / "neutral-anchored-relational-physics.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    report = [
        "# Neutral-anchored relational physics",
        "",
        "The empirical null is defined by ten calibration-neutral laws, not by an assumed raw zero.",
        (
            f"Neutral center = {center:.6g}; robust scale = {scale:.6g}. "
            "Ten different neutral laws validate the calibration."
        ),
        "",
        f"- Direct versus inverse AUC: {endpoints['direct_vs_inverse_auc']:.3f}",
        f"- Direct versus validation-neutral AUC: {endpoints['direct_vs_validation_neutral_auc']:.3f}",
        f"- Validation-neutral versus inverse AUC: {endpoints['validation_neutral_vs_inverse_auc']:.3f}",
        f"- Neutral-cut direct/inverse accuracy: {calibrated_accuracy:.3f}",
        f"- Explicit/rearranged surface AUC: {surface_endpoints['a']:.3f}/{surface_endpoints['b']:.3f}",
        f"- Ordinal Spearman rho: {rho:.3f}; permutation p={ordinal_p:.6g}",
        f"- Passed all frozen criteria: {passed}",
        "",
        "All prompts, raw outputs, matched-pair contrasts, and law-level results are retained in this directory.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
