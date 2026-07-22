#!/usr/bin/env python3
"""Test option-free physical-outcome alignment across different mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_graph_topology_rigorous import (  # noqa: E402
    cosine_layers,
    deterministic_binary_similarity,
    prompt_feature_matrices,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "cross-mechanism-outcome-2026-07-18"
FIG = OUT / "figures"
PROTOCOL_PATH = OUT / "protocol.json"
PRIMARY_BAND = (38.0, 92.0)
BOOTSTRAP_SEED = 20260720
N_BOOT = 30_000

COLORS = {
    "jacobian": "#16697A",
    "direct": "#B56576",
    "raw": "#6C5B7B",
    "word_tfidf": "#5B8C5A",
    "char_tfidf": "#D17C38",
    "numeric_direction": "#2F4858",
    "answer_order": "#A0A4A8",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def exact_p(null: np.ndarray, observed: float) -> float:
    return float(np.mean(null >= observed - 1e-15))


def validate_inputs(protocol: Mapping[str, object]) -> dict[str, str]:
    actual = {}
    for name, row in protocol["inputs"].items():  # type: ignore[index]
        path = ROOT / str(row["path"])
        actual[name] = sha256(path)
        if actual[name] != str(row["sha256"]):
            raise RuntimeError(
                f"fingerprint mismatch for {name}: "
                f"{actual[name]} != {row['sha256']}"
            )
    return actual


def orientation_assignments(
    family_names: Sequence[str],
) -> list[dict[str, bool]]:
    output = []
    for direct_indices in itertools.combinations(range(len(family_names)), 3):
        direct = set(direct_indices)
        output.append(
            {
                family: index in direct
                for index, family in enumerate(family_names)
            }
        )
    if len(output) != 20:
        raise RuntimeError("expected 20 balanced response-orientation assignments")
    return output


def labels_from_orientation(
    rows: Sequence[Mapping[str, object]],
    orientation: Mapping[str, bool],
) -> np.ndarray:
    """Positive outcome iff numeric direction agrees with direct orientation."""

    return np.asarray(
        [
            (str(row["numeric_direction"]) == "increase")
            == bool(orientation[str(row["family_id"])])
            for row in rows
        ],
        dtype=bool,
    )


def observed_orientation(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, bool]:
    output = {}
    for row in rows:
        if str(row["numeric_direction"]) != "increase":
            continue
        family = str(row["family_id"])
        value = str(row["expected_outcome"]) == str(row["outcome_positive"])
        if family in output and output[family] != value:
            raise RuntimeError(f"inconsistent response orientation in {family}")
        output[family] = value
    if len(output) != 6 or sum(output.values()) != 3:
        raise RuntimeError(f"unexpected observed response orientation: {output}")
    return output


def query_rows(
    similarity: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    families: np.ndarray,
    variants: np.ndarray,
    labels: np.ndarray,
    orientation: Mapping[str, bool],
) -> pd.DataFrame:
    output = []
    family_names = sorted(set(families))
    variant_names = sorted(set(variants))
    for source in range(len(rows)):
        source_family = str(families[source])
        for target_family in family_names:
            if target_family == source_family:
                continue
            for target_variant in variant_names:
                candidates = np.flatnonzero(
                    (families == target_family)
                    & (variants == target_variant)
                )
                if len(candidates) != 4:
                    raise RuntimeError("cross-mechanism query must have four cases")
                same = labels[candidates] == labels[source]
                if int(np.sum(same)) != 2:
                    raise RuntimeError(
                        "cross-mechanism query must have two positives and two negatives"
                    )
                positive = candidates[same]
                negative = candidates[~same]
                positive_scores = similarity[source, positive]
                negative_scores = similarity[source, negative]
                comparisons = positive_scores[:, None] - negative_scores[None, :]
                auc = float(
                    np.mean(comparisons > 0)
                    + 0.5 * np.mean(comparisons == 0)
                )
                ordered = sorted(
                    candidates.tolist(),
                    key=lambda index: (
                        -float(similarity[source, index]),
                        index,
                    ),
                )
                pair = tuple(sorted((source_family, target_family)))
                output.append(
                    {
                        "source_index": source,
                        "source_prompt_id": str(rows[source]["prompt_id"]),
                        "source_family": source_family,
                        "source_variant": str(variants[source]),
                        "source_numeric_direction": str(
                            rows[source]["numeric_direction"]
                        ),
                        "source_positive_outcome": bool(labels[source]),
                        "target_family": target_family,
                        "target_variant": target_variant,
                        "family_pair": " | ".join(pair),
                        "opposite_response_orientation": bool(
                            orientation[source_family]
                            != orientation[target_family]
                        ),
                        "top1_same_outcome": bool(
                            labels[ordered[0]] == labels[source]
                        ),
                        "pairwise_auc": auc,
                        "candidate_indices": " ".join(
                            str(value) for value in candidates
                        ),
                    }
                )
    frame = pd.DataFrame(output)
    if len(frame) != 1080:
        raise RuntimeError(f"expected 1080 rankings, found {len(frame)}")
    if int(frame["opposite_response_orientation"].sum()) != 648:
        raise RuntimeError("expected 648 counter-numeric rankings")
    return frame


def summarize_queries(frame: pd.DataFrame) -> dict[str, float]:
    opposite = frame[frame["opposite_response_orientation"]]
    same = frame[~frame["opposite_response_orientation"]]
    return {
        "overall_auc": float(frame["pairwise_auc"].mean()),
        "counter_numeric_auc": float(opposite["pairwise_auc"].mean()),
        "same_orientation_auc": float(same["pairwise_auc"].mean()),
        "overall_top1_accuracy": float(frame["top1_same_outcome"].mean()),
        "counter_numeric_top1_accuracy": float(
            opposite["top1_same_outcome"].mean()
        ),
        "same_orientation_top1_accuracy": float(
            same["top1_same_outcome"].mean()
        ),
    }


def exact_null(
    similarity: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    families: np.ndarray,
    variants: np.ndarray,
    assignments: Sequence[Mapping[str, bool]],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    output = []
    for orientation in assignments:
        labels = labels_from_orientation(rows, orientation)
        frame = query_rows(
            similarity,
            rows,
            families,
            variants,
            labels,
            orientation,
        )
        output.append(summarize_queries(frame))
    names = list(output[0])
    matrix = np.asarray(
        [[row[name] for name in names] for row in output],
        dtype=float,
    )
    return matrix, output


def mechanism_pair_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["family_pair", "opposite_response_orientation"],
            as_index=False,
        )
        .agg(
            n_rankings=("pairwise_auc", "size"),
            pairwise_auc=("pairwise_auc", "mean"),
            top1_accuracy=("top1_same_outcome", "mean"),
        )
        .sort_values("family_pair")
    )


def pair_bootstrap(
    pair_frame: pd.DataFrame,
    value: str,
    rng: np.random.Generator,
    *,
    opposite_only: bool | None = None,
) -> list[float]:
    frame = pair_frame
    if opposite_only is not None:
        frame = frame[
            frame["opposite_response_orientation"] == opposite_only
        ]
    values = frame[value].to_numpy(dtype=float)
    indices = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(samples, (0.025, 0.975))]


def paired_method_bootstrap(
    pair_frame: pd.DataFrame,
    first: str,
    second: str,
    metric: str,
    rng: np.random.Generator,
    *,
    opposite_only: bool,
) -> dict[str, object]:
    frame = pair_frame[
        pair_frame["opposite_response_orientation"] == opposite_only
    ]
    pivot = frame.pivot(
        index="family_pair", columns="method", values=metric
    )
    delta = (pivot[first] - pivot[second]).to_numpy(dtype=float)
    indices = rng.integers(0, len(delta), size=(N_BOOT, len(delta)))
    samples = delta[indices].mean(axis=1)
    return {
        "subset": "counter_numeric" if opposite_only else "same_orientation",
        "mean_difference": float(np.mean(delta)),
        "pair_bootstrap_95": [
            float(x) for x in np.quantile(samples, (0.025, 0.975))
        ],
        "positive_pairs": int(np.sum(delta > 0)),
        "n_pairs": int(len(delta)),
        "pair_differences": {
            str(pair): float(value)
            for pair, value in zip(pivot.index, delta)
        },
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.15,
        1.04,
        letter,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def make_figure(
    fixed: pd.DataFrame,
    layer: pd.DataFrame,
    pairs: pd.DataFrame,
    orientation: Mapping[str, bool],
) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.8))

    order = [
        "jacobian",
        "direct",
        "raw",
        "word_tfidf",
        "char_tfidf",
        "numeric_direction",
    ]
    labels = [
        "Jacobian",
        "Direct",
        "Raw",
        "Word TF-IDF",
        "Char TF-IDF",
        "Numeric direction",
    ]
    primary = fixed.set_index("method").loc[order]
    x = np.arange(len(order))
    width = 0.36
    axes[0, 0].bar(
        x - width / 2,
        primary["overall_auc"],
        width,
        color=[COLORS[name] for name in order],
        alpha=0.93,
        label="all cross-mechanism cases",
    )
    axes[0, 0].bar(
        x + width / 2,
        primary["counter_numeric_auc"],
        width,
        color="white",
        edgecolor=[COLORS[name] for name in order],
        hatch="//",
        linewidth=1.2,
        label="matching outcome needs opposite numeric direction",
    )
    axes[0, 0].axhline(0.5, color="#8B8E91", linestyle="--", linewidth=0.9)
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("same-outcome candidate AUC")
    axes[0, 0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0, 0].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=6.8,
    )
    panel_label(axes[0, 0], "A")

    for method, label in (
        ("jacobian", "Jacobian"),
        ("direct", "Direct"),
        ("raw", "Raw"),
    ):
        rows = layer[layer["method"] == method].sort_values("depth_percent")
        axes[0, 1].plot(
            rows["depth_percent"],
            rows["counter_numeric_auc"],
            color=COLORS[method],
            linewidth=1.6,
            marker="o",
            markersize=2.8,
            label=label,
        )
    axes[0, 1].axvspan(
        PRIMARY_BAND[0], PRIMARY_BAND[1], color="#D9E7EA", alpha=0.45
    )
    axes[0, 1].axhline(0.5, color="#8B8E91", linestyle="--", linewidth=0.9)
    axes[0, 1].set_ylim(0.2, 0.85)
    axes[0, 1].set_xlabel("layer depth (%)")
    axes[0, 1].set_ylabel("counter-numeric candidate AUC")
    axes[0, 1].legend(frameon=False, loc="lower right")
    panel_label(axes[0, 1], "B")

    family_names = sorted(orientation)
    short = {
        "crosslink-density-modulus": "Crosslink",
        "dislocation-density-strength": "Dislocation",
        "obstacle-spacing-orowan": "Orowan",
        "particle-fraction-modulus": "Particles",
        "pearlite-spacing-strength": "Pearlite",
        "porosity-modulus": "Porosity",
    }
    j_pairs = pairs[pairs["method"] == "jacobian"].set_index("family_pair")
    matrix = np.full((6, 6), np.nan)
    for first_index, first in enumerate(family_names):
        for second_index, second in enumerate(family_names):
            if first_index == second_index:
                continue
            pair = " | ".join(sorted((first, second)))
            matrix[first_index, second_index] = float(
                j_pairs.loc[pair, "pairwise_auc"]
            )
    image = axes[1, 0].imshow(
        matrix,
        vmin=0.25,
        vmax=0.85,
        cmap="RdYlBu",
        aspect="equal",
    )
    for row_index in range(6):
        for column_index in range(6):
            if row_index == column_index:
                continue
            value = matrix[row_index, column_index]
            color = "white" if value < 0.38 or value > 0.76 else "#202124"
            axes[1, 0].text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.3,
                color=color,
            )
    orientation_marks = [
        "+" if orientation[name] else "-"
        for name in family_names
    ]
    tick_labels = [
        f"{short[name]} {mark}"
        for name, mark in zip(family_names, orientation_marks)
    ]
    axes[1, 0].set_xticks(
        range(6), tick_labels, rotation=38, ha="right"
    )
    axes[1, 0].set_yticks(range(6), tick_labels)
    colorbar = figure.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    colorbar.set_label("AUC", fontsize=7.5, labelpad=2)
    colorbar.ax.tick_params(labelsize=6.8)
    panel_label(axes[1, 0], "C")

    display = ["jacobian", "direct", "raw", "numeric_direction"]
    x2 = np.arange(len(display))
    same_values = [
        float(fixed.set_index("method").loc[name, "same_orientation_auc"])
        for name in display
    ]
    opposite_values = [
        float(fixed.set_index("method").loc[name, "counter_numeric_auc"])
        for name in display
    ]
    for index, method in enumerate(display):
        axes[1, 1].plot(
            [index - 0.12, index + 0.12],
            [same_values[index], opposite_values[index]],
            color=COLORS[method],
            linewidth=1.3,
            alpha=0.85,
        )
        axes[1, 1].scatter(
            index - 0.12,
            same_values[index],
            color=COLORS[method],
            marker="o",
            s=30,
            label="same response orientation" if index == 0 else None,
        )
        axes[1, 1].scatter(
            index + 0.12,
            opposite_values[index],
            facecolor="white",
            edgecolor=COLORS[method],
            marker="o",
            s=34,
            linewidth=1.3,
            label="opposite response orientation" if index == 0 else None,
        )
    axes[1, 1].axhline(0.5, color="#8B8E91", linestyle="--", linewidth=0.9)
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_ylabel("same-outcome candidate AUC")
    axes[1, 1].set_xticks(
        x2, ["Jacobian", "Direct", "Raw", "Numeric\ndirection"]
    )
    axes[1, 1].legend(frameon=False, loc="upper center")
    panel_label(axes[1, 1], "D")

    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.14,
        top=0.97,
        wspace=0.42,
        hspace=0.50,
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(
            FIG / f"cross-mechanism-outcome.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=PROTOCOL_PATH
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    fingerprints = validate_inputs(protocol)

    manifest_path = ROOT / protocol["inputs"]["prompt_manifest"]["path"]
    states_path = ROOT / protocol["inputs"]["representations"]["path"]
    manifest = json.loads(manifest_path.read_text())
    metadata = {
        str(row["prompt_id"]): row for row in manifest["prompts"]
    }
    with np.load(states_path, allow_pickle=False) as arrays:
        prompt_ids = arrays["prompt_ids"].astype(str)
        positions = arrays["positions"].astype(str)
        layers = arrays["layers"].astype(int)
        raw = arrays["raw_states"].astype(np.float64)
        direct = arrays["direct_decoder_basis"].astype(np.float64)
        jacobian = arrays["jacobian_decoder_basis"].astype(np.float64)
    if positions.tolist() != ["question_end"]:
        raise RuntimeError(f"unexpected positions: {positions.tolist()}")
    rows = [metadata[prompt_id] for prompt_id in prompt_ids]
    families = np.asarray([str(row["family_id"]) for row in rows])
    variants = np.asarray([str(row["variant"]) for row in rows])
    stems = [str(row["stem"]) for row in rows]
    depths = layers / 41.0 * 100.0
    band = (depths >= PRIMARY_BAND[0]) & (depths <= PRIMARY_BAND[1])

    observed = observed_orientation(rows)
    family_names = sorted(observed)
    assignments = orientation_assignments(family_names)
    observed_index = [
        orientation == observed for orientation in assignments
    ]
    if sum(observed_index) != 1:
        raise RuntimeError("observed orientation missing from exact null")
    labels = labels_from_orientation(rows, observed)
    expected_labels = np.asarray(
        [
            str(row["expected_outcome"]) == str(row["outcome_positive"])
            for row in rows
        ]
    )
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError("orientation-derived labels disagree with manifest")

    sim_j_seed = cosine_layers(jacobian[:, 0])
    sim_j = np.mean(sim_j_seed, axis=0)
    sim_direct = cosine_layers(direct[0][None, ...])[0]
    sim_raw = cosine_layers(raw[0][None, ...])[0]
    lexical = prompt_feature_matrices(stems)
    presentation = np.asarray(
        [str(row["presentation_order"]) for row in rows]
    )
    numeric = np.asarray([str(row["numeric_direction"]) for row in rows])
    similarities = {
        "jacobian": np.mean(sim_j[band], axis=0),
        "direct": np.mean(sim_direct[band], axis=0),
        "raw": np.mean(sim_raw[band], axis=0),
        "word_tfidf": lexical["word_tfidf"],
        "char_tfidf": lexical["char_tfidf"],
        "numeric_direction": deterministic_binary_similarity(numeric),
        "answer_order": deterministic_binary_similarity(presentation),
    }
    for seed in range(3):
        similarities[f"jacobian_seed{seed}"] = np.mean(
            sim_j_seed[seed, band], axis=0
        )

    fixed_rows = []
    query_frames = []
    pair_frames = []
    null_payload = {}
    for method, similarity in similarities.items():
        queries = query_rows(
            similarity,
            rows,
            families,
            variants,
            labels,
            observed,
        )
        summary = summarize_queries(queries)
        null_matrix, null_rows = exact_null(
            similarity,
            rows,
            families,
            variants,
            assignments,
        )
        metric_names = list(null_rows[0])
        null_lookup = {
            metric: null_matrix[:, index]
            for index, metric in enumerate(metric_names)
        }
        pair = mechanism_pair_metrics(queries)
        opposite_pairs = pair[pair["opposite_response_orientation"]]
        fixed_rows.append(
            {
                "method": method,
                **summary,
                "overall_auc_exact_p": exact_p(
                    null_lookup["overall_auc"], summary["overall_auc"]
                ),
                "counter_numeric_auc_exact_p": exact_p(
                    null_lookup["counter_numeric_auc"],
                    summary["counter_numeric_auc"],
                ),
                "overall_top1_exact_p": exact_p(
                    null_lookup["overall_top1_accuracy"],
                    summary["overall_top1_accuracy"],
                ),
                "counter_numeric_top1_exact_p": exact_p(
                    null_lookup["counter_numeric_top1_accuracy"],
                    summary["counter_numeric_top1_accuracy"],
                ),
                "positive_counter_numeric_pairs": int(
                    np.sum(opposite_pairs["pairwise_auc"] > 0.5)
                ),
                "n_counter_numeric_pairs": int(len(opposite_pairs)),
                "overall_auc_pair_bootstrap_95": pair_bootstrap(
                    pair,
                    "pairwise_auc",
                    np.random.default_rng(BOOTSTRAP_SEED + len(fixed_rows)),
                ),
                "counter_numeric_auc_pair_bootstrap_95": pair_bootstrap(
                    pair,
                    "pairwise_auc",
                    np.random.default_rng(
                        BOOTSTRAP_SEED + 100 + len(fixed_rows)
                    ),
                    opposite_only=True,
                ),
            }
        )
        query_copy = queries.copy()
        query_copy.insert(0, "method", method)
        query_frames.append(query_copy)
        pair.insert(0, "method", method)
        pair_frames.append(pair)
        null_payload[method] = {
            metric: values.tolist()
            for metric, values in null_lookup.items()
        }

    fixed = pd.DataFrame(fixed_rows)
    queries = pd.concat(query_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)

    layer_rows = []
    layer_nulls = {
        "jacobian": [],
        "direct": [],
        "raw": [],
    }
    for method, layered in (
        ("jacobian", sim_j),
        ("direct", sim_direct),
        ("raw", sim_raw),
    ):
        for index, depth in enumerate(depths):
            observed_queries = query_rows(
                layered[index],
                rows,
                families,
                variants,
                labels,
                observed,
            )
            summary = summarize_queries(observed_queries)
            null_matrix, null_rows = exact_null(
                layered[index],
                rows,
                families,
                variants,
                assignments,
            )
            names = list(null_rows[0])
            null_lookup = {
                name: null_matrix[:, metric_index]
                for metric_index, name in enumerate(names)
            }
            layer_nulls[method].append(null_lookup)
            layer_rows.append(
                {
                    "method": method,
                    "layer": int(layers[index]),
                    "depth_percent": float(depth),
                    **summary,
                }
            )
    layer = pd.DataFrame(layer_rows)

    layer_scan = {}
    for method, method_nulls in layer_nulls.items():
        observed_rows = layer[layer["method"] == method].sort_values(
            "depth_percent"
        )
        for metric in ("overall_auc", "counter_numeric_auc"):
            observed_values = observed_rows[metric].to_numpy(dtype=float)
            null = np.column_stack(
                [row[metric] for row in method_nulls]
            )
            best = int(np.argmax(observed_values))
            layer_scan[f"{method}_{metric}"] = {
                "best_value": float(observed_values[best]),
                "best_layer": int(layers[best]),
                "best_depth_percent": float(depths[best]),
                "max_layer_exact_p": exact_p(
                    np.max(null, axis=1), float(observed_values[best])
                ),
            }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contrasts = {}
    for opposite_only in (False, True):
        suffix = "counter_numeric" if opposite_only else "same_orientation"
        for second in ("direct", "raw", "word_tfidf", "char_tfidf"):
            contrasts[f"jacobian_minus_{second}_{suffix}"] = (
                paired_method_bootstrap(
                    pairs,
                    "jacobian",
                    second,
                    "pairwise_auc",
                    rng,
                    opposite_only=opposite_only,
                )
            )

    primary = fixed.set_index("method").loc["jacobian"]
    overall_pass = bool(
        primary["overall_auc"] > 0.5
        and primary["overall_auc_exact_p"] <= 0.05
    )
    counter_pass = bool(
        primary["counter_numeric_auc"] > 0.5
        and primary["counter_numeric_auc_exact_p"] <= 0.05
    )
    breadth_pass = bool(primary["positive_counter_numeric_pairs"] >= 6)
    if overall_pass and counter_pass and breadth_pass:
        verdict = "strong"
    elif overall_pass and not counter_pass:
        verdict = "numeric_compatible"
    elif counter_pass or (overall_pass and not breadth_pass):
        verdict = "partial"
    else:
        verdict = "none"
    jacobian_specific = bool(
        contrasts["jacobian_minus_direct_counter_numeric"][
            "pair_bootstrap_95"
        ][0]
        > 0
    )

    fixed.to_csv(OUT / "fixed_statistics.csv", index=False)
    queries.to_csv(OUT / "all_query_rankings.csv", index=False)
    pairs.to_csv(OUT / "mechanism_pair_metrics.csv", index=False)
    layer.to_csv(OUT / "layer_metrics.csv", index=False)
    (OUT / "exact_orientation_nulls.json").write_text(
        json.dumps(json_safe(null_payload), indent=2) + "\n"
    )

    payload = {
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "input_fingerprints": fingerprints,
        "observed_response_orientation": observed,
        "n_exact_assignments": len(assignments),
        "primary": {
            key: json_safe(value) for key, value in primary.to_dict().items()
        },
        "frozen_verdict": {
            "cross_mechanism_evidence": verdict,
            "overall_pass": overall_pass,
            "counter_numeric_pass": counter_pass,
            "pair_breadth_pass": breadth_pass,
            "jacobian_specific": jacobian_specific,
        },
        "paired_mechanism_pair_contrasts": contrasts,
        "layer_scan": layer_scan,
        "all_fixed_results": fixed.to_dict(orient="records"),
        "guardrail": (
            "The endpoint was frozen after the within-mechanism natural "
            "question-end result was inspected. It is cross-mechanism "
            "observational geometry, not an independent replication or causal test."
        ),
    }
    (OUT / "statistics.json").write_text(
        json.dumps(json_safe(payload), indent=2) + "\n"
    )
    make_figure(fixed, layer, pairs, observed)

    (OUT / "RESULTS.md").write_text(
        "\n".join(
            [
                "# Cross-mechanism physical-outcome analysis",
                "",
                f"Frozen interpretation: **{verdict.upper()}** evidence.",
                "",
                "## Primary results",
                "",
                (
                    f"- All cross-mechanism rankings: AUC "
                    f"**{float(primary['overall_auc']):.3f}**, exact "
                    f"orientation-null `p={float(primary['overall_auc_exact_p']):.3f}`."
                ),
                (
                    f"- Counter-numeric rankings, where the same physical "
                    f"outcome requires opposite numerical directions: AUC "
                    f"**{float(primary['counter_numeric_auc']):.3f}**, exact "
                    f"`p={float(primary['counter_numeric_auc_exact_p']):.3f}`."
                ),
                (
                    f"- Pair breadth: "
                    f"{int(primary['positive_counter_numeric_pairs'])}/"
                    f"{int(primary['n_counter_numeric_pairs'])} opposite-response "
                    "mechanism pairs have AUC above 0.5."
                ),
                (
                    "- Jacobian-specific: "
                    f"**{'yes' if jacobian_specific else 'no'}** under the "
                    "frozen paired mechanism-pair interval rule."
                ),
                "",
                "The exact null contains only 20 balanced mechanism-orientation",
                "assignments, so 0.05 is the smallest attainable exact p-value.",
                "",
                "## Interpretation",
                "",
                (
                    "The counter-numeric endpoint cannot be passed by merely "
                    "grouping prompts whose numbers both increase or both "
                    "decrease. A positive result requires matching physical "
                    "consequences across mechanisms with opposite input trends."
                ),
                "",
                (
                    "This remains a post-hoc analysis of the already inspected "
                    "natural question-end cohort. It is observational, not causal."
                ),
                "",
                "## Complete artifacts",
                "",
                "- `PROTOCOL.md` and `protocol.json`.",
                "- `fixed_statistics.csv`.",
                "- `all_query_rankings.csv` (all 1,080 queries per method).",
                "- `mechanism_pair_metrics.csv` (all 15 pairs per method).",
                "- `layer_metrics.csv` (all 25 layers).",
                "- `exact_orientation_nulls.json` (all 20 assignments).",
                "- `statistics.json`.",
                "- `figures/cross-mechanism-outcome.{pdf,png,svg}`.",
                "",
                "## Reproduction",
                "",
                "```bash",
                "python scripts/analyze_cross_mechanism_outcome.py",
                "```",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
