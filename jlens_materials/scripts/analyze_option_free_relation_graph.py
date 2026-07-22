#!/usr/bin/env python3
"""Analyze signed-relation topology before any answer scaffold is visible.

This script implements the frozen protocol in
``experiments/option-free-relation-graph-2026-07-17``.  It reads archived
states from the answer-code study, validates every registered input
fingerprint, retains all prompts and layers, and writes row-level results,
machine-readable statistics, and publication-quality figures.
"""

from __future__ import annotations

import argparse
import hashlib
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
    case_preserving_exact_permutations,
    cosine_layers,
    deterministic_binary_similarity,
    edge_precision,
    per_query_precision,
    prompt_feature_matrices,
    relation_edges,
    relation_ranking_summary,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "option-free-relation-graph-2026-07-17"
FIG = OUT / "figures"
PROTOCOL_PATH = OUT / "protocol.json"
DEFAULT_STATES = (
    ROOT
    / "experiments"
    / "answer-code-binding-2026-07-17"
    / "representations.npz"
)
DEFAULT_MANIFEST = (
    ROOT
    / "experiments"
    / "answer-code-binding-2026-07-17"
    / "prompt_manifest.json"
)

PRIMARY_BAND = (38.0, 92.0)
LATE_BAND = (80.0, 96.0)
BOOTSTRAP_SEED = 20260718
N_BOOTSTRAP = 30_000

COLORS = {
    "jacobian": "#16697A",
    "direct": "#B56576",
    "raw": "#6C5B7B",
    "word_tfidf": "#5B8C5A",
    "char_tfidf": "#D17C38",
    "answer_order": "#A0A4A8",
    "numeric_oracle": "#2F4858",
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
    """Upper-tail exact probability; the observed assignment is in the null."""

    return float(np.mean(null >= observed - 1e-15))


def validate_inputs(
    protocol: Mapping[str, object],
    states_path: Path,
    manifest_path: Path,
) -> dict[str, str]:
    expected = protocol["inputs"]  # type: ignore[index]
    actual = {
        "representations": sha256(states_path),
        "prompt_manifest": sha256(manifest_path),
        "source_protocol": sha256(
            ROOT / str(expected["source_protocol"]["path"])  # type: ignore[index]
        ),
        "reused_graph_implementation": sha256(
            ROOT
            / str(expected["reused_graph_implementation"]["path"])  # type: ignore[index]
        ),
    }
    for name, digest in actual.items():
        registered = str(expected[name]["sha256"])  # type: ignore[index]
        if digest != registered:
            raise RuntimeError(
                f"input fingerprint mismatch for {name}: {digest} != {registered}"
            )
    return actual


def outcome_binary(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray(
        [
            1
            if str(row["expected_outcome"]) == str(row["outcome_positive"])
            else 0
            for row in rows
        ],
        dtype=np.int8,
    )


def relation_candidate_auc_null(
    similarity: np.ndarray,
    families: np.ndarray,
    variants: np.ndarray,
    triplets: np.ndarray,
    permutations: np.ndarray,
) -> np.ndarray:
    """Exact-null mean pairwise AUC across all eligible candidate rankings."""

    accum = np.zeros(len(permutations), dtype=np.float64)
    n_queries = 0
    for source in range(len(families)):
        for target_variant in sorted(set(variants)):
            if target_variant == variants[source]:
                continue
            candidates = np.flatnonzero(
                (families == families[source])
                & (variants == target_variant)
                & (triplets != triplets[source])
            )
            if len(candidates) != 3:
                raise RuntimeError("each relation ranking must have three candidates")
            scores = similarity[source, candidates]
            same = (
                permutations[:, candidates]
                == permutations[:, source, None]
            )
            query_auc = np.zeros(len(permutations), dtype=np.float64)
            comparisons = 0
            for first in range(3):
                for second in range(3):
                    if first == second:
                        continue
                    orientation = same[:, first] & ~same[:, second]
                    if scores[first] > scores[second]:
                        query_auc += orientation
                    elif scores[first] == scores[second]:
                        query_auc += 0.5 * orientation
                    comparisons += 1
            # Every balanced case-preserving assignment has two positive-
            # negative comparisons among the three eligible candidates.
            denominator = np.sum(same, axis=1) * np.sum(~same, axis=1)
            if not np.all(denominator == 2):
                raise RuntimeError(
                    "structured null changed candidate AUC denominator"
                )
            accum += query_auc / denominator
            n_queries += 1
    return accum / n_queries


def layer_auc(
    similarity: np.ndarray,
    families: np.ndarray,
    variants: np.ndarray,
    triplets: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    summary, _ = relation_ranking_summary(
        similarity, families, variants, triplets, outcomes
    )
    return float(summary["pairwise_auc"])


def family_metric_frame(
    method: str,
    position: str,
    band_name: str,
    edges: np.ndarray,
    ranking_frame: pd.DataFrame,
    outcomes: np.ndarray,
    families: np.ndarray,
) -> pd.DataFrame:
    query_precision = per_query_precision(edges, outcomes)
    precision = pd.DataFrame(
        {
            "family": families,
            "graph_precision": query_precision,
        }
    ).groupby("family", as_index=False)["graph_precision"].mean()
    auc = (
        ranking_frame.groupby("family", as_index=False)["pairwise_auc"]
        .mean()
        .rename(columns={"pairwise_auc": "candidate_auc"})
    )
    output = precision.merge(auc, on="family", validate="one_to_one")
    output.insert(0, "band", band_name)
    output.insert(0, "position", position)
    output.insert(0, "method", method)
    return output


def two_stage_bootstrap(
    values: pd.DataFrame,
    value_column: str,
    rng: np.random.Generator,
    *,
    n: int = N_BOOTSTRAP,
) -> list[float]:
    """Resample families, then rows within each sampled family."""

    by_family = {
        str(family): rows[value_column].to_numpy(dtype=float)
        for family, rows in values.groupby("family")
    }
    family_names = sorted(by_family)
    samples = np.empty(n, dtype=float)
    for index in range(n):
        chosen = rng.choice(family_names, size=len(family_names), replace=True)
        family_values = []
        for family in chosen:
            rows = by_family[str(family)]
            family_values.append(
                float(np.mean(rng.choice(rows, size=len(rows), replace=True)))
            )
        samples[index] = float(np.mean(family_values))
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def paired_family_bootstrap(
    family_frame: pd.DataFrame,
    first: str,
    second: str,
    metric: str,
    rng: np.random.Generator,
    *,
    n: int = N_BOOTSTRAP,
) -> dict[str, object]:
    pivot = family_frame.pivot(index="family", columns="method", values=metric)
    delta = (pivot[first] - pivot[second]).to_numpy(dtype=float)
    indices = rng.integers(0, len(delta), size=(n, len(delta)))
    null = delta[indices].mean(axis=1)
    return {
        "mean_difference": float(np.mean(delta)),
        "family_bootstrap_95": [
            float(value) for value in np.quantile(null, (0.025, 0.975))
        ],
        "positive_families": int(np.sum(delta > 0)),
        "family_differences": {
            str(family): float(value)
            for family, value in zip(pivot.index, delta)
        },
    }


def edge_rows(
    method: str,
    position: str,
    band_name: str,
    similarity: np.ndarray,
    edges: np.ndarray,
    rows: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    output = []
    for source, target in edges:
        first = rows[int(source)]
        second = rows[int(target)]
        output.append(
            {
                "position": position,
                "band": band_name,
                "method": method,
                "source_index": int(source),
                "target_index": int(target),
                "source_prompt_id": str(first["prompt_id"]),
                "target_prompt_id": str(second["prompt_id"]),
                "family": str(first["family_id"]),
                "source_case": str(first["case_id"]),
                "target_case": str(second["case_id"]),
                "source_variant": str(first["variant"]),
                "target_variant": str(second["variant"]),
                "source_outcome": str(first["expected_outcome"]),
                "target_outcome": str(second["expected_outcome"]),
                "same_outcome": bool(
                    first["expected_outcome"] == second["expected_outcome"]
                ),
                "same_future_answer_order": bool(
                    first["presentation_order"] == second["presentation_order"]
                ),
                "same_numeric_direction": bool(
                    first["numeric_direction"] == second["numeric_direction"]
                ),
                "cosine": float(similarity[int(source), int(target)]),
            }
        )
    return pd.DataFrame(output)


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
    fixed_frame: pd.DataFrame,
    family_frame: pd.DataFrame,
    layer_frame: pd.DataFrame,
    exact_null: np.ndarray,
) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.75))

    primary = fixed_frame[
        (fixed_frame["position"] == "checkpoint")
        & (fixed_frame["band"] == "primary")
        & fixed_frame["method"].isin(
            ["jacobian", "direct", "raw", "word_tfidf", "char_tfidf"]
        )
    ].copy()
    order = ["jacobian", "direct", "raw", "word_tfidf", "char_tfidf"]
    labels = ["Jacobian", "Direct", "Raw", "Word TF-IDF", "Char TF-IDF"]
    x = np.arange(len(order))
    precision = [
        float(primary.loc[primary["method"] == method, "graph_precision"].iloc[0])
        for method in order
    ]
    auc = [
        float(primary.loc[primary["method"] == method, "candidate_auc"].iloc[0])
        for method in order
    ]
    width = 0.36
    axes[0, 0].bar(
        x - width / 2,
        precision,
        width,
        color=[COLORS[method] for method in order],
        alpha=0.93,
        label="selected-edge precision",
    )
    axes[0, 0].bar(
        x + width / 2,
        auc,
        width,
        color="white",
        edgecolor=[COLORS[method] for method in order],
        linewidth=1.2,
        hatch="//",
        label="all-candidate AUC",
    )
    axes[0, 0].axhline(
        float(np.mean(exact_null)),
        color="#8B8E91",
        linestyle="--",
        linewidth=0.9,
    )
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("same-direction score")
    axes[0, 0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0, 0].legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.33),
    )
    panel_label(axes[0, 0], "A")

    checkpoint_layers = layer_frame[layer_frame["position"] == "checkpoint"]
    for method, label in (
        ("jacobian", "Jacobian"),
        ("direct", "Direct"),
        ("raw", "Raw"),
    ):
        rows_method = checkpoint_layers[
            checkpoint_layers["method"] == method
        ].sort_values("depth_percent")
        axes[0, 1].plot(
            rows_method["depth_percent"],
            rows_method["graph_precision"],
            color=COLORS[method],
            linewidth=1.6,
            marker="o",
            markersize=2.8,
            label=label,
        )
    axes[0, 1].axvspan(
        PRIMARY_BAND[0], PRIMARY_BAND[1], color="#D9E7EA", alpha=0.45
    )
    axes[0, 1].axhline(
        float(np.mean(exact_null)),
        color="#8B8E91",
        linestyle="--",
        linewidth=0.9,
        label="structured-null mean",
    )
    axes[0, 1].set_ylim(0.3, 1.02)
    axes[0, 1].set_xlabel("layer depth (%)")
    axes[0, 1].set_ylabel("selected-edge precision")
    axes[0, 1].legend(frameon=False, ncol=2, loc="lower right")
    panel_label(axes[0, 1], "B")

    family = family_frame[
        (family_frame["position"] == "checkpoint")
        & (family_frame["band"] == "primary")
        & family_frame["method"].isin(["jacobian", "direct", "raw"])
    ].copy()
    family_names = sorted(family["family"].unique())
    family_labels = [
        {
            "crosslink-density-modulus": "Crosslink",
            "dislocation-density-strength": "Dislocation",
            "obstacle-spacing-orowan": "Orowan",
            "particle-fraction-modulus": "Particles",
            "pearlite-spacing-strength": "Pearlite",
            "porosity-modulus": "Porosity",
        }.get(name, name)
        for name in family_names
    ]
    offsets = {"jacobian": -0.18, "direct": 0.0, "raw": 0.18}
    for method, label in (
        ("jacobian", "Jacobian"),
        ("direct", "Direct"),
        ("raw", "Raw"),
    ):
        values = (
            family[family["method"] == method]
            .set_index("family")
            .loc[family_names, "candidate_auc"]
            .to_numpy()
        )
        axes[1, 0].scatter(
            np.arange(len(family_names)) + offsets[method],
            values,
            s=28,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            label=label,
            zorder=3,
        )
    axes[1, 0].axhline(0.5, color="#8B8E91", linestyle="--", linewidth=0.9)
    axes[1, 0].set_ylim(0.2, 1.02)
    axes[1, 0].set_ylabel("all-candidate AUC")
    axes[1, 0].set_xticks(
        np.arange(len(family_names)), family_labels, rotation=25, ha="right"
    )
    axes[1, 0].legend(frameon=False, ncol=3, loc="lower center")
    panel_label(axes[1, 0], "C")

    scaffold = fixed_frame[
        (fixed_frame["band"] == "primary")
        & fixed_frame["method"].isin(["jacobian", "direct", "raw"])
    ]
    for method, label in (
        ("jacobian", "Jacobian"),
        ("direct", "Direct"),
        ("raw", "Raw"),
    ):
        method_rows = scaffold[scaffold["method"] == method].set_index(
            "position"
        )
        values = method_rows.loc[
            ["checkpoint", "final_prompt"], "candidate_auc"
        ].to_numpy(dtype=float)
        axes[1, 1].plot(
            [0, 1],
            values,
            color=COLORS[method],
            linewidth=1.5,
            marker="o",
            markersize=5,
            label=label,
        )
    axes[1, 1].axhline(0.5, color="#8B8E91", linestyle="--", linewidth=0.9)
    axes[1, 1].set_xlim(-0.15, 1.15)
    axes[1, 1].set_ylim(0.2, 1.02)
    axes[1, 1].set_xticks(
        [0, 1], ["before answer\nmapping", "after answer\nmapping"]
    )
    axes[1, 1].set_ylabel("all-candidate AUC")
    axes[1, 1].legend(frameon=False, loc="lower right")
    panel_label(axes[1, 1], "D")

    figure.subplots_adjust(
        left=0.09, right=0.985, bottom=0.13, top=0.97, wspace=0.32, hspace=0.50
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(
            FIG / f"option-free-relation-graph.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL_PATH.read_text())
    fingerprints = validate_inputs(protocol, args.states, args.manifest)
    manifest = json.loads(args.manifest.read_text())
    metadata = {
        str(row["prompt_id"]): row for row in manifest["prompts"]
    }
    with np.load(args.states, allow_pickle=False) as arrays:
        prompt_ids = arrays["prompt_ids"].astype(str)
        positions = arrays["positions"].astype(str)
        layers = arrays["layers"].astype(int)
        raw = arrays["raw_states"].astype(np.float64)
        direct = arrays["direct_decoder_basis"].astype(np.float64)
        jacobian = arrays["jacobian_decoder_basis"].astype(np.float64)
    if list(positions) != ["checkpoint", "final_prompt"]:
        raise RuntimeError(f"unexpected position order: {positions.tolist()}")
    rows = [metadata[str(prompt_id)] for prompt_id in prompt_ids]
    if len(rows) != 72 or len(set(prompt_ids)) != 72:
        raise RuntimeError("expected 72 unique prompt ids")
    families = np.asarray([str(row["family_id"]) for row in rows])
    variants = np.asarray([str(row["variant"]) for row in rows])
    triplets = np.asarray([str(row["triplet_id"]) for row in rows])
    outcomes = np.asarray([str(row["expected_outcome"]) for row in rows])
    binary = outcome_binary(rows)
    presentation_orders = np.asarray(
        [str(row["presentation_order"]) for row in rows]
    )
    numeric_directions = np.asarray(
        [str(row["numeric_direction"]) for row in rows]
    )
    stems = [str(row["stem"]) for row in rows]
    depths = layers / 41.0 * 100.0
    primary_mask = (depths >= PRIMARY_BAND[0]) & (
        depths <= PRIMARY_BAND[1]
    )
    late_mask = (depths >= LATE_BAND[0]) & (depths <= LATE_BAND[1])

    exact_assignments = case_preserving_exact_permutations(
        families, variants, triplets, outcomes
    )
    if exact_assignments.shape != (46_656, 72):
        raise RuntimeError(
            f"unexpected exact-null shape: {exact_assignments.shape}"
        )
    observed_in_null = np.any(
        np.all(exact_assignments == binary[None, :], axis=1)
    )
    if not observed_in_null:
        raise RuntimeError("observed assignment is absent from exact null")

    lexical = prompt_feature_matrices(stems)
    similarities: dict[str, dict[str, np.ndarray]] = {}
    layer_similarities: dict[str, dict[str, np.ndarray]] = {}
    individual_seed: dict[str, np.ndarray] = {}
    for position_index, position in enumerate(positions):
        sim_j_seed = cosine_layers(jacobian[:, position_index])
        sim_j = np.mean(sim_j_seed, axis=0)
        sim_direct = cosine_layers(direct[position_index][None, ...])[0]
        sim_raw = cosine_layers(raw[position_index][None, ...])[0]
        layer_similarities[str(position)] = {
            "jacobian": sim_j,
            "direct": sim_direct,
            "raw": sim_raw,
        }
        similarities[str(position)] = {
            "jacobian_primary": np.mean(sim_j[primary_mask], axis=0),
            "direct_primary": np.mean(sim_direct[primary_mask], axis=0),
            "raw_primary": np.mean(sim_raw[primary_mask], axis=0),
            "jacobian_late": np.mean(sim_j[late_mask], axis=0),
            "direct_late": np.mean(sim_direct[late_mask], axis=0),
            "raw_late": np.mean(sim_raw[late_mask], axis=0),
        }
        if position == "checkpoint":
            for seed in range(sim_j_seed.shape[0]):
                individual_seed[f"jacobian_seed{seed}_primary"] = np.mean(
                    sim_j_seed[seed, primary_mask], axis=0
                )
    similarities["checkpoint"].update(individual_seed)
    similarities["checkpoint"].update(
        {
            "word_tfidf_primary": lexical["word_tfidf"],
            "char_tfidf_primary": lexical["char_tfidf"],
            "answer_order_primary": deterministic_binary_similarity(
                presentation_orders
            ),
            "numeric_oracle_primary": deterministic_binary_similarity(
                numeric_directions
            ),
        }
    )

    fixed_rows = []
    edge_frames = []
    ranking_frames = []
    family_frames = []
    fixed_cache: dict[tuple[str, str, str], dict[str, object]] = {}
    primary_exact_null = None
    primary_auc_null = None
    for position, methods in similarities.items():
        for compound_name, similarity in methods.items():
            if compound_name.endswith("_primary"):
                band_name = "primary"
                method = compound_name[: -len("_primary")]
            elif compound_name.endswith("_late"):
                band_name = "late"
                method = compound_name[: -len("_late")]
            else:
                raise RuntimeError(f"unparsed method name: {compound_name}")
            edges = relation_edges(
                similarity, families, variants, triplets
            )
            precision = edge_precision(edges, outcomes)
            precision_null = np.mean(
                exact_assignments[:, edges[:, 0]]
                == exact_assignments[:, edges[:, 1]],
                axis=1,
            )
            ranking, ranking_frame = relation_ranking_summary(
                similarity, families, variants, triplets, outcomes
            )
            auc = float(ranking["pairwise_auc"])
            auc_null = relation_candidate_auc_null(
                similarity,
                families,
                variants,
                triplets,
                exact_assignments,
            )
            family_frame = family_metric_frame(
                method,
                position,
                band_name,
                edges,
                ranking_frame,
                outcomes,
                families,
            )
            family_frames.append(family_frame)
            fixed_rows.append(
                {
                    "position": position,
                    "band": band_name,
                    "method": method,
                    "graph_precision": precision,
                    "graph_exact_p": exact_p(precision_null, precision),
                    "graph_null_mean": float(np.mean(precision_null)),
                    "graph_null_q025": float(
                        np.quantile(precision_null, 0.025)
                    ),
                    "graph_null_q975": float(
                        np.quantile(precision_null, 0.975)
                    ),
                    "candidate_auc": auc,
                    "candidate_auc_exact_p": exact_p(auc_null, auc),
                    "candidate_auc_null_mean": float(np.mean(auc_null)),
                    "candidate_auc_null_q025": float(
                        np.quantile(auc_null, 0.025)
                    ),
                    "candidate_auc_null_q975": float(
                        np.quantile(auc_null, 0.975)
                    ),
                    "positive_auc_families": int(
                        np.sum(family_frame["candidate_auc"] > 0.5)
                    ),
                    "n_edges": int(len(edges)),
                    "n_rankings": int(ranking["n_rankings"]),
                }
            )
            edge_frames.append(
                edge_rows(
                    method,
                    position,
                    band_name,
                    similarity,
                    edges,
                    rows,
                )
            )
            rank_copy = ranking_frame.copy()
            rank_copy.insert(0, "band", band_name)
            rank_copy.insert(0, "position", position)
            rank_copy.insert(0, "method", method)
            ranking_frames.append(rank_copy)
            fixed_cache[(position, band_name, method)] = {
                "edges": edges,
                "precision_null": precision_null,
                "auc_null": auc_null,
                "ranking": ranking,
            }
            if (
                position == "checkpoint"
                and band_name == "primary"
                and method == "jacobian"
            ):
                primary_exact_null = precision_null
                primary_auc_null = auc_null

    fixed_frame = pd.DataFrame(fixed_rows).sort_values(
        ["position", "band", "method"]
    )
    edge_frame = pd.concat(edge_frames, ignore_index=True)
    ranking_frame = pd.concat(ranking_frames, ignore_index=True)
    family_frame = pd.concat(family_frames, ignore_index=True)

    layer_rows = []
    layer_edges: dict[str, list[np.ndarray]] = {
        "jacobian": [],
        "direct": [],
        "raw": [],
    }
    for position, methods in layer_similarities.items():
        for method, layered in methods.items():
            for layer_index, depth in enumerate(depths):
                similarity = layered[layer_index]
                edges = relation_edges(
                    similarity, families, variants, triplets
                )
                ranking, _ = relation_ranking_summary(
                    similarity,
                    families,
                    variants,
                    triplets,
                    outcomes,
                )
                layer_rows.append(
                    {
                        "position": position,
                        "method": method,
                        "layer": int(layers[layer_index]),
                        "depth_percent": float(depth),
                        "graph_precision": edge_precision(edges, outcomes),
                        "candidate_auc": float(ranking["pairwise_auc"]),
                    }
                )
                if position == "checkpoint":
                    layer_edges[method].append(edges)
    layer_frame = pd.DataFrame(layer_rows)

    layer_scan = {}
    for method, edges_by_layer in layer_edges.items():
        observed = (
            layer_frame[
                (layer_frame["position"] == "checkpoint")
                & (layer_frame["method"] == method)
            ]
            .sort_values("depth_percent")["graph_precision"]
            .to_numpy()
        )
        null = np.empty(
            (len(exact_assignments), len(edges_by_layer)), dtype=np.float32
        )
        for layer_index, edges in enumerate(edges_by_layer):
            null[:, layer_index] = np.mean(
                exact_assignments[:, edges[:, 0]]
                == exact_assignments[:, edges[:, 1]],
                axis=1,
            )
        best = int(np.argmax(observed))
        layer_scan[method] = {
            "best_graph_precision": float(observed[best]),
            "best_layer": int(layers[best]),
            "best_depth_percent": float(depths[best]),
            "max_layer_exact_p": exact_p(
                np.max(null, axis=1), float(observed[best])
            ),
        }

    primary_rows = family_frame[
        (family_frame["position"] == "checkpoint")
        & (family_frame["band"] == "primary")
        & family_frame["method"].isin(["jacobian", "direct", "raw"])
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contrasts = {}
    for metric in ("graph_precision", "candidate_auc"):
        contrasts[f"jacobian_minus_direct_{metric}"] = paired_family_bootstrap(
            primary_rows,
            "jacobian",
            "direct",
            metric,
            rng,
        )
        contrasts[f"jacobian_minus_raw_{metric}"] = paired_family_bootstrap(
            primary_rows,
            "jacobian",
            "raw",
            metric,
            rng,
        )

    primary = fixed_frame[
        (fixed_frame["position"] == "checkpoint")
        & (fixed_frame["band"] == "primary")
        & (fixed_frame["method"] == "jacobian")
    ].iloc[0]
    graph_pass = bool(primary["graph_exact_p"] <= 0.05)
    auc_pass = bool(
        primary["candidate_auc_exact_p"] <= 0.05
        and primary["candidate_auc"] > 0.5
    )
    breadth_pass = bool(primary["positive_auc_families"] >= 4)
    if graph_pass and auc_pass and breadth_pass:
        verdict = "strong"
    elif graph_pass or auc_pass:
        verdict = "partial"
    else:
        verdict = "none"
    direct_auc_ci = contrasts["jacobian_minus_direct_candidate_auc"][
        "family_bootstrap_95"
    ]
    jacobian_specific = bool(direct_auc_ci[0] > 0)

    prompt_inventory = pd.DataFrame(
        [
            {
                "array_index": index,
                "prompt_id": str(row["prompt_id"]),
                "family_id": str(row["family_id"]),
                "family_name": str(row["family_name"]),
                "case_id": str(row["case_id"]),
                "variant": str(row["variant"]),
                "expected_outcome": str(row["expected_outcome"]),
                "outcome_positive": str(row["outcome_positive"]),
                "outcome_negative": str(row["outcome_negative"]),
                "numeric_direction": str(row["numeric_direction"]),
                "future_presentation_order": str(row["presentation_order"]),
                "option_free_stem": str(row["stem"]),
                "complete_answer_code_prompt": str(row["user"]),
            }
            for index, row in enumerate(rows)
        ]
    )

    fixed_frame.to_csv(OUT / "fixed_graph_statistics.csv", index=False)
    edge_frame.to_csv(OUT / "all_selected_edges.csv", index=False)
    ranking_frame.to_csv(OUT / "all_candidate_rankings.csv", index=False)
    family_frame.to_csv(OUT / "family_metrics.csv", index=False)
    layer_frame.to_csv(OUT / "layer_metrics.csv", index=False)
    prompt_inventory.to_csv(OUT / "prompt_inventory.csv", index=False)
    np.savez_compressed(
        OUT / "primary_exact_nulls.npz",
        structured_assignments=exact_assignments,
        graph_precision=primary_exact_null,
        candidate_auc=primary_auc_null,
    )

    payload = {
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "input_fingerprints": fingerprints,
        "array_validation": {
            "n_prompts": int(len(rows)),
            "n_families": int(len(set(families))),
            "n_triplets": int(len(set(triplets))),
            "positions": positions.tolist(),
            "layers": layers.tolist(),
            "depth_percent": [float(value) for value in depths],
            "observed_assignment_present_in_exact_null": bool(observed_in_null),
            "n_exact_assignments": int(len(exact_assignments)),
        },
        "primary": {
            key: json_safe(value)
            for key, value in primary.to_dict().items()
        },
        "frozen_verdict": {
            "option_free_evidence": verdict,
            "graph_precision_pass": graph_pass,
            "candidate_auc_pass": auc_pass,
            "family_breadth_pass": breadth_pass,
            "jacobian_specific": jacobian_specific,
        },
        "paired_family_contrasts": contrasts,
        "layer_scan": layer_scan,
        "all_fixed_results": fixed_frame.to_dict(orient="records"),
        "guardrail": (
            "Checkpoint states precede all answer words and mappings. Positive "
            "topology is option-free within supplied mechanism families, not "
            "a complete ontology, literal chain of thought, or causal result."
        ),
    }
    (OUT / "statistics.json").write_text(
        json.dumps(json_safe(payload), indent=2) + "\n"
    )

    make_figure(
        fixed_frame,
        family_frame,
        layer_frame,
        np.asarray(primary_exact_null),
    )

    primary_dict = payload["primary"]
    readme = [
        "# Option-free signed-relation graph",
        "",
        f"Frozen interpretation: **{verdict.upper()}** option-free evidence.",
        "",
        "## Primary checkpoint result",
        "",
        (
            f"- Selected-edge precision: "
            f"{100 * float(primary_dict['graph_precision']):.1f}% "
            f"(exact structured-null "
            f"`p={float(primary_dict['graph_exact_p']):.6f}`)."
        ),
        (
            f"- Full-candidate AUC: "
            f"{float(primary_dict['candidate_auc']):.3f} "
            f"(exact structured-null "
            f"`p={float(primary_dict['candidate_auc_exact_p']):.6f}`)."
        ),
        (
            f"- Family breadth: "
            f"{int(primary_dict['positive_auc_families'])}/6 families "
            "have AUC above 0.5."
        ),
        (
            "- Jacobian-specific: "
            f"**{'yes' if jacobian_specific else 'no'}** under the frozen "
            "paired-family interval rule."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The checkpoint state has seen the complete scientific question "
            "and a common marker, but it cannot attend to the future answer "
            "words, A/B mapping, or response instruction. The result therefore "
            "tests signed physical organization before the answer scaffold."
        ),
        "",
        "This remains a post-hoc analysis of archived states. It is not causal.",
        "",
        "## Complete artifacts",
        "",
        "- `PROTOCOL.md` and `protocol.json`: frozen analysis and decision rule.",
        "- `prompt_inventory.csv`: all 72 exact scientific stems and full prompts.",
        "- `fixed_graph_statistics.csv`: every position, band, and method.",
        "- `all_selected_edges.csv`: every selected edge.",
        "- `all_candidate_rankings.csv`: every eligible candidate-ranking query.",
        "- `family_metrics.csv`: mechanism-level precision and AUC.",
        "- `layer_metrics.csv`: all 25 layers at both positions.",
        "- `primary_exact_nulls.npz`: all 46,656 structured assignments and null statistics.",
        "- `statistics.json`: machine-readable decision record.",
        "- `figures/option-free-relation-graph.{pdf,png,svg}`.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/analyze_option_free_relation_graph.py",
        "```",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(readme))


if __name__ == "__main__":
    main()
