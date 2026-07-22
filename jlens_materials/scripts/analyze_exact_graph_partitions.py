#!/usr/bin/env python3
"""Exact balanced graph partition and held-out-surface transfer."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from analyze_graph_isomorphism_generalization import (
    MANIFEST,
    METHODS,
    STATES,
    VARIANTS,
    load_data,
    plus_one_signflip_p,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "experiments"
    / "graph-isomorphism-generalization-2026-07-18"
)
SPECTRAL_PROTOCOL = OUT / "spectral_protocol.json"
PARTITION_PROTOCOL = OUT / "partition_protocol.json"
PARTITION_PROTOCOL_MD = OUT / "PARTITION_PROTOCOL.md"
SEED = 20260718
N_NULL = 10_000
N_BOOTSTRAP = 1_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_json(value: object) -> object:
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def validate_inputs() -> None:
    protocol = json.loads(PARTITION_PROTOCOL.read_text())
    spectral = json.loads(SPECTRAL_PROTOCOL.read_text())
    if sha256(SPECTRAL_PROTOCOL) != protocol[
        "parent_spectral_protocol_sha256"
    ]:
        raise RuntimeError("spectral protocol fingerprint mismatch")
    gauge = json.loads((OUT / "gauge_protocol.json").read_text())
    if (
        sha256(STATES)
        != gauge["inputs"]["representations_sha256"]
        or sha256(MANIFEST)
        != gauge["inputs"]["prompt_manifest_sha256"]
    ):
        raise RuntimeError("frozen input fingerprint mismatch")
    if spectral["study_id"] != "spectral-gauge-community-2026-07-18":
        raise RuntimeError("unexpected spectral parent")


def balanced_partitions(n_nodes: int) -> np.ndarray:
    half = n_nodes // 2
    rows = []
    # Fix node zero in community zero, eliminating complementary duplicates.
    for community_one in itertools.combinations(range(1, n_nodes), half):
        labels = np.zeros(n_nodes, dtype=bool)
        labels[list(community_one)] = True
        rows.append(labels)
    result = np.asarray(rows, dtype=bool)
    expected = math.comb(n_nodes - 1, half)
    if len(result) != expected:
        raise RuntimeError("partition enumeration mismatch")
    return result


FULL_PARTITIONS = balanced_partitions(12)
FIT_PARTITIONS = balanced_partitions(8)


def family_layout(
    family: str, data: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(np.asarray(data["families"]) == family)
    return (
        indices,
        np.asarray(data["cases"])[indices],
        np.asarray(data["variants"])[indices],
    )


def eligible_directed(
    similarity: np.ndarray,
    family: str,
    data: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices, cases, variants = family_layout(family, data)
    weights = np.zeros((12, 12), dtype=np.float64)
    mask = np.zeros((12, 12), dtype=bool)
    for source in range(12):
        for target in range(12):
            if (
                source != target
                and variants[source] != variants[target]
                and cases[source] != cases[target]
            ):
                weights[source, target] = max(
                    0.0,
                    float(similarity[indices[source], indices[target]]),
                )
                mask[source, target] = True
    return weights, mask, cases, variants


def symmetrized_edges(
    directed_weights: np.ndarray,
    directed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = []
    second = []
    weights = []
    for left in range(len(directed_weights)):
        for right in range(left + 1, len(directed_weights)):
            if directed_mask[left, right] or directed_mask[right, left]:
                first.append(left)
                second.append(right)
                weights.append(
                    directed_weights[left, right]
                    + directed_weights[right, left]
                )
    return (
        np.column_stack([first, second]).astype(np.int64),
        np.asarray(weights, dtype=np.float64),
    )


def score_matrix(
    partitions: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    same = (
        partitions[:, edges[:, 0]] == partitions[:, edges[:, 1]]
    )
    within_count = np.sum(same, axis=1)
    between_count = np.sum(~same, axis=1)
    valid = (within_count > 0) & (between_count > 0)
    coefficients = np.zeros(same.shape, dtype=np.float64)
    coefficients[valid] = (
        same[valid] / within_count[valid, None]
        - (~same[valid]) / between_count[valid, None]
    )
    return coefficients, valid


def exact_partition(
    partitions: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float, float, int]:
    coefficients, valid = score_matrix(partitions, edges)
    # An explicit reduction avoids spurious overflow warnings emitted by the
    # local Accelerate-backed BLAS for these small, finite matrices.
    scores = np.sum(coefficients * weights[None, :], axis=1)
    scores[~valid] = -np.inf
    order = np.argsort(-scores, kind="stable")
    best_index = int(order[0])
    second_index = int(order[1])
    return (
        partitions[best_index].copy(),
        float(scores[best_index]),
        float(scores[second_index]),
        best_index,
    )


def metrics(
    partition: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    ordinary = float(np.mean(partition == truth))
    return {
        "ari": float(adjusted_rand_score(truth, partition)),
        "nmi": float(normalized_mutual_info_score(truth, partition)),
        "ordinary_accuracy": ordinary,
        "gauge_accuracy": max(ordinary, 1.0 - ordinary),
    }


def fit_node_indices(variants: np.ndarray) -> np.ndarray:
    return np.flatnonzero(
        np.isin(variants, ["anchor", "physics_paraphrase"])
    )


def heldout_partition(
    directed_weights: np.ndarray,
    directed_mask: np.ndarray,
    cases: np.ndarray,
    variants: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    fit_indices = fit_node_indices(variants)
    heldout_indices = np.flatnonzero(
        variants == "lexical_counterfactual"
    )
    fit_lookup = {
        int(global_local): fit_local
        for fit_local, global_local in enumerate(fit_indices)
    }
    full_edges, full_weights = symmetrized_edges(
        directed_weights, directed_mask
    )
    keep = np.asarray(
        [
            int(left) in fit_lookup and int(right) in fit_lookup
            for left, right in full_edges
        ],
        dtype=bool,
    )
    fit_edges = np.asarray(
        [
            [fit_lookup[int(left)], fit_lookup[int(right)]]
            for left, right in full_edges[keep]
        ],
        dtype=np.int64,
    )
    fit_weights = full_weights[keep]
    fit_partition, best, second, _ = exact_partition(
        FIT_PARTITIONS, fit_edges, fit_weights
    )
    prediction = np.zeros(12, dtype=bool)
    prediction[fit_indices] = fit_partition
    sym_weights = directed_weights + directed_weights.T
    sym_mask = directed_mask | directed_mask.T
    for node in heldout_indices:
        means = []
        for community in (False, True):
            candidates = fit_indices[
                (fit_partition == community)
                & (cases[fit_indices] != cases[node])
                & sym_mask[node, fit_indices]
            ]
            means.append(
                float(np.mean(sym_weights[node, candidates]))
                if len(candidates)
                else -np.inf
            )
        prediction[node] = bool(means[1] > means[0])
    return prediction, best, best - second


def blockwise_resample(
    weights: np.ndarray,
    mask: np.ndarray,
    variants: np.ndarray,
    rng: np.random.Generator,
    *,
    replace: bool,
) -> np.ndarray:
    result = np.zeros_like(weights)
    for source_variant in VARIANTS:
        for target_variant in VARIANTS:
            if source_variant == target_variant:
                continue
            block = (
                (variants[:, None] == source_variant)
                & (variants[None, :] == target_variant)
                & mask
            )
            values = weights[block]
            if replace:
                sampled = rng.choice(values, size=len(values), replace=True)
            else:
                sampled = values.copy()
                rng.shuffle(sampled)
            result[block] = sampled
    return result


def observed_and_randomized(
    data: Mapping[str, object],
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    rng = np.random.default_rng(SEED)
    families_array = np.asarray(data["families"])
    physical = np.asarray(data["physical"], dtype=bool)
    band_mask = np.asarray(data["band_mask"])
    observed_rows = []
    null_outputs = {}
    bootstrap_outputs = {}
    bootstrap_summary_rows = []

    for method in METHODS:
        layers = np.asarray(data["similarities"][method])
        band = np.mean(layers[band_mask], axis=0)
        family_artifacts = {}
        for family in data["family_order"]:
            family = str(family)
            truth = physical[families_array == family]
            directed, mask, cases, variants = eligible_directed(
                band, family, data
            )
            edges, edge_weights = symmetrized_edges(directed, mask)
            partition, best, second, _ = exact_partition(
                FULL_PARTITIONS, edges, edge_weights
            )
            heldout, heldout_best, heldout_margin = heldout_partition(
                directed, mask, cases, variants
            )
            observed_rows.append(
                {
                    "method": method,
                    "family": family,
                    "endpoint": "complete",
                    **metrics(partition, truth),
                    "best_score": best,
                    "second_score": second,
                    "margin": best - second,
                }
            )
            observed_rows.append(
                {
                    "method": method,
                    "family": family,
                    "endpoint": "heldout_surface",
                    **metrics(heldout, truth),
                    "best_score": heldout_best,
                    "second_score": heldout_best - heldout_margin,
                    "margin": heldout_margin,
                }
            )
            family_artifacts[family] = {
                "truth": truth,
                "directed": directed,
                "mask": mask,
                "cases": cases,
                "variants": variants,
                "edges": edges,
                "partition": partition,
                "heldout": heldout,
            }

            complete_ari = np.empty(N_BOOTSTRAP, dtype=np.float32)
            heldout_ari = np.empty(N_BOOTSTRAP, dtype=np.float32)
            complete_identical = np.empty(N_BOOTSTRAP, dtype=bool)
            heldout_identical = np.empty(N_BOOTSTRAP, dtype=bool)
            coassignment = np.zeros((12, 12), dtype=np.float64)
            for bootstrap_index in range(N_BOOTSTRAP):
                sampled = blockwise_resample(
                    directed, mask, variants, rng, replace=True
                )
                _, sampled_weights = symmetrized_edges(sampled, mask)
                sampled_partition, *_ = exact_partition(
                    FULL_PARTITIONS, edges, sampled_weights
                )
                sampled_heldout, *_ = heldout_partition(
                    sampled, mask, cases, variants
                )
                complete_ari[bootstrap_index] = adjusted_rand_score(
                    partition, sampled_partition
                )
                heldout_ari[bootstrap_index] = adjusted_rand_score(
                    heldout, sampled_heldout
                )
                complete_identical[bootstrap_index] = bool(
                    np.array_equal(partition, sampled_partition)
                    or np.array_equal(partition, ~sampled_partition)
                )
                heldout_identical[bootstrap_index] = bool(
                    np.array_equal(heldout, sampled_heldout)
                    or np.array_equal(heldout, ~sampled_heldout)
                )
                coassignment += (
                    sampled_partition[:, None]
                    == sampled_partition[None, :]
                )
            key = f"{method}_{family}"
            bootstrap_outputs[f"{key}_complete_ari"] = complete_ari
            bootstrap_outputs[f"{key}_heldout_ari"] = heldout_ari
            bootstrap_outputs[f"{key}_coassignment"] = (
                coassignment / N_BOOTSTRAP
            ).astype(np.float32)
            bootstrap_summary_rows.append(
                {
                    "method": method,
                    "family": family,
                    "complete_mean_bootstrap_ari": float(
                        np.mean(complete_ari)
                    ),
                    "complete_identical_fraction": float(
                        np.mean(complete_identical)
                    ),
                    "heldout_mean_bootstrap_ari": float(
                        np.mean(heldout_ari)
                    ),
                    "heldout_identical_fraction": float(
                        np.mean(heldout_identical)
                    ),
                }
            )

        null_complete = np.empty(
            (N_NULL, len(data["family_order"])), dtype=np.float32
        )
        null_heldout = np.empty_like(null_complete)
        for family_index, family in enumerate(data["family_order"]):
            family = str(family)
            artifact = family_artifacts[family]
            for null_index in range(N_NULL):
                shuffled = blockwise_resample(
                    artifact["directed"],
                    artifact["mask"],
                    artifact["variants"],
                    rng,
                    replace=False,
                )
                _, shuffled_weights = symmetrized_edges(
                    shuffled, artifact["mask"]
                )
                partition, *_ = exact_partition(
                    FULL_PARTITIONS,
                    artifact["edges"],
                    shuffled_weights,
                )
                heldout, *_ = heldout_partition(
                    shuffled,
                    artifact["mask"],
                    artifact["cases"],
                    artifact["variants"],
                )
                null_complete[null_index, family_index] = (
                    adjusted_rand_score(artifact["truth"], partition)
                )
                null_heldout[null_index, family_index] = (
                    adjusted_rand_score(artifact["truth"], heldout)
                )
        null_outputs[f"{method}_complete_family_ari"] = null_complete
        null_outputs[f"{method}_heldout_family_ari"] = null_heldout
    return (
        pd.DataFrame(observed_rows),
        null_outputs,
        bootstrap_outputs,
        pd.DataFrame(bootstrap_summary_rows),
    )


def summarize(
    observed: pd.DataFrame,
    nulls: Mapping[str, np.ndarray],
) -> dict[str, object]:
    summary = {}
    family_order = sorted(observed["family"].unique())
    for method in METHODS:
        summary[method] = {}
        for endpoint in ("complete", "heldout_surface"):
            selected = observed[
                (observed["method"] == method)
                & (observed["endpoint"] == endpoint)
            ].set_index("family").loc[family_order]
            aris = selected["ari"].to_numpy(float)
            gauges = selected["gauge_accuracy"].to_numpy(float)
            null_key = (
                f"{method}_complete_family_ari"
                if endpoint == "complete"
                else f"{method}_heldout_family_ari"
            )
            null_mean = np.mean(
                np.asarray(nulls[null_key]), axis=1
            )
            observed_mean = float(np.mean(aris))
            p_value = float(
                (1 + np.sum(null_mean >= observed_mean - 1e-12))
                / (1 + len(null_mean))
            )
            summary[method][endpoint] = {
                "ari_by_family": dict(
                    zip(family_order, aris.tolist())
                ),
                "gauge_accuracy_by_family": dict(
                    zip(family_order, gauges.tolist())
                ),
                "mean_ari": observed_mean,
                "mean_gauge_accuracy": float(np.mean(gauges)),
                "positive_ari_families": int(np.sum(aris > 0)),
                "gauge_above_half_families": int(
                    np.sum(gauges > 0.5)
                ),
                "null_mean_ari": float(np.mean(null_mean)),
                "null_sd_ari": float(np.std(null_mean, ddof=1)),
                "null_exact_p": p_value,
            }
        summary[method]["strong_dense_partition_gate"] = bool(
            summary[method]["complete"]["positive_ari_families"] == 6
            and summary[method]["complete"][
                "gauge_above_half_families"
            ]
            == 6
            and summary[method]["complete"]["null_exact_p"] <= 0.05
            and summary[method]["heldout_surface"][
                "positive_ari_families"
            ]
            == 6
            and summary[method]["heldout_surface"][
                "gauge_above_half_families"
            ]
            == 6
            and summary[method]["heldout_surface"]["null_exact_p"]
            <= 0.05
        )

    for endpoint in ("complete", "heldout_surface"):
        pivot = observed[
            observed["endpoint"] == endpoint
        ].pivot(index="family", columns="method", values="ari")
        direct = (pivot["jacobian"] - pivot["direct"]).sort_index()
        raw = (pivot["jacobian"] - pivot["raw"]).sort_index()
        summary[f"jacobian_specificity_{endpoint}"] = {
            "minus_direct_by_family": direct.to_dict(),
            "minus_raw_by_family": raw.to_dict(),
            "over_direct_families": int(np.sum(direct.to_numpy() > 0)),
            "over_raw_families": int(np.sum(raw.to_numpy() > 0)),
            "minus_direct_signflip_p": plus_one_signflip_p(direct),
            "minus_raw_signflip_p": plus_one_signflip_p(raw),
        }
    return summary


def main() -> None:
    validate_inputs()
    data = load_data()
    observed, nulls, bootstraps, bootstrap_summary = (
        observed_and_randomized(data)
    )
    summary = summarize(observed, nulls)
    observed.to_csv(
        OUT / "exact_partition_metrics.csv", index=False
    )
    bootstrap_summary.to_csv(
        OUT / "exact_partition_bootstrap_summary.csv", index=False
    )
    np.savez_compressed(
        OUT / "exact_partition_nulls.npz", **nulls
    )
    np.savez_compressed(
        OUT / "exact_partition_bootstraps.npz", **bootstraps
    )
    results = {
        "study_id": json.loads(PARTITION_PROTOCOL.read_text())[
            "study_id"
        ],
        "partition_protocol_sha256": sha256(PARTITION_PROTOCOL),
        "partition_protocol_markdown_sha256": sha256(
            PARTITION_PROTOCOL_MD
        ),
        "spectral_protocol_sha256": sha256(SPECTRAL_PROTOCOL),
        "partitions_enumerated": {
            "complete": len(FULL_PARTITIONS),
            "fit_only": len(FIT_PARTITIONS),
        },
        "summary": summary,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (OUT / "study6_partition_statistics.json").write_text(
        json.dumps(safe_json(results), indent=2) + "\n"
    )
    print(json.dumps(safe_json(results), indent=2))


if __name__ == "__main__":
    main()
