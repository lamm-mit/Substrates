#!/usr/bin/env python3
"""Spectral graph-community and density-ablation experiment."""

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
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from analyze_graph_isomorphism_generalization import (
    MANIFEST,
    METHODS,
    STATES,
    VARIANTS,
    bh_adjust,
    load_data,
    plus_one_signflip_p,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "experiments"
    / "graph-isomorphism-generalization-2026-07-18"
)
GAUGE_PROTOCOL = OUT / "gauge_protocol.json"
SPECTRAL_PROTOCOL = OUT / "spectral_protocol.json"
SPECTRAL_PROTOCOL_MD = OUT / "SPECTRAL_PROTOCOL.md"
SEED = 20260718
N_NULL = 10_000
MODES = (
    "binary_top1",
    "weighted_top1",
    "weighted_top2",
    "weighted_complete_candidates",
)


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
    spectral = json.loads(SPECTRAL_PROTOCOL.read_text())
    gauge = json.loads(GAUGE_PROTOCOL.read_text())
    if sha256(GAUGE_PROTOCOL) != spectral["parent_gauge_protocol_sha256"]:
        raise RuntimeError("gauge protocol fingerprint mismatch")
    if (
        sha256(STATES)
        != gauge["inputs"]["representations_sha256"]
        or sha256(MANIFEST)
        != gauge["inputs"]["prompt_manifest_sha256"]
    ):
        raise RuntimeError("frozen input fingerprint mismatch")


def family_node_data(
    family: str, data: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    families = np.asarray(data["families"])
    indices = np.flatnonzero(families == family)
    return (
        indices,
        np.asarray(data["cases"])[indices],
        np.asarray(data["variants"])[indices],
    )


def affinity_adjacency(
    similarity: np.ndarray,
    family: str,
    data: Mapping[str, object],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    indices, cases, variants = family_node_data(family, data)
    local = {
        int(global_index): local_index
        for local_index, global_index in enumerate(indices)
    }
    if mode in {"binary_top1", "weighted_top1"}:
        k = 1
    elif mode == "weighted_top2":
        k = 2
    elif mode == "weighted_complete_candidates":
        k = 3
    else:
        raise ValueError(mode)
    directed = np.zeros((12, 12), dtype=np.float64)
    selected_weights = []
    for source_local, source_global in enumerate(indices):
        for target_variant in VARIANTS:
            if target_variant == variants[source_local]:
                continue
            eligible_local = np.flatnonzero(
                (variants == target_variant)
                & (cases != cases[source_local])
            )
            ordered = sorted(
                eligible_local.tolist(),
                key=lambda target_local: (
                    -float(
                        similarity[
                            source_global, indices[target_local]
                        ]
                    ),
                    int(target_local),
                ),
            )
            for target_local in ordered[:k]:
                weight = (
                    1.0
                    if mode == "binary_top1"
                    else max(
                        0.0,
                        float(
                            similarity[
                                source_global, indices[target_local]
                            ]
                        ),
                    )
                )
                directed[source_local, target_local] = weight
                selected_weights.append(weight)
    del local
    adjacency = directed + directed.T
    return adjacency, np.asarray(selected_weights, dtype=float)


def fiedler_partition(
    adjacency: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    degree_values = adjacency.sum(axis=1)
    inverse_sqrt = np.zeros_like(degree_values)
    positive = degree_values > 1e-12
    inverse_sqrt[positive] = degree_values[positive] ** -0.5
    laplacian = np.eye(len(adjacency)) - (
        inverse_sqrt[:, None]
        * adjacency
        * inverse_sqrt[None, :]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    fiedler = eigenvectors[:, 1].copy()
    largest = int(np.argmax(np.abs(fiedler)))
    if fiedler[largest] < 0:
        fiedler *= -1
    order = np.lexsort((np.arange(len(fiedler)), fiedler))
    partition = np.zeros(len(fiedler), dtype=bool)
    partition[order[6:]] = True
    eigengap = float(eigenvalues[2] - eigenvalues[1])
    return partition, eigengap, eigenvalues


def partition_metrics(
    partition: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    ordinary = float(np.mean(partition == labels))
    return {
        "ari": float(adjusted_rand_score(labels, partition)),
        "nmi": float(normalized_mutual_info_score(labels, partition)),
        "ordinary_accuracy": ordinary,
        "gauge_accuracy": max(ordinary, 1.0 - ordinary),
    }


def observed_spectral(
    data: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    layer_rows = []
    families = np.asarray(data["families"])
    physical = np.asarray(data["physical"], dtype=bool)
    numeric = np.asarray(data["numeric"], dtype=bool)
    layers = np.asarray(data["layers"])
    depths = np.asarray(data["depths"])
    band_mask = np.asarray(data["band_mask"])
    for method in METHODS:
        similarity_layers = np.asarray(data["similarities"][method])
        band_similarity = np.mean(
            similarity_layers[band_mask], axis=0
        )
        for family in data["family_order"]:
            family_indices = np.flatnonzero(families == family)
            family_physical = physical[family_indices]
            family_numeric = numeric[family_indices]
            for mode in MODES:
                adjacency, weights = affinity_adjacency(
                    band_similarity, str(family), data, mode
                )
                partition, eigengap, eigenvalues = fiedler_partition(
                    adjacency
                )
                metrics = partition_metrics(partition, family_physical)
                numeric_metrics = partition_metrics(
                    partition, family_numeric
                )
                rows.append(
                    {
                        "method": method,
                        "family": family,
                        "mode": mode,
                        **metrics,
                        "numeric_ari": numeric_metrics["ari"],
                        "numeric_gauge_accuracy": numeric_metrics[
                            "gauge_accuracy"
                        ],
                        "physical_numeric_partition_identical": bool(
                            np.array_equal(
                                family_physical, family_numeric
                            )
                            or np.array_equal(
                                family_physical, ~family_numeric
                            )
                        ),
                        "eigengap": eigengap,
                        "edges_directed": len(weights),
                        "weight_mean": float(np.mean(weights)),
                        "eigenvalues": json.dumps(
                            [float(value) for value in eigenvalues]
                        ),
                    }
                )
            for layer_index, (layer, depth) in enumerate(
                zip(layers, depths)
            ):
                adjacency, _ = affinity_adjacency(
                    similarity_layers[layer_index],
                    str(family),
                    data,
                    "weighted_top1",
                )
                partition, eigengap, _ = fiedler_partition(adjacency)
                layer_rows.append(
                    {
                        "method": method,
                        "family": family,
                        "layer": int(layer),
                        "depth_percent": float(depth),
                        **partition_metrics(
                            partition, family_physical
                        ),
                        "eigengap": eigengap,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(layer_rows)


def random_constrained_adjacency(
    family: str,
    data: Mapping[str, object],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    _, cases, variants = family_node_data(family, data)
    directed = np.zeros((12, 12), dtype=float)
    random_weights = weights.copy()
    rng.shuffle(random_weights)
    edge_index = 0
    for source in range(12):
        for target_variant in VARIANTS:
            if target_variant == variants[source]:
                continue
            candidates = np.flatnonzero(
                (variants == target_variant)
                & (cases != cases[source])
            )
            target = int(rng.choice(candidates))
            directed[source, target] = random_weights[edge_index]
            edge_index += 1
    if edge_index != 24:
        raise RuntimeError("unexpected null edge count")
    return directed + directed.T


def constrained_nulls(
    data: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    rng = np.random.default_rng(SEED)
    families_array = np.asarray(data["families"])
    physical = np.asarray(data["physical"], dtype=bool)
    band_mask = np.asarray(data["band_mask"])
    outputs = {}
    summaries = {}
    for method in METHODS:
        layers = np.asarray(data["similarities"][method])
        band = np.mean(layers[band_mask], axis=0)
        templates = {}
        labels = {}
        for family in data["family_order"]:
            _, weights = affinity_adjacency(
                band, str(family), data, "weighted_top1"
            )
            templates[str(family)] = weights
            labels[str(family)] = physical[
                families_array == family
            ]
        mean_ari = np.empty(N_NULL, dtype=np.float32)
        mean_gauge = np.empty(N_NULL, dtype=np.float32)
        for null_index in range(N_NULL):
            aris = []
            gauges = []
            for family in data["family_order"]:
                adjacency = random_constrained_adjacency(
                    str(family),
                    data,
                    templates[str(family)],
                    rng,
                )
                partition, _, _ = fiedler_partition(adjacency)
                metrics = partition_metrics(
                    partition, labels[str(family)]
                )
                aris.append(metrics["ari"])
                gauges.append(metrics["gauge_accuracy"])
            mean_ari[null_index] = float(np.mean(aris))
            mean_gauge[null_index] = float(np.mean(gauges))
        outputs[f"{method}_mean_ari"] = mean_ari
        outputs[f"{method}_mean_gauge_accuracy"] = mean_gauge
        summaries[method] = {
            "null_mean_ari": float(np.mean(mean_ari)),
            "null_sd_ari": float(np.std(mean_ari, ddof=1)),
            "null_mean_gauge_accuracy": float(np.mean(mean_gauge)),
        }
    return outputs, summaries


def synthetic_graph(
    *,
    positive: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cases = np.repeat(np.arange(4), 3)
    variants = np.tile(np.arange(3), 4)
    anchor = np.zeros(4, dtype=bool)
    anchor[rng.choice(4, size=2, replace=False)] = True
    labels = np.asarray(
        [
            anchor[case] if variant < 2 else ~anchor[case]
            for case, variant in zip(cases, variants)
        ],
        dtype=bool,
    )
    scores = rng.normal(0.0, 0.25, size=(12, 12))
    if positive:
        scores += (labels[:, None] == labels[None, :]).astype(float)
    directed = np.zeros((12, 12), dtype=float)
    for source in range(12):
        for target_variant in range(3):
            if target_variant == variants[source]:
                continue
            candidates = np.flatnonzero(
                (variants == target_variant)
                & (cases != cases[source])
            )
            target = min(
                candidates.tolist(),
                key=lambda index: (-scores[source, index], int(index)),
            )
            directed[source, target] = max(
                0.0, float(scores[source, target] + 1.0)
            )
    observed_labels = labels.copy()
    if rng.random() < 0.5:
        observed_labels = ~observed_labels
    return directed + directed.T, labels, observed_labels


def synthetic_controls() -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(SEED)
    rows = []
    for condition in ("positive", "negative"):
        for graph_index in range(1000):
            adjacency, canonical, observed = synthetic_graph(
                positive=condition == "positive", rng=rng
            )
            partition, eigengap, _ = fiedler_partition(adjacency)
            canonical_metrics = partition_metrics(
                partition, canonical
            )
            observed_metrics = partition_metrics(
                partition, observed
            )
            rows.append(
                {
                    "condition": condition,
                    "graph_index": graph_index,
                    "ari": canonical_metrics["ari"],
                    "gauge_accuracy": canonical_metrics[
                        "gauge_accuracy"
                    ],
                    "ordinary_accuracy_random_orientation": (
                        observed_metrics["ordinary_accuracy"]
                    ),
                    "eigengap": eigengap,
                }
            )
    frame = pd.DataFrame(rows)
    positive = frame[frame["condition"] == "positive"]
    negative = frame[frame["condition"] == "negative"]
    summary = {
        "positive_mean_ari": float(positive["ari"].mean()),
        "positive_mean_gauge_accuracy": float(
            positive["gauge_accuracy"].mean()
        ),
        "positive_mean_ordinary_random_orientation": float(
            positive["ordinary_accuracy_random_orientation"].mean()
        ),
        "negative_mean_ari": float(negative["ari"].mean()),
        "negative_mean_gauge_accuracy": float(
            negative["gauge_accuracy"].mean()
        ),
        "positive_control_passes": bool(
            positive["ari"].mean() >= 0.8
        ),
        "negative_control_passes": bool(
            abs(negative["ari"].mean()) <= 0.05
        ),
    }
    return frame, summary


def summarize(
    observed: pd.DataFrame,
    null_summary: dict[str, object],
    null_arrays: Mapping[str, np.ndarray],
    synthetic_summary: Mapping[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    primary = observed[
        observed["mode"] == "weighted_top1"
    ].copy()
    summaries = {}
    for method in METHODS:
        selected = primary[primary["method"] == method].set_index(
            "family"
        )
        family_order = sorted(selected.index)
        aris = selected.loc[family_order, "ari"].to_numpy(float)
        gauges = selected.loc[
            family_order, "gauge_accuracy"
        ].to_numpy(float)
        null = np.asarray(null_arrays[f"{method}_mean_ari"])
        observed_mean = float(np.mean(aris))
        p_value = float(
            (1 + np.sum(null >= observed_mean - 1e-12))
            / (1 + len(null))
        )
        summaries[method] = {
            "ari_by_family": dict(zip(family_order, aris.tolist())),
            "gauge_accuracy_by_family": dict(
                zip(family_order, gauges.tolist())
            ),
            "mean_ari": observed_mean,
            "mean_gauge_accuracy": float(np.mean(gauges)),
            "positive_ari_families": int(np.sum(aris > 0)),
            "gauge_above_half_families": int(np.sum(gauges > 0.5)),
            "constrained_null_exact_p": p_value,
            **null_summary[method],
        }
        summaries[method]["strong_spectral_gate"] = bool(
            summaries[method]["positive_ari_families"] == 6
            and summaries[method]["gauge_above_half_families"] == 6
            and p_value <= 0.05
            and bool(synthetic_summary["positive_control_passes"])
            and bool(synthetic_summary["negative_control_passes"])
        )
    pivot = primary.pivot(
        index="family", columns="method", values="ari"
    )
    jac_direct = (
        pivot["jacobian"] - pivot["direct"]
    ).sort_index()
    jac_raw = (pivot["jacobian"] - pivot["raw"]).sort_index()
    specificity = {
        "jacobian_minus_direct_by_family": jac_direct.to_dict(),
        "jacobian_minus_raw_by_family": jac_raw.to_dict(),
        "jacobian_over_direct_families": int(
            np.sum(jac_direct.to_numpy() > 0)
        ),
        "jacobian_over_raw_families": int(
            np.sum(jac_raw.to_numpy() > 0)
        ),
        "jacobian_minus_direct_signflip_p": plus_one_signflip_p(
            jac_direct
        ),
        "jacobian_minus_raw_signflip_p": plus_one_signflip_p(
            jac_raw
        ),
    }
    specificity["jacobian_specificity_passes"] = bool(
        specificity["jacobian_over_direct_families"] >= 5
        and specificity["jacobian_over_raw_families"] >= 5
        and specificity["jacobian_minus_direct_signflip_p"] <= 0.05
        and specificity["jacobian_minus_raw_signflip_p"] <= 0.05
    )
    summaries["jacobian_specificity"] = specificity

    ablation_rows = []
    for method in METHODS:
        for mode in MODES:
            values = observed[
                (observed["method"] == method)
                & (observed["mode"] == mode)
            ]["ari"].to_numpy(float)
            ablation_rows.append(
                {
                    "method": method,
                    "mode": mode,
                    "mean_ari": float(np.mean(values)),
                    "positive_families": int(np.sum(values > 0)),
                    "signflip_p": plus_one_signflip_p(values),
                }
            )
    ablations = pd.DataFrame(ablation_rows)
    ablations["bh_q"] = bh_adjust(ablations["signflip_p"])
    return summaries, ablations


def main() -> None:
    validate_inputs()
    data = load_data()
    observed, layer_rows = observed_spectral(data)
    null_arrays, null_summary = constrained_nulls(data)
    synthetic_rows, synthetic_summary = synthetic_controls()
    summaries, ablations = summarize(
        observed, null_summary, null_arrays, synthetic_summary
    )

    observed.to_csv(OUT / "spectral_community_metrics.csv", index=False)
    layer_rows.to_csv(
        OUT / "spectral_community_layers.csv", index=False
    )
    synthetic_rows.to_csv(
        OUT / "spectral_synthetic_controls.csv", index=False
    )
    ablations.to_csv(
        OUT / "spectral_density_ablations.csv", index=False
    )
    np.savez_compressed(
        OUT / "spectral_constrained_nulls.npz", **null_arrays
    )
    results = {
        "study_id": json.loads(SPECTRAL_PROTOCOL.read_text())[
            "study_id"
        ],
        "spectral_protocol_sha256": sha256(SPECTRAL_PROTOCOL),
        "spectral_protocol_markdown_sha256": sha256(
            SPECTRAL_PROTOCOL_MD
        ),
        "gauge_protocol_sha256": sha256(GAUGE_PROTOCOL),
        "primary": summaries,
        "synthetic_controls": synthetic_summary,
        "density_ablations": ablations.to_dict(orient="records"),
        "physical_numeric_relation_identity": bool(
            observed["physical_numeric_partition_identical"].all()
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (OUT / "study5_spectral_statistics.json").write_text(
        json.dumps(safe_json(results), indent=2) + "\n"
    )
    print(json.dumps(safe_json(results), indent=2))


if __name__ == "__main__":
    main()
