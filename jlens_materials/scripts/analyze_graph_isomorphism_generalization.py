#!/usr/bin/env python3
"""Frozen graph-isomorphism and cross-mechanism template analysis.

This script executes Studies 1 and 2 from
``experiments/graph-isomorphism-generalization-2026-07-18/PROTOCOL.md``.
It uses only archived option-free states and never runs Gemma.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "experiments"
    / "graph-isomorphism-generalization-2026-07-18"
)
PROTOCOL = OUT / "protocol.json"
PROTOCOL_MD = OUT / "PROTOCOL.md"
STATES = (
    ROOT
    / "experiments"
    / "option-free-question-end-2026-07-18"
    / "representations.npz"
)
MANIFEST = (
    ROOT
    / "experiments"
    / "late-physics-representation-replication-2026-07-17"
    / "prompt_manifest.json"
)
SEED = 20260718
N_NULL = 10_000
VARIANTS = ("anchor", "physics_paraphrase", "lexical_counterfactual")
METHODS = ("jacobian", "direct", "raw")


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


def validate_inputs(protocol: Mapping[str, object]) -> None:
    lookup = {
        "representations": STATES,
        "prompt_manifest": MANIFEST,
    }
    inputs = protocol["inputs"]  # type: ignore[index]
    for name, path in lookup.items():
        expected = str(inputs[name]["sha256"])  # type: ignore[index]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"input fingerprint mismatch for {path}: {actual} != {expected}"
            )


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise FloatingPointError("zero-norm state")
    return values / norms


def cosine_layers(values: np.ndarray) -> np.ndarray:
    """Pairwise cosine for ``[item, layer, feature]``."""

    unit = normalize(values.astype(np.float64))
    result = np.einsum("ild,jld->lij", unit, unit, optimize=True)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite cosine matrix")
    return result


def pearson(first: Sequence[float], second: Sequence[float]) -> float:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(pearsonr(x, y).statistic)


def signflip_p(values: Sequence[float], two_sided: bool = True) -> float:
    data = np.asarray(values, dtype=float)
    observed = float(np.mean(data))
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(data))))
    null = np.mean(signs * data[None, :], axis=1)
    if two_sided:
        return float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return float(np.mean(null >= observed - 1e-12))


def plus_one_signflip_p(values: Sequence[float]) -> float:
    data = np.asarray(values, dtype=float)
    observed = float(np.mean(data))
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(data))))
    null = np.mean(signs * data[None, :], axis=1)
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + len(null)))


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return np.clip(adjusted, 0.0, 1.0)


def load_data() -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text())
    metadata = {str(row["prompt_id"]): row for row in manifest["prompts"]}
    with np.load(STATES, allow_pickle=False) as archive:
        prompt_ids = archive["prompt_ids"].astype(str)
        layers = archive["layers"].astype(int)
        raw_states = archive["raw_states"][0].astype(np.float64)
        direct_states = archive["direct_decoder_basis"][0].astype(np.float64)
        jacobian_states = archive["jacobian_decoder_basis"][
            :, 0
        ].astype(np.float64)
    rows = []
    for prompt_id in prompt_ids:
        key = prompt_id.removesuffix("--answer-code")
        if key not in metadata:
            raise KeyError(f"missing prompt metadata: {prompt_id}")
        rows.append(metadata[key])
    families = np.asarray([str(row["family_id"]) for row in rows])
    cases = np.asarray([str(row["case_id"]) for row in rows])
    variants = np.asarray([str(row["variant"]) for row in rows])
    physical = np.asarray(
        [bool(row["expected_outcome"] == row["outcome_positive"]) for row in rows]
    )
    numeric = np.asarray(
        [bool(row["numeric_direction"] == "increase") for row in rows]
    )
    family_order = sorted(set(families))
    case_order = {
        family: sorted(set(cases[families == family]))
        for family in family_order
    }
    response_orientation = {}
    for family in family_order:
        indices = np.flatnonzero(
            (families == family) & (variants == "anchor")
        )
        agrees = physical[indices] == numeric[indices]
        if np.all(agrees):
            response_orientation[family] = "direct"
        elif np.all(~agrees):
            response_orientation[family] = "inverse"
        else:
            raise RuntimeError(f"mixed response orientation in {family}")

    jacobian_fit_cosines = np.asarray(
        [cosine_layers(fit) for fit in jacobian_states]
    )
    similarities = {
        "jacobian": np.mean(jacobian_fit_cosines, axis=0),
        "direct": cosine_layers(direct_states),
        "raw": cosine_layers(raw_states),
    }
    depths = layers / 41.0 * 100.0
    band_mask = (depths >= 38.0) & (depths <= 92.0)
    return {
        "prompt_ids": prompt_ids,
        "rows": rows,
        "families": families,
        "cases": cases,
        "variants": variants,
        "physical": physical,
        "numeric": numeric,
        "family_order": family_order,
        "case_order": case_order,
        "response_orientation": response_orientation,
        "layers": layers,
        "depths": depths,
        "band_mask": band_mask,
        "similarities": similarities,
        "jacobian_fit_cosines": jacobian_fit_cosines,
    }


def index_lookup(data: Mapping[str, object]) -> dict[tuple[str, str, str], int]:
    families = np.asarray(data["families"])
    cases = np.asarray(data["cases"])
    variants = np.asarray(data["variants"])
    return {
        (str(family), str(case), str(variant)): int(index)
        for index, (family, case, variant) in enumerate(
            zip(families, cases, variants)
        )
    }


def fitting_vector(
    similarity: np.ndarray,
    family: str,
    cases: Sequence[str],
    lookup: Mapping[tuple[str, str, str], int],
    permutation: Sequence[int] | None = None,
) -> np.ndarray:
    target_cases = (
        [cases[index] for index in permutation]
        if permutation is not None
        else list(cases)
    )
    values = []
    for first in range(4):
        for second in range(4):
            if first == second:
                continue
            values.append(
                similarity[
                    lookup[(family, target_cases[first], "anchor")],
                    lookup[
                        (
                            family,
                            target_cases[second],
                            "physics_paraphrase",
                        )
                    ],
                ]
            )
    return np.asarray(values, dtype=float)


def heldout_vector(
    similarity: np.ndarray,
    family: str,
    cases: Sequence[str],
    lookup: Mapping[tuple[str, str, str], int],
    permutation: Sequence[int] | None = None,
) -> np.ndarray:
    target_cases = (
        [cases[index] for index in permutation]
        if permutation is not None
        else list(cases)
    )
    values = []
    for first in range(4):
        for second in range(4):
            if first == second:
                continue
            source = lookup[
                (family, target_cases[first], "lexical_counterfactual")
            ]
            for variant in ("anchor", "physics_paraphrase"):
                target = lookup[(family, target_cases[second], variant)]
                values.append(similarity[source, target])
    return np.asarray(values, dtype=float)


def fit_pair_mapping(
    similarity: np.ndarray,
    source_family: str,
    target_family: str,
    data: Mapping[str, object],
) -> tuple[tuple[int, ...], list[dict[str, object]], float]:
    cases = data["case_order"]  # type: ignore[assignment]
    lookup = index_lookup(data)
    source_cases = cases[source_family]
    target_cases = cases[target_family]
    source_fit = fitting_vector(
        similarity, source_family, source_cases, lookup
    )
    scores = []
    for permutation in itertools.permutations(range(4)):
        target_fit = fitting_vector(
            similarity,
            target_family,
            target_cases,
            lookup,
            permutation,
        )
        scores.append(
            {
                "permutation": tuple(int(value) for value in permutation),
                "fit_correlation": pearson(source_fit, target_fit),
            }
        )
    best = sorted(
        scores,
        key=lambda row: (
            -float(row["fit_correlation"]),
            tuple(row["permutation"]),
        ),
    )[0]
    permutation = tuple(best["permutation"])
    source_test = heldout_vector(
        similarity, source_family, source_cases, lookup
    )
    target_test = heldout_vector(
        similarity,
        target_family,
        target_cases,
        lookup,
        permutation,
    )
    test_correlation = pearson(source_test, target_test)
    return permutation, scores, test_correlation


def case_labels(
    family: str,
    label: np.ndarray,
    data: Mapping[str, object],
) -> list[bool]:
    lookup = index_lookup(data)
    cases = data["case_order"]  # type: ignore[assignment]
    return [
        bool(label[lookup[(family, case, "anchor")]])
        for case in cases[family]
    ]


def alignment_fraction(
    source_labels: Sequence[bool],
    target_labels: Sequence[bool],
    permutation: Sequence[int],
) -> float:
    return float(
        np.mean(
            [
                bool(source_labels[index])
                == bool(target_labels[permutation[index]])
                for index in range(4)
            ]
        )
    )


def all_mapping_rows(
    data: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    family_order = list(data["family_order"])
    similarities = data["similarities"]  # type: ignore[assignment]
    layers = np.asarray(data["layers"])
    depths = np.asarray(data["depths"])
    band_mask = np.asarray(data["band_mask"])
    physical = np.asarray(data["physical"])
    numeric = np.asarray(data["numeric"])
    orientation = data["response_orientation"]  # type: ignore[assignment]
    case_order = data["case_order"]  # type: ignore[assignment]

    similarity_sets: list[
        tuple[str, str, int | str, float | str, np.ndarray]
    ] = []
    for method in METHODS:
        layered = np.asarray(similarities[method])
        similarity_sets.append(
            (
                method,
                "band",
                "38-92",
                "38-92",
                np.mean(layered[band_mask], axis=0),
            )
        )
        similarity_sets.extend(
            (
                method,
                "layer",
                int(layer),
                float(depth),
                layered[index],
            )
            for index, (layer, depth) in enumerate(zip(layers, depths))
        )
    for fit_index, fit_layers in enumerate(
        np.asarray(data["jacobian_fit_cosines"])
    ):
        similarity_sets.append(
            (
                f"jacobian_fit_{fit_index}",
                "band",
                "38-92",
                "38-92",
                np.mean(fit_layers[band_mask], axis=0),
            )
        )

    for method, scope, layer, depth, similarity in similarity_sets:
        for source_family, target_family in itertools.combinations(
            family_order, 2
        ):
            permutation, scores, heldout = fit_pair_mapping(
                similarity, source_family, target_family, data
            )
            source_physical = case_labels(source_family, physical, data)
            target_physical = case_labels(target_family, physical, data)
            source_numeric = case_labels(source_family, numeric, data)
            target_numeric = case_labels(target_family, numeric, data)
            physical_agreement = alignment_fraction(
                source_physical, target_physical, permutation
            )
            numeric_agreement = alignment_fraction(
                source_numeric, target_numeric, permutation
            )
            pair_rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "layer": layer,
                    "depth_percent": depth,
                    "source_family": source_family,
                    "target_family": target_family,
                    "source_orientation": orientation[source_family],
                    "target_orientation": orientation[target_family],
                    "opposite_response_orientation": (
                        orientation[source_family]
                        != orientation[target_family]
                    ),
                    "permutation": ",".join(map(str, permutation)),
                    "mapped_target_cases": "|".join(
                        case_order[target_family][index]
                        for index in permutation
                    ),
                    "fit_correlation": max(
                        float(row["fit_correlation"]) for row in scores
                    ),
                    "heldout_correlation": heldout,
                    "physical_agreement": physical_agreement,
                    "numeric_agreement": numeric_agreement,
                    "physical_minus_numeric": (
                        physical_agreement - numeric_agreement
                    ),
                }
            )
            for score in scores:
                candidate_rows.append(
                    {
                        "method": method,
                        "scope": scope,
                        "layer": layer,
                        "depth_percent": depth,
                        "source_family": source_family,
                        "target_family": target_family,
                        "permutation": ",".join(
                            map(str, score["permutation"])
                        ),
                        "fit_correlation": score["fit_correlation"],
                        "selected": tuple(score["permutation"]) == permutation,
                    }
                )
    return pd.DataFrame(pair_rows), pd.DataFrame(candidate_rows)


def structured_label_null(
    band_rows: pd.DataFrame,
    data: Mapping[str, object],
) -> tuple[np.ndarray, float]:
    family_order = list(data["family_order"])
    balanced = list(itertools.combinations(range(4), 2))
    maps = {}
    for row in band_rows.itertuples(index=False):
        maps[(str(row.source_family), str(row.target_family))] = tuple(
            int(value) for value in str(row.permutation).split(",")
        )
    null_values = np.empty(6 ** 6, dtype=np.float64)
    for assignment_index, choices in enumerate(
        itertools.product(balanced, repeat=6)
    ):
        labels = {
            family: np.asarray(
                [index in set(choice) for index in range(4)], dtype=bool
            )
            for family, choice in zip(family_order, choices)
        }
        agreements = []
        for first, second in itertools.combinations(family_order, 2):
            agreements.extend(
                labels[first][index]
                == labels[second][maps[(first, second)][index]]
                for index in range(4)
            )
        null_values[assignment_index] = float(np.mean(agreements))
    observed = float(
        np.average(band_rows["physical_agreement"], weights=np.full(15, 4))
    )
    p_value = float(np.mean(null_values >= observed - 1e-12))
    return null_values, p_value


def mapping_summary(
    mapping_rows: pd.DataFrame,
    data: Mapping[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    family_order = list(data["family_order"])
    method_rows = []
    summary: dict[str, object] = {}
    for method in METHODS:
        band = mapping_rows[
            (mapping_rows["method"] == method)
            & (mapping_rows["scope"] == "band")
        ].copy()
        correlations = band["heldout_correlation"].to_numpy(float)
        family_means = {
            family: float(
                band[
                    (band["source_family"] == family)
                    | (band["target_family"] == family)
                ]["heldout_correlation"].mean()
            )
            for family in family_order
        }
        opposite = band[band["opposite_response_orientation"]].copy()
        differences = opposite["physical_minus_numeric"].to_numpy(float)
        null, structured_p = structured_label_null(band, data)
        physical = float(band["physical_agreement"].mean())
        numeric = float(band["numeric_agreement"].mean())
        topology_gate = bool(
            np.median(correlations) > 0
            and signflip_p(correlations) <= 0.05
            and np.sum(correlations > 0) >= 10
            and np.sum(np.asarray(list(family_means.values())) > 0) >= 5
        )
        physical_gate = bool(
            physical > 0.5
            and structured_p <= 0.05
            and np.sum(band["physical_agreement"] > 0.5) >= 10
            and np.sum(differences > 0) >= 6
            and signflip_p(differences) <= 0.05
        )
        method_summary = {
            "heldout_topology_median": float(np.median(correlations)),
            "heldout_topology_mean": float(np.mean(correlations)),
            "heldout_topology_signflip_p": signflip_p(correlations),
            "heldout_topology_positive_pairs": int(
                np.sum(correlations > 0)
            ),
            "heldout_topology_family_means": family_means,
            "heldout_topology_positive_families": int(
                np.sum(np.asarray(list(family_means.values())) > 0)
            ),
            "physical_agreement": physical,
            "physical_structured_null_mean": float(np.mean(null)),
            "physical_structured_null_p": structured_p,
            "physical_pairs_above_half": int(
                np.sum(band["physical_agreement"] > 0.5)
            ),
            "numeric_agreement": numeric,
            "opposite_response_physical_over_numeric_pairs": int(
                np.sum(differences > 0)
            ),
            "opposite_response_difference_signflip_p": signflip_p(
                differences
            ),
            "strong_topology_gate": topology_gate,
            "strong_physical_gate": physical_gate,
        }
        summary[method] = method_summary
        method_rows.append({"method": method, **method_summary})
        if method == "jacobian":
            np.savez_compressed(
                OUT / "structured_label_null.npz",
                physical_agreement=null,
                observed=np.asarray([physical]),
            )

    jacobian = mapping_rows[
        (mapping_rows["method"] == "jacobian")
        & (mapping_rows["scope"] == "band")
    ]
    direct = mapping_rows[
        (mapping_rows["method"] == "direct")
        & (mapping_rows["scope"] == "band")
    ]
    merged = jacobian.merge(
        direct,
        on=["source_family", "target_family"],
        suffixes=("_jacobian", "_direct"),
    )
    family_contrasts = []
    for family in family_order:
        selected = merged[
            (merged["source_family"] == family)
            | (merged["target_family"] == family)
        ]
        family_contrasts.append(
            float(
                np.mean(
                    selected["heldout_correlation_jacobian"]
                    - selected["heldout_correlation_direct"]
                )
            )
        )
    summary["jacobian_specificity"] = {
        "family_contrasts": dict(zip(family_order, family_contrasts)),
        "positive_families": int(np.sum(np.asarray(family_contrasts) > 0)),
        "plus_one_signflip_p": plus_one_signflip_p(family_contrasts),
        "claim_passes": bool(
            np.sum(np.asarray(family_contrasts) > 0) >= 5
            and plus_one_signflip_p(family_contrasts) <= 0.05
        ),
    }
    return summary, pd.DataFrame(method_rows)


def selected_family_graph(
    similarity: np.ndarray,
    family: str,
    data: Mapping[str, object],
) -> nx.DiGraph:
    families = np.asarray(data["families"])
    cases = np.asarray(data["cases"])
    variants = np.asarray(data["variants"])
    physical = np.asarray(data["physical"])
    numeric = np.asarray(data["numeric"])
    global_indices = np.flatnonzero(families == family)
    local = {int(global_index): i for i, global_index in enumerate(global_indices)}
    graph = nx.DiGraph()
    for global_index in global_indices:
        graph.add_node(
            local[int(global_index)],
            global_index=int(global_index),
            variant=str(variants[global_index]),
            physical=int(physical[global_index]),
            numeric=int(numeric[global_index]),
            case=str(cases[global_index]),
        )
    for global_source in global_indices:
        for target_variant in VARIANTS:
            if target_variant == variants[global_source]:
                continue
            candidates = global_indices[
                (variants[global_indices] == target_variant)
                & (cases[global_indices] != cases[global_source])
            ]
            target = min(
                candidates.tolist(),
                key=lambda index: (
                    -float(similarity[global_source, index]),
                    int(index),
                ),
            )
            graph.add_edge(
                local[int(global_source)],
                local[int(target)],
                weight=float(similarity[global_source, target]),
            )
    if graph.number_of_nodes() != 12 or graph.number_of_edges() != 24:
        raise RuntimeError("unexpected family graph size")
    return graph


def random_family_graph(
    template: nx.DiGraph, rng: np.random.Generator
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node, attributes in template.nodes(data=True):
        graph.add_node(node, **attributes)
    for source, attributes in template.nodes(data=True):
        for target_variant in VARIANTS:
            if target_variant == attributes["variant"]:
                continue
            candidates = [
                node
                for node, target_attributes in template.nodes(data=True)
                if target_attributes["variant"] == target_variant
                and target_attributes["case"] != attributes["case"]
            ]
            target = int(rng.choice(candidates))
            graph.add_edge(source, target, weight=1.0)
    return graph


def wl_features(
    graph: nx.DiGraph,
    *,
    label_scheme: str,
    height: int = 3,
) -> Counter[str]:
    if label_scheme == "none":
        labels = {node: "node" for node in graph.nodes}
    elif label_scheme == "variant":
        labels = {
            node: str(attributes["variant"])
            for node, attributes in graph.nodes(data=True)
        }
    elif label_scheme == "variant_physical":
        labels = {
            node: f"{attributes['variant']}:{attributes['physical']}"
            for node, attributes in graph.nodes(data=True)
        }
    else:
        raise ValueError(label_scheme)
    features: Counter[str] = Counter()
    for iteration in range(height + 1):
        features.update(
            f"h{iteration}:{label}" for label in labels.values()
        )
        if iteration == height:
            break
        signatures = {}
        for node in graph.nodes:
            incoming = sorted(f"I:{labels[parent]}" for parent in graph.predecessors(node))
            outgoing = sorted(f"O:{labels[child]}" for child in graph.successors(node))
            signatures[node] = "|".join([labels[node], *incoming, *outgoing])
        # Hash the complete signature so that equal rooted neighborhoods receive
        # equal labels across independently processed graphs.  A graph-local
        # integer vocabulary would make WL features incomparable between graphs.
        labels = {
            node: hashlib.sha256(signature.encode("utf-8")).hexdigest()
            for node, signature in signatures.items()
        }
    return features


def counter_cosine(first: Counter[str], second: Counter[str]) -> float:
    keys = set(first) | set(second)
    x = np.asarray([first[key] for key in keys], dtype=float)
    y = np.asarray([second[key] for key in keys], dtype=float)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def graph_signatures(graph: nx.DiGraph) -> dict[str, object]:
    indegree = np.asarray(sorted(dict(graph.in_degree()).values()), dtype=float)
    outdegree = np.asarray(sorted(dict(graph.out_degree()).values()), dtype=float)
    mutual = sum(
        1
        for first, second in graph.edges
        if graph.has_edge(second, first)
    )
    reciprocity = mutual / max(1, graph.number_of_edges())
    # The registered graph object stores cosine similarity for later GNN
    # weighting, but this endpoint compares topology only.
    adjacency = nx.to_numpy_array(
        graph, nodelist=sorted(graph.nodes), weight=None
    )
    adjacency = np.maximum(adjacency, adjacency.T)
    degree = np.diag(adjacency.sum(axis=1))
    with np.errstate(divide="ignore"):
        inverse_sqrt = np.diag(
            np.where(np.diag(degree) > 0, np.diag(degree) ** -0.5, 0.0)
        )
    laplacian = np.eye(len(adjacency)) - inverse_sqrt @ adjacency @ inverse_sqrt
    spectrum = np.sort(np.linalg.eigvalsh(laplacian))
    return {
        "degree": np.concatenate([indegree, outdegree]),
        "reciprocity": float(reciprocity),
        "spectrum": spectrum,
    }


def isomorphic(
    first: nx.DiGraph, second: nx.DiGraph, scheme: str
) -> bool:
    if scheme == "none":
        return bool(nx.is_isomorphic(first, second))

    def match(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
        if scheme == "variant":
            return left["variant"] == right["variant"]
        if scheme == "variant_physical":
            return (
                left["variant"] == right["variant"]
                and left["physical"] == right["physical"]
            )
        raise ValueError(scheme)

    return bool(nx.is_isomorphic(first, second, node_match=match))


def graph_pair_metrics(
    first: nx.DiGraph, second: nx.DiGraph
) -> dict[str, float | bool]:
    first_signature = graph_signatures(first)
    second_signature = graph_signatures(second)
    return {
        "exact_none": isomorphic(first, second, "none"),
        "exact_variant": isomorphic(first, second, "variant"),
        "exact_variant_physical": isomorphic(
            first, second, "variant_physical"
        ),
        "wl_variant": counter_cosine(
            wl_features(first, label_scheme="variant"),
            wl_features(second, label_scheme="variant"),
        ),
        "degree_distance": float(
            np.linalg.norm(
                first_signature["degree"] - second_signature["degree"]
            )
        ),
        "reciprocity_distance": abs(
            float(first_signature["reciprocity"])
            - float(second_signature["reciprocity"])
        ),
        "spectral_distance": float(
            np.linalg.norm(
                first_signature["spectrum"]
                - second_signature["spectrum"]
            )
        ),
    }


def isomorphism_analysis(
    data: Mapping[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    family_order = list(data["family_order"])
    similarities = data["similarities"]  # type: ignore[assignment]
    layers = np.asarray(data["layers"])
    depths = np.asarray(data["depths"])
    band_mask = np.asarray(data["band_mask"])
    records = []
    for method in METHODS:
        layered = np.asarray(similarities[method])
        scopes = [
            ("band", "38-92", "38-92", np.mean(layered[band_mask], axis=0))
        ]
        scopes.extend(
            (
                "layer",
                int(layer),
                float(depth),
                layered[index],
            )
            for index, (layer, depth) in enumerate(zip(layers, depths))
        )
        for scope, layer, depth, similarity in scopes:
            graphs = {
                family: selected_family_graph(similarity, family, data)
                for family in family_order
            }
            for first, second in itertools.combinations(family_order, 2):
                records.append(
                    {
                        "method": method,
                        "scope": scope,
                        "layer": layer,
                        "depth_percent": depth,
                        "source_family": first,
                        "target_family": second,
                        **graph_pair_metrics(graphs[first], graphs[second]),
                    }
                )
    observed = pd.DataFrame(records)

    jacobian_layers = np.asarray(similarities["jacobian"])
    band_similarity = np.mean(jacobian_layers[band_mask], axis=0)
    templates = {
        family: selected_family_graph(band_similarity, family, data)
        for family in family_order
    }
    rng = np.random.default_rng(SEED)
    null_rows = []
    for null_index in range(N_NULL):
        graphs = {
            family: random_family_graph(templates[family], rng)
            for family in family_order
        }
        pair_metrics = [
            graph_pair_metrics(graphs[first], graphs[second])
            for first, second in itertools.combinations(family_order, 2)
        ]
        row = {"null_index": null_index}
        row.update(
            {
                key: float(
                    np.mean([float(item[key]) for item in pair_metrics])
                )
                for key in (
                    "exact_none",
                    "exact_variant",
                    "exact_variant_physical",
                    "wl_variant",
                    "degree_distance",
                    "reciprocity_distance",
                    "spectral_distance",
                )
            }
        )
        null_rows.append(row)
    null = pd.DataFrame(null_rows)
    band_observed = observed[
        (observed["method"] == "jacobian") & (observed["scope"] == "band")
    ]
    metric_directions = {
        "exact_none": "high",
        "exact_variant": "high",
        "exact_variant_physical": "high",
        "wl_variant": "high",
        "degree_distance": "low",
        "reciprocity_distance": "low",
        "spectral_distance": "low",
    }
    test_rows = []
    for metric, direction in metric_directions.items():
        value = float(band_observed[metric].astype(float).mean())
        if direction == "high":
            p_value = float(np.mean(null[metric] >= value - 1e-12))
        else:
            p_value = float(np.mean(null[metric] <= value + 1e-12))
        test_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "observed": value,
                "null_mean": float(null[metric].mean()),
                "null_sd": float(null[metric].std(ddof=1)),
                "exact_p": p_value,
            }
        )
    tests = pd.DataFrame(test_rows)
    tests["bh_q"] = bh_adjust(tests["exact_p"])
    primary = tests.set_index("metric").loc["wl_variant"]
    summary = {
        "primary_wl_observed": float(primary["observed"]),
        "primary_wl_null_mean": float(primary["null_mean"]),
        "primary_wl_exact_p": float(primary["exact_p"]),
        "primary_wl_bh_q": float(primary["bh_q"]),
        "primary_constrained_null_gate": bool(
            float(primary["exact_p"]) <= 0.05
        ),
        "exact_variant_pairs_in_band": int(
            band_observed["exact_variant"].sum()
        ),
        "exact_variant_physical_pairs_in_band": int(
            band_observed["exact_variant_physical"].sum()
        ),
    }
    return observed, null, {"summary": summary, "tests": test_rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    validate_inputs(protocol)
    data = load_data()
    mapping_rows, candidate_rows = all_mapping_rows(data)
    mapping_summary_data, mapping_table = mapping_summary(mapping_rows, data)
    iso_rows, null_rows, iso_summary = isomorphism_analysis(data)

    mapping_rows.to_csv(OUT / "pair_mappings.csv", index=False)
    candidate_rows.to_csv(OUT / "all_mapping_candidates.csv", index=False)
    mapping_table.to_csv(OUT / "mapping_method_summary.csv", index=False)
    iso_rows.to_csv(OUT / "isomorphism_pair_metrics.csv", index=False)
    null_rows.to_csv(OUT / "constrained_graph_null.csv", index=False)
    pd.DataFrame(iso_summary["tests"]).to_csv(
        OUT / "isomorphism_tests.csv", index=False
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
    }
    results = {
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256(PROTOCOL),
        "protocol_markdown_sha256": sha256(PROTOCOL_MD),
        "input_sha256": {
            "representations": sha256(STATES),
            "prompt_manifest": sha256(MANIFEST),
        },
        "dimensions": {
            "families": 6,
            "family_pairs": 15,
            "nodes_per_family_graph": 12,
            "edges_per_family_graph": 24,
            "layers": 25,
            "mapping_candidate_rows": len(candidate_rows),
            "mapping_result_rows": len(mapping_rows),
            "constrained_null_sets": len(null_rows),
        },
        "study_1_mapping": mapping_summary_data,
        "study_2_isomorphism": iso_summary,
        "environment": environment,
    }
    (OUT / "study12_statistics.json").write_text(
        json.dumps(safe_json(results), indent=2) + "\n"
    )
    print(json.dumps(safe_json(results), indent=2))


if __name__ == "__main__":
    main()
