#!/usr/bin/env python3
"""Rigorous graph controls for materials-mechanism representations.

This script implements the frozen protocol in
``experiments/graph-topology-rigorous-2026-07-17``.  It uses only archived
representations and writes complete row-level tables, statistics, and figures.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import adjusted_rand_score, roc_auc_score  # noqa: E402

from analyze_graph_topology import (  # noqa: E402
    FAMILY_SHORT,
    candidate_removed,
    community_labels,
    normalize,
    token_words,
    vocabulary_features,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "graph-topology-rigorous-2026-07-17"
FIG = OUT / "figures"
PROTOCOL_PATH = OUT / "protocol.json"
AMENDMENT_PATH = OUT / "protocol-amendment-v1.json"
AMENDMENT_V2_PATH = OUT / "protocol-amendment-v2.json"
AMENDMENT_V2_CORRECTION_PATH = OUT / "protocol-amendment-v2-correction.json"
HELDOUT_NPZ = ROOT / "experiments" / "materials-heldout-v1_latent_vectors.npz"
HELDOUT_STATS = ROOT / "experiments" / "materials-heldout-v1_statistics.json"
HELDOUT_PROMPTS = ROOT / "prompts" / "materials-heldout-v1-preregistered.json"
REPLICATION_DIR = (
    ROOT
    / "experiments"
    / "late-physics-representation-replication-2026-07-17"
)
REPLICATION_NPZ = REPLICATION_DIR / "representations.npz"
REPLICATION_MANIFEST = REPLICATION_DIR / "prompt_manifest.json"
REPLICATION_PROTOCOL = REPLICATION_DIR / "protocol.json"

SEED = 20260717
N_PERM = 50_000
N_BOOT = 50_000
BAND = (38.0, 92.0)
LATE_BAND = (80.0, 96.0)

COLORS = {
    "jacobian": "#16697A",
    "raw": "#6C5B7B",
    "direct": "#B56576",
    "token_embedding": "#D17C38",
    "word_tfidf": "#5B8C5A",
    "char_tfidf": "#73777D",
    "target_free_jacobian": "#247BA0",
    "target_free_direct": "#C06C84",
    "chance": "#9AA0A6",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value: object) -> object:
    """Recursively convert NumPy scalars before serializing audit artifacts."""

    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def validate_inputs(protocol: Mapping[str, object]) -> None:
    lookup = {
        "heldout_vectors": HELDOUT_NPZ,
        "heldout_statistics": HELDOUT_STATS,
        "heldout_prompts": HELDOUT_PROMPTS,
        "replication_vectors": REPLICATION_NPZ,
        "replication_manifest": REPLICATION_MANIFEST,
        "replication_protocol": REPLICATION_PROTOCOL,
    }
    for name, path in lookup.items():
        expected = str(protocol["inputs"][name]["sha256"])  # type: ignore[index]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"input fingerprint mismatch for {path}: {actual} != {expected}"
            )


def cosine_layers(values: np.ndarray) -> np.ndarray:
    """Return pairwise cosine matrices for ``[..., item, layer, feature]``."""

    values = normalize(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        output = np.einsum(
            "...ild,...jld->...lij", values, values, optimize=True
        )
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("non-finite layered cosine matrix")
    return output


def cosine_matrix(values: np.ndarray) -> np.ndarray:
    values = normalize(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        output = np.einsum("id,jd->ij", values, values, optimize=True)
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("non-finite cosine matrix")
    return output


def prompt_feature_matrices(texts: Sequence[str]) -> dict[str, np.ndarray]:
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    ).fit_transform(texts)
    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        norm="l2",
    ).fit_transform(texts)
    token_sets = [token_words(text) for text in texts]
    jaccard = np.zeros((len(texts), len(texts)), dtype=float)
    lengths = np.asarray([len(words) for words in token_sets], dtype=float)
    for i, first in enumerate(token_sets):
        for j in range(i, len(texts)):
            second = token_sets[j]
            union = first | second
            value = len(first & second) / len(union) if union else 0.0
            jaccard[i, j] = jaccard[j, i] = value
    return {
        "word_tfidf": (word @ word.T).toarray(),
        "char_tfidf": (char @ char.T).toarray(),
        "token_jaccard": jaccard,
        "length_difference": np.abs(lengths[:, None] - lengths[None, :]),
    }


def cross_fold_edges(
    similarity: np.ndarray,
    folds: np.ndarray,
    *,
    neighbors_per_fold: int = 1,
) -> np.ndarray:
    """Select a fixed number of directed neighbors in every other fold."""

    edges: list[tuple[int, int]] = []
    for source in range(len(folds)):
        for target_fold in sorted(set(folds)):
            if target_fold == folds[source]:
                continue
            candidates = np.flatnonzero(folds == target_fold)
            ordered = sorted(
                candidates.tolist(),
                key=lambda target: (-float(similarity[source, target]), target),
            )
            edges.extend(
                (source, target)
                for target in ordered[: min(neighbors_per_fold, len(ordered))]
            )
    return np.asarray(edges, dtype=np.int16)


def relation_edges(
    similarity: np.ndarray,
    families: np.ndarray,
    variants: np.ndarray,
    triplets: np.ndarray,
) -> np.ndarray:
    """Within-family, cross-variant nearest edges excluding the same case."""

    edges: list[tuple[int, int]] = []
    variant_values = sorted(set(variants))
    for source in range(len(families)):
        for target_variant in variant_values:
            if target_variant == variants[source]:
                continue
            candidates = np.flatnonzero(
                (families == families[source])
                & (variants == target_variant)
                & (triplets != triplets[source])
            )
            if not len(candidates):
                raise RuntimeError("empty disjoint relation candidate set")
            target = min(
                candidates.tolist(),
                key=lambda candidate: (
                    -float(similarity[source, candidate]),
                    candidate,
                ),
            )
            edges.append((source, target))
    return np.asarray(edges, dtype=np.int16)


def case_preserving_exact_permutations(
    families: np.ndarray,
    variants: np.ndarray,
    triplets: np.ndarray,
    outcomes: np.ndarray,
) -> np.ndarray:
    """Enumerate case labels while preserving the counterfactual sign flip."""

    family_names = sorted(set(families))
    family_configurations: list[
        tuple[np.ndarray, list[np.ndarray]]
    ] = []
    for family in family_names:
        family_indices = np.flatnonzero(families == family)
        family_triplets = sorted(set(triplets[family_indices]))
        if len(family_triplets) != 4:
            raise RuntimeError(f"{family} does not contain four material cases")
        for triplet in family_triplets:
            case = (families == family) & (triplets == triplet)
            anchor = outcomes[case & (variants == "anchor")]
            paraphrase = outcomes[
                case & (variants == "physics_paraphrase")
            ]
            counterfactual = outcomes[
                case & (variants == "lexical_counterfactual")
            ]
            if not (
                len(anchor) == len(paraphrase) == len(counterfactual) == 1
                and anchor[0] == paraphrase[0]
                and anchor[0] != counterfactual[0]
            ):
                raise RuntimeError(
                    "material case does not have the registered sign transform"
                )
        configurations = []
        for positive in itertools.combinations(range(4), 2):
            positive_set = set(positive)
            assignment = np.zeros(len(families), dtype=np.int8)
            for index, triplet in enumerate(family_triplets):
                case = (families == family) & (triplets == triplet)
                base = int(index in positive_set)
                assignment[
                    case & (variants != "lexical_counterfactual")
                ] = base
                assignment[
                    case & (variants == "lexical_counterfactual")
                ] = 1 - base
            configurations.append(assignment)
        family_configurations.append((family_indices, configurations))
    combinations = list(
        itertools.product(range(6), repeat=len(family_names))
    )
    output = np.empty((len(combinations), len(families)), dtype=np.int8)
    for row_index, choice in enumerate(combinations):
        row = np.empty(len(families), dtype=np.int8)
        for family_index, configuration_index in enumerate(choice):
            indices, configurations = family_configurations[family_index]
            row[indices] = configurations[configuration_index][indices]
        output[row_index] = row
    if len(output) != 6 ** len(family_names):
        raise RuntimeError("incorrect exact case-permutation count")
    return output


def relation_ranking_summary(
    similarity: np.ndarray,
    families: np.ndarray,
    variants: np.ndarray,
    triplets: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate the complete three-candidate ranking behind each graph edge."""

    rows: list[dict[str, object]] = []
    for source in range(len(families)):
        for target_variant in sorted(set(variants)):
            if target_variant == variants[source]:
                continue
            candidates = np.flatnonzero(
                (families == families[source])
                & (variants == target_variant)
                & (triplets != triplets[source])
            )
            same = outcomes[candidates] == outcomes[source]
            if len(candidates) != 3 or int(np.sum(same)) not in (1, 2):
                raise RuntimeError(
                    "relation ranking requires one or two positives among three candidates"
                )
            positives = candidates[same]
            negatives = candidates[~same]
            ordered = sorted(
                candidates.tolist(),
                key=lambda candidate: (
                    -float(similarity[source, candidate]),
                    candidate,
                ),
            )
            rank = min(ordered.index(int(positive)) + 1 for positive in positives)
            positive_scores = similarity[source, positives]
            negative_scores = similarity[source, negatives]
            comparisons = positive_scores[:, None] - negative_scores[None, :]
            auc = float(
                np.mean(comparisons > 0)
                + 0.5 * np.mean(comparisons == 0)
            )
            rows.append(
                {
                    "source_index": source,
                    "family": str(families[source]),
                    "source_variant": str(variants[source]),
                    "target_variant": str(target_variant),
                    "n_same_outcome_candidates": int(len(positives)),
                    "first_positive_rank": rank,
                    "top1_correct": bool(
                        outcomes[ordered[0]] == outcomes[source]
                    ),
                    "reciprocal_rank": 1.0 / rank,
                    "pairwise_auc": auc,
                }
            )
    frame = pd.DataFrame(rows)
    family = (
        frame.groupby("family")[
            ["top1_correct", "reciprocal_rank", "pairwise_auc"]
        ]
        .mean()
        .sort_index()
    )
    leave_one_out = []
    for omitted in family.index:
        retained = frame[frame["family"] != omitted]
        leave_one_out.append(
            {
                "omitted_family": str(omitted),
                "top1_accuracy": float(retained["top1_correct"].mean()),
                "mean_reciprocal_rank": float(
                    retained["reciprocal_rank"].mean()
                ),
                "pairwise_auc": float(retained["pairwise_auc"].mean()),
            }
        )
    output = {
        "n_rankings": int(len(frame)),
        "candidates_per_ranking": 3,
        "same_outcome_candidates_range": [1, 2],
        "top1_accuracy": float(frame["top1_correct"].mean()),
        "mean_reciprocal_rank": float(frame["reciprocal_rank"].mean()),
        "pairwise_auc": float(frame["pairwise_auc"].mean()),
        "family_metrics": {
            str(name): {
                "top1_accuracy": float(row["top1_correct"]),
                "mean_reciprocal_rank": float(row["reciprocal_rank"]),
                "pairwise_auc": float(row["pairwise_auc"]),
            }
            for name, row in family.iterrows()
        },
        "leave_one_family_out": leave_one_out,
        "leave_one_family_out_range": {
            metric: [
                float(min(row[metric] for row in leave_one_out)),
                float(max(row[metric] for row in leave_one_out)),
            ]
            for metric in (
                "top1_accuracy",
                "mean_reciprocal_rank",
                "pairwise_auc",
            )
        },
        "chance": {
            "top1_accuracy": float(
                frame["n_same_outcome_candidates"].mean() / 3.0
            ),
            "pairwise_auc": 0.5,
        },
    }
    return output, frame


def edge_precision(edges: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(labels[edges[:, 0]] == labels[edges[:, 1]]))


def per_query_precision(edges: np.ndarray, labels: np.ndarray) -> np.ndarray:
    output = np.zeros(len(labels), dtype=float)
    counts = np.zeros(len(labels), dtype=int)
    for source, target in edges:
        output[source] += labels[source] == labels[target]
        counts[source] += 1
    if np.any(counts == 0):
        raise RuntimeError("graph contains a node with no directed edges")
    return output / counts


def family_means(
    values: np.ndarray, families: np.ndarray
) -> tuple[list[str], np.ndarray]:
    names = sorted(set(families))
    return names, np.asarray(
        [np.mean(values[families == family]) for family in names], dtype=float
    )


def plus_one_p(null: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(null >= observed - 1e-15)) / (1 + len(null)))


def sign_flip_test(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    observed = float(np.mean(values))
    null = np.mean(signs * values[None, :], axis=1)
    return float(np.mean(null >= observed - 1e-15))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def permute_within_blocks(
    labels: np.ndarray,
    blocks: Sequence[tuple[object, ...]],
    *,
    rng: np.random.Generator,
    n: int = N_PERM,
) -> np.ndarray:
    output = np.empty((n, len(labels)), dtype=labels.dtype)
    unique = sorted(set(blocks))
    indices = [
        np.asarray([i for i, block in enumerate(blocks) if block == value])
        for value in unique
    ]
    for iteration in range(n):
        row = labels.copy()
        for block_indices in indices:
            row[block_indices] = rng.permutation(row[block_indices])
        output[iteration] = row
    return output


def precision_null(edges: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    return np.mean(
        permutations[:, edges[:, 0]] == permutations[:, edges[:, 1]],
        axis=1,
    )


def layer_scan_null(
    edges_by_layer: Sequence[np.ndarray],
    permutations: np.ndarray,
) -> np.ndarray:
    values = np.empty((len(permutations), len(edges_by_layer)), dtype=np.float32)
    for layer, edges in enumerate(edges_by_layer):
        values[:, layer] = precision_null(edges, permutations)
    return values


def prompt_covariate_design(
    feature_matrices: Mapping[str, np.ndarray],
    folds: np.ndarray,
    *,
    upper_only: bool = True,
    cross_fold_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if upper_only:
        first, second = np.triu_indices(len(folds), k=1)
    else:
        first, second = np.where(~np.eye(len(folds), dtype=bool))
    if cross_fold_only:
        keep = folds[first] != folds[second]
        first, second = first[keep], second[keep]
    length = feature_matrices["length_difference"]
    length_scaled = length / max(1.0, float(np.max(length)))
    columns = [
        np.ones(len(first)),
        feature_matrices["word_tfidf"][first, second],
        feature_matrices["char_tfidf"][first, second],
        feature_matrices["token_jaccard"][first, second],
        length_scaled[first, second],
        (folds[first] == folds[second]).astype(float),
    ]
    return np.column_stack(columns), first, second


def residualize_similarity(
    similarity: np.ndarray,
    design: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(similarity[first, second], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - np.einsum("ij,j->i", design, coefficient)
    if not np.all(np.isfinite(residual)):
        raise FloatingPointError("non-finite prompt-residual similarity")
    matrix = np.zeros_like(similarity, dtype=float)
    matrix[first, second] = residual
    matrix[second, first] = residual
    np.fill_diagonal(matrix, np.max(residual) + 1.0)
    return matrix, coefficient


def partial_family_coefficient(
    similarity: np.ndarray,
    design: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    labels: np.ndarray,
) -> float:
    same = (labels[first] == labels[second]).astype(float)
    full = np.column_stack((design, same))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coefficient = np.linalg.lstsq(
            full, similarity[first, second], rcond=None
        )[0][-1]
    if not np.isfinite(coefficient):
        raise FloatingPointError("non-finite partial family coefficient")
    return float(coefficient)


def partial_coefficient_null(
    similarity: np.ndarray,
    design: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    permutations: np.ndarray,
) -> np.ndarray:
    y = similarity[first, second]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        projection = np.einsum(
            "ij,jk->ik", design, np.linalg.pinv(design), optimize=True
        )
        y_residual = y - np.einsum(
            "ij,j->i", projection, y, optimize=True
        )
    if not np.all(np.isfinite(y_residual)):
        raise FloatingPointError("non-finite residualized dyadic outcome")
    output = np.empty(len(permutations), dtype=float)
    chunk = 500
    for start in range(0, len(permutations), chunk):
        stop = min(len(permutations), start + chunk)
        same = (
            permutations[start:stop, first]
            == permutations[start:stop, second]
        ).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            same_residual = same - np.einsum(
                "ij,jk->ik", same, projection, optimize=True
            )
            numerator = np.einsum(
                "ij,j->i", same_residual, y_residual, optimize=True
            )
        denominator = np.sum(same_residual * same_residual, axis=1)
        output[start:stop] = numerator / np.maximum(denominator, 1e-12)
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("non-finite partial-coefficient null")
    return output


def hard_negative_rows(
    similarities: Mapping[str, np.ndarray],
    lexical_score: np.ndarray,
    families: np.ndarray,
    folds: np.ndarray,
    slugs: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in range(len(families)):
        for target_fold in sorted(set(folds)):
            if target_fold == folds[source]:
                continue
            true_matches = np.flatnonzero(
                (folds == target_fold) & (families == families[source])
            )
            if len(true_matches) != 1:
                raise RuntimeError("cross-phrasing target is not unique")
            target = int(true_matches[0])
            candidates = np.flatnonzero(
                (folds == target_fold) & (families != families[source])
            )
            negative = min(
                candidates.tolist(),
                key=lambda index: (-float(lexical_score[source, index]), index),
            )
            lexical_margin = float(
                lexical_score[source, target] - lexical_score[source, negative]
            )
            base = {
                "source": source,
                "source_slug": str(slugs[source]),
                "family": str(families[source]),
                "source_fold": int(folds[source]),
                "target_fold": int(target_fold),
                "true_target": target,
                "true_slug": str(slugs[target]),
                "hard_negative": negative,
                "hard_negative_slug": str(slugs[negative]),
                "hard_negative_family": str(families[negative]),
                "prompt_only_margin": lexical_margin,
                "lexical_trap": bool(lexical_margin < 0),
            }
            for method, similarity in similarities.items():
                rows.append(
                    {
                        **base,
                        "method": method,
                        "state_margin": float(
                            similarity[source, target]
                            - similarity[source, negative]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_hard_negatives(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for subset_name, subset in (
        ("all", frame),
        ("lexical_traps", frame[frame["lexical_trap"]]),
    ):
        method_results = {}
        for method, rows in subset.groupby("method"):
            family = rows.groupby("family")["state_margin"].mean().sort_index()
            values = family.to_numpy(dtype=float)
            method_results[str(method)] = {
                "n_comparisons": int(len(rows)),
                "n_families": int(len(values)),
                "mean_family_margin": float(np.mean(values)),
                "family_bootstrap_95": bootstrap_mean(values, rng),
                "one_sided_exact_family_sign_flip_p": sign_flip_test(values),
                "positive_families": int(np.sum(values > 0)),
                "family_margins": {
                    str(name): float(value)
                    for name, value in family.items()
                },
            }
        output[subset_name] = method_results
    return output


def stability_scores(
    similarities: np.ndarray,
    folds: np.ndarray,
    band_mask: np.ndarray,
) -> np.ndarray:
    """Frequency for every directed cross-fold pair.

    ``similarities`` is ``[replicate, layer, item, item]``.
    """

    counts = np.zeros((len(folds), len(folds)), dtype=float)
    total = 0
    for replicate in range(similarities.shape[0]):
        for layer in np.flatnonzero(band_mask):
            edges = cross_fold_edges(similarities[replicate, layer], folds)
            counts[edges[:, 0], edges[:, 1]] += 1
            total += 1
    return counts / total


def stability_auc_null(
    scores: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    permutations: np.ndarray,
) -> tuple[float, np.ndarray]:
    first, second = np.where(folds[:, None] != folds[None, :])
    values = scores[first, second]
    observed_binary = labels[first] == labels[second]
    observed = float(roc_auc_score(observed_binary, values))
    ranks = rankdata(values, method="average")
    n_pos = int(np.sum(observed_binary))
    n_total = len(values)
    output = np.empty(len(permutations), dtype=float)
    chunk = 500
    base = n_pos * (n_pos + 1) / 2.0
    denominator = n_pos * (n_total - n_pos)
    for start in range(0, len(permutations), chunk):
        stop = min(len(permutations), start + chunk)
        same = (
            permutations[start:stop, first]
            == permutations[start:stop, second]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rank_sum = np.einsum(
                "ij,j->i", same, ranks, optimize=True
            )
        output[start:stop] = (rank_sum - base) / denominator
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("non-finite stability AUC null")
    return observed, output


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def concept_documents(records: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, list[str]]:
    documents: list[dict[str, float]] = []
    for record in records:
        words = token_words(str(record["prompt"]))
        filtered = [
            row
            for row in record["filtered_consensus_candidates"]  # type: ignore[index]
            if not candidate_removed(str(row["token"]), words)
        ][:20]
        documents.append(
            {
                str(row["token"]): float(row["consensus_score"])
                for row in filtered
            }
        )
    frequency = Counter(token for document in documents for token in document)
    vocabulary = sorted(frequency)
    lookup = {token: index for index, token in enumerate(vocabulary)}
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=float)
    for row, document in enumerate(documents):
        for token, score in document.items():
            matrix[row, lookup[token]] = score * math.log(
                (1 + len(documents)) / (1 + frequency[token])
            )
    return matrix, vocabulary


def concept_enrichment(
    matrix: np.ndarray,
    vocabulary: Sequence[str],
    families: np.ndarray,
    permutations: np.ndarray,
) -> pd.DataFrame:
    family_names = sorted(set(families))
    family_codes = {family: index for index, family in enumerate(family_names)}
    observed_codes = np.asarray([family_codes[value] for value in families])
    perm_codes = np.empty_like(permutations, dtype=np.int16)
    for family, code in family_codes.items():
        perm_codes[permutations == family] = code
    global_mean = np.mean(matrix, axis=0)
    observed_means = np.vstack(
        [np.mean(matrix[observed_codes == code], axis=0) for code in range(10)]
    )
    observed_score = np.max(observed_means - global_mean[None, :], axis=0)
    observed_family = np.argmax(observed_means - global_mean[None, :], axis=0)
    exceed = np.zeros(len(vocabulary), dtype=np.int64)
    for token_index in range(len(vocabulary)):
        support = np.flatnonzero(matrix[:, token_index] > 0)
        weights = matrix[support, token_index]
        if not len(support):
            continue
        scores = np.full((len(permutations), 10), -global_mean[token_index])
        for code in range(10):
            scores[:, code] += np.sum(
                (perm_codes[:, support] == code) * weights[None, :],
                axis=1,
            ) / 5.0
        null = np.max(scores, axis=1)
        exceed[token_index] = int(
            np.sum(null >= observed_score[token_index] - 1e-15)
        )
    p_values = (1 + exceed) / (1 + len(permutations))
    q_values = bh_adjust(p_values)
    rows = []
    for index, token in enumerate(vocabulary):
        family_index = int(observed_family[index])
        family = family_names[family_index]
        rows.append(
            {
                "token": token,
                "assigned_family": family,
                "family_mean": float(observed_means[family_index, index]),
                "global_mean": float(global_mean[index]),
                "enrichment": float(observed_score[index]),
                "prompt_support": int(np.sum(matrix[:, index] > 0)),
                "permutation_p": float(p_values[index]),
                "bh_q": float(q_values[index]),
                "fdr_05": bool(q_values[index] <= 0.05),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fdr_05", "bh_q", "enrichment", "token"],
        ascending=[False, True, False, True],
    )


def graph_from_directed(
    edges: np.ndarray,
    similarity: np.ndarray,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(similarity.shape[0]))
    for first, second in edges:
        weight = float(similarity[first, second])
        if graph.has_edge(int(first), int(second)):
            graph[int(first)][int(second)]["weight"] = max(
                graph[int(first)][int(second)]["weight"], weight
            )
        else:
            graph.add_edge(int(first), int(second), weight=weight)
    return graph


def graph_summary(
    similarity: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    permutations: np.ndarray,
    *,
    neighbors_per_fold: int = 1,
) -> tuple[dict[str, object], np.ndarray]:
    edges = cross_fold_edges(
        similarity, folds, neighbors_per_fold=neighbors_per_fold
    )
    observed = edge_precision(edges, labels)
    null = precision_null(edges, permutations)
    query = per_query_precision(edges, labels)
    names, family = family_means(query, labels)
    graph = graph_from_directed(edges, similarity)
    partition = community_labels(graph)
    return (
        {
            "directed_precision": observed,
            "permutation_p": plus_one_p(null, observed),
            "null_mean": float(np.mean(null)),
            "null_95": [float(value) for value in np.quantile(null, (0.025, 0.975))],
            "n_directed_edges": int(len(edges)),
            "n_undirected_edges": int(graph.number_of_edges()),
            "louvain_ari": float(adjusted_rand_score(labels, partition)),
            "n_communities": int(len(set(partition))),
            "family_precision": {
                name: float(value) for name, value in zip(names, family)
            },
        },
        edges,
    )


def undirected_edge_sensitivity(
    edges: np.ndarray,
    labels: np.ndarray,
    permutations: np.ndarray,
) -> dict[str, object]:
    directed = {(int(first), int(second)) for first, second in edges}
    union = sorted(
        {tuple(sorted((first, second))) for first, second in directed}
    )
    mutual = sorted(
        {
            tuple(sorted((first, second)))
            for first, second in directed
            if (second, first) in directed
        }
    )
    output: dict[str, object] = {}
    for name, values in (("union", union), ("mutual", mutual)):
        array = np.asarray(values, dtype=np.int16)
        if not len(array):
            output[name] = {
                "n_edges": 0,
                "homophily": None,
                "permutation_p": None,
            }
            continue
        observed = edge_precision(array, labels)
        null = precision_null(array, permutations)
        output[name] = {
            "n_edges": int(len(array)),
            "homophily": observed,
            "permutation_p": plus_one_p(null, observed),
            "null_95": [
                float(value) for value in np.quantile(null, (0.025, 0.975))
            ],
        }
    return output


def edge_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    first_set = {(int(a), int(b)) for a, b in first}
    second_set = {(int(a), int(b)) for a, b in second}
    return float(len(first_set & second_set) / len(first_set | second_set))


def relation_edge_diagnostics(
    method: str,
    similarity: np.ndarray,
    edges: np.ndarray,
    metadata: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    rows = []
    for source, target in edges:
        first = metadata[int(source)]
        second = metadata[int(target)]
        rows.append(
            {
                "method": method,
                "source_index": int(source),
                "target_index": int(target),
                "family": str(first["family_id"]),
                "source_variant": str(first["variant"]),
                "target_variant": str(second["variant"]),
                "source_presentation_order": str(first["presentation_order"]),
                "target_presentation_order": str(second["presentation_order"]),
                "same_presentation_order": bool(
                    first["presentation_order"] == second["presentation_order"]
                ),
                "source_numeric_direction": str(first["numeric_direction"]),
                "target_numeric_direction": str(second["numeric_direction"]),
                "same_numeric_direction": bool(
                    first["numeric_direction"] == second["numeric_direction"]
                ),
                "source_expected_outcome": str(first["expected_outcome"]),
                "target_expected_outcome": str(second["expected_outcome"]),
                "same_outcome": bool(
                    first["expected_outcome"] == second["expected_outcome"]
                ),
                "cosine": float(similarity[int(source), int(target)]),
            }
        )
    return pd.DataFrame(rows)


def deterministic_binary_similarity(
    values: Sequence[object],
) -> np.ndarray:
    values = np.asarray(values)
    similarity = (values[:, None] == values[None, :]).astype(float)
    # A tiny deterministic term makes ties reproducible without changing the
    # binary grouping represented by the baseline.
    index = np.arange(len(values), dtype=float)
    similarity -= 1e-9 * index[None, :]
    return similarity


def relation_family_contrasts(
    fixed_edges: Mapping[str, np.ndarray],
    outcomes: np.ndarray,
    families: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, object]:
    query = {
        method: per_query_precision(edges, outcomes)
        for method, edges in fixed_edges.items()
    }
    output = {}
    for first, second in (
        ("jacobian_band", "direct_band"),
        ("jacobian_band", "raw_band"),
        ("jacobian_band", "word_tfidf"),
        ("jacobian_band", "char_tfidf"),
        ("jacobian_late", "direct_late"),
        ("jacobian_late", "raw_late"),
    ):
        names, values = family_means(query[first] - query[second], families)
        output[f"{first}_minus_{second}"] = {
            "mean_difference": float(np.mean(values)),
            "family_bootstrap_95": bootstrap_mean(values, rng),
            "one_sided_exact_family_sign_flip_p": sign_flip_test(values),
            "positive_families": int(np.sum(values > 0)),
            "family_differences": {
                name: float(value) for name, value in zip(names, values)
            },
        }
    return output


def analyze_heldout(
    protocol: Mapping[str, object],
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, object]]:
    with np.load(HELDOUT_NPZ, allow_pickle=False) as data:
        transported = data["transported_states"].astype(np.float64)
        raw = data["raw_states"].astype(np.float64)
        lexical = data["lexical_states"].astype(np.float64)
        target = data["target_states"].astype(np.float64)
        families = data["families"].astype(str)
        depths = data["depths"].astype(float)
        layers = data["source_layers"].astype(int)
        slugs = data["slugs"].astype(str)
        phrasing_ids = data["phrasing_ids"].astype(str)
    folds = np.asarray(
        [int(value.rsplit("-", 1)[-1]) for value in phrasing_ids],
        dtype=np.int16,
    )
    prompt_manifest = json.loads(HELDOUT_PROMPTS.read_text())
    prompt_by_slug = {
        str(row["slug"]): str(row["text"])
        for row in prompt_manifest["prompts"]
    }
    texts = [prompt_by_slug[str(slug)] for slug in slugs]
    features = prompt_feature_matrices(texts)

    sim_j_seed = cosine_layers(transported)
    sim_j = np.mean(sim_j_seed, axis=0)
    sim_raw = cosine_layers(raw[None, ...])[0]
    sim_lexical = cosine_matrix(lexical)
    sim_target = cosine_matrix(target)
    band = (depths >= BAND[0]) & (depths <= BAND[1])
    band_j = np.mean(sim_j[band], axis=0)
    band_raw = np.mean(sim_raw[band], axis=0)

    heldout = json.loads(HELDOUT_STATS.read_text())
    jac_records = heldout["open_vocabulary"]["methods"]["jacobian"]["per_prompt"]
    direct_records = heldout["open_vocabulary"]["methods"]["logit"]["per_prompt"]
    if [row["slug"] for row in jac_records] != slugs.tolist():
        raise RuntimeError("heldout target-free records do not match vector order")
    vocabulary_j, vocabulary_j_words = vocabulary_features(jac_records)
    vocabulary_d, vocabulary_d_words = vocabulary_features(direct_records)
    sim_vocabulary_j = cosine_matrix(vocabulary_j)
    sim_vocabulary_d = cosine_matrix(vocabulary_d)

    label_blocks = [(int(fold),) for fold in folds]
    permutations = permute_within_blocks(
        families, label_blocks, rng=np.random.default_rng(SEED)
    )

    fixed_similarities = {
        "jacobian": band_j,
        "raw": band_raw,
        "token_embedding": sim_lexical,
        "target_state": sim_target,
        "word_tfidf": features["word_tfidf"],
        "char_tfidf": features["char_tfidf"],
        "target_free_jacobian": sim_vocabulary_j,
        "target_free_direct": sim_vocabulary_d,
    }
    primary = {}
    primary_edges = {}
    for method, similarity in fixed_similarities.items():
        primary[method], primary_edges[method] = graph_summary(
            similarity, families, folds, permutations
        )

    layer_rows = []
    layer_edges: dict[str, list[np.ndarray]] = {"jacobian": [], "raw": []}
    for index, depth in enumerate(depths):
        for method, similarity in (
            ("jacobian", sim_j[index]),
            ("raw", sim_raw[index]),
        ):
            edges = cross_fold_edges(similarity, folds)
            layer_edges[method].append(edges)
            layer_rows.append(
                {
                    "method": method,
                    "layer": int(layers[index]),
                    "depth_percent": float(depth),
                    "directed_precision": edge_precision(edges, families),
                }
            )
    layer_frame = pd.DataFrame(layer_rows)
    layer_frame.to_csv(OUT / "heldout_cross_phrasing_layer_metrics.csv", index=False)
    layer_scan = {}
    for method in ("jacobian", "raw"):
        observed = layer_frame[layer_frame["method"] == method][
            "directed_precision"
        ].to_numpy()
        null = layer_scan_null(layer_edges[method], permutations)
        best = int(np.argmax(observed))
        layer_scan[method] = {
            "best_precision": float(observed[best]),
            "best_layer": int(layers[best]),
            "best_depth_percent": float(depths[best]),
            "max_layer_corrected_p": plus_one_p(
                np.max(null, axis=1), float(observed[best])
            ),
            "band_mean_layer_precision": float(np.mean(observed[band])),
            "band_mean_layer_p": plus_one_p(
                np.mean(null[:, band], axis=1), float(np.mean(observed[band]))
            ),
        }

    design, first, second = prompt_covariate_design(
        features, folds, upper_only=True
    )
    band_residual_j, residual_coefs_j = residualize_similarity(
        band_j, design, first, second
    )
    band_residual_raw, residual_coefs_raw = residualize_similarity(
        band_raw, design, first, second
    )
    residual_primary = {}
    residual_edges = {}
    for method, similarity in (
        ("jacobian", band_residual_j),
        ("raw", band_residual_raw),
    ):
        residual_primary[method], residual_edges[method] = graph_summary(
            similarity, families, folds, permutations
        )

    residual_layer_rows = []
    residual_layer_edges: dict[str, list[np.ndarray]] = {
        "jacobian": [],
        "raw": [],
    }
    residual_similarities: dict[str, list[np.ndarray]] = {
        "jacobian": [],
        "raw": [],
    }
    for index, depth in enumerate(depths):
        for method, similarity in (
            ("jacobian", sim_j[index]),
            ("raw", sim_raw[index]),
        ):
            residual, _ = residualize_similarity(
                similarity, design, first, second
            )
            edges = cross_fold_edges(residual, folds)
            residual_layer_edges[method].append(edges)
            residual_similarities[method].append(residual)
            residual_layer_rows.append(
                {
                    "method": method,
                    "layer": int(layers[index]),
                    "depth_percent": float(depth),
                    "directed_precision": edge_precision(edges, families),
                }
            )
    residual_frame = pd.DataFrame(residual_layer_rows)
    residual_frame.to_csv(
        OUT / "heldout_lexical_residual_layer_metrics.csv", index=False
    )
    residual_scan = {}
    for method in ("jacobian", "raw"):
        observed = residual_frame[residual_frame["method"] == method][
            "directed_precision"
        ].to_numpy()
        null = layer_scan_null(residual_layer_edges[method], permutations)
        best = int(np.argmax(observed))
        residual_scan[method] = {
            "best_precision": float(observed[best]),
            "best_layer": int(layers[best]),
            "best_depth_percent": float(depths[best]),
            "max_layer_corrected_p": plus_one_p(
                np.max(null, axis=1), float(observed[best])
            ),
            "band_mean_layer_precision": float(np.mean(observed[band])),
            "band_mean_layer_p": plus_one_p(
                np.mean(null[:, band], axis=1), float(np.mean(observed[band]))
            ),
        }

    cross_design, cross_first, cross_second = prompt_covariate_design(
        features, folds, upper_only=True, cross_fold_only=True
    )
    partial = {}
    for method, similarity in (
        ("jacobian", band_j),
        ("raw", band_raw),
    ):
        observed = partial_family_coefficient(
            similarity,
            cross_design,
            cross_first,
            cross_second,
            families,
        )
        null = partial_coefficient_null(
            similarity,
            cross_design,
            cross_first,
            cross_second,
            permutations,
        )
        partial[method] = {
            "same_family_coefficient": observed,
            "permutation_p": plus_one_p(null, observed),
            "null_95": [float(value) for value in np.quantile(null, (0.025, 0.975))],
        }

    lexical_score = 0.5 * (
        features["word_tfidf"] + features["char_tfidf"]
    )
    hard_frame = hard_negative_rows(
        {
            "jacobian": band_j,
            "raw": band_raw,
            "token_embedding": sim_lexical,
            "word_tfidf": features["word_tfidf"],
            "char_tfidf": features["char_tfidf"],
            "lexical_residual_jacobian": band_residual_j,
        },
        lexical_score,
        families,
        folds,
        slugs,
    )
    hard_frame.to_csv(OUT / "heldout_lexical_hard_negatives.csv", index=False)
    hard_summary = summarize_hard_negatives(hard_frame, rng)

    stability_j = stability_scores(sim_j_seed, folds, band)
    stability_raw = stability_scores(sim_raw[None, ...], folds, band)
    auc_j, auc_j_null = stability_auc_null(
        stability_j, families, folds, permutations
    )
    auc_raw, auc_raw_null = stability_auc_null(
        stability_raw, families, folds, permutations
    )
    stability = {
        "jacobian": {
            "same_family_auc": auc_j,
            "permutation_p": plus_one_p(auc_j_null, auc_j),
            "null_95": [float(value) for value in np.quantile(auc_j_null, (0.025, 0.975))],
        },
        "raw": {
            "same_family_auc": auc_raw,
            "permutation_p": plus_one_p(auc_raw_null, auc_raw),
            "null_95": [float(value) for value in np.quantile(auc_raw_null, (0.025, 0.975))],
        },
    }

    family_contrasts = {}
    for baseline in (
        "raw",
        "token_embedding",
        "word_tfidf",
        "char_tfidf",
        "target_free_direct",
    ):
        jac_query = per_query_precision(primary_edges["jacobian"], families)
        base_query = per_query_precision(primary_edges[baseline], families)
        names, values = family_means(jac_query - base_query, families)
        family_contrasts[f"jacobian_minus_{baseline}"] = {
            "mean_difference": float(np.mean(values)),
            "family_bootstrap_95": bootstrap_mean(values, rng),
            "one_sided_exact_family_sign_flip_p": sign_flip_test(values),
            "positive_families": int(np.sum(values > 0)),
            "family_differences": {
                name: float(value) for name, value in zip(names, values)
            },
        }

    sensitivity = {}
    for count in (1, 2, 3):
        sensitivity[str(count)] = {}
        for method, similarity in (
            ("jacobian", band_j),
            ("raw", band_raw),
            ("word_tfidf", features["word_tfidf"]),
            ("char_tfidf", features["char_tfidf"]),
        ):
            result, _ = graph_summary(
                similarity,
                families,
                folds,
                permutations,
                neighbors_per_fold=count,
            )
            sensitivity[str(count)][method] = result
    lens_seed_results = []
    lens_seed_edges = []
    for seed_index in range(sim_j_seed.shape[0]):
        similarity = np.mean(sim_j_seed[seed_index, band], axis=0)
        result, edges = graph_summary(
            similarity, families, folds, permutations
        )
        lens_seed_results.append({"lens_seed": seed_index, **result})
        lens_seed_edges.append(edges)
    lens_seed_edge_jaccard = {
        f"{first}-{second}": edge_jaccard(
            lens_seed_edges[first], lens_seed_edges[second]
        )
        for first, second in ((0, 1), (0, 2), (1, 2))
    }
    undirected = {
        method: undirected_edge_sensitivity(
            primary_edges[method], families, permutations
        )
        for method in ("jacobian", "raw", "word_tfidf", "char_tfidf")
    }

    concept_matrix, concept_words = concept_documents(jac_records)
    enrichment = concept_enrichment(
        concept_matrix,
        concept_words,
        families,
        permutations,
    )
    enrichment.to_csv(OUT / "target_free_concept_enrichment.csv", index=False)
    concept_result = {
        "candidate_words": len(concept_words),
        "fdr_significant_words": int(enrichment["fdr_05"].sum()),
        "families_with_fdr_words": sorted(
            set(enrichment.loc[enrichment["fdr_05"], "assigned_family"])
        ),
        "top_significant_words": enrichment[enrichment["fdr_05"]]
        .head(50)
        .to_dict(orient="records"),
        "jacobian_vocabulary_size": len(vocabulary_j_words),
        "direct_vocabulary_size": len(vocabulary_d_words),
    }

    payload = {
        "primary_cross_phrasing_graph": primary,
        "primary_family_contrasts": family_contrasts,
        "layer_scan": layer_scan,
        "prompt_only_residualized_graph": residual_primary,
        "residualized_layer_scan": residual_scan,
        "prompt_adjusted_dyadic_coefficient": partial,
        "lexical_hard_negatives": hard_summary,
        "edge_stability": stability,
        "neighbors_per_fold_sensitivity": sensitivity,
        "per_lens_seed": lens_seed_results,
        "per_lens_seed_directed_edge_jaccard": lens_seed_edge_jaccard,
        "undirected_graph_sensitivity": undirected,
        "target_free_concept_network": concept_result,
        "residualization_coefficients": {
            "covariates": [
                "intercept",
                "word_tfidf_cosine",
                "character_tfidf_cosine",
                "token_jaccard",
                "scaled_length_difference",
                "same_phrasing_fold",
            ],
            "jacobian": [float(value) for value in residual_coefs_j],
            "raw": [float(value) for value in residual_coefs_raw],
        },
    }
    figure_data = {
        "depths": depths,
        "layers": layers,
        "families": families,
        "folds": folds,
        "slugs": slugs,
        "layer_frame": layer_frame,
        "residual_frame": residual_frame,
        "band_j": band_j,
        "band_residual_j": band_residual_j,
        "primary_edges": primary_edges["jacobian"],
        "residual_edges": residual_edges["jacobian"],
        "hard_frame": hard_frame,
        "stability_j": stability_j,
        "enrichment": enrichment,
        "concept_matrix": concept_matrix,
        "concept_words": concept_words,
    }
    return payload, figure_data


def analyze_replication(
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = json.loads(REPLICATION_MANIFEST.read_text())
    metadata = {str(row["prompt_id"]): row for row in manifest["prompts"]}
    with np.load(REPLICATION_NPZ, allow_pickle=False) as data:
        prompt_ids = data["prompt_ids"].astype(str)
        layers = data["layers"].astype(int)
        raw = data["raw_states"].astype(np.float64)
        direct = data["direct_decoder_basis"].astype(np.float64)
        jacobian = data["jacobian_decoder_basis"].astype(np.float64)
    rows = [metadata[str(prompt_id)] for prompt_id in prompt_ids]
    families = np.asarray([str(row["family_id"]) for row in rows])
    outcomes = np.asarray([str(row["expected_outcome"]) for row in rows])
    variants = np.asarray([str(row["variant"]) for row in rows])
    triplets = np.asarray([str(row["triplet_id"]) for row in rows])
    presentation_orders = np.asarray(
        [str(row["presentation_order"]) for row in rows]
    )
    numeric_directions = np.asarray(
        [str(row["numeric_direction"]) for row in rows]
    )
    texts = [str(row["user"]) for row in rows]
    depths = layers / 41.0 * 100.0
    features = prompt_feature_matrices(texts)

    sim_j_seed = cosine_layers(jacobian)
    sim_j = np.mean(sim_j_seed, axis=0)
    sim_direct = cosine_layers(direct[None, ...])[0]
    sim_raw = cosine_layers(raw[None, ...])[0]
    band = (depths >= BAND[0]) & (depths <= BAND[1])
    late = (depths >= LATE_BAND[0]) & (depths <= LATE_BAND[1])

    blocks = [
        (str(family), str(variant))
        for family, variant in zip(families, variants)
    ]
    permutations = permute_within_blocks(
        outcomes, blocks, rng=np.random.default_rng(SEED + 1)
    )
    exact_case_permutations = case_preserving_exact_permutations(
        families,
        variants,
        triplets,
        outcomes,
    )

    fixed = {
        "jacobian_band": np.mean(sim_j[band], axis=0),
        "jacobian_late": np.mean(sim_j[late], axis=0),
        "direct_band": np.mean(sim_direct[band], axis=0),
        "direct_late": np.mean(sim_direct[late], axis=0),
        "raw_band": np.mean(sim_raw[band], axis=0),
        "raw_late": np.mean(sim_raw[late], axis=0),
        "word_tfidf": features["word_tfidf"],
        "char_tfidf": features["char_tfidf"],
        "answer_order_only": deterministic_binary_similarity(
            presentation_orders
        ),
        "numeric_direction_oracle": deterministic_binary_similarity(
            numeric_directions
        ),
    }
    fixed_results = {}
    fixed_edges = {}
    exact_nulls = {}
    ranking_results = {}
    ranking_frames = []
    for method, similarity in fixed.items():
        edges = relation_edges(
            similarity, families, variants, triplets
        )
        observed = edge_precision(edges, outcomes)
        null = precision_null(edges, permutations)
        exact_null = precision_null(edges, exact_case_permutations)
        exact_nulls[method] = exact_null
        query = per_query_precision(edges, outcomes)
        names, values = family_means(query, families)
        ranking, ranking_frame = relation_ranking_summary(
            similarity,
            families,
            variants,
            triplets,
            outcomes,
        )
        ranking_results[method] = ranking
        ranking_frame.insert(0, "method", method)
        ranking_frames.append(ranking_frame)
        fixed_results[method] = {
            "directed_same_outcome_precision": observed,
            "permutation_p": plus_one_p(null, observed),
            "null_mean": float(np.mean(null)),
            "null_95": [float(value) for value in np.quantile(null, (0.025, 0.975))],
            "case_preserving_exact_p": plus_one_p(exact_null, observed),
            "case_preserving_exact_null_mean": float(
                np.mean(exact_null)
            ),
            "case_preserving_exact_null_95": [
                float(value)
                for value in np.quantile(exact_null, (0.025, 0.975))
            ],
            "n_case_preserving_exact_assignments": int(
                len(exact_case_permutations)
            ),
            "n_directed_edges": int(len(edges)),
            "family_precision": {
                name: float(value) for name, value in zip(names, values)
            },
            "family_bootstrap_95": bootstrap_mean(values, rng),
            "one_sided_exact_family_sign_flip_p": sign_flip_test(
                values - float(np.mean(null))
            ),
        }
        fixed_edges[method] = edges
    ranking_frame = pd.concat(ranking_frames, ignore_index=True)
    ranking_frame.to_csv(
        OUT / "replication_relation_candidate_ranks.csv",
        index=False,
    )

    diagnostic_frames = []
    for method, similarity in fixed.items():
        diagnostic_frames.append(
            relation_edge_diagnostics(
                method,
                similarity,
                fixed_edges[method],
                rows,
            )
        )
    edge_frame = pd.concat(diagnostic_frames, ignore_index=True)
    edge_frame.to_csv(
        OUT / "replication_relation_graph_selected_edges.csv",
        index=False,
    )
    diagnostics = {}
    for method, method_rows in edge_frame.groupby("method"):
        variant_table = (
            method_rows.groupby(
                ["source_variant", "target_variant"], as_index=False
            )
            .agg(
                n_edges=("same_outcome", "size"),
                same_outcome_precision=("same_outcome", "mean"),
                same_presentation_order_rate=(
                    "same_presentation_order",
                    "mean",
                ),
                same_numeric_direction_rate=(
                    "same_numeric_direction",
                    "mean",
                ),
            )
        )
        variant_p = []
        for row in variant_table.itertuples(index=False):
            selected = method_rows[
                (method_rows["source_variant"] == row.source_variant)
                & (method_rows["target_variant"] == row.target_variant)
            ]
            edges = selected[["source_index", "target_index"]].to_numpy(
                dtype=np.int16
            )
            null = precision_null(edges, permutations)
            variant_p.append(
                plus_one_p(null, float(row.same_outcome_precision))
            )
        variant_table["permutation_p"] = variant_p
        variant_table["bh_q_across_six_pairs"] = bh_adjust(
            np.asarray(variant_p)
        )
        order_table = (
            method_rows.groupby(
                "same_presentation_order", as_index=False
            )
            .agg(
                n_edges=("same_outcome", "size"),
                same_outcome_precision=("same_outcome", "mean"),
            )
        )
        order_p = []
        for row in order_table.itertuples(index=False):
            selected = method_rows[
                method_rows["same_presentation_order"]
                == row.same_presentation_order
            ]
            edges = selected[["source_index", "target_index"]].to_numpy(
                dtype=np.int16
            )
            null = precision_null(edges, permutations)
            order_p.append(
                plus_one_p(null, float(row.same_outcome_precision))
            )
        order_table["permutation_p"] = order_p
        order_table["bh_q_across_two_groups"] = bh_adjust(
            np.asarray(order_p)
        )
        diagnostics[str(method)] = {
            "by_ordered_variant_pair": variant_table.to_dict(
                orient="records"
            ),
            "by_presentation_order_match": order_table.to_dict(
                orient="records"
            ),
            "same_order_rate_among_correct_edges": float(
                method_rows.loc[
                    method_rows["same_outcome"],
                    "same_presentation_order",
                ].mean()
            ),
            "same_order_rate_among_incorrect_edges": float(
                method_rows.loc[
                    ~method_rows["same_outcome"],
                    "same_presentation_order",
                ].mean()
            ),
            "same_numeric_direction_rate": float(
                method_rows["same_numeric_direction"].mean()
            ),
        }

    layer_rows = []
    layer_edges: dict[str, list[np.ndarray]] = {
        "jacobian": [],
        "direct": [],
        "raw": [],
    }
    for index, depth in enumerate(depths):
        for method, similarity in (
            ("jacobian", sim_j[index]),
            ("direct", sim_direct[index]),
            ("raw", sim_raw[index]),
        ):
            edges = relation_edges(
                similarity, families, variants, triplets
            )
            layer_edges[method].append(edges)
            layer_rows.append(
                {
                    "method": method,
                    "layer": int(layers[index]),
                    "depth_percent": float(depth),
                    "directed_same_outcome_precision": edge_precision(
                        edges, outcomes
                    ),
                }
            )
    layer_frame = pd.DataFrame(layer_rows)
    layer_frame.to_csv(OUT / "replication_relation_graph_layers.csv", index=False)
    layer_scan = {}
    for method in ("jacobian", "direct", "raw"):
        observed = layer_frame[layer_frame["method"] == method][
            "directed_same_outcome_precision"
        ].to_numpy()
        null = layer_scan_null(layer_edges[method], permutations)
        best = int(np.argmax(observed))
        layer_scan[method] = {
            "best_precision": float(observed[best]),
            "best_layer": int(layers[best]),
            "best_depth_percent": float(depths[best]),
            "max_layer_corrected_p": plus_one_p(
                np.max(null, axis=1), float(observed[best])
            ),
            "workspace_band_mean_layer_precision": float(np.mean(observed[band])),
            "workspace_band_mean_layer_p": plus_one_p(
                np.mean(null[:, band], axis=1), float(np.mean(observed[band]))
            ),
            "late_band_mean_layer_precision": float(np.mean(observed[late])),
            "late_band_mean_layer_p": plus_one_p(
                np.mean(null[:, late], axis=1), float(np.mean(observed[late]))
            ),
        }

    payload = {
        "fixed_graphs": fixed_results,
        "paired_family_contrasts": relation_family_contrasts(
            fixed_edges,
            outcomes,
            families,
            rng,
        ),
        "answer_order_and_variant_falsification": diagnostics,
        "complete_candidate_ranking": ranking_results,
        "layer_scan": layer_scan,
        "guardrail": (
            "Within-mechanism relation topology is a post-hoc graph analysis "
            "of a cohort whose earlier registered similarity replication failed."
        ),
    }
    figure_data = {
        "depths": depths,
        "layers": layers,
        "families": families,
        "outcomes": outcomes,
        "variants": variants,
        "triplets": triplets,
        "layer_frame": layer_frame,
        "fixed_results": fixed_results,
        "fixed_edges": fixed_edges,
        "edge_frame": edge_frame,
        "ranking_frame": ranking_frame,
        "exact_null_jacobian_band": exact_nulls["jacobian_band"],
        "similarity": fixed["jacobian_late"],
    }
    return payload, figure_data


def panel_label(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.12,
        1.04,
        letter,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_mechanism_graph(
    axis: plt.Axes,
    similarity: np.ndarray,
    edges: np.ndarray,
    families: np.ndarray,
    folds: np.ndarray,
) -> None:
    graph = graph_from_directed(edges, similarity)
    positions = nx.spring_layout(
        graph,
        seed=SEED,
        weight="weight",
        k=0.58,
        iterations=600,
    )
    same = [
        edge for edge in graph.edges()
        if families[edge[0]] == families[edge[1]]
    ]
    other = [
        edge for edge in graph.edges()
        if families[edge[0]] != families[edge[1]]
    ]
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=other,
        width=0.45,
        alpha=0.28,
        edge_color="#B9BDC2",
        ax=axis,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=same,
        width=1.1,
        alpha=0.75,
        edge_color=COLORS["jacobian"],
        ax=axis,
    )
    palette = plt.get_cmap("tab10")
    color_lookup = {
        family: palette(index)
        for index, family in enumerate(sorted(set(families)))
    }
    markers = {1: "o", 2: "s", 3: "^", 4: "D", 5: "P"}
    for fold in sorted(set(folds)):
        nodes = np.flatnonzero(folds == fold).tolist()
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=nodes,
            node_color=[color_lookup[families[node]] for node in nodes],
            node_shape=markers[int(fold)],
            node_size=44,
            edgecolors="white",
            linewidths=0.45,
            ax=axis,
        )
    axis.set_axis_off()


def plot_mechanism_evidence(
    heldout: Mapping[str, object],
    data: Mapping[str, object],
) -> None:
    configure_matplotlib()
    depths = np.asarray(data["depths"])
    layer_frame = data["layer_frame"]
    residual_frame = data["residual_frame"]
    families = np.asarray(data["families"])
    hard_frame = data["hard_frame"]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0))

    axis = axes[0, 0]
    for method, label in (("jacobian", "Jacobian"), ("raw", "raw state")):
        rows = layer_frame[layer_frame["method"] == method]
        axis.plot(
            rows["depth_percent"],
            rows["directed_precision"],
            color=COLORS[method],
            lw=2.0 if method == "jacobian" else 1.6,
            label=label,
        )
    primary = heldout["primary_cross_phrasing_graph"]
    axis.axhline(
        primary["word_tfidf"]["directed_precision"],
        color=COLORS["word_tfidf"],
        ls="--",
        lw=1.25,
        label="word TF–IDF",
    )
    axis.axhline(
        primary["char_tfidf"]["directed_precision"],
        color=COLORS["char_tfidf"],
        ls=":",
        lw=1.35,
        label="character TF–IDF",
    )
    axis.axhline(0.1, color=COLORS["chance"], lw=0.9, ls=(0, (3, 2)))
    axis.axvspan(*BAND, color="#D9DEE3", alpha=0.35, zorder=-2)
    axis.set(
        xlabel="layer depth (%)",
        ylabel="cross-phrasing same-mechanism edges",
        xlim=(0, 100),
        ylim=(0, max(0.55, float(layer_frame["directed_precision"].max()) + 0.05)),
    )
    axis.legend(frameon=False, ncol=2, loc="upper right")
    panel_label(axis, "A")

    axis = axes[0, 1]
    for method, label in (("jacobian", "Jacobian"), ("raw", "raw state")):
        rows = residual_frame[residual_frame["method"] == method]
        axis.plot(
            rows["depth_percent"],
            rows["directed_precision"],
            color=COLORS[method],
            lw=2.0 if method == "jacobian" else 1.6,
            label=label,
        )
    axis.axhline(0.1, color=COLORS["chance"], lw=0.9, ls=(0, (3, 2)))
    axis.axvspan(*BAND, color="#D9DEE3", alpha=0.35, zorder=-2)
    axis.set(
        xlabel="layer depth (%)",
        ylabel="same-mechanism edges after\nprompt-only residualization",
        xlim=(0, 100),
        ylim=(0, max(0.4, float(residual_frame["directed_precision"].max()) + 0.05)),
    )
    axis.legend(frameon=False, loc="upper right")
    panel_label(axis, "B")

    axis = axes[1, 0]
    trap = hard_frame[hard_frame["lexical_trap"]].copy()
    family_order = sorted(set(families))
    y = np.arange(len(family_order))
    for offset, (method, label, marker) in zip(
        (-0.12, 0.12),
        (
            ("jacobian", "Jacobian", "o"),
            ("raw", "raw state", "s"),
        ),
    ):
        all_values = (
            hard_frame[hard_frame["method"] == method]
            .groupby("family")["state_margin"]
            .mean()
            .reindex(family_order)
        )
        trap_values = (
            trap[trap["method"] == method]
            .groupby("family")["state_margin"]
            .mean()
            .reindex(family_order)
        )
        axis.scatter(
            all_values,
            y + offset,
            color=COLORS[method],
            marker=marker,
            s=30,
            label=f"{label}: all",
            zorder=3,
        )
        for index, value in enumerate(trap_values):
            if np.isfinite(value):
                axis.scatter(
                    value,
                    y[index] + offset,
                    facecolors="none",
                    edgecolors=COLORS[method],
                    marker=marker,
                    s=52,
                    linewidths=1.0,
                    zorder=4,
                )
    axis.axvline(0, color=COLORS["chance"], lw=0.8)
    axis.set_yticks(
        y,
        [FAMILY_SHORT.get(family, family[:2].upper()) for family in family_order],
    )
    axis.set(
        xlabel="true-mechanism minus lexical-competitor cosine",
        ylabel="mechanism family",
    )
    axis.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=COLORS["jacobian"],
                markeredgecolor=COLORS["jacobian"],
                markersize=5, label="Jacobian",
            ),
            Line2D(
                [0], [0], marker="s", color="none",
                markerfacecolor=COLORS["raw"],
                markeredgecolor=COLORS["raw"],
                markersize=5, label="raw state",
            ),
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor="none",
                markeredgecolor="#55585C",
                markersize=6, label="lexical-trap subset",
            ),
        ],
        frameon=False,
        loc="upper right",
        ncol=1,
        handletextpad=0.4,
    )
    panel_label(axis, "C")

    axis = axes[1, 1]
    folds = np.asarray(data["folds"])
    order = np.asarray(
        sorted(
            range(len(families)),
            key=lambda index: (str(families[index]), int(folds[index])),
        ),
        dtype=int,
    )
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(len(order))
    adjacency = np.zeros((len(order), len(order)), dtype=int)
    for source, target in np.asarray(data["residual_edges"]):
        value = 2 if families[source] == families[target] else 1
        adjacency[inverse[source], inverse[target]] = value
    axis.imshow(
        adjacency,
        cmap=ListedColormap(["#FFFFFF", "#D8DCE0", COLORS["jacobian"]]),
        vmin=0,
        vmax=2,
        interpolation="nearest",
        aspect="equal",
    )
    for boundary in range(5, len(order), 5):
        axis.axhline(boundary - 0.5, color="#AEB3B8", lw=0.45)
        axis.axvline(boundary - 0.5, color="#AEB3B8", lw=0.45)
    ordered_families = [str(families[index]) for index in order]
    family_names = sorted(set(ordered_families))
    centers = [
        np.mean(
            [
                index
                for index, value in enumerate(ordered_families)
                if value == family
            ]
        )
        for family in family_names
    ]
    axis.set_xticks(
        centers,
        [FAMILY_SHORT.get(family, family[:2].upper()) for family in family_names],
        rotation=90,
    )
    axis.set_yticks(
        centers,
        [FAMILY_SHORT.get(family, family[:2].upper()) for family in family_names],
    )
    axis.set(
        xlabel="target prompt ordered by mechanism",
        ylabel="source prompt ordered by mechanism",
    )
    axis.tick_params(length=0)
    panel_label(axis, "D")
    axis.legend(
        handles=[
            Line2D(
                [0], [0], marker="s", color="none",
                markerfacecolor=COLORS["jacobian"],
                markeredgecolor="none", markersize=6,
                label="same mechanism",
            ),
            Line2D(
                [0], [0], marker="s", color="none",
                markerfacecolor="#D8DCE0",
                markeredgecolor="none", markersize=6,
                label="other mechanism",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        columnspacing=0.9,
        handletextpad=0.25,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.985,
        bottom=0.08,
        hspace=0.30,
        wspace=0.26,
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            FIG / f"mechanism-graph-evidence.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def plot_replication_and_concepts(
    replication: Mapping[str, object],
    rep_data: Mapping[str, object],
    heldout_data: Mapping[str, object],
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.1))
    axis = axes[0, 0]
    frame = rep_data["layer_frame"]
    for method, label in (
        ("jacobian", "Jacobian"),
        ("direct", "direct decoder"),
        ("raw", "raw state"),
    ):
        rows = frame[frame["method"] == method]
        axis.plot(
            rows["depth_percent"],
            rows["directed_same_outcome_precision"],
            color=COLORS[method],
            lw=1.9 if method == "jacobian" else 1.4,
            label=label,
        )
    axis.axhline(0.5, color=COLORS["chance"], lw=0.9, ls=(0, (3, 2)))
    axis.axvspan(*BAND, color="#D9DEE3", alpha=0.25, zorder=-2)
    axis.axvspan(*LATE_BAND, color="#BFC8D1", alpha=0.25, zorder=-2)
    axis.set(
        xlabel="layer depth (%)",
        ylabel="within-mechanism edges with\nsame physical direction",
        xlim=(0, 100),
        ylim=(0.25, 0.91),
    )
    axis.legend(frameon=False, loc="upper left")
    panel_label(axis, "A")

    axis = axes[0, 1]
    fixed = replication["fixed_graphs"]
    methods = [
        "jacobian_band",
        "jacobian_late",
        "direct_late",
        "raw_late",
        "word_tfidf",
        "char_tfidf",
        "answer_order_only",
        "numeric_direction_oracle",
    ]
    labels = [
        "Jacobian\n38–92%",
        "Jacobian\n80–96%",
        "direct\n80–96%",
        "raw\n80–96%",
        "word\nTF–IDF",
        "character\nTF–IDF",
        "answer\norder only",
        "numeric\ndirection",
    ]
    values = [
        fixed[method]["directed_same_outcome_precision"] for method in methods
    ]
    bar_colors = [
        COLORS["jacobian"],
        COLORS["jacobian"],
        COLORS["direct"],
        COLORS["raw"],
        COLORS["word_tfidf"],
        COLORS["char_tfidf"],
        "#A58F78",
        "#4A4A4A",
    ]
    positions = np.arange(len(methods))
    bars = axis.barh(positions, values, color=bar_colors, height=0.66)
    bars[-1].set_hatch("//")
    axis.axvline(0.5, color=COLORS["chance"], lw=0.9, ls=(0, (3, 2)))
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set(
        xlabel="same-direction edge fraction",
        xlim=(0.25, 1.06),
    )
    for bar, value in zip(bars, values):
        axis.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=7.2,
        )
    panel_label(axis, "B")

    axis = axes[1, 0]
    family_names = [
        "obstacle-spacing-orowan",
        "porosity-modulus",
        "pearlite-spacing-strength",
        "dislocation-density-strength",
        "particle-fraction-modulus",
        "crosslink-density-modulus",
    ]
    family_labels = [
        "Orowan spacing",
        "porosity–modulus",
        "pearlite spacing",
        "dislocation density",
        "particle fraction",
        "crosslink density",
    ]
    y = np.arange(len(family_names))
    for offset, (method, label, marker, color) in zip(
        (-0.18, -0.06, 0.06, 0.18),
        (
            ("jacobian_late", "Jacobian", "o", COLORS["jacobian"]),
            ("raw_late", "raw state", "s", COLORS["raw"]),
            ("word_tfidf", "word TF–IDF", "^", COLORS["word_tfidf"]),
            ("char_tfidf", "character TF–IDF", "D", COLORS["char_tfidf"]),
        ),
    ):
        method_values = [
            fixed[method]["family_precision"][family]
            for family in family_names
        ]
        axis.scatter(
            method_values,
            y + offset,
            marker=marker,
            color=color,
            s=28,
            label=label,
            zorder=3,
        )
    axis.axvline(0.5, color=COLORS["chance"], lw=0.9, ls=(0, (3, 2)))
    axis.set_yticks(y, family_labels)
    axis.invert_yaxis()
    axis.set(
        xlabel="same-direction edge fraction",
        ylabel="held-out mechanism",
        xlim=(0.2, 1.05),
    )
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    panel_label(axis, "C")

    axis = axes[1, 1]
    edge_frame = rep_data["edge_frame"]
    selected = edge_frame[edge_frame["method"] == "jacobian_late"]
    variants = ["anchor", "physics_paraphrase", "lexical_counterfactual"]
    short = ["anchor", "paraphrase", "counterfactual"]
    outcome = np.full((3, 3), np.nan)
    order_rate = np.full((3, 3), np.nan)
    for row, source_variant in enumerate(variants):
        for column, target_variant in enumerate(variants):
            if source_variant == target_variant:
                continue
            values_for_pair = selected[
                (selected["source_variant"] == source_variant)
                & (selected["target_variant"] == target_variant)
            ]
            outcome[row, column] = float(
                values_for_pair["same_outcome"].mean()
            )
            order_rate[row, column] = float(
                values_for_pair["same_presentation_order"].mean()
            )
    image = axis.imshow(
        outcome,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(3), short, rotation=20, ha="right")
    axis.set_yticks(np.arange(3), short)
    axis.set(
        xlabel="target surface variant",
        ylabel="source surface variant",
    )
    for row in range(3):
        for column in range(3):
            if not np.isfinite(outcome[row, column]):
                continue
            color = "white" if outcome[row, column] >= 0.65 else "#202124"
            axis.text(
                column,
                row,
                f"{outcome[row, column]:.2f}\n(order {order_rate[row, column]:.2f})",
                ha="center",
                va="center",
                color=color,
                fontsize=7.2,
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("same-direction edge fraction", fontsize=7.5)
    colorbar.ax.tick_params(labelsize=7)
    panel_label(axis, "D")

    fig.subplots_adjust(
        left=0.11,
        right=0.965,
        top=0.985,
        bottom=0.105,
        hspace=0.39,
        wspace=0.29,
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            FIG / f"relation-graph-falsification.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def plot_relation_ranking_robustness(
    replication: Mapping[str, object],
    rep_data: Mapping[str, object],
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.75))

    axis = axes[0]
    null = np.asarray(rep_data["exact_null_jacobian_band"], dtype=float)
    observed = float(
        replication["fixed_graphs"]["jacobian_band"][
            "directed_same_outcome_precision"
        ]
    )
    bins = np.linspace(
        float(np.min(null)) - 0.01,
        float(np.max(null)) + 0.01,
        22,
    )
    axis.hist(
        null,
        bins=bins,
        color="#CBD3D9",
        edgecolor="white",
        linewidth=0.45,
    )
    axis.axvline(
        float(np.mean(null)),
        color=COLORS["chance"],
        lw=1.2,
        ls=(0, (3, 2)),
        label=f"exact-null mean {np.mean(null):.2f}",
    )
    axis.axvline(
        observed,
        color=COLORS["jacobian"],
        lw=2.4,
        label=f"observed {observed:.2f}",
    )
    axis.set(
        xlabel="same-direction edge fraction",
        ylabel="balanced case assignments",
        xlim=(max(0.0, float(np.min(null)) - 0.03), observed + 0.04),
    )
    axis.legend(frameon=False, loc="upper right")
    panel_label(axis, "A")

    axis = axes[1]
    rankings = replication["complete_candidate_ranking"]
    methods = [
        "jacobian_band",
        "direct_band",
        "raw_band",
        "word_tfidf",
        "char_tfidf",
        "answer_order_only",
        "numeric_direction_oracle",
    ]
    labels = [
        "Jacobian",
        "direct decoder",
        "raw state",
        "word TF–IDF",
        "character TF–IDF",
        "answer order only",
        "numeric direction",
    ]
    colors = [
        COLORS["jacobian"],
        COLORS["direct"],
        COLORS["raw"],
        COLORS["word_tfidf"],
        COLORS["char_tfidf"],
        "#A58F78",
        "#4A4A4A",
    ]
    values = np.asarray(
        [rankings[method]["pairwise_auc"] for method in methods],
        dtype=float,
    )
    ranges = [
        rankings[method]["leave_one_family_out_range"]["pairwise_auc"]
        for method in methods
    ]
    lower = values - np.asarray([row[0] for row in ranges])
    upper = np.asarray([row[1] for row in ranges]) - values
    positions = np.arange(len(methods))
    bars = axis.barh(
        positions,
        values,
        color=colors,
        height=0.64,
        xerr=np.vstack((lower, upper)),
        error_kw={
            "ecolor": "#202124",
            "elinewidth": 0.8,
            "capsize": 2.0,
            "capthick": 0.8,
        },
    )
    bars[-1].set_hatch("//")
    axis.axvline(0.5, color=COLORS["chance"], lw=1.0, ls=(0, (3, 2)))
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set(
        xlabel="pairwise ranking AUC",
        xlim=(0.2, 1.06),
    )
    for bar, value in zip(bars, values):
        axis.text(
            max(0.22, value - 0.07),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=7.2,
        )
    panel_label(axis, "B")

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.96,
        bottom=0.17,
        wspace=0.34,
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(
            FIG / f"relation-ranking-robustness.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def write_results(
    heldout: Mapping[str, object],
    replication: Mapping[str, object],
) -> None:
    primary = heldout["primary_cross_phrasing_graph"]
    residual = heldout["prompt_only_residualized_graph"]
    hard = heldout["lexical_hard_negatives"]
    stability = heldout["edge_stability"]
    concept = heldout["target_free_concept_network"]
    fixed = replication["fixed_graphs"]
    rankings = replication["complete_candidate_ranking"]
    diagnostics = replication["answer_order_and_variant_falsification"]
    contrasts = replication["paired_family_contrasts"]
    late_diagnostics = diagnostics["jacobian_late"]
    variant_rows = late_diagnostics["by_ordered_variant_pair"]
    counterfactual_rows = [
        row
        for row in variant_rows
        if "lexical_counterfactual"
        in (row["source_variant"], row["target_variant"])
    ]
    counterfactual_precision = float(
        np.average(
            [row["same_outcome_precision"] for row in counterfactual_rows],
            weights=[row["n_edges"] for row in counterfactual_rows],
        )
    )
    counterfactual_max_q = max(
        float(row["bh_q_across_six_pairs"])
        for row in counterfactual_rows
    )
    seed_jaccards = heldout["per_lens_seed_directed_edge_jaccard"]
    jaccard_values = list(seed_jaccards.values())
    undirected = heldout["undirected_graph_sensitivity"]["jacobian"]
    versus_lexical = contrasts["jacobian_band_minus_word_tfidf"]
    lines = [
        "# Rigorous graph-topology results",
        "",
        "## Cross-phrasing mechanism graph",
        "",
        (
            "The frozen-band Jacobian graph placed "
            f"**{primary['jacobian']['directed_precision']:.1%}** of its "
            "cross-phrasing edges within the correct mechanism family "
            f"(blocked permutation p={primary['jacobian']['permutation_p']:.4g}; "
            "balanced chance 10%)."
        ),
        (
            "The matched raw-state value was "
            f"{primary['raw']['directed_precision']:.1%}; word and character "
            f"TF--IDF reached {primary['word_tfidf']['directed_precision']:.1%} "
            f"and {primary['char_tfidf']['directed_precision']:.1%}."
        ),
        "",
        "## Prompt-only residualization and lexical traps",
        "",
        (
            "After removing the part of pairwise state similarity predictable "
            "from word TF--IDF, character TF--IDF, token overlap, prompt length, "
            "and phrasing fold, the Jacobian graph retained "
            f"**{residual['jacobian']['directed_precision']:.1%}** "
            f"same-mechanism edges (p={residual['jacobian']['permutation_p']:.4g})."
        ),
        (
            "Across all frozen lexical hard negatives, the Jacobian family-mean "
            "true-minus-competitor cosine margin was "
            f"{hard['all']['jacobian']['mean_family_margin']:+.4f} "
            f"(95% family bootstrap "
            f"{hard['all']['jacobian']['family_bootstrap_95'][0]:+.4f} to "
            f"{hard['all']['jacobian']['family_bootstrap_95'][1]:+.4f})."
        ),
        (
            f"There were {hard['lexical_traps']['jacobian']['n_comparisons']} "
            "comparisons in which prompt-only TF--IDF preferred the wrong "
            "mechanism; their Jacobian family-mean margin was "
            f"{hard['lexical_traps']['jacobian']['mean_family_margin']:+.4f}."
        ),
        "",
        "## Repeated graph assembly",
        "",
        (
            "An edge's selection frequency across all registered band layers "
            "and all three lens fits identified same-mechanism pairs with "
            f"ROC--AUC **{stability['jacobian']['same_family_auc']:.3f}** "
            f"(p={stability['jacobian']['permutation_p']:.4g}); raw-state "
            f"stability gave {stability['raw']['same_family_auc']:.3f}."
        ),
        (
            "The three independently fitted lenses selected directed edge sets "
            f"with pairwise Jaccard overlap {min(jaccard_values):.3f}--"
            f"{max(jaccard_values):.3f}. Treating edges as undirected gave "
            f"{undirected['union']['homophily']:.1%} union homophily and "
            f"{undirected['mutual']['homophily']:.1%} mutual-edge homophily."
        ),
        "",
        "## Disjoint signed-relation graph",
        "",
        (
            "Within each of six new mechanisms and after excluding the same "
            "material case, the frozen-band Jacobian graph connected prompts "
            "with the same physically correct outcome on "
            f"**{fixed['jacobian_band']['directed_same_outcome_precision']:.1%}** "
            f"of edges (p={fixed['jacobian_band']['permutation_p']:.4g}; "
            "chance 50%)."
        ),
        (
            "Under the stricter exact null that preserves all three surface "
            "variants of each material case—including the registered "
            "counterfactual sign reversal—and enumerates all 46,656 balanced "
            "base-case assignments, the frozen-band result had null mean "
            f"{fixed['jacobian_band']['case_preserving_exact_null_mean']:.1%} "
            f"and p={fixed['jacobian_band']['case_preserving_exact_p']:.4g}."
        ),
        (
            "The corresponding direct, raw, word-TF--IDF, and character-TF--IDF "
            "values were "
            f"{fixed['direct_band']['directed_same_outcome_precision']:.1%}, "
            f"{fixed['raw_band']['directed_same_outcome_precision']:.1%}, "
            f"{fixed['word_tfidf']['directed_same_outcome_precision']:.1%}, and "
            f"{fixed['char_tfidf']['directed_same_outcome_precision']:.1%}."
        ),
        (
            "The deliberately non-semantic answer-order-only baseline reached "
            f"{fixed['answer_order_only']['directed_same_outcome_precision']:.1%}; "
            "a numeric-direction oracle reached "
            f"{fixed['numeric_direction_oracle']['directed_same_outcome_precision']:.1%}. "
            "The oracle is an interpretive ceiling, not model evidence."
        ),
        (
            "For the four ordered edge types that cross into or out of the "
            "lexical-counterfactual variant, the late Jacobian graph retained "
            f"**{counterfactual_precision:.1%}** same-direction precision. "
            "Every crossing was significant after BH correction across the "
            f"six ordered pairs (largest q={counterfactual_max_q:.4g}). "
            "Thus the aggregate is not explained by preserving the displayed "
            "answer order across surface variants."
        ),
        (
            "At the family level, the frozen-band Jacobian graph exceeded word "
            f"TF--IDF by {versus_lexical['mean_difference']:+.1%} "
            f"(exact one-sided six-family sign-flip "
            f"p={versus_lexical['one_sided_exact_family_sign_flip_p']:.4g}), "
            "but did not outperform matched raw or direct states. This is "
            "evidence for model-state geometry, not a Jacobian-specific gain."
        ),
        (
            "Using all three eligible candidates rather than only the winning "
            "edge, the frozen-band Jacobian ranking achieved pairwise AUC "
            f"**{rankings['jacobian_band']['pairwise_auc']:.3f}** "
            f"(chance 0.5) and mean reciprocal rank "
            f"{rankings['jacobian_band']['mean_reciprocal_rank']:.3f}. "
            "Its leave-one-mechanism-out AUC range was "
            f"{rankings['jacobian_band']['leave_one_family_out_range']['pairwise_auc'][0]:.3f}--"
            f"{rankings['jacobian_band']['leave_one_family_out_range']['pairwise_auc'][1]:.3f}."
        ),
        "",
        "## Target-free concept network",
        "",
        (
            f"{concept['fdr_significant_words']} of "
            f"{concept['candidate_words']} archived target-free consensus words "
            "survived blocked permutation testing with BH FDR 0.05, spanning "
            f"{len(concept['families_with_fdr_words'])} of ten mechanism families."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "The graph evidence is descriptive representational evidence. It "
            "does not establish causal use, consciousness, a literal hidden "
            "narrative, or unrestricted materials understanding. The disjoint "
            "cohort's earlier registered triplet-similarity endpoint failed; "
            "the graph result cannot erase that negative result."
        ),
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/analyze_graph_topology_rigorous.py",
        "```",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    amendment = json.loads(AMENDMENT_PATH.read_text())
    amendment_v2 = json.loads(AMENDMENT_V2_PATH.read_text())
    amendment_v2_correction = json.loads(
        AMENDMENT_V2_CORRECTION_PATH.read_text()
    )
    validate_inputs(protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    heldout, heldout_data = analyze_heldout(protocol, rng)
    replication, replication_data = analyze_replication(rng)
    payload = {
        "study_id": protocol["study_id"],
        "status": protocol["status"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "falsification_amendment": {
            "study_id": amendment["study_id"],
            "status": amendment["status"],
            "sha256": sha256(AMENDMENT_PATH),
        },
        "ranking_robustness_amendment": {
            "study_id": amendment_v2["study_id"],
            "status": amendment_v2["status"],
            "sha256": sha256(AMENDMENT_V2_PATH),
        },
        "ranking_robustness_correction": {
            "study_id": amendment_v2_correction["study_id"],
            "status": amendment_v2_correction["status"],
            "sha256": sha256(AMENDMENT_V2_CORRECTION_PATH),
        },
        "input_fingerprints": protocol["inputs"],
        "heldout": heldout,
        "disjoint_relation_replication": replication,
        "guardrail": protocol["guardrail"],
    }
    (OUT / "statistics.json").write_text(
        json.dumps(json_safe(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    plot_mechanism_evidence(heldout, heldout_data)
    plot_replication_and_concepts(
        replication,
        replication_data,
        heldout_data,
    )
    plot_relation_ranking_robustness(replication, replication_data)
    write_results(heldout, replication)
    execution = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "falsification_amendment": str(AMENDMENT_PATH.relative_to(ROOT)),
        "falsification_amendment_sha256": sha256(AMENDMENT_PATH),
        "ranking_robustness_amendment": str(
            AMENDMENT_V2_PATH.relative_to(ROOT)
        ),
        "ranking_robustness_amendment_sha256": sha256(AMENDMENT_V2_PATH),
        "ranking_robustness_correction": str(
            AMENDMENT_V2_CORRECTION_PATH.relative_to(ROOT)
        ),
        "ranking_robustness_correction_sha256": sha256(
            AMENDMENT_V2_CORRECTION_PATH
        ),
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted(OUT.rglob("*"))
            if path.is_file()
            and path.name
            not in {
                "execution_record.json",
                "validation.json",
                "VALIDATION.md",
            }
        },
    }
    (OUT / "execution_record.json").write_text(
        json.dumps(json_safe(execution), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(payload), indent=2))


if __name__ == "__main__":
    main()
