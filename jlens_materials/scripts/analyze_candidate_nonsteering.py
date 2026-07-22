#!/usr/bin/env python3
"""Run frozen candidate non-steering analyses for the materials J-lens paper."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "candidate-nonsteering-2026-07-16"
FIG = OUT / "figures"
PROTOCOL_PATH = OUT / "protocol.json"
STATS_PATH = ROOT / "experiments" / "materials-heldout-v1_statistics.json"
NPZ_PATH = ROOT / "experiments" / "materials-heldout-v1_latent_vectors.npz"
META_PATH = ROOT / "experiments" / "materials-heldout-v1_latent_vectors.meta.json"
RUN_PATHS = [ROOT / "runs" / f"gemma4-e4b-it-heldout-v1-seed{i}.json" for i in range(3)]
SEED = 20260716
N_BOOT = 30_000
N_CLASS_PERM = 10_000
N_RSA_PERM = 30_000
N_ROLE_PERM = 10_000
BAND = (38.0, 92.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def token_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z'-]{2,}", text.lower()))


def candidate_removed(token: str, prompt_words: set[str], mode: str) -> bool:
    if mode == "all":
        return False
    if token in prompt_words:
        return True
    if mode == "prompt_exact_removed":
        return False
    if mode != "prompt_morphology_removed":
        raise ValueError(mode)
    return any(
        min(len(token), len(word)) >= 5
        and (token.startswith(word) or word.startswith(token))
        for word in prompt_words
    )


def validate_inputs() -> tuple[dict, list[dict], dict]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    expected = protocol["inputs"]
    checks = {
        STATS_PATH: expected["heldout_statistics_sha256"],
        NPZ_PATH: expected["latent_vectors_sha256"],
        META_PATH: expected["latent_metadata_sha256"],
        **{RUN_PATHS[i]: expected[f"heldout_seed{i}_sha256"] for i in range(3)},
    }
    for path, fingerprint in checks.items():
        if sha256(path) != fingerprint:
            raise RuntimeError(f"fingerprint mismatch: {path}")
    heldout = json.loads(STATS_PATH.read_text())
    runs = [json.loads(path.read_text()) for path in RUN_PATHS]
    return heldout, runs, protocol


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def build_fold_features(
    records: list[dict],
    families: np.ndarray,
    folds: np.ndarray,
    *,
    mode: str,
    input_baseline: bool = False,
) -> list[dict]:
    output = []
    for heldout_fold in sorted(set(folds)):
        train = np.where(folds != heldout_fold)[0]
        test = np.where(folds == heldout_fold)[0]
        weighted: list[dict[str, float]] = []
        for record in records:
            words = token_words(record["prompt"])
            if input_baseline:
                weighted.append({word: 1.0 for word in words})
            else:
                weighted.append({
                    row["token"]: float(row["consensus_score"])
                    for row in record["filtered_consensus_candidates"]
                    if not candidate_removed(row["token"], words, mode)
                })
        document_frequency = Counter(
            token for index in train for token in weighted[index]
        )
        vocabulary = sorted(document_frequency)
        lookup = {token: col for col, token in enumerate(vocabulary)}
        idf = {
            token: math.log(len(train) / count)
            for token, count in document_frequency.items()
        }

        def matrix(indices: np.ndarray) -> np.ndarray:
            values = np.zeros((len(indices), len(vocabulary)), dtype=float)
            for row_index, source_index in enumerate(indices):
                for token, score in weighted[int(source_index)].items():
                    if token in lookup:
                        values[row_index, lookup[token]] = score * idf[token]
            return normalize_rows(values)

        x_train = matrix(train)
        x_test = matrix(test)
        output.append({
            "fold": int(heldout_fold),
            "train": train,
            "test": test,
            "x_train": x_train,
            "x_test": x_test,
            "k_test_train": np.einsum("id,jd->ij", x_test, x_train, optimize=True),
            "k_train_train": np.einsum("id,jd->ij", x_train, x_train, optimize=True),
            "empty_test_vectors": int(np.sum(np.linalg.norm(x_test, axis=1) == 0)),
        })
    return output


def predict_from_labels(
    fold_data: dict, train_labels: np.ndarray, label_names: list[str]
) -> np.ndarray:
    similarities = np.empty((len(fold_data["test"]), len(label_names)), dtype=float)
    for label_index, label in enumerate(label_names):
        mask = train_labels == label
        numerator = fold_data["k_test_train"][:, mask].sum(axis=1)
        norm_sq = float(fold_data["k_train_train"][np.ix_(mask, mask)].sum())
        similarities[:, label_index] = numerator / math.sqrt(max(norm_sq, 1e-12))
    return np.asarray(label_names)[np.argmax(similarities, axis=1)]


def evaluate_folds(
    fold_data: list[dict], labels: np.ndarray, label_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.empty(len(labels), dtype=object)
    for fold in fold_data:
        predictions[fold["test"]] = predict_from_labels(
            fold, labels[fold["train"]], label_names
        )
    return predictions, predictions == labels


def permutation_accuracies(
    fold_data: list[dict],
    permuted_labels: np.ndarray,
    true_folds: np.ndarray,
    label_names: list[str],
    *,
    batch_size: int = 250,
) -> np.ndarray:
    n_permutations = len(permuted_labels)
    correct = np.zeros(n_permutations, dtype=float)
    label_to_int = {label: index for index, label in enumerate(label_names)}
    encoded = np.vectorize(label_to_int.get)(permuted_labels)
    for fold in fold_data:
        train = fold["train"]
        test = fold["test"]
        k_qt = fold["k_test_train"]
        k_tt = fold["k_train_train"]
        for start in range(0, n_permutations, batch_size):
            stop = min(start + batch_size, n_permutations)
            train_codes = encoded[start:stop, train]
            indicators = np.eye(len(label_names), dtype=float)[train_codes]
            numerators = np.einsum("ij,bjc->bic", k_qt, indicators, optimize=True)
            norm_sq = np.einsum(
                "bjc,jk,bkc->bc", indicators, k_tt, indicators, optimize=True
            )
            similarities = numerators / np.sqrt(np.maximum(norm_sq[:, None, :], 1e-12))
            prediction_codes = np.argmax(similarities, axis=2)
            correct[start:stop] += np.sum(
                prediction_codes == encoded[start:stop, test], axis=1
            )
    return correct / len(true_folds)


def family_bootstrap(correct: np.ndarray, families: np.ndarray, rng: np.random.Generator) -> list[float]:
    names = sorted(set(families))
    means = np.asarray([np.mean(correct[families == name]) for name in names])
    indices = rng.integers(0, len(names), size=(N_BOOT, len(names)))
    draws = means[indices].mean(axis=1)
    return [float(means.mean()), *[float(v) for v in np.quantile(draws, [0.025, 0.975])]]


def analyze_semantic_classification(
    heldout: dict, runs: list[dict]
) -> tuple[pd.DataFrame, dict]:
    reference = {record["slug"]: record for record in runs[0]["prompts"]}
    jac_records = heldout["open_vocabulary"]["methods"]["jacobian"]["per_prompt"]
    direct_records = heldout["open_vocabulary"]["methods"]["logit"]["per_prompt"]
    slugs = np.asarray([row["slug"] for row in jac_records])
    families = np.asarray([row["family"] for row in jac_records])
    folds = np.asarray([
        int(reference[slug]["phrasing_id"].rsplit("-", 1)[-1]) for slug in slugs
    ])
    label_names = sorted(set(families))
    configs: dict[str, list[dict]] = {}
    for method, records in (("Jacobian", jac_records), ("direct", direct_records)):
        for mode in ("all", "prompt_exact_removed", "prompt_morphology_removed"):
            configs[f"{method}::{mode}"] = build_fold_features(
                records, families, folds, mode=mode
            )
    configs["input_TFIDF::all"] = build_fold_features(
        jac_records, families, folds, mode="all", input_baseline=True
    )

    rng = np.random.default_rng(SEED)
    permuted_labels = np.empty((N_CLASS_PERM, len(families)), dtype=object)
    for iteration in range(N_CLASS_PERM):
        candidate = families.copy()
        for fold in sorted(set(folds)):
            indices = np.where(folds == fold)[0]
            candidate[indices] = rng.permutation(candidate[indices])
        permuted_labels[iteration] = candidate

    result_rows = []
    item_rows = []
    permutation_by_config = {}
    observed_by_config = {}
    correctness_by_config = {}
    for config_name, fold_data in configs.items():
        predictions, correct = evaluate_folds(fold_data, families, label_names)
        accuracy = float(np.mean(correct))
        permutation = permutation_accuracies(
            fold_data, permuted_labels, folds, label_names
        )
        permutation_by_config[config_name] = permutation
        observed_by_config[config_name] = accuracy
        correctness_by_config[config_name] = correct
        method, mode = config_name.split("::")
        bootstrap_ci = family_bootstrap(correct, families, rng)
        result_rows.append({
            "config": config_name,
            "method": method,
            "filter": mode,
            "accuracy": accuracy,
            "macro_f1": macro_f1(families, predictions, label_names),
            "permutation_p": float((1 + np.sum(permutation >= accuracy - 1e-15)) / (1 + len(permutation))),
            "family_bootstrap_mean": bootstrap_ci[0],
            "family_bootstrap_low": bootstrap_ci[1],
            "family_bootstrap_high": bootstrap_ci[2],
            "empty_test_vectors": int(sum(fold["empty_test_vectors"] for fold in fold_data)),
        })
        for index, slug in enumerate(slugs):
            item_rows.append({
                "config": config_name,
                "slug": slug,
                "family": families[index],
                "fold": int(folds[index]),
                "prediction": predictions[index],
                "correct": bool(correct[index]),
            })
    maximum_null = np.max(np.column_stack(list(permutation_by_config.values())), axis=1)
    for row in result_rows:
        observed = row["accuracy"]
        row["max_config_corrected_p"] = float(
            (1 + np.sum(maximum_null >= observed - 1e-15)) / (1 + len(maximum_null))
        )

    # Paired Jacobian-versus-direct comparisons under identical filters.
    paired = {}
    for mode in ("all", "prompt_exact_removed", "prompt_morphology_removed"):
        j = correctness_by_config[f"Jacobian::{mode}"].astype(int)
        d = correctness_by_config[f"direct::{mode}"].astype(int)
        wins = int(np.sum((j == 1) & (d == 0)))
        losses = int(np.sum((j == 0) & (d == 1)))
        paired[mode] = {
            "jacobian_only_correct": wins,
            "direct_only_correct": losses,
            "both_correct": int(np.sum((j == 1) & (d == 1))),
            "both_wrong": int(np.sum((j == 0) & (d == 0))),
            "exact_two_sided_mcnemar_p": float(
                stats.binomtest(wins, wins + losses, 0.5).pvalue
                if wins + losses else 1.0
            ),
        }

    summary_frame = pd.DataFrame(result_rows).sort_values(
        ["accuracy", "config"], ascending=[False, True]
    )
    item_frame = pd.DataFrame(item_rows)
    summary_frame.to_csv(OUT / "semantic_classification_summary.csv", index=False)
    item_frame.to_csv(OUT / "semantic_classification_items.csv", index=False)
    per_family = (
        item_frame.groupby(["config", "family"], as_index=False).correct.mean()
    )
    per_family.to_csv(OUT / "semantic_classification_by_family.csv", index=False)
    payload = {
        "status": "retrospective candidate analysis under frozen protocol",
        "n_prompts": 50,
        "n_families": 10,
        "folds": 5,
        "results": summary_frame.to_dict(orient="records"),
        "paired_jacobian_vs_direct": paired,
        "permutation_resamples": N_CLASS_PERM,
        "maximum_over_configs_null_95": float(np.quantile(maximum_null, 0.95)),
    }
    dump_json(OUT / "semantic_classification_statistics.json", payload)
    return item_frame, payload


def centroid_distances(vectors: np.ndarray, families: np.ndarray, family_order: list[str]) -> np.ndarray:
    centroids = normalize_rows(np.asarray([
        normalize_rows(vectors[families == family]).mean(axis=0)
        for family in family_order
    ]))
    similarity = np.einsum("id,jd->ij", centroids, centroids, optimize=True)
    return (1.0 - similarity)[np.triu_indices(len(family_order), k=1)]


def expert_distances(protocol: dict, family_order: list[str], kind: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(family_order), len(family_order)), dtype=float)
    ontology = protocol["family_ontology"]
    for i, first in enumerate(family_order):
        for j, second in enumerate(family_order):
            if kind == "response_class":
                matrix[i, j] = float(
                    ontology[first]["response_class"] != ontology[second]["response_class"]
                )
            elif kind == "multi_attribute":
                a = set(ontology[first]["attributes"])
                b = set(ontology[second]["attributes"])
                matrix[i, j] = 1.0 - len(a & b) / len(a | b)
            else:
                raise ValueError(kind)
    return matrix, matrix[np.triu_indices(len(family_order), k=1)]


def rank_unit(values: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(values, method="average", axis=-1)
    ranks = ranks - ranks.mean(axis=-1, keepdims=True)
    return ranks / np.maximum(np.linalg.norm(ranks, axis=-1, keepdims=True), 1e-12)


def analyze_ontology_rsa(protocol: dict) -> tuple[pd.DataFrame, dict]:
    with np.load(NPZ_PATH, allow_pickle=False) as data:
        raw = normalize_rows(data["raw_states"].astype(float))
        transported = normalize_rows(data["transported_states"].astype(float))
        transported_mean = normalize_rows(transported.mean(axis=0))
        target = normalize_rows(data["target_states"].astype(float))
        lexical = normalize_rows(data["lexical_states"].astype(float))
        families = data["families"].astype(str)
        depths = data["depths"].astype(float)
        source_layers = data["source_layers"].astype(int)
    family_order = sorted(set(families))
    j_distances = np.asarray([
        centroid_distances(transported_mean[:, layer], families, family_order)
        for layer in range(len(depths))
    ])
    raw_distances = np.asarray([
        centroid_distances(raw[:, layer], families, family_order)
        for layer in range(len(depths))
    ])
    target_distances = centroid_distances(target, families, family_order)
    lexical_distances = centroid_distances(lexical, families, family_order)

    rng = np.random.default_rng(SEED)
    permutations = np.asarray([
        rng.permutation(len(family_order)) for _ in range(N_RSA_PERM)
    ])
    rows = []
    payload = {"ontology_permutation_resamples": N_RSA_PERM, "ontologies": {}}
    for kind in ("response_class", "multi_attribute"):
        matrix, expert = expert_distances(protocol, family_order, kind)
        expert_rank = rank_unit(expert[None, :])[0]
        j_rank = rank_unit(j_distances)
        raw_rank = rank_unit(raw_distances)
        target_rank = rank_unit(target_distances[None, :])[0]
        lexical_rank = rank_unit(lexical_distances[None, :])[0]
        j_corr = np.einsum("ld,d->l", j_rank, expert_rank, optimize=True)
        raw_corr = np.einsum("ld,d->l", raw_rank, expert_rank, optimize=True)
        target_corr = float(np.einsum("d,d->", target_rank, expert_rank, optimize=True))
        lexical_corr = float(np.einsum("d,d->", lexical_rank, expert_rank, optimize=True))

        perm_expert = np.empty((N_RSA_PERM, len(expert)), dtype=float)
        upper = np.triu_indices(len(family_order), k=1)
        for start in range(0, N_RSA_PERM, 1000):
            stop = min(start + 1000, N_RSA_PERM)
            for offset, permutation in enumerate(permutations[start:stop]):
                perm_expert[start + offset] = matrix[np.ix_(permutation, permutation)][upper]
        perm_rank = rank_unit(perm_expert)
        null_j = np.einsum("pd,ld->pl", perm_rank, j_rank, optimize=True)
        null_raw = np.einsum("pd,ld->pl", perm_rank, raw_rank, optimize=True)
        null_target = np.einsum("pd,d->p", perm_rank, target_rank, optimize=True)
        null_lexical = np.einsum("pd,d->p", perm_rank, lexical_rank, optimize=True)
        j_best = int(np.argmax(np.abs(j_corr)))
        raw_best = int(np.argmax(np.abs(raw_corr)))

        def corrected_p(null_values: np.ndarray, observed: float) -> float:
            maximum = np.max(np.abs(null_values), axis=1) if null_values.ndim == 2 else np.abs(null_values)
            return float((1 + np.sum(maximum >= abs(observed) - 1e-15)) / (1 + len(maximum)))

        leave_one = {"Jacobian": [], "raw": [], "target": [], "lexical": []}
        for omitted_index, omitted in enumerate(family_order):
            keep_family = [i for i in range(len(family_order)) if i != omitted_index]
            pair_mask = np.asarray([
                first in keep_family and second in keep_family
                for first, second in zip(*upper)
            ])
            for method, vector in (
                ("Jacobian", j_distances[j_best]),
                ("raw", raw_distances[raw_best]),
                ("target", target_distances),
                ("lexical", lexical_distances),
            ):
                value = stats.spearmanr(vector[pair_mask], expert[pair_mask]).statistic
                leave_one[method].append({"omitted_family": omitted, "rho": float(value)})

        kind_payload = {
            "family_order": family_order,
            "Jacobian": {
                "rho_by_layer": [float(v) for v in j_corr],
                "best_abs_layer": int(source_layers[j_best]),
                "best_abs_depth": float(depths[j_best]),
                "best_abs_rho": float(j_corr[j_best]),
                "max_layer_corrected_p": corrected_p(null_j, float(j_corr[j_best])),
            },
            "raw": {
                "rho_by_layer": [float(v) for v in raw_corr],
                "best_abs_layer": int(source_layers[raw_best]),
                "best_abs_depth": float(depths[raw_best]),
                "best_abs_rho": float(raw_corr[raw_best]),
                "max_layer_corrected_p": corrected_p(null_raw, float(raw_corr[raw_best])),
            },
            "target": {"rho": target_corr, "permutation_p": corrected_p(null_target, target_corr)},
            "lexical": {"rho": lexical_corr, "permutation_p": corrected_p(null_lexical, lexical_corr)},
            "leave_one_family_out": leave_one,
        }
        payload["ontologies"][kind] = kind_payload
        for method, correlations in (("Jacobian", j_corr), ("raw", raw_corr)):
            for index, correlation in enumerate(correlations):
                rows.append({
                    "ontology": kind, "method": method,
                    "layer": int(source_layers[index]), "depth": float(depths[index]),
                    "spearman_rho": float(correlation),
                })
        rows.extend([
            {"ontology": kind, "method": "target", "layer": np.nan, "depth": np.nan, "spearman_rho": target_corr},
            {"ontology": kind, "method": "lexical", "layer": np.nan, "depth": np.nan, "spearman_rho": lexical_corr},
        ])
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "ontology_rsa_layer_curves.csv", index=False)
    dump_json(OUT / "ontology_rsa_statistics.json", payload)
    return frame, payload


def first_sustained_depth(depths: np.ndarray, ranks: np.ndarray, threshold: int = 1000) -> float:
    hits = ranks <= threshold
    for index in range(len(hits) - 1):
        if hits[index] and hits[index + 1]:
            return float(depths[index])
    return float("nan")


def analyze_concept_roles(runs: list[dict], protocol: dict) -> tuple[pd.DataFrame, dict]:
    role_lookup = {
        concept: role
        for role, concepts in protocol["concept_roles"].items()
        for concept in concepts
    }
    indexes = [{record["slug"]: record for record in run["prompts"]} for run in runs]
    technical = []
    for seed, index in enumerate(indexes):
        for slug, record in index.items():
            for source_key, method in (("jacobian_lens", "Jacobian"), ("logit_lens", "direct")):
                for trajectory in record["concept_trajectories"][source_key]:
                    label = trajectory["label"]
                    if label not in role_lookup:
                        raise RuntimeError(f"unassigned concept role: {label}")
                    depths = np.asarray(trajectory["depths"], dtype=float)
                    ranks = np.asarray(trajectory["ranks"], dtype=float)
                    mask = (depths >= BAND[0]) & (depths <= BAND[1])
                    depths = depths[mask]
                    ranks = ranks[mask]
                    best_index = int(np.argmin(ranks))
                    technical.append({
                        "seed": seed, "slug": slug, "family": record["target_family"],
                        "concept": label, "role": role_lookup[label], "method": method,
                        "best_depth": float(depths[best_index]),
                        "best_rank": float(ranks[best_index]),
                        "first_sustained_top1000_depth": first_sustained_depth(depths, ranks),
                        "sustained_top1000": bool(np.isfinite(first_sustained_depth(depths, ranks))),
                    })
    technical_frame = pd.DataFrame(technical)
    technical_frame.to_csv(OUT / "concept_role_onsets_by_seed.csv", index=False)
    units = (
        technical_frame.groupby(["slug", "family", "concept", "role", "method"], as_index=False)
        .agg(
            best_depth=("best_depth", "mean"),
            best_rank=("best_rank", "mean"),
            first_sustained_top1000_depth=("first_sustained_top1000_depth", "mean"),
            sustained_top1000=("sustained_top1000", "mean"),
        )
    )
    units["sustained_top1000"] = units.sustained_top1000 > 0
    units.to_csv(OUT / "concept_role_onsets.csv", index=False)

    rng = np.random.default_rng(SEED)
    roles = list(protocol["concept_roles"])
    methods_payload = {}
    for method, method_rows in units.groupby("method"):
        role_summary = {}
        for role in roles:
            subset = method_rows[method_rows.role.eq(role)]
            onset = subset.first_sustained_top1000_depth.dropna().to_numpy()
            role_summary[role] = {
                "n_units": int(len(subset)),
                "sustained_event_fraction": float(subset.sustained_top1000.mean()),
                "n_sustained": int(subset.sustained_top1000.sum()),
                "median_sustained_onset_depth": float(np.median(onset)) if len(onset) else None,
                "median_best_depth": float(subset.best_depth.median()),
            }

        observed_medians = {
            role: method_rows[method_rows.role.eq(role)].best_depth.median()
            for role in roles
        }
        observed_stat = max(
            abs(observed_medians[a] - observed_medians[b])
            for a, b in itertools.combinations(roles, 2)
        )
        null = np.empty(N_ROLE_PERM, dtype=float)
        family_concepts = {
            family: method_rows[method_rows.family.eq(family)][["concept", "role"]]
            .drop_duplicates().to_records(index=False)
            for family in sorted(method_rows.family.unique())
        }
        for iteration in range(N_ROLE_PERM):
            mapping = {}
            for family, entries in family_concepts.items():
                concepts = [entry[0] for entry in entries]
                shuffled_roles = rng.permutation([entry[1] for entry in entries])
                mapping.update({(family, concept): role for concept, role in zip(concepts, shuffled_roles)})
            perm_roles = np.asarray([
                mapping[(row.family, row.concept)] for row in method_rows.itertuples()
            ])
            medians = {
                role: np.median(method_rows.best_depth.to_numpy()[perm_roles == role])
                for role in roles if np.any(perm_roles == role)
            }
            null[iteration] = max(
                abs(medians[a] - medians[b])
                for a, b in itertools.combinations(medians, 2)
            )

        bootstrap = {}
        family_names = sorted(method_rows.family.unique())
        family_role_values = {
            (family, role): method_rows[
                (method_rows.family.eq(family)) & (method_rows.role.eq(role))
            ].best_depth.to_numpy()
            for family in family_names for role in roles
        }
        for role in roles:
            estimates = []
            for _ in range(N_BOOT):
                selected = rng.choice(family_names, size=len(family_names), replace=True)
                arrays = [
                    family_role_values[(family, role)]
                    for family in selected if len(family_role_values[(family, role)])
                ]
                if arrays:
                    estimates.append(float(np.median(np.concatenate(arrays))))
            bootstrap[role] = [float(v) for v in np.quantile(estimates, [0.025, 0.975])]
        methods_payload[method] = {
            "roles": role_summary,
            "family_clustered_best_depth_ci95": bootstrap,
            "max_pairwise_median_best_depth_difference": float(observed_stat),
            "family_blocked_max_difference_permutation_p": float(
                (1 + np.sum(null >= observed_stat - 1e-15)) / (1 + len(null))
            ),
            "permutation_resamples": N_ROLE_PERM,
        }
    payload = {
        "registered_band_percent": list(BAND),
        "concept_roles": protocol["concept_roles"],
        "methods": methods_payload,
    }
    dump_json(OUT / "concept_role_onset_statistics.json", payload)
    return units, payload


def plot_semantic_classification(items: pd.DataFrame, payload: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(payload["results"])
    filters = ["all", "prompt_exact_removed", "prompt_morphology_removed"]
    labels = ["All target-free", "Exact prompt\nwords removed", "Morphology-aware\nremoval"]
    colors = {"Jacobian": "#147A8A", "direct": "#777777"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    ax = axes[0]
    x = np.arange(len(filters)); width = 0.34
    for offset, method in ((-width / 2, "Jacobian"), (width / 2, "direct")):
        values = [
            float(results[(results.method.eq(method)) & (results["filter"].eq(mode))].accuracy.iloc[0])
            for mode in filters
        ]
        ax.bar(x + offset, values, width, label=method, color=colors[method])
    lexical = float(results[results.method.eq("input_TFIDF")].accuracy.iloc[0])
    ax.axhline(0.1, color="#999999", ls=":", label="chance")
    ax.axhline(lexical, color="#B35C44", ls="--", label="prompt TF-IDF")
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.02)
    ax.set_ylabel("Leave-one-phrasing-out accuracy")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.text(0.01, 0.98, "A", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[1]
    config = "Jacobian::prompt_morphology_removed"
    subset = items[items.config.eq(config)]
    names = sorted(subset.family.unique())
    matrix = pd.crosstab(subset.family, subset.prediction).reindex(
        index=names, columns=names, fill_value=0
    ).to_numpy()
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=5)
    for i in range(len(names)):
        for j in range(len(names)):
            if matrix[i, j]:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] >= 3 else "black")
    short = [name.replace("high-temperature-deformation", "high-T deform.")
             .replace("hot-air-surface-layer", "surface oxide")
             .replace("line-defect-motion", "line defect")
             .replace("particle-strengthening", "particle strength.")
             .replace("rapid-transformation", "transformation")
             .replace("notch-resistance", "notch")
             .replace("boundary-attack", "boundary attack") for name in names]
    ax.set_xticks(range(len(names)), short, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), short, fontsize=7)
    ax.set_xlabel("Predicted family"); ax.set_ylabel("True family")
    ax.text(-0.14, 1.04, "B", transform=ax.transAxes, ha="left", va="bottom",
            fontweight="bold", clip_on=False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="prompts")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"semantic-classification.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    x = np.arange(len(filters))
    containers = []
    for offset, method in ((-width / 2, "Jacobian"), (width / 2, "direct")):
        values = [
            float(results[(results.method.eq(method)) & (results["filter"].eq(mode))].accuracy.iloc[0])
            for mode in filters
        ]
        containers.append(
            ax.bar(x + offset, values, width, label=method, color=colors[method])
        )
    ax.axhline(0.1, color="#999999", ls=":", label="chance")
    ax.axhline(lexical, color="#B35C44", ls="--", label="prompt TF-IDF")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Leave-one-phrasing-out accuracy")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    for container in containers:
        ax.bar_label(
            container,
            labels=[f"{100 * bar.get_height():.0f}%" for bar in container],
            padding=3,
            fontsize=8,
        )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIG / f"semantic-classification-summary.{suffix}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)


def plot_rsa(frame: pd.DataFrame, payload: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=True)
    colors = {"Jacobian": "#147A8A", "raw": "#7562A8"}
    for ax, kind, panel in zip(axes, ("response_class", "multi_attribute"), ("A", "B")):
        subset = frame[(frame.ontology.eq(kind)) & frame.method.isin(["Jacobian", "raw"])]
        for method, group in subset.groupby("method"):
            ax.plot(group.depth, group.spearman_rho, marker="o", ms=3,
                    color=colors[method], label=method)
        target = payload["ontologies"][kind]["target"]["rho"]
        lexical = payload["ontologies"][kind]["lexical"]["rho"]
        ax.axhline(target, color="#4F8A5B", ls="--", label="target state")
        ax.axhline(lexical, color="#B35C44", ls=":", label="prompt embedding")
        ax.axhline(0, color="#999999", lw=0.7)
        ax.set_xlabel("Layer depth (%)")
        ax.set_title(kind.replace("_", " "))
        ax.text(0.01, 0.98, panel, transform=ax.transAxes, va="top", fontweight="bold")
    axes[0].set_ylabel("Spearman alignment with materials ontology")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"materials-ontology-rsa.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roles(units: pd.DataFrame, payload: dict) -> None:
    roles = list(payload["concept_roles"])
    labels = ["Entity/state", "Process/mechanism", "Response/property"]
    colors = {"Jacobian": "#147A8A", "direct": "#777777"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, method, panel in zip(axes, ("Jacobian", "direct"), ("A", "B")):
        subset = units[units.method.eq(method)]
        data = [subset[subset.role.eq(role)].best_depth.to_numpy() for role in roles]
        positions = np.arange(len(roles))
        box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
        for patch in box["boxes"]:
            patch.set_facecolor(colors[method]); patch.set_alpha(0.75)
        for position, values in zip(positions, data):
            jitter = np.linspace(-0.16, 0.16, len(values))
            ax.scatter(np.full(len(values), position) + jitter, values, s=8,
                       color="#333333", alpha=0.35)
        ax.set_xticks(positions, labels, rotation=12)
        ax.set_ylim(BAND[0] - 2, BAND[1] + 2)
        ax.set_ylabel("Best-rank layer depth (%)")
        ax.set_title(method)
        ax.text(0.01, 0.98, panel, transform=ax.transAxes, va="top", fontweight="bold")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"concept-role-depths.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(classification: dict, rsa: dict, roles: dict) -> None:
    results = {row["config"]: row for row in classification["results"]}
    j_clean = results["Jacobian::prompt_morphology_removed"]
    d_clean = results["direct::prompt_morphology_removed"]
    lexical = results["input_TFIDF::all"]
    rsa_multi = rsa["ontologies"]["multi_attribute"]
    role_j = roles["methods"]["Jacobian"]
    lines = [
        "# Candidate non-steering results",
        "",
        "These outputs are retrospective candidates under the frozen `PROTOCOL.md` and are reported with that limitation in the Supplementary Information.",
        "",
        "## Target-free mechanism classification",
        "",
        f"After morphology-aware prompt-word removal, Jacobian target-free vocabulary classifies {j_clean['accuracy']:.1%} of the 50 held-out prompts by mechanism family (macro-F1 {j_clean['macro_f1']:.3f}); direct vocabulary reaches {d_clean['accuracy']:.1%}. The prompt TF-IDF baseline reaches {lexical['accuracy']:.1%}. The Jacobian max-configuration-corrected permutation p-value is {j_clean['max_config_corrected_p']:.4g}.",
        f"Under the strict filter, Jacobian alone is correct for {classification['paired_jacobian_vs_direct']['prompt_morphology_removed']['jacobian_only_correct']} prompts and direct alone for {classification['paired_jacobian_vs_direct']['prompt_morphology_removed']['direct_only_correct']} (exact paired p={classification['paired_jacobian_vs_direct']['prompt_morphology_removed']['exact_two_sided_mcnemar_p']:.4g}).",
        "",
        "## Materials-ontology alignment",
        "",
        f"For the multi-attribute ontology, the strongest Jacobian layer has rho={rsa_multi['Jacobian']['best_abs_rho']:+.3f} at {rsa_multi['Jacobian']['best_abs_depth']:.1f}% depth (max-layer-corrected p={rsa_multi['Jacobian']['max_layer_corrected_p']:.4g}). Raw states peak at rho={rsa_multi['raw']['best_abs_rho']:+.3f}; the target and prompt-embedding baselines are {rsa_multi['target']['rho']:+.3f} and {rsa_multi['lexical']['rho']:+.3f}.",
        "",
        "## Concept-role depth",
        "",
        f"The maximum Jacobian difference among entity/state, process/mechanism, and response/property median best depths is {role_j['max_pairwise_median_best_depth_difference']:.1f} percentage points (family-blocked p={role_j['family_blocked_max_difference_permutation_p']:.4g}). Event rates and exact medians are in `concept_role_onset_statistics.json`.",
        "",
        "## Decision guidance",
        "",
        "The lexical-decontaminated classifier is the most directly useful addition if it remains above the corrected null: it turns target-free ribbons into a cross-validated quantitative result. Ontology RSA is valuable only if both frozen ontologies and leave-one-family-out checks agree. Concept-role ordering should remain supplementary unless it is strong, stable, and similar under both readouts.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    heldout, runs, protocol = validate_inputs()
    classification_items, classification = analyze_semantic_classification(heldout, runs)
    rsa_frame, rsa = analyze_ontology_rsa(protocol)
    role_units, roles = analyze_concept_roles(runs, protocol)
    plot_semantic_classification(classification_items, classification)
    plot_rsa(rsa_frame, rsa)
    plot_roles(role_units, roles)
    combined = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "semantic_classification": classification,
        "ontology_rsa": rsa,
        "concept_roles": roles,
    }
    dump_json(OUT / "candidate_nonsteering_results.json", combined)
    write_report(classification, rsa, roles)
    print((OUT / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
