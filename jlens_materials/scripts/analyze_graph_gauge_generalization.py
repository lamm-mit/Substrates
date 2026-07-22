#!/usr/bin/env python3
"""Gauge-invariant and permutation-group graph follow-up.

Executes Studies 4A, 4B, and the nonparametric portion of 4C from the frozen
``GAUGE_PROTOCOL.md``.  It only reads archived representations and results.
"""

from __future__ import annotations

import ast
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
from sklearn.metrics import roc_auc_score

from analyze_graph_isomorphism_generalization import (
    MANIFEST,
    METHODS,
    STATES,
    VARIANTS,
    fit_pair_mapping,
    heldout_vector,
    index_lookup,
    load_data,
    pearson,
    plus_one_signflip_p,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "experiments"
    / "graph-isomorphism-generalization-2026-07-18"
)
PROTOCOL = OUT / "protocol.json"
GAUGE_PROTOCOL = OUT / "gauge_protocol.json"
GAUGE_PROTOCOL_MD = OUT / "GAUGE_PROTOCOL.md"
MAPPINGS = OUT / "pair_mappings.csv"
CANDIDATES = OUT / "all_mapping_candidates.csv"
SEED = 20260718
N_CYCLE_NULL = 100_000
N_ATLAS_NULL = 10_000
TEMPERATURE = 0.05
PERMS = np.asarray(
    list(itertools.permutations(range(4))), dtype=np.int8
)
PERM_CODES = PERMS @ np.asarray([64, 16, 4, 1], dtype=np.int16)
CODE_TO_INDEX = np.full(256, -1, dtype=np.int16)
CODE_TO_INDEX[PERM_CODES] = np.arange(24, dtype=np.int16)
IDENTITY_INDEX = int(
    np.flatnonzero(np.all(PERMS == np.arange(4), axis=1))[0]
)
INVERSE_PERMS = np.argsort(PERMS, axis=1).astype(np.int8)


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
    parent = json.loads(PROTOCOL.read_text())
    followup = json.loads(GAUGE_PROTOCOL.read_text())
    expected_parent = followup["parent_protocol_sha256"]
    if sha256(PROTOCOL) != expected_parent:
        raise RuntimeError("parent protocol fingerprint mismatch")
    if (
        sha256(STATES)
        != followup["inputs"]["representations_sha256"]
        or sha256(MANIFEST)
        != followup["inputs"]["prompt_manifest_sha256"]
    ):
        raise RuntimeError("input fingerprint mismatch")
    if parent["study_id"] != "graph-isomorphism-generalization-2026-07-18":
        raise RuntimeError("unexpected parent study")


def parse_permutation(value: object) -> tuple[int, ...]:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    result = tuple(int(item) for item in parsed)
    if sorted(result) != list(range(4)):
        raise ValueError(f"invalid permutation: {value}")
    return result


def directed_map_lookup(
    frame: pd.DataFrame, family_order: Sequence[str]
) -> dict[tuple[str, str], np.ndarray]:
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for row in frame.itertuples(index=False):
        first = str(row.source_family)
        second = str(row.target_family)
        permutation = np.asarray(
            parse_permutation(row.permutation), dtype=np.int8
        )
        lookup[(first, second)] = permutation
        lookup[(second, first)] = np.argsort(permutation).astype(np.int8)
    expected = len(family_order) * (len(family_order) - 1)
    if len(lookup) != expected:
        raise RuntimeError("directed map lookup is incomplete")
    return lookup


def mapped_label_vectors(
    source_family: str,
    target_family: str,
    permutation: Sequence[int],
    labels: np.ndarray,
    data: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    cases = data["case_order"]
    lookup = index_lookup(data)
    source = []
    target = []
    for source_case_index, source_case in enumerate(cases[source_family]):
        target_case = cases[target_family][permutation[source_case_index]]
        for variant in VARIANTS:
            source.append(labels[lookup[(source_family, source_case, variant)]])
            target.append(labels[lookup[(target_family, target_case, variant)]])
    return np.asarray(source, dtype=bool), np.asarray(target, dtype=bool)


def agreement(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(first == second))


def gauge_agreement(first: np.ndarray, second: np.ndarray) -> float:
    ordinary = agreement(first, second)
    return max(ordinary, 1.0 - ordinary)


def response_code(data: Mapping[str, object]) -> np.ndarray:
    numeric = np.asarray(data["numeric"], dtype=bool)
    variants = np.asarray(data["variants"])
    return np.logical_xor(
        numeric,
        variants == "lexical_counterfactual",
    )


def structured_gauge_null(
    frame: pd.DataFrame, data: Mapping[str, object]
) -> np.ndarray:
    family_order = list(data["family_order"])
    cases = data["case_order"]
    lookup = index_lookup(data)
    balanced = list(itertools.combinations(range(4), 2))
    pair_specs = [
        (
            family_order.index(str(row.source_family)),
            family_order.index(str(row.target_family)),
            np.asarray(parse_permutation(row.permutation), dtype=int),
        )
        for row in frame.itertuples(index=False)
    ]
    results = np.empty(len(balanced) ** len(family_order), dtype=np.float32)
    for assignment_index, assignment in enumerate(
        itertools.product(range(len(balanced)), repeat=len(family_order))
    ):
        case_labels = np.zeros((len(family_order), 4), dtype=bool)
        for family_index, choice in enumerate(assignment):
            case_labels[family_index, list(balanced[choice])] = True
        pair_values = []
        for source_index, target_index, permutation in pair_specs:
            source_anchor = case_labels[source_index]
            target_anchor = case_labels[target_index, permutation]
            source_nodes = np.concatenate(
                [source_anchor, source_anchor, ~source_anchor]
            )
            target_nodes = np.concatenate(
                [target_anchor, target_anchor, ~target_anchor]
            )
            # The concatenation order is immaterial because both vectors use
            # the identical order; it represents A, P, then C.
            pair_values.append(
                gauge_agreement(source_nodes, target_nodes)
            )
        results[assignment_index] = float(np.mean(pair_values))
    return results


def study_4a(
    mappings: pd.DataFrame, data: Mapping[str, object]
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray]:
    physical = np.asarray(data["physical"], dtype=bool)
    numeric = np.asarray(data["numeric"], dtype=bool)
    prompt_code = response_code(data)
    pair_rows = []
    summaries = {}
    jacobian_null = np.asarray([])
    for method in METHODS:
        frame = mappings[
            (mappings["method"] == method)
            & (mappings["scope"] == "band")
        ].copy()
        for row in frame.itertuples(index=False):
            permutation = parse_permutation(row.permutation)
            values: dict[str, float] = {}
            for label_name, labels in (
                ("physical", physical),
                ("numeric", numeric),
                ("prompt_code", prompt_code),
            ):
                source, target = mapped_label_vectors(
                    str(row.source_family),
                    str(row.target_family),
                    permutation,
                    labels,
                    data,
                )
                values[f"{label_name}_ordinary"] = agreement(source, target)
                values[f"{label_name}_gauge"] = gauge_agreement(source, target)
            pair_rows.append(
                {
                    "method": method,
                    "source_family": row.source_family,
                    "target_family": row.target_family,
                    "permutation": str(permutation),
                    **values,
                }
            )
        method_pairs = pd.DataFrame(pair_rows)
        method_pairs = method_pairs[method_pairs["method"] == method]
        null = structured_gauge_null(frame, data)
        observed = float(method_pairs["physical_gauge"].mean())
        exact_p = float(
            (1 + np.sum(null >= observed - 1e-12)) / (1 + len(null))
        )
        summaries[method] = {
            "physical_ordinary_mean": float(
                method_pairs["physical_ordinary"].mean()
            ),
            "physical_gauge_mean": observed,
            "physical_perfect_gauge_pairs": int(
                np.sum(method_pairs["physical_gauge"] >= 1.0 - 1e-12)
            ),
            "structured_null_mean": float(np.mean(null)),
            "structured_exact_p": exact_p,
            "numeric_gauge_mean": float(
                method_pairs["numeric_gauge"].mean()
            ),
            "prompt_code_gauge_mean": float(
                method_pairs["prompt_code_gauge"].mean()
            ),
            "physical_minus_prompt_code": float(
                method_pairs["physical_gauge"].mean()
                - method_pairs["prompt_code_gauge"].mean()
            ),
        }
        summaries[method]["strong_gate"] = bool(
            exact_p <= 0.05
            and summaries[method]["physical_perfect_gauge_pairs"] >= 10
            and summaries[method]["physical_gauge_mean"]
            > summaries[method]["numeric_gauge_mean"]
            and summaries[method]["physical_gauge_mean"]
            > summaries[method]["prompt_code_gauge_mean"]
        )
        if method == "jacobian":
            jacobian_null = null
    return pd.DataFrame(pair_rows), summaries, jacobian_null


def permutation_index(permutation: Sequence[int]) -> int:
    code = int(np.asarray(permutation) @ np.asarray([64, 16, 4, 1]))
    index = int(CODE_TO_INDEX[code])
    if index < 0:
        raise ValueError("not a permutation")
    return index


def cycle_statistics(
    directed: Mapping[tuple[str, str], np.ndarray],
    family_order: Sequence[str],
) -> tuple[pd.DataFrame, float]:
    rows = []
    for first, second, third in itertools.combinations(family_order, 3):
        cycle = directed[(third, first)][
            directed[(second, third)][directed[(first, second)]]
        ]
        fixed = int(np.sum(cycle == np.arange(4)))
        rows.append(
            {
                "family_1": first,
                "family_2": second,
                "family_3": third,
                "cycle_permutation": str(tuple(int(item) for item in cycle)),
                "identity": bool(fixed == 4),
                "fixed_positions": fixed,
                "fixed_fraction": fixed / 4.0,
                "hamming_distance": 4 - fixed,
            }
        )
    frame = pd.DataFrame(rows)
    return frame, float(frame["fixed_fraction"].mean())


def null_cycle_fixed_fraction(
    family_order: Sequence[str],
    pair_probabilities: Mapping[tuple[str, str], np.ndarray] | None,
    n_null: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pairs = list(itertools.combinations(family_order, 2))
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    if pair_probabilities is None:
        draws = rng.integers(0, 24, size=(n_null, len(pairs)))
    else:
        draws = np.column_stack(
            [
                rng.choice(
                    24,
                    size=n_null,
                    p=pair_probabilities[pair],
                )
                for pair in pairs
            ]
        )
    forward = PERMS[draws]
    inverse = INVERSE_PERMS[draws]

    def directed_batch(source: str, target: str) -> np.ndarray:
        if source < target:
            return forward[:, pair_index[(source, target)]]
        return inverse[:, pair_index[(target, source)]]

    totals = np.zeros(n_null, dtype=np.float64)
    triples = list(itertools.combinations(family_order, 3))
    identity = np.arange(4)[None, :]
    for first, second, third in triples:
        first_second = directed_batch(first, second)
        second_third = directed_batch(second, third)
        third_first = directed_batch(third, first)
        step_two = np.take_along_axis(
            second_third, first_second, axis=1
        )
        cycle = np.take_along_axis(third_first, step_two, axis=1)
        totals += np.mean(cycle == identity, axis=1)
    return (totals / len(triples)).astype(np.float32)


def candidate_score_tables(
    candidates: pd.DataFrame,
    method: str,
    family_order: Sequence[str],
) -> dict[tuple[str, str], np.ndarray]:
    selected = candidates[
        (candidates["method"] == method)
        & (candidates["scope"] == "band")
    ]
    tables = {}
    for pair in itertools.combinations(family_order, 2):
        frame = selected[
            (selected["source_family"] == pair[0])
            & (selected["target_family"] == pair[1])
        ]
        table = np.full(24, -np.inf, dtype=np.float64)
        for row in frame.itertuples(index=False):
            table[permutation_index(parse_permutation(row.permutation))] = (
                float(row.fit_correlation)
            )
        if not np.all(np.isfinite(table)):
            raise RuntimeError(f"incomplete score table: {pair}, {method}")
        tables[pair] = table
    return tables


def score_probabilities(
    tables: Mapping[tuple[str, str], np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    probabilities = {}
    for pair, scores in tables.items():
        logits = (scores - np.max(scores)) / TEMPERATURE
        weights = np.exp(logits)
        probabilities[pair] = weights / np.sum(weights)
    return probabilities


def global_atlas_exact(
    tables: Mapping[tuple[str, str], np.ndarray],
    family_order: Sequence[str],
    *,
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, float, np.ndarray]:
    total_assignments = 24 ** (len(family_order) - 1)
    best_score = -np.inf
    best_assignment: np.ndarray | None = None
    random_scores = np.empty(N_ATLAS_NULL, dtype=np.float64)
    rng = np.random.default_rng(SEED + 701)
    random_assignment = np.column_stack(
        [
            np.full(N_ATLAS_NULL, IDENTITY_INDEX, dtype=np.int16),
            rng.integers(
                0,
                24,
                size=(N_ATLAS_NULL, len(family_order) - 1),
                dtype=np.int16,
            ),
        ]
    )

    def score_assignments(assignments: np.ndarray) -> np.ndarray:
        group_perms = PERMS[assignments]
        group_inverse = INVERSE_PERMS[assignments]
        result = np.zeros(len(assignments), dtype=np.float64)
        rows = np.arange(len(assignments))[:, None]
        del rows
        for first_index, second_index in itertools.combinations(
            range(len(family_order)), 2
        ):
            derived = np.take_along_axis(
                group_perms[:, second_index],
                group_inverse[:, first_index],
                axis=1,
            )
            codes = derived @ np.asarray(
                [64, 16, 4, 1], dtype=np.int16
            )
            indices = CODE_TO_INDEX[codes]
            pair = (
                family_order[first_index],
                family_order[second_index],
            )
            result += tables[pair][indices]
        return result

    random_scores[:] = score_assignments(random_assignment)
    powers = np.asarray(
        [24 ** index for index in range(len(family_order) - 1)],
        dtype=np.int64,
    )
    for start in range(0, total_assignments, chunk_size):
        stop = min(total_assignments, start + chunk_size)
        numbers = np.arange(start, stop, dtype=np.int64)
        digits = ((numbers[:, None] // powers[None, :]) % 24).astype(
            np.int16
        )
        assignments = np.column_stack(
            [
                np.full(len(numbers), IDENTITY_INDEX, dtype=np.int16),
                digits,
            ]
        )
        scores = score_assignments(assignments)
        local_index = int(np.argmax(scores))
        local_score = float(scores[local_index])
        if local_score > best_score + 1e-12:
            best_score = local_score
            best_assignment = assignments[local_index].copy()
    if best_assignment is None:
        raise RuntimeError("atlas search failed")
    return best_assignment, best_score, random_scores


def atlas_directed_maps(
    assignment: np.ndarray, family_order: Sequence[str]
) -> dict[tuple[str, str], np.ndarray]:
    group = PERMS[assignment]
    inverse = INVERSE_PERMS[assignment]
    result = {}
    for first_index, second_index in itertools.permutations(
        range(len(family_order)), 2
    ):
        result[
            (family_order[first_index], family_order[second_index])
        ] = group[second_index][inverse[first_index]]
    return result


def atlas_heldout(
    directed: Mapping[tuple[str, str], np.ndarray],
    data: Mapping[str, object],
    similarity: np.ndarray,
) -> pd.DataFrame:
    cases = data["case_order"]
    lookup = index_lookup(data)
    rows = []
    for first, second in itertools.combinations(data["family_order"], 2):
        permutation = directed[(first, second)]
        source = heldout_vector(
            similarity, first, cases[first], lookup
        )
        target = heldout_vector(
            similarity,
            second,
            cases[second],
            lookup,
            permutation,
        )
        rows.append(
            {
                "source_family": first,
                "target_family": second,
                "heldout_correlation": pearson(source, target),
            }
        )
    return pd.DataFrame(rows)


def study_4b(
    mappings: pd.DataFrame,
    candidates: pd.DataFrame,
    data: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, np.ndarray]]:
    family_order = list(data["family_order"])
    rng = np.random.default_rng(SEED)
    cycle_frames = []
    atlas_frames = []
    summaries = {}
    null_outputs = {}
    for method in METHODS:
        mapping_frame = mappings[
            (mappings["method"] == method)
            & (mappings["scope"] == "band")
        ]
        directed = directed_map_lookup(mapping_frame, family_order)
        cycles, observed = cycle_statistics(directed, family_order)
        cycles.insert(0, "method", method)
        cycle_frames.append(cycles)
        tables = candidate_score_tables(candidates, method, family_order)
        probabilities = score_probabilities(tables)
        uniform_null = null_cycle_fixed_fraction(
            family_order, None, N_CYCLE_NULL, rng
        )
        score_null = null_cycle_fixed_fraction(
            family_order, probabilities, N_CYCLE_NULL, rng
        )
        uniform_p = float(
            (1 + np.sum(uniform_null >= observed - 1e-12))
            / (1 + len(uniform_null))
        )
        score_p = float(
            (1 + np.sum(score_null >= observed - 1e-12))
            / (1 + len(score_null))
        )
        assignment, atlas_score, atlas_null = global_atlas_exact(
            tables, family_order
        )
        atlas_maps = atlas_directed_maps(assignment, family_order)
        similarity_layers = np.asarray(data["similarities"][method])
        band_similarity = np.mean(
            similarity_layers[np.asarray(data["band_mask"])], axis=0
        )
        heldout = atlas_heldout(atlas_maps, data, band_similarity)
        heldout.insert(0, "method", method)
        atlas_frames.append(heldout)
        atlas_p = float(
            (1 + np.sum(atlas_null >= atlas_score - 1e-12))
            / (1 + len(atlas_null))
        )
        assignment_dict = {
            family: tuple(int(value) for value in PERMS[index])
            for family, index in zip(family_order, assignment)
        }
        summaries[method] = {
            "cycle_fixed_fraction": observed,
            "cycle_identity_rate": float(cycles["identity"].mean()),
            "uniform_cycle_null_mean": float(np.mean(uniform_null)),
            "uniform_cycle_exact_p": uniform_p,
            "score_resampled_cycle_null_mean": float(np.mean(score_null)),
            "score_resampled_cycle_exact_p": score_p,
            "atlas_assignment": assignment_dict,
            "atlas_total_fit_score": atlas_score,
            "random_atlas_score_mean": float(np.mean(atlas_null)),
            "random_atlas_score_exact_p": atlas_p,
            "atlas_heldout_mean_correlation": float(
                heldout["heldout_correlation"].mean()
            ),
            "atlas_heldout_positive_pairs": int(
                np.sum(heldout["heldout_correlation"] > 0)
            ),
        }
        summaries[method]["strong_atlas_gate"] = bool(
            uniform_p <= 0.05
            and score_p <= 0.05
            and atlas_p <= 0.05
            and summaries[method]["atlas_heldout_positive_pairs"] >= 10
        )
        null_outputs[f"{method}_uniform_cycle"] = uniform_null
        null_outputs[f"{method}_score_cycle"] = score_null
        null_outputs[f"{method}_random_atlas_score"] = atlas_null
    return (
        pd.concat(cycle_frames, ignore_index=True),
        pd.concat(atlas_frames, ignore_index=True),
        summaries,
        null_outputs,
    )


def candidate_pairs_for_family(
    family: str, data: Mapping[str, object]
) -> list[tuple[int, int]]:
    families = np.asarray(data["families"])
    cases = np.asarray(data["cases"])
    variants = np.asarray(data["variants"])
    indices = np.flatnonzero(families == family)
    return [
        (int(source), int(target))
        for source in indices
        for target in indices
        if variants[source] != variants[target]
        and cases[source] != cases[target]
    ]


def relation_auc(
    similarity: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    labels: np.ndarray,
) -> float:
    outcomes = np.asarray(
        [labels[first] == labels[second] for first, second in pairs],
        dtype=int,
    )
    scores = np.asarray(
        [similarity[first, second] for first, second in pairs],
        dtype=float,
    )
    return float(roc_auc_score(outcomes, scores))


def balanced_relation_null(
    similarity: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    family_indices: np.ndarray,
) -> np.ndarray:
    results = []
    for positives in itertools.combinations(range(12), 6):
        local = np.zeros(12, dtype=bool)
        local[list(positives)] = True
        labels = np.zeros(72, dtype=bool)
        labels[family_indices] = local
        results.append(relation_auc(similarity, pairs, labels))
    return np.asarray(results, dtype=np.float32)


def study_4c_nonparametric(
    data: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, np.ndarray]]:
    physical = np.asarray(data["physical"], dtype=bool)
    numeric = np.asarray(data["numeric"], dtype=bool)
    prompt_code = response_code(data)
    families = np.asarray(data["families"])
    family_order = list(data["family_order"])
    band_mask = np.asarray(data["band_mask"])
    rows = []
    nulls = {}
    for method in METHODS:
        layers = np.asarray(data["similarities"][method])
        band = np.mean(layers[band_mask], axis=0)
        for family in family_order:
            pairs = candidate_pairs_for_family(family, data)
            family_indices = np.flatnonzero(families == family)
            physical_auc = relation_auc(band, pairs, physical)
            numeric_auc = relation_auc(band, pairs, numeric)
            prompt_auc = relation_auc(band, pairs, prompt_code)
            label_null = balanced_relation_null(
                band, pairs, family_indices
            )
            label_p = float(
                (1 + np.sum(label_null >= physical_auc - 1e-12))
                / (1 + len(label_null))
            )
            shift_aucs = []
            for shift in range(1, len(layers)):
                shifted_mask = np.roll(band_mask, shift)
                shifted = np.mean(layers[shifted_mask], axis=0)
                shift_aucs.append(
                    relation_auc(shifted, pairs, physical)
                )
            rows.append(
                {
                    "method": method,
                    "family": family,
                    "physical_auc": physical_auc,
                    "numeric_auc": numeric_auc,
                    "prompt_code_auc": prompt_auc,
                    "balanced_label_null_mean": float(
                        np.mean(label_null)
                    ),
                    "balanced_label_exact_p": label_p,
                    "circular_shift_mean_auc": float(
                        np.mean(shift_aucs)
                    ),
                    "physical_minus_shift_mean": float(
                        physical_auc - np.mean(shift_aucs)
                    ),
                }
            )
            nulls[f"{method}_{family}_balanced_labels"] = label_null
    frame = pd.DataFrame(rows)
    summaries = {}
    raw = frame[frame["method"] == "raw"].set_index("family")
    for method in METHODS:
        selected = frame[frame["method"] == method].set_index("family")
        aucs = selected.loc[family_order, "physical_auc"].to_numpy()
        prompt_contrast = (
            selected.loc[family_order, "physical_auc"].to_numpy()
            - selected.loc[family_order, "prompt_code_auc"].to_numpy()
        )
        raw_contrast = (
            selected.loc[family_order, "physical_auc"].to_numpy()
            - raw.loc[family_order, "physical_auc"].to_numpy()
        )
        summaries[method] = {
            "physical_auc_by_family": dict(
                zip(family_order, aucs.tolist())
            ),
            "mean_physical_auc": float(np.mean(aucs)),
            "families_above_half": int(np.sum(aucs > 0.5)),
            "auc_minus_half_plus_one_signflip_p": plus_one_signflip_p(
                aucs - 0.5
            ),
            "physical_over_prompt_families": int(
                np.sum(prompt_contrast > 0)
            ),
            "physical_minus_prompt_signflip_p": plus_one_signflip_p(
                prompt_contrast
            ),
            "physical_over_raw_families": int(
                np.sum(raw_contrast > 0)
            ),
            "physical_minus_raw_signflip_p": plus_one_signflip_p(
                raw_contrast
            ),
        }
        summaries[method]["strong_nonparametric_gate"] = bool(
            summaries[method]["families_above_half"] == 6
            and summaries[method]["auc_minus_half_plus_one_signflip_p"]
            <= 0.05
            and summaries[method]["physical_over_prompt_families"] >= 5
            and summaries[method]["physical_minus_prompt_signflip_p"]
            <= 0.05
            and summaries[method]["physical_over_raw_families"] >= 5
            and summaries[method]["physical_minus_raw_signflip_p"]
            <= 0.05
        )
    return frame, summaries, nulls


def main() -> None:
    validate_inputs()
    data = load_data()
    mappings = pd.read_csv(MAPPINGS)
    candidates = pd.read_csv(CANDIDATES)
    gauge_pairs, gauge_summary, gauge_null = study_4a(mappings, data)
    (
        cycles,
        atlas_heldout,
        atlas_summary,
        atlas_nulls,
    ) = study_4b(mappings, candidates, data)
    relation_rows, relation_summary, relation_nulls = (
        study_4c_nonparametric(data)
    )
    gauge_pairs.to_csv(OUT / "gauge_pair_agreement.csv", index=False)
    cycles.to_csv(OUT / "permutation_cycles.csv", index=False)
    atlas_heldout.to_csv(
        OUT / "global_atlas_heldout.csv", index=False
    )
    relation_rows.to_csv(
        OUT / "relation_nonparametric.csv", index=False
    )
    np.savez_compressed(
        OUT / "gauge_and_relation_nulls.npz",
        jacobian_gauge_structured=gauge_null,
        **atlas_nulls,
        **relation_nulls,
    )
    results = {
        "study_id": json.loads(GAUGE_PROTOCOL.read_text())["study_id"],
        "gauge_protocol_sha256": sha256(GAUGE_PROTOCOL),
        "gauge_protocol_markdown_sha256": sha256(GAUGE_PROTOCOL_MD),
        "parent_protocol_sha256": sha256(PROTOCOL),
        "study_4a_gauge_alignment": gauge_summary,
        "study_4b_permutation_atlas": atlas_summary,
        "study_4c_nonparametric_relation": relation_summary,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (OUT / "study4abc_statistics.json").write_text(
        json.dumps(safe_json(results), indent=2) + "\n"
    )
    print(json.dumps(safe_json(results), indent=2))


if __name__ == "__main__":
    main()
