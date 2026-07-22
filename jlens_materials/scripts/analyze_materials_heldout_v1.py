#!/usr/bin/env python3
"""Preregistered analysis of the three-seed materials held-out v1 study.

The script keeps lens-fit seeds as repeated measurements, estimates population
uncertainty over mechanism families and phrasings, and builds target-free
candidate sets for a later blinded family-identification task.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paper_protocol as pp  # noqa: E402


RUN_PATHS = [
    ROOT / "runs" / f"gemma4-e4b-it-heldout-v1-seed{seed}.json"
    for seed in range(3)
]
EXPECTED_RUN_HASHES = {
    "gemma4-e4b-it-heldout-v1-seed0.json":
        "8297783dd1314808f7a08d94abc2427f44554f68e494bf0fa940d493a327d46e",
    "gemma4-e4b-it-heldout-v1-seed1.json":
        "6397e3faf19ef52082ef25845c3d6e51815c15c803d2b269c25edffc616169cc",
    "gemma4-e4b-it-heldout-v1-seed2.json":
        "ae7d9e25df1a9179f2fce6dcd30efbdfd610bd8ae09563cc09e85234accad1c8",
}
EXP_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "figures" / "materials-heldout-v1"
STATS_PATH = EXP_DIR / "materials-heldout-v1_statistics.json"
PROMPT_CSV = EXP_DIR / "materials-heldout-v1_prompt_results.csv"
CONCEPT_CSV = EXP_DIR / "materials-heldout-v1_concept_results.csv"
CANDIDATE_CSV = EXP_DIR / "materials-heldout-v1_open_vocabulary_candidates.csv"
BLIND_CSV = EXP_DIR / "MATERIALS_HELDOUT_V1_BLINDED_SETS.csv"
KEY_CSV = EXP_DIR / "MATERIALS_HELDOUT_V1_BLINDED_KEY.csv"
REPORT_PATH = EXP_DIR / "MATERIALS_HELDOUT_V1_ANALYSIS.md"
CONTROLLED_FIGURE = FIG_DIR / "heldout-controlled-recovery"
DISCOVERY_FIGURE = FIG_DIR / "heldout-open-vocabulary"
REPRO_FIGURE = FIG_DIR / "figure-s1-fit-reproducibility"
RNG_SEED = 20260714
N_BOOT = 20_000
N_CORR_BOOT = 5_000
TOP_FAMILY_CANDIDATES = 8
DISPLAY_FAMILY = {
    "boundary-attack": "boundary attack",
    "cleavage": "cleavage",
    "cyclic": "cyclic damage",
    "ductile": "ductile failure",
    "high-temperature-deformation": "high-temperature deformation",
    "hot-air-surface-layer": "surface oxidation",
    "line-defect-motion": "line-defect motion",
    "notch-resistance": "notch resistance",
    "particle-strengthening": "particle strengthening",
    "rapid-transformation": "rapid transformation",
}


# Frozen, target-agnostic English function-word filter. It contains no
# materials-science terms and is the same list used in the development study.
FUNCTION_WORDS = set("""
a about above across after afterwards again against all almost along already
also although always am among an and another any are around as at away back be
became because become been before began behind being below between both but by
can cannot could did do does done down due during each either else enough
especially even ever every few for from further furthermore get got had has
have having he hence her here hers herself him himself his how however i if in
indeed into is it its itself just later least less like likely may me meanwhile
might mine more most much must my myself nearly neither never nevertheless no
nobody nor not nothing now of off often on once one only or other others
otherwise our ours ourselves out over own perhaps quite rather really said same
several she should since so some somebody someone something sometimes still
such than that the their theirs them themselves then thence there therefore
these they this those though through thus to too toward under until up upon us
very was we were what whatever when where whereas whether which while who whom
whose why will with within without would yet you your yours yourself yourselves
subsequently consequently simultaneously accordingly eventually ultimately
initially thereto thereafter
""".split())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed_records(run: dict) -> dict[str, dict]:
    return {
        record["slug"]: record
        for record in run["prompts"]
        if record.get("shape") == "ASSOCIATION"
    }


def validate_runs(runs: list[dict], run_paths: list[Path]) -> tuple[list[str], list[dict[str, dict]]]:
    if len(runs) != 3:
        raise ValueError("exactly three lens-fit seeds are required")
    indexes = [indexed_records(run) for run in runs]
    slugs = list(indexes[0])
    if len(slugs) != 50:
        raise ValueError(f"expected 50 association prompts, found {len(slugs)}")
    for seed, (run, index, path) in enumerate(zip(runs, indexes, run_paths, strict=True)):
        expected_hash = EXPECTED_RUN_HASHES.get(path.name)
        if expected_hash and sha256(path) != expected_hash:
            raise ValueError(f"raw run hash changed after freezing: {path}")
        if run.get("errors"):
            raise ValueError(f"seed {seed} contains run errors")
        if not run.get("methodology", {}).get("paper_protocol_complete"):
            raise ValueError(f"seed {seed} is not paper-protocol complete")
        if list(index) != slugs:
            raise ValueError(f"seed {seed} prompt set or order differs")
        if not all(record.get("valid_for_metrics") for record in index.values()):
            raise ValueError(f"seed {seed} contains excluded prompts")
        for record in index.values():
            config = record.get("open_vocabulary_config", {})
            if config.get("surprising_top") != 64:
                raise ValueError(f"{record['slug']} did not retain 64 candidates")
            if not config.get("matched_logit_open_vocabulary_stored"):
                raise ValueError(f"{record['slug']} lacks matched logit candidates")
    return slugs, indexes


def item_auc(zero_indexed_ranks: list[int], ks: list[int]) -> float:
    values = [
        float(np.mean([0 <= rank < k for rank in zero_indexed_ranks]))
        for k in ks
    ]
    return float(pp.log_k_auc(ks, values))


def percentile_summary(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "n_resamples": int(len(values)),
    }


def hierarchical_effect_bootstrap(rows: list[dict], *, seed: int) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(float(row["delta_auc_mean_seed"]))
    families = sorted(grouped)
    values = [np.asarray(grouped[family], dtype=float) for family in families]
    rng = np.random.default_rng(seed)
    estimates = np.empty(N_BOOT, dtype=float)
    for iteration in range(N_BOOT):
        selected_families = rng.integers(0, len(values), size=len(values))
        family_means = []
        for family_index in selected_families:
            family_values = values[int(family_index)]
            selected_prompts = rng.integers(0, len(family_values), size=len(family_values))
            family_means.append(float(np.mean(family_values[selected_prompts])))
        estimates[iteration] = float(np.mean(family_means))
    summary = percentile_summary(estimates)
    summary["observed_mean"] = float(np.mean([
        row["delta_auc_mean_seed"] for row in rows
    ]))
    summary["unit"] = "10 families; five phrasings resampled within selected family"
    return summary


def exact_family_sign_flip(family_effects: dict[str, float]) -> dict:
    values = np.asarray([family_effects[family] for family in sorted(family_effects)])
    observed = float(np.mean(values))
    null = np.asarray([
        float(np.mean(values * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    return {
        "observed_mean": observed,
        "p_one_sided_j_greater": float(np.mean(null >= observed - 1e-15)),
        "p_two_sided": float(np.mean(np.abs(null) >= abs(observed) - 1e-15)),
        "n_permutations": int(len(null)),
        "unit": "mechanism-family mean effect",
    }


def spearman_value(first: np.ndarray, second: np.ndarray) -> float:
    if len(np.unique(first)) < 2 or len(np.unique(second)) < 2:
        return float("nan")
    return float(spearmanr(first, second).statistic)


def family_clustered_spearman(rows: list[dict], first_key: str, second_key: str, *, seed: int) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    families = sorted(grouped)
    observed = spearman_value(
        np.asarray([row[first_key] for row in rows], dtype=float),
        np.asarray([row[second_key] for row in rows], dtype=float),
    )
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(N_CORR_BOOT):
        selected = rng.integers(0, len(families), size=len(families))
        sample = [row for index in selected for row in grouped[families[int(index)]]]
        estimate = spearman_value(
            np.asarray([row[first_key] for row in sample], dtype=float),
            np.asarray([row[second_key] for row in sample], dtype=float),
        )
        if math.isfinite(estimate):
            estimates.append(estimate)
    array = np.asarray(estimates, dtype=float)
    return {
        "rho": observed,
        "family_clustered_low": float(np.quantile(array, 0.025)),
        "family_clustered_high": float(np.quantile(array, 0.975)),
        "n_rows": len(rows),
        "n_resamples": len(estimates),
    }


def build_controlled_rows(
    runs: list[dict], slugs: list[str], indexes: list[dict[str, dict]]
) -> tuple[list[dict], list[dict], list[int]]:
    ks = list(runs[0]["metrics"]["by_shape"]["ASSOCIATION"]["ks"])
    prompt_rows: list[dict] = []
    concept_rows: list[dict] = []
    for slug in slugs:
        records = [index[slug] for index in indexes]
        reference = records[0]
        labels_by_seed = [
            [item["label"] for item in record["emergence"]]
            for record in records
        ]
        if any(labels != labels_by_seed[0] for labels in labels_by_seed[1:]):
            raise ValueError(f"tracked concepts differ across seeds for {slug}")
        j_aucs = [
            item_auc([int(item["best_rank"]) for item in record["emergence"]], ks)
            for record in records
        ]
        logit_auc = item_auc(
            [int(item["logit_lens_best_rank"]) for item in reference["emergence"]],
            ks,
        )
        family = reference.get("target_family") or reference["category"]
        prompt_rows.append({
            "slug": slug,
            "family": family,
            "phrasing_id": reference.get("phrasing_id"),
            "prompt": reference["prompt_text"],
            "generated_completion": reference.get("generated_completion"),
            "n_resolved_concepts": len(reference["emergence"]),
            "dropped_concepts": ";".join(reference.get("tracked_dropped", [])),
            "j_auc_seed0": j_aucs[0],
            "j_auc_seed1": j_aucs[1],
            "j_auc_seed2": j_aucs[2],
            "j_auc_mean_seed": float(np.mean(j_aucs)),
            "logit_auc": logit_auc,
            "delta_auc_mean_seed": float(np.mean(j_aucs) - logit_auc),
        })
        for concept_index, label in enumerate(labels_by_seed[0]):
            emergence = [record["emergence"][concept_index] for record in records]
            j_ranks = [int(item["best_rank"]) + 1 for item in emergence]
            logit_rank = int(emergence[0]["logit_lens_best_rank"]) + 1
            concept_rows.append({
                "slug": slug,
                "family": family,
                "concept": label,
                "j_rank_seed0": j_ranks[0],
                "j_rank_seed1": j_ranks[1],
                "j_rank_seed2": j_ranks[2],
                "j_log_rank_seed0": math.log10(j_ranks[0]),
                "j_log_rank_seed1": math.log10(j_ranks[1]),
                "j_log_rank_seed2": math.log10(j_ranks[2]),
                "j_log_rank_mean_seed": float(np.mean(np.log10(j_ranks))),
                "logit_rank": logit_rank,
                "logit_log_rank": math.log10(logit_rank),
                "log10_rank_advantage": float(
                    math.log10(logit_rank) - np.mean(np.log10(j_ranks))
                ),
            })
    return prompt_rows, concept_rows, ks


def build_controlled_statistics(
    runs: list[dict], prompt_rows: list[dict], concept_rows: list[dict], ks: list[int]
) -> dict:
    per_seed = []
    for seed in range(3):
        j_values = np.asarray([row[f"j_auc_seed{seed}"] for row in prompt_rows])
        logit_values = np.asarray([row["logit_auc"] for row in prompt_rows])
        per_seed.append({
            "seed": seed,
            "j_auc_mean": float(np.mean(j_values)),
            "logit_auc_mean": float(np.mean(logit_values)),
            "delta_auc": float(np.mean(j_values - logit_values)),
            "j_wins": int(np.sum(j_values > logit_values + 1e-12)),
            "ties": int(np.sum(np.abs(j_values - logit_values) <= 1e-12)),
            "j_losses": int(np.sum(j_values < logit_values - 1e-12)),
        })

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in prompt_rows:
        grouped[row["family"]].append(row)
    families = {}
    for family, rows in sorted(grouped.items()):
        family_j = np.asarray([row["j_auc_mean_seed"] for row in rows])
        family_logit = np.asarray([row["logit_auc"] for row in rows])
        families[family] = {
            "n_phrasings": len(rows),
            "j_auc_mean": float(np.mean(family_j)),
            "logit_auc_mean": float(np.mean(family_logit)),
            "delta_auc_mean": float(np.mean(family_j - family_logit)),
            "prompt_deltas": (family_j - family_logit).tolist(),
        }

    j_mean = np.asarray([row["j_auc_mean_seed"] for row in prompt_rows])
    logit = np.asarray([row["logit_auc"] for row in prompt_rows])
    family_effects = {
        family: row["delta_auc_mean"] for family, row in families.items()
    }
    correlations = {}
    for first in range(3):
        for second in range(first + 1, 3):
            label = f"seed{first}_vs_seed{second}"
            correlations[label] = family_clustered_spearman(
                concept_rows,
                f"j_log_rank_seed{first}",
                f"j_log_rank_seed{second}",
                seed=RNG_SEED + 10 * first + second,
            )
    correlations["mean_j_vs_logit"] = family_clustered_spearman(
        concept_rows,
        "j_log_rank_mean_seed",
        "logit_log_rank",
        seed=RNG_SEED + 100,
    )

    pass_at_k_j = np.asarray([
        run["metrics"]["by_shape"]["ASSOCIATION"]["jacobian_lens"]["pass_at_k"]
        for run in runs
    ], dtype=float)
    return {
        "overall": {
            "n_prompts": len(prompt_rows),
            "n_families": len(families),
            "n_resolved_prompt_concepts": len(concept_rows),
            "j_auc_mean": float(np.mean(j_mean)),
            "logit_auc_mean": float(np.mean(logit)),
            "delta_auc": float(np.mean(j_mean - logit)),
            "relative_auc_gain": float(np.mean(j_mean) / np.mean(logit) - 1),
            "j_wins": int(np.sum(j_mean > logit + 1e-12)),
            "ties": int(np.sum(np.abs(j_mean - logit) <= 1e-12)),
            "j_losses": int(np.sum(j_mean < logit - 1e-12)),
            "hierarchical_family_bootstrap": hierarchical_effect_bootstrap(
                prompt_rows, seed=RNG_SEED
            ),
            "exact_family_sign_flip": exact_family_sign_flip(family_effects),
        },
        "per_seed": per_seed,
        "families": families,
        "seed_rank_correlations": correlations,
        "pass_at_k": {
            "ks": ks,
            "j_by_seed": pass_at_k_j.tolist(),
            "j_mean": np.mean(pass_at_k_j, axis=0).tolist(),
            "logit": runs[0]["metrics"]["by_shape"]["ASSOCIATION"]["logit_lens"]["pass_at_k"],
        },
    }


def normalize_token(value: str) -> str | None:
    token = value.strip().lower()
    if not re.fullmatch(r"[a-z][a-z'-]{2,}", token):
        return None
    return token


def target_terms(record: dict) -> set[str]:
    terms = set()
    for concept in record.get("tracked", []):
        for value in (concept.get("label"), *concept.get("surfaces", [])):
            token = normalize_token(value or "")
            if token:
                terms.add(token)
    return terms


def consensus_candidates(
    records: list[dict], *, source_key: str, filter_function_words: bool
) -> list[dict]:
    by_seed = []
    for record in records:
        candidates = {}
        for row in record.get(source_key, []):
            token = normalize_token(row.get("concept", ""))
            if token is None or (filter_function_words and token in FUNCTION_WORDS):
                continue
            candidates[token] = row
        by_seed.append(candidates)
    shared = set.intersection(*(set(seed_rows) for seed_rows in by_seed))
    rows = []
    for token in shared:
        source = [seed_rows[token] for seed_rows in by_seed]
        rows.append({
            "token": token,
            "consensus_score": float(np.mean([row["score"] for row in source])),
            "worst_best_rank": int(max(row["best_rank"] for row in source)),
            "mean_best_depth": float(np.mean([row["best_depth"] for row in source])),
            "scores_by_seed": [float(row["score"]) for row in source],
            "depths_by_seed": [float(row["best_depth"]) for row in source],
            "seed_support": 3,
        })
    rows.sort(key=lambda row: (
        -row["consensus_score"], row["worst_best_rank"], row["token"]
    ))
    return rows


def method_discovery(
    method: str,
    source_key: str,
    slugs: list[str],
    indexes: list[dict[str, dict]],
) -> dict:
    per_prompt = []
    filtered_by_slug: dict[str, list[dict]] = {}
    raw_by_slug: dict[str, list[dict]] = {}
    for slug in slugs:
        records = [index[slug] for index in indexes]
        reference = records[0]
        raw = consensus_candidates(
            records, source_key=source_key, filter_function_words=False
        )
        filtered = consensus_candidates(
            records, source_key=source_key, filter_function_words=True
        )
        terms = target_terms(reference)
        for candidate in filtered:
            candidate["exact_predeclared_overlap"] = candidate["token"] in terms
        raw_by_slug[slug] = raw
        filtered_by_slug[slug] = filtered
        per_prompt.append({
            "slug": slug,
            "family": reference.get("target_family") or reference["category"],
            "prompt": reference["prompt_text"],
            "predeclared_terms_annotation_only": sorted(terms),
            "raw_consensus_count": len(raw),
            "filtered_consensus_count": len(filtered),
            "filtered_consensus_candidates": filtered,
        })

    raw_df: dict[str, int] = defaultdict(int)
    raw_score: dict[str, float] = defaultdict(float)
    filtered_df: dict[str, int] = defaultdict(int)
    for slug in slugs:
        for row in raw_by_slug[slug]:
            raw_df[row["token"]] += 1
            raw_score[row["token"]] += row["consensus_score"]
        for row in filtered_by_slug[slug]:
            filtered_df[row["token"]] += 1
    global_scaffold = sorted(({
        "token": token,
        "prompt_support": raw_df[token],
        "total_consensus_score": raw_score[token],
    } for token in raw_df), key=lambda row: (
        -row["prompt_support"], -row["total_consensus_score"], row["token"]
    ))

    family_slugs: dict[str, list[str]] = defaultdict(list)
    for row in per_prompt:
        family_slugs[row["family"]].append(row["slug"])
    families = {}
    for family, members in sorted(family_slugs.items()):
        family_targets = set().union(*(
            set(next(row["predeclared_terms_annotation_only"] for row in per_prompt
                     if row["slug"] == slug))
            for slug in members
        ))
        values: dict[str, list[dict]] = defaultdict(list)
        for slug in members:
            for row in filtered_by_slug[slug]:
                values[row["token"]].append(row)
        candidates = []
        for token, rows in values.items():
            inverse_document_frequency = math.log(len(slugs) / filtered_df[token])
            score = (
                sum(row["consensus_score"] for row in rows)
                * inverse_document_frequency
                / len(members)
            )
            candidates.append({
                "token": token,
                "family_specificity_score": float(score),
                "prompt_support": len(rows),
                "prompt_denominator": len(members),
                "global_prompt_frequency": filtered_df[token],
                "inverse_document_frequency": float(inverse_document_frequency),
                "mean_consensus_score_when_present": float(np.mean([
                    row["consensus_score"] for row in rows
                ])),
                "exact_predeclared_overlap": token in family_targets,
            })
        candidates.sort(key=lambda row: (
            -row["family_specificity_score"], -row["prompt_support"], row["token"]
        ))
        families[family] = {
            "n_prompts": len(members),
            "predeclared_terms_annotation_only": sorted(family_targets),
            "candidates": candidates[:TOP_FAMILY_CANDIDATES],
        }

    overlap = {}
    for cutoff in (1, 3, 5, 8):
        hits = [
            any(row["exact_predeclared_overlap"] for row in result["candidates"][:cutoff])
            for result in families.values()
        ]
        overlap[str(cutoff)] = {
            "families_with_exact_overlap": int(sum(hits)),
            "n_families": len(hits),
            "fraction": float(np.mean(hits)),
        }

    return {
        "method": method,
        "source_key": source_key,
        "n_prompts": len(slugs),
        "n_families": len(families),
        "global_scaffold": global_scaffold[:20],
        "families": families,
        "exact_predeclared_overlap_at_k": overlap,
        "per_prompt": per_prompt,
    }


def build_discovery_statistics(
    slugs: list[str], indexes: list[dict[str, dict]]
) -> dict:
    methods = {
        "jacobian": method_discovery("jacobian", "surprising", slugs, indexes),
        "logit": method_discovery("logit", "logit_surprising", slugs, indexes),
    }
    return {
        "analysis_status": "preregistered held-out target-free candidate generation",
        "candidate_source": (
            "unrestricted full-vocabulary top-1 tokens across all prompt positions "
            "and registered 38-92% source-layer band"
        ),
        "consensus_rule": "candidate must occur in all three stored seed lists",
        "filter_rule": (
            "exclude prompt/output tokens upstream; retain lowercase alphabetic "
            "tokens; remove frozen target-agnostic English function words"
        ),
        "ranking_rule": (
            "within-family sum of three-seed consensus score times "
            "log(50/global prompt frequency), divided by five phrasings"
        ),
        "predeclared_terms_use": (
            "annotation after candidate ranking only; not used to generate, "
            "filter, retain, or rank candidates"
        ),
        "candidate_retention_per_seed": 64,
        "top_family_candidates": TOP_FAMILY_CANDIDATES,
        "methods": methods,
        "function_words": sorted(FUNCTION_WORDS),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_csv(discovery: dict) -> None:
    rows = []
    for method, result in discovery["methods"].items():
        for prompt in result["per_prompt"]:
            for rank, candidate in enumerate(
                prompt["filtered_consensus_candidates"], start=1
            ):
                rows.append({
                    "method": method,
                    "slug": prompt["slug"],
                    "family": prompt["family"],
                    "candidate_rank": rank,
                    "candidate": candidate["token"],
                    "consensus_score": candidate["consensus_score"],
                    "worst_best_rank": candidate["worst_best_rank"],
                    "mean_best_depth": candidate["mean_best_depth"],
                    "exact_predeclared_overlap_after_ranking": (
                        candidate["exact_predeclared_overlap"]
                    ),
                    "prompt": prompt["prompt"],
                })
    write_csv(CANDIDATE_CSV, rows)


def write_blinded_sets(discovery: dict) -> None:
    entries = [
        (method, family, result)
        for method, method_result in discovery["methods"].items()
        for family, result in method_result["families"].items()
    ]
    rng = random.Random(RNG_SEED)
    rng.shuffle(entries)
    blind_rows = []
    key_rows = []
    for index, (method, family, result) in enumerate(entries, start=1):
        set_id = f"SET-{index:02d}"
        blind_row = {"set_id": set_id}
        for position in range(TOP_FAMILY_CANDIDATES):
            candidate = (
                result["candidates"][position]
                if position < len(result["candidates"]) else None
            )
            blind_row[f"candidate_{position + 1}"] = (
                candidate["token"] if candidate else ""
            )
            blind_row[f"support_{position + 1}"] = (
                f"{candidate['prompt_support']}/5" if candidate else ""
            )
        blind_rows.append(blind_row)
        key_rows.append({
            "set_id": set_id,
            "method": method,
            "mechanism_family": family,
        })
    write_csv(BLIND_CSV, blind_rows)
    write_csv(KEY_CSV, key_rows)


def plot_controlled(controlled: dict, prompt_rows: list[dict], concept_rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    jacobian_color = "#087F8C"
    logit_color = "#66727A"
    negative_color = "#C46C5A"
    neutral = "#687078"
    figure = plt.figure(figsize=(13.4, 8.7), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[1, :])]

    pass_at_k = controlled["pass_at_k"]
    axis = axes[0]
    for values in pass_at_k["j_by_seed"]:
        axis.plot(pass_at_k["ks"], values, color=jacobian_color, alpha=0.25, linewidth=1.2)
    axis.plot(pass_at_k["ks"], pass_at_k["j_mean"], "o-", color=jacobian_color,
              linewidth=2.4, label="Jacobian lens (3-fit mean)")
    axis.plot(pass_at_k["ks"], pass_at_k["logit"], "s-", color=logit_color,
              linewidth=2.0, label="Direct unembedding")
    axis.set_xscale("log")
    axis.set_xlabel("Vocabulary cutoff k (lower is stricter)")
    axis.set_ylabel("Fraction of predeclared concepts within top k")
    axis.set_title("A", loc="left", fontweight="bold")
    axis.grid(True, alpha=0.18)
    axis.legend(frameon=False, loc="upper left")

    axis = axes[1]
    x_values = np.asarray([row["logit_auc"] for row in prompt_rows])
    y_values = np.asarray([row["j_auc_mean_seed"] for row in prompt_rows])
    limit = max(float(np.max(x_values)), float(np.max(y_values)), 0.02) * 1.08
    axis.scatter(x_values, y_values, color="#A7ADB2", alpha=0.60, s=34, edgecolor="none")
    axis.plot([0, limit], [0, limit], linestyle="--", color=neutral, linewidth=1)
    axis.set_xlim(-0.006, limit)
    axis.set_ylim(-0.006, limit)
    axis.set_xlabel("Direct-unembedding prompt AUC")
    axis.set_ylabel("Jacobian-lens prompt AUC (3-fit mean)")
    axis.set_title("B", loc="left", fontweight="bold")
    axis.grid(True, alpha=0.18)

    axis = axes[2]
    families = sorted(
        controlled["families"],
        key=lambda family: controlled["families"][family]["delta_auc_mean"],
    )
    positions = np.arange(len(families))
    for position, family in zip(positions, families, strict=True):
        raw = controlled["families"][family]["prompt_deltas"]
        mean_value = controlled["families"][family]["delta_auc_mean"]
        color = jacobian_color if mean_value > 0 else negative_color if mean_value < 0 else neutral
        axis.scatter(raw, np.full(len(raw), position), color="#A7ADB2", alpha=0.45, s=25)
        axis.plot([0, mean_value], [position, position], color=color, linewidth=2.2, alpha=0.75)
        axis.scatter(mean_value, position, marker="D", color=color, s=52, zorder=3)
        axis.text(mean_value + (0.008 if mean_value >= 0 else -0.008), position,
                  f"{mean_value:+.3f}", va="center",
                  ha="left" if mean_value >= 0 else "right", fontsize=8, color=color)
    axis.axvline(0, color=neutral, linewidth=1)
    axis.set_yticks(positions, [DISPLAY_FAMILY[family] for family in families])
    axis.set_xlabel("Jacobian minus direct-unembedding AUC")
    axis.set_title("C", loc="left", fontweight="bold")
    axis.grid(True, axis="x", alpha=0.18)
    axis.set_xlim(-0.32, 0.36)

    for suffix in ("png", "pdf", "svg"):
        figure.savefig(CONTROLLED_FIGURE.with_suffix(f".{suffix}"), dpi=240,
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_reproducibility(concept_rows: list[dict]) -> None:
    """Supplementary version of the former main-text Figure 2D."""
    labels = ["J seed 0", "J seed 1", "J seed 2", "Direct"]
    arrays = [
        np.asarray([row[f"j_log_rank_seed{seed}"] for row in concept_rows])
        for seed in range(3)
    ] + [np.asarray([row["logit_log_rank"] for row in concept_rows])]
    matrix = np.asarray([
        [spearman_value(first, second) for second in arrays]
        for first in arrays
    ])
    figure, axis = plt.subplots(figsize=(6.8, 5.5), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="GnBu")
    axis.set_xticks(range(4), labels, rotation=25, ha="right")
    axis.set_yticks(range(4), labels)
    for row in range(4):
        for column in range(4):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.3f}", ha="center", va="center",
                      color="white" if value < 0.72 else "black")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman rank correlation")

    for suffix in ("png", "pdf", "svg"):
        figure.savefig(REPRO_FIGURE.with_suffix(f".{suffix}"), dpi=240,
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_discovery(discovery: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(2, 1, figsize=(14.4, 10.8), constrained_layout=True)
    method_labels = {
        "jacobian": "A  Jacobian lens",
        "logit": "B  Direct unembedding",
    }
    families = sorted(discovery["methods"]["jacobian"]["families"])
    maximum = 0.0
    matrices = {}
    for method in ("jacobian", "logit"):
        matrix = np.asarray([
            [
                discovery["methods"][method]["families"][family]["candidates"][column]["family_specificity_score"]
                if column < len(discovery["methods"][method]["families"][family]["candidates"])
                else 0.0
                for column in range(5)
            ]
            for family in families
        ])
        matrices[method] = np.sqrt(matrix)
        maximum = max(maximum, float(np.max(matrices[method])))

    images = []
    for axis, method in zip(axes, ("jacobian", "logit"), strict=True):
        matrix = matrices[method]
        image = axis.imshow(matrix, cmap="GnBu", aspect="auto", vmin=0, vmax=maximum)
        images.append(image)
        axis.set_xticks(range(5), [f"candidate {index}" for index in range(1, 6)])
        axis.set_yticks(range(len(families)), [DISPLAY_FAMILY[family] for family in families])
        axis.set_title(method_labels[method], loc="left", fontweight="bold")
        axis.tick_params(length=0)
        for row_index, family in enumerate(families):
            candidates = discovery["methods"][method]["families"][family]["candidates"]
            for column in range(5):
                if column >= len(candidates):
                    continue
                candidate = candidates[column]
                marker = "*" if candidate["exact_predeclared_overlap"] else ""
                label = f"{candidate['token']}{marker}\n{candidate['prompt_support']}/5"
                color = "white" if matrix[row_index, column] > 0.64 * maximum else "#17222A"
                axis.text(column, row_index, label, ha="center", va="center",
                          fontsize=8.4, color=color)
    colorbar = figure.colorbar(images[0], ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label("sqrt(background-corrected consensus score)")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(DISCOVERY_FIGURE.with_suffix(f".{suffix}"), dpi=240,
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)


def fmt_ci(result: dict) -> str:
    return (
        f"{result['rho']:.3f} "
        f"({result['family_clustered_low']:.3f} to "
        f"{result['family_clustered_high']:.3f})"
    )


def write_report(stats: dict) -> None:
    controlled = stats["controlled"]
    discovery = stats["open_vocabulary"]
    overall = controlled["overall"]
    interval = overall["hierarchical_family_bootstrap"]
    sign_flip = overall["exact_family_sign_flip"]
    correlations = controlled["seed_rank_correlations"]
    lines = [
        "# Materials held-out v1 analysis",
        "",
        "## What was tested",
        "",
        "Fifty descriptions were frozen before execution: ten materials-mechanism families with five unseen descriptions per family. Each description omitted the physical terms that were scored. The same Gemma-4-E4B-it checkpoint was evaluated through three Jacobian lenses fitted independently on three 1,000-record WikiText-103 samples. Direct unembedding used the same model, prompts, positions, layers, and vocabulary.",
        "",
        "The study has two deliberately separate parts. The controlled test asks whether physical terms declared before execution become highly ranked. The discovery test asks what words assemble without supplying those terms, and creates blinded candidate sets for mechanism identification.",
        "",
        "## Controlled result",
        "",
        f"Mean held-out Jacobian recovery AUC was **{overall['j_auc_mean']:.4f}**, versus **{overall['logit_auc_mean']:.4f}** for direct unembedding. The absolute advantage was **{overall['delta_auc']:+.4f}** ({100 * overall['relative_auc_gain']:.1f}% relative). The family-hierarchical 95% interval was {interval['low']:+.4f} to {interval['high']:+.4f}. The exact family sign-flip p-values were {sign_flip['p_one_sided_j_greater']:.4f} one-sided and {sign_flip['p_two_sided']:.4f} two-sided.",
        "",
        f"At the prompt level, the seed-averaged Jacobian lens won {overall['j_wins']} comparisons, tied {overall['ties']}, and lost {overall['j_losses']}. These counts are descriptive because prompts within the same physical family are related.",
        "",
        "![Held-out controlled recovery](../figures/materials-heldout-v1/heldout-controlled-recovery.png)",
        "",
        "### Reproducibility across independently fitted lenses",
        "",
        f"Across {overall['n_resolved_prompt_concepts']} prompt-concept pairs, the full-vocabulary rank correlations were:",
        "",
        f"- seed 0 versus 1: rho = {fmt_ci(correlations['seed0_vs_seed1'])};",
        f"- seed 0 versus 2: rho = {fmt_ci(correlations['seed0_vs_seed2'])};",
        f"- seed 1 versus 2: rho = {fmt_ci(correlations['seed1_vs_seed2'])}.",
        "",
        "These intervals resample physical families. High agreement means the measurement is stable to the WikiText sample used to fit the lens; it does not by itself establish causal use or human-like understanding.",
        "",
        "### Family-level effects",
        "",
        "| physical family | mean Jacobian AUC | direct AUC | difference |",
        "|---|---:|---:|---:|",
    ]
    for family, result in sorted(controlled["families"].items()):
        lines.append(
            f"| {family} | {result['j_auc_mean']:.4f} | "
            f"{result['logit_auc_mean']:.4f} | {result['delta_auc_mean']:+.4f} |"
        )
    lines.extend([
        "",
        "## Target-free discovery result",
        "",
        "For each method, the code scanned top-ranked vocabulary tokens across every prompt position and registered source layer, removed tokens present in the prompt or one-token continuation, retained 64 candidates per run, required agreement across all three stored seed lists, removed a frozen list of ordinary English function words, and downweighted words that appeared across many prompts. The predeclared scientific terms were not used at any of these stages.",
        "",
        "![Held-out open vocabulary](../figures/materials-heldout-v1/heldout-open-vocabulary.png)",
        "",
        "The table below is descriptive overlap added only after discovery ranking. It is not the blinded semantic endpoint.",
        "",
        "| method | exact target in top 1 | top 3 | top 5 | top 8 |",
        "|---|---:|---:|---:|---:|",
    ])
    for method, result in discovery["methods"].items():
        overlap = result["exact_predeclared_overlap_at_k"]
        lines.append(
            f"| {method} | {overlap['1']['families_with_exact_overlap']}/10 | "
            f"{overlap['3']['families_with_exact_overlap']}/10 | "
            f"{overlap['5']['families_with_exact_overlap']}/10 | "
            f"{overlap['8']['families_with_exact_overlap']}/10 |"
        )
    lines.extend([
        "",
        "### Family candidate sets",
        "",
        "Asterisks mark exact predeclared-term overlap after ranking. Parentheses give support across the five unseen phrasings.",
        "",
        "| method | family | top target-free candidates |",
        "|---|---|---|",
    ])
    for method, method_result in discovery["methods"].items():
        for family, result in sorted(method_result["families"].items()):
            cells = []
            for candidate in result["candidates"]:
                marker = "*" if candidate["exact_predeclared_overlap"] else ""
                cells.append(
                    f"`{candidate['token']}`{marker} ({candidate['prompt_support']}/5)"
                )
            lines.append(f"| {method} | {family} | {', '.join(cells)} |")
    lines.extend([
        "",
        "## What remains before a primary discovery claim",
        "",
        "The 20 shuffled sets in `MATERIALS_HELDOUT_V1_BLINDED_SETS.csv` must be classified among the ten frozen family labels by at least three materials-science raters. The answer key is stored separately. An automated language-model rater can be reported only as a reproducible secondary analysis.",
        "",
        "The primary controlled-versus-discovery correlation remains pending the human blinded-identification outcome. A separately labeled automated sensitivity analysis may use the same frozen sets, but must not replace the human endpoint or a correlation chosen after inspecting candidate words.",
        "",
        "## Interpretation boundary",
        "",
        "These results show which scientific words are linearly readable after a learned Jacobian transport and which words arise under target-free decoding. They do not expose private prose, prove a literal chain of thought, or demonstrate causal reasoning. Seed agreement is a stability control. A causal claim would require interventions that change model behavior in the predicted direction.",
        "",
        "## Reproducible artifacts",
        "",
        "- Frozen design: `materials-heldout-v1-preregistration.md`",
        "- Preflight: `materials-heldout-v1-preflight.json`",
        "- Machine-readable statistics: `materials-heldout-v1_statistics.json`",
        "- Prompt-level controlled results: `materials-heldout-v1_prompt_results.csv`",
        "- Concept-level ranks: `materials-heldout-v1_concept_results.csv`",
        "- All target-free prompt candidates: `materials-heldout-v1_open_vocabulary_candidates.csv`",
        "- Blinded sets: `MATERIALS_HELDOUT_V1_BLINDED_SETS.csv`",
        "- Separate key: `MATERIALS_HELDOUT_V1_BLINDED_KEY.csv`",
        "- Reproduce: `python scripts/analyze_materials_heldout_v1.py`",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    run_paths = RUN_PATHS
    runs = [json.loads(path.read_text()) for path in run_paths]
    slugs, indexes = validate_runs(runs, run_paths)
    prompt_rows, concept_rows, ks = build_controlled_rows(runs, slugs, indexes)
    controlled = build_controlled_statistics(runs, prompt_rows, concept_rows, ks)
    discovery = build_discovery_statistics(slugs, indexes)
    stats = {
        "analysis_status": "preregistered held-out analysis",
        "analysis_seed": RNG_SEED,
        "bootstrap_resamples": N_BOOT,
        "correlation_bootstrap_resamples": N_CORR_BOOT,
        "runs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in run_paths
        ],
        "controlled": controlled,
        "open_vocabulary": discovery,
        "pending_primary_endpoints": [
            "three-or-more blinded materials-science raters",
            "controlled-recovery versus blinded-discovery correlation",
        ],
    }
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(PROMPT_CSV, prompt_rows)
    write_csv(CONCEPT_CSV, concept_rows)
    write_candidate_csv(discovery)
    write_blinded_sets(discovery)
    stats["open_vocabulary"]["blinded_artifacts"] = {
        "sets": {
            "path": str(BLIND_CSV.relative_to(ROOT)),
            "sha256": sha256(BLIND_CSV),
        },
        "separate_key": {
            "path": str(KEY_CSV.relative_to(ROOT)),
            "sha256": sha256(KEY_CSV),
        },
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    plot_controlled(controlled, prompt_rows, concept_rows)
    plot_reproducibility(concept_rows)
    plot_discovery(discovery)
    write_report(stats)
    print(json.dumps({
        "overall": controlled["overall"],
        "seed_rank_correlations": controlled["seed_rank_correlations"],
        "open_vocabulary_overlap": {
            method: result["exact_predeclared_overlap_at_k"]
            for method, result in discovery["methods"].items()
        },
    }, indent=2))
    for path in (
        STATS_PATH, PROMPT_CSV, CONCEPT_CSV, CANDIDATE_CSV,
        BLIND_CSV, KEY_CSV, REPORT_PATH,
        CONTROLLED_FIGURE.with_suffix(".pdf"),
        REPRO_FIGURE.with_suffix(".pdf"),
        DISCOVERY_FIGURE.with_suffix(".pdf"),
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
