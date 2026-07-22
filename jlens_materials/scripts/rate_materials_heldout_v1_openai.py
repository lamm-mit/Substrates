#!/usr/bin/env python3
"""Run a reproducible, secondary OpenAI rater on frozen blinded word sets.

This does not replace the preregistered materials-expert ratings. The model sees
only shuffled candidate words, support counts, and the ten allowed labels. The
answer key is loaded only after all raw blinded responses have been saved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402


BLIND_CSV = ROOT / "experiments" / "MATERIALS_HELDOUT_V1_BLINDED_SETS.csv"
KEY_CSV = ROOT / "experiments" / "MATERIALS_HELDOUT_V1_BLINDED_KEY.csv"
STATS_PATH = ROOT / "experiments" / "materials-heldout-v1_statistics.json"
RAW_PATH = ROOT / "experiments" / "materials-heldout-v1_automated_blinded_raw.json"
RESULTS_PATH = ROOT / "experiments" / "materials-heldout-v1_automated_blinded_results.json"
RATINGS_CSV = ROOT / "experiments" / "materials-heldout-v1_automated_blinded_ratings.csv"
REPORT_PATH = ROOT / "experiments" / "MATERIALS_HELDOUT_V1_AUTOMATED_RATER.md"
INTEGRATED_REPORT_PATH = ROOT / "experiments" / "MATERIALS_HELDOUT_V1_ANALYSIS.md"
FIGURE_PATH = ROOT / "figures" / "materials-heldout-v1" / "automated-family-identification"
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
ANALYSIS_SEED = 20260714
N_NULL = 100_000
N_BOOT = 20_000


LABELS = [
    "boundary-attack",
    "cleavage",
    "cyclic",
    "ductile",
    "high-temperature-deformation",
    "hot-air-surface-layer",
    "line-defect-motion",
    "notch-resistance",
    "particle-strengthening",
    "rapid-transformation",
]

SYSTEM = """\
You are serving as a blinded materials-science classifier. You will receive
sets of vocabulary words decoded from an unknown representation. Infer which
one of ten physical mechanism families best matches each set. Use only the
candidate words, their support counts, and the supplied label names. You do not
know the decoding method, the original prompt, or the answer key.

Classify every set independently. Labels may be reused any number of times; do
not impose a one-to-one assignment across sets. Weak or noisy sets still require
the single best label. Return only the requested JSON array, with no Markdown.
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def format_user_prompt(
    rows: list[dict[str, str]], labels: list[str]
) -> str:
    lines = [
        "Allowed labels (choose exactly one for each set):",
        *[f"- {label}" for label in labels],
        "",
        "Blinded candidate sets:",
    ]
    for row in rows:
        candidates = []
        for index in range(1, 9):
            token = row.get(f"candidate_{index}", "").strip()
            support = row.get(f"support_{index}", "").strip()
            if token:
                candidates.append(f"{token} [{support} prompts]")
        lines.append(f"{row['set_id']}: " + "; ".join(candidates))
    lines.extend([
        "",
        "Return a JSON array with exactly one object per set, using this schema:",
        '[{"set_id":"SET-01","predicted_family":"one-allowed-label",'
        '"confidence":0.0,"rationale":"at most 20 words"}]',
        "Confidence must be between 0 and 1. Keep each rationale under 20 words.",
    ])
    return "\n".join(lines)


def parse_response(text: str, expected_set_ids: set[str]) -> list[dict]:
    stripped = text.strip()
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON array")
    data = json.loads(stripped[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("response JSON is not a list")
    parsed = []
    seen = set()
    for row in data:
        set_id = str(row.get("set_id", ""))
        label = str(row.get("predicted_family", ""))
        if set_id not in expected_set_ids or set_id in seen:
            raise ValueError(f"invalid or duplicate set id {set_id!r}")
        if label not in LABELS:
            raise ValueError(f"invalid family label {label!r}")
        confidence = float(row.get("confidence", 0.0))
        if not 0 <= confidence <= 1:
            raise ValueError(f"invalid confidence for {set_id}")
        parsed.append({
            "set_id": set_id,
            "predicted_family": label,
            "confidence": confidence,
            "rationale": str(row.get("rationale", "")),
        })
        seen.add(set_id)
    if seen != expected_set_ids:
        missing = sorted(expected_set_ids - seen)
        raise ValueError(f"response omitted set ids: {missing}")
    return parsed


def run_blinded_passes(model: str, passes: int, *, force: bool) -> dict:
    blind_hash = sha256(BLIND_CSV)
    if RAW_PATH.is_file() and not force:
        existing = json.loads(RAW_PATH.read_text())
        if (
            existing.get("blinded_sets_sha256") == blind_hash
            and existing.get("model") == model
            and len(existing.get("passes", [])) == passes
            and all(item.get("parsed_ratings") for item in existing["passes"])
        ):
            print(f"reusing {passes} saved blinded API passes from {RAW_PATH}")
            return existing

    blind_rows = read_csv(BLIND_CSV)
    expected_set_ids = {row["set_id"] for row in blind_rows}
    raw = {
        "analysis_status": "secondary automated blinded rater; not human expert data",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "system_prompt": SYSTEM,
        "blinded_sets_path": str(BLIND_CSV.relative_to(ROOT)),
        "blinded_sets_sha256": blind_hash,
        "answer_key_not_loaded_during_calls": True,
        "passes": [],
        "failed_attempts": [],
    }
    for pass_index in range(passes):
        rng = random.Random(ANALYSIS_SEED + pass_index)
        row_order = list(blind_rows)
        label_order = list(LABELS)
        rng.shuffle(row_order)
        rng.shuffle(label_order)
        user_prompt = format_user_prompt(row_order, label_order)
        parsed = None
        response = ""
        for attempt_index, effort in enumerate(("low", "minimal"), start=1):
            response = llm.complete(
                "openai",
                model,
                SYSTEM,
                [{"type": "text", "text": user_prompt}],
                max_tokens=8000,
                effort=effort,
            )
            try:
                parsed = parse_response(response, expected_set_ids)
            except (ValueError, json.JSONDecodeError) as exc:
                raw["failed_attempts"].append({
                    "pass_index": pass_index,
                    "attempt_index": attempt_index,
                    "reasoning_effort": effort,
                    "user_prompt": user_prompt,
                    "raw_response": response,
                    "parse_error": f"{type(exc).__name__}: {exc}",
                })
                RAW_PATH.write_text(json.dumps(raw, indent=2) + "\n")
                if attempt_index == 2:
                    raise
                continue
            break
        if parsed is None:
            raise RuntimeError(f"pass {pass_index} produced no valid rating")
        raw["passes"].append({
            "pass_index": pass_index,
            "randomization_seed": ANALYSIS_SEED + pass_index,
            "set_order": [row["set_id"] for row in row_order],
            "label_order": label_order,
            "successful_reasoning_effort": effort,
            "user_prompt": user_prompt,
            "raw_response": response,
            "parsed_ratings": parsed,
        })
        RAW_PATH.write_text(json.dumps(raw, indent=2) + "\n")
        print(f"saved blinded pass {pass_index + 1}/{passes}")
    return raw


def majority_prediction(ratings: list[dict]) -> tuple[str, dict, str]:
    counts = Counter(row["predicted_family"] for row in ratings)
    maximum = max(counts.values())
    tied = [label for label, count in counts.items() if count == maximum]
    tie_break = "vote count"
    if len(tied) > 1:
        mean_confidence = {
            label: float(np.mean([
                row["confidence"] for row in ratings
                if row["predicted_family"] == label
            ]))
            for label in tied
        }
        best_confidence = max(mean_confidence.values())
        tied = [
            label for label in tied
            if abs(mean_confidence[label] - best_confidence) <= 1e-12
        ]
        tie_break = "mean confidence among vote-count ties"
    if len(tied) > 1:
        tied.sort()
        tie_break = "lexicographic after vote-count and confidence tie"
    return tied[0], dict(sorted(counts.items())), tie_break


def fleiss_kappa(rows: list[dict], passes: int) -> float:
    by_set: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_set[row["set_id"]][row["predicted_family"]] += 1
    matrix = np.asarray([
        [by_set[set_id][label] for label in LABELS]
        for set_id in sorted(by_set)
    ], dtype=float)
    item_agreement = (
        np.sum(matrix ** 2, axis=1) - passes
    ) / (passes * (passes - 1))
    observed = float(np.mean(item_agreement))
    category_frequency = np.sum(matrix, axis=0) / (matrix.shape[0] * passes)
    expected = float(np.sum(category_frequency ** 2))
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def shuffled_null(
    predictions: list[str], truths: list[str], *, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    predictions_array = np.asarray(predictions)
    truths_array = np.asarray(truths)
    observed = float(np.mean(predictions_array == truths_array))
    estimates = np.empty(N_NULL, dtype=float)
    for index in range(N_NULL):
        estimates[index] = float(np.mean(
            predictions_array == truths_array[rng.permutation(len(truths_array))]
        ))
    return {
        "observed_accuracy": observed,
        "null_mean": float(np.mean(estimates)),
        "null_95_low": float(np.quantile(estimates, 0.025)),
        "null_95_high": float(np.quantile(estimates, 0.975)),
        "p_greater_or_equal": float(
            (np.sum(estimates >= observed - 1e-15) + 1) / (N_NULL + 1)
        ),
        "n_shuffles": N_NULL,
    }


def exact_paired_sign_flip(family_differences: list[float]) -> dict:
    values = np.asarray(family_differences, dtype=float)
    observed = float(np.mean(values))
    null = np.asarray([
        float(np.mean(values * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    return {
        "observed_jacobian_minus_logit_accuracy": observed,
        "p_one_sided_j_greater": float(np.mean(null >= observed - 1e-15)),
        "p_two_sided": float(np.mean(np.abs(null) >= abs(observed) - 1e-15)),
        "n_permutations": int(len(null)),
    }


def bootstrap_spearman(x_values: list[float], y_values: list[float], *, seed: int) -> dict:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    observed = float(spearmanr(x, y).statistic) if len(np.unique(y)) > 1 else float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(N_BOOT):
        selected = rng.integers(0, len(x), size=len(x))
        sample_x = x[selected]
        sample_y = y[selected]
        if len(np.unique(sample_x)) < 2 or len(np.unique(sample_y)) < 2:
            continue
        estimate = float(spearmanr(sample_x, sample_y).statistic)
        if math.isfinite(estimate):
            estimates.append(estimate)
    return {
        "rho": observed,
        "bootstrap_low": float(np.quantile(estimates, 0.025)) if estimates else None,
        "bootstrap_high": float(np.quantile(estimates, 0.975)) if estimates else None,
        "n_families": len(x),
        "n_valid_resamples": len(estimates),
    }


def analyze(raw: dict) -> tuple[dict, list[dict]]:
    # The answer key is intentionally opened only after all API responses exist.
    key_rows = read_csv(KEY_CSV)
    key = {row["set_id"]: row for row in key_rows}
    ratings = []
    for pass_result in raw["passes"]:
        for rating in pass_result["parsed_ratings"]:
            truth = key[rating["set_id"]]
            ratings.append({
                "pass_index": pass_result["pass_index"],
                **rating,
                "method": truth["method"],
                "true_family": truth["mechanism_family"],
                "correct": rating["predicted_family"] == truth["mechanism_family"],
            })

    by_set: dict[str, list[dict]] = defaultdict(list)
    for row in ratings:
        by_set[row["set_id"]].append(row)
    majority_rows = []
    for set_id in sorted(by_set):
        prediction, vote_counts, tie_break = majority_prediction(by_set[set_id])
        truth = key[set_id]
        correct_votes = sum(row["correct"] for row in by_set[set_id])
        majority_rows.append({
            "set_id": set_id,
            "method": truth["method"],
            "true_family": truth["mechanism_family"],
            "majority_prediction": prediction,
            "majority_correct": prediction == truth["mechanism_family"],
            "vote_fraction_correct": correct_votes / len(by_set[set_id]),
            "vote_counts": vote_counts,
            "tie_break": tie_break,
        })

    per_pass = []
    for pass_index in range(len(raw["passes"])):
        pass_rows = [row for row in ratings if row["pass_index"] == pass_index]
        per_pass.append({
            "pass_index": pass_index,
            "overall_accuracy": float(np.mean([row["correct"] for row in pass_rows])),
            "jacobian_accuracy": float(np.mean([
                row["correct"] for row in pass_rows if row["method"] == "jacobian"
            ])),
            "logit_accuracy": float(np.mean([
                row["correct"] for row in pass_rows if row["method"] == "logit"
            ])),
        })

    method_metrics = {}
    for method in ("jacobian", "logit"):
        method_majority = [row for row in majority_rows if row["method"] == method]
        predictions = [row["majority_prediction"] for row in method_majority]
        truths = [row["true_family"] for row in method_majority]
        method_ratings = [row for row in ratings if row["method"] == method]
        method_metrics[method] = {
            "majority_accuracy": float(np.mean([
                row["majority_correct"] for row in method_majority
            ])),
            "correct_families": int(sum(row["majority_correct"] for row in method_majority)),
            "n_families": len(method_majority),
            "mean_individual_pass_accuracy": float(np.mean([
                result[f"{method}_accuracy"] for result in per_pass
            ])),
            "fleiss_kappa": fleiss_kappa(method_ratings, len(raw["passes"])),
            "shuffled_label_null": shuffled_null(
                predictions, truths, seed=ANALYSIS_SEED + (1 if method == "jacobian" else 2)
            ),
        }

    family_rows = {}
    for family in LABELS:
        family_rows[family] = {}
        for method in ("jacobian", "logit"):
            family_rows[family][method] = next(
                row for row in majority_rows
                if row["method"] == method and row["true_family"] == family
            )
    family_differences = [
        float(family_rows[family]["jacobian"]["majority_correct"])
        - float(family_rows[family]["logit"]["majority_correct"])
        for family in LABELS
    ]

    study_stats = json.loads(STATS_PATH.read_text())
    controlled_families = study_stats["controlled"]["families"]
    h4 = {}
    for method in ("jacobian", "logit"):
        recovery_key = "j_auc_mean" if method == "jacobian" else "logit_auc_mean"
        recovery = [controlled_families[family][recovery_key] for family in LABELS]
        majority_success = [
            float(family_rows[family][method]["majority_correct"])
            for family in LABELS
        ]
        vote_fraction = [
            float(family_rows[family][method]["vote_fraction_correct"])
            for family in LABELS
        ]
        h4[method] = {
            "majority_success": bootstrap_spearman(
                recovery, majority_success,
                seed=ANALYSIS_SEED + (10 if method == "jacobian" else 11),
            ),
            "vote_fraction_sensitivity": bootstrap_spearman(
                recovery, vote_fraction,
                seed=ANALYSIS_SEED + (20 if method == "jacobian" else 21),
            ),
            "family_values": [
                {
                    "family": family,
                    "controlled_recovery_auc": recovery[index],
                    "majority_success": majority_success[index],
                    "vote_fraction_correct": vote_fraction[index],
                }
                for index, family in enumerate(LABELS)
            ],
        }

    results = {
        "analysis_status": (
            "secondary automated blinded analysis; does not satisfy the "
            "preregistered human-rater endpoint"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": raw["provider"],
        "model": raw["model"],
        "n_order_randomized_passes": len(raw["passes"]),
        "blinded_sets_sha256": raw["blinded_sets_sha256"],
        "answer_key_sha256": sha256(KEY_CSV),
        "per_pass": per_pass,
        "overall_fleiss_kappa": fleiss_kappa(ratings, len(raw["passes"])),
        "methods": method_metrics,
        "paired_family_comparison": exact_paired_sign_flip(family_differences),
        "majority_by_set": majority_rows,
        "controlled_discovery_relationship_h4_secondary": h4,
        "limitations": [
            "five passes of one model are not five independent human raters",
            "automated semantic judgments are a secondary analysis",
            "human materials-science ratings remain required for the primary endpoint",
        ],
    }
    return results, ratings


def write_ratings_csv(rows: list[dict]) -> None:
    with RATINGS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(results: dict) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    jacobian_color = "#087F8C"
    logit_color = "#66727A"
    neutral = "#70757A"
    figure = plt.figure(figsize=(15.2, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=[0.43, 1.0])
    protocol_axis = figure.add_subplot(grid[0, :])
    axes = [figure.add_subplot(grid[1, index]) for index in range(3)]

    protocol_axis.set_xlim(0, 1)
    protocol_axis.set_ylim(0, 1)
    protocol_axis.axis("off")
    protocol_boxes = [
        (0.035, "20 candidate-word sets\n10 families x 2 readouts"),
        (0.285, "Shuffle sets and hide\nprompt, method, and answer key"),
        (0.535, "Choose 1 of 10 family labels\nfrom the candidate words alone"),
        (0.785, "Repeat 5 orderings\nthen take majority vote"),
    ]
    box_width = 0.18
    box_y = 0.36
    box_height = 0.43
    for x_value, label in protocol_boxes:
        protocol_axis.add_patch(
            FancyBboxPatch(
                (x_value, box_y), box_width, box_height,
                boxstyle="round,pad=0.012,rounding_size=0.015",
                facecolor="#f4f7f8", edgecolor="#7c8b92", linewidth=1.1,
            )
        )
        protocol_axis.text(
            x_value + box_width / 2, box_y + box_height / 2, label,
            ha="center", va="center", fontsize=9.2,
        )
    for first, second in zip(protocol_boxes[:-1], protocol_boxes[1:], strict=True):
        protocol_axis.add_patch(
            FancyArrowPatch(
                (first[0] + box_width + 0.008, box_y + box_height / 2),
                (second[0] - 0.008, box_y + box_height / 2),
                arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#626c72",
            )
        )
    protocol_axis.text(
        0.5, 0.12,
        "This tests whether a word set carries recognizable materials meaning; it does not ask the rater to solve the original engineering prompt.",
        ha="center", va="center", fontsize=8.8, color="#596168",
    )

    axis = axes[0]
    pass_indices = np.arange(len(results["per_pass"]))
    width = 0.34
    axis.bar(pass_indices - width / 2,
             [row["jacobian_accuracy"] for row in results["per_pass"]],
             width=width, color=jacobian_color, label="Jacobian")
    axis.bar(pass_indices + width / 2,
             [row["logit_accuracy"] for row in results["per_pass"]],
             width=width, color=logit_color, label="Direct")
    axis.axhline(0.1, color=neutral, linestyle="--", linewidth=1, label="10-label chance")
    axis.set_xticks(pass_indices, [str(index + 1) for index in pass_indices])
    axis.set_ylim(0, 1.03)
    axis.set_xlabel("Order-randomized automated pass")
    axis.set_ylabel("Classification accuracy")
    axis.legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.64, 1.015),
        ncol=3, fontsize=8,
    )
    axis.grid(True, axis="y", alpha=0.18)

    axis = axes[1]
    rows = {row["true_family"]: {} for row in results["majority_by_set"]}
    for row in results["majority_by_set"]:
        rows[row["true_family"]][row["method"]] = row
    correctness = np.asarray([
        [
            float(rows[family]["jacobian"]["majority_correct"]),
            float(rows[family]["logit"]["majority_correct"]),
        ]
        for family in LABELS
    ])
    axis.imshow(
        correctness,
        vmin=0,
        vmax=1,
        cmap=ListedColormap(["#e5e8ea", "#4a9096"]),
        aspect="auto",
    )
    axis.set_yticks(np.arange(len(LABELS)), [DISPLAY_FAMILY[label] for label in LABELS])
    axis.set_xticks([0, 1], ["Jacobian", "Direct"])
    for row_index in range(len(LABELS)):
        for col_index in range(2):
            correct = bool(correctness[row_index, col_index])
            axis.text(
                col_index, row_index, "correct" if correct else "miss",
                ha="center", va="center", fontsize=8.2,
                color="white" if correct else "#30383d",
            )
    axis.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis = axes[2]
    h4 = results["controlled_discovery_relationship_h4_secondary"]["jacobian"]
    for row in h4["family_values"]:
        axis.scatter(row["controlled_recovery_auc"], row["vote_fraction_correct"],
                     color=jacobian_color, s=45, alpha=0.82)
        if row["vote_fraction_correct"] < 1 or row["family"] == "boundary-attack":
            offset = (5, -10) if row["family"] == "boundary-attack" else (5, 3)
            axis.annotate(DISPLAY_FAMILY[row["family"]],
                          (row["controlled_recovery_auc"], row["vote_fraction_correct"]),
                          xytext=offset, textcoords="offset points", fontsize=8)
    sensitivity = h4["vote_fraction_sensitivity"]
    axis.set_xlabel("Controlled Jacobian recovery AUC")
    axis.set_ylabel("Automated vote fraction correct")
    axis.set_ylim(-0.04, 1.04)
    axis.text(0.98, 0.06, f"Spearman rho = {sensitivity['rho']:.2f}",
              transform=axis.transAxes, ha="right", va="bottom", color=neutral)
    axis.grid(True, alpha=0.18)

    protocol_axis.text(
        0.0,
        1.02,
        "A",
        transform=protocol_axis.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    for panel_axis, label in zip(axes, "BCD", strict=True):
        panel_axis.text(
            0.0,
            1.06,
            label,
            transform=panel_axis.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )

    for suffix in ("png", "pdf", "svg"):
        figure.savefig(FIGURE_PATH.with_suffix(f".{suffix}"), dpi=240,
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_report(results: dict) -> None:
    jacobian = results["methods"]["jacobian"]
    logit = results["methods"]["logit"]
    paired = results["paired_family_comparison"]
    h4 = results["controlled_discovery_relationship_h4_secondary"]["jacobian"]
    lines = [
        "# Secondary automated blinded family identification",
        "",
        "## Status",
        "",
        "This is a reproducible automated secondary analysis. It does **not** satisfy the preregistered requirement for at least three materials-science raters. Five order-randomized passes of one model are repeated judgments from one automated system, not five independent experts.",
        "",
        "## What the model saw",
        "",
        f"The rater was `{results['model']}` through the OpenAI Responses API. It saw only the 20 shuffled sets of eight candidate words, each word's support across five phrasings, and the ten allowed mechanism-family names. It did not receive the original prompts, the decoding method, or the answer key. The exact system prompt, five exact user prompts, raw responses, and parsed responses are retained in `materials-heldout-v1_automated_blinded_raw.json`.",
        "",
        "Each set was classified independently; labels could be reused. Majority vote was computed across the five order-randomized passes. Vote-count ties were resolved first by mean confidence and then lexicographically, as encoded before scoring.",
        "",
        "![Automated blinded rating](../figures/materials-heldout-v1/automated-family-identification.png)",
        "",
        "## Results",
        "",
        f"Jacobian candidate sets were identified correctly for **{jacobian['correct_families']}/{jacobian['n_families']} families ({jacobian['majority_accuracy']:.0%})**. Direct-unembedding sets were correct for **{logit['correct_families']}/{logit['n_families']} ({logit['majority_accuracy']:.0%})**.",
        "",
        f"Across paired physical families, the Jacobian-minus-direct accuracy difference was {paired['observed_jacobian_minus_logit_accuracy']:+.2f}; the exact sign-flip p-values were {paired['p_one_sided_j_greater']:.4f} one-sided and {paired['p_two_sided']:.4f} two-sided.",
        "",
        f"Agreement across the five automated passes was Fleiss kappa = {results['overall_fleiss_kappa']:.3f}. The shuffled-label p-values were {jacobian['shuffled_label_null']['p_greater_or_equal']:.4f} for Jacobian sets and {logit['shuffled_label_null']['p_greater_or_equal']:.4f} for direct sets.",
        "",
        "| pass | Jacobian accuracy | direct accuracy | overall |",
        "|---:|---:|---:|---:|",
    ]
    for row in results["per_pass"]:
        lines.append(
            f"| {row['pass_index'] + 1} | {row['jacobian_accuracy']:.0%} | "
            f"{row['logit_accuracy']:.0%} | {row['overall_accuracy']:.0%} |"
        )
    lines.extend([
        "",
        "## Controlled-versus-discovered relationship",
        "",
        f"For the Jacobian method, family-level controlled recovery and automated majority identification had Spearman rho = {h4['majority_success']['rho']:.3f}. Using the fraction of the five automated passes that were correct gave rho = {h4['vote_fraction_sensitivity']['rho']:.3f}, with a family-bootstrap interval of {h4['vote_fraction_sensitivity']['bootstrap_low']} to {h4['vote_fraction_sensitivity']['bootstrap_high']}.",
        "",
        "This small, ten-family secondary correlation is not a substitute for the preregistered human-rater H4 analysis. A weak relationship would mean that finding a preselected word and assembling a semantically identifiable word neighborhood capture different aspects of the representation.",
        "",
        "## Per-set majority outcomes",
        "",
        "| set | hidden method | true family | majority prediction | correct | correct votes |",
        "|---|---|---|---|---:|---:|",
    ])
    for row in results["majority_by_set"]:
        lines.append(
            f"| {row['set_id']} | {row['method']} | {row['true_family']} | "
            f"{row['majority_prediction']} | {'yes' if row['majority_correct'] else 'no'} | "
            f"{row['vote_fraction_correct']:.0%} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The automated comparison tests whether target-free decoded word neighborhoods are recognizable as materials mechanisms. It does not test whether the model internally wrote a hidden explanation, and it cannot establish causal reasoning. The human expert rating sheet remains the primary next step.",
        "",
        "## Artifacts",
        "",
        "- Raw blinded prompts and responses: `materials-heldout-v1_automated_blinded_raw.json`",
        "- Unblinded machine-readable results: `materials-heldout-v1_automated_blinded_results.json`",
        "- Every individual rating: `materials-heldout-v1_automated_blinded_ratings.csv`",
        "- Reproduce or reuse saved calls: `python scripts/rate_materials_heldout_v1_openai.py`",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines))


def update_integrated_report(results: dict) -> None:
    begin = "<!-- BEGIN AUTOMATED BLINDED SECONDARY -->"
    end = "<!-- END AUTOMATED BLINDED SECONDARY -->"
    jacobian = results["methods"]["jacobian"]
    logit = results["methods"]["logit"]
    paired = results["paired_family_comparison"]
    h4 = results["controlled_discovery_relationship_h4_secondary"]["jacobian"]
    section = "\n".join([
        begin,
        "",
        "## Completed automated blinded secondary analysis",
        "",
        f"Five order-randomized blinded passes of `{results['model']}` identified "
        f"{jacobian['correct_families']}/{jacobian['n_families']} Jacobian family sets "
        f"and {logit['correct_families']}/{logit['n_families']} direct-unembedding sets "
        "by majority vote. The paired family accuracy difference was "
        f"{paired['observed_jacobian_minus_logit_accuracy']:+.2f} "
        f"(exact p={paired['p_two_sided']:.3f}); repeat-pass Fleiss kappa was "
        f"{results['overall_fleiss_kappa']:.3f}. Both methods failed on cleavage.",
        "",
        "The candidate neighborhoods are therefore strongly materials-informative under this "
        "automated test, but not uniquely Jacobian. Controlled Jacobian recovery and automated "
        f"vote fraction were weakly related (rho={h4['vote_fraction_sensitivity']['rho']:.3f}, "
        f"family-bootstrap interval {h4['vote_fraction_sensitivity']['bootstrap_low']:.3f} to "
        f"{h4['vote_fraction_sensitivity']['bootstrap_high']:.3f}).",
        "",
        "This does not complete the primary human-rater endpoint: five passes of one automated "
        "model are not independent materials experts. See "
        "[`MATERIALS_HELDOUT_V1_AUTOMATED_RATER.md`](MATERIALS_HELDOUT_V1_AUTOMATED_RATER.md) "
        "for every result and the raw-response audit trail.",
        "",
        end,
        "",
    ])
    text = INTEGRATED_REPORT_PATH.read_text()
    if begin in text and end in text:
        prefix, remainder = text.split(begin, 1)
        _, suffix = remainder.split(end, 1)
        text = prefix.rstrip() + "\n\n" + section + suffix.lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + section
    INTEGRATED_REPORT_PATH.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=llm.default_model("openai"))
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.passes < 3:
        raise SystemExit("use at least three automated passes")
    raw = run_blinded_passes(args.model, args.passes, force=args.force)
    results, ratings = analyze(raw)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    write_ratings_csv(ratings)
    plot_results(results)
    write_report(results)
    update_integrated_report(results)

    study_stats = json.loads(STATS_PATH.read_text())
    study_stats["automated_blinded_secondary"] = results
    STATS_PATH.write_text(json.dumps(study_stats, indent=2) + "\n")
    print(json.dumps({
        "model": results["model"],
        "overall_fleiss_kappa": results["overall_fleiss_kappa"],
        "methods": results["methods"],
        "paired_family_comparison": results["paired_family_comparison"],
        "h4": results["controlled_discovery_relationship_h4_secondary"],
    }, indent=2))
    for path in (RAW_PATH, RESULTS_PATH, RATINGS_CSV, REPORT_PATH,
                 INTEGRATED_REPORT_PATH,
                 FIGURE_PATH.with_suffix(".pdf"), STATS_PATH):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
