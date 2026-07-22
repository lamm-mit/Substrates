#!/usr/bin/env python3
"""Validate the robustness experiment bundle.

The script is intentionally read-only except for its JSON validation report.
It checks protocol fingerprints, prompt completeness, scaffold exclusion,
array dimensions, table cardinalities, and frozen numerical endpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "experiments" / "review-robustness-audit-2026-07-18"
REPORT = AUDIT / "validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text())


def csv_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return abs(actual - expected) <= tolerance


def main() -> None:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, **details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **details})

    study_dirs = {
        "checkpoint": "experiments/option-free-relation-graph-2026-07-17",
        "question_end": "experiments/option-free-question-end-2026-07-18",
        "cross_mechanism": "experiments/cross-mechanism-outcome-2026-07-18",
        "multitoken": (
            "experiments/multitoken-sequence-robustness-2026-07-18"
        ),
        "cross_mechanism_patching": (
            "experiments/cross-mechanism-activation-patching-2026-07-18"
        ),
    }
    protocols = {
        name: read_json(f"{directory}/protocol.json")
        for name, directory in study_dirs.items()
    }
    statistics = {
        name: read_json(f"{directory}/statistics.json")
        for name, directory in study_dirs.items()
    }

    for name, directory in study_dirs.items():
        protocol_path = ROOT / directory / "protocol.json"
        actual = sha256(protocol_path)
        recorded = statistics[name]["protocol_sha256"]
        check(
            f"{name}: protocol fingerprint",
            actual == recorded,
            actual=actual,
            recorded=recorded,
        )

    manifest_spec = protocols["question_end"]["inputs"]["prompt_manifest"]
    manifest_path = ROOT / manifest_spec["path"]
    check(
        "source manifest fingerprint",
        sha256(manifest_path) == manifest_spec["sha256"],
        actual=sha256(manifest_path),
        recorded=manifest_spec["sha256"],
    )
    manifest = json.loads(manifest_path.read_text())
    prompts = manifest["prompts"]
    check("source prompt count", len(prompts) == 72, observed=len(prompts))
    family_counts = Counter(row["family_id"] for row in prompts)
    variant_counts = Counter(row["variant"] for row in prompts)
    outcome_counts = Counter(row["expected_outcome"] for row in prompts)
    check(
        "balanced family inventory",
        len(family_counts) == 6 and set(family_counts.values()) == {12},
        counts=dict(sorted(family_counts.items())),
    )
    check(
        "balanced surface variants",
        variant_counts
        == Counter(
            {
                "anchor": 24,
                "physics_paraphrase": 24,
                "lexical_counterfactual": 24,
            }
        ),
        counts=dict(sorted(variant_counts.items())),
    )
    check(
        "balanced physical outcomes",
        outcome_counts == Counter({"higher": 36, "lower": 36})
        or outcome_counts == Counter(
            {"higher": 18, "lower": 18, "greater": 18, "smaller": 18}
        ),
        counts=dict(sorted(outcome_counts.items())),
    )
    forbidden = (
        "answer exactly",
        "output a",
        "output b",
        "internal checkpoint",
        "if the scientific answer",
    )
    scaffold_leaks = [
        row["prompt_id"]
        for row in prompts
        if any(marker in row["stem"].lower() for marker in forbidden)
    ]
    check(
        "natural question stems contain no answer scaffold",
        not scaffold_leaks,
        leaking_prompt_ids=scaffold_leaks,
    )

    question_states_path = (
        ROOT
        / protocols["cross_mechanism"]["inputs"]["representations"]["path"]
    )
    recorded_states_hash = protocols["cross_mechanism"]["inputs"][
        "representations"
    ]["sha256"]
    check(
        "natural question-end state fingerprint",
        sha256(question_states_path) == recorded_states_hash,
        actual=sha256(question_states_path),
        recorded=recorded_states_hash,
    )
    with np.load(question_states_path) as arrays:
        shapes = {key: list(arrays[key].shape) for key in arrays.files}
        expected_shapes = {
            "prompt_ids": [72],
            "positions": [1],
            "layers": [25],
            "raw_states": [1, 72, 25, 2560],
            "direct_decoder_basis": [1, 72, 25, 2560],
            "jacobian_decoder_basis": [3, 1, 72, 25, 2560],
        }
        check(
            "natural question-end array dimensions",
            shapes == expected_shapes,
            observed=shapes,
            expected=expected_shapes,
        )

    checkpoint = statistics["checkpoint"]
    check(
        "checkpoint frozen endpoint",
        checkpoint["frozen_verdict"]["option_free_evidence"] == "none"
        and close(checkpoint["primary"]["graph_precision"], 0.5208333333333334)
        and close(checkpoint["primary"]["candidate_auc"], 0.4861111111111111),
        observed=checkpoint["primary"],
    )
    question = statistics["question_end"]
    check(
        "natural question-end frozen endpoint",
        question["frozen_verdict"]["option_free_evidence"] == "strong"
        and close(question["primary"]["graph_precision"], 0.6736111111111112)
        and close(question["primary"]["candidate_auc"], 0.6423611111111112),
        observed=question["primary"],
    )

    rankings = csv_rows(
        "experiments/cross-mechanism-outcome-2026-07-18/"
        "all_query_rankings.csv"
    )
    method_counts = Counter(row["method"] for row in rankings)
    counter_counts = Counter(
        row["method"]
        for row in rankings
        if row["opposite_response_orientation"] == "True"
    )
    check(
        "cross-mechanism query cardinality",
        len(method_counts) == 10
        and set(method_counts.values()) == {1080}
        and set(counter_counts.values()) == {648},
        all_queries_per_method=dict(sorted(method_counts.items())),
        counter_numeric_queries_per_method=dict(sorted(counter_counts.items())),
    )
    cross = statistics["cross_mechanism"]
    check(
        "cross-mechanism frozen endpoint",
        cross["frozen_verdict"]["cross_mechanism_evidence"] == "none"
        and close(cross["primary"]["overall_auc"], 0.513425925925926)
        and close(cross["primary"]["counter_numeric_auc"], 0.470679012345679),
        observed=cross["primary"],
    )

    sequence_rows = csv_rows(
        "experiments/multitoken-sequence-robustness-2026-07-18/"
        "layer_sequence_scores.csv"
    )
    sequence_counts = Counter(row["method"] for row in sequence_rows)
    check(
        "multi-token layer-score cardinality",
        len(sequence_counts) == 5 and set(sequence_counts.values()) == {250},
        rows_per_method=dict(sorted(sequence_counts.items())),
    )
    multitoken = statistics["multitoken"]
    current_runner = ROOT / "scripts/run_multitoken_sequence_robustness.py"
    amendment_path = ROOT / multitoken["protocol_amendment"]
    check(
        "multi-token amended runner fingerprint",
        sha256(current_runner) == multitoken["runner_sha256"],
        actual=sha256(current_runner),
        recorded=multitoken["runner_sha256"],
    )
    check(
        "multi-token amendment fingerprint",
        sha256(amendment_path) == multitoken["protocol_amendment_sha256"],
        actual=sha256(amendment_path),
        recorded=multitoken["protocol_amendment_sha256"],
    )
    check(
        "multi-token frozen endpoint",
        multitoken["frozen_verdict"]["sequence_robustness"] == "fail"
        and multitoken["frozen_verdict"]["positive_jacobian_prompts"] == 7
        and multitoken["frozen_verdict"]["required_positive_prompts"] == 8,
        observed=multitoken["frozen_verdict"],
    )

    patching = statistics["cross_mechanism_patching"]
    patching_rows = csv_rows(
        "experiments/cross-mechanism-activation-patching-2026-07-18/"
        "all_patch_rows.csv"
    )
    check(
        "cross-mechanism patch cardinality",
        len(patching_rows) == 1920
        and len({row["receiver_prompt_id"] for row in patching_rows}) == 24
        and len({row["donor_prompt_id"] for row in patching_rows}) == 24
        and {
            int(row["layer"]) for row in patching_rows
        }
        == {16, 24, 32, 37},
        n_rows=len(patching_rows),
    )
    check(
        "cross-mechanism patch frozen endpoint",
        patching["frozen_verdict"][
            "option_free_cross_mechanism_transfer"
        ]
        == "partial"
        and patching["frozen_verdict"]["positive_donor_families"] == 4
        and close(patching["primary"]["all"]["mean"], 0.384951780239741)
        and close(
            patching["structured_donor_label_nulls"]["all"][
                "exact_two_sided_p"
            ],
            0.031935871056241426,
        ),
        observed=patching["frozen_verdict"],
    )
    patch_audit = read_json(
        "experiments/cross-mechanism-activation-patching-2026-07-18/"
        "validation.json"
    )
    check(
        "independent cross-mechanism patch audit",
        patch_audit["passed"]
        and patch_audit["n_checks"] == 27
        and patch_audit["n_passed"] == 27,
        observed={
            "passed": patch_audit["passed"],
            "n_checks": patch_audit["n_checks"],
            "n_passed": patch_audit["n_passed"],
        },
    )
    with np.load(
        ROOT
        / "experiments"
        / "cross-mechanism-activation-patching-2026-07-18"
        / "primary_exact_donor_label_nulls.npz"
    ) as nulls:
        check(
            "structured donor-label null cardinality",
            all(len(nulls[name]) == 46656 for name in (
                "all",
                "cross_vocabulary",
                "opposite_orientation",
                "both",
            ))
            and nulls["assignment_indices"].shape == (46656, 6),
            assignment_shape=list(nulls["assignment_indices"].shape),
        )

    artifact_paths = [
        "experiments/option-free-relation-graph-2026-07-17/"
        "prompt_inventory.csv",
        "experiments/option-free-question-end-2026-07-18/"
        "representations.npz",
        "experiments/option-free-question-end-2026-07-18/"
        "all_candidate_rankings.csv",
        "experiments/cross-mechanism-outcome-2026-07-18/"
        "all_query_rankings.csv",
        "experiments/cross-mechanism-outcome-2026-07-18/"
        "exact_orientation_nulls.json",
        "experiments/multitoken-sequence-robustness-2026-07-18/"
        "layer_sequence_scores.csv",
        "experiments/multitoken-sequence-robustness-2026-07-18/raw.json",
        "experiments/cross-mechanism-activation-patching-2026-07-18/"
        "raw.json",
        "experiments/cross-mechanism-activation-patching-2026-07-18/"
        "all_patch_rows.csv",
        "experiments/cross-mechanism-activation-patching-2026-07-18/"
        "primary_exact_donor_label_nulls.npz",
        "experiments/cross-mechanism-activation-patching-2026-07-18/"
        "validation.json",
    ]
    artifact_hashes = {
        relative: sha256(ROOT / relative) for relative in artifact_paths
    }

    passed = all(row["passed"] for row in checks)
    report = {
        "study_id": "review-robustness-audit-2026-07-18",
        "passed": passed,
        "n_checks": len(checks),
        "n_passed": sum(row["passed"] for row in checks),
        "checks": checks,
        "artifact_sha256": artifact_hashes,
        "guardrail": (
            "This validation confirms archive integrity and registered "
            "calculation outputs. It is not a new inferential analysis."
        ),
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        failed = [row["name"] for row in checks if not row["passed"]]
        raise SystemExit("validation failed: " + ", ".join(failed))
    print(f"validated {report['n_passed']}/{report['n_checks']} checks")
    print(REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
