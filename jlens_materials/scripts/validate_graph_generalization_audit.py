#!/usr/bin/env python3
"""Validate the complete graph-generalization experiment bundle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = (
    ROOT
    / "experiments"
    / "graph-isomorphism-generalization-2026-07-18"
)
MANIFEST = (
    ROOT
    / "experiments"
    / "late-physics-representation-replication-2026-07-17"
    / "prompt_manifest.json"
)
STATES = (
    ROOT
    / "experiments"
    / "option-free-question-end-2026-07-18"
    / "representations.npz"
)
EXPECTED_ROWS = {
    "all_mapping_candidates.csv": 29160,
    "constrained_graph_null.csv": 10000,
    "exact_partition_bootstrap_summary.csv": 18,
    "exact_partition_metrics.csv": 36,
    "gauge_pair_agreement.csv": 45,
    "gin_ensemble_node_logits.csv": 792,
    "gin_family_metrics.csv": 66,
    "gin_node_logits.csv": 15840,
    "gin_seed_results.csv": 1320,
    "global_atlas_heldout.csv": 45,
    "isomorphism_pair_metrics.csv": 1170,
    "isomorphism_tests.csv": 7,
    "mapping_method_summary.csv": 3,
    "pair_mappings.csv": 1215,
    "permutation_cycles.csv": 60,
    "relation_gin_ensemble_pair_logits.csv": 3888,
    "relation_gin_family_metrics.csv": 54,
    "relation_gin_pair_logits.csv": 77760,
    "relation_gin_seed_results.csv": 1080,
    "relation_nonparametric.csv": 18,
    "spectral_community_layers.csv": 450,
    "spectral_community_metrics.csv": 72,
    "spectral_density_ablations.csv": 12,
    "spectral_synthetic_controls.csv": 2000,
}
HASH_FILES = [
    MANIFEST,
    STATES,
    EXP / "PROTOCOL.md",
    EXP / "protocol.json",
    EXP / "GAUGE_PROTOCOL.md",
    EXP / "gauge_protocol.json",
    EXP / "SPECTRAL_PROTOCOL.md",
    EXP / "spectral_protocol.json",
    EXP / "PARTITION_PROTOCOL.md",
    EXP / "partition_protocol.json",
    EXP / "REPORT.md",
    EXP / "ARTIFACT_INVENTORY.md",
    EXP / "study12_statistics.json",
    EXP / "study3_gin_statistics.json",
    EXP / "study4abc_statistics.json",
    EXP / "study4c_relation_gin_statistics.json",
    EXP / "study5_spectral_statistics.json",
    EXP / "study6_partition_statistics.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_finite_frame(frame: pd.DataFrame, name: str) -> None:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.size and not np.isfinite(numeric.to_numpy()).all():
        raise AssertionError(f"non-finite numeric value in {name}")


def relation_identity() -> dict[str, str]:
    prompts = json.loads(MANIFEST.read_text())["prompts"]
    result = {}
    for family in sorted({row["family_id"] for row in prompts}):
        rows = [row for row in prompts if row["family_id"] == family]
        physical = np.asarray(
            [
                row["expected_outcome"] == row["outcome_positive"]
                for row in rows
            ],
            dtype=bool,
        )
        numeric = np.asarray(
            [row["numeric_direction"] == "increase" for row in rows],
            dtype=bool,
        )
        if np.all(physical == numeric):
            result[family] = "direct"
        elif np.all(physical == ~numeric):
            result[family] = "inverse"
        else:
            raise AssertionError(
                f"physical/numeric factorization fails for {family}"
            )
        for first in range(len(rows)):
            for second in range(len(rows)):
                if (physical[first] == physical[second]) != (
                    numeric[first] == numeric[second]
                ):
                    raise AssertionError(
                        f"relation identity fails for {family}"
                    )
    return result


def validate_seed_grid(
    path: Path, configurations: int, expected_rows: int
) -> None:
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise AssertionError(f"unexpected rows in {path.name}")
    counts = frame.groupby(
        ["configuration", "heldout_family"]
    ).size()
    if len(counts) != configurations * 6 or not np.all(
        counts.to_numpy() == 20
    ):
        raise AssertionError(f"incomplete seed grid in {path.name}")
    if sorted(frame["seed_index"].unique()) != list(range(20)):
        raise AssertionError(f"unexpected seeds in {path.name}")


def main() -> None:
    if len(json.loads(MANIFEST.read_text())["prompts"]) != 72:
        raise AssertionError("prompt manifest does not contain 72 prompts")
    with np.load(STATES, allow_pickle=False) as archive:
        if archive["prompt_ids"].shape != (72,):
            raise AssertionError("representation prompt alignment mismatch")
        if archive["layers"].shape != (25,):
            raise AssertionError("representation layer count mismatch")
        for key in (
            "raw_states",
            "direct_decoder_basis",
            "jacobian_decoder_basis",
        ):
            if not np.isfinite(archive[key]).all():
                raise AssertionError(f"non-finite archived state: {key}")

    row_counts = {}
    for name, expected in EXPECTED_ROWS.items():
        path = EXP / name
        frame = pd.read_csv(path)
        row_counts[name] = len(frame)
        if len(frame) != expected:
            raise AssertionError(
                f"{name}: {len(frame)} rows, expected {expected}"
            )
        assert_finite_frame(frame, name)

    validate_seed_grid(
        EXP / "gin_seed_results.csv", configurations=11, expected_rows=1320
    )
    validate_seed_grid(
        EXP / "relation_gin_seed_results.csv",
        configurations=9,
        expected_rows=1080,
    )

    for name in (
        "graph-generalization-model-tests.png",
        "graph-generalization-model-tests.pdf",
        "graph-generalization-partition-tests.png",
        "graph-generalization-partition-tests.pdf",
        "graph-generalization-network-examples.png",
        "graph-generalization-network-examples.pdf",
        "graph-identifiability-summary.png",
        "graph-identifiability-summary.pdf",
    ):
        path = EXP / "figures" / name
        if not path.exists() or path.stat().st_size <= 1000:
            raise AssertionError(f"missing or empty figure: {name}")

    for name in (
        "study12_statistics.json",
        "study3_gin_statistics.json",
        "study4abc_statistics.json",
        "study4c_relation_gin_statistics.json",
        "study5_spectral_statistics.json",
        "study6_partition_statistics.json",
    ):
        value = json.loads((EXP / name).read_text())
        if not value.get("study_id"):
            raise AssertionError(f"missing study ID: {name}")

    orientations = relation_identity()
    hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in HASH_FILES
    }
    report = {
        "status": "pass",
        "prompt_count": 72,
        "layer_count": 25,
        "mechanism_count": 6,
        "absolute_gin_runs": 1320,
        "relation_gin_runs": 1080,
        "row_counts": row_counts,
        "mechanism_orientations": orientations,
        "physical_numeric_relation_identity": True,
        "sha256": hashes,
    }
    output = EXP / "VALIDATION.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
