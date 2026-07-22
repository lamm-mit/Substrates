#!/usr/bin/env python3
"""Audit invariants and reported values for the rigorous graph analysis."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "graph-topology-rigorous-2026-07-17"
STATS = OUT / "statistics.json"
EDGES = OUT / "replication_relation_graph_selected_edges.csv"
RANKS = OUT / "replication_relation_candidate_ranks.csv"
MANIFEST = (
    ROOT
    / "experiments"
    / "late-physics-representation-replication-2026-07-17"
    / "prompt_manifest.json"
)
VECTORS = (
    ROOT
    / "experiments"
    / "late-physics-representation-replication-2026-07-17"
    / "representations.npz"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value} in {path}")
        ),
    )


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite statistic at {path}: {value}")


def main() -> None:
    checks: list[dict[str, object]] = []

    def record(name: str, detail: str) -> None:
        checks.append({"check": name, "status": "pass", "detail": detail})

    statistics = strict_json(STATS)
    assert_finite(statistics)
    record("strict_json_and_finite_values", "statistics.json is strict and finite")

    for name, entry in statistics["input_fingerprints"].items():
        path = ROOT / entry["path"]
        assert path.is_file(), path
        assert sha256(path) == entry["sha256"], name
    record("input_fingerprints", "all six frozen input SHA-256 hashes match")

    heldout = statistics["heldout"]
    primary = heldout["primary_cross_phrasing_graph"]
    assert all(row["n_directed_edges"] == 200 for row in primary.values())
    assert primary["jacobian"]["null_95"][1] < primary["jacobian"]["directed_precision"]
    assert (
        heldout["prompt_only_residualized_graph"]["jacobian"]["directed_precision"]
        > 0.10
    )
    record(
        "heldout_graph_design",
        "all methods have 50 prompts × 4 cross-phrasing edges; Jacobian exceeds its null",
    )

    jaccards = list(
        heldout["per_lens_seed_directed_edge_jaccard"].values()
    )
    assert min(jaccards) >= 0.95
    assert (
        heldout["undirected_graph_sensitivity"]["jacobian"]["mutual"][
            "homophily"
        ]
        > heldout["undirected_graph_sensitivity"]["jacobian"]["union"][
            "homophily"
        ]
    )
    record(
        "lens_and_graph_sensitivity",
        f"lens edge Jaccard {min(jaccards):.3f}–{max(jaccards):.3f}; mutual edges are more homophilic",
    )

    manifest = strict_json(MANIFEST)
    metadata = {str(row["prompt_id"]): row for row in manifest["prompts"]}
    with np.load(VECTORS, allow_pickle=False) as vectors:
        prompt_ids = vectors["prompt_ids"].astype(str)
    rows = [metadata[prompt_id] for prompt_id in prompt_ids]

    edge_frame = pd.read_csv(EDGES)
    methods = sorted(edge_frame["method"].unique())
    assert len(methods) == 10
    assert len(edge_frame) == 10 * 72 * 2
    for method, method_rows in edge_frame.groupby("method"):
        assert len(method_rows) == 144, method
        assert method_rows.groupby("source_index").size().eq(2).all(), method
        assert (method_rows["source_index"] != method_rows["target_index"]).all()
        for edge in method_rows.itertuples(index=False):
            source = rows[int(edge.source_index)]
            target = rows[int(edge.target_index)]
            assert source["family_id"] == target["family_id"]
            assert source["variant"] != target["variant"]
            assert source["triplet_id"] != target["triplet_id"]
            assert bool(edge.same_outcome) == (
                source["expected_outcome"] == target["expected_outcome"]
            )
    record(
        "relation_edge_invariants",
        "1,440 audited edges: two per source, cross-variant, same-family, different material case",
    )

    fixed = statistics["disjoint_relation_replication"]["fixed_graphs"]
    for method, method_rows in edge_frame.groupby("method"):
        observed = float(method_rows["same_outcome"].mean())
        expected = float(fixed[method]["directed_same_outcome_precision"])
        assert math.isclose(observed, expected, abs_tol=1e-12), method
    record(
        "reported_precision_recalculation",
        "all ten graph precisions exactly match row-level selected edges",
    )
    assert all(
        row["n_case_preserving_exact_assignments"] == 46_656
        for row in fixed.values()
    )
    assert (
        fixed["jacobian_band"]["case_preserving_exact_p"] < 0.05
    )
    record(
        "case_preserving_exact_null",
        "all fixed graphs use 46,656 balanced case-level assignments; Jacobian p<0.05",
    )

    rank_frame = pd.read_csv(RANKS)
    assert len(rank_frame) == 10 * 72 * 2
    assert rank_frame["first_positive_rank"].between(1, 3).all()
    assert rank_frame["n_same_outcome_candidates"].between(1, 2).all()
    assert rank_frame["pairwise_auc"].between(0, 1).all()
    rankings = statistics["disjoint_relation_replication"][
        "complete_candidate_ranking"
    ]
    jacobian_ranking = rankings["jacobian_band"]
    assert jacobian_ranking["pairwise_auc"] > 0.5
    assert (
        jacobian_ranking["leave_one_family_out_range"]["pairwise_auc"][0]
        > 0.5
    )
    record(
        "complete_candidate_rankings",
        "1,440 three-candidate rankings audited; Jacobian AUC and every leave-one-family-out AUC exceed chance",
    )

    diagnostics = statistics["disjoint_relation_replication"][
        "answer_order_and_variant_falsification"
    ]["jacobian_late"]
    crossings = [
        row
        for row in diagnostics["by_ordered_variant_pair"]
        if "lexical_counterfactual"
        in (row["source_variant"], row["target_variant"])
    ]
    assert len(crossings) == 4
    assert min(row["same_outcome_precision"] for row in crossings) >= 0.75
    assert max(row["bh_q_across_six_pairs"] for row in crossings) < 0.05
    order_groups = {
        bool(row["same_presentation_order"]): row
        for row in diagnostics["by_presentation_order_match"]
    }
    assert (
        order_groups[False]["same_outcome_precision"]
        > order_groups[True]["same_outcome_precision"]
    )
    record(
        "surface_form_falsification",
        "all four counterfactual crossings are ≥75% and BH q<0.05; opposite-order edges are more accurate",
    )

    assert fixed["answer_order_only"]["directed_same_outcome_precision"] < 0.5
    assert math.isclose(
        fixed["numeric_direction_oracle"]["directed_same_outcome_precision"],
        1.0,
    )
    record(
        "artifact_baselines",
        "answer-order-only is below chance; numeric-direction oracle is the 100% ceiling",
    )

    figures = [
        OUT / "figures" / "mechanism-graph-evidence.png",
        OUT / "figures" / "mechanism-graph-evidence.pdf",
        OUT / "figures" / "mechanism-graph-evidence.svg",
        OUT / "figures" / "relation-graph-falsification.png",
        OUT / "figures" / "relation-graph-falsification.pdf",
        OUT / "figures" / "relation-graph-falsification.svg",
        OUT / "figures" / "relation-ranking-robustness.png",
        OUT / "figures" / "relation-ranking-robustness.pdf",
        OUT / "figures" / "relation-ranking-robustness.svg",
    ]
    assert all(path.stat().st_size > 10_000 for path in figures)
    record("figure_artifacts", "all three figures exist as non-empty PNG, PDF, and SVG")

    validation = {
        "status": "pass",
        "n_checks": len(checks),
        "checks": checks,
        "audited_artifacts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [STATS, EDGES, RANKS, *figures]
        },
    }
    (OUT / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Graph-analysis validation",
        "",
        f"**Status: PASS ({len(checks)} checks).**",
        "",
    ]
    lines.extend(
        f"- **{row['check']}** — {row['detail']}" for row in checks
    )
    lines.extend(
        [
            "",
            "Reproduce:",
            "",
            "```bash",
            "python scripts/audit_graph_topology_rigorous.py",
            "```",
            "",
        ]
    )
    (OUT / "VALIDATION.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
