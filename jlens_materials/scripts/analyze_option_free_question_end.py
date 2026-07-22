#!/usr/bin/env python3
"""Analyze the prospectively frozen natural question-end robustness run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_graph_topology_rigorous import (  # noqa: E402
    case_preserving_exact_permutations,
    cosine_layers,
    deterministic_binary_similarity,
    edge_precision,
    prompt_feature_matrices,
    relation_edges,
    relation_ranking_summary,
)
from analyze_option_free_relation_graph import (  # noqa: E402
    COLORS,
    LATE_BAND,
    N_BOOTSTRAP,
    PRIMARY_BAND,
    edge_rows,
    exact_p,
    family_metric_frame,
    json_safe,
    panel_label,
    paired_family_bootstrap,
    relation_candidate_auc_null,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "option-free-question-end-2026-07-18"
FIG = OUT / "figures"
PROTOCOL_PATH = OUT / "protocol.json"
DEFAULT_STATES = OUT / "representations.npz"
DEFAULT_RAW = OUT / "raw.json"
CHECKPOINT_OUT = (
    ROOT / "experiments" / "option-free-relation-graph-2026-07-17"
)
BOOTSTRAP_SEED = 20260719


def validate(protocol: dict, states: Path, raw: Path) -> dict[str, str]:
    actual = {
        "runner": sha256(
            ROOT / protocol["inputs"]["runner"]["path"]
        ),
        "prompt_manifest": sha256(
            ROOT / protocol["inputs"]["prompt_manifest"]["path"]
        ),
        "states": sha256(states),
        "raw": sha256(raw),
    }
    for name in ("runner", "prompt_manifest"):
        expected = protocol["inputs"][name]["sha256"]
        if actual[name] != expected:
            raise RuntimeError(
                f"fingerprint mismatch for {name}: "
                f"{actual[name]} != {expected}"
            )
    raw_payload = json.loads(raw.read_text())
    if raw_payload["provenance"]["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("raw output is not bound to the frozen protocol")
    if raw_payload["provenance"]["runner_sha256"] != actual["runner"]:
        raise RuntimeError("raw output runner fingerprint mismatch")
    return actual


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_figure(
    fixed: pd.DataFrame,
    family: pd.DataFrame,
    layer: pd.DataFrame,
    null: np.ndarray,
) -> None:
    configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.7))
    order = ["jacobian", "direct", "raw", "word_tfidf", "char_tfidf"]
    labels = ["Jacobian", "Direct", "Raw", "Word TF-IDF", "Char TF-IDF"]
    primary = fixed[
        (fixed["band"] == "primary") & fixed["method"].isin(order)
    ].set_index("method")
    x = np.arange(len(order))
    width = 0.36
    axes[0, 0].bar(
        x - width / 2,
        primary.loc[order, "graph_precision"],
        width,
        color=[COLORS[name] for name in order],
        label="selected-edge precision",
    )
    axes[0, 0].bar(
        x + width / 2,
        primary.loc[order, "candidate_auc"],
        width,
        color="white",
        edgecolor=[COLORS[name] for name in order],
        hatch="//",
        linewidth=1.2,
        label="all-candidate AUC",
    )
    axes[0, 0].axhline(
        float(np.mean(null)),
        color="#8B8E91",
        linestyle="--",
        linewidth=0.9,
    )
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("same-direction score")
    axes[0, 0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0, 0].legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.33),
    )
    panel_label(axes[0, 0], "A")

    for method, label_name in (
        ("jacobian", "Jacobian"),
        ("direct", "Direct"),
        ("raw", "Raw"),
    ):
        rows = layer[layer["method"] == method].sort_values("depth_percent")
        axes[0, 1].plot(
            rows["depth_percent"],
            rows["graph_precision"],
            color=COLORS[method],
            linewidth=1.6,
            marker="o",
            markersize=2.8,
            label=label_name,
        )
    axes[0, 1].axvspan(
        PRIMARY_BAND[0], PRIMARY_BAND[1], color="#D9E7EA", alpha=0.45
    )
    axes[0, 1].axhline(
        float(np.mean(null)),
        color="#8B8E91",
        linestyle="--",
        linewidth=0.9,
        label="structured-null mean",
    )
    axes[0, 1].set_ylim(0.3, 1.02)
    axes[0, 1].set_xlabel("layer depth (%)")
    axes[0, 1].set_ylabel("selected-edge precision")
    axes[0, 1].legend(frameon=False, ncol=2, loc="lower right")
    panel_label(axes[0, 1], "B")

    subset = family[
        (family["band"] == "primary")
        & family["method"].isin(["jacobian", "direct", "raw"])
    ]
    family_names = sorted(subset["family"].unique())
    short = {
        "crosslink-density-modulus": "Crosslink",
        "dislocation-density-strength": "Dislocation",
        "obstacle-spacing-orowan": "Orowan",
        "particle-fraction-modulus": "Particles",
        "pearlite-spacing-strength": "Pearlite",
        "porosity-modulus": "Porosity",
    }
    offsets = {"jacobian": -0.18, "direct": 0.0, "raw": 0.18}
    for method, label_name in (
        ("jacobian", "Jacobian"),
        ("direct", "Direct"),
        ("raw", "Raw"),
    ):
        values = (
            subset[subset["method"] == method]
            .set_index("family")
            .loc[family_names, "candidate_auc"]
        )
        axes[1, 0].scatter(
            np.arange(len(family_names)) + offsets[method],
            values,
            s=28,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            label=label_name,
        )
    axes[1, 0].axhline(0.5, color="#8B8E91", linestyle="--", linewidth=0.9)
    axes[1, 0].set_ylim(0.2, 1.02)
    axes[1, 0].set_ylabel("all-candidate AUC")
    axes[1, 0].set_xticks(
        np.arange(len(family_names)),
        [short[name] for name in family_names],
        rotation=25,
        ha="right",
    )
    axes[1, 0].legend(frameon=False, ncol=3, loc="lower center")
    panel_label(axes[1, 0], "C")

    checkpoint = pd.read_csv(CHECKPOINT_OUT / "fixed_graph_statistics.csv")
    positions = [
        ("question_end", fixed, "question end"),
        (
            "checkpoint",
            checkpoint[
                (checkpoint["position"] == "checkpoint")
                & (checkpoint["band"] == "primary")
            ],
            "checkpoint\nmarker",
        ),
        (
            "final_prompt",
            checkpoint[
                (checkpoint["position"] == "final_prompt")
                & (checkpoint["band"] == "primary")
            ],
            "after answer\nmapping",
        ),
    ]
    for method, label_name in (
        ("jacobian", "Jacobian"),
        ("direct", "Direct"),
        ("raw", "Raw"),
    ):
        values = []
        for _, table, _ in positions:
            values.append(
                float(
                    table.loc[
                        (table["method"] == method)
                        & (table["band"] == "primary"),
                        "candidate_auc",
                    ].iloc[0]
                )
            )
        axes[1, 1].plot(
            range(3),
            values,
            color=COLORS[method],
            marker="o",
            markersize=5,
            linewidth=1.5,
            label=label_name,
        )
    axes[1, 1].axhline(0.5, color="#8B8E91", linestyle="--", linewidth=0.9)
    axes[1, 1].set_ylim(0.2, 1.02)
    axes[1, 1].set_xticks(
        range(3), [row[2] for row in positions]
    )
    axes[1, 1].set_ylabel("all-candidate AUC")
    axes[1, 1].legend(frameon=False, loc="lower right")
    panel_label(axes[1, 1], "D")

    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.13,
        top=0.97,
        wspace=0.32,
        hspace=0.50,
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(
            FIG / f"option-free-question-end.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL_PATH.read_text())
    fingerprints = validate(protocol, args.states, args.raw)
    manifest_path = ROOT / protocol["inputs"]["prompt_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    metadata = {
        str(row["prompt_id"]): row for row in manifest["prompts"]
    }
    with np.load(args.states, allow_pickle=False) as arrays:
        prompt_ids = arrays["prompt_ids"].astype(str)
        positions = arrays["positions"].astype(str)
        layers = arrays["layers"].astype(int)
        raw = arrays["raw_states"].astype(np.float64)
        direct = arrays["direct_decoder_basis"].astype(np.float64)
        jacobian = arrays["jacobian_decoder_basis"].astype(np.float64)
    if positions.tolist() != ["question_end"]:
        raise RuntimeError(f"unexpected positions: {positions.tolist()}")
    rows = [metadata[prompt_id] for prompt_id in prompt_ids]
    families = np.asarray([str(row["family_id"]) for row in rows])
    variants = np.asarray([str(row["variant"]) for row in rows])
    triplets = np.asarray([str(row["triplet_id"]) for row in rows])
    outcomes = np.asarray([str(row["expected_outcome"]) for row in rows])
    presentation = np.asarray(
        [str(row["presentation_order"]) for row in rows]
    )
    numeric = np.asarray([str(row["numeric_direction"]) for row in rows])
    stems = [str(row["stem"]) for row in rows]
    depths = layers / 41.0 * 100.0
    primary_mask = (depths >= PRIMARY_BAND[0]) & (
        depths <= PRIMARY_BAND[1]
    )
    late_mask = (depths >= LATE_BAND[0]) & (depths <= LATE_BAND[1])

    binary = np.asarray(
        [
            int(row["expected_outcome"] == row["outcome_positive"])
            for row in rows
        ],
        dtype=np.int8,
    )
    assignments = case_preserving_exact_permutations(
        families, variants, triplets, outcomes
    )
    if not np.any(np.all(assignments == binary[None, :], axis=1)):
        raise RuntimeError("observed assignment missing from exact null")

    sim_j_seed = cosine_layers(jacobian[:, 0])
    sim_j = np.mean(sim_j_seed, axis=0)
    sim_direct = cosine_layers(direct[0][None, ...])[0]
    sim_raw = cosine_layers(raw[0][None, ...])[0]
    lexical = prompt_feature_matrices(stems)
    methods = {
        "jacobian_primary": np.mean(sim_j[primary_mask], axis=0),
        "direct_primary": np.mean(sim_direct[primary_mask], axis=0),
        "raw_primary": np.mean(sim_raw[primary_mask], axis=0),
        "jacobian_late": np.mean(sim_j[late_mask], axis=0),
        "direct_late": np.mean(sim_direct[late_mask], axis=0),
        "raw_late": np.mean(sim_raw[late_mask], axis=0),
        "word_tfidf_primary": lexical["word_tfidf"],
        "char_tfidf_primary": lexical["char_tfidf"],
        "answer_order_primary": deterministic_binary_similarity(presentation),
        "numeric_oracle_primary": deterministic_binary_similarity(numeric),
    }
    for seed in range(3):
        methods[f"jacobian_seed{seed}_primary"] = np.mean(
            sim_j_seed[seed, primary_mask], axis=0
        )

    fixed_rows = []
    edges_all = []
    rankings_all = []
    families_all = []
    primary_null = None
    primary_auc_null = None
    for compound, similarity in methods.items():
        if compound.endswith("_primary"):
            method = compound[: -len("_primary")]
            band = "primary"
        else:
            method = compound[: -len("_late")]
            band = "late"
        edges = relation_edges(similarity, families, variants, triplets)
        graph = edge_precision(edges, outcomes)
        graph_null = np.mean(
            assignments[:, edges[:, 0]] == assignments[:, edges[:, 1]],
            axis=1,
        )
        ranking, ranking_frame = relation_ranking_summary(
            similarity, families, variants, triplets, outcomes
        )
        auc = float(ranking["pairwise_auc"])
        auc_null = relation_candidate_auc_null(
            similarity, families, variants, triplets, assignments
        )
        family_frame = family_metric_frame(
            method,
            "question_end",
            band,
            edges,
            ranking_frame,
            outcomes,
            families,
        )
        families_all.append(family_frame)
        fixed_rows.append(
            {
                "position": "question_end",
                "band": band,
                "method": method,
                "graph_precision": graph,
                "graph_exact_p": exact_p(graph_null, graph),
                "graph_null_mean": float(np.mean(graph_null)),
                "graph_null_q025": float(np.quantile(graph_null, 0.025)),
                "graph_null_q975": float(np.quantile(graph_null, 0.975)),
                "candidate_auc": auc,
                "candidate_auc_exact_p": exact_p(auc_null, auc),
                "candidate_auc_null_mean": float(np.mean(auc_null)),
                "candidate_auc_null_q025": float(
                    np.quantile(auc_null, 0.025)
                ),
                "candidate_auc_null_q975": float(
                    np.quantile(auc_null, 0.975)
                ),
                "positive_auc_families": int(
                    np.sum(family_frame["candidate_auc"] > 0.5)
                ),
                "n_edges": int(len(edges)),
                "n_rankings": int(ranking["n_rankings"]),
            }
        )
        edges_all.append(
            edge_rows(
                method,
                "question_end",
                band,
                similarity,
                edges,
                rows,
            )
        )
        rank_copy = ranking_frame.copy()
        rank_copy.insert(0, "band", band)
        rank_copy.insert(0, "position", "question_end")
        rank_copy.insert(0, "method", method)
        rankings_all.append(rank_copy)
        if method == "jacobian" and band == "primary":
            primary_null = graph_null
            primary_auc_null = auc_null

    fixed = pd.DataFrame(fixed_rows).sort_values(["band", "method"])
    family = pd.concat(families_all, ignore_index=True)
    edges = pd.concat(edges_all, ignore_index=True)
    rankings = pd.concat(rankings_all, ignore_index=True)

    layer_rows = []
    layer_edges = {"jacobian": [], "direct": [], "raw": []}
    for method, layered in (
        ("jacobian", sim_j),
        ("direct", sim_direct),
        ("raw", sim_raw),
    ):
        for index, depth in enumerate(depths):
            similarity = layered[index]
            selected = relation_edges(
                similarity, families, variants, triplets
            )
            ranking, _ = relation_ranking_summary(
                similarity, families, variants, triplets, outcomes
            )
            layer_rows.append(
                {
                    "position": "question_end",
                    "method": method,
                    "layer": int(layers[index]),
                    "depth_percent": float(depth),
                    "graph_precision": edge_precision(selected, outcomes),
                    "candidate_auc": float(ranking["pairwise_auc"]),
                }
            )
            layer_edges[method].append(selected)
    layer = pd.DataFrame(layer_rows)

    layer_scan = {}
    for method, method_edges in layer_edges.items():
        observed = (
            layer[layer["method"] == method]
            .sort_values("depth_percent")["graph_precision"]
            .to_numpy()
        )
        null = np.empty(
            (len(assignments), len(method_edges)), dtype=np.float32
        )
        for index, selected in enumerate(method_edges):
            null[:, index] = np.mean(
                assignments[:, selected[:, 0]]
                == assignments[:, selected[:, 1]],
                axis=1,
            )
        best = int(np.argmax(observed))
        layer_scan[method] = {
            "best_graph_precision": float(observed[best]),
            "best_layer": int(layers[best]),
            "best_depth_percent": float(depths[best]),
            "max_layer_exact_p": exact_p(
                np.max(null, axis=1), float(observed[best])
            ),
        }

    primary_family = family[
        (family["band"] == "primary")
        & family["method"].isin(["jacobian", "direct", "raw"])
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contrasts = {}
    for metric in ("graph_precision", "candidate_auc"):
        contrasts[f"jacobian_minus_direct_{metric}"] = paired_family_bootstrap(
            primary_family,
            "jacobian",
            "direct",
            metric,
            rng,
            n=N_BOOTSTRAP,
        )
        contrasts[f"jacobian_minus_raw_{metric}"] = paired_family_bootstrap(
            primary_family,
            "jacobian",
            "raw",
            metric,
            rng,
            n=N_BOOTSTRAP,
        )

    primary = fixed[
        (fixed["band"] == "primary") & (fixed["method"] == "jacobian")
    ].iloc[0]
    graph_pass = bool(primary["graph_exact_p"] <= 0.05)
    auc_pass = bool(
        primary["candidate_auc_exact_p"] <= 0.05
        and primary["candidate_auc"] > 0.5
    )
    breadth_pass = bool(primary["positive_auc_families"] >= 4)
    if graph_pass and auc_pass and breadth_pass:
        verdict = "strong"
    elif graph_pass or auc_pass:
        verdict = "partial"
    else:
        verdict = "none"
    jacobian_specific = bool(
        contrasts["jacobian_minus_direct_candidate_auc"][
            "family_bootstrap_95"
        ][0]
        > 0
    )

    fixed.to_csv(OUT / "fixed_graph_statistics.csv", index=False)
    family.to_csv(OUT / "family_metrics.csv", index=False)
    edges.to_csv(OUT / "all_selected_edges.csv", index=False)
    rankings.to_csv(OUT / "all_candidate_rankings.csv", index=False)
    layer.to_csv(OUT / "layer_metrics.csv", index=False)
    np.savez_compressed(
        OUT / "primary_exact_nulls.npz",
        structured_assignments=assignments,
        graph_precision=primary_null,
        candidate_auc=primary_auc_null,
    )

    payload = {
        "study_id": protocol["study_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "input_fingerprints": fingerprints,
        "primary": {
            key: json_safe(value) for key, value in primary.to_dict().items()
        },
        "frozen_verdict": {
            "option_free_evidence": verdict,
            "graph_precision_pass": graph_pass,
            "candidate_auc_pass": auc_pass,
            "family_breadth_pass": breadth_pass,
            "jacobian_specific": jacobian_specific,
        },
        "paired_family_contrasts": contrasts,
        "layer_scan": layer_scan,
        "all_fixed_results": fixed.to_dict(orient="records"),
        "guardrail": (
            "This prospectively frozen positional robustness run reuses the "
            "same inspected cohort and endpoint. It is not an independent "
            "replication or a causal experiment."
        ),
    }
    (OUT / "statistics.json").write_text(
        json.dumps(json_safe(payload), indent=2) + "\n"
    )
    make_figure(fixed, family, layer, np.asarray(primary_null))

    (OUT / "RESULTS.md").write_text(
        "\n".join(
            [
                "# Option-free natural question-end graph",
                "",
                f"Frozen interpretation: **{verdict.upper()}** option-free evidence.",
                "",
                "## Primary result",
                "",
                (
                    f"- Selected-edge precision: "
                    f"{100 * float(primary['graph_precision']):.1f}% "
                    f"(exact structured-null "
                    f"`p={float(primary['graph_exact_p']):.6f}`)."
                ),
                (
                    f"- Full-candidate AUC: "
                    f"{float(primary['candidate_auc']):.3f} "
                    f"(exact structured-null "
                    f"`p={float(primary['candidate_auc_exact_p']):.6f}`)."
                ),
                (
                    f"- Family breadth: "
                    f"{int(primary['positive_auc_families'])}/6 AUCs exceed 0.5."
                ),
                (
                    "- Jacobian-specific: "
                    f"**{'yes' if jacobian_specific else 'no'}** under the "
                    "frozen paired-family interval rule."
                ),
                "",
                "## Meaning",
                "",
                (
                    "The captured state follows the complete scientific "
                    "question but contains no answer choices, answer words, "
                    "arbitrary code, response instruction, or checkpoint marker."
                ),
                "",
                (
                    "This is a prospectively frozen positional robustness run "
                    "on the same previously inspected cohort. It is not an "
                    "independent replication and is not causal."
                ),
                "",
                "## Reproduction",
                "",
                "```bash",
                "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\",
                "  python scripts/run_option_free_question_end_states.py",
                "python scripts/analyze_option_free_question_end.py",
                "```",
                "",
                "All exact stems are retained in the source manifest and in the",
                "option-free checkpoint experiment's `prompt_inventory.csv`.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
