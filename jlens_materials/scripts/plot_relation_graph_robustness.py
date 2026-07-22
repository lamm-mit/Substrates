#!/usr/bin/env python3
"""Build publication figures for the signed-relation robustness audit.

The script does not run Gemma or change any registered endpoint.  It renders
the already archived 72-node natural-question graph, all 25 layerwise graphs,
and the union/persistence graph from the exact edge-selection rule used in the
frozen analysis.  It also combines the positional and cross-mechanism
falsification results into a revised main-paper figure.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_graph_topology_rigorous import (  # noqa: E402
    cosine_layers,
    relation_edges,
)
from analyze_option_free_relation_graph import COLORS  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "option-free-question-end-2026-07-18"
CHECKPOINT = ROOT / "experiments" / "option-free-relation-graph-2026-07-17"
CROSS = ROOT / "experiments" / "cross-mechanism-outcome-2026-07-18"
OUT = ROOT / "experiments" / "relation-graph-visualization-2026-07-18"
FIG = OUT / "figures"
PAPER_FIG = ROOT / "paper" / "figures"
STATES = SOURCE / "representations.npz"
MANIFEST = (
    ROOT
    / "experiments"
    / "answer-code-binding-2026-07-17"
    / "prompt_manifest.json"
)

FAMILY_ORDER = [
    "obstacle-spacing-orowan",
    "porosity-modulus",
    "pearlite-spacing-strength",
    "dislocation-density-strength",
    "particle-fraction-modulus",
    "crosslink-density-modulus",
]
FAMILY_SHORT = {
    "obstacle-spacing-orowan": "Orowan",
    "porosity-modulus": "Porosity",
    "pearlite-spacing-strength": "Pearlite",
    "dislocation-density-strength": "Dislocation",
    "particle-fraction-modulus": "Particles",
    "crosslink-density-modulus": "Crosslink",
}
FAMILY_COLORS = {
    "obstacle-spacing-orowan": "#3B82A0",
    "porosity-modulus": "#C7794A",
    "pearlite-spacing-strength": "#6D9F58",
    "dislocation-density-strength": "#8067A8",
    "particle-fraction-modulus": "#C9A13B",
    "crosslink-density-modulus": "#4E8C84",
}
CORRECT_EDGE = "#168A83"
INCORRECT_EDGE = "#D66B59"
VARIANT_MARKERS = {
    "anchor": "o",
    "physics_paraphrase": "s",
    "lexical_counterfactual": "^",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_data() -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text())
    metadata = {
        str(row["prompt_id"]): row for row in manifest["prompts"]
    }
    with np.load(STATES, allow_pickle=False) as arrays:
        prompt_ids = arrays["prompt_ids"].astype(str)
        positions = arrays["positions"].astype(str)
        layers = arrays["layers"].astype(int)
        jacobian = arrays["jacobian_decoder_basis"].astype(np.float64)
    if positions.tolist() != ["question_end"]:
        raise RuntimeError(f"unexpected positions: {positions.tolist()}")
    rows = [metadata[prompt_id] for prompt_id in prompt_ids]
    families = np.asarray([str(row["family_id"]) for row in rows])
    variants = np.asarray([str(row["variant"]) for row in rows])
    triplets = np.asarray([str(row["triplet_id"]) for row in rows])
    outcomes = np.asarray([str(row["expected_outcome"]) for row in rows])
    positive = np.asarray([str(row["outcome_positive"]) for row in rows])
    binary = outcomes == positive
    seed_similarities = cosine_layers(jacobian[:, 0])
    similarities = np.mean(seed_similarities, axis=0)
    depths = layers / 41.0 * 100.0
    primary = (depths >= 38.0) & (depths <= 92.0)
    primary_similarity = np.mean(similarities[primary], axis=0)
    primary_edges = relation_edges(
        primary_similarity, families, variants, triplets
    )
    layer_edges = [
        relation_edges(similarity, families, variants, triplets)
        for similarity in similarities
    ]
    return {
        "rows": rows,
        "prompt_ids": prompt_ids,
        "families": families,
        "variants": variants,
        "triplets": triplets,
        "outcomes": outcomes,
        "binary": binary,
        "layers": layers,
        "depths": depths,
        "primary_edges": primary_edges,
        "layer_edges": layer_edges,
    }


def node_positions(
    families: np.ndarray,
    triplets: np.ndarray,
    variants: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Fixed six-island layout shared by every graph panel."""

    centers = {
        family: (
            3.15 * (index % 3),
            -2.60 * (index // 3),
        )
        for index, family in enumerate(FAMILY_ORDER)
    }
    positions: dict[int, tuple[float, float]] = {}
    variant_radius = {
        "anchor": 0.52,
        "physics_paraphrase": 0.78,
        "lexical_counterfactual": 1.04,
    }
    for family in FAMILY_ORDER:
        indices = np.flatnonzero(families == family)
        cases = sorted(set(triplets[indices]))
        center_x, center_y = centers[family]
        for node in indices:
            case_index = cases.index(triplets[node])
            angle = np.pi / 4.0 + case_index * np.pi / 2.0
            radius = variant_radius[str(variants[node])]
            positions[int(node)] = (
                center_x + radius * np.cos(angle),
                center_y + radius * np.sin(angle),
            )
    return positions


def draw_relation_graph(
    axis: plt.Axes,
    edges: np.ndarray,
    data: dict[str, object],
    *,
    node_size: float,
    edge_width: float,
    edge_alpha: float,
    labels: bool,
    arrowheads: bool = False,
) -> float:
    families = np.asarray(data["families"])
    variants = np.asarray(data["variants"])
    outcomes = np.asarray(data["outcomes"])
    binary = np.asarray(data["binary"])
    positions = node_positions(
        families,
        np.asarray(data["triplets"]),
        variants,
    )

    for source, target in edges:
        first = positions[int(source)]
        second = positions[int(target)]
        correct = outcomes[int(source)] == outcomes[int(target)]
        color = CORRECT_EDGE if correct else INCORRECT_EDGE
        if arrowheads:
            patch = FancyArrowPatch(
                first,
                second,
                arrowstyle="-|>",
                mutation_scale=3.2,
                linewidth=edge_width,
                color=color,
                alpha=edge_alpha,
                shrinkA=2.5,
                shrinkB=2.5,
                connectionstyle="arc3,rad=0.04",
                zorder=1,
            )
            axis.add_patch(patch)
        else:
            axis.plot(
                [first[0], second[0]],
                [first[1], second[1]],
                color=color,
                linewidth=edge_width,
                alpha=edge_alpha,
                zorder=1,
                solid_capstyle="round",
            )

    for variant, marker in VARIANT_MARKERS.items():
        indices = np.flatnonzero(variants == variant)
        for outcome_value, face in ((True, "#183A54"), (False, "white")):
            chosen = indices[binary[indices] == outcome_value]
            axis.scatter(
                [positions[int(index)][0] for index in chosen],
                [positions[int(index)][1] for index in chosen],
                s=node_size,
                marker=marker,
                facecolor=face,
                edgecolor="#183A54",
                linewidth=max(0.35, node_size / 90.0),
                zorder=3,
            )

    if labels:
        centers = {
            family: (
                3.15 * (index % 3),
                -2.60 * (index // 3),
            )
            for index, family in enumerate(FAMILY_ORDER)
        }
        for family, (x_value, y_value) in centers.items():
            axis.text(
                x_value,
                y_value + 1.25,
                FAMILY_SHORT[family],
                color=FAMILY_COLORS[family],
                fontsize=7.2,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

    correct_fraction = float(
        np.mean(outcomes[edges[:, 0]] == outcomes[edges[:, 1]])
    )
    axis.set_xlim(-1.35, 7.65)
    axis.set_ylim(-3.75, 1.55)
    axis.set_aspect("equal")
    axis.axis("off")
    return correct_fraction


def main_figure(data: dict[str, object]) -> None:
    fixed = pd.read_csv(SOURCE / "fixed_graph_statistics.csv")
    checkpoint = pd.read_csv(CHECKPOINT / "fixed_graph_statistics.csv")
    cross = pd.read_csv(CROSS / "fixed_statistics.csv")

    figure = plt.figure(figsize=(7.25, 5.65))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        bottom=0.105,
        top=0.975,
        wspace=0.32,
        hspace=0.42,
    )
    axes = np.asarray(
        [
            [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])],
            [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
        ]
    )

    methods = ["jacobian", "direct", "raw"]
    method_names = {"jacobian": "Jacobian", "direct": "Direct", "raw": "Raw"}

    position_tables = [
        (
            fixed[fixed["band"] == "primary"],
            "natural\nquestion end",
        ),
        (
            checkpoint[
                (checkpoint["position"] == "checkpoint")
                & (checkpoint["band"] == "primary")
            ],
            "checkpoint\nmarker",
        ),
        (
            checkpoint[
                (checkpoint["position"] == "final_prompt")
                & (checkpoint["band"] == "primary")
            ],
            "after answer\nmapping",
        ),
    ]
    for method in methods:
        values = [
            float(
                table.loc[table["method"] == method, "candidate_auc"].iloc[0]
            )
            for table, _ in position_tables
        ]
        axes[0, 0].plot(
            np.arange(3),
            values,
            color=COLORS[method],
            marker="o",
            markersize=4.5,
            linewidth=1.45,
            label=method_names[method],
        )
    axes[0, 0].axhline(
        0.5, color="#8D9297", linestyle="--", linewidth=0.8
    )
    axes[0, 0].set_xticks(
        np.arange(3), [label for _, label in position_tables]
    )
    axes[0, 0].set_ylim(0.30, 0.88)
    axes[0, 0].set_ylabel("all-candidate AUC")
    axes[0, 0].legend(frameon=False, loc="upper left")

    order = ["jacobian", "direct", "raw", "word_tfidf", "char_tfidf"]
    names = ["Jacobian", "Direct", "Raw", "Word", "Character"]
    primary = fixed[
        (fixed["band"] == "primary") & fixed["method"].isin(order)
    ].set_index("method")
    x_values = np.arange(len(order))
    width = 0.34
    axes[0, 1].bar(
        x_values - width / 2,
        primary.loc[order, "graph_precision"],
        width,
        color=[COLORS[name] for name in order],
        label="selected edges",
    )
    axes[0, 1].bar(
        x_values + width / 2,
        primary.loc[order, "candidate_auc"],
        width,
        facecolor="white",
        edgecolor=[COLORS[name] for name in order],
        linewidth=1.1,
        hatch="//",
        label="all candidates",
    )
    axes[0, 1].axhline(
        float(primary.loc["jacobian", "graph_null_mean"]),
        color="#8D9297",
        linestyle="--",
        linewidth=0.8,
        label="structured-null mean",
    )
    axes[0, 1].set_xticks(x_values, names, rotation=22, ha="right")
    axes[0, 1].set_ylim(0.3, 0.78)
    axes[0, 1].set_ylabel("same-direction score")
    axes[0, 1].legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )

    cross = cross.set_index("method")
    categories = [
        ("within supplied mechanism", None),
        ("across mechanisms", "overall_auc"),
        ("cross-mechanism,\ncounter-numeric", "counter_numeric_auc"),
    ]
    for method_index, method in enumerate(methods):
        values = [
            float(primary.loc[method, "candidate_auc"]),
            float(cross.loc[method, "overall_auc"]),
            float(cross.loc[method, "counter_numeric_auc"]),
        ]
        axes[1, 0].plot(
            np.arange(3),
            values,
            color=COLORS[method],
            marker="o",
            markersize=4.5,
            linewidth=1.45,
            label=method_names[method],
        )
    axes[1, 0].axhline(
        0.5, color="#8D9297", linestyle="--", linewidth=0.8
    )
    axes[1, 0].set_xticks(
        np.arange(3), [name for name, _ in categories]
    )
    axes[1, 0].set_ylim(0.38, 0.70)
    axes[1, 0].set_ylabel("all-candidate AUC")

    precision = draw_relation_graph(
        axes[1, 1],
        np.asarray(data["primary_edges"]),
        data,
        node_size=18,
        edge_width=0.52,
        edge_alpha=0.38,
        labels=True,
        arrowheads=False,
    )
    # Node and edge encodings are stated in the caption.  Omitting an
    # in-panel legend keeps the dense, actual graph readable at column width.

    for label, axis in zip("ABCD", axes.ravel()):
        axis.text(
            -0.16,
            1.07,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    save_figure(figure, "relation-graph-robustness")
    plt.close(figure)


def layer_atlas(data: dict[str, object]) -> pd.DataFrame:
    layers = np.asarray(data["layers"])
    depths = np.asarray(data["depths"])
    outcomes = np.asarray(data["outcomes"])
    layer_edges = list(data["layer_edges"])
    figure, axes = plt.subplots(5, 5, figsize=(10.8, 7.55))
    records = []
    for index, (axis, layer, depth, edges) in enumerate(
        zip(axes.ravel(), layers, depths, layer_edges)
    ):
        precision = draw_relation_graph(
            axis,
            np.asarray(edges),
            data,
            node_size=5.0,
            edge_width=0.25,
            edge_alpha=0.26,
            labels=False,
        )
        axis.text(
            0.5,
            0.98,
            f"L{layer} · {depth:.0f}% · {100 * precision:.0f}% correct",
            transform=axis.transAxes,
            fontsize=6.0,
            ha="center",
            va="top",
        )
        records.extend(
            {
                "layer": int(layer),
                "depth_percent": float(depth),
                "source_index": int(source),
                "target_index": int(target),
                "same_outcome": bool(
                    outcomes[int(source)] == outcomes[int(target)]
                ),
            }
            for source, target in edges
        )
    figure.subplots_adjust(
        left=0.02,
        right=0.995,
        bottom=0.035,
        top=0.99,
        wspace=0.025,
        hspace=0.09,
    )
    save_figure(figure, "relation-graph-layer-atlas")
    plt.close(figure)
    return pd.DataFrame(records)


def persistence_figure(
    data: dict[str, object], layer_edge_frame: pd.DataFrame
) -> pd.DataFrame:
    families = np.asarray(data["families"])
    variants = np.asarray(data["variants"])
    outcomes = np.asarray(data["outcomes"])
    binary = np.asarray(data["binary"])
    positions = node_positions(
        families,
        np.asarray(data["triplets"]),
        variants,
    )
    counts = Counter(
        zip(
            layer_edge_frame["source_index"].astype(int),
            layer_edge_frame["target_index"].astype(int),
        )
    )
    rows = []
    for (source, target), count in sorted(counts.items()):
        rows.append(
            {
                "source_index": source,
                "target_index": target,
                "layers_selected": count,
                "fraction_of_layers": count / 25.0,
                "same_outcome": bool(outcomes[source] == outcomes[target]),
            }
        )
    persistence = pd.DataFrame(rows)

    figure, axis = plt.subplots(figsize=(9.7, 5.35))
    for row in persistence.itertuples(index=False):
        first = positions[int(row.source_index)]
        second = positions[int(row.target_index)]
        color = CORRECT_EDGE if row.same_outcome else INCORRECT_EDGE
        axis.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color=color,
            linewidth=0.25 + 2.8 * float(row.fraction_of_layers),
            alpha=0.08 + 0.72 * float(row.fraction_of_layers),
            zorder=1,
            solid_capstyle="round",
        )
    for variant, marker in VARIANT_MARKERS.items():
        indices = np.flatnonzero(variants == variant)
        for outcome_value, face in ((True, "#183A54"), (False, "white")):
            chosen = indices[binary[indices] == outcome_value]
            axis.scatter(
                [positions[int(index)][0] for index in chosen],
                [positions[int(index)][1] for index in chosen],
                s=35,
                marker=marker,
                facecolor=face,
                edgecolor="#183A54",
                linewidth=0.65,
                zorder=3,
            )
    centers = {
        family: (
            3.15 * (index % 3),
            -2.60 * (index // 3),
        )
        for index, family in enumerate(FAMILY_ORDER)
    }
    for family, (x_value, y_value) in centers.items():
        axis.text(
            x_value,
            y_value + 1.28,
            FAMILY_SHORT[family],
            color=FAMILY_COLORS[family],
            fontsize=9.2,
            fontweight="bold",
            ha="center",
        )
    axis.set_xlim(-1.45, 7.75)
    axis.set_ylim(-3.85, 1.65)
    axis.set_aspect("equal")
    axis.axis("off")
    legend = [
        Line2D(
            [0], [0], color=CORRECT_EDGE, lw=2.2, label="same direction"
        ),
        Line2D(
            [0], [0], color=INCORRECT_EDGE, lw=2.2, label="opposite direction"
        ),
        Line2D(
            [0],
            [0],
            color="#71777D",
            lw=0.6,
            alpha=0.5,
            label="rare across layers",
        ),
        Line2D(
            [0],
            [0],
            color="#71777D",
            lw=3.0,
            alpha=0.8,
            label="persistent across layers",
        ),
    ]
    axis.legend(
        handles=legend,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
    )
    figure.subplots_adjust(left=0.02, right=0.99, bottom=0.08, top=0.99)
    save_figure(figure, "relation-graph-persistence")
    plt.close(figure)
    return persistence


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        path = FIG / f"{stem}.{suffix}"
        figure.savefig(path, dpi=350, bbox_inches="tight")
        if suffix == "pdf":
            (PAPER_FIG / path.name).write_bytes(path.read_bytes())


def write_artifacts(
    data: dict[str, object],
    layer_edges: pd.DataFrame,
    persistence: pd.DataFrame,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    layer_edges.to_csv(OUT / "layerwise_selected_edges.csv", index=False)
    persistence.to_csv(OUT / "edge_persistence.csv", index=False)
    primary_edges = np.asarray(data["primary_edges"])
    prompt_ids = np.asarray(data["prompt_ids"])
    outcomes = np.asarray(data["outcomes"])
    primary = pd.DataFrame(
        {
            "source_index": primary_edges[:, 0],
            "target_index": primary_edges[:, 1],
            "source_prompt_id": prompt_ids[primary_edges[:, 0]],
            "target_prompt_id": prompt_ids[primary_edges[:, 1]],
            "same_outcome": (
                outcomes[primary_edges[:, 0]]
                == outcomes[primary_edges[:, 1]]
            ),
        }
    )
    primary.to_csv(OUT / "primary_graph_edges.csv", index=False)
    payload = {
        "type": "derived visualization of frozen archived results",
        "inputs": {
            str(STATES.relative_to(ROOT)): sha256(STATES),
            str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST),
            str(
                (SOURCE / "fixed_graph_statistics.csv").relative_to(ROOT)
            ): sha256(SOURCE / "fixed_graph_statistics.csv"),
            str(
                (CHECKPOINT / "fixed_graph_statistics.csv").relative_to(ROOT)
            ): sha256(CHECKPOINT / "fixed_graph_statistics.csv"),
            str(
                (CROSS / "fixed_statistics.csv").relative_to(ROOT)
            ): sha256(CROSS / "fixed_statistics.csv"),
        },
        "counts": {
            "nodes": 72,
            "selected_edges_per_graph": 144,
            "registered_layers": 25,
            "layerwise_edge_rows": int(len(layer_edges)),
            "unique_directed_edges_across_layers": int(len(persistence)),
            "same_direction_primary_edges": int(primary["same_outcome"].sum()),
        },
        "guardrail": (
            "Candidate targets are restricted to the supplied mechanism "
            "family. The atlas is an actual graph visualization, but it is "
            "not a mechanism-free or universal materials graph."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Relation-graph publication visualizations",
                "",
                "This directory contains derived visualizations of the frozen "
                "natural-question, positional, and cross-mechanism audits. "
                "No model inference or endpoint selection occurs here.",
                "",
                "## Figures",
                "",
                "- `figures/relation-graph-robustness.*`: revised four-panel "
                "main-paper figure.",
                "- `figures/relation-graph-layer-atlas.*`: all 25 actual "
                "72-node, 144-edge layer graphs.",
                "- `figures/relation-graph-persistence.*`: union graph; line "
                "width and opacity encode how often an edge is selected.",
                "",
                "## Machine-readable visual data",
                "",
                "- `primary_graph_edges.csv`: all edges in the displayed "
                "natural-question band graph.",
                "- `layerwise_selected_edges.csv`: 3,600 selected edge rows "
                "across 25 layers.",
                "- `edge_persistence.csv`: one row per unique directed edge "
                "with its layer frequency.",
                "- `summary.json`: input hashes, counts, and guardrail.",
                "",
                "Nodes are exact prompts. Node shape is surface variant; fill "
                "is the registered positive/negative outcome orientation. "
                "Teal edges preserve the physically correct outcome and coral "
                "edges do not. Six islands appear because the frozen candidate "
                "rule supplies the mechanism family; they are not discovered "
                "communities.",
                "",
                "Regenerate with:",
                "",
                "```bash",
                "python scripts/plot_relation_graph_robustness.py",
                "```",
                "",
            ]
        )
    )


def main() -> None:
    configure()
    data = load_data()
    main_figure(data)
    layer_edges = layer_atlas(data)
    persistence = persistence_figure(data, layer_edges)
    write_artifacts(data, layer_edges, persistence)


if __name__ == "__main__":
    main()
