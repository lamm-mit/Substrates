#!/usr/bin/env python3
"""Create publication figures for the graph-generalization audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from analyze_graph_isomorphism_generalization import load_data
from run_graph_isomorphism_gin import selected_family_graph


ROOT = Path(__file__).resolve().parents[1]
EXP = (
    ROOT
    / "experiments"
    / "graph-isomorphism-generalization-2026-07-18"
)
FIG = EXP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PAPER_FIG = ROOT / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)

FAMILY_LABELS = {
    "crosslink-density-modulus": "Crosslink",
    "dislocation-density-strength": "Dislocation",
    "obstacle-spacing-orowan": "Orowan",
    "particle-fraction-modulus": "Particles",
    "pearlite-spacing-strength": "Pearlite",
    "porosity-modulus": "Porosity",
}
FAMILY_ORDER = list(FAMILY_LABELS)
COLORS = {
    "jacobian": "#3B5BA9",
    "direct": "#2A9D8F",
    "raw": "#7768AE",
    "mlp": "#7A8793",
    "shuffle": "#B6BDC5",
    "numeric": "#9D4EDD",
    "positive": "#2A9D8F",
    "negative": "#7768AE",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "grid.color": "#D8DCE2",
            "grid.linewidth": 0.5,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def ordered(frame: pd.DataFrame, column: str) -> np.ndarray:
    return (
        frame.set_index("heldout_family")
        .reindex(FAMILY_ORDER)[column]
        .to_numpy(float)
    )


def figure_model_tests() -> None:
    absolute = pd.read_csv(EXP / "gin_family_metrics.csv")
    relation = pd.read_csv(EXP / "relation_gin_family_metrics.csv")
    nonparam = pd.read_csv(EXP / "relation_nonparametric.csv")
    x = np.arange(len(FAMILY_ORDER))
    labels = [FAMILY_LABELS[item] for item in FAMILY_ORDER]
    fig, axes = plt.subplots(
        1, 3, figsize=(11.8, 3.35), constrained_layout=True
    )

    absolute_specs = [
        ("topology_gin_physical", "GIN: physical", COLORS["jacobian"], "o"),
        ("topology_mlp_physical", "No graph: physical", COLORS["mlp"], "s"),
        (
            "topology_gin_edge_shuffle_physical",
            "Shuffled graph",
            COLORS["shuffle"],
            "^",
        ),
        ("topology_gin_numeric", "GIN: numeric", COLORS["numeric"], "D"),
    ]
    for config, label, color, marker in absolute_specs:
        frame = absolute[absolute["configuration"] == config]
        axes[0].plot(
            x,
            ordered(frame, "auc"),
            marker=marker,
            markersize=4,
            linewidth=1.25,
            color=color,
            label=label,
        )
    axes[0].axhline(0.5, color="#61666D", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Held-out-mechanism ROC–AUC")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(axis="y")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        frameon=False,
        ncol=2,
    )
    panel_label(axes[0], "A")

    relation_specs = [
        (
            "relation_jacobian_gin_physical",
            "Relation GIN",
            COLORS["jacobian"],
            "o",
        ),
        (
            "relation_jacobian_mlp_physical",
            "Pair MLP",
            COLORS["mlp"],
            "s",
        ),
        (
            "relation_jacobian_gin_edge_shuffle_physical",
            "Shuffled graph",
            COLORS["shuffle"],
            "^",
        ),
    ]
    for config, label, color, marker in relation_specs:
        frame = relation[relation["configuration"] == config]
        axes[1].plot(
            x,
            ordered(frame, "auc"),
            marker=marker,
            markersize=4,
            linewidth=1.25,
            color=color,
            label=label,
        )
    axes[1].axhline(0.5, color="#61666D", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylim(0.30, 0.70)
    axes[1].grid(axis="y")
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        frameon=False,
        ncol=3,
    )
    panel_label(axes[1], "B")

    for method, color, marker in (
        ("jacobian", COLORS["jacobian"], "o"),
        ("direct", COLORS["direct"], "s"),
        ("raw", COLORS["raw"], "^"),
    ):
        frame = (
            nonparam[nonparam["method"] == method]
            .set_index("family")
            .reindex(FAMILY_ORDER)
        )
        axes[2].plot(
            x,
            frame["physical_auc"],
            marker=marker,
            markersize=4,
            linewidth=1.25,
            color=color,
            label=method.capitalize(),
        )
    axes[2].axhline(0.5, color="#61666D", linewidth=0.8, linestyle="--")
    axes[2].set_xticks(x, labels, rotation=35, ha="right")
    axes[2].set_ylim(0.48, 0.78)
    axes[2].grid(axis="y")
    axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        frameon=False,
        ncol=3,
    )
    panel_label(axes[2], "C")
    fig.savefig(FIG / "graph-generalization-model-tests.png")
    fig.savefig(FIG / "graph-generalization-model-tests.pdf")
    plt.close(fig)


def figure_partition_tests() -> None:
    spectral = pd.read_csv(EXP / "spectral_community_metrics.csv")
    exact = pd.read_csv(EXP / "exact_partition_metrics.csv")
    nulls = np.load(EXP / "exact_partition_nulls.npz")
    mode_order = [
        "binary_top1",
        "weighted_top1",
        "weighted_top2",
        "weighted_complete_candidates",
    ]
    mode_labels = ["Top-1\nbinary", "Top-1\nweighted", "Top-2\nweighted", "All\nweighted"]
    fig, axes = plt.subplots(
        1, 3, figsize=(11.8, 3.45), constrained_layout=True
    )

    jac = spectral[spectral["method"] == "jacobian"]
    for family in FAMILY_ORDER:
        values = (
            jac[jac["family"] == family]
            .set_index("mode")
            .reindex(mode_order)["ari"]
            .to_numpy(float)
        )
        axes[0].plot(
            np.arange(4),
            values,
            color="#AEB7C3",
            linewidth=0.8,
            marker="o",
            markersize=2.8,
            alpha=0.85,
        )
    mean_values = (
        jac.groupby("mode")["ari"].mean().reindex(mode_order).to_numpy()
    )
    axes[0].plot(
        np.arange(4),
        mean_values,
        color=COLORS["jacobian"],
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Six-mechanism mean",
    )
    axes[0].axhline(0, color="#61666D", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(np.arange(4), mode_labels)
    axes[0].set_ylabel("Adjusted Rand index")
    axes[0].set_ylim(-0.18, 1.08)
    axes[0].grid(axis="y")
    axes[0].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
    )
    panel_label(axes[0], "A")

    width = 0.24
    x = np.arange(len(FAMILY_ORDER))
    for offset, (method, color) in enumerate(
        [
            ("jacobian", COLORS["jacobian"]),
            ("direct", COLORS["direct"]),
            ("raw", COLORS["raw"]),
        ]
    ):
        frame = (
            spectral[
                (spectral["method"] == method)
                & (spectral["mode"] == "weighted_top1")
            ]
            .set_index("family")
            .reindex(FAMILY_ORDER)
        )
        axes[1].bar(
            x + (offset - 1) * width,
            frame["ari"],
            width=width,
            color=color,
            label=method.capitalize(),
        )
    axes[1].axhline(0, color="#61666D", linewidth=0.8)
    axes[1].set_xticks(
        x, [FAMILY_LABELS[item] for item in FAMILY_ORDER], rotation=35, ha="right"
    )
    axes[1].set_ylim(-0.18, 1.08)
    axes[1].grid(axis="y")
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=3,
    )
    panel_label(axes[1], "B")

    endpoint_order = ["complete", "heldout_surface"]
    endpoint_labels = ["Exact full\ngraph", "Counterfactual\nheld out"]
    rng = np.random.default_rng(20260718)
    for method_index, (method, color) in enumerate(
        [
            ("jacobian", COLORS["jacobian"]),
            ("direct", COLORS["direct"]),
            ("raw", COLORS["raw"]),
        ]
    ):
        for endpoint_index, endpoint in enumerate(endpoint_order):
            frame = exact[
                (exact["method"] == method)
                & (exact["endpoint"] == endpoint)
            ]
            center = endpoint_index + (method_index - 1) * 0.18
            mean = float(frame["ari"].mean())
            axes[2].bar(
                center,
                mean,
                width=0.16,
                color=color,
                alpha=0.78,
                label=method.capitalize() if endpoint_index == 0 else None,
            )
            jitter = rng.uniform(-0.045, 0.045, size=len(frame))
            axes[2].scatter(
                np.full(len(frame), center) + jitter,
                frame["ari"],
                s=12,
                facecolor="white",
                edgecolor=color,
                linewidth=0.7,
                zorder=3,
            )
            key = (
                f"{method}_complete_family_ari"
                if endpoint == "complete"
                else f"{method}_heldout_family_ari"
            )
            null_mean = np.asarray(nulls[key]).mean(axis=1)
            axes[2].errorbar(
                center,
                float(np.mean(null_mean)),
                yerr=float(np.std(null_mean, ddof=1)),
                fmt="_",
                color="#35393E",
                linewidth=1,
                capsize=2,
                zorder=4,
            )
    axes[2].axhline(0, color="#61666D", linewidth=0.8)
    axes[2].set_xticks(np.arange(2), endpoint_labels)
    axes[2].set_ylim(-0.18, 0.12)
    axes[2].grid(axis="y")
    axes[2].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=3,
    )
    panel_label(axes[2], "C")
    fig.savefig(FIG / "graph-generalization-partition-tests.png")
    fig.savefig(FIG / "graph-generalization-partition-tests.pdf")
    plt.close(fig)


def figure_graph_examples() -> None:
    data = load_data()
    similarity = np.asarray(data["similarities"]["jacobian"])
    band = np.mean(similarity[np.asarray(data["band_mask"])], axis=0)
    examples = [
        "particle-fraction-modulus",
        "crosslink-density-modulus",
        "obstacle-spacing-orowan",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8))
    fig.subplots_adjust(
        left=0.035, right=0.995, top=0.94, bottom=0.18, wspace=0.13
    )
    variant_x = {
        "anchor": 0.0,
        "physics_paraphrase": 1.0,
        "lexical_counterfactual": 2.0,
    }
    shapes = {
        "anchor": "o",
        "physics_paraphrase": "s",
        "lexical_counterfactual": "D",
    }
    for panel_index, (axis, family) in enumerate(zip(axes, examples)):
        graph = selected_family_graph(band, family, data)
        cases = sorted({attributes["case"] for _, attributes in graph.nodes(data=True)})
        case_y = {case: 3 - index for index, case in enumerate(cases)}
        positions = {
            node: (
                variant_x[attributes["variant"]],
                case_y[attributes["case"]],
            )
            for node, attributes in graph.nodes(data=True)
        }
        weights = np.asarray(
            [attributes["weight"] for *_, attributes in graph.edges(data=True)]
        )
        widths = 0.4 + 1.4 * (
            (weights - weights.min()) / max(1e-9, np.ptp(weights))
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edge_color="#8D959E",
            alpha=0.50,
            width=widths,
            arrows=True,
            arrowsize=7,
            connectionstyle="arc3,rad=0.05",
            node_size=220,
        )
        for variant, marker in shapes.items():
            nodes = [
                node
                for node, attributes in graph.nodes(data=True)
                if attributes["variant"] == variant
            ]
            node_colors = [
                COLORS["positive"]
                if graph.nodes[node]["physical"]
                else COLORS["negative"]
                for node in nodes
            ]
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=nodes,
                node_color=node_colors,
                node_shape=marker,
                node_size=185,
                edgecolors="white",
                linewidths=0.8,
                ax=axis,
            )
        labels = {
            node: str(index + 1)
            for index, node in enumerate(sorted(graph.nodes))
        }
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            font_size=6.5,
            font_color="white",
            ax=axis,
        )
        axis.set_xlim(-0.35, 2.35)
        axis.set_ylim(-0.55, 3.55)
        axis.set_xticks(
            [0, 1, 2], ["Anchor", "Paraphrase", "Counterfactual"]
        )
        axis.tick_params(axis="x", length=0)
        axis.set_yticks([])
        axis.set_xlabel("")
        axis.spines[:].set_visible(False)
        panel_label(axis, chr(ord("A") + panel_index))
    legend_handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["positive"],
            markeredgecolor="none",
            markersize=6,
            label="Positive direction",
        ),
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["negative"],
            markeredgecolor="none",
            markersize=6,
            label="Negative direction",
        ),
        mpl.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="#61666D",
            markersize=6,
            label="Anchor",
        ),
        mpl.lines.Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            color="#61666D",
            markersize=6,
            label="Paraphrase",
        ),
        mpl.lines.Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            color="#61666D",
            markersize=6,
            label="Counterfactual",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
    )
    fig.savefig(FIG / "graph-generalization-network-examples.png")
    fig.savefig(FIG / "graph-generalization-network-examples.pdf")
    plt.close(fig)


def figure_identifiability_summary() -> None:
    absolute = pd.read_csv(EXP / "gin_family_metrics.csv")
    relation = pd.read_csv(EXP / "relation_gin_family_metrics.csv")
    density = pd.read_csv(EXP / "spectral_density_ablations.csv")
    exact = pd.read_csv(EXP / "exact_partition_metrics.csv")
    nulls = np.load(EXP / "exact_partition_nulls.npz")

    fig, axes = plt.subplots(
        2, 2, figsize=(10.8, 7.1), constrained_layout=True
    )

    # A: the exact direct/inverse-law identifiability relation.
    axis = axes[0, 0]
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    box = dict(boxstyle="round,pad=0.45", edgecolor="#8D959E", linewidth=0.9)
    axis.text(
        0.25,
        0.84,
        "Direct law  $s=+1$",
        ha="center",
        va="center",
        color="#24313A",
        bbox={**box, "facecolor": "#E6F3F1"},
    )
    axis.text(
        0.75,
        0.84,
        "Inverse law  $s=-1$",
        ha="center",
        va="center",
        color="#24313A",
        bbox={**box, "facecolor": "#EEEAF7"},
    )
    axis.text(
        0.25,
        0.62,
        "dislocation density $\\uparrow$\nyield strength $\\uparrow$",
        ha="center",
        va="center",
        color="#24313A",
    )
    axis.text(
        0.25,
        0.38,
        "particle fraction $\\uparrow$\nmodulus $\\uparrow$",
        ha="center",
        va="center",
        color="#24313A",
    )
    axis.text(
        0.75,
        0.62,
        "porosity $\\uparrow$\nmodulus $\\downarrow$",
        ha="center",
        va="center",
        color="#24313A",
    )
    axis.text(
        0.75,
        0.38,
        "obstacle spacing $\\uparrow$\nbypass stress $\\downarrow$",
        ha="center",
        va="center",
        color="#24313A",
    )
    axis.text(
        0.5,
        0.18,
        r"$y_{fi}=s_f x_{fi}$"
        "\n"
        r"within one mechanism:  $y_{fi}y_{fj}=x_{fi}x_{fj}$",
        ha="center",
        va="center",
        color="#24313A",
        bbox={**box, "facecolor": "#F4F5F6"},
    )
    panel_label(axis, "A")

    # B: whole-mechanism generalization.
    axis = axes[0, 1]
    x = np.arange(len(FAMILY_ORDER))
    labels = [FAMILY_LABELS[item] for item in FAMILY_ORDER]
    curves = [
        (
            absolute[absolute["configuration"] == "topology_gin_numeric"],
            "Input-direction check ($x$ supplied)",
            COLORS["numeric"],
            "D",
        ),
        (
            absolute[absolute["configuration"] == "topology_gin_physical"],
            "Physical polarity",
            COLORS["jacobian"],
            "o",
        ),
        (
            relation[
                relation["configuration"]
                == "relation_jacobian_gin_physical"
            ],
            "Same/different relation",
            COLORS["direct"],
            "s",
        ),
    ]
    for frame, label, color, marker in curves:
        axis.plot(
            x,
            ordered(frame, "auc"),
            color=color,
            marker=marker,
            markersize=4.2,
            linewidth=1.4,
            label=label,
        )
    axis.axhline(0.5, color="#61666D", linewidth=0.8, linestyle="--")
    axis.set_ylabel("Held-out-mechanism ROC–AUC")
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(axis="y")
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=1.0,
        handlelength=1.8,
        fontsize=7.0,
        borderaxespad=0,
    )
    panel_label(axis, "B")

    # C: graph-density ablation.
    axis = axes[1, 0]
    mode_order = [
        "binary_top1",
        "weighted_top1",
        "weighted_top2",
        "weighted_complete_candidates",
    ]
    mode_labels = [
        "Top-1\nbinary",
        "Top-1\nweighted",
        "Top-2\nweighted",
        "All\nweighted",
    ]
    for method, color, marker in (
        ("jacobian", COLORS["jacobian"], "o"),
        ("direct", COLORS["direct"], "s"),
        ("raw", COLORS["raw"], "^"),
    ):
        frame = density[density["method"] == method].set_index("mode")
        values = frame.reindex(mode_order)["mean_ari"].to_numpy(float)
        axis.plot(
            np.arange(4),
            values,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.5,
            label=method.capitalize(),
        )
    axis.axhline(0, color="#61666D", linewidth=0.8, linestyle="--")
    axis.set_xticks(np.arange(4), mode_labels)
    axis.set_ylabel("Community agreement (adjusted Rand index)")
    axis.set_ylim(-0.12, 1.02)
    axis.grid(axis="y")
    axis.legend(frameon=False, loc="upper left", ncol=3)
    panel_label(axis, "C")

    # D: exact balanced partitions and held-out counterfactual surface.
    axis = axes[1, 1]
    endpoint_order = ["complete", "heldout_surface"]
    endpoint_labels = ["All 12 nodes", "Counterfactual\nheld out"]
    width = 0.20
    rng = np.random.default_rng(20260718)
    for method_index, (method, color) in enumerate(
        [
            ("jacobian", COLORS["jacobian"]),
            ("direct", COLORS["direct"]),
            ("raw", COLORS["raw"]),
        ]
    ):
        for endpoint_index, endpoint in enumerate(endpoint_order):
            frame = exact[
                (exact["method"] == method) & (exact["endpoint"] == endpoint)
            ]
            center = endpoint_index + (method_index - 1) * width
            axis.bar(
                center,
                float(frame["ari"].mean()),
                width=0.17,
                color=color,
                alpha=0.82,
                label=method.capitalize() if endpoint_index == 0 else None,
            )
            jitter = rng.uniform(-0.035, 0.035, size=len(frame))
            axis.scatter(
                np.full(len(frame), center) + jitter,
                frame["ari"],
                s=15,
                facecolor="white",
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
            key = (
                f"{method}_complete_family_ari"
                if endpoint == "complete"
                else f"{method}_heldout_family_ari"
            )
            null_mean = np.asarray(nulls[key]).mean(axis=1)
            axis.errorbar(
                center,
                float(np.mean(null_mean)),
                yerr=float(np.std(null_mean, ddof=1)),
                fmt="_",
                color="#35393E",
                linewidth=1,
                capsize=2,
                zorder=4,
            )
    axis.axhline(0, color="#61666D", linewidth=0.8)
    axis.set_xticks(np.arange(2), endpoint_labels)
    axis.set_ylabel("Exact-partition agreement")
    axis.set_ylim(-0.15, 0.09)
    axis.grid(axis="y")
    axis.legend(frameon=False, loc="lower right", ncol=3)
    panel_label(axis, "D")

    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"graph-identifiability-summary.{suffix}")
        fig.savefig(PAPER_FIG / f"graph-identifiability-summary.{suffix}")
    plt.close(fig)


def main() -> None:
    style()
    figure_model_tests()
    figure_partition_tests()
    figure_graph_examples()
    figure_identifiability_summary()
    for stem in (
        "graph-generalization-model-tests",
        "graph-generalization-partition-tests",
        "graph-generalization-network-examples",
    ):
        for suffix in ("png", "pdf"):
            source = FIG / f"{stem}.{suffix}"
            target = PAPER_FIG / f"{stem}.{suffix}"
            target.write_bytes(source.read_bytes())
    inventory = {
        "figures": [
            "graph-generalization-model-tests",
            "graph-generalization-partition-tests",
            "graph-generalization-network-examples",
            "graph-identifiability-summary",
        ],
        "formats": ["png", "pdf"],
    }
    (FIG / "figure_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
