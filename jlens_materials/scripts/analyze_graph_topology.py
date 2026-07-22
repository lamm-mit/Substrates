#!/usr/bin/env python3
"""Analyze label-free graph topology in held-out materials representations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "graph-topology-2026-07-17"
FIG = OUT / "figures"
PROTOCOL_PATH = OUT / "protocol.json"
NPZ_PATH = ROOT / "experiments" / "materials-heldout-v1_latent_vectors.npz"
STATS_PATH = ROOT / "experiments" / "materials-heldout-v1_statistics.json"
META_PATH = ROOT / "experiments" / "materials-heldout-v1_latent_vectors.meta.json"
SEED = 20260717
N_PERM = 10_000
PRIMARY_K = 4
BAND = (38.0, 92.0)

FAMILY_SHORT = {
    "boundary-attack": "BA",
    "cleavage": "CL",
    "cyclic": "CY",
    "ductile": "DU",
    "high-temperature-deformation": "HT",
    "hot-air-surface-layer": "OX",
    "line-defect-motion": "LD",
    "notch-resistance": "NR",
    "particle-strengthening": "PS",
    "rapid-transformation": "RT",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def token_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z'-]{2,}", text.lower()))


def candidate_removed(token: str, prompt_words: set[str]) -> bool:
    if token in prompt_words:
        return True
    return any(
        min(len(token), len(word)) >= 5
        and (token.startswith(word) or word.startswith(token))
        for word in prompt_words
    )


def validate_inputs(protocol: dict) -> None:
    paths = {
        NPZ_PATH: protocol["inputs"]["latent_vectors_sha256"],
        STATS_PATH: protocol["inputs"]["heldout_statistics_sha256"],
        META_PATH: protocol["inputs"]["latent_metadata_sha256"],
    }
    for path, expected in paths.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"fingerprint mismatch for {path}: {actual} != {expected}")


def knn_graph(vectors: np.ndarray, k: int, *, mutual: bool = False) -> tuple[nx.Graph, np.ndarray]:
    vectors = normalize(vectors)
    # Apple Accelerate can emit stale floating-point status warnings for a
    # finite BLAS result. Suppress those warnings locally and retain an
    # explicit finite-value gate so a genuine numerical failure aborts.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        similarity = np.einsum("id,jd->ij", vectors, vectors, optimize=True)
    if not np.all(np.isfinite(similarity)):
        raise FloatingPointError("non-finite cosine similarity in graph construction")
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    directed = {(i, int(j)) for i, row in enumerate(neighbors) for j in row}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(vectors)))
    for i, j in directed:
        if mutual and (j, i) not in directed:
            continue
        a, b = sorted((i, j))
        graph.add_edge(a, b, weight=float(max(similarity[a, b], 0.0) + 1e-9))
    return graph, similarity


def edge_homophily(graph: nx.Graph, labels: np.ndarray) -> float:
    edges = list(graph.edges())
    if not edges:
        return float("nan")
    return float(np.mean([labels[a] == labels[b] for a, b in edges]))


def community_labels(graph: nx.Graph) -> np.ndarray:
    communities = nx.algorithms.community.louvain_communities(
        graph, weight="weight", resolution=1.0, seed=SEED
    )
    labels = np.full(graph.number_of_nodes(), -1, dtype=int)
    for index, community in enumerate(communities):
        labels[list(community)] = index
    if np.any(labels < 0):
        raise RuntimeError("Louvain left an unassigned node")
    return labels


def graph_metrics(vectors: np.ndarray, families: np.ndarray, k: int) -> dict:
    graph, similarity = knn_graph(vectors, k)
    communities = community_labels(graph)
    return {
        "graph": graph,
        "similarity": similarity,
        "homophily": edge_homophily(graph, families),
        "ari": float(adjusted_rand_score(families, communities)),
        "communities": communities,
        "n_communities": int(len(set(communities))),
        "n_edges": int(graph.number_of_edges()),
        "mean_degree": float(np.mean([degree for _, degree in graph.degree()])),
    }


def permuted_labels(families: np.ndarray, folds: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    output = np.empty((N_PERM, len(families)), dtype=families.dtype)
    for iteration in range(N_PERM):
        labels = families.copy()
        for fold in sorted(set(folds)):
            indices = np.flatnonzero(folds == fold)
            labels[indices] = rng.permutation(labels[indices])
        output[iteration] = labels
    return output


def homophily_null(graphs: list[nx.Graph], permutations: np.ndarray) -> np.ndarray:
    output = np.empty((len(permutations), len(graphs)), dtype=float)
    edge_arrays = [np.asarray(list(graph.edges()), dtype=int) for graph in graphs]
    for layer, edges in enumerate(edge_arrays):
        output[:, layer] = np.mean(
            permutations[:, edges[:, 0]] == permutations[:, edges[:, 1]], axis=1
        )
    return output


def ari_null(partitions: list[np.ndarray], permutations: np.ndarray) -> np.ndarray:
    output = np.empty((len(permutations), len(partitions)), dtype=float)
    for iteration, labels in enumerate(permutations):
        for layer, partition in enumerate(partitions):
            output[iteration, layer] = adjusted_rand_score(labels, partition)
    return output


def p_plus_one(null: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(null >= observed - 1e-15)) / (1 + len(null)))


def vocabulary_features(records: list[dict], *, prompt_words: bool = False) -> tuple[np.ndarray, list[str]]:
    documents: list[dict[str, float]] = []
    for record in records:
        words = token_words(record["prompt"])
        if prompt_words:
            document = {word: 1.0 for word in words}
        else:
            filtered = [
                row for row in record["filtered_consensus_candidates"]
                if not candidate_removed(str(row["token"]), words)
            ][:20]
            document = {
                str(row["token"]): float(row["consensus_score"])
                for row in filtered
            }
        documents.append(document)
    frequency = Counter(token for document in documents for token in document)
    vocabulary = sorted(frequency)
    lookup = {token: index for index, token in enumerate(vocabulary)}
    values = np.zeros((len(documents), len(vocabulary)), dtype=float)
    for row, document in enumerate(documents):
        for token, score in document.items():
            idf = math.log((1 + len(documents)) / (1 + frequency[token]))
            values[row, lookup[token]] = score * idf
    return normalize(values), vocabulary


def write_layer_rows(
    depths: np.ndarray,
    source_layers: np.ndarray,
    j_metrics: list[dict],
    raw_metrics: list[dict],
) -> pd.DataFrame:
    rows = []
    for index, depth in enumerate(depths):
        for method, metrics in (("Jacobian", j_metrics), ("raw", raw_metrics)):
            row = metrics[index]
            rows.append({
                "method": method,
                "source_layer": int(source_layers[index]),
                "depth": float(depth),
                "homophily": row["homophily"],
                "louvain_ari": row["ari"],
                "n_communities": row["n_communities"],
                "n_edges": row["n_edges"],
                "mean_degree": row["mean_degree"],
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "layer_graph_metrics.csv", index=False)
    return frame


def family_color_map(families: np.ndarray) -> dict[str, tuple]:
    palette = plt.get_cmap("tab10")
    return {family: palette(index) for index, family in enumerate(sorted(set(families)))}


def draw_prompt_graph(
    axis: plt.Axes,
    graph: nx.Graph,
    families: np.ndarray,
    folds: np.ndarray,
    colors: dict[str, tuple],
    title: str,
    seed: int,
) -> None:
    positions = nx.spring_layout(graph, seed=seed, weight="weight", k=0.62, iterations=300)
    same = [(a, b) for a, b in graph.edges() if families[a] == families[b]]
    other = [(a, b) for a, b in graph.edges() if families[a] != families[b]]
    nx.draw_networkx_edges(graph, positions, edgelist=other, edge_color="#b8bcc2", width=0.55, alpha=0.45, ax=axis)
    nx.draw_networkx_edges(graph, positions, edgelist=same, edge_color="#3f6f78", width=1.15, alpha=0.75, ax=axis)
    for fold, marker in zip(sorted(set(folds)), ["o", "s", "^", "D", "P"]):
        nodes = np.flatnonzero(folds == fold).tolist()
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=nodes,
            node_color=[colors[str(families[node])] for node in nodes],
            node_shape=marker,
            node_size=48,
            linewidths=0.45,
            edgecolors="white",
            ax=axis,
        )
    axis.set_title(title, fontsize=9, loc="left")
    axis.set_axis_off()


def plot_results(
    depths: np.ndarray,
    families: np.ndarray,
    folds: np.ndarray,
    j_metrics: list[dict],
    raw_metrics: list[dict],
    fixed: dict[str, dict],
    best_index: int,
) -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = family_color_map(families)
    teal = "#2a788e"
    purple = "#6f5aa8"
    gray = "#73777d"
    orange = "#d07c39"
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.1))

    axis = axes[0, 0]
    axis.plot(depths, [row["homophily"] for row in j_metrics], color=teal, lw=2.1, label="Jacobian states")
    axis.plot(depths, [row["homophily"] for row in raw_metrics], color=purple, lw=1.8, label="raw states")
    axis.axhline(fixed["lexical"]["homophily"], color=orange, lw=1.4, ls="--", label="prompt words")
    axis.axhline(fixed["target"]["homophily"], color=gray, lw=1.3, ls=":", label="target state")
    axis.axvspan(BAND[0], BAND[1], color="#d9dde2", alpha=0.35, zorder=-2)
    axis.axhline(4 / 49, color="#a7abb1", lw=0.9, ls=(0, (2, 2)), label="balanced chance")
    axis.set(xlabel="layer depth (%)", ylabel="same-family edge fraction", ylim=(0, 0.66))
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.text(-0.12, 1.04, "A", transform=axis.transAxes, fontweight="bold", fontsize=10)

    axis = axes[0, 1]
    axis.plot(depths, [row["ari"] for row in j_metrics], color=teal, lw=2.1, label="Jacobian states")
    axis.plot(depths, [row["ari"] for row in raw_metrics], color=purple, lw=1.8, label="raw states")
    axis.axhline(fixed["lexical"]["ari"], color=orange, lw=1.4, ls="--", label="prompt words")
    axis.axhline(fixed["target"]["ari"], color=gray, lw=1.3, ls=":", label="target state")
    axis.axvspan(BAND[0], BAND[1], color="#d9dde2", alpha=0.35, zorder=-2)
    axis.axhline(0, color="#a7abb1", lw=0.8)
    axis.set(xlabel="layer depth (%)", ylabel="label-free community ARI", ylim=(-0.08, 0.72))
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.text(-0.12, 1.04, "B", transform=axis.transAxes, fontweight="bold", fontsize=10)

    draw_prompt_graph(
        axes[1, 0],
        j_metrics[best_index]["graph"],
        families,
        folds,
        colors,
        f"Best Jacobian layer ({depths[best_index]:.1f}% depth)",
        SEED,
    )
    axes[1, 0].text(-0.12, 1.04, "C", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=10)

    draw_prompt_graph(
        axes[1, 1],
        fixed["vocabulary_jacobian"]["graph"],
        families,
        folds,
        colors,
        "Target-free discovered-word graph",
        SEED + 1,
    )
    axes[1, 1].text(-0.12, 1.04, "D", transform=axes[1, 1].transAxes, fontweight="bold", fontsize=10)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[family],
                   markeredgecolor="none", markersize=5, label=FAMILY_SHORT.get(family, family[:2].upper()))
        for family in sorted(colors)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=10, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.075, hspace=0.28, wspace=0.22)
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIG / f"materials-graph-topology.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    validate_inputs(protocol)
    with np.load(NPZ_PATH, allow_pickle=False) as data:
        transported = normalize(data["transported_states"].astype(np.float64))
        transported_mean = normalize(np.mean(transported, axis=0))
        raw = normalize(data["raw_states"].astype(np.float64))
        lexical = normalize(data["lexical_states"].astype(np.float64))
        target = normalize(data["target_states"].astype(np.float64))
        families = data["families"].astype(str)
        depths = data["depths"].astype(float)
        source_layers = data["source_layers"].astype(int)
        phrasing_ids = data["phrasing_ids"].astype(str)
    folds = np.asarray([int(value.rsplit("-", 1)[-1]) for value in phrasing_ids])
    permutations = permuted_labels(families, folds)
    band_mask = (depths >= BAND[0]) & (depths <= BAND[1])

    j_metrics = [graph_metrics(transported_mean[:, layer], families, PRIMARY_K) for layer in range(len(depths))]
    raw_metrics = [graph_metrics(raw[:, layer], families, PRIMARY_K) for layer in range(len(depths))]
    lexical_metrics = graph_metrics(lexical, families, PRIMARY_K)
    target_metrics = graph_metrics(target, families, PRIMARY_K)

    heldout = json.loads(STATS_PATH.read_text())
    jac_records = heldout["open_vocabulary"]["methods"]["jacobian"]["per_prompt"]
    direct_records = heldout["open_vocabulary"]["methods"]["logit"]["per_prompt"]
    if [record["slug"] for record in jac_records] != [record["slug"] for record in direct_records]:
        raise RuntimeError("candidate method prompt order differs")
    if [record["family"] for record in jac_records] != families.tolist():
        raise RuntimeError("candidate and latent-vector family order differs")
    vocabulary_j, _ = vocabulary_features(jac_records)
    vocabulary_d, _ = vocabulary_features(direct_records)
    prompt_words, _ = vocabulary_features(jac_records, prompt_words=True)
    vocabulary_metrics = {
        "vocabulary_jacobian": graph_metrics(vocabulary_j, families, PRIMARY_K),
        "vocabulary_direct": graph_metrics(vocabulary_d, families, PRIMARY_K),
        "prompt_words": graph_metrics(prompt_words, families, PRIMARY_K),
    }

    j_graphs = [row["graph"] for row in j_metrics]
    raw_graphs = [row["graph"] for row in raw_metrics]
    j_h_null = homophily_null(j_graphs, permutations)
    raw_h_null = homophily_null(raw_graphs, permutations)
    j_ari_null = ari_null([row["communities"] for row in j_metrics], permutations)
    raw_ari_null = ari_null([row["communities"] for row in raw_metrics], permutations)
    j_h = np.asarray([row["homophily"] for row in j_metrics])
    raw_h = np.asarray([row["homophily"] for row in raw_metrics])
    j_ari = np.asarray([row["ari"] for row in j_metrics])
    raw_ari = np.asarray([row["ari"] for row in raw_metrics])

    fixed = {
        "lexical": lexical_metrics,
        "target": target_metrics,
        **vocabulary_metrics,
    }
    fixed_nulls = {}
    for name, metrics in fixed.items():
        graph = metrics["graph"]
        edges = np.asarray(list(graph.edges()), dtype=int)
        h_null = np.mean(permutations[:, edges[:, 0]] == permutations[:, edges[:, 1]], axis=1)
        a_null = np.asarray([
            adjusted_rand_score(labels, metrics["communities"])
            for labels in permutations
        ])
        fixed_nulls[name] = {"homophily": h_null, "ari": a_null}

    def layer_summary(
        observed_h: np.ndarray,
        observed_ari: np.ndarray,
        null_h: np.ndarray,
        null_ari: np.ndarray,
    ) -> dict:
        best_h = int(np.argmax(observed_h))
        best_ari = int(np.argmax(observed_ari))
        return {
            "band_mean_homophily": float(np.mean(observed_h[band_mask])),
            "band_mean_homophily_p": p_plus_one(np.mean(null_h[:, band_mask], axis=1), float(np.mean(observed_h[band_mask]))),
            "best_homophily": float(observed_h[best_h]),
            "best_homophily_layer": int(source_layers[best_h]),
            "best_homophily_depth": float(depths[best_h]),
            "max_layer_corrected_homophily_p": p_plus_one(np.max(null_h, axis=1), float(observed_h[best_h])),
            "best_louvain_ari": float(observed_ari[best_ari]),
            "best_ari_layer": int(source_layers[best_ari]),
            "best_ari_depth": float(depths[best_ari]),
            "max_layer_corrected_ari_p": p_plus_one(np.max(null_ari, axis=1), float(observed_ari[best_ari])),
        }

    sensitivity = {}
    for k in (3, 4, 5, 6):
        j_k = [graph_metrics(transported_mean[:, layer], families, k)["homophily"] for layer in range(len(depths))]
        raw_k = [graph_metrics(raw[:, layer], families, k)["homophily"] for layer in range(len(depths))]
        sensitivity[str(k)] = {
            "jacobian_band_mean_homophily": float(np.mean(np.asarray(j_k)[band_mask])),
            "raw_band_mean_homophily": float(np.mean(np.asarray(raw_k)[band_mask])),
            "jacobian_best_homophily": float(np.max(j_k)),
            "raw_best_homophily": float(np.max(raw_k)),
        }

    fixed_results = {}
    for name, metrics in fixed.items():
        fixed_results[name] = {
            "homophily": metrics["homophily"],
            "homophily_p": p_plus_one(fixed_nulls[name]["homophily"], metrics["homophily"]),
            "louvain_ari": metrics["ari"],
            "louvain_ari_p": p_plus_one(fixed_nulls[name]["ari"], metrics["ari"]),
            "n_communities": metrics["n_communities"],
            "n_edges": metrics["n_edges"],
        }

    payload = {
        "status": protocol["status"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "input_sha256": protocol["inputs"],
        "n_permutations": N_PERM,
        "primary_k": PRIMARY_K,
        "registered_band_percent": list(BAND),
        "jacobian_states": layer_summary(j_h, j_ari, j_h_null, j_ari_null),
        "raw_states": layer_summary(raw_h, raw_ari, raw_h_null, raw_ari_null),
        "jacobian_minus_raw": {
            "band_mean_homophily_difference": float(np.mean(j_h[band_mask] - raw_h[band_mask])),
            "descriptive_only": True,
        },
        "fixed_graphs": fixed_results,
        "k_sensitivity": sensitivity,
        "interpretation": {
            "supports_family_graph_structure_if": "Jacobian homophily and community ARI exceed the corrected label-permutation null",
            "supports_jacobian_specificity_if": "Jacobian topology is materially stronger than raw-state and prompt-word baselines",
            "causal_claim": False,
        },
    }
    (OUT / "statistics.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_layer_rows(depths, source_layers, j_metrics, raw_metrics)
    pd.DataFrame([
        {"graph": name, **values}
        for name, values in fixed_results.items()
    ]).to_csv(OUT / "fixed_graph_metrics.csv", index=False)

    best_index = int(np.argmax(j_h))
    plot_results(depths, families, folds, j_metrics, raw_metrics, fixed, best_index)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
