#!/usr/bin/env python3
"""Analyze and plot the frozen held-out latent-geometry extraction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.manifold import trustworthiness

import umap


ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = ROOT / "experiments" / "materials-heldout-v1_latent_vectors.npz"
META_PATH = ROOT / "experiments" / "materials-heldout-v1_latent_vectors.meta.json"
PROTOCOL_PATH = ROOT / "experiments" / "materials-heldout-v1-latent-geometry-protocol.md"
STATS_PATH = ROOT / "experiments" / "materials-heldout-v1_latent_geometry_statistics.json"
COORDS_PATH = ROOT / "experiments" / "materials-heldout-v1_latent_geometry_coordinates.csv"
REPORT_PATH = ROOT / "experiments" / "MATERIALS_HELDOUT_V1_LATENT_GEOMETRY.md"
FIG_DIR = ROOT / "figures" / "materials-heldout-v1"
FIG_STEM = FIG_DIR / "figure5-latent-geometry"
PCA_STEM = FIG_DIR / "latent-geometry-pca-sensitivity"
RNG_SEED = 20260715
N_PERM = 5_000
UMAP_SEEDS = [RNG_SEED, 0, 1, 2, 3, 4]
FAMILY_LABELS = {
    "boundary-attack": "boundary attack",
    "cleavage": "cleavage",
    "cyclic": "cyclic damage",
    "ductile": "ductile failure",
    "high-temperature-deformation": "high-T deformation",
    "hot-air-surface-layer": "surface oxidation",
    "line-defect-motion": "line-defect motion",
    "notch-resistance": "notch resistance",
    "particle-strengthening": "particle strengthening",
    "rapid-transformation": "rapid transformation",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(array: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(denom, 1e-12)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"{label}  {title}", loc="left", fontweight="bold", pad=8)


def fold_numbers(phrasing_ids: np.ndarray) -> np.ndarray:
    values = []
    for value in phrasing_ids:
        match = re.search(r"-(\d+)$", str(value))
        if match is None:
            raise ValueError(f"cannot resolve phrasing fold from {value!r}")
        values.append(int(match.group(1)))
    return np.asarray(values, dtype=int)


def nearest_centroid_accuracy(
    vectors: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> tuple[float, np.ndarray]:
    classes = np.unique(labels)
    predictions = np.empty(len(labels), dtype=labels.dtype)
    for fold in np.unique(folds):
        train = folds != fold
        test = ~train
        centroids = []
        for label in classes:
            rows = vectors[train & (labels == label)]
            if len(rows) == 0:
                raise ValueError(f"permutation produced no training rows for {label}")
            centroids.append(normalize(rows.mean(axis=0, keepdims=True))[0])
        # Use float64 for all inferential dot products.  The archived vectors
        # are float16 for compactness; on Apple Accelerate, float32 BLAS can
        # inherit spurious floating-point status flags during long repeated
        # products even when the returned values are finite.
        centroid_array = np.asarray(centroids, dtype=np.float64)
        similarities = vectors[test] @ centroid_array.T
        predictions[test] = classes[np.argmax(similarities, axis=1)]
    return float(np.mean(predictions == labels)), predictions


def nearest_centroid_accuracy_safe(
    vectors: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> tuple[float, np.ndarray]:
    """Small explicit-dot implementation for Apple BLAS-sensitive re-rendering."""
    classes = np.unique(labels)
    predictions = np.empty(len(labels), dtype=labels.dtype)
    for fold in np.unique(folds):
        train = folds != fold
        test_indices = np.flatnonzero(folds == fold)
        centroids = np.asarray([
            normalize(vectors[train & (labels == label)].mean(axis=0, keepdims=True))[0]
            for label in classes
        ], dtype=np.float64)
        for index in test_indices:
            similarities = np.asarray([
                float(np.sum(vectors[index] * centroid)) for centroid in centroids
            ])
            predictions[index] = classes[int(np.argmax(similarities))]
    return float(np.mean(predictions == labels)), predictions


def distance_ratio(vectors: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    distances = 1.0 - np.clip(vectors @ vectors.T, -1.0, 1.0)
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), 1)
    within = float(np.mean(distances[upper & same]))
    between = float(np.mean(distances[upper & ~same]))
    return between / max(within, 1e-12), within, between


def word_alignment(
    vectors: np.ndarray,
    families: np.ndarray,
    word_vectors: np.ndarray,
    word_families: np.ndarray,
    word_methods: np.ndarray,
) -> tuple[float, np.ndarray]:
    keep = word_methods == "jacobian"
    words = word_vectors[keep]
    labels = word_families[keep]
    similarities = vectors @ words.T
    margins = np.empty(len(vectors), dtype=float)
    for index, family in enumerate(families):
        own = similarities[index, labels == family]
        other = similarities[index, labels != family]
        margins[index] = float(np.max(own) - np.max(other))
    return float(np.mean(margins)), margins


def seed_spread(transported: np.ndarray) -> np.ndarray:
    values = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        values.append(1.0 - np.sum(transported[first] * transported[second], axis=-1))
    return np.mean(np.asarray(values), axis=(0, 1))


def permutation_null(
    vectors_by_layer: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(RNG_SEED)
    observed = np.asarray(
        [nearest_centroid_accuracy(vectors_by_layer[:, layer], labels, folds)[0] for layer in range(vectors_by_layer.shape[1])]
    )
    maxima = np.empty(N_PERM, dtype=float)
    unique_folds = np.unique(folds)
    for permutation in range(N_PERM):
        shuffled = labels.copy()
        for fold in unique_folds:
            indices = np.flatnonzero(folds == fold)
            shuffled[indices] = rng.permutation(shuffled[indices])
        maxima[permutation] = max(
            nearest_centroid_accuracy(vectors_by_layer[:, layer], shuffled, folds)[0]
            for layer in range(vectors_by_layer.shape[1])
        )
    observed_max = float(np.max(observed))
    p_value = float((1 + np.sum(maxima >= observed_max - 1e-12)) / (N_PERM + 1))
    return maxima, observed_max, p_value


def fit_projections(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict], PCA]:
    """Fit a display-only UMAP and a linear PCA sensitivity view.

    The display is restricted to the best registered layer.  A joint UMAP of
    all 25 layers was generated during development but was dominated by layer
    progression and obscured the between-family question the projection was
    intended to illustrate.  This choice does not affect any inferential test.
    """
    components = min(25, vectors.shape[0] - 1, vectors.shape[1])
    pca50 = PCA(n_components=components, svd_solver="full", random_state=RNG_SEED)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")
        reduced = pca50.fit_transform(vectors)
    reduced = normalize(reduced)
    embeddings = []
    diagnostics = []
    main = None
    for seed in UMAP_SEEDS:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=10,
            min_dist=0.20,
            metric="cosine",
            random_state=seed,
            transform_seed=seed,
            n_jobs=1,
        )
        embedding = reducer.fit_transform(reduced)
        value = float(trustworthiness(reduced, embedding, n_neighbors=5, metric="cosine"))
        if main is None:
            main = embedding
            disparity = 0.0
        else:
            _, _, disparity = procrustes(main, embedding)
        diagnostics.append({"seed": int(seed), "trustworthiness_5": value, "procrustes_disparity_to_main": float(disparity)})
        embeddings.append(embedding)
    pca2 = PCA(n_components=2, svd_solver="full", random_state=RNG_SEED).fit_transform(vectors)
    return embeddings[0], pca2, diagnostics, pca50


def family_colors(families: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    return {family: mpl.colors.to_hex(cmap(index)) for index, family in enumerate(families)}


def plot_geometry(
    coordinates: np.ndarray,
    pca_coordinates: np.ndarray,
    transported_mean: np.ndarray,
    families: np.ndarray,
    folds: np.ndarray,
    best_predictions: np.ndarray,
    depths: np.ndarray,
    accuracy_j: np.ndarray,
    accuracy_raw: np.ndarray,
    lexical_accuracy: float,
    target_accuracy: float,
    null_95: float,
    permutation_p: float,
    ratios: np.ndarray,
    spreads: np.ndarray,
    best_layer: int,
) -> None:
    unique_families = sorted(set(families.tolist()))
    colors = family_colors(unique_families)
    fig = plt.figure(figsize=(14.0, 12.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.43, 1.0, 1.0])
    ax_protocol = fig.add_subplot(grid[0, :])
    ax0 = fig.add_subplot(grid[1, 0])
    ax1 = fig.add_subplot(grid[1, 1])
    ax2 = fig.add_subplot(grid[2, 0])
    ax3 = fig.add_subplot(grid[2, 1])

    ax_protocol.set_xlim(0, 1)
    ax_protocol.set_ylim(0, 1)
    ax_protocol.axis("off")
    ax_protocol.text(0.0, 0.98, "A", ha="left", va="top", fontsize=12, fontweight="bold")
    protocol_boxes = [
        (0.035, "50 descriptions\n10 families x 5 phrasings"),
        (0.285, "Train 10 centroids\n4 phrasings per family"),
        (0.535, "Classify the held-out\n5th phrasing"),
        (0.785, "Repeat all 5 folds\nat each of 25 layers"),
    ]
    box_width = 0.18
    box_y = 0.37
    box_height = 0.42
    for x_value, label in protocol_boxes:
        ax_protocol.add_patch(
            FancyBboxPatch(
                (x_value, box_y), box_width, box_height,
                boxstyle="round,pad=0.012,rounding_size=0.015",
                facecolor="#f4f7f8", edgecolor="#7c8b92", linewidth=1.1,
            )
        )
        ax_protocol.text(
            x_value + box_width / 2, box_y + box_height / 2, label,
            ha="center", va="center", fontsize=9.3,
        )
    for first, second in zip(protocol_boxes[:-1], protocol_boxes[1:], strict=True):
        ax_protocol.add_patch(
            FancyArrowPatch(
                (first[0] + box_width + 0.008, box_y + box_height / 2),
                (second[0] - 0.008, box_y + box_height / 2),
                arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#626c72",
            )
        )
    ax_protocol.text(
        0.5, 0.13,
        "The same nearest-centroid test is applied to Jacobian, raw-state, target-state, and prompt-word vectors; shuffled family labels define the corrected null.",
        ha="center", va="center", fontsize=8.8, color="#596168",
    )

    panel_label(ax0, "B", "")
    ax0.plot(depths, accuracy_j, color="#087F8C", marker="o", markersize=3.5, linewidth=2.1, label="Jacobian transported (3-fit mean)")
    ax0.plot(depths, accuracy_raw, color="#66727A", marker="s", markersize=3.0, linewidth=1.7, label="raw hidden state")
    ax0.axhline(0.10, color="#777777", linestyle="--", linewidth=1.0, label="10-label chance")
    ax0.axhline(null_95, color="#333333", linestyle="-.", linewidth=1.0, label=f"95% max-layer null ({null_95:.0%})")
    ax0.axvline(depths[best_layer], color="#087F8C", alpha=0.16, linewidth=7)
    ax0.text(
        depths[best_layer] + 3.5, 0.73,
        f"31/50 at {depths[best_layer]:.1f}% depth\ncorrected p={permutation_p:.4f}",
        fontsize=8.8, va="top", color="#30383d",
    )
    ax0.set_xlim(float(depths[0]), float(depths[-1]))
    ax0.set_ylim(0, 0.82)
    ax0.set_xlabel("normalized source-layer depth")
    ax0.set_ylabel("leave-one-phrasing-out accuracy")
    ax0.legend(loc="upper left", frameon=False)

    panel_label(ax1, "C", "")
    markers = ["o", "s", "^", "D", "P"]
    short = ["BA", "CL", "CY", "DU", "HT", "OX", "LD", "NR", "PS", "RT"]
    short_codes = dict(zip(unique_families, short, strict=True))
    for family in unique_families:
        indices = np.flatnonzero(families == family)
        for index in indices:
            marker = markers[int(folds[index] - 1) % len(markers)]
            ax1.scatter(coordinates[index, 0], coordinates[index, 1], s=50, marker=marker, color=colors[family], edgecolor="white", linewidth=0.45, alpha=0.90)
        centroid = np.mean(coordinates[indices], axis=0)
        ax1.scatter(centroid[0], centroid[1], s=95, marker="*", color=colors[family], edgecolor="#30353a", linewidth=0.4)
        ax1.text(centroid[0] + 0.10, centroid[1] + 0.04, short_codes[family], fontsize=7.5, fontweight="bold", color="#30353a")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.grid(False)

    panel_label(ax2, "D", "")
    matrix = confusion_matrix(families, best_predictions, labels=unique_families, normalize="true")
    image = ax2.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax2.set_xticks(range(len(short)), short)
    ax2.set_yticks(range(len(short)), short)
    ax2.set_xlabel("predicted family")
    ax2.set_ylabel("observed family")
    for row in range(len(short)):
        for col in range(len(short)):
            if matrix[row, col] >= 0.2:
                ax2.text(col, row, f"{matrix[row, col]:.1f}", ha="center", va="center", fontsize=7.2, color="white" if matrix[row, col] > 0.55 else "#30353a")
    fig.colorbar(image, ax=ax2, fraction=0.046, pad=0.03)

    panel_label(ax3, "E", "")
    labels = ["chance", "corrected null", "target layer", "raw hidden", "Jacobian transport", "prompt-word embedding"]
    values = [0.10, null_95, target_accuracy, accuracy_raw[best_layer], accuracy_j[best_layer], lexical_accuracy]
    bar_colors = ["#B8BEC3", "#737C83", "#8172A8", "#66727A", "#087F8C", "#4F86A6"]
    ypos = np.arange(len(values))
    ax3.barh(ypos, values, color=bar_colors, alpha=0.88)
    ax3.set_yticks(ypos, labels)
    ax3.set_xlim(0, 0.90)
    ax3.set_xlabel("leave-one-phrasing-out accuracy")
    for y, value in zip(ypos, values, strict=True):
        ax3.text(value + 0.015, y, f"{value:.0%}", va="center", fontsize=8.6)
    save_all(fig, FIG_STEM)

    fig_pca, ax = plt.subplots(figsize=(8.5, 6.8))
    for family in unique_families:
        indices = np.flatnonzero(families == family)
        ax.scatter(pca_coordinates[indices, 0], pca_coordinates[indices, 1], color=colors[family], s=36, alpha=0.85, label=FAMILY_LABELS[family])
    ax.set_title("PCA sensitivity view at the best registered layer", fontweight="bold")
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.legend(ncol=2, frameon=False, fontsize=8)
    save_all(fig_pca, PCA_STEM)


def write_coordinates(
    path: Path,
    coordinates: np.ndarray,
    families: np.ndarray,
    slugs: np.ndarray,
    phrasing_ids: np.ndarray,
    source_layer: int,
    depth: float,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slug", "family", "phrasing_id", "source_layer", "depth", "umap_x", "umap_y"])
        writer.writeheader()
        for index, slug in enumerate(slugs):
            writer.writerow({"slug": slug, "family": families[index], "phrasing_id": phrasing_ids[index], "source_layer": source_layer, "depth": depth, "umap_x": float(coordinates[index, 0]), "umap_y": float(coordinates[index, 1])})


def plot_archived_geometry() -> None:
    """Re-render Figure 5 without refitting UMAP or rerunning permutations."""
    data = np.load(NPZ_PATH, allow_pickle=False)
    stats = json.loads(STATS_PATH.read_text())
    families = data["families"].astype(str)
    folds = fold_numbers(data["phrasing_ids"].astype(str))
    depths = data["depths"].astype(float)
    source_layers = data["source_layers"].astype(int)
    transported = normalize(data["transported_states"].astype(np.float64))
    transported_mean = normalize(np.mean(transported, axis=0))
    raw = normalize(data["raw_states"].astype(np.float64))
    controlled = stats["classification"]
    accuracy_j = np.asarray(controlled["jacobian_mean"]["accuracy_by_layer"], dtype=float)
    accuracy_raw = np.asarray(controlled["raw"]["accuracy_by_layer"], dtype=float)
    best_layer = int(np.flatnonzero(source_layers == int(controlled["jacobian_mean"]["best_layer"]))[0])
    _, best_predictions = nearest_centroid_accuracy_safe(
        transported_mean[:, best_layer], families, folds,
    )
    coordinate_rows = list(csv.DictReader(COORDS_PATH.open()))
    coordinate_index = {
        row["slug"]: (float(row["umap_x"]), float(row["umap_y"]))
        for row in coordinate_rows
    }
    coordinates = np.asarray([coordinate_index[str(slug)] for slug in data["slugs"]], dtype=float)
    pca_coordinates = PCA(n_components=2, svd_solver="full", random_state=RNG_SEED).fit_transform(
        transported_mean[:, best_layer]
    )
    plot_geometry(
        coordinates,
        pca_coordinates,
        transported_mean,
        families,
        folds,
        best_predictions,
        depths,
        accuracy_j,
        accuracy_raw,
        float(controlled["lexical"]["accuracy"]),
        float(controlled["target"]["accuracy"]),
        float(controlled["jacobian_mean"]["null_95"]),
        float(controlled["jacobian_mean"]["max_layer_permutation_p"]),
        np.asarray(stats["geometry"]["between_within_ratio_by_layer"], dtype=float),
        np.asarray(stats["seed_variability"]["by_layer"], dtype=float),
        best_layer,
    )


def write_report(stats: dict) -> None:
    controlled = stats["classification"]
    text = f"""# Exploratory held-out latent geometry

## What was projected

The analysis extracted the contextual residual at the frozen final-prompt
position for all 50 held-out descriptions and 25 registered source layers. Each
residual was transported through all three independently fitted Jacobian maps.
For a legible display, the seed-mean states at the quantitatively best registered
layer were embedded together; the choice of display does not alter the
full-dimensional test.

UMAP is not used for inference. Classification, distances, word alignment, and
permutation tests use the original 2,560-dimensional normalized vectors.

## Quantitative result

- Best leave-one-phrasing-out Jacobian accuracy: **{controlled['jacobian_mean']['max_accuracy']:.1%}** at {controlled['jacobian_mean']['best_depth']:.1f}% depth.
- Best raw-residual accuracy: **{controlled['raw']['max_accuracy']:.1%}**.
- Mean-input-embedding baseline: **{controlled['lexical']['accuracy']:.1%}**.
- Target-layer accuracy: **{controlled['target']['accuracy']:.1%}**.
- Max-over-layer balanced-label permutation p-value: **{controlled['jacobian_mean']['max_layer_permutation_p']:.4g}**.
- The 95th percentile of the permutation maximum was **{controlled['jacobian_mean']['null_95']:.1%}**.
- Mean pairwise distance between transported lens fits was **{stats['seed_variability']['mean_pairwise_cosine_distance']:.3e}**.

The main-text decision specified before extraction was **{str(stats['main_text_decision']['include']).lower()}**: {stats['main_text_decision']['reason']}.

## Projection checks

The main UMAP 5-neighbor trustworthiness was {stats['projection']['umap_runs'][0]['trustworthiness_5']:.3f}. The complete seed diagnostics and PCA sensitivity figure are retained with the results.

## Interpretation boundary

Any family organization shows geometric structure in contextual
representations. It does not reveal a literal hidden reasoning path or establish
that the decoded words causally determine the answer.

## Artifacts

- Statistics: `materials-heldout-v1_latent_geometry_statistics.json`
- Coordinates: `materials-heldout-v1_latent_geometry_coordinates.csv`
- Main figure: `../figures/materials-heldout-v1/figure5-latent-geometry.pdf`
- PCA sensitivity: `../figures/materials-heldout-v1/latent-geometry-pca-sensitivity.pdf`
- Vector metadata: `materials-heldout-v1_latent_vectors.meta.json`
- Frozen protocol: `materials-heldout-v1-latent-geometry-protocol.md`
"""
    REPORT_PATH.write_text(text)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")
    configure_style()
    data = np.load(NPZ_PATH)
    meta = json.loads(META_PATH.read_text())
    if meta["output_npz_sha256"] != sha256(NPZ_PATH):
        raise ValueError("latent vector NPZ hash differs from extraction metadata")
    if meta["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise ValueError("latent geometry protocol changed after extraction")

    # Promote the compact float16 archive to float64 before normalization.
    # This makes every reported distance, margin, centroid, and permutation
    # statistic independent of half-precision and platform BLAS behavior.
    raw = normalize(data["raw_states"].astype(np.float64))
    transported = normalize(data["transported_states"].astype(np.float64))
    target = normalize(data["target_states"].astype(np.float64))
    lexical = normalize(data["lexical_states"].astype(np.float64))
    word_vectors = normalize(data["word_vectors"].astype(np.float64))
    families = data["families"].astype(str)
    phrasing_ids = data["phrasing_ids"].astype(str)
    folds = fold_numbers(phrasing_ids)
    depths = data["depths"].astype(float)
    words = data["word_tokens"].astype(str)
    word_families = data["word_families"].astype(str)
    word_methods = data["word_methods"].astype(str)
    slugs = data["slugs"].astype(str)

    transported_mean = normalize(np.mean(transported, axis=0))
    accuracy_j = np.asarray([nearest_centroid_accuracy(transported_mean[:, layer], families, folds)[0] for layer in range(len(depths))])
    accuracy_raw = np.asarray([nearest_centroid_accuracy(raw[:, layer], families, folds)[0] for layer in range(len(depths))])
    accuracy_seeds = np.asarray([[nearest_centroid_accuracy(transported[seed, :, layer], families, folds)[0] for layer in range(len(depths))] for seed in range(3)])
    lexical_accuracy = nearest_centroid_accuracy(lexical, families, folds)[0]
    target_accuracy = nearest_centroid_accuracy(target, families, folds)[0]

    ratios = []
    within = []
    between = []
    alignments = []
    for layer in range(len(depths)):
        ratio, w_value, b_value = distance_ratio(transported_mean[:, layer], families)
        ratios.append(ratio)
        within.append(w_value)
        between.append(b_value)
        alignments.append(word_alignment(transported_mean[:, layer], families, word_vectors, word_families, word_methods)[0])
    ratios = np.asarray(ratios)
    alignments = np.asarray(alignments)
    spreads = seed_spread(transported)

    # Apple Accelerate can emit stale floating-point status warnings from BLAS
    # even when every result is finite.  Silence only those warnings, and retain
    # explicit finite-value checks below so a genuine numerical failure aborts.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")
        maxima, observed_max, permutation_p = permutation_null(transported_mean, families, folds)
    null_95 = float(np.quantile(maxima, 0.95))

    best_j = int(np.argmax(accuracy_j))
    best_vectors = transported_mean[:, best_j]
    best_accuracy, best_predictions = nearest_centroid_accuracy(best_vectors, families, folds)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")
        coordinates, pca_coordinates, projection_diagnostics, pca50 = fit_projections(best_vectors)
    if not all(np.all(np.isfinite(array)) for array in (accuracy_j, accuracy_raw, ratios, alignments, spreads, maxima, coordinates, pca_coordinates)):
        raise FloatingPointError("latent-geometry analysis produced a non-finite value")
    main_text_supported = bool(permutation_p < 0.05 and np.median([row["trustworthiness_5"] for row in projection_diagnostics]) >= 0.90)
    reason = (
        "full-dimensional classification exceeded the max-over-layer permutation null and UMAP neighborhood trustworthiness was stable"
        if main_text_supported
        else "the frozen full-dimensional significance and projection-stability gate was not satisfied"
    )

    best_raw = int(np.argmax(accuracy_raw))
    stats = {
        "analysis_status": "post hoc exploratory geometry; protocol frozen before vector extraction",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "vector_npz_sha256": sha256(NPZ_PATH),
        "analysis_seed": RNG_SEED,
        "n_prompts": len(families),
        "n_families": len(np.unique(families)),
        "n_layers": len(depths),
        "classification": {
            "method": "leave-one-phrasing-out nearest centroid; cosine similarity; five balanced folds",
            "jacobian_mean": {
                "accuracy_by_layer": accuracy_j.tolist(),
                "max_accuracy": observed_max,
                "best_layer": int(data["source_layers"][best_j]),
                "best_depth": float(depths[best_j]),
                "null_95": null_95,
                "max_layer_permutation_p": permutation_p,
                "n_permutations": N_PERM,
                "permutation_rule": "independently shuffle ten balanced labels within each phrasing fold; retain maximum over 25 layers",
            },
            "jacobian_by_seed": [
                {"seed": seed, "accuracy_by_layer": accuracy_seeds[seed].tolist(), "max_accuracy": float(np.max(accuracy_seeds[seed]))}
                for seed in range(3)
            ],
            "raw": {
                "accuracy_by_layer": accuracy_raw.tolist(),
                "max_accuracy": float(np.max(accuracy_raw)),
                "best_layer": int(data["source_layers"][best_raw]),
                "best_depth": float(depths[best_raw]),
            },
            "lexical": {"accuracy": lexical_accuracy},
            "target": {"accuracy": target_accuracy},
            "chance": 0.1,
        },
        "geometry": {
            "between_within_ratio_by_layer": ratios.tolist(),
            "within_family_cosine_distance_by_layer": within,
            "between_family_cosine_distance_by_layer": between,
            "max_ratio": float(np.max(ratios)),
            "max_ratio_depth": float(depths[int(np.argmax(ratios))]),
            "own_family_word_margin_by_layer": alignments.tolist(),
            "max_word_margin": float(np.max(alignments)),
            "max_word_margin_depth": float(depths[int(np.argmax(alignments))]),
        },
        "seed_variability": {
            "mean_pairwise_cosine_distance": float(np.mean(spreads)),
            "max_pairwise_cosine_distance": float(np.max(spreads)),
            "by_layer": spreads.tolist(),
        },
        "projection": {
            "scope": "50 seed-mean Jacobian-transported prompt states at the best registered layer; display only",
            "input_rows": len(best_vectors),
            "pca_components": int(pca50.n_components_),
            "pca_explained_variance": float(np.sum(pca50.explained_variance_ratio_)),
            "umap_parameters": {"metric": "cosine", "n_neighbors": 10, "min_dist": 0.20, "main_seed": RNG_SEED},
            "umap_runs": projection_diagnostics,
            "inference_uses_projection": False,
        },
        "main_text_decision": {"include": main_text_supported, "reason": reason},
        "limitations": [
            "geometry endpoint was designed after the held-out vocabulary results were inspected",
            "UMAP preserves local neighborhoods but not global axes or distances",
            "discovered-word alignment is descriptive because the word sets came from the same prompt suite",
            "classification tests representation geometry, not causal use or conscious exploration",
        ],
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    write_coordinates(COORDS_PATH, coordinates, families, slugs, phrasing_ids, int(data["source_layers"][best_j]), float(depths[best_j]))
    plot_geometry(coordinates, pca_coordinates, transported_mean, families, folds, best_predictions, depths, accuracy_j, accuracy_raw, lexical_accuracy, target_accuracy, null_95, permutation_p, ratios, spreads, best_j)
    write_report(stats)
    print(json.dumps({"best_accuracy": observed_max, "best_depth": float(depths[best_j]), "permutation_p": permutation_p, "null_95": null_95, "lexical_accuracy": lexical_accuracy, "target_accuracy": target_accuracy, "main_text": main_text_supported}, indent=2))


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        configure_style()
        plot_archived_geometry()
    else:
        main()
