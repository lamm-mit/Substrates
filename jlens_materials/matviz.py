# Copyright 2026.  Apache-2.0.
"""Static PNG/SVG figures for the Jacobian lens, in the transformer-circuits style.

The repo's own `jlens.vis` renders an *interactive* d3 HTML page.  For papers,
slides and static reports you usually want frozen vector/raster figures.  This
module consumes the same `jlens.vis.SliceData` object and emits matplotlib
figures that mirror the paper's four workhorse figure types:

    slice_grid        layer x position grid of the top-1 lens token, each cell
                      shaded by its full-vocab rank.  (paper Fig 3 / Fig 5
                      "argmax * layer x pos".)

    rank_trajectory   rank-vs-layer line chart for a set of tracked concept
                      tokens at one readout position, log rank axis, rank 1 at
                      the top.  (paper Fig 5 bottom, Fig 17 line chart.)  This
                      is where "intermediates surface in computed order at
                      successively later layers" becomes visible.

    rank_heatmap      position x layer heatmap of ONE token's rank.  (paper
                      Fig 5 rank heatmap, Fig 17 heatmap.)

    hit_rate_bars     grouped bars with Wilson CIs for condition comparisons.
                      (paper Fig 8 / 19 / 20.)

    emergence_depth   lollipop of the first sustained rank-threshold crossing
                      within a fixed workspace band; the best rank is annotated.

Everything is model-agnostic: pass a `SliceData` plus a `LensModel` (only used
for `n_layers` to reindex layers to 0-100 like the paper, and for the
tokenizer to decode).  Nothing here imports torch beyond what SliceData
already carries as numpy.

Design notes
------------
Layers are *reindexed to [0, 100]* exactly as the paper does, so depth reads as
a percentage and figures are comparable across models with different layer
counts.  The rank colour scale is log10(rank): rank 1 is bright, and anything
past ~vocab is dark -- the same "1 -> 10k" ramp the paper uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ---- house style ---------------------------------------------------------- #

# A calm, print-friendly palette in the spirit of transformer-circuits figures.
INK = "#1b1b1b"
MUTE = "#8a8a8a"
GRID = "#e7e7e7"
ACCENT = "#c1553b"   # warm terracotta (the paper's highlight colour)
SERIES = ["#c1553b", "#2b7a8c", "#6a51a3", "#4c8c3f", "#c98a1e",
          "#a03050", "#3a6ea5", "#7a7a7a"]

# rank colormap: rank 1 -> bright gold, deep -> indigo (log scale).  Mirrors
# the paper's 1..10k ramp (bright = the model is "poised" to say this token).
_RANK_CMAP = LinearSegmentedColormap.from_list(
    "jlens_rank",
    ["#fde725", "#7ad151", "#22a884", "#2a788e", "#414487", "#440154"],
)


def _install_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": MUTE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTE,
        "ytick.color": MUTE,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


_install_style()


# ---- helpers -------------------------------------------------------------- #

def reindex_layers(layer_indices, n_layers: int) -> np.ndarray:
    """Map absolute layer indices to the paper's [0, 100] depth scale."""
    denom = max(n_layers - 1, 1)
    return 100.0 * np.asarray(layer_indices, dtype=float) / denom


def _decode(slice_data, token_id: int) -> str:
    s = slice_data.vocab_fragment.get(int(token_id), f"<{token_id}>")
    s = s.replace("\n", "\\n").replace("\t", "\\t")
    return s if s.strip() else repr(s).strip("'")


def _rank_norm(vocab_size: int) -> Normalize:
    top = np.log10(max(vocab_size, 1000))
    return Normalize(vmin=0.0, vmax=top)


def _pos_labels(slice_data, stride: int) -> tuple[np.ndarray, list[str]]:
    offset = slice_data.ctx_offset
    strs = slice_data.context_token_strs[offset:]
    idx = np.arange(0, len(strs), stride)
    labels = [strs[i].replace("\n", "\\n").strip()[:6] or "·" for i in idx]
    return idx, labels


def _save(fig, out_base: str | Path, formats=("png", "svg")) -> list[str]:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = f"{out_base}.{fmt}"
        fig.savefig(path, format=fmt)
        written.append(path)
    plt.close(fig)
    return written


# ---- figure 1: slice grid ------------------------------------------------- #

def plot_slice_grid(
    slice_data,
    model,
    out_base: str | Path,
    *,
    title: str = "",
    subtitle: str = "",
    max_positions: int | None = 40,
    show_token_text: bool = True,
    highlight_positions: tuple[int, ...] = (),
    formats=("png", "svg"),
) -> list[str]:
    """Layer x position grid of the top-1 lens token, shaded by rank.

    This is the signature "read the model's mind across depth" figure.  Rows are
    layers (reindexed to depth %), columns are token positions; each cell shows
    the single word the activation is most disposed to make the model say, tinted
    by how highly that word ranks over the full vocabulary.
    """
    top_ids = slice_data.top_ids[:, :, 0]      # [seq, nlayers]
    top_rank = slice_data.top_ranks[:, :, 0]   # [seq, nlayers]
    seq_len, n_layer_rows = top_ids.shape

    pos_sel = np.arange(seq_len)
    if max_positions is not None and seq_len > max_positions:
        pos_sel = np.arange(seq_len - max_positions, seq_len)
    top_ids = top_ids[pos_sel]
    top_rank = top_rank[pos_sel]

    depth = reindex_layers(slice_data.layers, model.n_layers)
    norm = _rank_norm(slice_data.vocab_size or 50000)
    log_rank = np.log10(top_rank.T + 1.0)      # [nlayers, npos]

    cell_w, cell_h = 0.62, 0.30
    fig_w = max(6.0, len(pos_sel) * cell_w)
    fig_h = max(4.0, n_layer_rows * cell_h)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(log_rank, aspect="auto", cmap=_RANK_CMAP, norm=norm,
              origin="lower", interpolation="nearest")

    if show_token_text:
        for li in range(n_layer_rows):
            for pi in range(len(pos_sel)):
                r = top_rank.T[li, pi]
                txt = _decode(slice_data, top_ids.T[li, pi])[:9]
                lum = 1.0 - norm(np.log10(r + 1.0))
                color = "white" if lum < 0.45 else INK
                ax.text(pi, li, txt, ha="center", va="center",
                        fontsize=6.0, color=color, clip_on=True)

    # y axis: depth %
    ytick = np.linspace(0, n_layer_rows - 1, min(6, n_layer_rows)).astype(int)
    ax.set_yticks(ytick)
    ax.set_yticklabels([f"{depth[i]:.0f}" for i in ytick])
    ax.set_ylabel("layer depth  (reindexed 0-100)")

    # x axis: token strings
    pos_strs = [slice_data.context_token_strs[slice_data.ctx_offset + p]
                for p in pos_sel]
    ax.set_xticks(np.arange(len(pos_sel)))
    ax.set_xticklabels(
        [s.replace("\n", "\\n").strip()[:7] or "·" for s in pos_strs],
        rotation=90, fontsize=6.0,
    )
    ax.set_xlabel("token position ->")

    for hp in highlight_positions:
        rel = hp - (seq_len - len(pos_sel)) if hp >= 0 else hp + len(pos_sel)
        if 0 <= rel < len(pos_sel):
            ax.axvline(rel, color=ACCENT, lw=1.4, alpha=0.8)

    sm = mpl.cm.ScalarMappable(cmap=_RANK_CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("lens rank of top-1 token", fontsize=8)
    cb.set_ticks([0, 1, 2, 3, np.log10(slice_data.vocab_size or 50000)])
    cb.set_ticklabels(["1", "10", "100", "1k", "vocab"])

    if title:
        ax.set_title(title, loc="left", pad=14, fontweight="bold")
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes,
                fontsize=8, color=MUTE, va="bottom")
    return _save(fig, out_base, formats)


# ---- figure 2: rank trajectories ------------------------------------------ #

def _rank_curve(slice_data, cols: int | list[int], position: int | None,
                positions: list[int] | None = None) -> np.ndarray:
    """Rank-per-layer curve, minimized over valid synonyms/score positions.

    position is an int  -> the rank at that single token position.
    position is None     -> the min rank over ALL positions at each layer
                            (the paper's 'hit anywhere in the span' reduction).
    """
    cols = [cols] if isinstance(cols, int) else cols
    ranks = slice_data.rank_tensor[:, :, cols].astype(float)  # [seq,nlayer,nform]
    ranks = np.where(ranks < 0, np.nan, ranks)
    ranks = np.nanmin(ranks, axis=2)                           # [seq,nlayer]
    if positions is not None:
        return np.nanmin(ranks[positions], axis=0)
    if position is None:
        return np.nanmin(ranks, axis=0)                      # [nlayer]
    seq_rel = position if position >= 0 else ranks.shape[0] + position
    return ranks[seq_rel]


def plot_rank_trajectories(
    slice_data,
    model,
    tracked,
    out_base: str | Path,
    *,
    position: int | None = -1,
    positions: list[int] | None = None,
    title: str = "",
    subtitle: str = "",
    band: tuple[float, float] | None = None,
    formats=("png", "svg"),
) -> list[str]:
    """Rank-vs-depth line chart for tracked concept tokens.

    `tracked` is a list of (token_id, label).  Y is full-vocab rank on a log
    scale with rank 1 at the top; the layer at which a line dives toward the top
    is the depth at which that concept enters the workspace.  `position=None`
    plots the best (min) rank over all token positions at each layer -- the
    paper's 'concept appears anywhere in the span' reduction, best for raw-text
    completions where intermediates surface off the final token.
    """
    depth = reindex_layers(slice_data.layers, model.n_layers)
    id_to_col = {int(t): i for i, t in enumerate(slice_data.tracked_token_ids)}

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if band is not None:
        ax.axvspan(band[0], band[1], color=ACCENT, alpha=0.06, lw=0,
                   label="workspace band")

    plotted = 0
    for k, item in enumerate(tracked):
        if hasattr(item, "token_ids"):
            token_ids, label = item.token_ids, item.label
        else:
            tid, label = item
            token_ids = (tid,)
        cols = [id_to_col[int(t)] for t in token_ids if int(t) in id_to_col]
        if not cols:
            continue
        ranks = _rank_curve(slice_data, cols, position, positions) + 1.0
        c = SERIES[k % len(SERIES)]
        ax.plot(depth, ranks, "-o", ms=3.2, lw=1.7, color=c, label=label)
        # annotate the best (min-rank) point
        if np.isfinite(ranks).any():
            bi = int(np.nanargmin(ranks))
            ax.annotate(label, (depth[bi], ranks[bi]), fontsize=7.5,
                        color=c, xytext=(3, -2), textcoords="offset points")
        plotted += 1

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylim(bottom=(slice_data.vocab_size or 50000), top=0.8)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1", "10", "100", "1k", "10k"])
    ax.set_ylabel("lens rank  (1 = top of workspace)")
    ax.set_xlabel("layer depth  (reindexed 0-100)")
    ax.set_xlim(-2, 102)
    ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.axhline(1, color=MUTE, lw=0.6, ls=":")
    if plotted:
        ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9, ncol=2)
    if title:
        ax.set_title(title, loc="left", pad=12, fontweight="bold")
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=8, color=MUTE, va="bottom")
    return _save(fig, out_base, formats)


# ---- figure 3: single-token rank heatmap ---------------------------------- #

def plot_rank_heatmap(
    slice_data,
    model,
    token_id: int,
    label: str,
    out_base: str | Path,
    *,
    title: str = "",
    max_positions: int | None = 60,
    formats=("png", "svg"),
) -> list[str]:
    """Position x layer heatmap of ONE concept token's rank across the prompt."""
    col = {int(t): i for i, t in enumerate(slice_data.tracked_token_ids)}.get(
        int(token_id))
    if col is None:
        raise ValueError(f"token {label!r} not tracked in this slice")
    ranks = slice_data.rank_tensor[:, :, col].astype(float)  # [seq, nlayer]
    ranks = np.where(ranks < 0, np.nan, ranks)
    seq_len = ranks.shape[0]

    pos_sel = np.arange(seq_len)
    if max_positions is not None and seq_len > max_positions:
        pos_sel = np.arange(seq_len - max_positions, seq_len)
    grid = np.log10(ranks[pos_sel].T + 1.0)  # [nlayer, npos]

    depth = reindex_layers(slice_data.layers, model.n_layers)
    norm = _rank_norm(slice_data.vocab_size or 50000)

    fig, ax = plt.subplots(figsize=(max(6.0, len(pos_sel) * 0.16), 3.4))
    im = ax.imshow(grid, aspect="auto", cmap=_RANK_CMAP, norm=norm,
                   origin="lower", interpolation="nearest")
    ytick = np.linspace(0, len(depth) - 1, min(6, len(depth))).astype(int)
    ax.set_yticks(ytick)
    ax.set_yticklabels([f"{depth[i]:.0f}" for i in ytick])
    ax.set_ylabel("depth")
    stride = max(1, len(pos_sel) // 24)
    xt = np.arange(0, len(pos_sel), stride)
    ax.set_xticks(xt)
    ax.set_xticklabels(
        [slice_data.context_token_strs[slice_data.ctx_offset + pos_sel[i]]
         .replace("\n", "\\n").strip()[:5] or "·" for i in xt],
        rotation=90, fontsize=6)
    ax.set_xlabel("token position ->")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(f"rank of “{label}”", fontsize=8)
    cb.set_ticks([0, 1, 2, 3])
    cb.set_ticklabels(["1", "10", "100", "1k"])
    ax.set_title(title or f"Where the lens holds “{label}”", loc="left",
                 fontweight="bold", pad=10)
    return _save(fig, out_base, formats)


# ---- figure 4: emergence-depth lollipop ----------------------------------- #

@dataclass
class Emergence:
    label: str
    onset_depth: float  # first sustained threshold crossing in the fixed band
    best_depth: float   # reindexed layer of min rank
    best_rank: int
    reached_top: bool   # min rank == 0 (rank-1)
    best_pos: int = -1  # token position of the peak (only set when scanning)


def concept_emergence(
    slice_data, model, tracked, *, position: int | None = None,
    positions: list[int] | None = None,
    band: tuple[float, float] | None = None,
    threshold: int = 5,
    sustain: int = 2,
) -> list[Emergence]:
    """Score concepts in a predetermined span and independently fixed band.

    ``best_depth`` is retained for descriptive plots. ``onset_depth`` is the
    first layer where rank is below ``threshold`` for ``sustain`` consecutive
    evaluated layers; unlike the old global peak, this is a genuine onset
    statistic. Synonymous token IDs are minimized within each concept.
    """
    depth = reindex_layers(slice_data.layers, model.n_layers)
    id_to_col = {int(t): i for i, t in enumerate(slice_data.tracked_token_ids)}
    layer_mask = np.ones(len(depth), dtype=bool)
    if band is not None:
        layer_mask = (depth >= band[0]) & (depth <= band[1])
    layer_indices = np.flatnonzero(layer_mask)
    if not len(layer_indices):
        raise ValueError(f"no evaluated layers fall inside workspace band {band}")
    out = []
    for item in tracked:
        if hasattr(item, "token_ids"):
            token_ids, label = item.token_ids, item.label
        else:
            tid, label = item
            token_ids = (tid,)
        cols = [id_to_col[int(t)] for t in token_ids if int(t) in id_to_col]
        if not cols:
            out.append(Emergence(label, np.nan, np.nan, -1, False))
            continue
        ranks = slice_data.rank_tensor[:, :, cols].astype(float)
        ranks = np.where(ranks >= 0, ranks, np.nan)
        ranks = np.nanmin(ranks, axis=2)                    # [seq,nlayer]
        if positions is not None:
            selected_positions = positions
            ranks = ranks[selected_positions]
        elif position is not None:
            seq_rel = position if position >= 0 else ranks.shape[0] + position
            selected_positions = [seq_rel]
            ranks = ranks[seq_rel:seq_rel + 1]
        else:
            selected_positions = list(range(ranks.shape[0]))
        ranks = ranks[:, layer_indices]
        valid = np.isfinite(ranks)
        if not valid.any():
            out.append(Emergence(label, np.nan, np.nan, -1, False))
            continue
        masked = np.where(valid, ranks, np.inf)
        flat = int(np.argmin(masked))
        pos_i, band_layer_i = np.unravel_index(flat, masked.shape)
        layer_i = int(layer_indices[band_layer_i])
        r = int(ranks[pos_i, band_layer_i])
        curve = np.nanmin(ranks, axis=0)
        hit = curve < threshold
        onset = np.nan
        run = max(1, sustain)
        for i in range(0, max(0, len(hit) - run + 1)):
            if bool(np.all(hit[i:i + run])):
                onset = float(depth[layer_indices[i]])
                break
        out.append(Emergence(
            label=label,
            onset_depth=onset,
            best_depth=float(depth[layer_i]),
            best_rank=r,
            reached_top=r == 0,
            best_pos=int(selected_positions[pos_i]),
        ))
    return out


def plot_emergence_depth(
    emergences: list[Emergence],
    out_base: str | Path,
    *,
    title: str = "",
    subtitle: str = "",
    formats=("png", "svg"),
) -> list[str]:
    """Lollipop: sustained onset where available, with peak rank annotated."""
    ems = [e for e in emergences if np.isfinite(e.best_depth)]
    ems = sorted(ems, key=lambda e: (e.onset_depth if np.isfinite(e.onset_depth)
                                    else e.best_depth))
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(ems) + 1.4))
    norm = _rank_norm(50000)
    for y, e in enumerate(ems):
        c = _RANK_CMAP(norm(np.log10(e.best_rank + 1.0)))
        plotted_depth = (e.onset_depth if np.isfinite(e.onset_depth)
                         else e.best_depth)
        ax.plot([0, plotted_depth], [y, y], color=GRID, lw=1.2, zorder=1)
        ax.scatter([plotted_depth], [y], s=130, color=c, zorder=2,
                   edgecolor=INK, linewidth=0.6)
        kind = "onset" if np.isfinite(e.onset_depth) else "peak only"
        tag = f"{e.label}  ({kind}; peak rank {e.best_rank + 1})"
        ax.text(plotted_depth + 1.5, y, tag, va="center", fontsize=8,
                color=INK if e.best_rank < 30 else MUTE)
    ax.set_yticks([])
    ax.set_xlim(-2, 118)
    ax.set_xlabel("sustained emergence depth  (reindexed 0-100)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8,
                color=MUTE, va="bottom")
    return _save(fig, out_base, formats)


# ---- figure 5: hit-rate bars ---------------------------------------------- #

def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def plot_hit_rate_bars(
    conditions: list[str],
    counts: list[tuple[int, int]],   # (hits, trials) per condition
    out_base: str | Path,
    *,
    title: str = "",
    subtitle: str = "",
    ylabel: str = "hit rate  (concept reaches lens top-k)",
    colors: list[str] | None = None,
    formats=("png", "svg"),
) -> list[str]:
    """Grouped bars with Wilson 95% CIs (paper Fig 8/19/20 style)."""
    fig, ax = plt.subplots(figsize=(1.2 * len(conditions) + 2.0, 4.0))
    xs = np.arange(len(conditions))
    colors = colors or [SERIES[i % len(SERIES)] for i in range(len(conditions))]
    for x, (k, n), c in zip(xs, counts, colors):
        p, lo, hi = _wilson(k, n)
        ax.bar(x, p, width=0.62, color=c, alpha=0.9)
        # Wilson interval is centred off the raw proportion, so clamp the
        # distances from the bar height p to be non-negative for errorbar().
        ax.errorbar(x, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]],
                    color=INK, lw=1.0, capsize=4)
        ax.text(x, p + 0.02, f"{p*100:.0f}%\n{k}/{n}", ha="center",
                va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8,
                color=MUTE, va="bottom")
    return _save(fig, out_base, formats)


def plot_pass_at_k_curves(
    ks: list[int],
    series: dict[str, list[float]],
    out_base: str | Path,
    *,
    aucs: dict[str, float] | None = None,
    title: str = "Lens recovery pass@k",
    subtitle: str = "item-level mean; x axis is logarithmic",
    formats=("png", "svg"),
) -> list[str]:
    """Paper-style pass@k curves for J-lens and registered baselines."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    aucs = aucs or {}
    for i, (label, values) in enumerate(series.items()):
        auc = aucs.get(label)
        suffix = f" (AUC={auc:.3f})" if auc is not None and np.isfinite(auc) else ""
        ax.plot(ks, values, "-o", lw=1.8, ms=4,
                color=SERIES[i % len(SERIES)], label=label + suffix)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if series:
        ax.legend(fontsize=8, loc="lower right")
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8,
                color=MUTE, va="bottom")
    return _save(fig, out_base, formats)


def plot_hidden_thoughts(
    items: list[dict],
    out_base: str | Path,
    *,
    title: str = "",
    subtitle: str = "exploratory top-1 candidates — prompt/output filtered "
                     "where an actual completion is available",
    top: int = 12,
    value_key: str = "score",
    formats=("png", "svg"),
) -> list[str]:
    """Horizontal bars of exploratory surfaced candidates.

    ``items`` are the per-prompt ``surprising`` records from ``runs/<tag>.json``
    (``concept`` · ``score`` = band cells · ``best_rank`` · ``near_token``). Bar
    length is the band-cell count; each bar is annotated with its peak rank and
    the nearby prompt token where the concept surfaced. Rank-1 concepts are drawn
    saturated, weaker ranks faded.
    """
    items = sorted(items, key=lambda d: -d.get(value_key, 0))[:top]
    if not items:
        fig, ax = plt.subplots(figsize=(6.6, 1.7))
        ax.axis("off")
        ax.text(0.5, 0.55, "no exploratory candidates surfaced",
                ha="center", va="center", fontsize=12, color=INK)
        ax.text(0.5, 0.28, "all candidates were filtered by prompt/output controls",
                ha="center", va="center", fontsize=9, color=MUTE)
        if title:
            ax.set_title(title, loc="left", fontweight="bold")
        return _save(fig, out_base, formats)

    labels = [d["concept"] for d in items]
    vals = [float(d.get(value_key, 0)) for d in items]
    ranks = [int(d.get("best_rank", 0)) for d in items]
    near = [str(d.get("near_token", "") or "") for d in items]
    y = np.arange(len(items))[::-1]        # strongest pinned to the top
    vmax = max(vals) or 1.0

    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(items) + 1.5))
    for yi, v, r, nt in zip(y, vals, ranks, near):
        # surprising records store best_rank 1-indexed (1 = tops the lens)
        ax.barh(yi, v, height=0.66, color=ACCENT, alpha=(0.92 if r <= 1 else 0.5))
        note = f"r{r}" + (f"  ·  near “{nt}”" if nt else "")
        ax.text(v + vmax * 0.02, yi, note, va="center", ha="left",
                fontsize=7.8, color=MUTE)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("workspace-band cells where the concept tops the lens", fontsize=8)
    ax.set_xlim(0, vmax * 1.5)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8,
                color=MUTE, va="bottom")
    return _save(fig, out_base, formats)


# ---- comparison figures (across runs / models) ---------------------------- #

def plot_grouped_bars(
    categories: list[str],
    series_labels: list[str],
    values: list[list[float]],       # [n_series][n_categories]
    out_base: str | Path,
    *,
    title: str = "",
    subtitle: str = "",
    ylabel: str = "",
    ylim: tuple[float, float] | None = None,
    pct: bool = True,
    formats=("png", "svg"),
) -> list[str]:
    """Grouped bars: one group per category, one bar per series (e.g. model)."""
    n_series, n_cat = len(series_labels), len(categories)
    x = np.arange(n_cat)
    w = 0.8 / max(n_series, 1)
    fig, ax = plt.subplots(figsize=(max(6.0, n_cat * (0.5 + 0.28 * n_series) + 2), 4.2))
    for i, (lab, vals) in enumerate(zip(series_labels, values)):
        off = (i - (n_series - 1) / 2) * w
        c = SERIES[i % len(SERIES)]
        bars = ax.bar(x + off, vals, width=w * 0.92, color=c, label=lab)
        for b, v in zip(bars, vals):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            txt = f"{v*100:.0f}%" if pct else f"{v:g}"
            ax.text(b.get_x() + b.get_width() / 2, v, txt, ha="center",
                    va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    elif pct:
        ax.set_ylim(0, 1.12)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, ncol=min(n_series, 4), loc="upper right",
              framealpha=0.9)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8,
                color=MUTE, va="bottom")
    return _save(fig, out_base, formats)


def _stream_series(layer_readouts, max_concepts=12):
    """From per-layer top tokens, build a salience-vs-depth series per concept.
    Salience at a layer = (K - display_position) if the token is in that layer's
    top-K, else 0 (top-1 -> K, ..., top-K -> 1). Returns (depths, labels,
    matrix[n_concepts, n_depths]) keeping the max_concepts most salient overall,
    ordered by first-emergence depth (a lineage reading)."""
    depths = [float(r["depth"]) for r in layer_readouts]
    K = max((len(r["top_tokens"]) for r in layer_readouts), default=8)
    sal: dict[str, list[float]] = {}
    for li, r in enumerate(layer_readouts):
        for p, t in enumerate(r["top_tokens"]):
            if not t or not t.isascii():  # skip non-renderable (e.g. CJK) tokens
                continue
            sal.setdefault(t, [0.0] * len(layer_readouts))[li] = float(K - p)
    top = sorted(sal.items(), key=lambda kv: sum(kv[1]), reverse=True)[:max_concepts]

    def first(series):
        return next((i for i, v in enumerate(series) if v > 0), len(series))

    top.sort(key=lambda kv: first(kv[1]))
    labels = [t for t, _ in top]
    mat = np.array([s for _, s in top], dtype=float) if top else np.zeros((0, len(depths)))
    return np.array(depths), labels, mat


def plot_concept_stream(
    layer_readouts,
    out_base: str | Path,
    *,
    title: str = "",
    subtitle: str = "",
    max_concepts: int = 12,
    formats=("png", "svg"),
) -> list[str]:
    """Static 'thought stream' (ThemeRiver): each concept the lens surfaces is a
    band flowing left->right across depth, thickness ~ salience — a lineage view
    of the model's evolving disposition to speak."""
    depths, labels, S = _stream_series(layer_readouts, max_concepts)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    if len(labels):
        total = S.sum(0)
        cur = -total / 2.0                      # centered baseline (ThemeRiver)
        for k, lab in enumerate(labels):
            lo, hi = cur.copy(), cur + S[k]
            c = SERIES[k % len(SERIES)]
            ax.fill_between(depths, lo, hi, color=c, alpha=0.85, lw=0.4,
                            edgecolor="white")
            i = int(np.argmax(S[k]))
            if S[k][i] > 0:
                lum = 0.3  # bands are saturated; white labels read well
                ax.text(depths[i], (lo[i] + hi[i]) / 2, lab, ha="center",
                        va="center", fontsize=7.5,
                        color="white" if k % len(SERIES) in (2, 5) else INK)
            cur = hi
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xlabel("layer depth  (reindexed 0-100)  ->  the flow of 'thinking'")
    ax.spines[["top", "right", "left"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8,
                color=MUTE, va="bottom")
    return _save(fig, out_base, formats)


def plot_concept_rank_dots(
    concepts: list[str],
    models: list[str],
    ranks: list[list[float]],        # [n_models][n_concepts]; 1-based, NaN=missing
    out_base: str | Path,
    *,
    title: str = "",
    subtitle: str = "",
    vocab_size: int = 50000,
    formats=("png", "svg"),
) -> list[str]:
    """Per-prompt comparison: each concept's best lens rank as a dot per model
    on a log axis (left = rank 1 = strongest), with a connector between models
    (a dumbbell). Missing values are skipped."""
    ncat = len(concepts)
    y = np.arange(ncat)
    fig, ax = plt.subplots(figsize=(7.0, 0.5 * ncat + 1.6))
    # dumbbell connector per concept
    for j in range(ncat):
        vals = [max(ranks[i][j], 1) for i in range(len(models))
                if ranks[i][j] == ranks[i][j]]
        if len(vals) >= 2:
            ax.plot([min(vals), max(vals)], [y[j], y[j]], color=GRID, lw=2.0,
                    zorder=1)
    for i, m in enumerate(models):
        xx = [max(ranks[i][j], 1) for j in range(ncat) if ranks[i][j] == ranks[i][j]]
        yy = [y[j] for j in range(ncat) if ranks[i][j] == ranks[i][j]]
        ax.scatter(xx, yy, s=95, color=SERIES[i % len(SERIES)], label=m,
                   edgecolor=INK, linewidth=0.5, zorder=3)
        for xv, yv in zip(xx, yy):
            ax.annotate(f"{int(xv)}", (xv, yv), fontsize=6.5, color=MUTE,
                        xytext=(0, 6), textcoords="offset points", ha="center")
    ax.set_xscale("log")
    ax.set_xlim(0.8, max(vocab_size, 1000))
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1k", "10k"])
    ax.set_xlabel("best lens rank  (1 = strongest; left = stronger)")
    ax.set_yticks(y)
    ax.set_yticklabels(concepts, fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # legend outside the axes (right) so it never overlaps the dumbbells;
    # savefig bbox='tight' keeps it in frame
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5),
              framealpha=0.9, borderaxespad=0)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8,
                color=MUTE, va="bottom")
    return _save(fig, out_base, formats)


def plot_rank_matrix(
    row_labels: list[str],
    col_labels: list[str],
    ranks: list[list[float]],        # [n_rows][n_cols]; 1-based rank, NaN=missing
    out_base: str | Path,
    *,
    title: str = "",
    subtitle: str = "",
    vocab_size: int = 50000,
    formats=("png", "svg"),
) -> list[str]:
    """Heatmap of a concept's best lens rank across models (rows=concepts,
    cols=models). Bright = high rank (concept strongly present); grey = the
    concept wasn't tracked / didn't appear for that model."""
    R = np.array(ranks, dtype=float)
    logv = np.ma.masked_invalid(np.log10(R))
    norm = _rank_norm(vocab_size)
    cmap = _RANK_CMAP.copy()
    cmap.set_bad("#ededed")
    fig, ax = plt.subplots(figsize=(max(4.5, len(col_labels) * 1.5 + 2.5),
                                    0.42 * len(row_labels) + 1.8))
    ax.imshow(logv, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            r = R[i, j]
            if np.isnan(r):
                continue
            lum = 1.0 - norm(np.log10(r))
            ax.text(j, i, f"{int(r)}", ha="center", va="center", fontsize=7,
                    color="white" if lum < 0.45 else INK)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("best lens rank", fontsize=8)
    cb.set_ticks([0, 1, 2, 3]); cb.set_ticklabels(["1", "10", "100", "1k"])
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8,
                color=MUTE, va="bottom")
    return _save(fig, out_base, formats)
