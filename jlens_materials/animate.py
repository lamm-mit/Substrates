# Copyright 2026.  Apache-2.0.
"""Animate the dynamics of "thinking": a bar-race of the concepts the Jacobian
lens surfaces as the readout sweeps through model depth.

Depth is time. At each depth the lens is disposed to make the model say some
ranked set of tokens; as depth increases those concepts rise, fall, and give way
to others. This renders that as a movie (GIF always; MP4 if ffmpeg is present) —
the animated companion to the static "thought stream" figure in the report.

    python animate.py --run runs/qwen2forcausallm.json                  # all prompts -> GIFs
    python animate.py --run runs/qwen2forcausallm.json --slug fracture-fatigue --mp4
    python animate.py --run runs/qwen2forcausallm.json --fps 20 --top-n 12

Uses only ``layer_readouts`` from the run JSON (no model needed). Output goes to
``figures/animation/<slug>__thoughts.{gif,mp4}``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless; render frames to buffer
import numpy as np               # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

import matviz                    # noqa: E402  (installs house style; SERIES/INK/ACCENT)

HERE = Path(__file__).resolve().parent


def make_movie(rec: dict, outdir: Path, *, fps: int = 18, top_n: int = 10,
               substeps: int = 7, mp4: bool = False,
               max_concepts: int = 16) -> list[str]:
    depths, labels, S = matviz._stream_series(rec["layer_readouts"],
                                              max_concepts=max_concepts)
    if not len(labels):
        print(f"  [{rec['slug']}] no surfaced concepts to animate")
        return []
    # interpolate onto a fine depth axis for smooth motion
    fine = np.linspace(float(depths.min()), float(depths.max()),
                       max(2, (len(depths) - 1) * substeps))
    Sf = np.vstack([np.interp(fine, depths, S[k]) for k in range(len(labels))])
    colors = {lab: matviz.SERIES[i % len(matviz.SERIES)]
              for i, lab in enumerate(labels)}
    K = float(S.max()) if S.size else 8.0

    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=90)

    def draw(fi: int):
        ax.clear()
        v = Sf[:, fi]
        order = [k for k in np.argsort(v)[::-1][:top_n] if v[k] > 0]
        y = (top_n - 1) - np.arange(len(order))   # strongest pinned to the top
        ax.barh(y, [v[k] for k in order], height=0.72,
                color=[colors[labels[k]] for k in order], alpha=0.9)
        for yi, k in zip(y, order):
            ax.text(v[k] + K * 0.02, yi, labels[k], va="center", fontsize=9.5,
                    color=matviz.INK)
        ax.set_xlim(0, K * 1.18)
        ax.set_ylim(-0.6, max(top_n, 1) - 0.4)
        ax.set_yticks([])
        ax.set_xlabel("lens salience  (disposition to say, top of vocabulary)",
                      fontsize=8)
        ax.set_title(rec["title"], loc="left", fontsize=10, fontweight="bold")
        ax.text(0.995, 1.04, f"depth {fine[fi]:.0f}%", transform=ax.transAxes,
                ha="right", fontsize=11, color=matviz.ACCENT, fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(labelsize=8)

    anim = animation.FuncAnimation(fig, draw, frames=len(fine),
                                   interval=1000 / fps)
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / f"{rec['slug']}__thoughts"
    written = []
    gif = f"{base}.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=fps))
    written.append(gif)
    if mp4:
        if shutil.which("ffmpeg"):
            mp4p = f"{base}.mp4"
            anim.save(mp4p, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
            written.append(mp4p)
        else:
            print("  (ffmpeg not found; wrote GIF only)")
    plt.close(fig)
    print(f"  [{rec['slug']}] wrote {', '.join(Path(w).name for w in written)}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<model>.json")
    ap.add_argument("--slug", default=None, help="animate one prompt (default: all)")
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--top-n", type=int, default=10, help="bars shown per frame")
    ap.add_argument("--substeps", type=int, default=7, help="interpolation smoothness")
    ap.add_argument("--mp4", action="store_true", help="also write MP4 (needs ffmpeg)")
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text())
    recs = [p for p in run["prompts"]
            if args.slug is None or p["slug"] == args.slug]
    if not recs:
        raise SystemExit(f"no prompt matching --slug {args.slug!r}")
    # namespace movies per model, matching figures/<tag>/...
    tag = run.get("tag") or Path(args.run).stem
    outdir = HERE / "figures" / tag / "animation"
    print(f"animating {len(recs)} prompt(s) from {run['model']}  (-> {outdir})")
    for rec in recs:
        make_movie(rec, outdir, fps=args.fps, top_n=args.top_n,
                   substeps=args.substeps, mp4=args.mp4)
    print(f"done -> {outdir}")


if __name__ == "__main__":
    main()
