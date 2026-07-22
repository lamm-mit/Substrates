# Copyright 2026.  Apache-2.0.
"""Compare Jacobian-lens runs across models (e.g. Qwen vs Gemma).

Loads two or more ``runs/<model>.json`` records and produces:

    figures/compare/<tag>__hitrate.png       grouped bars: item-level pass@5
                                              by evaluation shape and model
    figures/compare/<tag>__concept_rank.png   heatmap: each shared concept's best
                                              lens rank, concepts x models
    figures/compare/<tag>__swap.png           (if swaps present) target install
                                              rank after swap, per model
    runs/comparison_<tag>.json                the aligned numbers
    runs/comparison_<tag>_analysis.json       cross-model LLM synthesis
    reports/comparison_<tag>_report.{tex,pdf} the comparison report

    python compare.py --runs runs/qwen2forcausallm.json runs/gpt2lmheadmodel.json
    python compare.py --runs runs/*.json --provider openai --tag qwen_vs_gemma

Provider/auth and the offline fallback work exactly as in analyze.py.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matviz
import report_latex as rl
import llm

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"


def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


# --------------------------------------------------------------------------- #
# data alignment
# --------------------------------------------------------------------------- #

def hit_rate_table(runs, top_k=5):
    shapes = sorted({shape for r in runs for shape in (
                        r.get("metrics", {}).get("by_shape", {}).keys() or
                        {p["shape"] for p in r["prompts"]})},
                    key=lambda s: ["MULTIHOP", "ASSOCIATION", "RECOGNITION",
                                   "MODULATION", "REPORT_SWAP"].index(s)
                    if s in ("MULTIHOP", "ASSOCIATION", "RECOGNITION",
                             "MODULATION", "REPORT_SWAP") else 99)
    labels = [_short(r["model"]) for r in runs]
    values, counts = [], {}
    for r in runs:
        paper_metrics = r.get("metrics", {}).get("by_shape", {})
        if paper_metrics:
            row = []
            for shape in shapes:
                metric = paper_metrics.get(shape)
                if not metric or top_k not in metric["ks"]:
                    row.append(float("nan"))
                    continue
                value = metric["jacobian_lens"]["pass_at_k"][metric["ks"].index(top_k)]
                row.append(value)
                counts[(_short(r["model"]), shape)] = (None, metric["n_items"])
            values.append(row)
            continue
        byshape: dict[str, list[bool]] = {}
        for p in r["prompts"]:
            hits = [0 <= e["best_rank"] < top_k for e in p["emergence"]]
            byshape.setdefault(p["shape"], []).extend(hits)
        row = []
        for s in shapes:
            v = byshape.get(s)
            row.append(sum(v) / len(v) if v else float("nan"))
            if v:
                counts[(_short(r["model"]), s)] = (int(sum(v)), len(v))
        values.append(row)
    return shapes, labels, values, counts


def concept_rank_matrix(runs, max_rows=22):
    """Per (prompt, concept) best lens rank across models, for concepts present
    in >= 2 runs. Rows sorted by the best rank any model achieved."""
    labels = [_short(r["model"]) for r in runs]
    table: dict[tuple[str, str], dict[int, int]] = {}
    for mi, r in enumerate(runs):
        for p in r["prompts"]:
            for e in p["emergence"]:
                table.setdefault((p["slug"], e["label"]), {})[mi] = e["best_rank"] + 1
    rows = [k for k, v in table.items() if len(v) >= 2]
    rows.sort(key=lambda k: min(table[k].values()))
    rows = rows[:max_rows]
    row_labels = [f"{slug.split('-', 1)[-1]}: {c}" for (slug, c) in rows]
    ranks = [[table[k].get(mi, float("nan")) for mi in range(len(runs))]
             for k in rows]
    return row_labels, labels, ranks


def swap_rows(runs):
    out = []
    for r in runs:
        for p in r["prompts"]:
            sw = p.get("swap")
            if sw:
                out.append({"model": _short(r["model"]), "slug": p["slug"],
                            "source": sw["source"], "target": sw["target"],
                            "alpha": sw["alpha"],
                            "source_rank_clean": sw["source_rank_clean"] + 1,
                            "source_rank_swapped": sw["source_rank_swapped"] + 1,
                            "target_rank_clean": sw["target_rank_clean"] + 1,
                            "target_rank_swapped": sw["target_rank_swapped"] + 1})
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def make_figures(runs, tag):
    out = (FIGDIR / "compare")
    out.mkdir(parents=True, exist_ok=True)
    figs = {}

    shapes, labels, values, _ = hit_rate_table(runs)
    figs["hitrate"] = matviz.plot_grouped_bars(
        shapes, labels, values, out / f"{tag}__hitrate",
        title="Concept surfacing rate by experiment type — model comparison",
        subtitle="item-level pass@5 over registered score spans and fixed bands",
        ylabel="pass@5")[0]

    row_labels, cols, ranks = concept_rank_matrix(runs)
    if row_labels:
        figs["concept_rank"] = matviz.plot_rank_matrix(
            row_labels, cols, ranks, out / f"{tag}__concept_rank",
            title="Best lens rank per shared concept, across models",
            subtitle="bright = concept strongly surfaces; grey = not present for "
                     "that model")[0]

    sw = swap_rows(runs)
    if sw:
        # target install rank after swap (lower = better), grouped by prompt
        by_slug: dict[str, dict[str, float]] = {}
        for s in sw:
            by_slug.setdefault(s["slug"], {})[s["model"]] = s["target_rank_swapped"]
        cats = list(by_slug)
        models = sorted({s["model"] for s in sw})
        vals = [[by_slug[c].get(m, float("nan")) for c in cats] for m in models]
        figs["swap"] = matviz.plot_grouped_bars(
            [c.split("-", 1)[-1] for c in cats], models, vals,
            out / f"{tag}__swap",
            title="Causal swap: target-concept rank after the swap (lower = installed)",
            subtitle="post-swap full-vocab rank of the swapped-in target concept",
            ylabel="target rank after swap", pct=False)[0]
    return figs


# --------------------------------------------------------------------------- #
# LLM comparative analysis
# --------------------------------------------------------------------------- #

SYSTEM_CMP = """\
You are an interpretability researcher comparing Jacobian-lens ("J-lens") runs \
across different language models, in the framework of "Verbalizable \
Representations Form a Global Workspace in Language Models" (Anthropic, 2026).

You are given: (a) per-model top-5 concept-surfacing rates by experiment type \
(MULTIHOP / ASSOCIATION / RECOGNITION / MODULATION / REPORT_SWAP); (b) a matrix \
of each shared concept's best lens rank per model; (c) optionally causal-swap \
outcomes per model. Write a comparative analysis: which model surfaces \
verbalizable scientific concepts more strongly and where (which shapes, which \
concepts), how vocabulary size / model capability explains differences \
(single-token limitation, §9.1), how the causal swaps compare, and a practical \
recommendation for which model to use for scientific interpretability. Cite the \
numbers. Be precise and honest; note where a difference is small or where a \
model simply lacks a concept.
"""


SYSTEM_PP = """\
You are an interpretability researcher comparing how several language models \
represent ONE prompt, read out with the Jacobian lens (global-workspace \
framework, Anthropic 2026). You are given, for each model: the best lens rank \
(and depth) of each tracked/discovered concept for this prompt, and that model's \
own per-prompt read-out analysis. Write a tight 1-2 paragraph comparison for \
THIS prompt: which model surfaces the unspoken intermediates more strongly or \
earlier (cite ranks/depths), where the models agree or diverge, and what \
explains the difference (capability, vocabulary, single-token limitation). Be \
concrete and honest; if a model simply lacks the concept, say so.
"""


def _img(path):
    p = Path(path)
    if not p.is_file():
        return None
    return {"type": "image_png",
            "data": base64.standard_b64encode(p.read_bytes()).decode()}


# --------------------------------------------------------------------------- #
# per-prompt cross-model layer
# --------------------------------------------------------------------------- #

def load_analyses(run_paths):
    """For each run path, load its sibling <stem>_analysis.json (if present) as
    slug -> 'what we see' text. Returns a list aligned with the runs."""
    out = []
    for p in run_paths:
        ana = Path(p).with_name(Path(p).stem + "_analysis.json")
        if ana.is_file():
            data = json.loads(ana.read_text())
            out.append({a["slug"]: a["analysis"] for a in data.get("per_prompt", [])})
        else:
            out.append({})
    return out


def shared_prompts(runs):
    """Prompts (by slug) present in >=2 runs, in first-appearance order."""
    order, meta = [], {}
    for mi, r in enumerate(runs):
        for p in r["prompts"]:
            if p["slug"] not in meta:
                meta[p["slug"]] = {"title": p["title"], "shape": p["shape"],
                                   "domain": p["domain"], "models": []}
                order.append(p["slug"])
            meta[p["slug"]]["models"].append(mi)
    return [(s, meta[s]) for s in order if len(meta[s]["models"]) >= 2]


def _emergence_of(run, slug):
    for p in run["prompts"]:
        if p["slug"] == slug:
            return {e["label"]: (e["best_rank"] + 1, e["best_depth"])
                    for e in p["emergence"]}
    return {}


def _per_prompt_analysis(title, shape, per_model, fig_png, provider, model,
                         offline):
    if offline or model is None:
        parts = []
        for pm in per_model:
            top = [c for c, (r, _) in pm["emergence"].items() if r <= 5]
            parts.append(f"{pm['model']}: {len(top)} concept(s) in lens top-5"
                         + (f" ({', '.join(top)})" if top else ""))
        return "[offline template] " + "; ".join(parts) + "."
    lines = [f"Prompt: {title}  (shape {shape}). Compare the models on THIS prompt."]
    for pm in per_model:
        lines.append(f"\n### {pm['model']}")
        lines.append("  concept best ranks: " + ", ".join(
            f"{c}=rank {r}@{d}%" for c, (r, d) in pm["emergence"].items()))
        if pm["analysis"]:
            lines.append("  its own read-out analysis: " + pm["analysis"])
    lines.append("\nThe attached figure shows each concept's best rank per model "
                 "(left = stronger). Write the 1-2 paragraph cross-model "
                 "comparison for this prompt.")
    blocks = [{"type": "text", "text": "\n".join(lines)}]
    if (b := _img(fig_png)):
        blocks.append(b)
    try:
        return llm.complete(provider, model, SYSTEM_PP, blocks, max_tokens=1500)
    except Exception as exc:  # noqa: BLE001
        return f"[per-prompt comparison unavailable: {type(exc).__name__}: {exc}]"


def per_prompt_layer(runs, analyses, tag, provider, model, offline):
    """One record per shared prompt: a cross-model rank figure, each model's own
    read-out analysis, and an LLM cross-model comparison for that prompt."""
    labels = [_short(r["model"]) for r in runs]
    outdir = FIGDIR / "compare" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    records = []
    for slug, m in shared_prompts(runs):
        per_model, concepts = [], []
        for mi in m["models"]:
            em = _emergence_of(runs[mi], slug)
            per_model.append({"model": labels[mi], "emergence": em,
                              "analysis": analyses[mi].get(slug, "")})
            for c in em:
                if c not in concepts:
                    concepts.append(c)
        mods = [pm["model"] for pm in per_model]
        ranks = [[pm["emergence"].get(c, (float("nan"), None))[0] for c in concepts]
                 for pm in per_model]
        fig = matviz.plot_concept_rank_dots(
            concepts, mods, ranks, outdir / f"{slug}__ranks",
            title=f"{m['title']} — concept ranks across models",
            subtitle="best lens rank per concept; left = stronger, missing = "
                     "not surfaced")[0]
        cross = _per_prompt_analysis(m["title"], m["shape"], per_model, fig,
                                     provider, model, offline)
        print(f"  [{slug}] per-prompt comparison ({'offline' if offline else provider})")
        records.append({"slug": slug, "title": m["title"], "shape": m["shape"],
                        "domain": m["domain"],
                        "figure": str(Path(fig).relative_to(HERE)),
                        "per_model": per_model, "cross": cross})
    return records


def comparative_analysis(runs, figs, provider, model, offline):
    shapes, labels, values, counts = hit_rate_table(runs)
    lines = [f"Models compared: {labels}", "",
             "## Top-5 surfacing rate by shape (model: shape=rate):"]
    for li, lab in enumerate(labels):
        parts = [f"{s}={values[li][j]:.2f}" if values[li][j] == values[li][j]
                 else f"{s}=n/a" for j, s in enumerate(shapes)]
        lines.append(f"  {lab}: " + ", ".join(parts))
    row_labels, cols, ranks = concept_rank_matrix(runs)
    if row_labels:
        lines += ["", "## Best lens rank per shared concept (rank; lower=stronger):"]
        for rl_, rr in zip(row_labels, ranks):
            cells = ", ".join(f"{cols[j]}={'n/a' if rr[j]!=rr[j] else int(rr[j])}"
                              for j in range(len(cols)))
            lines.append(f"  {rl_}: {cells}")
    sw = swap_rows(runs)
    if sw:
        lines += ["", "## Causal swaps (source/target full-vocab rank clean->swapped):"]
        for s in sw:
            lines.append(f"  [{s['model']}] {s['slug']} {s['source']}->{s['target']} "
                         f"(a={s['alpha']}): source {s['source_rank_clean']}->"
                         f"{s['source_rank_swapped']}, target "
                         f"{s['target_rank_clean']}->{s['target_rank_swapped']}")

    if offline:
        best = max(labels, key=lambda L: _mean_hit(runs, L))
        return ("[offline template] Comparison of " + ", ".join(labels) +
                f". Highest overall top-5 surfacing rate: {best}. See the "
                "grouped-bar and concept-rank figures for the per-shape and "
                "per-concept breakdown.")

    blocks = [{"type": "text", "text": "\n".join(lines) +
               "\n\nThe attached figures are the per-shape hit-rate comparison "
               "and the per-concept rank matrix. Write the comparative analysis "
               "(3-5 paragraphs)."}]
    for key in ("hitrate", "concept_rank", "swap"):
        if key in figs and (b := _img(figs[key])):
            blocks.append(b)
    try:
        return llm.complete(provider, model, SYSTEM_CMP, blocks, max_tokens=6000)
    except Exception as exc:  # noqa: BLE001
        return f"[comparison analysis unavailable: {type(exc).__name__}: {exc}]"


def _mean_hit(runs, label):
    for r in runs:
        if _short(r["model"]) == label:
            hits = [0 <= e["best_rank"] < 5 for p in r["prompts"]
                    for e in p["emergence"]]
            return sum(hits) / len(hits) if hits else 0.0
    return 0.0


# --------------------------------------------------------------------------- #
# comparison report (LaTeX -> PDF)
# --------------------------------------------------------------------------- #

def build_report(runs, tag, figs, analysis_text, per_prompt, provider, model,
                 offline, compile_pdf=True):
    labels = [_short(r["model"]) for r in runs]
    L = [rl.PREAMBLE % {
        "title": rl.tex_escape(f"Model comparison: {' vs '.join(labels)}"),
        "author": ("offline template" if offline
                   else rl.tex_escape(f"Comparison by {provider}:{model}"))}]
    L.append(r"\noindent " + rl.tex_escape(
        "Models: " + "; ".join(f"{_short(r['model'])} ({r['n_layers']} layers, "
                               f"d_model {r['d_model']})" for r in runs)) + "\n")

    L.append(r"\section*{Comparative synthesis}")
    L.append(rl.md_to_tex(analysis_text))

    cap = {"hitrate": "Top-5 concept-surfacing rate by experiment type, per model.",
           "concept_rank": "Best lens rank of each shared concept, per model "
                           "(grey = not present for that model).",
           "swap": "Causal swap: post-swap rank of the swapped-in target "
                   "concept, per model (lower = more successfully installed)."}
    for key in ("hitrate", "concept_rank", "swap"):
        if key in figs:
            L.append(r"\begin{figure}[H]\centering")
            L.append(rf"\includegraphics[width=0.95\linewidth]"
                     rf"{{{Path(figs[key]).resolve()}}}")
            L.append(rf"\caption{{{cap[key]}}}\end{{figure}}")

    # per-shape numeric table
    shapes, slabels, values, _ = hit_rate_table(runs)
    L.append(r"\clearpage\section*{Top-5 surfacing rate (numbers)}")
    L.append(r"\begin{table}[H]\centering\small\begin{tabular}{l" +
             "r" * len(shapes) + r"}\toprule")
    L.append("model & " + " & ".join(rl.tex_escape(s) for s in shapes) + r" \\ \midrule")
    for li, lab in enumerate(slabels):
        cells = " & ".join("--" if values[li][j] != values[li][j]
                           else f"{values[li][j]*100:.0f}\\%" for j in range(len(shapes)))
        L.append(f"{rl.tex_escape(lab)} & {cells} \\\\")
    L.append(r"\bottomrule\end{tabular}\end{table}")

    sw = swap_rows(runs)
    if sw:
        L.append(r"\section*{Causal swaps}")
        L.append(r"\begin{table}[H]\centering\small\begin{tabular}{lllrr}\toprule")
        L.append(r"model & prompt & swap & source rank & target rank \\ \midrule")
        for s in sw:
            L.append(f"{rl.tex_escape(s['model'])} & "
                     f"{rl.tex_escape(s['slug'].split('-',1)[-1])} & "
                     f"{rl.tex_escape(s['source']+'->'+s['target'])} & "
                     f"{s['source_rank_clean']}$\\to${s['source_rank_swapped']} & "
                     f"{s['target_rank_clean']}$\\to${s['target_rank_swapped']} \\\\")
        L.append(r"\bottomrule\end{tabular}")
        L.append(r"\caption{Source rank rises = suppressed; target rank falls = "
                 r"installed.}\end{table}")

    # ---- per-prompt cross-model analysis (keeps each model's own read-out) ---
    if per_prompt:
        L.append(r"\clearpage\section{Per-prompt cross-model analysis}")
        L.append(r"For each prompt shared across the models: each model's own "
                 r"read-out analysis, the concept ranks side by side, and a "
                 r"cross-model comparison.")
        for rec in per_prompt:
            L.append(rf"\subsection{{{rl.tex_escape(rec['title'])} "
                     rf"\normalfont\texttt{{{rl.tex_escape(rec['shape'])}}}}}")
            for pm in rec["per_model"]:
                if pm["analysis"]:
                    L.append(rf"\paragraph{{{rl.tex_escape(pm['model'])} --- "
                             rf"what it saw.}} " + rl.md_to_tex(pm["analysis"]))
            figp = (HERE / rec["figure"]).resolve()
            if figp.is_file():
                L.append(r"\begin{figure}[H]\centering")
                L.append(rf"\includegraphics[width=0.85\linewidth]{{{figp}}}")
                L.append(r"\caption{Best lens rank of each concept, per model "
                         r"(left = stronger).}\end{figure}")
            L.append(r"\paragraph{Across models.} " + rl.md_to_tex(rec["cross"]))

    L.append(r"\end{document}")
    reports = HERE / "reports"
    reports.mkdir(exist_ok=True)
    tex = reports / f"comparison_{tag}_report.tex"
    tex.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {tex}")
    if compile_pdf:
        rl.compile_pdf(tex)
    return tex


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="two or more runs/<model>.json files to compare")
    ap.add_argument("--tag", default=None, help="output name (default: joined model names)")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    runs = [json.loads(Path(p).read_text()) for p in args.runs]
    if len(runs) < 2:
        raise SystemExit("need at least two runs to compare")
    analyses = load_analyses(args.runs)  # per-model 'what we see' (if present)
    tag = args.tag or "_vs_".join(_short(r["model"]) for r in runs)
    tag = tag.replace("/", "-").replace(".", "")
    model = args.model or (None if args.offline else llm.default_model(args.provider))

    print(f"comparing: {[_short(r['model']) for r in runs]}  (tag={tag})")
    figs = make_figures(runs, tag)
    analysis = comparative_analysis(runs, figs, args.provider, model, args.offline)
    per_prompt = per_prompt_layer(runs, analyses, tag, args.provider, model,
                                  args.offline)

    comp = {"tag": tag, "models": [r["model"] for r in runs],
            "provider": args.provider, "offline": args.offline,
            "hitrate": dict(zip(("shapes", "labels", "values"),
                                hit_rate_table(runs)[:3])),
            "swaps": swap_rows(runs), "analysis": analysis,
            "per_prompt": [{k: v for k, v in rec.items() if k != "per_model"}
                           | {"models": [pm["model"] for pm in rec["per_model"]]}
                           for rec in per_prompt]}
    (HERE / "runs" / f"comparison_{tag}.json").write_text(
        json.dumps(comp, indent=2, ensure_ascii=False))
    print(f"wrote runs/comparison_{tag}.json")

    build_report(runs, tag, figs, analysis, per_prompt, args.provider, model,
                 args.offline, compile_pdf=not args.no_compile)
    print(f"done -> reports/comparison_{tag}_report.pdf")


if __name__ == "__main__":
    main()
