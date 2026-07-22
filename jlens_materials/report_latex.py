# Copyright 2026.  Apache-2.0.
"""Assemble a Jacobian-lens run + its LLM analysis into a LaTeX document and
compile it to PDF.

Reads ``runs/<model>.json`` (figures + numeric records from ``run_lens.py``)
and ``runs/<model>_analysis.json`` (the "what we see" write-ups from
``analyze.py``) and writes ``reports/<model>_report.tex`` +
``reports/<model>_report.pdf``.

    python report_latex.py --run runs/qwen2forcausallm.json            # .tex + .pdf
    python report_latex.py --run runs/qwen2forcausallm.json --no-compile  # .tex only

Compilation uses ``latexmk`` (falls back to ``pdflatex`` x2).  A TeX install is
required for the PDF step; the ``.tex`` is always written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import domain_prompts as dp

# macOS MacTeX default; harmless elsewhere.
_TEXBIN = "/Library/TeX/texbin"

FIG_ORDER = ["grid", "stream", "trajectory", "emergence", "heatmap"]
FIG_CAPTION = {
    "grid": "Layer $\\times$ position slice: top-1 lens word per cell, shaded "
            "by full-vocab rank.",
    "stream": "Thought stream: each concept the lens surfaces flows across "
              "depth (band thickness $\\sim$ salience) --- the dynamics of "
              "what the model is disposed to say.",
    "trajectory": "Rank of each tracked concept vs depth over the predeclared "
                  "score span (rank~1 at top); a sustained line toward the top "
                  "marks workspace entry.",
    "emergence": "First sustained threshold crossing in the fixed band; peak rank is annotated.",
    "heatmap": "Position $\\times$ layer rank map for the strongest concept.",
}
LEGACY_FIG_CAPTION = {
    **FIG_CAPTION,
    "trajectory": "Legacy exploratory trajectory: best tracked-concept rank "
                  "was searched over all prompt positions at each depth.",
    "emergence": "Legacy peak-depth summary, not the fixed-band sustained-onset "
                 "metric used by format-v2 runs.",
}
FIG_WIDTH = {"grid": "\\linewidth", "stream": "0.95\\linewidth",
             "trajectory": "0.86\\linewidth", "emergence": "0.7\\linewidth",
             "heatmap": "0.86\\linewidth"}

_BUILTIN_BY_SLUG = {p.slug: p for p in dp.ALL_PROMPTS}


# --------------------------------------------------------------------------- #
# text -> LaTeX
# --------------------------------------------------------------------------- #

_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
            # Normalize common model-generated Unicode to pdfTeX-safe forms.
            # ASCII TeX punctuation also avoids font-dependent missing glyphs.
            "‐": "-", "‑": "-", "–": "--", "—": "---",
            "‘": "`", "’": "'", "“": "``", "”": "''", "…": "...",
            "×": r"$\times$", "→": r"$\rightarrow$", "⇒": r"$\Rightarrow$",
            # OT1 renders bare < and > as inverted ¡/¿ — escape them (fixes
            # "->" and any comparison operators in LLM analysis text)
            "<": r"\textless{}", ">": r"\textgreater{}"}


def tex_escape(s: str) -> str:
    out = []
    for ch in s:
        out.append(_ESCAPES.get(ch, ch))
    return "".join(out)


def md_to_tex(text: str) -> str:
    """Escape LaTeX specials, then re-apply a little markdown (**bold**, `code`)
    and paragraph breaks. Robust to whatever the LLM emits."""
    # Pull out `code` before **bold**. Model prose occasionally contains
    # markdown-looking ``**`` inside a code span (or an unmatched span that
    # reaches into one). Stashing bold first can then put bold sentinels inside
    # the saved code value; those sentinels are reintroduced only after the
    # bold-restoration pass and leak NUL bytes into the generated TeX.
    bolds, codes = [], []

    def _stash_bold(m):
        bolds.append(m.group(1)); return f"\x00B{len(bolds)-1}\x00"

    def _stash_code(m):
        codes.append(m.group(1)); return f"\x00C{len(codes)-1}\x00"

    text = re.sub(r"`([^`]+?)`", _stash_code, text)
    text = re.sub(r"\*\*(.+?)\*\*", _stash_bold, text)
    text = tex_escape(text)
    # restore spans (their contents are escaped separately)
    text = re.sub(r"\x00B(\d+)\x00",
                  lambda m: r"\textbf{" + tex_escape(bolds[int(m.group(1))]) + "}",
                  text)
    text = re.sub(r"\x00C(\d+)\x00",
                  lambda m: r"\texttt{" + tex_escape(codes[int(m.group(1))]) + "}",
                  text)
    # blank line -> paragraph break; strip a leading "[offline template]" tag bold
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n\n".join(paras)


# --------------------------------------------------------------------------- #
# document assembly
# --------------------------------------------------------------------------- #

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{parskip}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\setkeys{Gin}{keepaspectratio}
\emergencystretch=2em
\title{%(title)s}
\author{%(author)s}
\date{Generated by the jlens\_materials pipeline}
\begin{document}
\maketitle
"""


def _surprising_block(surp: list[dict]) -> list[str]:
    """LaTeX for exploratory, non-quantitative surfaced candidates."""
    L = [r"\paragraph{Exploratory surfaced candidates.} "
         "Heuristic top-ranked tokens absent from the prompt and, where checked, "
         "from an actual generated or teacher-forced completion. This is not a "
         "paper metric and may include tokenizer artifacts."]
    L.append(r"\begin{table}[H]\centering\small\begin{tabular}{lrrl}\toprule")
    L.append(r"concept & band cells & peak rank & near token \\ \midrule")
    for s in surp[:10]:
        L.append(f"{tex_escape(s['concept'])} & {s['score']:g} & {s['best_rank']} "
                 f"& {tex_escape(s['near_token'])} \\\\")
    L.append(r"\bottomrule\end{tabular}")
    L.append(r"\caption{Exploratory surfaced candidates, strongest first "
             r"(`band cells' = how many workspace-band positions/layers it "
             r"tops).}\end{table}")
    return L


def _swap_block(sw: dict) -> list[str]:
    """LaTeX for a causal-swap result: clean vs swapped next-token, plus the
    source/target full-vocab rank change."""
    if sw.get("protocol") == "verbal_report" and sw.get("trials"):
        clean_source = sw.get("clean_source", sw.get("source", "?"))
        L = [
            r"\paragraph{Multi-candidate verbal-report swaps.} "
            + tex_escape(
                f"Clean one-word report: {clean_source}. Each row installs a "
                "predeclared sibling candidate across the fixed band."
            ),
            r"\begin{table}[H]\centering\small",
            r"\begin{tabular}{lrrlc}\toprule",
            r"target & source rank & target rank & swapped top-1 & success \\ \midrule",
        ]
        for trial in sw["trials"]:
            target = trial["target"]
            if target.strip().casefold() == str(clean_source).casefold():
                target += " (case-only self-target)"
            source_rank = (
                f"{trial['source_rank_clean']+1}$\\to$"
                f"{trial['source_rank_swapped']+1}"
            )
            target_rank = (
                f"{trial['target_rank_clean']+1}$\\to$"
                f"{trial['target_rank_swapped']+1}"
            )
            top_after = trial.get("swapped_top", [["", 0]])[0][0]
            success = "yes" if trial.get("protocol_success") else "no"
            L.append(
                f"{tex_escape(target)} & {source_rank} & {target_rank} & "
                f"{tex_escape(top_after) or '{}'} & {success} \\\\"
            )
        L.append(r"\bottomrule\end{tabular}")
        L.append(
            rf"\caption{{Verbal-report candidate swaps; protocol success rate "
            rf"{100 * sw.get('protocol_success_rate', 0):.1f}\%. A case-only "
            r"self-target is diagnostic noise and is excluded by current runner code.}"
        )
        L.append(r"\end{table}")
        return L

    L = [r"\paragraph{Causal swap (Fig 4C).} "
         + tex_escape(f"J-lens coordinate {sw['source']} -> {sw['target']} "
                      f"(alpha={sw['alpha']}) at every band layer "
                      f"{sw['band_layers'][0]}-{sw['band_layers'][-1]}. "
                      "Does intervening on the concept change the output?")]
    L.append(r"\begin{table}[H]\centering\small")
    L.append(r"\begin{tabular}{rlr@{\quad}lr}\toprule")
    L.append(r"\# & clean next-token & prob & after swap & prob \\ \midrule")
    for i, ((cw, cp), (sw_w, sp)) in enumerate(
            zip(sw["clean_top"], sw["swapped_top"]), 1):
        L.append(f"{i} & {tex_escape(cw) or '{}'} & {cp:.3f} & "
                 f"{tex_escape(sw_w) or '{}'} & {sp:.3f} \\\\")
    L.append(r"\bottomrule\end{tabular}")
    sr = f"{sw['source_rank_clean']+1}$\\to${sw['source_rank_swapped']+1}"
    tr = f"{sw['target_rank_clean']+1}$\\to${sw['target_rank_swapped']+1}"
    L.append(rf"\caption{{Source ``{tex_escape(sw['source'])}'' full-vocab rank "
             rf"{sr} (higher = suppressed); target "
             rf"``{tex_escape(sw['target'])}'' rank {tr} (lower = installed).}}")
    L.append(r"\end{table}")
    return L


def _fig_abspath(rec, base_dir, kind):
    rel = rec.get("figures", {}).get(kind)
    if not rel:
        return None
    p = (base_dir / rel).resolve()
    return str(p) if p.is_file() else None


def _prompt_text_for_report(rec: dict) -> tuple[str, str]:
    """Return (label, prompt text) for a case-study report section."""
    for key in ("prompt_text", "resolved_prompt", "prompt_used"):
        if rec.get(key):
            return "Prompt used", rec[key]
    p = _BUILTIN_BY_SLUG.get(rec.get("slug"))
    if p:
        if p.text:
            return "Prompt used (authored; older run did not store resolved prompt)", p.text
        parts = []
        if p.system:
            parts.append(f"System: {p.system}")
        if p.user:
            parts.append(f"User: {p.user}")
        if p.assistant_prefill:
            parts.append(f"Assistant prefill: {p.assistant_prefill}")
        return ("Prompt used (authored chat fields; older run did not store "
                "resolved chat-template text)", "\n\n".join(parts))
    if rec.get("prompt_tail"):
        return "Prompt tail (older run did not store the full prompt)", "..." + rec["prompt_tail"]
    return "Prompt used", "(not recorded)"


def _prompt_block(rec: dict) -> list[str]:
    label, text = _prompt_text_for_report(rec)
    lines = [
        rf"\paragraph{{{tex_escape(label)}.}}",
        r"\begin{quote}\footnotesize\ttfamily\raggedright",
    ]
    for line in text.splitlines() or [""]:
        lines.append(tex_escape(line) + r"\par")
    lines.append(r"\end{quote}")
    return lines


def build_tex(run_path: str) -> Path:
    run_path = Path(run_path)
    run = json.loads(run_path.read_text())
    base_dir = run_path.resolve().parent.parent  # jlens_materials/
    ana_path = run_path.with_name(run_path.stem + "_analysis.json")
    analysis = json.loads(ana_path.read_text()) if ana_path.is_file() else None
    ana_by_slug = ({a["slug"]: a for a in analysis["per_prompt"]}
                   if analysis else {})
    legacy = run.get("format_version", 1) < 2

    if analysis:
        if analysis["offline"]:
            author = "Interpretation: offline template (no LLM)"
        else:
            author = tex_escape(f"Interpretation by {analysis.get('provider','?')}"
                                f":{analysis['analysis_model']}")
    else:
        author = "No LLM analysis available"

    L = [PREAMBLE % {
        "title": tex_escape(f"Jacobian-lens report: {run['model']}"),
        "author": author}]

    lens_n = run.get("lens_n_prompts", "?")
    L.append(r"\noindent " + tex_escape(
        f"Model: {run['model']} ({run['n_layers']} layers, d_model "
        f"{run['d_model']}); lens fitted on {lens_n} prompts. Shapes: "
        f"{', '.join(run['shapes'])}. Domains: {', '.join(run['domains'])}.")
        + "\n")
    methodology = run.get("methodology", {})
    if methodology:
        L.append(r"\par\noindent\textbf{Methodology status.} " + tex_escape(
            f"Claims level: {methodology.get('claims_level', 'unknown')}. "
            f"Recipe: {methodology.get('recipe', {}).get('name', '?')}; "
            f"fixed workspace band: {methodology.get('workspace_band', '?')}; "
            "predetermined score spans with synonym-min ranks.") + "\n")
    elif legacy:
        L.append(
            r"\par\noindent\fcolorbox{red!60!black}{red!4}{"
            r"\parbox{0.94\linewidth}{\textbf{Legacy exploratory run.} "
            r"This record predates lens provenance, fixed-position scoring, "
            r"and preregistered-band metadata. Its figures are descriptive "
            r"only; its hit rates are not paper-protocol results.}}" + "\n"
        )
    if run.get("errors"):
        L.append(
            r"\par\noindent\textbf{Run incomplete.} "
            + tex_escape(
                f"{len(run['errors'])} prompt(s) failed. See the run JSON for "
                "the recorded exception messages."
            ) + "\n"
        )
    if run.get("insufficient_sample_shapes"):
        L.append(
            r"\par\noindent\textbf{Sample-size guard.} Quantitative claims "
            r"are disabled for the following groups:"
        )
        L.append(r"\begin{itemize}\small\raggedright")
        for group in run["insufficient_sample_shapes"]:
            breakable = tex_escape(group).replace(
                ":", r":\allowbreak{}"
            ).replace("-", r"-\allowbreak{}")
            L.append(r"\item " + breakable)
        L.append(r"\end{itemize}")

    if analysis and analysis.get("synthesis"):
        L.append(r"\section*{Executive synthesis}")
        L.append(md_to_tex(analysis["synthesis"]))

    # run-level 'hidden thoughts': unprompted concepts, aggregated across prompts
    agg: dict[str, list] = {}
    for rec in run["prompts"]:
        for s in rec.get("surprising", []):
            e = agg.setdefault(s["concept"], [0.0, 0])
            e[0] += s["score"]; e[1] += 1
    if agg:
        ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])[:15]
        L.append(r"\section*{Exploratory surfaced candidates across the run}")
        L.append(r"Heuristic top-ranked candidates, filtered against prompts and "
                 r"actual completions where available. They are descriptive and "
                 r"not included in the paper-protocol metrics.")
        L.append(r"\begin{table}[H]\centering\small\begin{tabular}{lrr}\toprule")
        L.append(r"concept & total band cells & \# prompts \\ \midrule")
        for c, (sc, n) in ranked:
            L.append(f"{tex_escape(c)} & {sc:g} & {n} \\\\")
        L.append(r"\bottomrule\end{tabular}\end{table}")

    by_shape = run.get("metrics", {}).get("by_shape", {})
    if by_shape:
        L.append(r"\section*{Item-level lens evaluation}")
        L.append(r"\begin{table}[H]\centering\small")
        L.append(r"\begin{tabular}{lrrrr}\toprule")
        L.append(r"shape & items & sufficient & J-lens AUC & logit AUC \\ \midrule")
        for shape, metric in by_shape.items():
            sufficient = "yes" if metric["sufficient_sample"] else "no"
            L.append(
                f"{tex_escape(shape)} & {metric['n_items']} & {sufficient} & "
                f"{metric['jacobian_lens']['auc_log_k']:.3f} & "
                f"{metric['logit_lens']['auc_log_k']:.3f} \\\\")
        L.append(r"\bottomrule\end{tabular}")
        L.append(r"\caption{Pass@k area under the curve against log k.}\end{table}")

    modulation = run.get("metrics", {}).get("directed_modulation_controls", {})
    if modulation:
        L.append(r"\section*{Directed-modulation controls}")
        L.append(r"\begin{table}[H]\centering\small")
        L.append(r"\begin{tabular}{llrrc}\toprule")
        L.append(r"control group & condition & items & hit rate & sufficient \\ \midrule")
        summaries = []
        for group, conditions in modulation.items():
            for condition in ("focus", "suppress", "control"):
                metric = conditions.get(condition)
                if not metric:
                    continue
                sufficient = "yes" if metric["sufficient_sample"] else "no"
                L.append(
                    f"{tex_escape(group)} & {condition} & {metric['n_items']} & "
                    f"{metric['hit_rate']:.3f} & {sufficient} \\\\"
                )
            summary = conditions.get("summary", {})
            if summary:
                summaries.append(
                    f"{group}: {summary.get('distinct_phrasings', 0)} distinct "
                    f"phrasings (paper target "
                    f"{summary.get('paper_phrasing_target', 24)})"
                )
        L.append(r"\bottomrule\end{tabular}")
        caption = "Matched focus, suppress, and neutral modulation trials."
        if summaries:
            caption += " " + "; ".join(summaries) + "."
        L.append(rf"\caption{{{tex_escape(caption)}}}\end{{table}}")

    causal = run.get("metrics", {}).get("causal_swaps", {})
    if causal.get("n_interventions", 0):
        protocol_rate = causal.get("protocol_success_rate")
        causal_rate = causal.get("causal_success_rate")
        protocol_display = "--" if protocol_rate is None else f"{protocol_rate:.3f}"
        causal_display = "--" if causal_rate is None else f"{causal_rate:.3f}"
        sufficient = "yes" if causal["sufficient_sample"] else "no"
        L.append(r"\section*{Causal-intervention summary}")
        L.append(r"\begin{table}[H]\centering\small")
        L.append(r"\begin{tabular}{rrrrrc}\toprule")
        L.append(r"interventions & graded & protocol success & counterfactual & causal success & sufficient \\ \midrule")
        L.append(
            f"{causal['n_interventions']} & {causal['n_graded']} & "
            f"{protocol_display} & {causal.get('n_counterfactual', 0)} & "
            f"{causal_display} & {sufficient} \\\\"
        )
        L.append(r"\bottomrule\end{tabular}")
        L.append(r"\caption{Registered behavioral outcomes for causal swaps; "
                 r"rank movement alone is not counted as causal success.}\end{table}")

    _tag = run.get("tag")
    hitbars = ((base_dir / "figures" / _tag / "summary__hit_rates.png") if _tag
               else (base_dir / "figures" / "summary__hit_rates.png")).resolve()
    if legacy and hitbars.is_file():
        L.append(r"\section*{Legacy exploratory concept surfacing}")
        L.append(r"\begin{figure}[H]\centering")
        L.append(rf"\includegraphics[width=0.9\linewidth]{{{hitbars}}}")
        L.append(r"\caption{Legacy exploratory fraction of tracked concepts "
                 r"reaching the lens top-5 anywhere in depth and prompt "
                 r"position, with Wilson 95\% CIs. This is not a fixed-span "
                 r"paper-protocol metric.}")
        L.append(r"\end{figure}")

    for domain in run["domains"]:
        recs = [r for r in run["prompts"] if r["domain"] == domain]
        if not recs:
            continue
        L.append(rf"\clearpage\section{{Domain: {tex_escape(domain)}}}")
        for rec in recs:
            L.append(rf"\subsection{{{tex_escape(rec['title'])} "
                     rf"\normalfont\texttt{{{tex_escape(rec['shape'])}}}}}")
            L.append(md_to_tex(rec["description"]))
            L += _prompt_block(rec)

            if "valid_for_metrics" in rec:
                if rec["valid_for_metrics"]:
                    status = "Included in the registered aggregate."
                else:
                    reasons = rec.get("excluded_reasons") or ["not a registered metric item"]
                    status = "Excluded: " + "; ".join(reasons) + "."
                L.append(r"\paragraph{Metric status.} " + tex_escape(status))
                baseline = rec.get("baseline", {})
                if baseline.get("required"):
                    expected = ", ".join(baseline.get("expected", []))
                    greedy = baseline.get("greedy_token", "")
                    correct = "yes" if baseline.get("correct") else "no"
                    L.append(
                        r"\paragraph{Clean baseline.} "
                        + tex_escape(
                            f"Expected {expected}; greedy token {greedy}; "
                            f"correct: {correct}."
                        )
                    )
                if rec.get("generated_completion"):
                    completion = rec["generated_completion"]
                    for source, visible in (
                        ("\\", "[backslash]"), ("\r", "[carriage-return]"),
                        ("\n", "[newline]"), ("\t", "[tab]"),
                    ):
                        completion = completion.replace(source, visible)
                    completion = f'"{completion}"'
                    L.append(r"\paragraph{Generated completion used for the "
                             r"output-absence control.}")
                    L.append(r"\begin{quote}\small\ttfamily\raggedright "
                             + tex_escape(completion) + r"\end{quote}")

            # emergence table
            L.append(r"\begin{table}[H]\centering\small")
            L.append(r"\begin{tabular}{lrrrrc}\toprule")
            L.append(r"concept & J rank & logit rank & peak depth & onset & rank-1? \\ \midrule")
            for e in rec["emergence"]:
                peak_depth = "--" if e["best_depth"] is None else f"{e['best_depth']}\\%"
                onset = ("--" if e.get("onset_depth") is None
                         else f"{e['onset_depth']}\\%")
                logit_rank = e.get("logit_lens_best_rank")
                logit_display = "--" if logit_rank is None else str(logit_rank + 1)
                depth = f"{logit_display} & {peak_depth}"
                L.append(f"{tex_escape(e['label'])} & {e['best_rank']+1} & "
                         f"{depth} & {onset} & {'yes' if e['reached_top1'] else ''} \\\\")
            L.append(r"\bottomrule\end{tabular}")
            if legacy:
                cap = ("Legacy exploratory concept recovery searched prompt "
                       "positions; peak depth is descriptive and sustained onset "
                       "was not recorded.")
            else:
                cap = ("Concept recovery over the predetermined score span and "
                       "fixed workspace band; onset is the first sustained "
                       "threshold crossing.")
            if rec["tracked_dropped"]:
                cap += (" Dropped (multi-token, \\S9.1): "
                        + tex_escape(", ".join(rec["tracked_dropped"])) + ".")
            L.append(rf"\caption{{{cap}}}\end{{table}}")

            if rec["slug"] in ana_by_slug:
                L.append(r"\paragraph{What we see.}")
                L.append(md_to_tex(ana_by_slug[rec["slug"]]["analysis"]))

            if rec.get("surprising"):
                L += _surprising_block(rec["surprising"])

            if rec.get("swap"):
                L += _swap_block(rec["swap"])

            for kind in FIG_ORDER:
                fp = _fig_abspath(rec, base_dir, kind)
                if fp:
                    L.append(r"\begin{figure}[H]\centering")
                    L.append(rf"\includegraphics[width={FIG_WIDTH[kind]}]{{{fp}}}")
                    captions = LEGACY_FIG_CAPTION if legacy else FIG_CAPTION
                    L.append(rf"\caption{{{captions[kind]}}}")
                    L.append(r"\end{figure}")

    L.append(r"\clearpage\section*{Limitations}")
    L.append(r"\begin{itemize}")
    L.append(r"\item \textbf{Single-token vocabulary} (\S9.1): multi-token "
             r"concepts cannot be tracked and are dropped (listed per prompt). "
             r"A larger vocabulary (e.g.\ Gemma's 256k) keeps more scientific "
             r"terms trackable.")
    L.append(r"\item \textbf{Model capability gates everything}: a concept "
             r"surfaces only if the model has it; reproducing the paper's "
             r"protein-sequence or abstraction results needs a capable, ideally "
             r"science-exposed model.")
    if legacy:
        L.append(r"\item \textbf{Legacy methodology}: this run lacks the v2 "
                 r"provenance and fixed-scoring record. Its workspace band and "
                 r"prompt-position selection cannot be audited as preregistered, "
                 r"so it is suitable only for qualitative inspection.")
    else:
        L.append(r"\item \textbf{Fixed workspace band}: the report uses the "
                 r"band recorded in the run metadata. For a new model family, "
                 r"calibrate or preregister that band on data disjoint from the "
                 r"reported test items.")
    L.append(r"\item \textbf{The LLM analysis is an interpretive aid} that reads "
             r"the same figures you can; verify its claims against the numbers "
             r"in \texttt{runs/*.json}.")
    L.append(r"\end{itemize}")
    L.append(r"\end{document}")

    reports = base_dir / "reports"
    reports.mkdir(exist_ok=True)
    tex_path = reports / f"{run_path.stem}_report.tex"
    tex_path.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {tex_path}")
    return tex_path


def compile_pdf(tex_path: Path) -> Path | None:
    env = dict(os.environ)
    env["PATH"] = _TEXBIN + os.pathsep + env.get("PATH", "")
    workdir = tex_path.parent
    latexmk = shutil.which("latexmk", path=env["PATH"])
    pdflatex = shutil.which("pdflatex", path=env["PATH"])
    if latexmk:
        cmd = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
               tex_path.name]
        runs = [cmd]
    elif pdflatex:
        cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
               tex_path.name]
        runs = [cmd, cmd]  # twice for refs
    else:
        print("  no latexmk/pdflatex found; wrote .tex only "
              "(install MacTeX/TeX Live to compile)")
        return None
    for c in runs:
        r = subprocess.run(c, cwd=workdir, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            log = (workdir / (tex_path.stem + ".log"))
            tail = log.read_text(errors="ignore")[-1500:] if log.is_file() else r.stdout[-1500:]
            print(f"  LaTeX compile failed:\n{tail}")
            return None
    pdf = tex_path.with_suffix(".pdf")
    print(f"wrote {pdf}" if pdf.is_file() else "  compile produced no PDF")
    return pdf if pdf.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<model>.json")
    ap.add_argument("--no-compile", action="store_true",
                    help="write the .tex only, skip PDF compilation")
    args = ap.parse_args()
    tex = build_tex(args.run)
    if not args.no_compile:
        compile_pdf(tex)


if __name__ == "__main__":
    main()
