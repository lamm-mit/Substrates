# Copyright 2026.  Apache-2.0.
"""Assemble a Jacobian-lens run + its LLM analysis into a Markdown report.

Reads ``runs/<model>.json`` (figures + numeric records from ``run_lens.py``)
and ``runs/<model>_analysis.json`` (the "what we see" write-ups from
``analyze.py``) and produces ``reports/<model>_report.md``: an executive
synthesis, then one section per prompt embedding its four figures and the
model's analysis, then a limitations note.

    python report.py --run runs/qwen2forcausallm.json

This closes the loop: fit -> apply -> figures -> LLM analysis -> report.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import domain_prompts as dp

FIG_ORDER = ["grid", "stream", "trajectory", "emergence", "heatmap"]
FIG_CAPTION = {
    "grid": "Layer x position slice: the top-1 lens word per cell, shaded by "
            "rank. Reads the model's disposition across depth and position.",
    "stream": "Thought stream: each surfaced concept flows across depth (band "
              "thickness ~ salience) — the dynamics of what the model is "
              "disposed to say.",
    "trajectory": "Rank of each tracked concept vs depth over the predeclared "
                  "score span (rank 1 at top). A sustained line toward the top "
                  "marks workspace entry.",
    "emergence": "First sustained threshold crossing in the fixed band; peak rank is annotated.",
    "heatmap": "Position x layer rank map for the single strongest concept.",
}
LEGACY_FIG_CAPTION = {
    **FIG_CAPTION,
    "trajectory": "Legacy exploratory trajectory: best tracked-concept rank "
                  "was searched over all prompt positions at each depth.",
    "emergence": "Legacy peak-depth summary, not the fixed-band sustained-onset "
                 "metric used by format-v2 runs.",
}

_BUILTIN_BY_SLUG = {p.slug: p for p in dp.ALL_PROMPTS}


def _authored_prompt_text(rec: dict) -> tuple[str, str]:
    """Prompt text for reports.

    New run records store ``prompt_text``: the exact resolved string fed to the
    tokenizer/model, including chat-template tokens for instruct models. Older
    run JSONs only stored ``prompt_tail``; for built-in prompts we can still
    show the full authored prompt fields so readers get the scientific context.
    """
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


def _code_fence(text: str) -> str:
    max_ticks = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    return "`" * max(3, max_ticks + 1)


def _fig_paths(rec: dict, base_dir: Path) -> dict[str, str]:
    """Map figure-kind -> markdown-relative path (relative to the report file)."""
    out = {}
    for kind, rel in rec.get("figures", {}).items():
        p = (base_dir / rel)
        if p.is_file():
            # report lives in reports/, figures in figures/ — use ../
            out[kind] = "../" + rel
    return out


def build(run_path: str) -> Path:
    run_path = Path(run_path)
    run = json.loads(run_path.read_text())
    base_dir = run_path.resolve().parent.parent  # jlens_materials/
    analysis_path = run_path.with_name(run_path.stem + "_analysis.json")
    analysis = (json.loads(analysis_path.read_text())
                if analysis_path.is_file() else None)
    ana_by_slug = ({a["slug"]: a for a in analysis["per_prompt"]}
                   if analysis else {})
    legacy = run.get("format_version", 1) < 2

    md: list[str] = []
    md.append(f"# Jacobian-lens report — `{run['model']}`\n")
    lens_n = run.get("lens_n_prompts", "?")
    md.append(f"*{run['n_layers']} layers, d_model {run['d_model']}, "
              f"lens fitted on {lens_n} prompts. Experiment shapes: "
              f"{', '.join(run['shapes'])}. Domains: "
              f"{', '.join(run['domains'])}.*\n")
    methodology = run.get("methodology", {})
    if methodology:
        md.append(
            f"> Claims level: **{methodology.get('claims_level', 'unknown')}**. "
            f"Recipe `{methodology.get('recipe', {}).get('name', '?')}`; fixed "
            f"workspace band {methodology.get('workspace_band', '?')}; scoring "
            f"uses predetermined spans and synonym-min ranks.\n"
        )
    elif legacy:
        md.append(
            "> **Legacy exploratory run:** this record predates lens provenance, "
            "fixed-position scoring, and preregistered-band metadata. Its figures "
            "are descriptive only; do not use its hit rates as paper-protocol "
            "results.\n"
        )
    if run.get("errors"):
        md.append(
            f"> **Run incomplete:** {len(run['errors'])} prompt(s) failed. "
            "See the run JSON for the recorded exception messages.\n"
        )
    if run.get("insufficient_sample_shapes"):
        md.append(
            "> **Sample-size guard:** quantitative claims are disabled for "
            f"{', '.join(run['insufficient_sample_shapes'])}.\n"
        )
    if analysis:
        tag = ("offline template — no LLM was used"
               if analysis["offline"]
               else f"analysis by `{analysis['analysis_model']}`")
        md.append(f"> Interpretation {tag}.\n")

    # --- executive synthesis ------------------------------------------------
    if analysis and analysis.get("synthesis"):
        md.append("## Executive synthesis\n")
        md.append(analysis["synthesis"] + "\n")

    # run-level hidden thoughts (unprompted concepts aggregated across prompts)
    agg: dict[str, list] = {}
    for rec in run["prompts"]:
        for s in rec.get("surprising", []):
            e = agg.setdefault(s["concept"], [0.0, 0])
            e[0] += s["score"]; e[1] += 1
    if agg:
        ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])[:15]
        md.append("## Exploratory surfaced candidates across the run\n")
        md.append("Heuristic top-ranked candidates, filtered against prompts and "
                  "actual completions where available. These are descriptive and "
                  "are not included in paper-protocol metrics.\n")
        md.append("| concept | total band cells | # prompts |")
        md.append("|--|--|--|")
        for c, (sc, n) in ranked:
            md.append(f"| {c} | {sc:g} | {n} |")
        md.append("")

    by_shape = run.get("metrics", {}).get("by_shape", {})
    if by_shape:
        md.append("## Item-level lens evaluation\n")
        md.append("| shape | independent items | sample sufficient? | J-lens AUC | logit-lens AUC |")
        md.append("|---|---:|---|---:|---:|")
        for shape, metric in by_shape.items():
            md.append(
                f"| {shape} | {metric['n_items']} | "
                f"{'yes' if metric['sufficient_sample'] else 'no'} | "
                f"{metric['jacobian_lens']['auc_log_k']:.3f} | "
                f"{metric['logit_lens']['auc_log_k']:.3f} |"
            )
            tag = run.get("tag")
            if tag:
                image = f"../figures/{tag}/summary__{shape.lower()}__pass_at_k.png"
                md.append(f"\n![{shape} pass at k]({image})\n")

    modulation = run.get("metrics", {}).get("directed_modulation_controls", {})
    if modulation:
        md.append("## Directed-modulation controls\n")
        md.append("| control group | condition | items | hit rate | sample sufficient? |")
        md.append("|---|---|---:|---:|---|")
        for group, conditions in modulation.items():
            for condition in ("focus", "suppress", "control"):
                metric = conditions.get(condition)
                if not metric:
                    continue
                md.append(
                    f"| {group} | {condition} | {metric['n_items']} | "
                    f"{metric['hit_rate']:.3f} | "
                    f"{'yes' if metric['sufficient_sample'] else 'no'} |"
                )
            summary = conditions.get("summary", {})
            if summary:
                md.append(
                    f"\n*{group}: {summary.get('distinct_phrasings', 0)} distinct "
                    f"phrasings; paper target {summary.get('paper_phrasing_target', 24)}.*\n"
                )

    causal = run.get("metrics", {}).get("causal_swaps", {})
    if causal.get("n_interventions", 0):
        md.append("## Causal-intervention summary\n")
        md.append("| interventions | graded | protocol success | counterfactual items | causal success | sufficient? |")
        md.append("|---:|---:|---:|---:|---:|---|")
        protocol_rate = causal.get("protocol_success_rate")
        causal_rate = causal.get("causal_success_rate")
        md.append(
            f"| {causal['n_interventions']} | {causal['n_graded']} | "
            f"{'—' if protocol_rate is None else f'{protocol_rate:.3f}'} | "
            f"{causal.get('n_counterfactual', 0)} | "
            f"{'—' if causal_rate is None else f'{causal_rate:.3f}'} | "
            f"{'yes' if causal['sufficient_sample'] else 'no'} |"
        )

    # --- per-prompt sections, grouped by domain -----------------------------
    for domain in run["domains"]:
        recs = [r for r in run["prompts"] if r["domain"] == domain]
        if not recs:
            continue
        md.append(f"\n---\n\n## Domain: {domain}\n")
        for rec in recs:
            md.append(f"### {rec['title']}  ·  `{rec['shape']}`\n")
            md.append(f"{rec['description']}\n")
            prompt_label, prompt_text = _authored_prompt_text(rec)
            fence = _code_fence(prompt_text)
            md.append(f"**{prompt_label}**\n\n{fence}text\n{prompt_text}\n{fence}\n")
            if "valid_for_metrics" in rec:
                if rec["valid_for_metrics"]:
                    md.append("**Metric status:** included in the registered aggregate.\n")
                else:
                    reasons = rec.get("excluded_reasons") or ["not a registered metric item"]
                    md.append(
                        "**Metric status:** excluded — " + "; ".join(reasons) + ".\n"
                    )
                baseline = rec.get("baseline", {})
                if baseline.get("required"):
                    expected = ", ".join(baseline.get("expected", []))
                    greedy = baseline.get("greedy_token", "")
                    md.append(
                        f"**Clean baseline:** expected `{expected}`; greedy token "
                        f"`{greedy}`; correct: "
                        f"{'yes' if baseline.get('correct') else 'no'}.\n"
                    )
                if rec.get("generated_completion"):
                    # JSON quoting preserves whitespace-only completions (for
                    # example two newline tokens) as visible, auditable text.
                    completion = json.dumps(
                        rec["generated_completion"], ensure_ascii=False
                    )
                    completion_fence = _code_fence(completion)
                    md.append(
                        f"**Generated completion used for output-absence control**\n\n"
                        f"{completion_fence}text\n{completion}\n{completion_fence}\n"
                    )
            # concept emergence table
            md.append("| concept | J-lens best rank | logit-lens best rank | peak depth | sustained onset | rank-1? |")
            md.append("|---|---:|---:|---:|---:|---|")
            for e in rec["emergence"]:
                depth = "—" if e["best_depth"] is None else f"{e['best_depth']}%"
                onset = ("—" if e.get("onset_depth") is None
                         else f"{e['onset_depth']}%")
                logit_rank = e.get("logit_lens_best_rank")
                logit_display = "—" if logit_rank is None else str(logit_rank + 1)
                md.append(f"| {e['label']} | {e['best_rank'] + 1} | {logit_display} | "
                          f"{depth} | {onset} | "
                          f"{'yes' if e['reached_top1'] else ''} |")
            if rec["tracked_dropped"]:
                md.append(f"\n*Dropped (multi-token, §9.1): "
                          f"{', '.join(rec['tracked_dropped'])}.*")
            md.append("")
            if legacy:
                md.append(
                    "*Legacy scoring searched prompt positions; peak depth is "
                    "descriptive and sustained onset was not recorded.*\n"
                )

            # analysis
            if rec["slug"] in ana_by_slug:
                md.append("**What we see** — " + ana_by_slug[rec["slug"]]["analysis"]
                          + "\n")

            # hidden thoughts (unprompted internal concepts)
            surp = rec.get("surprising")
            if surp:
                md.append("**Exploratory surfaced candidates** — heuristic "
                          "top-ranked tokens absent from the prompt and, where "
                          "checked, an actual generated/teacher-forced completion:")
                md.append("\n| concept | band cells | peak rank | near token |")
                md.append("|--|--|--|--|")
                for s in surp[:10]:
                    md.append(f"| {s['concept']} | {s['score']:g} | "
                              f"{s['best_rank']} | {s['near_token']} |")
                md.append("")

            # causal swap
            sw = rec.get("swap")
            if sw:
                if sw.get("protocol") == "verbal_report" and sw.get("trials"):
                    clean_source = sw.get("clean_source", sw.get("source", "?"))
                    md.append(
                        f"**Multi-candidate verbal-report swaps** — clean report "
                        f"`{clean_source}`; success rate "
                        f"{100 * sw.get('protocol_success_rate', 0):.1f}%."
                    )
                    md.append("\n| target | source rank | target rank | swapped top-1 | success |")
                    md.append("|---|---:|---:|---|---|")
                    for trial in sw["trials"]:
                        target = trial["target"]
                        if target.strip().casefold() == str(clean_source).casefold():
                            target += " (case-only self-target)"
                        top_after = trial.get("swapped_top", [["", 0]])[0][0]
                        md.append(
                            f"| {target} | {trial['source_rank_clean']+1}→"
                            f"{trial['source_rank_swapped']+1} | "
                            f"{trial['target_rank_clean']+1}→"
                            f"{trial['target_rank_swapped']+1} | "
                            f"{top_after or '·'} | "
                            f"{'yes' if trial.get('protocol_success') else 'no'} |"
                        )
                    md.append(
                        "\n*Case-only self-targets are diagnostic noise and are "
                        "excluded by the current runner.*\n"
                    )
                else:
                    md.append(f"**Causal swap (Fig 4C)** — J-lens coordinate "
                              f"`{sw['source']}` → `{sw['target']}` (α={sw['alpha']}) "
                              f"at band layers {sw['band_layers'][0]}–{sw['band_layers'][-1]}:")
                    md.append("\n| # | clean | prob | after swap | prob |")
                    md.append("|--|--|--|--|--|")
                    for i, ((cw, cp), (sww, sp)) in enumerate(
                            zip(sw["clean_top"], sw["swapped_top"]), 1):
                        md.append(f"| {i} | {cw or '·'} | {cp:.3f} | {sww or '·'} | {sp:.3f} |")
                    md.append(f"\n*Source `{sw['source']}` rank "
                              f"{sw['source_rank_clean']+1}→{sw['source_rank_swapped']+1} "
                              f"(higher = suppressed); target `{sw['target']}` rank "
                              f"{sw['target_rank_clean']+1}→{sw['target_rank_swapped']+1} "
                              f"(lower = installed).*\n")

            # figures
            figs = _fig_paths(rec, base_dir)
            for kind in FIG_ORDER:
                if kind in figs:
                    md.append(f"![{kind}]({figs[kind]})")
                    captions = LEGACY_FIG_CAPTION if legacy else FIG_CAPTION
                    md.append(f"*{captions[kind]}*\n")

    # --- limitations --------------------------------------------------------
    md.append("\n---\n\n## Limitations\n")
    md.append("- **Single-token vocabulary** (paper §9.1): multi-token "
              "concepts cannot be tracked and are dropped (listed per prompt). "
              "A larger vocabulary (e.g. Gemma's 256k) keeps more scientific "
              "terms trackable.\n"
              "- **Model capability gates everything**: a concept only surfaces "
              "if the model has it. Reproducing the paper's protein-sequence or "
              "abstraction results needs a capable, ideally science-exposed "
              "model.\n")
    if legacy:
        md.append(
            "- **Legacy methodology:** this run lacks the v2 provenance and "
            "fixed-scoring record. Its workspace band and prompt-position "
            "selection cannot be audited as preregistered, so it is suitable "
            "only for qualitative inspection."
        )
    else:
        md.append(
            "- **Fixed workspace band:** the report uses the band recorded in "
            "the run metadata. For a new model family, calibrate or preregister "
            "that band on data disjoint from the reported test items."
        )

    reports = base_dir / "reports"
    reports.mkdir(exist_ok=True)
    out = reports / f"{run_path.stem}_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<model>.json")
    args = ap.parse_args()
    build(args.run)


if __name__ == "__main__":
    main()
