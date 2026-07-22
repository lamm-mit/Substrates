# Copyright 2026.  Apache-2.0.
"""LLM-powered analysis of a Jacobian-lens run.

Reads a run record written by ``run_lens.py`` (``runs/<model>.json``) and, for
each prompt, asks a frontier model to interpret *what the figures show* —
grounded in both the numeric per-layer readouts AND the rendered figures
themselves (sent as images via the vision API).  It then writes a run-level
synthesis.  Output is ``runs/<model>_analysis.json`` (structured) which
``report.py`` / ``report_latex.py`` turn into a report.

This automates the "what we see" write-up: instead of a human squinting at each
trajectory, a capable model reads the lens output through the lens of the
global-workspace paper and says what surfaced, in what order, and whether it is
signal or a small-model artifact.

Provider choice (``--provider``): ``anthropic`` (Claude, default Opus 4.8) or
``openai`` (GPT-5.5).  Auth uses each provider's standard env var
(``ANTHROPIC_API_KEY`` / an ``ant auth login`` profile, or ``OPENAI_API_KEY``).
With no credentials it falls back to a deterministic template written straight
from the numbers, so the report pipeline still runs — clearly marked ``offline``.

    python analyze.py --run runs/qwen2forcausallm.json                       # Claude Opus 4.8
    python analyze.py --run runs/qwen2forcausallm.json --provider openai      # GPT-5.5
    python analyze.py --run runs/qwen2forcausallm.json --provider anthropic --model claude-fable-5
    python analyze.py --run runs/qwen2forcausallm.json --offline             # skip the API
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import llm

SYSTEM = """\
You are an interpretability researcher analysing Jacobian-lens ("J-lens") \
readouts, in the framework of the paper "Verbalizable Representations Form a \
Global Workspace in Language Models" (Anthropic, 2026).

Background you can rely on:
- The J-lens transports an intermediate residual-stream activation into the \
final-layer basis and decodes it with the model's own unembedding, giving the \
ranked vocabulary tokens the activation is *disposed to make the model say* — \
including concepts the model never actually outputs.
- Layers are reindexed to depth 0-100. The paper finds a mid-to-late \
"workspace band" where interpretable, verbalizable content lives; the last few \
layers switch to representing the literal next output token; early layers are \
usually uninterpretable.
- Key claims to test against the data: (1) unspoken *intermediate* concepts \
surface in the lens before the answer; (2) in multi-step reasoning the \
intermediates appear *in the order they are computed*, at successively later \
layers; (3) the lens can only track concepts that are single vocabulary tokens \
(a real limitation); (4) a concept only surfaces if the model actually has it \
— on a weak model the lens faithfully reports its absence.

Protocol discipline:
- Treat the recorded fixed score span and workspace band as the only scored \
region. Do not promote an all-position visual pattern into a metric.
- Compare every J-lens rank with the supplied vanilla logit-lens rank. A J-lens \
signal is most interesting when it is substantially stronger than the matched \
logit baseline; say plainly when the logit lens is equal or better.
- Respect the clean-answer and input/output-absence controls. If an item is \
excluded, it may still contain a qualitative pattern, but never call it a clean \
or registered protocol success. State the exclusion reason.
- This run may be exploratory or sample-insufficient. Do not infer population \
rates from a handful of prompts.

Write a precise, honest analysis. Distinguish genuine signal from small-model \
or single-token artifacts. When the figure shows a real ordered emergence or a \
clean concept, say so and cite the depths/ranks. When it shows nothing (flat \
lines, only subword fragments), say that plainly and explain why. Do not \
overclaim. Use complete sentences and name the concepts explicitly.
"""


def _img_block(path: Path) -> dict | None:
    """Neutral image block (provider-agnostic); llm.py translates per provider."""
    if not path.is_file():
        return None
    return {"type": "image_png",
            "data": base64.standard_b64encode(path.read_bytes()).decode()}


def _prompt_user_content(rec: dict, base_dir: Path) -> list[dict]:
    """Text summary of the numeric readouts + the rendered figures as images."""
    full_prompt = (rec.get("prompt_text") or rec.get("resolved_prompt")
                   or rec.get("prompt_used"))
    if full_prompt:
        prompt_label = "Exact prompt used"
        prompt_text = full_prompt
    else:
        prompt_label = "Prompt tail"
        prompt_text = ("..." + rec["prompt_tail"]) if rec.get("prompt_tail") else ""
    lines = [
        f"# Prompt: {rec['title']}  ({rec['shape']} / {rec['domain']})",
        f"Description: {rec['description']}",
        f"Scientific rationale: {rec['note']}",
        f"{prompt_label}:\n{prompt_text}",
        f"Recorded fixed workspace band: {rec['band'][0]}-{rec['band'][1]}% depth",
        f"Protocol: {rec.get('protocol', 'legacy/unknown')}",
        f"Registered metric status: "
        f"{'included' if rec.get('valid_for_metrics') else 'excluded'}",
        "",
    ]
    baseline = rec.get("baseline", {})
    if baseline.get("required"):
        lines += [
            "## Clean-answer control:",
            f"  expected first-token answer(s): {baseline.get('expected', [])}",
            f"  model greedy token: {baseline.get('greedy_token')!r}",
            f"  registered correct: {baseline.get('correct')}",
        ]
    else:
        lines += ["## Clean-answer control: not required for this item."]
    if rec.get("generated_completion") is not None:
        lines.append(
            f"  generated completion used for output-absence checking: "
            f"{rec.get('generated_completion')!r}"
        )
    if rec.get("excluded_reasons"):
        lines.append(
            "  exclusion reason(s): " + "; ".join(rec["excluded_reasons"])
        )
    lines += [
        "",
        ("## Auto-discovered rank comparison (exploratory):"
         if rec.get("discovered") else
         "## Tracked-rank comparison over the fixed score span and band:"),
    ]
    for e in rec["emergence"]:
        depth = "n/a" if e["best_depth"] is None else f"{e['best_depth']}%"
        star = "  <- reached rank 1" if e["reached_top1"] else ""
        logit_rank = e.get("logit_lens_best_rank")
        logit_text = "n/a" if logit_rank is None else str(logit_rank + 1)
        lines.append(
            f"  - {e['label']}: J-lens rank {e['best_rank'] + 1} at depth "
            f"{depth}; logit-lens rank {logit_text}; sustained onset "
            f"{e.get('onset_depth')}{star}"
        )
    if rec["tracked_dropped"]:
        lines.append(f"  (dropped — multi-token, not trackable: "
                     f"{rec['tracked_dropped']})")
    lines += ["", "## Per-layer top lens tokens at the readout position "
              "(depth: tokens):"]
    for r in rec["layer_readouts"]:
        toks = ", ".join(r["top_tokens"][:6])
        lines.append(f"  L{r['layer']:>2} (depth {r['depth']:>5}%): {toks}")
    surp = rec.get("surprising")
    if surp:
        lines += ["", "## Exploratory surfaced candidates — heuristic top-1 "
                  "tokens absent from the prompt and, when output_checked=true, "
                  "from an actual generated/teacher-forced completion:"]
        for s in surp[:10]:
            lines.append(f"  - {s['concept']} (tops {s['score']:g} band cells; "
                         f"peak rank {s['best_rank']} at depth {s['best_depth']}%, "
                         f"near '{s['near_token']}')")

    sw = rec.get("swap")
    if sw:
        lines += ["", "## Causal swap (intervention, not read-out):"]
        if sw.get("protocol") == "verbal_report" and sw.get("trials"):
            lines += [
                f"  verbal-report clean source: {sw.get('clean_source')!r}",
                f"  candidate trials: {len(sw['trials'])}; protocol success "
                f"rate: {sw.get('protocol_success_rate')}",
            ]
            for trial in sw["trials"]:
                same_surface = (
                    trial["target"].strip().casefold()
                    == sw.get("clean_source", "").casefold()
                )
                top_after = trial.get("swapped_top", [["", 0]])[0][0]
                lines.append(
                    f"  - target {trial['target']!r}"
                    + (" [case-only self-target; not a sibling concept]" if same_surface else "")
                    + f": source rank {trial['source_rank_clean']+1} -> "
                    f"{trial['source_rank_swapped']+1}; target rank "
                    f"{trial['target_rank_clean']+1} -> "
                    f"{trial['target_rank_swapped']+1}; swapped top-1 "
                    f"{top_after!r}; protocol success {trial.get('protocol_success')}"
                )
        else:
            lines += [
                f"  J-lens coordinate {sw['source']} -> {sw['target']} "
                f"(alpha={sw['alpha']}) at every band layer.",
                f"  clean next-token top: "
                f"{[w for w, _ in sw['clean_top'][:5]]}",
                f"  after-swap next-token top: "
                f"{[w for w, _ in sw['swapped_top'][:5]]}",
                f"  source '{sw['source']}' full-vocab rank "
                f"{sw['source_rank_clean']+1} -> {sw['source_rank_swapped']+1} "
                f"(higher = more suppressed); target '{sw['target']}' rank "
                f"{sw['target_rank_clean']+1} -> {sw['target_rank_swapped']+1} "
                f"(lower = more installed).",
            ]

    lines += ["", "The attached images are the rank-trajectory chart and the "
              "layer x position slice grid for this prompt. Analyse what the "
              "lens reveals about the model's unspoken reasoning here"
              + (", and whether the causal swap confirms it (did suppressing the "
                 "source concept and adding the target actually change the "
                 "output, or only suppress?)" if sw else "")
              + (". Explicitly review the exploratory candidate list as a "
                 "heuristic: which entries may be meaningful and which are "
                 "tokenizer/noise artifacts? Do not treat it as a paper metric."
                 if surp else "")
              + " Keep it to 1-3 tight paragraphs."]

    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    figs = rec.get("figures", {})
    for key in ("trajectory", "grid"):
        if key in figs:
            blk = _img_block(base_dir / figs[key])
            if blk:
                content.append(blk)
    return content


def _offline_analysis(rec: dict) -> str:
    """Deterministic template from the numbers, when no API is available."""
    hits = [e for e in rec["emergence"] if 0 <= e["best_rank"] < 5]
    top1 = [e for e in rec["emergence"] if e["reached_top1"]]
    parts = []
    if top1:
        names = ", ".join(f"'{e['label']}' (depth {e['best_depth']}%)"
                          for e in top1)
        parts.append(f"Concepts reaching lens rank 1: {names}.")
    if hits:
        names = ", ".join(f"'{e['label']}' (rank {e['best_rank']+1}, "
                          f"depth {e['best_depth']}%)" for e in hits)
        parts.append(f"Concepts reaching the lens top-5: {names}.")
    if not hits:
        parts.append("No tracked concept reached the lens top-5 at any depth "
                     "— either the model lacks these concepts or they are "
                     "multi-token and untrackable.")
    if rec["tracked_dropped"]:
        parts.append(f"Not trackable (multi-token, paper limitation section 9.1): "
                     f"{', '.join(rec['tracked_dropped'])}.")
    surp = rec.get("surprising")
    if surp:
        parts.append("Exploratory surfaced candidates (heuristic, prompt/output "
                     "filtered where an actual completion was available): "
                     + ", ".join(s["concept"] for s in surp[:6]) + ".")
    return "[offline template] " + " ".join(parts)


def analyze(run_path: str, *, provider: str = "anthropic",
            model: str | None = None, offline: bool = False,
            base_url: str | None = None) -> dict:
    model = model or llm.default_model(provider)
    run = json.loads(Path(run_path).read_text())
    base_dir = Path(run_path).resolve().parent.parent  # jlens_materials/

    analyses = []
    for rec in run["prompts"]:
        if offline:
            text, used = _offline_analysis(rec), "offline"
        else:
            try:
                blocks = _prompt_user_content(rec, base_dir)
                text = llm.complete(provider, model, SYSTEM, blocks,
                                    base_url=base_url)
                used = f"{provider}:{model}"
            except Exception as exc:  # noqa: BLE001
                print(f"  [{rec['slug']}] {provider} error: "
                      f"{type(exc).__name__}: {exc}; using offline template")
                text, used = _offline_analysis(rec), "offline"
                offline = True  # stop retrying a dead provider
        print(f"  [{rec['slug']}] analysed ({used})")
        analyses.append({"slug": rec["slug"], "title": rec["title"],
                         "shape": rec["shape"], "domain": rec["domain"],
                         "model_used": used, "analysis": text})

    synthesis = _synthesize(provider, model, run, analyses, offline, base_dir,
                            base_url=base_url)

    out = {"model": run["model"], "provider": provider, "analysis_model": model,
           "offline": offline, "per_prompt": analyses, "synthesis": synthesis}
    out_path = Path(run_path).with_name(Path(run_path).stem + "_analysis.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    return out


def top_hidden_thoughts(run, top=15):
    """Aggregate exploratory surfaced candidates across the run,
    summed by concept, strongest first: (concept, total_score, n_prompts)."""
    agg: dict[str, list] = {}
    for rec in run["prompts"]:
        for s in rec.get("surprising", []):
            e = agg.setdefault(s["concept"], [0.0, 0])
            e[0] += s["score"]; e[1] += 1
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])
    return [(c, round(sc, 1), n) for c, (sc, n) in ranked[:top]]


def _synthesize(provider, model, run, analyses, offline, base_dir=None,
                base_url=None) -> str:
    hidden = top_hidden_thoughts(run)
    hidden_str = ", ".join(f"{c} (×{n})" for c, _, n in hidden[:12])
    if offline:
        by_shape: dict[str, list[str]] = {}
        for a, rec in zip(analyses, run["prompts"]):
            hit = any(0 <= e["best_rank"] < 5 for e in rec["emergence"])
            by_shape.setdefault(a["shape"], []).append("hit" if hit else "miss")
        summary = "; ".join(
            f"{s}: {v.count('hit')}/{len(v)} prompts with a top-5 concept"
            for s, v in by_shape.items())
        extra = (f" Most frequent exploratory surfaced candidates across prompts: "
                 f"{hidden_str}." if hidden else "")
        return ("[offline template] Run over "
                f"{run['model']} ({run['n_layers']} layers). By experiment "
                f"type — {summary}.{extra}")
    joined = "\n\n".join(f"### {a['title']} ({a['shape']})\n{a['analysis']}"
                         for a in analyses)
    methodology = run.get("methodology", {})
    sufficiency = run.get("insufficient_sample_shapes", [])
    blocks = [{"type": "text", "text":
        f"Below are per-prompt J-lens analyses for model {run['model']} "
        f"({run['n_layers']} layers), across domains {run.get('domains', [])}. "
        f"Run claims level: {methodology.get('claims_level', 'legacy/unknown')}; "
        f"insufficient-sample groups: {sufficiency}. Do not make quantitative "
        f"paper-level claims for exploratory or insufficient groups. "
        f"Write a run-level synthesis (3-5 paragraphs): which experiment types "
        f"(MULTIHOP / ASSOCIATION / RECOGNITION / MODULATION / REPORT_SWAP) "
        f"showed the workspace phenomena most clearly, how the results compare "
        f"to the paper's claims, what the single-token and model-capability "
        f"limitations imply, and what a reader should conclude about using this "
        f"target model versus a larger or more capable one for scientific "
        f"interpretability. Respect every per-item metric exclusion in the "
        f"analyses below and distinguish qualitative signals from registered "
        f"protocol successes. "
        f"Also comment on the exploratory surfaced candidates (most frequent: "
        f"{hidden_str}), explicitly treating them as a heuristic rather than a "
        f"paper metric and separating plausible signals from artifacts.\n\n{joined}"}]
    _tag = run.get("tag")
    _figs = (base_dir or Path(".")) / "figures"
    hitbars = (_figs / _tag / "summary__hit_rates.png") if _tag else (_figs / "summary__hit_rates.png")
    blk = (_img_block(hitbars) if run.get("format_version", 1) < 2 and
           hitbars.is_file() else None)
    if blk:
        blocks.append(blk)
    try:
        return llm.complete(provider, model, SYSTEM, blocks, max_tokens=6000,
                            base_url=base_url)
    except Exception as exc:  # noqa: BLE001
        return f"[synthesis unavailable: {type(exc).__name__}: {exc}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<model>.json from run_lens.py")
    ap.add_argument("--provider", choices=["anthropic", "openai"],
                    default="anthropic",
                    help="anthropic (Claude, default Opus 4.8) or openai (GPT-5.5)")
    ap.add_argument("--model", default=None,
                    help="override the model id (default per provider: "
                         "claude-opus-4-8 / gpt-5.5)")
    ap.add_argument("--offline", action="store_true",
                    help="skip the API; deterministic template analysis")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible endpoint for a local model server "
                         "(mistral.rs, LM Studio, vLLM, Ollama), e.g. "
                         "http://localhost:1234/v1. Only used with "
                         "--provider openai; ignored for anthropic.")
    args = ap.parse_args()
    analyze(args.run, provider=args.provider, model=args.model,
            offline=args.offline, base_url=args.base_url)


if __name__ == "__main__":
    main()
