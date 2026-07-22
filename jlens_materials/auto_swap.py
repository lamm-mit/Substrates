# Copyright 2026.  Apache-2.0.
"""Suggest and run J-lens coordinate swaps from an existing run.

The normal run pipeline is discovery-first: fit/apply a lens, inspect what
concepts surfaced, then decide what causal swap to test. This CLI automates the
middle of that loop:

    runs/<tag>.json -> proposals -> approval -> alpha sweep -> plots/json

The default provider is offline, using deterministic same-category domain
lexicons. Anthropic/OpenAI can be used to refine proposals, but the final list is
still filtered against concepts that actually surfaced in the run.

These automatically proposed same-domain swaps are manipulation sanity checks.
They become graded causal experiments only when the item declares a clean and
counterfactual answer in advance.

Examples:
    python auto_swap.py --run runs/qwen2forcausallm.json --dry-run
    python auto_swap.py --run runs/qwen2forcausallm.json --model Qwen/Qwen2.5-0.5B-Instruct \
        --lens lens_qwen2forcausallm.pt --provider offline --auto-approve --slugs fracture-report-swap
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "jlens-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

import llm
import swap as swapmod
import domain_prompts as dp


HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Domain lexicons
# --------------------------------------------------------------------------- #

DOMAIN_GROUPS: dict[str, dict[str, list[str]]] = {
    "fracture": {
        "failure_modes": [
            "fracture", "fatigue", "creep", "corrosion", "buckling",
            "cleavage", "wear",
        ],
        "mechanisms": [
            "crack", "dislocation", "plastic", "yield", "deformation",
            "brittle", "ductile", "necking",
        ],
        "fracture_parameters": [
            "stress", "strain", "intensity", "toughness", "energy",
            "surface", "length", "propagate", "opening",
        ],
    },
    "protein": {
        "secondary_structure": [
            "helix", "sheet", "strand", "turn", "coil", "backbone",
            "hydrogen", "bond",
        ],
        "folding": [
            "protein", "sequence", "amino", "residue", "fold", "structure",
            "native", "core", "hydrophobic",
        ],
        "residues": [
            "glycine", "alanine", "leucine", "serine", "proline",
            "cysteine",
        ],
        "membrane": [
            "membrane", "hydrophobic", "transmembrane", "lipid", "helix",
        ],
        "aggregation": [
            "amyloid", "fibril", "aggregate", "misfold", "beta", "plaque",
        ],
        "chemistry": [
            "disulfide", "cysteine", "bridge", "bond", "sulfur", "cross",
        ],
    },
}


@dataclass
class Proposal:
    slug: str
    title: str
    domain: str
    shape: str
    source: str
    target: str
    source_rank: int
    source_depth: float | None
    rationale: str
    expected: str
    confidence: float
    provider: str = "offline"
    target_token_id: int | None = None
    source_token_id: int | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_run(run_path: str | Path) -> dict:
    return json.loads(Path(run_path).read_text())


def _base_dir(run_path: str | Path) -> Path:
    p = Path(run_path).resolve()
    return p.parent.parent if p.parent.name == "runs" else HERE


def _csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def _clean_word(s: str) -> str:
    return re.sub(r"^\s+", "", str(s or "")).strip()


def _norm_word(s: str) -> str:
    return _clean_word(s).lower()


def _rank1(e: dict) -> int:
    return int(e.get("best_rank", 10**9)) + 1


def _source_rank0(e: dict) -> int:
    return int(e.get("best_rank", 10**9))


def _depth(e: dict) -> float | None:
    d = e.get("best_depth")
    return None if d is None else float(d)


def _single_token_id(tokenizer, word: str) -> int | None:
    forms = [" " + word, word, " " + word.capitalize(), word.capitalize(),
             " " + word.lower(), word.lower()]
    for form in forms:
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def _target_candidates(domain: str, source: str, prompt_labels: list[str]) -> list[str]:
    """Same-category replacements from the domain lexicon, then prompt siblings."""
    src = _norm_word(source)
    out: list[str] = []
    for group in DOMAIN_GROUPS.get(domain, {}).values():
        if src in {_norm_word(x) for x in group}:
            for item in group:
                if _norm_word(item) != src and item not in out:
                    out.append(item)
    for label in prompt_labels:
        if _norm_word(label) != src and label not in out:
            out.append(label)
    return out


def _confidence_from_rank(rank0: int) -> float:
    # Rank 1 concepts get ~0.9; rank 20 concepts get ~0.62.
    return round(max(0.35, min(0.95, 0.92 - 0.015 * rank0)), 2)


def _proposal_key(p: Proposal) -> tuple[str, str, str]:
    return (_norm_word(p.slug), _norm_word(p.source), _norm_word(p.target))


def _filtered_prompts(run: dict, *, slugs: set[str] | None,
                      domains: set[str] | None) -> list[dict]:
    prompts = run.get("prompts", [])
    if slugs:
        prompts = [p for p in prompts if p.get("slug") in slugs]
    if domains:
        prompts = [p for p in prompts if p.get("domain") in domains]
    return prompts


def deterministic_proposals(
    run: dict,
    *,
    slugs: set[str] | None = None,
    domains: set[str] | None = None,
    max_source_rank: int = 20,
    allow_weak: bool = False,
    per_prompt: int = 2,
    max_proposals: int = 20,
) -> list[Proposal]:
    """Choose loaded source concepts and same-category targets without an LLM."""
    proposals: list[Proposal] = []
    for rec in _filtered_prompts(run, slugs=slugs, domains=domains):
        ems = rec.get("emergence", [])
        labels = [_clean_word(e.get("label", "")) for e in ems]
        band = rec.get("band") or [0.0, 100.0]
        band_lo, band_hi = float(band[0]), float(band[1])
        candidates = []
        for e in ems:
            label = _clean_word(e.get("label", ""))
            if not label or _depth(e) is None:
                continue
            depth = _depth(e)
            if depth is None or depth < band_lo or depth > band_hi:
                continue
            rank0 = _source_rank0(e)
            if rank0 < 0:
                continue
            if not allow_weak and rank0 >= max_source_rank:
                continue
            candidates.append(e)
        candidates.sort(key=lambda e: (_source_rank0(e), _depth(e) or 999.0))

        made_for_prompt = 0
        for e in candidates:
            source = _clean_word(e["label"])
            targets = _target_candidates(rec.get("domain", ""), source, labels)
            if not targets:
                continue
            target = targets[0]
            rank0 = _source_rank0(e)
            depth = _depth(e)
            proposals.append(Proposal(
                slug=rec.get("slug", ""),
                title=rec.get("title", rec.get("slug", "")),
                domain=rec.get("domain", ""),
                shape=rec.get("shape", ""),
                source=source,
                target=target,
                source_rank=rank0 + 1,
                source_depth=depth,
                rationale=(
                    f"'{source}' surfaced strongly in the run "
                    f"(rank {rank0 + 1} at depth {depth}%). "
                    f"'{target}' is a same-domain counterfactual target."
                ),
                expected=(
                    f"The swap should suppress '{source}', install '{target}', "
                    "and move the next-token distribution toward the target."
                ),
                confidence=_confidence_from_rank(rank0),
            ))
            made_for_prompt += 1
            if made_for_prompt >= per_prompt:
                break
        if len(proposals) >= max_proposals:
            break
    return proposals[:max_proposals]


# --------------------------------------------------------------------------- #
# LLM proposal refinement
# --------------------------------------------------------------------------- #

SYSTEM = """\
You propose causal Jacobian-lens coordinate swaps for scientific interpretability.
Use only source concepts that actually surfaced in the supplied J-lens run.
Prefer same-category targets, and avoid weak sources unless there is a clear
reason. Return ONLY strict JSON with this shape:
{"proposals":[{"slug":"...","source":"...","target":"...","rationale":"...",
"expected":"...","confidence":0.0}]}
"""


def _json_from_text(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _run_summary_for_llm(run: dict, seeds: list[Proposal]) -> str:
    lines = [
        f"Model: {run.get('model')}",
        "Candidate loaded concepts by prompt. Ranks are 1-based; lower is stronger.",
        "",
    ]
    for rec in run.get("prompts", []):
        rows = []
        for e in rec.get("emergence", []):
            if e.get("best_depth") is None or int(e.get("best_rank", -1)) < 0:
                continue
            rows.append(f"{e['label']}=rank {_rank1(e)}@{e['best_depth']}%")
        if rows:
            lines.append(f"- {rec['slug']} ({rec.get('domain')}/{rec.get('shape')}): "
                         + ", ".join(rows[:10]))
    lines += ["", "Deterministic seed proposals:"]
    for p in seeds:
        lines.append(
            f"- {p.slug}: {p.source}->{p.target}; source rank {p.source_rank}; "
            f"rationale: {p.rationale}"
        )
    lines += ["", "Return at most 12 proposals. Keep source strings exactly as shown."]
    return "\n".join(lines)


def llm_refine_proposals(
    run: dict,
    seeds: list[Proposal],
    *,
    provider: str,
    model: str | None,
    allow_weak: bool,
    max_source_rank: int,
) -> list[Proposal]:
    """Ask a provider to refine proposals, then filter against surfaced concepts."""
    model = model or llm.default_model(provider)
    text = llm.complete(
        provider,
        model,
        SYSTEM,
        [{"type": "text", "text": _run_summary_for_llm(run, seeds)}],
        max_tokens=2500,
        effort="medium",
    )
    parsed = _json_from_text(text)
    raw = parsed.get("proposals", parsed if isinstance(parsed, list) else [])

    rec_by_slug = {p.get("slug"): p for p in run.get("prompts", [])}
    out: list[Proposal] = []
    for item in raw:
        slug = str(item.get("slug", ""))
        rec = rec_by_slug.get(slug)
        if rec is None:
            continue
        source = _clean_word(item.get("source", ""))
        target = _clean_word(item.get("target", ""))
        if not source or not target or _norm_word(source) == _norm_word(target):
            continue

        match = None
        for e in rec.get("emergence", []):
            if _norm_word(e.get("label", "")) == _norm_word(source):
                match = e
                break
        if match is None:
            continue
        band = rec.get("band") or [0.0, 100.0]
        depth = _depth(match)
        if depth is None or depth < float(band[0]) or depth > float(band[1]):
            continue
        rank0 = _source_rank0(match)
        if not allow_weak and rank0 >= max_source_rank:
            continue
        conf = item.get("confidence", 0.5)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.5
        out.append(Proposal(
            slug=slug,
            title=rec.get("title", slug),
            domain=rec.get("domain", ""),
            shape=rec.get("shape", ""),
            source=source,
            target=target,
            source_rank=rank0 + 1,
            source_depth=_depth(match),
            rationale=str(item.get("rationale", "")).strip()
            or "LLM-proposed same-run counterfactual.",
            expected=str(item.get("expected", "")).strip()
            or f"Move the answer toward '{target}'.",
            confidence=round(max(0.0, min(1.0, conf)), 2),
            provider=f"{provider}:{model}",
        ))

    # De-duplicate while preserving provider order. Fall back to seeds if the
    # model returned nothing usable.
    seen = set()
    deduped = []
    for p in out:
        if _proposal_key(p) not in seen:
            seen.add(_proposal_key(p))
            deduped.append(p)
    return deduped or seeds


# --------------------------------------------------------------------------- #
# Token validation, approval, and summaries
# --------------------------------------------------------------------------- #

def validate_tokens(proposals: list[Proposal], model_name: str) -> list[Proposal]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    valid = []
    for p in proposals:
        sid = _single_token_id(tokenizer, p.source)
        tid = _single_token_id(tokenizer, p.target)
        if sid is None or tid is None:
            print(
                f"skip {p.slug}: {p.source}->{p.target} is not single-token "
                f"(source={sid}, target={tid})",
                file=sys.stderr,
            )
            continue
        p.source_token_id = sid
        p.target_token_id = tid
        valid.append(p)
    return valid


def _fmt_depth(depth: float | None) -> str:
    return "n/a" if depth is None or math.isnan(depth) else f"{depth:.1f}%"


def _short(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "."


def print_proposals(proposals: list[Proposal]) -> None:
    if not proposals:
        print("No swap proposals survived filtering.")
        return
    print("\nSwap proposals")
    print("-" * 118)
    print(f"{'#':>2}  {'prompt':<28} {'source -> target':<27} {'rank@depth':<14} "
          f"{'tok':<9} {'conf':<5} rationale")
    print("-" * 118)
    for i, p in enumerate(proposals, 1):
        tok = "?"
        if p.target_token_id is not None:
            tok = str(p.target_token_id)
        rank_depth = f"{p.source_rank}@{_fmt_depth(p.source_depth)}"
        pair = f"{p.source} -> {p.target}"
        print(f"{i:>2}  {_short(p.slug, 28):<28} {_short(pair, 27):<27} "
              f"{rank_depth:<14} {tok:<9} {p.confidence:<5.2f} "
              f"{_short(p.rationale, 45)}")
    print("-" * 118)


def approve_proposals(proposals: list[Proposal], *, auto_approve: bool) -> list[Proposal]:
    if auto_approve:
        return proposals
    if not sys.stdin.isatty():
        raise SystemExit("Refusing to run swaps without an interactive terminal. "
                         "Use --auto-approve or --dry-run.")
    approved: list[Proposal] = []
    approve_all = False
    for i, p in enumerate(proposals, 1):
        if approve_all:
            approved.append(p)
            continue
        while True:
            ans = input(f"Approve proposal {i} ({p.source}->{p.target})? "
                        "[y/n/all/skip] ").strip().lower()
            if ans in {"y", "yes"}:
                approved.append(p)
                break
            if ans in {"n", "no"}:
                break
            if ans in {"all", "a"}:
                approved.append(p)
                approve_all = True
                break
            if ans in {"skip", "s", "q", "quit"}:
                return approved
            print("Please answer y, n, all, or skip.")
    return approved


# --------------------------------------------------------------------------- #
# Execution and plots
# --------------------------------------------------------------------------- #

def _prompt_text_by_slug(run: dict) -> dict[str, str]:
    return {p["slug"]: (p.get("prompt_text") or p.get("full_prompt") or
                        p.get("prompt_tail_full") or p.get("prompt") or
                        p.get("prompt_tail", ""))
            for p in run.get("prompts", [])}


def _resolve_prompt_text(run: dict, proposal: Proposal, tokenizer=None) -> str:
    for rec in run.get("prompts", []):
        if rec.get("slug") == proposal.slug:
            # run_lens stores the full resolved prompt only inside swap results
            # for REPORT_SWAP prompts. For normal prompts, prompt_tail is all we
            # have in the run JSON; those are still useful for manual/dry-run
            # proposals, but execution needs the original prompt.
            if rec.get("swap", {}).get("prompt"):
                return rec["swap"]["prompt"]
            if rec.get("prompt_text"):
                return rec["prompt_text"]
            if "full_prompt" in rec:
                return rec["full_prompt"]
            if tokenizer is not None:
                try:
                    return dp.resolve_text(dp.by_slug(proposal.slug), tokenizer)
                except KeyError:
                    pass
            return rec.get("prompt_tail", "")
    raise KeyError(proposal.slug)


def _metric_from_swap(result: dict) -> dict:
    clean_top = result.get("clean_top", [])
    swapped_top = result.get("swapped_top", [])
    target = _norm_word(result.get("target", ""))

    def clean_tok(x):
        return _norm_word(x[0]) if x else ""

    swapped_words = [clean_tok(x) for x in swapped_top[:5]]
    return {
        "metric_kind": "manipulation_sanity_check",
        "source_suppressed": result["source_rank_swapped"] > result["source_rank_clean"],
        "target_installed": result["target_rank_swapped"] < result["target_rank_clean"],
        "target_top1": bool(swapped_words and swapped_words[0] == target),
        "target_top5": target in swapped_words,
        "top_changed": bool(clean_top and swapped_top and clean_tok(clean_top[0]) != clean_tok(swapped_top[0])),
        "source_rank_delta": result["source_rank_swapped"] - result["source_rank_clean"],
        "target_rank_delta": result["target_rank_clean"] - result["target_rank_swapped"],
        "causal_success": result.get("causal_success"),
    }


def execute_swaps(run: dict, proposals: list[Proposal], *, model_name: str,
                  lens_path: str, alphas: list[float], method: str) -> list[dict]:
    print(f"\nLoading model and lens for {len(proposals)} approved proposal(s)...")
    model, lens = swapmod._load(model_name, lens_path)
    band = tuple(run.get("methodology", {}).get("workspace_band", (38.0, 92.0)))
    records = []
    for p in proposals:
        prompt = _resolve_prompt_text(run, p, model.tokenizer)
        if not prompt:
            print(f"skip {p.slug}: no executable prompt text in run JSON", file=sys.stderr)
            continue
        alpha_records = []
        print(f"\n{p.slug}: {p.source} -> {p.target}")
        for alpha in alphas:
            try:
                out = swapmod.run_swap(model, lens, prompt, p.source, p.target,
                                       alpha=alpha, method=method, band=band)
            except Exception as exc:  # noqa: BLE001
                alpha_records.append({"alpha": alpha, "error": f"{type(exc).__name__}: {exc}"})
                print(f"  alpha={alpha:g}: ERROR {type(exc).__name__}: {exc}")
                continue
            metrics = _metric_from_swap(out)
            alpha_records.append({"alpha": alpha, "swap": out, "metrics": metrics})
            print(
                f"  alpha={alpha:g}: source rank "
                f"{out['source_rank_clean']+1}->{out['source_rank_swapped']+1}; "
                f"target rank {out['target_rank_clean']+1}->{out['target_rank_swapped']+1}; "
                f"top {out['clean_top'][0][0]!r}->{out['swapped_top'][0][0]!r}"
            )
        records.append({"proposal": asdict(p), "alphas": alpha_records})
    return records


def plot_swap_effects(records: list[dict], out_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    by_slug: dict[str, list[dict]] = {}
    for rec in records:
        by_slug.setdefault(rec["proposal"]["slug"], []).append(rec)

    for slug, rows in by_slug.items():
        labels, target_delta, source_delta, colors = [], [], [], []
        for rec in rows:
            prop = rec["proposal"]
            pair = f"{prop['source']}->{prop['target']}"
            for ar in rec["alphas"]:
                if "metrics" not in ar:
                    continue
                labels.append(f"{pair}\na={ar['alpha']:g}")
                target_delta.append(ar["metrics"]["target_rank_delta"])
                source_delta.append(ar["metrics"]["source_rank_delta"])
                colors.append("#2b7a8c" if ar["metrics"]["target_top5"] else "#c1553b")
        if not labels:
            continue
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 2, figsize=(max(8.0, 1.1 * len(labels)), 4.0))
        axes[0].bar(x, target_delta, color=colors, alpha=0.9)
        axes[0].axhline(0, color="#8a8a8a", lw=0.8)
        axes[0].set_title("Target-direction rank movement (diagnostic)")
        axes[0].set_ylabel("clean rank - swapped rank\n(positive = target moved up)")
        axes[1].bar(x, source_delta, color="#6a51a3", alpha=0.9)
        axes[1].axhline(0, color="#8a8a8a", lw=0.8)
        axes[1].set_title("Source suppression")
        axes[1].set_ylabel("swapped rank - clean rank\n(positive = source moved down)")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.grid(True, axis="y", color="#e7e7e7", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Swap effects: {slug}", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        path = out_dir / f"{slug}__swap_effect.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written


def write_results(base_dir: Path, tag: str, run_path: str, model_name: str,
                  lens_path: str, proposals: list[Proposal], approved: list[Proposal],
                  results: list[dict], provider: str, analysis_model: str | None,
                  plot_paths: list[str], method: str) -> Path:
    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run": str(run_path),
        "model": model_name,
        "lens": str(lens_path),
        "method": method,
        "tag": tag,
        "provider": provider,
        "analysis_model": analysis_model,
        "proposals": [asdict(p) for p in proposals],
        "approved": [asdict(p) for p in approved],
        "results": results,
        "plots": [str(Path(p).relative_to(base_dir)) for p in plot_paths],
    }
    runs_dir = base_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    path = runs_dir / f"{tag}_swaps.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_alphas(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("need at least one alpha")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Suggest, approve, and run J-lens swaps.")
    ap.add_argument("--run", required=True, help="runs/<tag>.json from run_lens.py")
    ap.add_argument("--model", default=None, help="HF model id (default: run['model'])")
    ap.add_argument("--lens", default=None, help="lens_<tag>.pt (default: sibling lens file)")
    ap.add_argument("--provider", choices=["offline", "anthropic", "openai"],
                    default="offline")
    ap.add_argument("--analysis-model", default=None,
                    help="provider model override for proposal generation")
    ap.add_argument("--slugs", default=None, help="comma-separated prompt slugs")
    ap.add_argument("--domains", default=None, help="comma-separated domains")
    ap.add_argument("--alpha", type=_parse_alphas, default=[1.0],
                    help="comma-separated alpha sweep, e.g. 0.5,1,2,4")
    ap.add_argument("--method", choices=["pinv", "projection"], default="pinv",
                    help="swap method: paper-style pseudoinverse coordinates "
                         "(pinv) or old projection approximation")
    ap.add_argument("--auto-approve", action="store_true",
                    help="run all validated proposals without prompting")
    ap.add_argument("--dry-run", action="store_true",
                    help="print proposals only; do not load model weights or run swaps")
    ap.add_argument("--allow-weak", action="store_true",
                    help="allow source concepts worse than --max-source-rank")
    ap.add_argument("--max-source-rank", type=int, default=20,
                    help="1-based maximum source rank to propose by default")
    ap.add_argument("--per-prompt", type=int, default=2)
    ap.add_argument("--max-proposals", type=int, default=20)
    args = ap.parse_args()

    run_path = Path(args.run)
    run = _load_run(run_path)
    base_dir = _base_dir(run_path)
    tag = run.get("tag") or run_path.stem
    model_name = args.model or run.get("model")
    lens_path = args.lens or str(base_dir / f"lens_{tag}.pt")
    slugs = _csv_set(args.slugs)
    domains = _csv_set(args.domains)

    seeds = deterministic_proposals(
        run,
        slugs=slugs,
        domains=domains,
        max_source_rank=args.max_source_rank,
        allow_weak=args.allow_weak,
        per_prompt=args.per_prompt,
        max_proposals=args.max_proposals,
    )
    proposals = seeds
    provider_used = args.provider
    if args.provider != "offline":
        try:
            proposals = llm_refine_proposals(
                run,
                seeds,
                provider=args.provider,
                model=args.analysis_model,
                allow_weak=args.allow_weak,
                max_source_rank=args.max_source_rank,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{args.provider} proposal call failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            print("Falling back to offline deterministic proposals.", file=sys.stderr)
            provider_used = "offline"
            proposals = seeds

    if not args.dry_run:
        if not model_name:
            raise SystemExit("--model is required when the run JSON has no model field")
        if not Path(lens_path).is_file():
            raise SystemExit(f"lens file not found: {lens_path}")
        proposals = validate_tokens(proposals, model_name)

    print_proposals(proposals)
    if args.dry_run:
        print("\nDry run: no model weights loaded and no swaps executed.")
        return

    approved = approve_proposals(proposals, auto_approve=args.auto_approve)
    if not approved:
        print("No swaps approved.")
        return

    results = execute_swaps(
        run,
        approved,
        model_name=model_name,
        lens_path=lens_path,
        alphas=args.alpha,
        method=args.method,
    )
    plot_dir = base_dir / "figures" / tag / "swaps"
    plots = plot_swap_effects(results, plot_dir)
    out_json = write_results(
        base_dir,
        tag,
        str(run_path),
        model_name,
        lens_path,
        proposals,
        approved,
        results,
        provider_used,
        args.analysis_model,
        plots,
        args.method,
    )
    print(f"\nWrote swap results: {out_json}")
    if plots:
        print("Wrote plots:")
        for p in plots:
            print(f"  {p}")


if __name__ == "__main__":
    main()
