# Copyright 2026.  Apache-2.0.
"""Run the Jacobian lens over the fracture / protein prompt sets and emit figures.

Pipeline (mirrors walkthrough.ipynb, plus the static figures in `matviz`):

    1. load an HF decoder  ->  jlens.from_hf  (Gemma config baked in below)
    2. load OR fit a JacobianLens
    3. for each domain Prompt:
         compute_slice  ->  SliceData
         matviz.plot_slice_grid / rank_trajectories / rank_heatmap /
                emergence_depth
    4. aggregate valid independent items into pass@k/AUC with a logit-lens baseline

Gemma notes  (verified on `google/gemma-4-E4B-it`)
--------------------------------------------------
* `jlens.hf._LAYOUTS` already covers the Gemma family: `Layout("model")` matches
  the text-only `Gemma*ForCausalLM` (text decoder at `.model`, blocks `.layers`,
  norm `.norm`, embed `.embed_tokens`, head `.lm_head`), and the multimodal
  `Gemma*ForConditionalGeneration` wrappers are caught by
  `Layout("model.language_model")`.  So `from_hf(hf, tok)` "just works"; if a
  future variant nests the decoder elsewhere, pass an explicit `Layout(...)`.
* `final_logit_softcapping` (where present) — `HFLensModel.unembed` reads it off
  the config and applies the tanh cap, so lens logits stay on the model's scale
  (Gemma-3/4 set it to null; also handled).
* Gemma ties embeddings (`lm_head` shares `embed_tokens.weight`).  Irrelevant to
  the lens: it only calls `unembed`.
* Gemma is gated on the Hub: `huggingface-cli login` (or HF_TOKEN) first, and
  accept the licence for `google/gemma-4-E4B-it`.
* An *instruct* checkpoint runs all five shapes (raw-text completions and
  chat-template prompts).  A *base* (`-pt`) checkpoint has no chat template, so
  restrict it to the raw-text shapes (MULTIHOP / ASSOCIATION / RECOGNITION).

Run
---
    python run_lens.py --model gpt2 --fit --recipe demo --corpus builtin --n-fit 8
    python run_lens.py --model google/gemma-4-E4B-it --fit --recipe paper \
        --corpus wikitext --prompts prompts/mechanics-paper-example.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

try:  # tqdm ships with transformers; degrade to a no-op if somehow absent
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_):
        return x

# use the vendored jlens (committed under _vendor_jlens/) without installing
_REPO = Path(__file__).resolve().parent / "_vendor_jlens"
if (_REPO / "jlens").is_dir():
    sys.path.insert(0, str(_REPO))

import jlens                                    # noqa: E402
from jlens.vis import compute_slice             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import domain_prompts as dp                      # noqa: E402
import lens_hub                                  # noqa: E402
import matviz                                    # noqa: E402
import paper_protocol as pp                      # noqa: E402
import swap as swapmod                           # noqa: E402

FIGDIR = Path(__file__).resolve().parent / "figures"


# --------------------------------------------------------------------------- #
# model + lens
# --------------------------------------------------------------------------- #

_DTYPES = {"float32": torch.float32,
           "bfloat16": torch.bfloat16,
           "float16": torch.float16}


def load_model(name: str, *, dtype=torch.float32, device: str | None = None,
               revision: str | None = None, trust_remote_code: bool = False):
    import transformers

    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, revision=revision, trust_remote_code=trust_remote_code
    )
    hf = hf.to(device)
    tok = transformers.AutoTokenizer.from_pretrained(
        name, revision=revision, trust_remote_code=trust_remote_code
    )
    model = jlens.from_hf(hf, tok)
    model._requested_revision = revision
    print(f"loaded {name}: n_layers={model.n_layers} d_model={model.d_model} "
          f"device={device} dtype={dtype}")
    return model


def arch_tag(model_id: str, *, revision: str | None = None,
             trust_remote_code: bool = False) -> str:
    """Deterministic lens/run tag for a HF model id, WITHOUT loading weights:
    the class ``AutoModelForCausalLM`` will instantiate, lowercased. This equals
    ``type(loaded_model).__name__.lower()`` (what the files are named after), so
    pipeline.sh and run_lens.py always agree — even for multimodal Gemma, where
    ``config.architectures[0]`` differs from the causal-LM class."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    try:
        return AutoModelForCausalLM._model_mapping[type(cfg)].__name__.lower()
    except Exception:  # noqa: BLE001
        return (cfg.architectures or ["model"])[0].lower()


def _save_lens_bundle(lens, path: str | Path, metadata: dict) -> Path:
    """Save lens weights and provenance, creating an explicit output directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lens.save(str(destination))
    return pp.write_lens_metadata(destination, metadata)


def get_or_fit_lens(
    model,
    *,
    model_id: str,
    tag: str,
    lens_path: str | None,
    do_fit: bool,
    recipe: pp.FitRecipe,
    corpus: str,
    corpus_revision: str | None,
    corpus_subset: str | None,
    corpus_split: str,
    corpus_text_field: str,
    corpus_seed: int,
    corpus_min_chars: int,
    source_layers=None,
    dim_batch: int = 64,
    allow_unverified_lens: bool = False,
):
    identity = pp.model_identity(model, model_id)
    if lens_path and Path(lens_path).is_file():
        lens = jlens.JacobianLens.load(lens_path)
        metadata = pp.read_lens_metadata(lens_path)
        pp.validate_lens_metadata(
            metadata, identity=identity,
            require=(recipe.name == "paper" and not allow_unverified_lens),
        )
        if recipe.name == "paper" and metadata is not None:
            fitted_recipe = metadata.get("recipe", {})
            try:
                fitted_recipe_obj = pp.FitRecipe(**fitted_recipe)
            except TypeError as exc:
                raise ValueError("lens has incomplete paper-recipe provenance") from exc
            if not pp.is_paper_faithful_recipe(fitted_recipe_obj):
                raise ValueError(
                    "paper evaluation requested with a non-compliant lens; refit with "
                    "--recipe paper or use --recipe demo"
                )
            expected_layers = pp.evenly_spaced_source_layers(
                model.n_layers, -2, 25
            )
            if lens.source_layers != expected_layers:
                raise ValueError(
                    "paper lens source layers do not match the 25-layer "
                    f"registered grid: {lens.source_layers}"
                )
            if metadata.get("corpus", {}).get("records", 0) < 1000:
                raise ValueError("paper lens provenance contains fewer than 1000 records")
            if metadata.get("corpus", {}).get("unique_records", 0) < 1000:
                raise ValueError("paper lens provenance contains fewer than 1000 unique records")
        print(f"loaded lens: {lens}")
        return lens, metadata
    if not do_fit:
        raise SystemExit("no lens file and --fit not set; nothing to do")
    fit_corpus = pp.load_fit_corpus(
        corpus=corpus,
        n=recipe.n_fit,
        seed=corpus_seed,
        revision=corpus_revision,
        subset=corpus_subset,
        split=corpus_split,
        text_field=corpus_text_field,
        min_chars=corpus_min_chars,
        strict=recipe.strict_corpus,
    )
    if source_layers is None:
        source_layers = pp.evenly_spaced_source_layers(
            model.n_layers, recipe.target_layer, recipe.report_layer_count
        )
    fit_metadata = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_reference": "https://transformer-circuits.pub/2026/workspace/index.html",
        "upstream_reference": "lamm-mit/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "model": identity,
        "recipe": pp.recipe_dict(recipe),
        "corpus": fit_corpus.metadata,
        "source_layers": source_layers,
        "target_layer": pp.resolve_layer_index(recipe.target_layer, model.n_layers),
        "dim_batch": dim_batch,
    }
    fingerprint = pp.fit_fingerprint(fit_metadata)
    ckpt = FIGDIR.parent / f"lens_{tag}.{fingerprint}.ckpt.pt"
    # outer bar over fit prompts; jlens.fit iterates this list (each prompt runs
    # ceil(d_model/dim_batch) backward passes — the slow part on big models).
    bar = tqdm(fit_corpus.texts, desc="fitting lens", unit="prompt")
    lens = jlens.fit(model, bar, source_layers=source_layers,
                     target_layer=recipe.target_layer,
                     dim_batch=dim_batch, max_seq_len=recipe.max_seq_len,
                     skip_first=recipe.skip_first,
                     checkpoint_path=str(ckpt), checkpoint_every=4)
    out = lens_path or str(FIGDIR.parent / f"lens_{tag}.pt")
    meta_path = _save_lens_bundle(lens, out, fit_metadata)
    print(f"fitted and saved lens -> {out}: {lens}; provenance -> {meta_path}")
    return lens, fit_metadata


# --------------------------------------------------------------------------- #
# concept word -> token id
# --------------------------------------------------------------------------- #

def tracked_ids(tokenizer, prompt: dp.Prompt):
    """Resolve concepts and every valid synonym to single-token alternatives."""
    out, dropped = pp.resolve_concepts(tokenizer, prompt)
    if dropped:
        print(f"    [{prompt.slug}] multi-token, not tracked: {dropped}")
    return out, dropped


@torch.no_grad()
def next_token_logits(model, text: str, *, max_seq_len: int = 512) -> torch.Tensor:
    """The model's actual next-token logits, independent of display masking."""
    from jlens.hooks import ActivationRecorder

    input_ids = model.encode(text, max_length=max_seq_len)
    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final]) as recorder:
        model.forward(input_ids)
        residual = recorder.activations[final][0, -1:].float()
    return model.unembed(residual).float()[0].cpu()


@torch.no_grad()
def generate_completion(model, text: str, *, max_new_tokens: int) -> tuple[str, list[int]]:
    """Greedy completion used for answer and surface-absence controls."""
    if max_new_tokens <= 0:
        return "", []
    hf = getattr(model, "_hf_model", None)
    if hf is None or not hasattr(hf, "generate"):
        raise ValueError("the model adapter does not expose greedy generation")
    input_ids = model.encode(text, max_length=512)
    pad = model.tokenizer.pad_token_id
    if pad is None:
        pad = model.tokenizer.eos_token_id
    output = hf.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad,
        use_cache=True,
    )
    generated = [int(x) for x in output[0, input_ids.shape[1]:].tolist()]
    return model.tokenizer.decode(generated, skip_special_tokens=True), generated


def _answer_ids(tokenizer, answers: list[str]) -> set[int]:
    ids: set[int] = set()
    for answer in answers:
        for form in (" " + answer, answer, " " + answer.capitalize(),
                     answer.capitalize(), " " + answer.lower(), answer.lower()):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
    return ids


def baseline_answer_record(model, text: str, prompt: dp.Prompt) -> dict:
    answers = [x for x in (prompt.answer, *prompt.acceptable_answers) if x]
    if not answers:
        return {"required": False, "correct": None, "expected": []}
    logits = next_token_logits(model, text)
    top_id = int(logits.argmax())
    valid_ids = _answer_ids(model.tokenizer, answers)
    decoded = model.tokenizer.decode([top_id], clean_up_tokenization_spaces=False).strip()
    return {
        "required": True,
        "correct": top_id in valid_ids,
        "expected": answers,
        "greedy_token_id": top_id,
        "greedy_token": decoded,
    }


def logit_lens_min_ranks(
    model,
    lens,
    text: str,
    concepts: list[pp.ConceptTokens],
    *,
    positions: list[int],
    band: tuple[float, float],
    max_seq_len: int,
) -> tuple[dict[str, int], list[dict]]:
    """Vanilla logit-lens baseline and auditable per-layer rank curves.

    The summary uses the exact same band/span reduction as the J-lens.  The
    trajectories retain one-indexed ranks at every evaluated layer so figures
    can show the comparison without rerunning the model or reading values back
    from a rendered image.
    """
    depths = matviz.reindex_layers(lens.source_layers, model.n_layers)
    layers = [layer for layer, depth in zip(lens.source_layers, depths)
              if band[0] <= depth <= band[1]]
    if not layers or not concepts:
        return {}, []
    logits_by_layer, _, _ = lens.apply(
        model,
        text,
        layers=layers,
        positions=positions,
        max_seq_len=max_seq_len,
        use_jacobian=False,
    )
    result: dict[str, int] = {}
    curves: dict[str, list[int]] = {concept.label: [] for concept in concepts}
    for concept in concepts:
        best = None
        for logits in logits_by_layer.values():
            layer_best = None
            for token_id in concept.token_ids:
                target = logits[:, token_id]
                ranks = (logits > target[:, None]).sum(dim=1)
                value = int(ranks.min())
                best = value if best is None else min(best, value)
                layer_best = value if layer_best is None else min(layer_best, value)
            curves[concept.label].append(
                -1 if layer_best is None else int(layer_best) + 1
            )
        if best is not None:
            result[concept.label] = best
    depth_by_layer = {
        int(layer): round(float(depth), 1)
        for layer, depth in zip(lens.source_layers, depths)
    }
    trajectories = [
        {
            "label": concept.label,
            "layers": [int(layer) for layer in layers],
            "depths": [depth_by_layer[int(layer)] for layer in layers],
            "ranks": curves[concept.label],
        }
        for concept in concepts
    ]
    return result, trajectories


def jacobian_rank_trajectories(
    slice_data,
    model,
    concepts: list[pp.ConceptTokens],
    *,
    positions: list[int],
) -> list[dict]:
    """Serialize one-indexed J-lens concept ranks across all fitted layers."""
    depth = matviz.reindex_layers(slice_data.layers, model.n_layers)
    id_to_col = {int(t): i for i, t in enumerate(slice_data.tracked_token_ids)}
    trajectories = []
    for concept in concepts:
        cols = [id_to_col[int(t)] for t in concept.token_ids
                if int(t) in id_to_col]
        if not cols:
            continue
        ranks = slice_data.rank_tensor[:, :, cols].astype(float)
        ranks = np.where(ranks >= 0, ranks, np.nan)
        ranks = np.nanmin(ranks, axis=2)
        curve = np.nanmin(ranks[positions], axis=0)
        trajectories.append({
            "label": concept.label,
            "layers": [int(layer) for layer in slice_data.layers],
            "depths": [round(float(value), 1) for value in depth],
            "ranks": [(-1 if not np.isfinite(value) else int(value) + 1)
                      for value in curve],
        })
    return trajectories


# --------------------------------------------------------------------------- #
# per-prompt run
# --------------------------------------------------------------------------- #

def layer_readouts(slice_data, model, *, position: int = -1, top: int = 8) -> list[dict]:
    """Per-layer top lens tokens at one position: the semantic trajectory the
    LLM analysis reads. Uses the already-computed slice grid (word-like tokens
    when mask_display=True)."""
    seq = slice_data.top_ids.shape[0]
    pos = position if position >= 0 else seq + position
    depth = matviz.reindex_layers(slice_data.layers, model.n_layers)
    out = []
    for li, layer in enumerate(slice_data.layers):
        ids = slice_data.top_ids[pos, li, :top]
        toks = [slice_data.vocab_fragment.get(int(t), "").strip() for t in ids]
        # Static reports use an English engineering audience.  Filtering here
        # avoids unreadable subword fragments and missing-font glyphs while the
        # unfiltered numeric ranks remain available in ``concept_trajectories``.
        toks = [t for t in toks if t and t.isascii()
                and any(char.isalpha() for char in t)]
        out.append({"layer": int(layer), "depth": round(float(depth[li]), 1),
                    "top_tokens": toks[:top]})
    return out


def surprising_concepts(slice_data, model, text: str, *, band=(38.0, 92.0),
                        generated_token_ids: list[int] | None = None,
                        top: int = 12) -> list[dict]:
    """Exploratory top-1 candidates, filtered against prompt and real output.

    This remains a heuristic discovery view, not one of the paper's quantitative
    metrics. Unlike the old implementation, the final-layer next-token rows are
    not described as emitted output; output exclusion uses an actual generated
    or teacher-forced completion when one is available.
    """
    depth = matviz.reindex_layers(slice_data.layers, model.n_layers)
    final = model.n_layers - 1
    final_idx = (slice_data.layers.index(final) if final in slice_data.layers
                 else len(slice_data.layers) - 1)
    seq = slice_data.top_ids.shape[0]
    prompt_low = text.lower()

    out_ids = set(generated_token_ids or [])

    band_layers = [li for li, d in enumerate(depth)
                   if band[0] <= d <= band[1] and li != final_idx]
    score: dict[int, float] = {}
    best: dict[int, tuple] = {}
    for li in band_layers:
        for pos in range(seq):
            tid = int(slice_data.top_ids[pos, li, 0])          # top-1 per cell
            rank = int(slice_data.top_ranks[pos, li, 0])
            score[tid] = score.get(tid, 0.0) + 1.0
            if tid not in best or rank < best[tid][0]:
                best[tid] = (rank, float(depth[li]), pos)

    def wordlike(s: str) -> bool:
        return (len(s) >= 3 and s.isascii() and any(c.isalpha() for c in s)
                and all(c.isalnum() or c in "'-" for c in s))

    rows = []
    for tid, sc in score.items():
        if generated_token_ids is not None and tid in out_ids:
            continue
        raw = slice_data.vocab_fragment.get(tid, "")   # keeps the leading space
        s = raw.strip()
        # word-like, and a word START (space-prefixed or capitalised) — this
        # drops most subword-continuation fragments ("tlement", "izable", ...)
        if not wordlike(s) or not (raw[:1] == " " or s[:1].isupper()):
            continue
        if s.lower() in prompt_low:                    # already in the prompt
            continue
        r, d, pos = best[tid]
        near = slice_data.context_token_strs[slice_data.ctx_offset + pos]
        rows.append({"concept": s, "score": round(sc, 1), "best_rank": r + 1,
                     "best_depth": round(d, 1),
                     "near_token": near.replace("\n", " ").strip()[:14],
                     "output_checked": generated_token_ids is not None})
    rows.sort(key=lambda x: (-x["score"], x["best_rank"]))
    return rows[:top]


def run_swap_for(model, lens, prompt: dp.Prompt, text: str, *,
                 alpha: float = 1.0,
                 band: tuple[float, float] = (38.0, 92.0)) -> dict | None:
    """Run a graded probe swap or the paper's multi-candidate verbal report."""
    if prompt.protocol == "verbal_report":
        candidate_targets = list(prompt.candidates[:10]) or (
            [prompt.swap_to] if prompt.swap_to else []
        )
        if not candidate_targets:
            return None
        clean_id = int(next_token_logits(model, text).argmax())
        clean_source = model.tokenizer.decode(
            [clean_id], clean_up_tokenization_spaces=False
        ).strip()
        trials, errors = [], []
        for target in candidate_targets:
            try:
                # The clean report may differ from a candidate only in case
                # (e.g. "Vacancy" vs "vacancy") and therefore use a different
                # token id. That is not a sibling-concept intervention.
                if target.strip().casefold() == clean_source.casefold():
                    continue
                if swapmod._single_token(model.tokenizer, target) == clean_id:
                    continue
                trials.append(swapmod.run_swap(
                    model, lens, text, "", target, alpha=alpha, band=band
                ))
            except Exception as exc:  # noqa: BLE001
                errors.append({"target": target, "error": f"{type(exc).__name__}: {exc}"})
        if not trials:
            print(f"    [{prompt.slug}] verbal-report swaps skipped: {errors}")
            return None
        summary = dict(trials[0])
        summary.update({
            "protocol": "verbal_report",
            "clean_source": clean_source,
            "trials": trials,
            "errors": errors,
            "n_targets": len(trials),
            "protocol_success_rate": float(np.mean([
                trial["protocol_success"] for trial in trials
            ])),
        })
        print(f"    [{prompt.slug}] verbal report source={clean_source!r}; "
              f"{sum(t['protocol_success'] for t in trials)}/{len(trials)} "
              "candidate swaps reached rank 1")
        return summary

    if not (prompt.swap_from and prompt.swap_to):
        return None
    try:
        out = swapmod.run_swap(
            model,
            lens,
            text,
            prompt.swap_from or "",
            prompt.swap_to,
            alpha=alpha,
            band=band,
            expected_answer=prompt.answer,
            acceptable_answers=prompt.acceptable_answers,
            swap_answer=prompt.swap_answer,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    [{prompt.slug}] swap skipped: {type(exc).__name__}: {exc}")
        return None
    print(f"    [{prompt.slug}] swap {out['source']}->{prompt.swap_to}: "
          f"source rank {out['source_rank_clean']+1}->{out['source_rank_swapped']+1}, "
          f"target rank {out['target_rank_clean']+1}->{out['target_rank_swapped']+1}")
    return out


def run_prompt(
    model,
    lens,
    prompt: dp.Prompt,
    *,
    tag: str,
    workspace_band: tuple[float, float],
    strict_protocol: bool,
    eval_max_seq_len: int = 512,
    emergence_threshold: int = 5,
    emergence_sustain: int = 2,
    generation_max_new_tokens: int = 16,
    layer_stride: int = 1,
    mask_display: bool = True,
    discover: int = 0,
    layer_readout_top: int = 8,
    surprising_top: int = 12,
    open_vocab_logit_baseline: bool = False,
    do_swap: bool = False,
    swap_alpha: float = 1.0,
) -> dict:
    if layer_readout_top < 1 or surprising_top < 1:
        raise ValueError("open-vocabulary retention counts must be positive")
    text = dp.resolve_text(prompt, model.tokenizer)
    concepts, dropped = tracked_ids(model.tokenizer, prompt)
    pinned = pp.flatten_concept_ids(concepts)
    score_positions = pp.resolve_score_positions(
        model.tokenizer, text, prompt, strict=strict_protocol
    )
    prompt_token_ids = [int(x) for x in model.encode(
        text, max_length=eval_max_seq_len)[0].tolist()]
    if max(score_positions) >= len(prompt_token_ids):
        raise ValueError(
            f"score position {max(score_positions)} was truncated from a "
            f"{len(prompt_token_ids)}-token evaluation context"
        )

    if prompt.readout_selector == "assistant_response":
        generated_text = prompt.assistant_prefill
        generated_ids = [prompt_token_ids[pos] for pos in score_positions]
    elif generation_max_new_tokens > 0:
        generated_text, generated_ids = generate_completion(
            model, text, max_new_tokens=generation_max_new_tokens
        )
    else:
        generated_text, generated_ids = "", None

    violations = pp.protocol_violations(
        prompt,
        concepts,
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_ids,
    )
    design_violations = [v for v in violations
                         if v.startswith("intermediates present in input") or
                         v.startswith("output absence requested but")]
    if design_violations and strict_protocol:
        raise ValueError("; ".join(design_violations))
    baseline = baseline_answer_record(model, text, prompt)
    excluded_reasons = list(violations)
    if baseline["required"] and not baseline["correct"]:
        excluded_reasons.append("clean model did not produce the expected answer")

    # Discovery mode remains explicitly exploratory and is never included in
    # paper pass@k aggregates.
    auto = discover > 0 and not concepts
    slice_data = compute_slice(
        model,
        lens,
        text,
        top_n=max(10, layer_readout_top),
        layer_stride=layer_stride,
        mask_display=mask_display,
        pinned_token_ids=pinned,
        max_tracked=(discover if auto else None),
        max_seq_len=eval_max_seq_len,
    )
    if auto:
        concepts = [pp.ConceptTokens(
            model.tokenizer.decode([int(token_id)]).strip() or f"<{token_id}>",
            (int(token_id),),
            (model.tokenizer.decode([int(token_id)]).strip(),),
        ) for token_id in slice_data.tracked_token_ids[:discover]]
        print(f"    [{prompt.slug}] exploratory discoveries: "
              f"{[concept.label for concept in concepts]}")

    ems = matviz.concept_emergence(
        slice_data,
        model,
        concepts,
        positions=score_positions,
        band=workspace_band,
        threshold=emergence_threshold,
        sustain=emergence_sustain,
    )
    logit_baseline, logit_trajectories = logit_lens_min_ranks(
        model,
        lens,
        text,
        concepts,
        positions=score_positions,
        band=workspace_band,
        max_seq_len=eval_max_seq_len,
    )
    base = FIGDIR / tag / prompt.domain / prompt.slug
    lr = layer_readouts(
        slice_data,
        model,
        position=score_positions[-1],
        top=layer_readout_top,
    )
    highlight = tuple(dict.fromkeys((score_positions[0], score_positions[-1])))

    written = []
    written += matviz.plot_slice_grid(
        slice_data,
        model,
        f"{base}__grid",
        title=prompt.title,
        subtitle=prompt.description,
        highlight_positions=highlight,
    )
    written += matviz.plot_concept_stream(
        lr,
        f"{base}__stream",
        title=f"{prompt.title} - top-token readout stream",
        subtitle=("descriptive lens readout at the fixed score position; "
                  "not a literal chain of thought"),
    )
    if concepts:
        written += matviz.plot_rank_trajectories(
            slice_data,
            model,
            concepts,
            f"{base}__trajectory",
            positions=score_positions,
            band=workspace_band,
            title=f"{prompt.title} - concept ranks across depth",
            subtitle=(f"{prompt.description}  [fixed {prompt.readout_selector} "
                      f"span; synonym-min ranks]"),
        )
        written += matviz.plot_emergence_depth(
            ems,
            f"{base}__emergence",
            title=f"{prompt.title} - sustained concept emergence",
            subtitle=(f"preregistered workspace band {workspace_band[0]:.0f}-"
                      f"{workspace_band[1]:.0f}% depth"),
        )
        best = min((e for e in ems if np.isfinite(e.best_depth)),
                   key=lambda e: e.best_rank, default=None)
        if best is not None:
            concept = next(c for c in concepts if c.label == best.label)
            written += matviz.plot_rank_heatmap(
                slice_data,
                model,
                concept.token_ids[0],
                best.label,
                f"{base}__heatmap",
                title=f"{prompt.title} - '{best.label}' across the prompt",
            )

    surprising = surprising_concepts(
        slice_data,
        model,
        text,
        band=workspace_band,
        generated_token_ids=generated_ids,
        top=surprising_top,
    )
    logit_layer_readouts = []
    logit_surprising = []
    if open_vocab_logit_baseline:
        logit_slice = compute_slice(
            model,
            lens,
            text,
            top_n=max(10, layer_readout_top),
            max_tracked=0,
            pinned_token_ids=set(),
            layer_stride=layer_stride,
            mask_display=mask_display,
            max_seq_len=eval_max_seq_len,
            use_jacobian=False,
        )
        logit_layer_readouts = layer_readouts(
            logit_slice,
            model,
            position=score_positions[-1],
            top=layer_readout_top,
        )
        logit_surprising = surprising_concepts(
            logit_slice,
            model,
            text,
            band=workspace_band,
            generated_token_ids=generated_ids,
            top=surprising_top,
        )
    swap_result = (
        run_swap_for(
            model,
            lens,
            prompt,
            text,
            alpha=swap_alpha,
            band=workspace_band,
        ) if do_swap else None
    )
    valid_for_metrics = (prompt.protocol != "exploratory" and not auto
                         and not excluded_reasons and bool(ems))
    print(
        f"  [{prompt.slug}] wrote {len(written)} files; fixed band "
        f"{workspace_band[0]:.0f}-{workspace_band[1]:.0f}; "
        f"metrics={'included' if valid_for_metrics else 'excluded'}"
    )

    record = {
        "slug": prompt.slug,
        "shape": prompt.shape,
        "protocol": prompt.protocol,
        "domain": prompt.domain,
        "title": prompt.title,
        "description": prompt.description,
        "note": prompt.note,
        "prompt_text": text,
        "prompt_tail": text[-160:],
        "readout_selector": prompt.readout_selector,
        "readout_at": prompt.readout_at,
        "score_positions": score_positions,
        "band": [round(workspace_band[0], 1), round(workspace_band[1], 1)],
        "band_source": "preregistered_cli_or_config",
        "condition": prompt.condition,
        "control_group": prompt.control_group,
        "phrasing_id": prompt.phrasing_id,
        "carrier_id": prompt.carrier_id,
        "target_family": prompt.target_family,
        "category": prompt.category,
        "answer": prompt.answer,
        "baseline": baseline,
        "generated_completion": generated_text,
        "generated_token_ids": generated_ids,
        "protocol_violations": violations,
        "valid_for_metrics": valid_for_metrics,
        "excluded_reasons": excluded_reasons,
        "discovered": bool(auto),
        "open_vocabulary_config": {
            "layer_readout_top": int(layer_readout_top),
            "surprising_top": int(surprising_top),
            "candidate_generation_uses_tracked_terms": False,
            "matched_logit_open_vocabulary_stored": bool(open_vocab_logit_baseline),
        },
        "tracked_dropped": dropped,
        "tracked": [
            {"label": c.label, "surfaces": list(c.surfaces),
             "token_ids": list(c.token_ids)} for c in concepts
        ],
        "emergence": [
            {
                "label": e.label,
                "best_rank": int(e.best_rank),
                "best_depth": (None if not np.isfinite(e.best_depth)
                               else round(float(e.best_depth), 1)),
                "onset_depth": (None if not np.isfinite(e.onset_depth)
                                else round(float(e.onset_depth), 1)),
                "reached_top1": bool(e.reached_top),
                "best_pos": int(e.best_pos),
                "logit_lens_best_rank": logit_baseline.get(e.label),
            }
            for e in ems
        ],
        "layer_readouts": lr,
        "concept_trajectories": {
            "jacobian_lens": jacobian_rank_trajectories(
                slice_data, model, concepts, positions=score_positions
            ),
            "logit_lens": logit_trajectories,
            "rank_indexing": "one-indexed; rank 1 is the highest-readout token",
            "position_reduction": "minimum over preregistered score positions and declared single-token synonyms",
        },
        "surprising": surprising,
        "surprising_label": "exploratory surfaced candidates",
        "logit_layer_readouts": logit_layer_readouts,
        "logit_surprising": logit_surprising,
        "figures": {
            Path(path).stem.split("__")[-1]: str(Path(path).relative_to(FIGDIR.parent))
            for path in written if path.endswith(".png")
        },
    }
    if swap_result is not None:
        record["swap"] = swap_result
        if swap_result.get("baseline_correct") is False:
            record["valid_for_metrics"] = False
            record["excluded_reasons"].append("swap clean-answer baseline failed")
    return {
        "slug": prompt.slug,
        "shape": prompt.shape,
        "emergences": ems,
        "band": workspace_band,
        "files": written,
        "record": record,
    }


def aggregate_metrics(
    results: list[dict],
    tag: str,
    *,
    ks: list[int],
    seed: int,
    min_items: int,
) -> dict:
    """Item-level pass@k/AUC, logit-lens baseline, controls, and swaps."""
    by_shape: dict[str, list[dict]] = {}
    for result in results:
        record = result["record"]
        if record.get("valid_for_metrics") and record.get("protocol") == "lens_eval":
            by_shape.setdefault(result["shape"], []).append(record)

    metrics: dict[str, dict] = {}
    for shape, records in sorted(by_shape.items()):
        lens_items = [[int(e["best_rank"]) for e in record["emergence"]
                       if int(e["best_rank"]) >= 0] for record in records]
        baseline_items = [[int(e["logit_lens_best_rank"])
                           for e in record["emergence"]
                           if e.get("logit_lens_best_rank") is not None]
                          for record in records]
        lens_values = pp.item_pass_scores(lens_items, ks)
        baseline_values = pp.item_pass_scores(baseline_items, ks)
        group = {
            "n_items": len(records),
            "minimum_required": min_items,
            "sufficient_sample": len(records) >= min_items,
            "ks": ks,
            "jacobian_lens": {
                "pass_at_k": lens_values,
                "auc_log_k": pp.log_k_auc(ks, lens_values),
            },
            "logit_lens": {
                "pass_at_k": baseline_values,
                "auc_log_k": pp.log_k_auc(ks, baseline_values),
            },
        }
        cis = {}
        for k in ks:
            per_item = [float(np.mean([0 <= rank < k for rank in ranks]))
                        for ranks in lens_items if ranks]
            mean, lo, hi = pp.bootstrap_mean_ci(per_item, seed=seed)
            cis[str(k)] = {"mean": mean, "low": lo, "high": hi}
        group["jacobian_lens"]["bootstrap_95"] = cis
        metrics[shape] = group
        matviz.plot_pass_at_k_curves(
            ks,
            {"Jacobian lens": lens_values, "Logit lens": baseline_values},
            FIGDIR / tag / f"summary__{shape.lower()}__pass_at_k",
            aucs={"Jacobian lens": group["jacobian_lens"]["auc_log_k"],
                  "Logit lens": group["logit_lens"]["auc_log_k"]},
            title=f"{shape}: intermediate recovery",
            subtitle=(f"{len(records)} independent items; fixed score spans and "
                      "workspace band"),
        )

    modulation: dict[str, dict[str, list[float]]] = {}
    modulation_records: dict[str, list[dict]] = {}
    for result in results:
        record = result["record"]
        if record.get("protocol") != "directed_modulation" or not record.get("valid_for_metrics"):
            continue
        group = record.get("control_group") or "ungrouped"
        condition = record.get("condition") or "unspecified"
        score = float(any(0 <= e["best_rank"] < 1 for e in record["emergence"]))
        modulation.setdefault(group, {}).setdefault(condition, []).append(score)
        modulation_records.setdefault(group, []).append(record)
    # Directed modulation has its own preregistered 24-phrasing target in the
    # source protocol. The general 50-item threshold applies to the independent
    # lens-evaluation distributions, not to each modulation condition.
    modulation_min = 24
    control_metrics = {}
    for group, conditions in modulation.items():
        control_metrics[group] = {}
        for condition, values in conditions.items():
            mean, lo, hi = pp.bootstrap_mean_ci(values, seed=seed)
            control_metrics[group][condition] = {
                "n_items": len(values), "hit_rate": mean,
                "bootstrap_95": [lo, hi],
                "minimum_required": modulation_min,
                "sufficient_sample": len(values) >= modulation_min,
            }
        phrasing_ids = {record["phrasing_id"] for record in modulation_records[group]
                        if record.get("phrasing_id")}
        control_metrics[group]["summary"] = {
            "n_trials": len(modulation_records[group]),
            "distinct_phrasings": len(phrasing_ids),
            "paper_phrasing_target": 24,
        }

    swap_records = [r["record"]["swap"] for r in results if r["record"].get("swap")]
    swaps = []
    for swap_record in swap_records:
        swaps.extend(swap_record.get("trials", [swap_record]))
    graded = [s for s in swaps if s.get("protocol_success") is not None]
    counterfactual = [s for s in swaps if s.get("causal_success") is not None]
    swap_metrics = {
        "n_interventions": len(swaps),
        "n_graded": len(graded),
        "minimum_required": min_items,
        "sufficient_sample": len(graded) >= min_items,
        "protocol_success_rate": (float(np.mean([s["protocol_success"] for s in graded]))
                                  if graded else None),
        "n_counterfactual": len(counterfactual),
        "causal_success_rate": (
            float(np.mean([s["causal_success"] for s in counterfactual]))
            if counterfactual else None
        ),
    }
    return {
        "by_shape": metrics,
        "directed_modulation_controls": control_metrics,
        "causal_swaps": swap_metrics,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--model-revision", default=None,
                    help="HF commit/tag; recorded in lens provenance")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--lens", default=None, help="path to a fitted lens .pt")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument(
        "--fit-only",
        action="store_true",
        help=("fit (and optionally upload) the lens, then exit before loading "
              "or evaluating any prompts; requires --fit"),
    )
    ap.add_argument("--hf-lens-repo", default=None, metavar="OWNER/REPO",
                    help="private Hugging Face model repo used to store/fetch the lens bundle")
    ap.add_argument("--hf-lens-path", default=None, metavar="PATH.pt",
                    help="lens path inside --hf-lens-repo (default: basename of --lens)")
    ap.add_argument("--hf-lens-revision", default="main", metavar="REVISION",
                    help="Hub branch/tag/commit for download; branch for upload (default: main)")
    ap.add_argument("--hf-upload-lens", action="store_true",
                    help="after fitting/loading, upload the .pt and .meta.json in one Hub commit")
    ap.add_argument("--hf-force-download", action="store_true",
                    help="replace the local lens bundle from the Hub even if local files exist")
    ap.add_argument("--hf-private", action=argparse.BooleanOptionalAction, default=True,
                    help="create/require a private Hub repo for upload (default: true)")
    ap.add_argument("--hf-commit-message", default=None,
                    help="optional Hub commit message for --hf-upload-lens")
    ap.add_argument("--recipe", choices=sorted(pp.FIT_RECIPES), default="demo",
                    help="paper = quantitative protocol; demo = fast exploration")
    ap.add_argument("--n-fit", type=int, default=None,
                    help="override recipe sample count")
    ap.add_argument("--fit-max-seq-len", type=int, default=None)
    ap.add_argument("--target-layer", type=int, default=None,
                    help="override recipe target layer; negative counts from end")
    ap.add_argument("--report-layers", type=int, default=None,
                    help="number of evenly spaced fitted/reported source layers")
    ap.add_argument("--source-layers", default=None,
                    help="explicit comma-separated source layers")
    ap.add_argument("--corpus", default="wikitext",
                    help="wikitext, HF dataset id, local txt/json/jsonl, or builtin (demo only)")
    ap.add_argument("--corpus-revision", default=None)
    ap.add_argument("--corpus-subset", default=None)
    ap.add_argument("--corpus-split", default="train")
    ap.add_argument("--corpus-text-field", default="text")
    ap.add_argument("--corpus-seed", type=int, default=0)
    ap.add_argument("--corpus-min-chars", type=int, default=600)
    ap.add_argument("--allow-unverified-lens", action="store_true",
                    help="permit a legacy lens without provenance (exploratory only)")
    ap.add_argument("--shapes", default=None,
                    help="comma list; subset of MULTIHOP,ASSOCIATION,"
                         "RECOGNITION,MODULATION,REPORT_SWAP,PROBE_SWAP "
                         "(default: built-ins use the first three; custom "
                         "prompts use whatever shapes they declare)")
    ap.add_argument("--domains", default=None,
                    help="comma list (default: fracture,protein for built-ins; "
                         "all present for custom prompts)")
    ap.add_argument("--prompts", default=None,
                    help="load prompts from a JSON file or a directory of "
                         "*.json instead of the built-ins (see prompts/)")
    ap.add_argument("--prompt-json", default=None,
                    help="define a prompt on the fly: a raw JSON object or list "
                         "of objects, e.g. '{\"shape\":\"MULTIHOP\","
                         "\"text\":\"...\",\"tracked\":[\"crack\"]}'")
    ap.add_argument("--discover", type=int, default=0, metavar="N",
                    help="auto-discover the N concepts that surface, for prompts "
                         "with no 'tracked' list (no need to pre-name concepts)")
    ap.add_argument("--layer-readout-top", type=int, default=8, metavar="N",
                    help="retain the unrestricted top N word-like tokens at the "
                         "fixed readout position for every sampled layer")
    ap.add_argument("--surprising-top", type=int, default=12, metavar="N",
                    help="retain the top N input/output-absent open-vocabulary "
                         "candidates across prompt positions and band layers")
    ap.add_argument("--open-vocab-logit-baseline", action="store_true",
                    help="also retain matched vanilla-logit-lens open-vocabulary "
                         "readouts for blinded discovery comparisons")
    ap.add_argument("--swaps", action="store_true",
                    help="also run the causal J-lens swap for prompts that "
                         "declare probe-swap answers or verbal-report candidates; records "
                         "clean-vs-swapped output into the run JSON")
    ap.add_argument("--swap-alpha", type=float, default=1.0,
                    help="swap strength (1.0 clean swap = paper primary; "
                         "2.0 double-strength, can destabilise small models)")
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--workspace-band", default="38,92", metavar="LO,HI",
                    help="fixed/preregistered depth-percent band; never inferred from test items")
    ap.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction,
                    default=None, help="fail prompt-design violations (default on for paper)")
    ap.add_argument("--eval-max-seq-len", type=int, default=512)
    ap.add_argument("--generation-max-new-tokens", type=int, default=16)
    ap.add_argument("--emergence-threshold", type=int, default=5)
    ap.add_argument("--emergence-sustain", type=int, default=2)
    ap.add_argument("--pass-at-k", default="1,2,5,10,20,50,100")
    ap.add_argument("--min-items-per-shape", type=int, default=None,
                    help="default 50 for paper, 1 for demo")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="write a partial run and return success despite prompt failures")
    ap.add_argument("--dtype", choices=list(_DTYPES), default="float32",
                    help="model weight dtype (default float32 = most accurate). "
                         "bfloat16 ~halves memory and works well on GPU; float16 "
                         "also halves memory but is less numerically stable. The "
                         "fitted Jacobian is stored in float32 regardless, so the "
                         "lens itself stays full-precision.")
    ap.add_argument("--dim-batch", type=int, default=64, metavar="N",
                    help="output dims computed per backward pass during --fit "
                         "(default 64). Lower (e.g. 16 or 8) cuts the fit's peak "
                         "memory at the cost of more passes — same total FLOPs, "
                         "just slower. Only affects fitting.")
    ap.add_argument("--tag", default=None,
                    help="override the lens/run filename tag (default: the arch "
                         "class, e.g. --tag gemma4-e4b-it for a readable name, or "
                         "to keep two same-architecture checkpoints distinct)")
    args = ap.parse_args()

    hub_detail_flags = (
        args.hf_lens_path,
        args.hf_upload_lens,
        args.hf_force_download,
        args.hf_commit_message,
    )
    if any(hub_detail_flags) and not args.hf_lens_repo:
        raise SystemExit("Hub lens options require --hf-lens-repo OWNER/REPO")
    if args.hf_lens_repo and not args.lens:
        raise SystemExit("--hf-lens-repo requires an explicit local --lens PATH.pt")
    if args.fit_only and not args.fit:
        raise SystemExit("--fit-only requires --fit")

    recipe = pp.resolve_recipe(
        args.recipe,
        n_fit=args.n_fit,
        max_seq_len=args.fit_max_seq_len,
        target_layer=args.target_layer,
        report_layer_count=args.report_layers,
    )
    if args.recipe == "paper" and not pp.is_paper_faithful_recipe(recipe):
        raise SystemExit(
            "paper recipe overrides must retain n_fit>=1000, max_seq_len=128, "
            "target_layer=-2, and report_layers=25; use --recipe demo for custom "
            "or reduced-cost fitting"
        )
    strict_protocol = (recipe.name == "paper" if args.strict_protocol is None
                       else args.strict_protocol)
    if recipe.name == "paper" and not strict_protocol:
        raise SystemExit(
            "--recipe paper requires strict protocol checks; use --recipe demo "
            "for exploratory scoring"
        )
    if recipe.name == "paper" and args.allow_unverified_lens:
        raise SystemExit(
            "--allow-unverified-lens is exploratory only; use --recipe demo or "
            "load a paper-fit lens with a valid .meta.json sidecar"
        )
    if recipe.name == "paper" and args.layer_stride != 1:
        raise SystemExit(
            "--recipe paper requires --layer-stride 1 so all 25 registered "
            "source layers are scored"
        )
    workspace_band = tuple(float(x) for x in args.workspace_band.split(","))
    if len(workspace_band) != 2 or not 0 <= workspace_band[0] <= workspace_band[1] < 100:
        raise SystemExit("--workspace-band must be two ordered percentages below 100")
    ks = sorted({int(x) for x in args.pass_at_k.split(",") if int(x) > 0})
    if len(ks) < 2:
        raise SystemExit("--pass-at-k needs at least two positive k values for AUC")
    min_items = (args.min_items_per_shape if args.min_items_per_shape is not None
                 else (50 if recipe.name == "paper" else 1))
    if recipe.name == "paper" and min_items < 50:
        raise SystemExit(
            "--recipe paper requires at least 50 independent items per "
            "evaluation distribution"
        )
    source_layers = ([int(x) for x in args.source_layers.split(",")]
                     if args.source_layers else None)
    tag = args.tag or arch_tag(
        args.model, revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
    )
    if args.hf_lens_repo:
        local_lens = Path(args.lens)
        local_bundle_complete = (
            local_lens.is_file() and lens_hub.metadata_path(local_lens).is_file()
        )
        should_download = args.hf_force_download or (
            not args.fit and not local_bundle_complete
        )
        if should_download:
            local, local_meta = lens_hub.download_lens_bundle(
                repo_id=args.hf_lens_repo,
                local_path=local_lens,
                path_in_repo=args.hf_lens_path,
                revision=args.hf_lens_revision,
                force=args.hf_force_download,
            )
            print(
                f"downloaded lens bundle from hf://{args.hf_lens_repo}/"
                f"{lens_hub.normalize_hub_path(args.hf_lens_path, local_lens)} "
                f"-> {local} and {local_meta}"
            )
    model = load_model(
        args.model,
        dtype=_DTYPES[args.dtype],
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
    )
    if recipe.name == "paper" and len(pp.evenly_spaced_source_layers(
            model.n_layers, -2, 25)) != 25:
        raise SystemExit(
            f"{args.model} has too few source layers for the paper's 25-layer "
            "reporting grid; use --recipe demo for this model"
        )
    if recipe.name == "paper" and source_layers is not None:
        expected_layers = pp.evenly_spaced_source_layers(model.n_layers, -2, 25)
        if source_layers != expected_layers:
            raise SystemExit(
                "--recipe paper requires the registered 25-layer grid; remove "
                "--source-layers or pass the exact grid"
            )
    lens, lens_metadata = get_or_fit_lens(
        model,
        model_id=args.model,
        tag=tag,
        lens_path=args.lens,
        do_fit=args.fit,
        recipe=recipe,
        corpus=args.corpus,
        corpus_revision=args.corpus_revision,
        corpus_subset=args.corpus_subset,
        corpus_split=args.corpus_split,
        corpus_text_field=args.corpus_text_field,
        corpus_seed=args.corpus_seed,
        corpus_min_chars=args.corpus_min_chars,
        source_layers=source_layers,
        dim_batch=args.dim_batch,
        allow_unverified_lens=args.allow_unverified_lens,
    )
    if args.hf_upload_lens:
        commit_url = lens_hub.upload_lens_bundle(
            repo_id=args.hf_lens_repo,
            local_path=args.lens,
            path_in_repo=args.hf_lens_path,
            revision=args.hf_lens_revision,
            private=args.hf_private,
            commit_message=args.hf_commit_message,
        )
        lens_metadata = pp.read_lens_metadata(args.lens)
        print(f"uploaded lens bundle to {commit_url}")

    if args.fit_only:
        print(f"fit-only complete -> lens bundle {args.lens}")
        return

    # Prompt source: custom JSON (file/dir/inline) replaces the built-ins.
    custom = args.prompts or args.prompt_json
    if custom:
        candidates = []
        if args.prompts:
            candidates += dp.load_prompts(args.prompts)
        if args.prompt_json:
            candidates += dp.load_prompts(args.prompt_json)
    else:
        candidates = dp.ALL_PROMPTS

    # Resolve shape/domain filters: explicit flags win; else built-in defaults
    # for the built-in set, or "everything present" for custom prompts.
    if args.shapes:
        shapes = set(args.shapes.split(","))
    else:
        shapes = ({p.shape for p in candidates} if custom
                  else {"MULTIHOP", "ASSOCIATION", "RECOGNITION"})
    if args.domains:
        domains = set(args.domains.split(","))
    else:
        domains = ({p.domain for p in candidates} if custom
                   else {"fracture", "protein"})

    prompts = [p for p in candidates
               if p.shape in shapes and p.domain in domains]
    if not prompts:
        raise SystemExit("no prompts remain after the shape/domain filters")
    if recipe.name == "paper" and any(p.protocol == "exploratory" for p in prompts):
        slugs = [p.slug for p in prompts if p.protocol == "exploratory"]
        raise SystemExit(
            "--recipe paper cannot score exploratory prompt records: "
            f"{slugs}; use registered lens_eval/modulation/swap items or "
            "--recipe demo"
        )
    print(f"running {len(prompts)} prompts: "
          f"{[p.slug for p in prompts]}")

    results = []
    errors = []
    for p in tqdm(prompts, desc="lensing prompts", unit="prompt"):
        try:
            results.append(run_prompt(
                model,
                lens,
                p,
                tag=tag,
                workspace_band=workspace_band,
                strict_protocol=strict_protocol,
                eval_max_seq_len=args.eval_max_seq_len,
                emergence_threshold=args.emergence_threshold,
                emergence_sustain=args.emergence_sustain,
                generation_max_new_tokens=args.generation_max_new_tokens,
                layer_stride=args.layer_stride,
                discover=args.discover,
                layer_readout_top=args.layer_readout_top,
                surprising_top=args.surprising_top,
                open_vocab_logit_baseline=args.open_vocab_logit_baseline,
                do_swap=args.swaps,
                swap_alpha=args.swap_alpha,
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{p.slug}] FAILED: {type(exc).__name__}: {exc}")
            errors.append({"slug": p.slug, "type": type(exc).__name__, "message": str(exc)})
    metrics = aggregate_metrics(
        results, tag, ks=ks, seed=args.corpus_seed, min_items=min_items
    )
    expected_eval_shapes = {p.shape for p in prompts if p.protocol == "lens_eval"}
    insufficient = [
        shape for shape in sorted(expected_eval_shapes)
        if shape not in metrics["by_shape"] or
        not metrics["by_shape"][shape]["sufficient_sample"]
    ]
    if any(p.protocol == "directed_modulation" for p in prompts):
        for group, conditions in metrics["directed_modulation_controls"].items():
            for condition in ("focus", "suppress", "control"):
                values = conditions.get(condition)
                if not values or not values["sufficient_sample"]:
                    insufficient.append(f"MODULATION:{group}:{condition}")
            if conditions.get("summary", {}).get("distinct_phrasings", 0) < 24:
                insufficient.append(f"MODULATION:{group}:24-distinct-phrasings")
    has_swap_prompts = any(
        p.protocol in {"probe_swap", "verbal_report"} for p in prompts
    )
    if has_swap_prompts and not args.swaps:
        insufficient.append("SWAPS:not-enabled")
    elif has_swap_prompts and not metrics["causal_swaps"]["sufficient_sample"]:
        insufficient.append("SWAPS")

    # Persist a structured run record for the analysis / report pipeline.
    runs_dir = FIGDIR.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    paper_ready = (
        pp.is_paper_faithful_recipe(recipe)
        and strict_protocol
        and lens_metadata is not None
        and not errors
        and not insufficient
    )
    claims_level = (
        "paper-protocol quantitative" if paper_ready else
        "paper-protocol incomplete; quantitative claims disabled"
        if recipe.name == "paper" else
        "exploratory demo"
    )
    run_json = {
        "format_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_identity": pp.model_identity(model, args.model),
        "tag": tag,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "lens_n_prompts": lens.n_prompts,
        "lens_provenance": lens_metadata,
        "methodology": {
            "recipe": pp.recipe_dict(recipe),
            "claims_level": claims_level,
            "paper_protocol_complete": paper_ready,
            "strict_protocol": strict_protocol,
            "workspace_band": list(workspace_band),
            "workspace_band_source": "preregistered_cli_or_config",
            "score_reduction": "fixed span x fixed band x synonym minimum",
            "emergence_threshold": args.emergence_threshold,
            "emergence_sustain": args.emergence_sustain,
            "pass_at_k": ks,
            "minimum_items_per_shape": min_items,
            "evaluation_max_seq_len": args.eval_max_seq_len,
            "generation_max_new_tokens": args.generation_max_new_tokens,
            "model_revision": args.model_revision,
            "dtype": args.dtype,
            "layer_stride": args.layer_stride,
            "prompt_source": args.prompts or ("inline-json" if args.prompt_json
                                               else "authoritative-builtins"),
            "swaps_enabled": args.swaps,
            "swap_alpha": args.swap_alpha,
        },
        "shapes": sorted(shapes),
        "domains": sorted(domains),
        "metrics": metrics,
        "errors": errors,
        "insufficient_sample_shapes": insufficient,
        "prompts": [r["record"] for r in results if "record" in r],
    }
    out_path = runs_dir / f"{tag}.json"
    out_path.write_text(json.dumps(run_json, indent=2, ensure_ascii=False))
    print(f"done -> figures in {FIGDIR}; run record -> {out_path}")
    failure_reasons = []
    if errors:
        failure_reasons.append(f"{len(errors)} prompt(s) failed")
    if strict_protocol and insufficient:
        failure_reasons.append(
            f"insufficient independent items for {insufficient}; need {min_items} per shape"
        )
    if failure_reasons and not args.continue_on_error:
        raise SystemExit("run incomplete: " + "; ".join(failure_reasons))


if __name__ == "__main__":
    main()
