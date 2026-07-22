# Copyright 2026. Apache-2.0.
"""Paper-faithful protocol helpers for fitting and evaluating a Jacobian lens.

This module intentionally sits outside the vendored :mod:`jlens` package.  The
vendored package stays byte-for-byte compatible with the public reference
implementation (apart from its progress display), while this file owns the
experiment choices that the paper leaves to the caller: fitting corpus,
target/source layers, scored positions, workspace band, token synonyms,
provenance, and item-level statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FitRecipe:
    name: str
    n_fit: int
    max_seq_len: int
    target_layer: int | None
    report_layer_count: int | None
    skip_first: int
    strict_corpus: bool
    description: str


FIT_RECIPES: dict[str, FitRecipe] = {
    "paper": FitRecipe(
        name="paper",
        n_fit=1000,
        max_seq_len=128,
        target_layer=-2,
        report_layer_count=25,
        skip_first=16,
        strict_corpus=True,
        description=(
            "Paper-scale estimator: 1000 independent 128-token sequences, "
            "penultimate target layer, and 25 evenly spaced reported layers."
        ),
    ),
    "demo": FitRecipe(
        name="demo",
        n_fit=12,
        max_seq_len=64,
        target_layer=None,
        report_layer_count=None,
        skip_first=16,
        strict_corpus=False,
        description=(
            "Fast exploratory fit. Results from this profile are not a "
            "quantitative replication of the paper."
        ),
    ),
}


def is_paper_faithful_recipe(recipe: FitRecipe) -> bool:
    reference = FIT_RECIPES["paper"]
    return (
        recipe.name == "paper"
        and recipe.n_fit >= reference.n_fit
        and recipe.max_seq_len == reference.max_seq_len
        and recipe.target_layer == reference.target_layer
        and recipe.report_layer_count == reference.report_layer_count
        and recipe.skip_first == reference.skip_first
    )


@dataclass(frozen=True)
class FitCorpus:
    texts: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ConceptTokens:
    """All valid single-token surface forms for one conceptual target."""

    label: str
    token_ids: tuple[int, ...]
    surfaces: tuple[str, ...]


def resolve_recipe(
    name: str,
    *,
    n_fit: int | None = None,
    max_seq_len: int | None = None,
    target_layer: int | None = None,
    report_layer_count: int | None = None,
) -> FitRecipe:
    if name not in FIT_RECIPES:
        raise ValueError(f"unknown recipe {name!r}; choose from {sorted(FIT_RECIPES)}")
    base = FIT_RECIPES[name]
    return FitRecipe(
        name=base.name,
        n_fit=base.n_fit if n_fit is None else n_fit,
        max_seq_len=base.max_seq_len if max_seq_len is None else max_seq_len,
        target_layer=base.target_layer if target_layer is None else target_layer,
        report_layer_count=(base.report_layer_count if report_layer_count is None
                            else report_layer_count),
        skip_first=base.skip_first,
        strict_corpus=base.strict_corpus,
        description=base.description,
    )


def resolve_layer_index(layer: int | None, n_layers: int) -> int:
    resolved = n_layers - 1 if layer is None else layer
    if resolved < 0:
        resolved += n_layers
    if not 0 <= resolved < n_layers:
        raise ValueError(f"target layer {layer!r} is invalid for {n_layers} layers")
    return resolved


def evenly_spaced_source_layers(
    n_layers: int, target_layer: int | None, count: int | None
) -> list[int] | None:
    """Layers below ``target_layer``; ``None`` retains every source layer."""
    target = resolve_layer_index(target_layer, n_layers)
    if target <= 0:
        raise ValueError("target layer leaves no source layers")
    if count is None or count >= target:
        return list(range(target))
    if count <= 0:
        raise ValueError("report layer count must be positive")
    # Rounding linspace can duplicate values for small stacks; unique/sorted is
    # deliberate and the endpoints are always retained.
    return sorted({int(round(x)) for x in np.linspace(0, target - 1, count)})


_BUILTIN_CORPUS = (
    "The history of materials science spans thousands of years, from the first "
    "smelting of copper to modern superalloys used in turbine blades. " * 5,
    "Proteins are chains of amino acids that fold into precise structures. "
    "Sequence, solvent, temperature, and cellular machinery shape the fold. " * 6,
    "When a solid is loaded it can deform elastically, flow plastically, or "
    "fracture. Engineers quantify stress, defects, fatigue, and toughness. " * 6,
    "The city sat quietly beneath a grey sky as the river carried small boats "
    "past the bridge and the market stalls along the harbour. " * 7,
    "Thermodynamics connects energy, entropy, chemical reactions, heat engines, "
    "phase transformations, and molecular structure. " * 7,
    "She opened the notebook and recorded temperature, pressure, displacement, "
    "and time for every measurement in the laboratory. " * 7,
    "Crystalline materials contain vacancies, dislocations, grain boundaries, "
    "and interfaces whose motion controls macroscopic properties. " * 7,
    "DNA is transcribed into RNA, and RNA is translated into protein by the "
    "ribosome one codon and one amino acid at a time. " * 7,
)


def _texts_from_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    elif path.suffix.lower() == ".json":
        blob = json.loads(path.read_text(encoding="utf-8"))
        rows = blob.get("texts", blob.get("items", blob)) if isinstance(blob, dict) else blob
    else:
        return [part.strip() for part in path.read_text(encoding="utf-8").split("\n\n")
                if part.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"corpus {path} must contain a list or a texts/items list")
    texts = []
    for row in rows:
        text = row if isinstance(row, str) else row.get("text", row.get("prompt", ""))
        if str(text).strip():
            texts.append(str(text))
    return texts


def _take_text_records(
    records: Iterable[Any], *, n: int, text_field: str, min_chars: int
) -> list[str]:
    texts: list[str] = []
    for record in records:
        text = record if isinstance(record, str) else record.get(text_field, "")
        text = str(text).strip()
        if len(text) < min_chars:
            continue
        texts.append(text)
        if len(texts) == n:
            break
    return texts


def corpus_sha256(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def load_fit_corpus(
    *,
    corpus: str,
    n: int,
    seed: int = 0,
    revision: str | None = None,
    split: str = "train",
    subset: str | None = None,
    text_field: str = "text",
    min_chars: int = 600,
    strict: bool = True,
) -> FitCorpus:
    """Load independent fitting records and return content-addressed metadata.

    ``corpus`` may be ``wikitext``, ``builtin``, a local file, or any Hugging
    Face dataset id.  WikiText is a reproducible pretraining-like proxy; it is
    not claimed to be the paper authors' proprietary pretraining corpus.
    """
    if n <= 0:
        raise ValueError("n_fit must be positive")
    source: dict[str, Any]
    path = Path(corpus).expanduser()
    if path.is_file():
        pool = _texts_from_file(path)
        rng = random.Random(seed)
        rng.shuffle(pool)
        texts = [x for x in pool if len(x) >= min_chars][:n]
        source = {"kind": "file", "path": str(path.resolve())}
    elif corpus == "builtin":
        pool = list(_BUILTIN_CORPUS)
        texts = pool[:n]
        source = {"kind": "builtin", "name": "eight-demo-passages"}
    else:
        from datasets import load_dataset

        dataset_id = "Salesforce/wikitext" if corpus == "wikitext" else corpus
        dataset_subset = subset or (
            "wikitext-103-raw-v1" if dataset_id == "Salesforce/wikitext" else None
        )
        kwargs: dict[str, Any] = {
            "path": dataset_id,
            "split": split,
            "streaming": True,
        }
        if dataset_subset:
            kwargs["name"] = dataset_subset
        if revision:
            kwargs["revision"] = revision
        records = load_dataset(**kwargs)
        if hasattr(records, "shuffle"):
            records = records.shuffle(seed=seed, buffer_size=max(10_000, n * 10))
        texts = _take_text_records(
            records, n=n, text_field=text_field, min_chars=min_chars
        )
        source = {
            "kind": "huggingface",
            "dataset": dataset_id,
            "subset": dataset_subset,
            "split": split,
            "revision": revision or "main",
            "text_field": text_field,
        }

    if len(texts) < n:
        message = f"corpus {corpus!r} yielded {len(texts)} independent texts; requested {n}"
        if strict:
            raise ValueError(message)
        print(f"  WARNING: {message}; demo fit will use the available records")
    if not texts:
        raise ValueError(f"corpus {corpus!r} yielded no usable texts")
    unique_records = len(set(texts))
    if strict and unique_records != len(texts):
        raise ValueError(
            f"corpus {corpus!r} contains duplicate fitting records "
            f"({unique_records} unique of {len(texts)})"
        )
    metadata = {
        **source,
        "requested_records": n,
        "records": len(texts),
        "unique_records": unique_records,
        "seed": seed,
        "min_chars": min_chars,
        "sha256": corpus_sha256(texts),
    }
    return FitCorpus(tuple(texts), metadata)


def model_identity(model: Any, requested_id: str) -> dict[str, Any]:
    hf = getattr(model, "_hf_model", None)
    cfg = getattr(hf, "config", None)
    tok = model.tokenizer
    dtype = None
    if hf is not None:
        try:
            dtype = str(next(hf.parameters()).dtype).replace("torch.", "")
        except StopIteration:
            pass
    return {
        "requested_id": requested_id,
        "requested_revision": getattr(model, "_requested_revision", None),
        "model_class": type(hf).__name__ if hf is not None else type(model).__name__,
        "config_name_or_path": getattr(cfg, "_name_or_path", None),
        "model_revision": getattr(cfg, "_commit_hash", None),
        "tokenizer_class": type(tok).__name__,
        "tokenizer_name_or_path": getattr(tok, "name_or_path", None),
        "vocab_size": len(tok) if hasattr(tok, "__len__") else None,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "dtype": dtype,
    }


def lens_metadata_path(lens_path: str | Path) -> Path:
    return Path(f"{lens_path}.meta.json")


def write_lens_metadata(lens_path: str | Path, metadata: dict[str, Any]) -> Path:
    path = lens_metadata_path(lens_path)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_lens_metadata(lens_path: str | Path) -> dict[str, Any] | None:
    path = lens_metadata_path(lens_path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def validate_lens_metadata(
    metadata: dict[str, Any] | None,
    *,
    identity: dict[str, Any],
    require: bool,
) -> None:
    if metadata is None:
        if require:
            raise ValueError(
                "lens has no provenance sidecar; refit it with this runner or pass "
                "--allow-unverified-lens for exploratory use"
            )
        print("  WARNING: loaded lens has no provenance metadata")
        return
    fitted = metadata.get("model", {})
    for key in ("requested_id", "model_class", "n_layers", "d_model", "vocab_size"):
        expected, actual = identity.get(key), fitted.get(key)
        if expected is not None and actual is not None and expected != actual:
            raise ValueError(
                f"lens/model mismatch for {key}: lens={actual!r}, model={expected!r}"
            )
    for key in ("requested_revision", "model_revision", "tokenizer_name_or_path"):
        expected, actual = identity.get(key), fitted.get(key)
        if expected and actual and expected != actual:
            raise ValueError(
                f"lens/model mismatch for {key}: lens={actual!r}, model={expected!r}"
            )


def fit_fingerprint(metadata: dict[str, Any]) -> str:
    stable = {key: value for key, value in metadata.items() if key != "created_at"}
    raw = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _surface_forms(word: str) -> tuple[str, ...]:
    forms = (" " + word, word, " " + word.capitalize(), word.capitalize(),
             " " + word.lower(), word.lower())
    return tuple(dict.fromkeys(forms))


def resolve_concepts(tokenizer: Any, prompt: Any) -> tuple[list[ConceptTokens], list[str]]:
    """Resolve labels plus synonyms to valid, non-special single token IDs."""
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    unk = getattr(tokenizer, "unk_token_id", None)
    concepts: list[ConceptTokens] = []
    dropped: list[str] = []
    token_owner: dict[int, str] = {}
    for label in prompt.tracked:
        surfaces = [label, *prompt.synonyms.get(label, [])]
        ids: list[int] = []
        accepted: list[str] = []
        for surface in surfaces:
            accepted_surface = False
            for form in _surface_forms(surface):
                token_ids = tokenizer.encode(form, add_special_tokens=False)
                if len(token_ids) != 1:
                    continue
                token_id = int(token_ids[0])
                if token_id == unk or token_id in special or token_id in ids:
                    continue
                ids.append(token_id)
                accepted_surface = True
            if accepted_surface:
                accepted.append(surface)
        if ids:
            conflicts = {token_id: token_owner[token_id] for token_id in ids
                         if token_id in token_owner and token_owner[token_id] != label}
            if conflicts:
                raise ValueError(
                    f"ambiguous concept token IDs for {label!r}: {conflicts}; "
                    "targets/synonym sets must be disjoint"
                )
            token_owner.update({token_id: label for token_id in ids})
            concepts.append(ConceptTokens(label, tuple(ids), tuple(accepted)))
        else:
            dropped.append(label)
    return concepts, dropped


def flatten_concept_ids(concepts: Sequence[ConceptTokens]) -> set[int]:
    return {token_id for concept in concepts for token_id in concept.token_ids}


def _find_last_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle), -1, -1):
        if list(haystack[start:start + len(needle)]) == list(needle):
            return start
    return None


def resolve_score_positions(
    tokenizer: Any, text: str, prompt: Any, *, strict: bool
) -> list[int]:
    """Resolve the protocol's predetermined readout position or response span."""
    ids = tokenizer.encode(text, add_special_tokens=True)
    if not ids:
        raise ValueError(f"prompt {prompt.slug!r} tokenized to an empty sequence")
    selector = prompt.readout_selector
    if selector in {"before_answer", "final_prompt_token"}:
        return [len(ids) - 1]
    if selector == "explicit":
        pos = prompt.readout_at if prompt.readout_at >= 0 else len(ids) + prompt.readout_at
        if not 0 <= pos < len(ids):
            raise ValueError(f"readout_at={prompt.readout_at} is outside {len(ids)} tokens")
        return [pos]
    if selector == "last_newline":
        newline_positions = [i for i, token_id in enumerate(ids)
                             if "\n" in tokenizer.decode([token_id])]
        if newline_positions:
            return [newline_positions[-1]]
        raise ValueError("readout_selector=last_newline but no newline token was found")
    if selector == "assistant_response":
        response = prompt.assistant_prefill
        candidates = [response, " " + response]
        for candidate in candidates:
            response_ids = tokenizer.encode(candidate, add_special_tokens=False)
            start = _find_last_subsequence(ids, response_ids)
            if start is not None:
                return list(range(start, start + len(response_ids)))
        if strict:
            raise ValueError(
                f"could not locate assistant_prefill tokens for {prompt.slug!r}"
            )
        print(f"  WARNING: could not resolve response span for {prompt.slug}; using last token")
        return [len(ids) - 1]
    if selector == "all_prompt":
        if strict:
            raise ValueError(
                "readout_selector=all_prompt is exploratory and cannot be used in strict mode"
            )
        return list(range(len(ids)))
    raise ValueError(f"unknown readout_selector {selector!r}")


def concept_presence(
    concepts: Sequence[ConceptTokens], token_ids: Sequence[int]
) -> dict[str, bool]:
    present = set(int(x) for x in token_ids)
    return {c.label: any(token_id in present for token_id in c.token_ids)
            for c in concepts}


def protocol_violations(
    prompt: Any,
    concepts: Sequence[ConceptTokens],
    *,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int] | None,
) -> list[str]:
    violations: list[str] = []
    if prompt.must_be_absent_from_input:
        found = [name for name, yes in concept_presence(concepts, prompt_token_ids).items()
                 if yes]
        if found:
            violations.append(f"intermediates present in input: {found}")
    if prompt.must_be_absent_from_output:
        if generated_token_ids is None:
            violations.append("output absence requested but no completion was generated")
        else:
            found = [name for name, yes in concept_presence(
                concepts, generated_token_ids).items() if yes]
            if found:
                violations.append(f"intermediates present in generated output: {found}")
    return violations


def item_pass_scores(rank_items: Sequence[Sequence[int]], ks: Sequence[int]) -> list[float]:
    """Paper pass@k: mean, over items, of target/intermediate recovery."""
    scores = []
    for k in ks:
        per_item = [float(np.mean([0 <= rank < k for rank in ranks]))
                    for ranks in rank_items if ranks]
        scores.append(float(np.mean(per_item)) if per_item else float("nan"))
    return scores


def log_k_auc(ks: Sequence[int], values: Sequence[float]) -> float:
    if len(ks) < 2 or len(ks) != len(values):
        return float("nan")
    x = np.log(np.asarray(ks, dtype=float))
    y = np.asarray(values, dtype=float)
    if not np.isfinite(y).all() or x[-1] == x[0]:
        return float("nan")
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(y, x=x) / (x[-1] - x[0]))


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int = 0, n_boot: int = 2000
) -> tuple[float, float, float]:
    arr = np.asarray([x for x in values if math.isfinite(x)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        value = float(arr[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def recipe_dict(recipe: FitRecipe) -> dict[str, Any]:
    return asdict(recipe)
