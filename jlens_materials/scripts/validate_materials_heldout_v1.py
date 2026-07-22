#!/usr/bin/env python3
"""Pre-execution validation for the frozen materials held-out v1 suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}


def max_ngram_jaccard(text: str, comparison: list[str], n: int = 5) -> float:
    left = ngrams(text, n)
    if not left:
        return 0.0
    best = 0.0
    for candidate in comparison:
        right = ngrams(candidate, n)
        union = left | right
        best = max(best, len(left & right) / len(union) if union else 0.0)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="prompts/materials-heldout-v1-preregistered.json",
    )
    parser.add_argument(
        "--development-manifest",
        default="prompts/materials-paper-v2-preregistered.json",
    )
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument(
        "--revision",
        default="a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
    )
    parser.add_argument(
        "--output",
        default="experiments/materials-heldout-v1-preflight.json",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow tokenizer files to be fetched instead of requiring the local cache",
    )
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    development_path = ROOT / args.development_manifest
    output_path = ROOT / args.output
    manifest = json.loads(manifest_path.read_text())
    development = json.loads(development_path.read_text())
    prompts = manifest["prompts"]
    old_prompts = development["prompts"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=not args.allow_download,
    )

    slugs = [row["slug"] for row in prompts]
    counts = Counter(row["target_family"] for row in prompts)
    expected_families = set(counts)
    structural_errors: list[str] = []
    if len(prompts) != 50:
        structural_errors.append(f"expected 50 prompts, found {len(prompts)}")
    if len(set(slugs)) != len(slugs):
        structural_errors.append("prompt slugs are not unique")
    if len(counts) != 10 or any(value != 5 for value in counts.values()):
        structural_errors.append(f"expected 10 families x 5 prompts, found {dict(counts)}")

    input_leaks: list[dict] = []
    dropped_terms: list[dict] = []
    invalid_fields: list[dict] = []
    for row in prompts:
        if row.get("shape") != "ASSOCIATION" or row.get("protocol") != "lens_eval":
            invalid_fields.append({"slug": row["slug"], "reason": "shape/protocol"})
        if row.get("readout_selector") != "final_prompt_token":
            invalid_fields.append({"slug": row["slug"], "reason": "readout selector"})
        prompt_ids = set(tokenizer.encode(row["text"], add_special_tokens=True))
        for term in row["tracked"]:
            token_ids: set[int] = set()
            for surface in (f" {term}", term, f" {term.capitalize()}", term.capitalize()):
                encoded = tokenizer.encode(surface, add_special_tokens=False)
                if len(encoded) == 1:
                    token_ids.add(int(encoded[0]))
            if not token_ids:
                dropped_terms.append({"slug": row["slug"], "term": term})
            overlap = sorted(prompt_ids & token_ids)
            if overlap:
                input_leaks.append({
                    "slug": row["slug"],
                    "term": term,
                    "token_ids": overlap,
                })

    old_association_prompts = [
        row for row in old_prompts
        if row.get("shape") == "ASSOCIATION" and row.get("text")
    ]
    old_texts = [row["text"] for row in old_association_prompts]
    exact_duplicates = sorted(
        row["slug"] for row in prompts if row["text"] in set(old_texts)
    )
    maximum_jaccards = [max_ngram_jaccard(row["text"], old_texts) for row in prompts]
    old_families = {
        row.get("target_family") or row.get("category")
        for row in old_association_prompts
    }

    report = {
        "status": "pass" if not (
            structural_errors or invalid_fields or input_leaks or exact_duplicates
        ) else "fail",
        "executed_before_heldout_lens_outputs": True,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "generator_sha256": sha256(ROOT / "scripts/generate_materials_heldout_v1.py"),
        "model": args.model,
        "model_revision": args.revision,
        "tokenizer_class": type(tokenizer).__name__,
        "prompt_count": len(prompts),
        "unique_slug_count": len(set(slugs)),
        "family_counts": dict(sorted(counts.items())),
        "same_family_set_as_development": expected_families == old_families,
        "structural_errors": structural_errors,
        "invalid_fields": invalid_fields,
        "tokenizer_resolved_input_leaks": input_leaks,
        "unresolved_single_token_terms": dropped_terms,
        "exact_development_prompt_duplicates": exact_duplicates,
        "development_overlap": {
            "metric": "maximum pairwise Jaccard similarity of lowercase word 5-gram sets",
            "mean": sum(maximum_jaccards) / len(maximum_jaccards),
            "maximum": max(maximum_jaccards),
            "per_prompt": dict(zip(slugs, maximum_jaccards, strict=True)),
        },
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "development_overlap"}, indent=2))
    print(
        "development 5-gram overlap: "
        f"mean={report['development_overlap']['mean']:.6f}, "
        f"max={report['development_overlap']['maximum']:.6f}"
    )
    print(f"wrote {output_path}")
    if report["status"] != "pass":
        raise SystemExit("held-out preflight validation failed")


if __name__ == "__main__":
    main()
