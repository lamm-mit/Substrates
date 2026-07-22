#!/usr/bin/env python3
"""Freeze the arbitrary answer-code binding falsification study."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

import freeze_lexical_adversarial_representation as shared

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "experiments"
    / "late-physics-representation-replication-2026-07-17"
)
OUT = ROOT / "experiments" / "answer-code-binding-2026-07-17"
MANIFEST = OUT / "prompt_manifest.json"
PROTOCOL = OUT / "protocol.json"
PROTOCOL_MD = OUT / "PROTOCOL.md"
RUNNER = ROOT / "scripts" / "run_answer_code_binding.py"


def transform_prompt(row: dict, anchor_outcome: str) -> dict:
    """Replace the semantic answer pair with a deliberately conflicting code."""
    positive = row["outcome_positive"]
    negative = row["outcome_negative"]
    if row["variant"] == "anchor":
        mapping = {anchor_outcome: "A"}
        mapping[negative if anchor_outcome == positive else positive] = "B"
    else:
        mapping = {anchor_outcome: "B"}
        mapping[negative if anchor_outcome == positive else positive] = "A"
    expected_physical = row["expected_outcome"]
    first, second = (
        (positive, negative)
        if row["presentation_order"] == "positive-first"
        else (negative, positive)
    )
    user = (
        f"{row['stem']} Internal checkpoint. Now apply this arbitrary reporting "
        f"code: if the scientific answer is {first}, output {mapping[first]}; "
        f"if it is {second}, output {mapping[second]}. Answer exactly one "
        "uppercase letter: A or B."
    )
    return {
        **row,
        "prompt_id": f"{row['prompt_id']}--answer-code",
        "source_prompt_id": row["prompt_id"],
        "expected_physical_outcome": expected_physical,
        "expected_code": mapping[expected_physical],
        "code_mapping": mapping,
        "checkpoint_word": "checkpoint",
        "user": user,
    }


def build_manifest(source: dict) -> dict:
    source_by_triplet = {
        row["triplet_id"]: row for row in source["triplets"]
    }
    prompts = []
    prompt_ids = {}
    for row in source["prompts"]:
        triplet = source_by_triplet[row["triplet_id"]]
        transformed = transform_prompt(row, triplet["anchor_expected_outcome"])
        prompts.append(transformed)
        prompt_ids[(row["triplet_id"], row["variant"])] = transformed["prompt_id"]
    triplets = []
    for row in source["triplets"]:
        triplets.append({
            **row,
            "source_prompt_ids": row["prompt_ids"],
            "prompt_ids": {
                variant: prompt_ids[(row["triplet_id"], variant)]
                for variant in [
                    "anchor",
                    "physics_paraphrase",
                    "lexical_counterfactual",
                ]
            },
            "registered_code_pattern": {
                "anchor": "A",
                "physics_paraphrase": "B",
                "lexical_counterfactual": "A",
            },
        })
    return {
        "study_id": "answer-code-binding-2026-07-17",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": (
            "Prospective falsification of the replicated late transition. "
            "Anchor and physical paraphrase retain the same scientific answer "
            "but receive different arbitrary output codes; anchor and physical "
            "counterfactual have opposite scientific answers but share code A. "
            "A checkpoint marker precedes the mapping, so its state cannot "
            "encode the future arbitrary assignment."
        ),
        "source_study_id": source["study_id"],
        "families": source["families"],
        "triplets": triplets,
        "prompts": prompts,
    }


def main() -> None:
    if PROTOCOL.exists() or MANIFEST.exists():
        raise FileExistsError("Frozen files already exist; do not overwrite.")
    source_manifest_path = SOURCE / "prompt_manifest.json"
    source_statistics_path = SOURCE / "statistics.json"
    if not source_manifest_path.exists() or not source_statistics_path.exists():
        raise FileNotFoundError("The completed disjoint replication is required.")
    source = json.loads(source_manifest_path.read_text())
    manifest = build_manifest(source)
    shared.dump(MANIFEST, manifest)

    tokenizer = AutoTokenizer.from_pretrained(
        shared.MODEL,
        revision=shared.MODEL_REVISION,
        local_files_only=True,
    )
    words = {"checkpoint", "A", "B"}
    for family in manifest["families"]:
        words.add(family["outcome_positive"])
        words.add(family["outcome_negative"])
    tokenization = {
        word: [
            int(value)
            for value in tokenizer.encode(word, add_special_tokens=False)
        ]
        for word in sorted(words)
    }
    if any(len(ids) != 1 for ids in tokenization.values()):
        raise RuntimeError(f"Registered readouts must be one token: {tokenization}")
    lenses = [
        {**row, "sha256": shared.sha256(ROOT / row["path"])}
        for row in shared.LENS_ROWS
    ]
    protocol = {
        "study_id": manifest["study_id"],
        "status": (
            "prospectively frozen before any model forward pass for this "
            "answer-code cohort; motivated by the completed late-transition "
            "replication"
        ),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Does the late transition represent the scientific relation before "
            "the arbitrary answer mapping is visible, and then bind that "
            "relation to the requested A/B code after the mapping?"
        ),
        "model": shared.MODEL,
        "model_revision": shared.MODEL_REVISION,
        "lenses": lenses,
        "source_layers": shared.LAYERS,
        "checkpoint_word": "checkpoint",
        "registered_band_percent": [80.0, 96.0],
        "inputs": {
            "prompt_manifest": str(MANIFEST.relative_to(ROOT)),
            "prompt_manifest_sha256": shared.sha256(MANIFEST),
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": shared.sha256(RUNNER),
            "source_prompt_manifest": str(source_manifest_path.relative_to(ROOT)),
            "source_prompt_manifest_sha256": shared.sha256(source_manifest_path),
            "source_statistics": str(source_statistics_path.relative_to(ROOT)),
            "source_statistics_sha256": shared.sha256(source_statistics_path),
        },
        "tokenization_preflight": tokenization,
        "registered_endpoints": {
            "checkpoint_physics_separation": (
                "For each triplet, sign the positive-minus-negative physical "
                "readout so that the anchor answer is positive, then compute "
                "mean(anchor, physics paraphrase) minus counterfactual. Average "
                "over the frozen 80--96% window."
            ),
            "final_code_separation": (
                "At the final prompt position compute mean(A-minus-B for anchor "
                "and counterfactual) minus A-minus-B for the paraphrase. Average "
                "over the frozen 80--96% window."
            ),
            "code_binding_transition": (
                "Final-position code separation minus checkpoint-position code "
                "separation over the frozen 80--96% window."
            ),
            "inference": (
                "30,000 two-stage bootstrap resamples over six families and "
                "four triplets per family; fixed analysis seeds."
            ),
        },
        "success_rule": {
            "confidence": (
                "All three Jacobian-ensemble endpoint 95% intervals are above zero."
            ),
            "breadth": (
                "For both checkpoint physics separation and final code "
                "separation, at least 18/24 triplets and 5/6 family means are positive."
            ),
            "all_outputs_retained": True,
        },
        "secondary_endpoints": [
            "direct-decoding versions and paired Jacobian-minus-direct contrasts",
            "clean A/B answer accuracy and consistency",
            "layer-resolved endpoint trajectories",
            "centered full-state anchor-paraphrase minus anchor-counterfactual geometry at both positions",
        ],
        "guardrails": [
            "The prompt explicitly names scientific answer words; readout is a registered relational contrast, not open-vocabulary discovery.",
            "The checkpoint state cannot attend to the future mapping, but it includes the complete scientific question and the words 'Internal checkpoint'.",
            "A positive result would support staged relation formation and answer-code binding, not a literal chain of thought or human-like understanding.",
            "Every prompt, family, layer, lens seed, behavior error, and state is retained.",
        ],
    }
    shared.dump(PROTOCOL, protocol)
    PROTOCOL_MD.write_text(
        "\n".join([
            "# Frozen arbitrary answer-code binding falsification",
            "",
            f"Frozen: `{protocol['frozen_at']}`",
            "",
            "## Why this experiment",
            "",
            (
                "The preceding disjoint cohort showed a large late-minus-middle "
                "shift, but direct decoding and the Jacobian lens behaved almost "
                "identically. That effect could therefore be scientific relation "
                "formation, imminent answer-token preparation, or both. This "
                "study creates a conflict between physics and output labels."
            ),
            "",
            "## Exact conflict",
            "",
            (
                "Within each triplet, anchor and physical paraphrase have the "
                "same scientific answer but are mapped to different letters. "
                "Anchor and physical counterfactual have opposite scientific "
                "answers but are both mapped to A. The word `checkpoint` occurs "
                "before the mapping. Its state has seen the full scientific "
                "question but cannot attend to the future arbitrary code."
            ),
            "",
            "## Frozen tests",
            "",
            f"1. {protocol['registered_endpoints']['checkpoint_physics_separation']}",
            f"2. {protocol['registered_endpoints']['final_code_separation']}",
            f"3. {protocol['registered_endpoints']['code_binding_transition']}",
            "",
            "## Success rule",
            "",
            f"- {protocol['success_rule']['confidence']}",
            f"- {protocol['success_rule']['breadth']}",
            "- All outputs are retained.",
            "",
            "## Fingerprints",
            "",
            f"- transformed manifest: `{protocol['inputs']['prompt_manifest_sha256']}`",
            f"- execution runner: `{protocol['inputs']['runner_sha256']}`",
            f"- source manifest: `{protocol['inputs']['source_prompt_manifest_sha256']}`",
            f"- motivating statistics: `{protocol['inputs']['source_statistics_sha256']}`",
            *[
                f"- lens seed {row['seed']}: `{row['sha256']}`"
                for row in lenses
            ],
            "",
        ]) + "\n"
    )
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    print(f"manifest sha256: {shared.sha256(MANIFEST)}")
    print(f"wrote {PROTOCOL.relative_to(ROOT)}")
    print(f"protocol sha256: {shared.sha256(PROTOCOL)}")


if __name__ == "__main__":
    main()
