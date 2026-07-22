#!/usr/bin/env python3
"""Freeze a disjoint replication of the late physical-equivalence transition."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import freeze_lexical_adversarial_representation as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "late-physics-representation-replication-2026-07-17"
MANIFEST = OUT / "prompt_manifest.json"
PROTOCOL = OUT / "protocol.json"
PROTOCOL_MD = OUT / "PROTOCOL.md"
RUNNER = ROOT / "scripts" / "run_lexical_adversarial_representation.py"
DISCOVERY_STATS = (
    ROOT / "experiments" / "lexical-adversarial-representation-2026-07-17"
    / "statistics.json"
)


def compact(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 10000 or abs(value) < 0.01:
        exponent = int(np.floor(np.log10(abs(value))))
        coefficient = value / (10 ** exponent)
        return f"{coefficient:g}e{exponent}"
    return base.fmt(value, 4)


def orowan_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    material = case["material"]
    if not paraphrase:
        return (
            f"In a {material}, mean clear spacing between non-shearable "
            f"obstacles changes from {base.fmt(start, 1)} nanometers to "
            f"{base.fmt(end, 1)} nanometers while particle size, volume "
            "fraction, dislocation character, shear modulus, and Burgers "
            "vector remain fixed. What happens to Orowan bypass stress?"
        )
    direction = "narrows" if end < start else "widens"
    return (
        f"For the same {material}, every obstacle attribute is unchanged except "
        f"the slip-plane gap, which {direction} from "
        f"{base.fmt(start / 1000)} micrometers to "
        f"{base.fmt(end / 1000)} micrometers. Select the trend in stress "
        "required for a dislocation to bow between impenetrable particles."
    )


def porosity_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    material = case["material"]
    if not paraphrase:
        return (
            f"A {material} specimen changes from {base.fmt(start, 1)} percent "
            f"to {base.fmt(end, 1)} percent connected porosity while solid "
            "composition, grain structure, moisture, and crack density remain "
            "fixed. What happens to effective elastic modulus?"
        )
    direction = "falls" if end < start else "rises"
    return (
        f"In the same {material}, void volume fraction {direction} from "
        f"{base.fmt(start / 100)} to {base.fmt(end / 100)} while the "
        "load-bearing solid skeleton and all non-porosity defects are "
        "unchanged. Select the trend in macroscopic stiffness."
    )


def pearlite_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    material = case["material"]
    if not paraphrase:
        return (
            f"In fully pearlitic {material}, ferrite-cementite interlamellar "
            f"spacing changes from {base.fmt(start, 1)} nanometers to "
            f"{base.fmt(end, 1)} nanometers while composition, colony size, "
            "texture, prior-austenite grain size, and porosity remain fixed. "
            "What happens to yield strength?"
        )
    direction = "contracts" if end < start else "expands"
    return (
        f"The same pearlitic {material} keeps its chemistry and colony "
        f"architecture, but the separation between adjacent ferrite and "
        f"cementite plates {direction} from {base.fmt(start / 1000)} "
        f"micrometers to {base.fmt(end / 1000)} micrometers. Select the trend "
        "in flow strength."
    )


def dislocation_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    material = case["material"]
    if not paraphrase:
        return (
            f"A {material} specimen changes dislocation density from "
            f"{compact(start)} per square meter to {compact(end)} per square "
            "meter while grain size, solute content, precipitates, texture, "
            "porosity, and temperature remain fixed. What happens to yield "
            "strength under Taylor strengthening?"
        )
    direction = "rises" if end > start else "falls"
    return (
        f"In the same {material}, line-defect content {direction} from "
        f"{compact(start / 1e12)} per square micrometer to "
        f"{compact(end / 1e12)} per square micrometer. No grain, solute, "
        "particle, texture, pore, or thermal variable changes. Select the "
        "trend in flow stress."
    )


def particle_fraction_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    material = case["material"]
    particles = case["particles"]
    if not paraphrase:
        return (
            f"A {material} composite changes from {base.fmt(start, 1)} percent "
            f"to {base.fmt(end, 1)} percent {particles} particle volume "
            "fraction while matrix composition, particle size, dispersion, "
            "porosity, interface quality, and loading direction remain fixed. "
            "What happens to effective elastic modulus?"
        )
    direction = "grows" if end > start else "shrinks"
    return (
        f"In the same {material}, the fraction of stiff {particles} "
        f"reinforcement {direction} from {base.fmt(start / 100)} to "
        f"{base.fmt(end / 100)} of total volume. Matrix, particle geometry, "
        "distribution, void content, bonding, and load axis are unchanged. "
        "Select the trend in composite stiffness."
    )


def crosslink_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    material = case["material"]
    if not paraphrase:
        return (
            f"A {material} network changes crosslink density from "
            f"{base.fmt(start, 1)} moles per cubic meter to "
            f"{base.fmt(end, 1)} moles per cubic meter while temperature, "
            "chain chemistry, filler content, crystallinity, and strain rate "
            "remain fixed. What happens to rubbery shear modulus?"
        )
    direction = "increases" if end > start else "decreases"
    return (
        f"For the same {material}, the number of elastically active network "
        f"junctions {direction} from {base.fmt(start / 1000)} moles per liter "
        f"to {base.fmt(end / 1000)} moles per liter. Thermal state, polymer "
        "chemistry, fillers, order, and loading rate do not change. Select the "
        "trend in small-strain network stiffness."
    )


FAMILIES = [
    {
        "family_id": "obstacle-spacing-orowan",
        "family_name": "Obstacle spacing and Orowan stress",
        "outcome_positive": "higher",
        "outcome_negative": "lower",
        "positive_numeric_direction": "decrease",
        "render": orowan_stem,
        "cases": [
            {"case_id": "nickel-superalloy-180-60", "material": "nickel-based superalloy", "high": 180, "low": 60},
            {"case_id": "aluminum-copper-150-50", "material": "aluminum-copper alloy", "high": 150, "low": 50},
            {"case_id": "ferritic-steel-120-40", "material": "ferritic steel", "high": 120, "low": 40},
            {"case_id": "copper-alloy-90-30", "material": "copper alloy", "high": 90, "low": 30},
        ],
    },
    {
        "family_id": "porosity-modulus",
        "family_name": "Porosity and elastic modulus",
        "outcome_positive": "greater",
        "outcome_negative": "smaller",
        "positive_numeric_direction": "decrease",
        "render": porosity_stem,
        "cases": [
            {"case_id": "sintered-alumina-12-3", "material": "sintered alumina", "high": 12, "low": 3},
            {"case_id": "porous-titanium-20-5", "material": "porous titanium", "high": 20, "low": 5},
            {"case_id": "silicon-nitride-8-2", "material": "silicon nitride", "high": 8, "low": 2},
            {"case_id": "sandstone-16-4", "material": "quartz-rich sandstone", "high": 16, "low": 4},
        ],
    },
    {
        "family_id": "pearlite-spacing-strength",
        "family_name": "Pearlite spacing and yield strength",
        "outcome_positive": "higher",
        "outcome_negative": "lower",
        "positive_numeric_direction": "decrease",
        "render": pearlite_stem,
        "cases": [
            {"case_id": "eutectoid-steel-300-100", "material": "eutectoid steel", "high": 300, "low": 100},
            {"case_id": "rail-steel-240-80", "material": "rail steel", "high": 240, "low": 80},
            {"case_id": "wire-steel-180-60", "material": "high-carbon wire steel", "high": 180, "low": 60},
            {"case_id": "pearlitic-iron-360-120", "material": "pearlitic cast iron", "high": 360, "low": 120},
        ],
    },
    {
        "family_id": "dislocation-density-strength",
        "family_name": "Dislocation density and flow strength",
        "outcome_positive": "higher",
        "outcome_negative": "lower",
        "positive_numeric_direction": "increase",
        "render": dislocation_stem,
        "cases": [
            {"case_id": "copper-1e14-1e12", "material": "high-purity copper", "high": 1e14, "low": 1e12},
            {"case_id": "aluminum-8e13-8e11", "material": "high-purity aluminum", "high": 8e13, "low": 8e11},
            {"case_id": "nickel-5e14-5e12", "material": "nickel", "high": 5e14, "low": 5e12},
            {"case_id": "ferritic-steel-2e14-2e12", "material": "ferritic steel", "high": 2e14, "low": 2e12},
        ],
    },
    {
        "family_id": "particle-fraction-modulus",
        "family_name": "Stiff-particle fraction and modulus",
        "outcome_positive": "greater",
        "outcome_negative": "smaller",
        "positive_numeric_direction": "increase",
        "render": particle_fraction_stem,
        "cases": [
            {"case_id": "aluminum-sic-30-10", "material": "aluminum-matrix", "particles": "silicon-carbide", "high": 30, "low": 10},
            {"case_id": "magnesium-alumina-24-8", "material": "magnesium-matrix", "particles": "alumina", "high": 24, "low": 8},
            {"case_id": "nickel-wc-18-6", "material": "nickel-matrix", "particles": "tungsten-carbide", "high": 18, "low": 6},
            {"case_id": "epoxy-silica-36-12", "material": "epoxy-matrix", "particles": "silica", "high": 36, "low": 12},
        ],
    },
    {
        "family_id": "crosslink-density-modulus",
        "family_name": "Crosslink density and rubbery modulus",
        "outcome_positive": "greater",
        "outcome_negative": "smaller",
        "positive_numeric_direction": "increase",
        "render": crosslink_stem,
        "cases": [
            {"case_id": "natural-rubber-800-200", "material": "natural-rubber", "high": 800, "low": 200},
            {"case_id": "silicone-600-150", "material": "silicone-elastomer", "high": 600, "low": 150},
            {"case_id": "polyurethane-1000-250", "material": "polyurethane-elastomer", "high": 1000, "low": 250},
            {"case_id": "epdm-720-180", "material": "EPDM-elastomer", "high": 720, "low": 180},
        ],
    },
]


def build_manifest() -> dict:
    previous = base.FAMILIES
    try:
        base.FAMILIES = FAMILIES
        manifest = base.build_manifest()
    finally:
        base.FAMILIES = previous
    manifest["study_id"] = "late-physics-representation-replication-2026-07-17"
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["design"] = (
        "Disjoint prospective replication of a late-layer transition observed "
        "in lexical-adversarial-representation-2026-07-17. No mechanism family "
        "or material system is reused."
    )
    return manifest


def main() -> None:
    if PROTOCOL.exists() or MANIFEST.exists():
        raise FileExistsError(
            "Frozen replication files already exist; do not overwrite them."
        )
    if not DISCOVERY_STATS.exists():
        raise FileNotFoundError(DISCOVERY_STATS)
    manifest = build_manifest()
    preflight = base.lexical_preflight(manifest)
    base.dump(MANIFEST, manifest)

    tokenizer = AutoTokenizer.from_pretrained(
        base.MODEL, revision=base.MODEL_REVISION, local_files_only=True
    )
    answer_words = sorted({
        row["outcome_positive"] for row in manifest["families"]
    } | {
        row["outcome_negative"] for row in manifest["families"]
    })
    tokenization = {
        word: [int(value) for value in tokenizer.encode(
            word, add_special_tokens=False
        )]
        for word in answer_words
    }
    if any(len(ids) != 1 for ids in tokenization.values()):
        raise RuntimeError(tokenization)
    lenses = [
        {**row, "sha256": base.sha256(ROOT / row["path"])}
        for row in base.LENS_ROWS
    ]
    protocol = {
        "study_id": manifest["study_id"],
        "status": (
            "prospectively frozen before any forward pass on this disjoint "
            "replication cohort; late window was selected from the completed "
            "discovery cohort and is not an independent choice"
        ),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Does the late-layer shift from lexical similarity toward physical "
            "equivalence replicate across six new materials mechanism families?"
        ),
        "model": base.MODEL,
        "model_revision": base.MODEL_REVISION,
        "lenses": lenses,
        "source_layers": base.LAYERS,
        "registered_band_percent": [80.0, 96.0],
        "secondary_full_band_percent": [38.0, 92.0],
        "target_free_layers": [34, 36, 37, 39],
        "target_free_top_k_retained": 256,
        "target_free_display_k": 30,
        "frozen_stopwords": base.FROZEN_STOPWORDS,
        "inputs": {
            "prompt_manifest": str(MANIFEST.relative_to(ROOT)),
            "prompt_manifest_sha256": base.sha256(MANIFEST),
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": base.sha256(RUNNER),
            "discovery_statistics": str(DISCOVERY_STATS.relative_to(ROOT)),
            "discovery_statistics_sha256": base.sha256(DISCOVERY_STATS),
        },
        "tokenization_preflight": tokenization,
        "lexical_adversarial_preflight": preflight,
        "primary_endpoint": {
            "representation": (
                "mean of three Jacobian transported states in the final "
                "decoder basis, centered across all replication prompts at each layer"
            ),
            "triplet_margin": (
                "cosine(anchor, physics_paraphrase) minus "
                "cosine(anchor, lexical_counterfactual)"
            ),
            "scalar": "mean triplet margin over the frozen 80-96 percent late window",
            "inference": (
                "30000 two-stage bootstrap resamples over families and triplets; "
                "seed 20260718"
            ),
        },
        "secondary_endpoints": [
            "late-window direct and raw-state margins",
            "Jacobian-minus-direct late-window contrast",
            "38-92 percent full-band margins retained for comparison with discovery",
            "late 80-96 minus middle 38-70 percent transition contrast",
            "clean scientific answer consistency",
            "target-free top-word Jaccard margins at layers 34, 36, 37, and 39",
        ],
        "success_rule": {
            "primary": "two-stage 95 percent CI for late Jacobian margin above zero",
            "breadth": "at least 18 of 24 triplets and five of six family means positive",
            "all_outputs_retained": True,
        },
        "guardrails": [
            "The late window was motivated by a completed prior cohort and must be labeled as a prospective replication, not an original preregistered hypothesis.",
            "Similarity does not establish causal use or a literal chain of thought.",
            "No prompt, family, layer, clean error, or target-free word is excluded after execution.",
        ],
    }
    base.dump(PROTOCOL, protocol)
    PROTOCOL_MD.write_text(
        "\n".join([
            "# Frozen disjoint replication of the late physical-equivalence transition",
            "",
            f"Frozen: `{protocol['frozen_at']}`",
            "",
            "## Motivation and evidential status",
            "",
            (
                "The completed discovery cohort failed its frozen 38--92% "
                "band endpoint but showed an unregistered late rise that crossed "
                "zero near 80% depth. This new cohort freezes 80--96% as the "
                "primary window before running any of its prompts. It uses six "
                "new mechanisms and 24 new material systems. The window is "
                "prospective for this cohort but motivated by prior output."
            ),
            "",
            "## Primary endpoint",
            "",
            (
                "Centered three-fit Jacobian ensemble cosine(anchor, physically "
                "equivalent paraphrase) minus cosine(anchor, near-verbatim "
                "physical counterfactual), averaged over 80--96% depth. Positive "
                "means physics outranks wording. Inference uses the frozen "
                "two-stage family/triplet bootstrap."
            ),
            "",
            "## Frozen success rule",
            "",
            f"- {protocol['success_rule']['primary']}.",
            f"- {protocol['success_rule']['breadth']}.",
            "- Every output is retained.",
            "",
            "## Disjoint families",
            "",
            *[
                f"- {row['family_name']}"
                for row in manifest["families"]
            ],
            "",
            "## Fingerprints",
            "",
            f"- manifest: `{protocol['inputs']['prompt_manifest_sha256']}`",
            f"- runner: `{protocol['inputs']['runner_sha256']}`",
            f"- motivating statistics: `{protocol['inputs']['discovery_statistics_sha256']}`",
            "",
        ]) + "\n"
    )
    print(f"wrote {MANIFEST}")
    print(f"wrote {PROTOCOL}")
    print(f"wrote {PROTOCOL_MD}")
    print(f"manifest sha256: {base.sha256(MANIFEST)}")
    print(f"protocol sha256: {base.sha256(PROTOCOL)}")


if __name__ == "__main__":
    main()
