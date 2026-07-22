#!/usr/bin/env python3
"""Create and fingerprint the prospective lexical-adversarial prompt suite."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "lexical-adversarial-representation-2026-07-17"
MANIFEST = OUT / "prompt_manifest.json"
PROTOCOL = OUT / "protocol.json"
PROTOCOL_MD = OUT / "PROTOCOL.md"
RUNNER = ROOT / "scripts" / "run_lexical_adversarial_representation.py"

MODEL = "google/gemma-4-E4B-it"
MODEL_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
LAYERS = [
    0, 2, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23, 24, 26,
    28, 29, 31, 32, 34, 36, 37, 39,
]
TARGET_FREE_LAYERS = [13, 20, 28, 34, 39]
LENS_ROWS = [
    {"seed": 0, "path": "lenses/hub/gemma4-e4b-it/paper/seed0.pt"},
    {"seed": 1, "path": "lenses/hub/gemma4-e4b-it/paper/seed1.pt"},
    {"seed": 2, "path": "lenses/hub/gemma4-e4b-it/paper/seed2.pt"},
]
FROZEN_STOPWORDS = sorted(set("""
a about after again against all also am an and any are as at be because been
before being between both but by can could did do does doing down during each
exactly for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just lower higher increase
decrease greater smaller likely may me more most must my myself no nor not of
off on once one only or ordered other our ours ourselves out over own pair
same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were
what when where which while who whom why will with word would you your yours
yourself yourselves answer choose select resulting result trend relative
initial state happens become becomes
""".split()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def fmt(value: float, digits: int = 3) -> str:
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def ordered_user(stem: str, positive: str, negative: str, order: str) -> tuple[str, list[str]]:
    words = [positive, negative] if order == "positive-first" else [negative, positive]
    return (
        f"{stem} Answer exactly one lowercase word from this ordered pair: "
        f"{words[0]}, {words[1]}.",
        words,
    )


def grain_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    material = case["material"]
    if not paraphrase:
        return (
            f"A conventional polycrystalline {material} changes from "
            f"{fmt(start, 1)}-micrometer to {fmt(end, 1)}-micrometer equiaxed "
            "grains while composition, texture, precipitates, porosity, and "
            "dislocation density are held fixed. Relative to the initial "
            "microstructure, what happens to yield strength?"
        )
    direction = "contracts" if end < start else "expands"
    return (
        f"In the same {material}, no microstructural variable changes except "
        f"mean crystallite diameter. It {direction} from {fmt(start / 1000)} "
        f"millimeters to {fmt(end / 1000)} millimeters. The alloy remains in "
        "the conventional Hall-Petch regime. Select the resulting trend in "
        "flow strength."
    )


def fracture_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    material = case["material"]
    if not paraphrase:
        return (
            f"A {material} plate contains a through crack whose half-length "
            f"changes from {fmt(start)} millimeters to {fmt(end)} millimeters "
            "while elastic modulus, fracture energy, geometry factor, "
            "thickness, and loading mode remain fixed. What happens to the "
            "remote tensile stress required for unstable fracture?"
        )
    direction = "shortens" if end < start else "lengthens"
    return (
        f"Keeping toughness and specimen geometry unchanged in the same "
        f"{material}, the flaw {direction} from {fmt(start * 1000, 1)} "
        f"micrometers to {fmt(end * 1000, 1)} micrometers in half-length. "
        "Under Griffith scaling, select the trend in critical far-field stress."
    )


def diffusion_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    system = case["material"]
    species = case["species"]
    if not paraphrase:
        return (
            f"In single-phase {system}, temperature changes from {fmt(start, 1)} "
            f"kelvin to {fmt(end, 1)} kelvin while composition, phase, pressure, "
            f"and point-defect chemistry are held fixed. What happens to the "
            f"diffusion coefficient of {species}?"
        )
    direction = "heated" if end > start else "cooled"
    return (
        f"Keeping the activation barrier and prefactor unchanged for {species} "
        f"in the same {system}, the specimen is {direction} from "
        f"{fmt(start - 273.15, 2)} degrees Celsius to "
        f"{fmt(end - 273.15, 2)} degrees Celsius. Under Arrhenius behavior, "
        "select the trend in diffusivity."
    )


def fiber_angle_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    material = case["material"]
    if not paraphrase:
        return (
            f"In unidirectional {material}, fiber angle relative to the tensile "
            f"axis changes from {fmt(start, 1)} degrees to {fmt(end, 1)} degrees "
            "while constituent moduli, fiber fraction, porosity, and interface "
            "quality remain fixed. What happens to the effective axial modulus?"
        )
    direction = "more aligned with" if end < start else "farther from"
    return (
        f"The same {material} lamina is rotated so its reinforcement becomes "
        f"{direction} the loading direction: orientation changes from "
        f"{fmt(np.deg2rad(start))} radians to {fmt(np.deg2rad(end))} radians. "
        "All constituent and interface properties are unchanged. Select the "
        "trend in longitudinal stiffness."
    )


def fiber_fraction_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    material = case["material"]
    if not paraphrase:
        return (
            f"A unidirectional {material} composite changes from "
            f"{fmt(start, 1)} percent to {fmt(end, 1)} percent aligned fiber "
            "volume fraction while fiber orientation, constituent moduli, "
            "porosity, and interface quality remain fixed. What happens to the "
            "longitudinal elastic modulus?"
        )
    direction = "rises" if end > start else "falls"
    return (
        f"In the same {material} system, the fraction of load-parallel "
        f"reinforcement {direction} from {fmt(start / 100)} to "
        f"{fmt(end / 100)} of the total volume. Matrix, reinforcement, "
        "alignment, and bonding are unchanged. Select the trend in axial "
        "stiffness."
    )


def martensite_stem(case: dict, start: float, end: float, paraphrase: bool) -> str:
    material = case["material"]
    if not paraphrase:
        return (
            f"A {material} microstructure changes from {fmt(start, 1)} percent "
            f"to {fmt(end, 1)} percent martensite, with the balance ferrite, "
            "while carbon content, prior-austenite grain size, tempering state, "
            "and residual stress remain fixed. What happens to bulk hardness?"
        )
    direction = "grows" if end > start else "shrinks"
    return (
        f"In the same {material}, the hard transformed-phase fraction "
        f"{direction} from {fmt(start / 100)} to {fmt(end / 100)} of the "
        "microstructure and ferrite supplies the remainder. Chemistry, parent "
        "grain scale, temper, and residual stress do not change. Select the "
        "trend in indentation hardness."
    )


FAMILIES = [
    {
        "family_id": "grain-size-strength",
        "family_name": "Grain size and yield strength",
        "outcome_positive": "higher",
        "outcome_negative": "lower",
        "positive_numeric_direction": "decrease",
        "render": grain_stem,
        "cases": [
            {"case_id": "nickel-superalloy-80-20", "material": "wrought nickel-based superalloy", "high": 80, "low": 20},
            {"case_id": "ferritic-stainless-60-15", "material": "ferritic stainless steel", "high": 60, "low": 15},
            {"case_id": "aluminum-alloy-48-12", "material": "precipitation-hardened aluminum alloy", "high": 48, "low": 12},
            {"case_id": "alpha-brass-36-9", "material": "alpha brass", "high": 36, "low": 9},
        ],
    },
    {
        "family_id": "crack-size-fracture",
        "family_name": "Crack size and critical fracture stress",
        "outcome_positive": "higher",
        "outcome_negative": "lower",
        "positive_numeric_direction": "decrease",
        "render": fracture_stem,
        "cases": [
            {"case_id": "fused-silica-800-200", "material": "fused-silica", "high": 0.8, "low": 0.2},
            {"case_id": "alumina-600-150", "material": "dense alumina", "high": 0.6, "low": 0.15},
            {"case_id": "silicon-carbide-400-100", "material": "silicon-carbide", "high": 0.4, "low": 0.1},
            {"case_id": "soda-lime-glass-1000-250", "material": "soda-lime-glass", "high": 1.0, "low": 0.25},
        ],
    },
    {
        "family_id": "temperature-diffusion",
        "family_name": "Temperature and diffusivity",
        "outcome_positive": "increase",
        "outcome_negative": "decrease",
        "positive_numeric_direction": "increase",
        "render": diffusion_stem,
        "cases": [
            {"case_id": "nickel-self-1300-1100", "material": "nickel", "species": "nickel atoms", "high": 1300, "low": 1100},
            {"case_id": "carbon-austenite-1250-1050", "material": "austenitic iron", "species": "interstitial carbon", "high": 1250, "low": 1050},
            {"case_id": "oxygen-ysz-1000-800", "material": "yttria-stabilized zirconia", "species": "oxygen ions", "high": 1000, "low": 800},
            {"case_id": "lithium-lco-420-300", "material": "lithium cobalt oxide", "species": "lithium ions", "high": 420, "low": 300},
        ],
    },
    {
        "family_id": "fiber-angle-stiffness",
        "family_name": "Fiber angle and axial stiffness",
        "outcome_positive": "greater",
        "outcome_negative": "smaller",
        "positive_numeric_direction": "decrease",
        "render": fiber_angle_stem,
        "cases": [
            {"case_id": "carbon-epoxy-80-10", "material": "carbon-fiber epoxy", "high": 80, "low": 10},
            {"case_id": "glass-epoxy-75-15", "material": "glass-fiber epoxy", "high": 75, "low": 15},
            {"case_id": "aramid-epoxy-70-20", "material": "aramid-fiber epoxy", "high": 70, "low": 20},
            {"case_id": "sic-titanium-65-5", "material": "silicon-carbide-fiber titanium", "high": 65, "low": 5},
        ],
    },
    {
        "family_id": "fiber-fraction-stiffness",
        "family_name": "Fiber fraction and longitudinal stiffness",
        "outcome_positive": "greater",
        "outcome_negative": "smaller",
        "positive_numeric_direction": "increase",
        "render": fiber_fraction_stem,
        "cases": [
            {"case_id": "carbon-peek-55-25", "material": "carbon-fiber PEEK", "high": 55, "low": 25},
            {"case_id": "glass-polyester-50-20", "material": "glass-fiber polyester", "high": 50, "low": 20},
            {"case_id": "aramid-epoxy-45-15", "material": "aramid-fiber epoxy", "high": 45, "low": 15},
            {"case_id": "sic-titanium-40-10", "material": "silicon-carbide-fiber titanium", "high": 40, "low": 10},
        ],
    },
    {
        "family_id": "martensite-hardness",
        "family_name": "Martensite fraction and hardness",
        "outcome_positive": "higher",
        "outcome_negative": "lower",
        "positive_numeric_direction": "increase",
        "render": martensite_stem,
        "cases": [
            {"case_id": "medium-carbon-steel-80-20", "material": "medium-carbon steel", "high": 80, "low": 20},
            {"case_id": "dual-phase-steel-60-15", "material": "dual-phase steel", "high": 60, "low": 15},
            {"case_id": "low-alloy-steel-70-10", "material": "low-alloy steel", "high": 70, "low": 10},
            {"case_id": "martensitic-stainless-75-25", "material": "martensitic stainless steel", "high": 75, "low": 25},
        ],
    },
]


def build_manifest() -> dict:
    prompts: list[dict] = []
    triplets: list[dict] = []
    family_rows: list[dict] = []
    global_index = 0
    for family in FAMILIES:
        positive = family["outcome_positive"]
        negative = family["outcome_negative"]
        family_rows.append({
            key: family[key] for key in [
                "family_id", "family_name", "outcome_positive",
                "outcome_negative", "positive_numeric_direction",
            ]
        })
        for case_index, case in enumerate(family["cases"]):
            anchor_positive = case_index % 2 == 0
            positive_is_increase = family["positive_numeric_direction"] == "increase"
            if anchor_positive == positive_is_increase:
                start, end = case["low"], case["high"]
            else:
                start, end = case["high"], case["low"]
            counter_start, counter_end = end, start
            expected_anchor = positive if anchor_positive else negative
            expected_counter = negative if anchor_positive else positive
            order = "positive-first" if global_index % 2 == 0 else "negative-first"
            triplet_id = f"{family['family_id']}--{case['case_id']}"
            variants = [
                (
                    "anchor",
                    family["render"](case, start, end, False),
                    expected_anchor,
                    start,
                    end,
                ),
                (
                    "physics_paraphrase",
                    family["render"](case, start, end, True),
                    expected_anchor,
                    start,
                    end,
                ),
                (
                    "lexical_counterfactual",
                    family["render"](case, counter_start, counter_end, False),
                    expected_counter,
                    counter_start,
                    counter_end,
                ),
            ]
            prompt_ids = {}
            for variant, stem, expected, value_start, value_end in variants:
                user, words = ordered_user(stem, positive, negative, order)
                prompt_id = f"{triplet_id}--{variant}"
                prompt_ids[variant] = prompt_id
                prompts.append({
                    "prompt_id": prompt_id,
                    "triplet_id": triplet_id,
                    "family_id": family["family_id"],
                    "family_name": family["family_name"],
                    "case_id": case["case_id"],
                    "variant": variant,
                    "presentation_order": order,
                    "presented_words": words,
                    "outcome_positive": positive,
                    "outcome_negative": negative,
                    "expected_outcome": expected,
                    "numeric_start": value_start,
                    "numeric_end": value_end,
                    "numeric_direction": (
                        "increase" if value_end > value_start else "decrease"
                    ),
                    "anchor_expected_positive": anchor_positive,
                    "stem": stem,
                    "user": user,
                })
            triplets.append({
                "triplet_id": triplet_id,
                "family_id": family["family_id"],
                "case_id": case["case_id"],
                "anchor_expected_outcome": expected_anchor,
                "counterfactual_expected_outcome": expected_counter,
                "presentation_order": order,
                "prompt_ids": prompt_ids,
            })
            global_index += 1
    return {
        "study_id": "lexical-adversarial-representation-2026-07-17",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": (
            "Each frozen triplet contains an anchor, a physically equivalent "
            "paraphrase with different wording and converted units, and a "
            "near-verbatim numerical reversal with the opposite physical answer."
        ),
        "families": family_rows,
        "triplets": triplets,
        "prompts": prompts,
    }


def lexical_preflight(manifest: dict) -> dict:
    prompt_by_id = {row["prompt_id"]: row for row in manifest["prompts"]}
    texts = [row["user"] for row in manifest["prompts"]]
    indices = {row["prompt_id"]: index for index, row in enumerate(manifest["prompts"])}
    output: dict[str, dict] = {}
    for name, vectorizer in [
        ("word_tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        ("character_tfidf", TfidfVectorizer(
            lowercase=True, analyzer="char_wb", ngram_range=(3, 5)
        )),
    ]:
        matrix = vectorizer.fit_transform(texts)
        rows = []
        for triplet in manifest["triplets"]:
            ids = triplet["prompt_ids"]
            anchor = indices[ids["anchor"]]
            paraphrase = indices[ids["physics_paraphrase"]]
            counter = indices[ids["lexical_counterfactual"]]
            ap = float(cosine_similarity(matrix[anchor], matrix[paraphrase])[0, 0])
            ac = float(cosine_similarity(matrix[anchor], matrix[counter])[0, 0])
            rows.append({
                "triplet_id": triplet["triplet_id"],
                "anchor_paraphrase_similarity": ap,
                "anchor_counterfactual_similarity": ac,
                "physics_minus_lexical_margin": ap - ac,
            })
        margins = np.asarray([row["physics_minus_lexical_margin"] for row in rows])
        output[name] = {
            "all_lexical_counterfactuals_closer": bool(np.all(margins < 0)),
            "n_counterfactuals_closer": int(np.sum(margins < 0)),
            "n_triplets": len(rows),
            "mean_physics_minus_lexical_margin": float(margins.mean()),
            "rows": rows,
        }
    if not all(
        result["all_lexical_counterfactuals_closer"] for result in output.values()
    ):
        failed = {
            key: [
                row["triplet_id"] for row in value["rows"]
                if row["physics_minus_lexical_margin"] >= 0
            ]
            for key, value in output.items()
        }
        raise RuntimeError(f"lexical-adversarial preflight failed: {failed}")
    return output


def main() -> None:
    if PROTOCOL.exists() or MANIFEST.exists():
        raise FileExistsError(
            "Frozen files already exist; do not overwrite them. Create a new "
            "dated study directory for a revised protocol."
        )
    if not RUNNER.exists():
        raise FileNotFoundError(RUNNER)
    manifest = build_manifest()
    preflight = lexical_preflight(manifest)
    dump(MANIFEST, manifest)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISION, local_files_only=True
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
        raise RuntimeError(f"answer words must be single tokens: {tokenization}")

    lenses = []
    for row in LENS_ROWS:
        path = ROOT / row["path"]
        lenses.append({**row, "sha256": sha256(path)})
    protocol = {
        "study_id": manifest["study_id"],
        "status": (
            "prospectively frozen before any model forward pass, hidden-state "
            "inspection, lens readout, or target-free vocabulary output"
        ),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "When wording and physics disagree, do Gemma's internal states "
            "become more similar for physically equivalent descriptions than "
            "for near-verbatim descriptions with the opposite physical relation?"
        ),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "lenses": lenses,
        "source_layers": LAYERS,
        "registered_band_percent": [38.0, 92.0],
        "target_free_layers": TARGET_FREE_LAYERS,
        "target_free_top_k_retained": 256,
        "target_free_display_k": 30,
        "frozen_stopwords": FROZEN_STOPWORDS,
        "inputs": {
            "prompt_manifest": str(MANIFEST.relative_to(ROOT)),
            "prompt_manifest_sha256": sha256(MANIFEST),
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": sha256(RUNNER),
        },
        "tokenization_preflight": tokenization,
        "lexical_adversarial_preflight": preflight,
        "primary_endpoint": {
            "representation": (
                "mean of three Jacobian transported states in the final "
                "decoder basis, centered across all frozen prompts at each layer"
            ),
            "triplet_margin": (
                "cosine(anchor, physics_paraphrase) minus "
                "cosine(anchor, lexical_counterfactual)"
            ),
            "scalar": "mean triplet margin over the fixed 38-92 percent band",
            "independent_units": "24 material-system triplets in six mechanism families",
            "inference": (
                "30000 two-stage bootstrap resamples: mechanism families, then "
                "triplets within resampled families; seed 20260717"
            ),
        },
        "secondary_endpoints": [
            "direct-unembedding-basis and raw-residual triplet margins",
            "paired Jacobian-minus-direct triplet margin",
            "fraction of triplets and family means with positive band margin",
            "clean answer accuracy and anchor-paraphrase-counterfactual consistency",
            (
                "target-free top-30 word-set Jaccard margin at the five frozen "
                "layers after triplet-union prompt morphology filtering"
            ),
            "word and character TF-IDF lexical margins",
        ],
        "success_rule": {
            "primary": (
                "the two-stage 95 percent bootstrap interval for the Jacobian "
                "ensemble band margin is above zero"
            ),
            "breadth": (
                "at least 18 of 24 triplets and at least five of six family "
                "means have positive Jacobian band margins"
            ),
            "all_outputs_retained": True,
        },
        "guardrails": [
            "Similarity is evidence of representation organization, not causal use.",
            "A positive result does not reveal a literal chain of thought.",
            "The suite tests six monotonic textbook relations, not unrestricted materials reasoning.",
            "No prompt, family, layer, or word is excluded after execution.",
        ],
    }
    dump(PROTOCOL, protocol)

    PROTOCOL_MD.write_text(
        "\n".join([
            "# Frozen lexical-adversarial representation protocol",
            "",
            f"Frozen: `{protocol['frozen_at']}`",
            "",
            "## Scientific question",
            "",
            protocol["scientific_question"],
            "",
            "## Design",
            "",
            (
                "The suite contains 24 independently parameterized materials "
                "triplets across six mechanism families. Each triplet contains "
                "(i) an anchor, (ii) a scientifically equivalent paraphrase "
                "with changed terminology and converted units, and (iii) a "
                "near-verbatim numerical reversal with the opposite physical "
                "answer. Answer order is fixed within a triplet and balanced "
                "across triplets."
            ),
            "",
            (
                "Before freezing, both word and character TF-IDF selected the "
                f"lexical counterfactual as the closer neighbor in all "
                f"{len(manifest['triplets'])} triplets. This is a design "
                "preflight, not a model result."
            ),
            "",
            "## Frozen primary endpoint",
            "",
            (
                "At every registered layer, center the three-fit mean Jacobian "
                "target states across all 72 prompts and calculate cosine "
                "similarity. For each triplet, subtract anchor-to-counterfactual "
                "similarity from anchor-to-paraphrase similarity. Positive "
                "values mean that physical equivalence outranks lexical overlap. "
                "Average over the fixed 38--92% layer band and use the frozen "
                "two-stage family/triplet bootstrap."
            ),
            "",
            "## Secondary endpoints",
            "",
            *[f"- {item}" for item in protocol["secondary_endpoints"]],
            "",
            "## Guardrails",
            "",
            *[f"- {item}" for item in protocol["guardrails"]],
            "",
            "## Fingerprints",
            "",
            f"- prompt manifest: `{protocol['inputs']['prompt_manifest_sha256']}`",
            f"- execution runner: `{protocol['inputs']['runner_sha256']}`",
            *[
                f"- lens seed {row['seed']}: `{row['sha256']}`"
                for row in protocol["lenses"]
            ],
            "",
        ]) + "\n"
    )
    print(f"wrote {MANIFEST}")
    print(f"wrote {PROTOCOL}")
    print(f"wrote {PROTOCOL_MD}")
    print(f"manifest sha256: {sha256(MANIFEST)}")
    print(f"protocol sha256: {sha256(PROTOCOL)}")


if __name__ == "__main__":
    main()
