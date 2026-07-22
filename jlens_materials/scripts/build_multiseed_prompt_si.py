#!/usr/bin/env python3
"""Build a standalone SI inventory of all 50 paper association prompts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "prompts" / "materials-paper-v2-preregistered.json"
RUN_PATHS = [
    ROOT / "runs" / f"gemma4-e4b-it-paper-seed{seed}.json"
    for seed in range(3)
]
OUTPUT = ROOT / "experiments" / "PAPER_ASSOCIATION_PROMPTS_MULTI_SEED_SI.md"

FAMILY_LABELS = {
    "ductile": "Ductile dimpled failure",
    "boundary-attack": "Boundary-localized stainless attack",
    "cyclic": "Cyclic loading and fatigue",
    "cleavage": "Low-temperature cleavage",
    "high-temperature-deformation": "High-temperature time-dependent deformation",
    "particle-strengthening": "Hard-particle strengthening",
    "rapid-transformation": "Rapid diffusionless transformation",
    "line-defect-motion": "Line-defect motion",
    "notch-resistance": "Notch resistance and toughness",
    "hot-air-surface-layer": "High-temperature surface reaction",
}


def clean(value: object) -> str:
    return str(value or "").replace("\n", " ").strip()


def one_indexed(value: object) -> str:
    if value is None or int(value) < 0:
        return "–"
    return f"{int(value) + 1:,}"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    prompts = [item for item in manifest["prompts"] if item.get("shape") == "ASSOCIATION"]
    counts = Counter(item["category"] for item in prompts)
    if len(prompts) != 50 or set(counts) != set(FAMILY_LABELS) or set(counts.values()) != {5}:
        raise ValueError(f"expected ten five-item families; observed {dict(counts)}")

    runs = [json.loads(path.read_text()) for path in RUN_PATHS]
    records = [{item["slug"]: item for item in run["prompts"]} for run in runs]
    expected_slugs = {item["slug"] for item in prompts}
    for seed, (run, indexed) in enumerate(zip(runs, records)):
        if set(indexed) != expected_slugs:
            raise ValueError(f"seed {seed} prompt set does not match the frozen manifest")
        if run.get("errors") or run.get("insufficient_sample_shapes"):
            raise ValueError(f"seed {seed} is incomplete")
        if not run["methodology"].get("paper_protocol_complete"):
            raise ValueError(f"seed {seed} is not paper-protocol complete")

    lines = [
        "# Supplementary Information: complete 50-prompt materials association dataset",
        "",
        "This document lists every exact prompt in the primary Gemma-4 materials-science "
        "association experiment. It is generated from the frozen prompt manifest and the three "
        "paper-protocol run records. No prompt or candidate concept was selected from its result.",
        "",
        "## Experimental design in plain language",
        "",
        "The dataset contains ten materials-mechanism families with five independently worded "
        "descriptions per family. Each description omits the technical words that will later be "
        "searched for. Gemma processes the description, and both the Jacobian lens and the matched "
        "logit lens rank the same predeclared candidate words at the final prompt token. Lower "
        "vocabulary rank is stronger. Results are reduced over the fixed 38–92% network-depth band; "
        "the band is not selected separately for any prompt.",
        "",
        "The three Jacobian lenses share the same model, recipe, source-layer grid, and target layer. "
        "They differ only in the independently shuffled 1,000-record WikiText fitting sample. "
        "The three samples have different aggregate content hashes but are not asserted to be "
        "pairwise disjoint.",
        "",
        "A prompt is valid only when every scored candidate is absent from the tokenized input and "
        "from the clean generated continuation. Concept traces are linear readout measurements, not "
        "literal transcripts of private prose and not evidence by themselves that a concept caused "
        "the generated answer.",
        "",
        "## Run and lens provenance",
        "",
        "| seed | run record | fitting records | corpus SHA-256 | model revision | protocol status |",
        "|---:|---|---:|---|---|---|",
    ]
    for seed, (path, run) in enumerate(zip(RUN_PATHS, runs)):
        provenance = run["lens_provenance"]
        lines.append(
            f"| {seed} | [`{path.name}`](../runs/{path.name}) | {run['lens_n_prompts']:,} | "
            f"`{provenance['corpus']['sha256']}` | "
            f"`{run['model_identity']['model_revision']}` | paper-protocol complete |"
        )
    lines.extend([
        "",
        "Common lens recipe: 128-token fitting sequences; target layer 40 (penultimate); source "
        "layers `0, 2, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23, 24, 26, 28, 29, "
        "31, 32, 34, 36, 37, 39`; Gemma revision "
        f"`{runs[0]['model_identity']['model_revision']}`.",
        "",
        "## Complete prompt inventory and results",
        "",
        "Ranks below are one-indexed. The logit-lens rank is identical across seeds because it uses "
        "the same model, prompt, token position, and source-layer grid; it does not depend on the "
        "fitted Jacobian lens. `–` means that the sustained-top-5 criterion was not met.",
        "",
    ])

    family_counts: Counter[str] = Counter()
    current_family = None
    for prompt in prompts:
        family = prompt["category"]
        if family != current_family:
            current_family = family
            lines.extend([f"## {FAMILY_LABELS[family]}", ""])
        family_counts[family] += 1
        slug = prompt["slug"]
        seed_records = [indexed[slug] for indexed in records]
        reference = seed_records[0]
        for seed, record in enumerate(seed_records):
            if record["prompt_text"] != prompt["text"]:
                raise ValueError(f"seed {seed} text mismatch for {slug}")
            retained = [item["label"] for item in record["tracked"]]
            dropped = [
                item.get("label", "") if isinstance(item, dict) else str(item)
                for item in record.get("tracked_dropped", [])
            ]
            if [item for item in prompt["tracked"] if item not in dropped] != retained:
                raise ValueError(f"seed {seed} tracked-concept mismatch for {slug}")

        retained_labels = [item["label"] for item in reference["tracked"]]
        dropped_labels = [
            item.get("label", "") if isinstance(item, dict) else str(item)
            for item in reference.get("tracked_dropped", [])
        ]

        lines.extend([
            f"### {family_counts[family]}. `{slug}` — {prompt['title']}",
            "",
            f"**Exact model input:** {clean(prompt['text'])}",
            "",
            f"**Predeclared concepts:** {', '.join(f'`{item}`' for item in prompt['tracked'])}",
            "",
            f"**Directly ranked concepts:** {', '.join(f'`{item}`' for item in retained_labels)}. "
            f"**Tokenizer-dropped multi-token concepts:** "
            f"{', '.join(f'`{item}`' for item in dropped_labels) if dropped_labels else 'none'}.",
            "",
            f"**Fixed readout:** `{reference['readout_selector']}` at token position "
            f"`{reference['score_positions']}`. **Clean continuation:** "
            f"`{clean(reference['generated_completion'])}`. **Included:** "
            f"`{all(item['valid_for_metrics'] for item in seed_records)}` in all three seeds.",
            "",
            "| concept | seed 0: J rank (depth) | seed 1: J rank (depth) | seed 2: J rank (depth) | logit rank | sustained top-5 onset by seed |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for label in retained_labels:
            seed_emergence = [
                next(item for item in record["emergence"] if item["label"] == label)
                for record in seed_records
            ]
            j_cells = [
                f"{one_indexed(item['best_rank'])} ({float(item['best_depth']):.1f}%)"
                for item in seed_emergence
            ]
            onsets = [
                "–" if item["onset_depth"] is None else f"{float(item['onset_depth']):.1f}%"
                for item in seed_emergence
            ]
            lines.append(
                f"| `{label}` | {j_cells[0]} | {j_cells[1]} | {j_cells[2]} | "
                f"{one_indexed(seed_emergence[0]['logit_lens_best_rank'])} | "
                f"{', '.join(f'S{seed}: {onset}' for seed, onset in enumerate(onsets))} |"
            )
        lines.append("")

    lines.extend([
        "## Reading the inventory correctly",
        "",
        "Each five-prompt family is a collection of alternate descriptions of one mechanism, not "
        "five unrelated scientific concepts. Accordingly, family-clustered or hierarchical "
        "uncertainty should be used for population-level claims. The individual entries remain "
        "valuable for inspecting phrasing sensitivity and for selecting illustrative examples, "
        "provided that any selection is clearly labeled as illustrative rather than confirmatory.",
        "",
        "Generated from:",
        "",
        "- `prompts/materials-paper-v2-preregistered.json`",
        "- `runs/gemma4-e4b-it-paper-seed0.json`",
        "- `runs/gemma4-e4b-it-paper-seed1.json`",
        "- `runs/gemma4-e4b-it-paper-seed2.json`",
        "",
        "Regenerate with `python scripts/build_multiseed_prompt_si.py` from the "
        "`jlens_materials` directory.",
        "",
    ])
    OUTPUT.write_text("\n".join(lines))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
