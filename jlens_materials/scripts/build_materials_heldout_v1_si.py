#!/usr/bin/env python3
"""Build the complete prompt-and-result SI for materials held-out v1."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "prompts" / "materials-heldout-v1-preregistered.json"
RUN_PATHS = [
    ROOT / "runs" / f"gemma4-e4b-it-heldout-v1-seed{seed}.json"
    for seed in range(3)
]
STATS = ROOT / "experiments" / "materials-heldout-v1_statistics.json"
OUTPUT = ROOT / "experiments" / "MATERIALS_HELDOUT_V1_COMPLETE_SI.md"


FAMILY_LABELS = {
    "ductile": "Energy-absorbing dimpled separation",
    "boundary-attack": "Chromium-depleted interfacial attack",
    "cyclic": "Progressive damage under repeated loading",
    "cleavage": "Faceted low-plasticity separation",
    "high-temperature-deformation": "Time-dependent deformation under heat",
    "particle-strengthening": "Nonshearable particle obstacles",
    "rapid-transformation": "Quench-induced coordinated transformation",
    "line-defect-motion": "Motion of linear lattice imperfections",
    "notch-resistance": "Resistance to unstable flaw extension",
    "hot-air-surface-layer": "Oxygen-rich high-temperature surface film",
}


def clean(value: object) -> str:
    return str(value or "").replace("\n", " ").strip()


def one_indexed(value: object) -> str:
    if value is None or int(value) < 0:
        return "–"
    return f"{int(value) + 1:,}"


def candidate_text(rows: list[dict], limit: int = 10) -> str:
    if not rows:
        return "none survived strict consensus"
    return ", ".join(
        f"`{row['token']}` ({row['consensus_score']:.1f})"
        for row in rows[:limit]
    )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    prompts = manifest["prompts"]
    counts = Counter(prompt["target_family"] for prompt in prompts)
    if len(prompts) != 50 or len(counts) != 10 or set(counts.values()) != {5}:
        raise ValueError(f"expected 10 x 5 prompts, found {dict(counts)}")
    runs = [json.loads(path.read_text()) for path in RUN_PATHS]
    indexes = [{row["slug"]: row for row in run["prompts"]} for run in runs]
    expected_slugs = [prompt["slug"] for prompt in prompts]
    for seed, (run, index) in enumerate(zip(runs, indexes, strict=True)):
        if list(index) != expected_slugs:
            raise ValueError(f"seed {seed} prompt order/set mismatch")
        if run.get("errors") or not run["methodology"]["paper_protocol_complete"]:
            raise ValueError(f"seed {seed} is incomplete")

    stats = json.loads(STATS.read_text())
    open_by_method = {
        method: {
            row["slug"]: row
            for row in result["per_prompt"]
        }
        for method, result in stats["open_vocabulary"]["methods"].items()
    }
    prompt_results = {
        row["slug"]: row
        for row in csv.DictReader(
            (ROOT / "experiments" / "materials-heldout-v1_prompt_results.csv").open()
        )
    }

    lines = [
        "# Supplementary Information: complete materials held-out v1 dataset",
        "",
        "This document exposes every exact input, every predeclared physical term, the one-token continuation used for leakage checks, all three fitted-lens ranks, the matched direct-unembedding rank, and the leading target-free prompt candidates. It is generated from the frozen manifest and checksum-locked raw runs; no example was selected because it looked attractive.",
        "",
        "## How to read the tables",
        "",
        "Rank 1 is the strongest word in Gemma's 262,144-token vocabulary. Jacobian and direct ranks use the same final-prompt-token position and the same fixed 38–92% source-layer band. The best rank is the strongest point in that registered band. The target-free lists were generated without consulting the predeclared words; scores count how often a word was the unrestricted top readout across positions and layers, averaged over the three lens records after strict consensus.",
        "",
        "The three Jacobian columns are repeated fits of the readout on different 1,000-record WikiText-103 samples. They are not three different Gemma models. Direct unembedding has no fitted lens, so its value is identical across runs.",
        "",
        "These are semantic readouts, not a literal chain of thought and not proof that the word caused an answer.",
        "",
        "## Frozen provenance",
        "",
        f"- Manifest: `{MANIFEST.relative_to(ROOT)}`",
        "- Manifest SHA-256: `8c034cf33d287d379fddf842971914ec035a22b9e31d29f457258bf85c52e203`",
        "- Model: `google/gemma-4-E4B-it`",
        f"- Model revision: `{runs[0]['model_identity']['model_revision']}`",
        "- Lens fitting: 1,000 unique 128-token WikiText-103 records per seed; penultimate target; 25 registered source layers",
        "",
        "| seed | raw run | raw SHA-256 | fitting corpus SHA-256 |",
        "|---:|---|---|---|",
    ]
    for seed, (path, run_meta) in enumerate(zip(RUN_PATHS, stats["runs"], strict=True)):
        lines.append(
            f"| {seed} | [`{path.name}`](../runs/{path.name}) | "
            f"`{run_meta['sha256']}` | `{runs[seed]['lens_provenance']['corpus']['sha256']}` |"
        )
    lines.extend([
        "",
        "## Complete prompt inventory",
        "",
    ])

    current_family = None
    family_index: Counter[str] = Counter()
    for prompt in prompts:
        family = prompt["target_family"]
        if family != current_family:
            current_family = family
            lines.extend([f"## {FAMILY_LABELS[family]}", ""])
        family_index[family] += 1
        slug = prompt["slug"]
        records = [index[slug] for index in indexes]
        reference = records[0]
        for seed, record in enumerate(records):
            if record["prompt_text"] != prompt["text"]:
                raise ValueError(f"seed {seed} prompt text mismatch for {slug}")
        retained = [row["label"] for row in reference["tracked"]]
        dropped = [str(row) for row in reference.get("tracked_dropped", [])]
        completion_values = [clean(record["generated_completion"]) for record in records]
        if len(set(completion_values)) != 1:
            raise ValueError(f"clean completion differs across seed records for {slug}")
        prompt_auc = prompt_results[slug]
        lines.extend([
            f"### {family_index[family]}. `{slug}`",
            "",
            f"> {clean(prompt['text'])}",
            "",
            f"**Predeclared before execution:** {', '.join(f'`{term}`' for term in prompt['tracked'])}.",
            "",
            f"**Single-token rank endpoints:** {', '.join(f'`{term}`' for term in retained)}. "
            f"**Documented but tokenizer-dropped:** {', '.join(f'`{term}`' for term in dropped) if dropped else 'none'}.",
            "",
            f"**Readout and leakage control:** final prompt token at position `{reference['score_positions'][0]}`; one-token continuation `{completion_values[0]}`; included in all three runs: `{all(record['valid_for_metrics'] for record in records)}`.",
            "",
            f"**Prompt AUC:** Jacobian seed mean `{float(prompt_auc['j_auc_mean_seed']):.4f}`; direct `{float(prompt_auc['logit_auc']):.4f}`; difference `{float(prompt_auc['delta_auc_mean_seed']):+.4f}`.",
            "",
            "| concept | J seed 0 rank (depth) | J seed 1 rank (depth) | J seed 2 rank (depth) | direct rank |",
            "|---|---:|---:|---:|---:|",
        ])
        for label in retained:
            emergence = [
                next(row for row in record["emergence"] if row["label"] == label)
                for record in records
            ]
            cells = [
                f"{one_indexed(row['best_rank'])} ({float(row['best_depth']):.1f}%)"
                for row in emergence
            ]
            lines.append(
                f"| `{label}` | {cells[0]} | {cells[1]} | {cells[2]} | "
                f"{one_indexed(emergence[0]['logit_lens_best_rank'])} |"
            )
        j_candidates = open_by_method["jacobian"][slug]["filtered_consensus_candidates"]
        logit_candidates = open_by_method["logit"][slug]["filtered_consensus_candidates"]
        lines.extend([
            "",
            f"**Leading target-free Jacobian candidates:** {candidate_text(j_candidates)}.",
            "",
            f"**Leading target-free direct candidates:** {candidate_text(logit_candidates)}.",
            "",
        ])

    lines.extend([
        "## Family-level target-free sets used for blinded rating",
        "",
        "The exact shuffled sheet is `MATERIALS_HELDOUT_V1_BLINDED_SETS.csv`; its answer key is separate. The table below is unblinded for audit after rating.",
        "",
        "| method | physical family | eight ranked candidate words |",
        "|---|---|---|",
    ])
    for method, result in stats["open_vocabulary"]["methods"].items():
        for family, family_result in sorted(result["families"].items()):
            candidates = ", ".join(
                f"`{row['token']}` ({row['prompt_support']}/5)"
                for row in family_result["candidates"]
            )
            lines.append(f"| {method} | {family} | {candidates} |")
    lines.extend([
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/build_materials_heldout_v1_si.py",
        "```",
        "",
    ])
    OUTPUT.write_text("\n".join(lines))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
