#!/usr/bin/env python3
"""Seed-consensus open-vocabulary discovery for the 50 materials prompts.

Candidate generation is independent of the predeclared concept lists. Those
lists are consulted only after ranking to annotate exact overlaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_PATHS = [
    ROOT / "runs" / f"gemma4-e4b-it-paper-seed{seed}.json"
    for seed in range(3)
]
EXP_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "figures" / "gemma4-paper-multiseed"
STATS_PATH = EXP_DIR / "open-vocabulary-consensus_statistics.json"
CANDIDATE_CSV = EXP_DIR / "open-vocabulary-consensus_candidates.csv"
BLIND_CSV = EXP_DIR / "OPEN_VOCABULARY_BLINDED_FAMILY_SETS.csv"
KEY_CSV = EXP_DIR / "OPEN_VOCABULARY_BLINDED_FAMILY_KEY.csv"
REPORT_PATH = EXP_DIR / "OPEN_VOCABULARY_CONSENSUS_ANALYSIS.md"
FIGURE_PATH = FIG_DIR / "open-vocabulary-consensus"
ANALYSIS_SEED = 20260714
TOP_FAMILY_CANDIDATES = 8


# Target-agnostic common English function words. The list is stored here so the
# filter is frozen and auditable; it contains no materials-science vocabulary.
FUNCTION_WORDS = set("""
a about above across after afterwards again against all almost along already
also although always am among an and another any are around as at away back be
became because become been before began behind being below between both but by
can cannot could did do does done down due during each either else enough
especially even ever every few for from further furthermore get got had has
have having he hence her here hers herself him himself his how however i if in
indeed into is it its itself just later least less like likely may me meanwhile
might mine more most much must my myself nearly neither never nevertheless no
nobody nor not nothing now of off often on once one only or other others
otherwise our ours ourselves out over own perhaps quite rather really said same
several she should since so some somebody someone something sometimes still
such than that the their theirs them themselves then thence there therefore
these they this those though through thus to too toward under until up upon us
very was we were what whatever when where whereas whether which while who whom
whose why will with within without would yet you your yours yourself yourselves
subsequently consequently simultaneously accordingly eventually ultimately
initially thereto thereafter
""".split())


def normalize_token(value: str) -> str | None:
    token = value.strip().lower()
    if not re.fullmatch(r"[a-z][a-z'-]{2,}", token):
        return None
    return token


def indexed_records(run: dict) -> dict[str, dict]:
    return {
        record["slug"]: record
        for record in run["prompts"]
        if record.get("shape") == "ASSOCIATION"
    }


def validate(runs: list[dict]) -> tuple[list[str], list[dict[str, dict]]]:
    indexes = [indexed_records(run) for run in runs]
    slugs = list(indexes[0])
    if len(slugs) != 50:
        raise ValueError(f"Expected 50 association prompts, found {len(slugs)}")
    for seed, (run, index) in enumerate(zip(runs, indexes)):
        if run.get("errors"):
            raise ValueError(f"Seed {seed} contains run errors")
        if not run.get("methodology", {}).get("paper_protocol_complete"):
            raise ValueError(f"Seed {seed} is not paper-protocol complete")
        if list(index) != slugs:
            raise ValueError(f"Seed {seed} prompt set/order differs")
        if not all(record.get("valid_for_metrics") for record in index.values()):
            raise ValueError(f"Seed {seed} contains invalid association records")
    return slugs, indexes


def target_terms(record: dict) -> set[str]:
    terms = set()
    for concept in record.get("tracked", []):
        for value in (concept.get("label"), *concept.get("surfaces", [])):
            normalized = normalize_token(value or "")
            if normalized:
                terms.add(normalized)
    return terms


def consensus_candidates(records: list[dict], *, filter_function_words: bool) -> list[dict]:
    by_seed = []
    for record in records:
        candidates = {}
        for row in record.get("surprising", []):
            token = normalize_token(row.get("concept", ""))
            if token is None or (filter_function_words and token in FUNCTION_WORDS):
                continue
            candidates[token] = row
        by_seed.append(candidates)

    shared = set.intersection(*(set(seed_rows) for seed_rows in by_seed))
    rows = []
    for token in shared:
        source = [seed_rows[token] for seed_rows in by_seed]
        rows.append({
            "token": token,
            "consensus_score": float(np.mean([row["score"] for row in source])),
            "worst_best_rank": int(max(row["best_rank"] for row in source)),
            "mean_best_depth": float(np.mean([row["best_depth"] for row in source])),
            "depths_by_seed": [float(row["best_depth"]) for row in source],
            "scores_by_seed": [float(row["score"]) for row in source],
            "seed_support": 3,
        })
    rows.sort(key=lambda row: (-row["consensus_score"], row["worst_best_rank"], row["token"]))
    return rows


def build_analysis(
    runs: list[dict],
    slugs: list[str],
    indexes: list[dict[str, dict]],
    run_paths: list[Path],
) -> dict:
    per_prompt = []
    raw_candidates: dict[str, list[dict]] = {}
    filtered_candidates: dict[str, list[dict]] = {}
    for slug in slugs:
        records = [index[slug] for index in indexes]
        reference = records[0]
        raw = consensus_candidates(records, filter_function_words=False)
        filtered = consensus_candidates(records, filter_function_words=True)
        raw_candidates[slug] = raw
        filtered_candidates[slug] = filtered
        terms = target_terms(reference)
        for row in filtered:
            row["exact_predeclared_overlap"] = row["token"] in terms
        per_prompt.append({
            "slug": slug,
            "family": reference["category"],
            "prompt": reference["prompt_text"],
            "predeclared_terms_annotation_only": sorted(terms),
            "raw_consensus_candidates": raw,
            "filtered_consensus_candidates": filtered,
        })

    n_prompts = len(slugs)
    prompt_df: dict[str, int] = defaultdict(int)
    global_score: dict[str, float] = defaultdict(float)
    for slug in slugs:
        for row in raw_candidates[slug]:
            prompt_df[row["token"]] += 1
            global_score[row["token"]] += row["consensus_score"]
    global_scaffold = sorted(
        ({
            "token": token,
            "prompt_support": prompt_df[token],
            "total_consensus_score": global_score[token],
        } for token in prompt_df),
        key=lambda row: (-row["prompt_support"], -row["total_consensus_score"], row["token"]),
    )

    # Standard unsmoothed inverse-document frequency. Candidate generation and
    # ranking remain independent of the predeclared concepts.
    filtered_df: dict[str, int] = defaultdict(int)
    for slug in slugs:
        for row in filtered_candidates[slug]:
            filtered_df[row["token"]] += 1

    family_slugs: dict[str, list[str]] = defaultdict(list)
    for prompt in per_prompt:
        family_slugs[prompt["family"]].append(prompt["slug"])
    families = {}
    for family, members in sorted(family_slugs.items()):
        family_targets = set().union(*(
            set(next(prompt["predeclared_terms_annotation_only"] for prompt in per_prompt
                     if prompt["slug"] == slug))
            for slug in members
        ))
        values: dict[str, list[dict]] = defaultdict(list)
        for slug in members:
            for row in filtered_candidates[slug]:
                values[row["token"]].append(row)
        candidates = []
        for token, rows in values.items():
            inverse_document_frequency = math.log(n_prompts / filtered_df[token])
            score = sum(row["consensus_score"] for row in rows) * inverse_document_frequency / len(members)
            candidates.append({
                "token": token,
                "family_specificity_score": score,
                "prompt_support": len(rows),
                "prompt_denominator": len(members),
                "global_prompt_frequency": filtered_df[token],
                "inverse_document_frequency": inverse_document_frequency,
                "mean_consensus_score_when_present": float(np.mean([
                    row["consensus_score"] for row in rows
                ])),
                "exact_predeclared_overlap": token in family_targets,
            })
        candidates.sort(key=lambda row: (
            -row["family_specificity_score"],
            -row["prompt_support"],
            row["token"],
        ))
        families[family] = {
            "n_prompts": len(members),
            "predeclared_terms_annotation_only": sorted(family_targets),
            "candidates": candidates[:TOP_FAMILY_CANDIDATES],
        }

    overlap = {}
    for cutoff in (1, 3, 5, 8):
        family_hits = [
            any(row["exact_predeclared_overlap"] for row in result["candidates"][:cutoff])
            for result in families.values()
        ]
        overlap[str(cutoff)] = {
            "families_with_exact_overlap": int(sum(family_hits)),
            "n_families": len(family_hits),
            "fraction": float(np.mean(family_hits)),
        }

    return {
        "analysis_status": "exploratory open-vocabulary discovery on an already-inspected prompt suite",
        "analysis_seed": ANALYSIS_SEED,
        "candidate_source": (
            "per-seed surprising lists: unrestricted full-vocabulary top-1 tokens across all prompt "
            "positions and 38-92% source layers; prompt/output tokens excluded upstream"
        ),
        "consensus_rule": "candidate must occur in all three independently fitted lens lists",
        "ranking_rule": (
            "within-family mean of consensus score times log(50/global prompt frequency); "
            "common English function words removed using the stored target-agnostic list"
        ),
        "important_censoring": (
            "each source run retained only its top 12 surprising candidates, so this is a censored "
            "candidate analysis rather than exhaustive vocabulary discovery"
        ),
        "predeclared_terms_use": "annotation after ranking only; never used to generate or rank candidates",
        "runs": [
            str(path.resolve().relative_to(ROOT))
            if path.resolve().is_relative_to(ROOT) else str(path.resolve())
            for path in run_paths
        ],
        "n_prompts": n_prompts,
        "n_families": len(families),
        "global_scaffold": global_scaffold[:20],
        "families": families,
        "exact_predeclared_overlap_at_k": overlap,
        "per_prompt": per_prompt,
        "function_words": sorted(FUNCTION_WORDS),
    }


def write_candidate_csv(stats: dict) -> None:
    rows = []
    for prompt in stats["per_prompt"]:
        for rank, candidate in enumerate(prompt["filtered_consensus_candidates"], start=1):
            rows.append({
                "slug": prompt["slug"],
                "family": prompt["family"],
                "candidate_rank": rank,
                "candidate": candidate["token"],
                "consensus_score": candidate["consensus_score"],
                "worst_best_rank": candidate["worst_best_rank"],
                "mean_best_depth": candidate["mean_best_depth"],
                "exact_predeclared_overlap_after_ranking": candidate["exact_predeclared_overlap"],
                "prompt": prompt["prompt"],
            })
    with CANDIDATE_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_blinded_sets(stats: dict) -> None:
    families = sorted(stats["families"])
    rng = random.Random(ANALYSIS_SEED)
    rng.shuffle(families)
    blind_rows = []
    key_rows = []
    for index, family in enumerate(families, start=1):
        set_id = f"SET-{index:02d}"
        candidates = stats["families"][family]["candidates"]
        blind_row = {"set_id": set_id}
        for position in range(TOP_FAMILY_CANDIDATES):
            candidate = candidates[position] if position < len(candidates) else None
            blind_row[f"candidate_{position + 1}"] = candidate["token"] if candidate else ""
            blind_row[f"support_{position + 1}"] = (
                f"{candidate['prompt_support']}/5" if candidate else ""
            )
        blind_rows.append(blind_row)
        key_rows.append({"set_id": set_id, "mechanism_family": family})
    with BLIND_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blind_rows[0]))
        writer.writeheader()
        writer.writerows(blind_rows)
    with KEY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)


def plot_results(stats: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(14.5, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.72, 1.9])

    ax = fig.add_subplot(grid[0, 0])
    global_rows = stats["global_scaffold"][:12][::-1]
    labels = [row["token"] for row in global_rows]
    values = [row["prompt_support"] for row in global_rows]
    ax.barh(np.arange(len(labels)), values, color="#287C8E", alpha=0.88)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlim(0, 50)
    ax.set_xlabel("prompts with three-seed consensus (of 50)")
    ax.set_title("A  Shared explanatory scaffold", loc="left", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.18)
    for position, value in enumerate(values):
        ax.text(value + 0.7, position, str(value), va="center", fontsize=9)

    ax = fig.add_subplot(grid[0, 1])
    families = sorted(stats["families"])
    n_rows = len(families)
    n_columns = 5
    scores = np.asarray([
        [
            stats["families"][family]["candidates"][column]["family_specificity_score"]
            if column < len(stats["families"][family]["candidates"]) else 0.0
            for column in range(n_columns)
        ]
        for family in families
    ])
    transformed = np.sqrt(scores)
    image = ax.imshow(transformed, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_columns), [f"candidate {index}" for index in range(1, n_columns + 1)])
    ax.set_yticks(range(n_rows), [family.replace("-", " ") for family in families])
    ax.set_title("B  Background-corrected semantic neighborhoods", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for row_index, family in enumerate(families):
        candidates = stats["families"][family]["candidates"]
        for column in range(n_columns):
            if column >= len(candidates):
                continue
            candidate = candidates[column]
            marker = "*" if candidate["exact_predeclared_overlap"] else ""
            label = f"{candidate['token']}{marker}\n{candidate['prompt_support']}/5 prompts"
            color = "white" if transformed[row_index, column] > 0.58 * transformed.max() else "black"
            ax.text(column, row_index, label, ha="center", va="center", fontsize=8.4, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    colorbar.set_label("sqrt(background-corrected consensus score)")

    fig.suptitle(
        "Open-vocabulary readout reveals both language scaffolding and mechanism-specific assemblies",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5, -0.01,
        "Candidates were generated without a target list and required agreement across all three lens fits.  "
        "* exact predeclared-term overlap, annotated only after ranking.  Exploratory; not a literal chain of thought.",
        ha="center", fontsize=9, color="#6E7781",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURE_PATH.with_suffix(f".{suffix}"), dpi=240,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(stats: dict) -> None:
    overlap = stats["exact_predeclared_overlap_at_k"]
    lines = [
        "# Three-seed open-vocabulary discovery across 50 materials prompts",
        "",
        "## Question",
        "",
        "What words does the Jacobian lens surface consistently when no candidate vocabulary is supplied to the discovery algorithm? This analysis complements, rather than replaces, the predeclared-concept rank analysis.",
        "",
        "## Method in plain language",
        "",
        "For every prompt and every lens-fit seed, the run already stored the unrestricted full-vocabulary token that ranked first across every prompt position and each source layer in the fixed 38--92% depth band. Tokens appearing in the prompt or generated continuation were removed upstream. A candidate was retained here only when it appeared in the stored candidate list for **all three** independently fitted lenses.",
        "",
        "The predeclared materials terms were not used to generate, retain, filter, or rank candidates. After ranking was complete, exact overlaps were marked with an asterisk for comparison. Common English function words were removed only from the family-specific display; the unfiltered cross-prompt result is retained because it reveals a shared language scaffold.",
        "",
        "Family candidates were ranked using a transparent background correction: the three-seed consensus score was multiplied by `log(50 / number of prompts containing the token)` and averaged over the five phrasings. Therefore, a word that appears everywhere is treated as general language scaffolding rather than family-specific evidence.",
        "",
        "![Open-vocabulary consensus](../figures/gemma4-paper-multiseed/open-vocabulary-consensus.png)",
        "",
        "## Main result",
        "",
        "Two structures appear naturally:",
        "",
        "1. A cross-domain explanatory scaffold dominated by words such as `because`, `metallurgical`, `several`, `this`, `whereas`, and `during`. This suggests that the readout often organizes an explanatory continuation before or alongside a specific mechanism word.",
        "2. Mechanism-sensitive semantic neighborhoods. Examples include `corrosion` for boundary attack; `brittle` and `fracture` for cleavage; `fatigue` and `cracks` for cyclic failure; `oxide`/`oxides` for hot-air surface layers; `dislocations`, `atoms`, and `deformation` for line-defect motion; and `microstructure`, `diffusion`, `nucleation`, and the token fragment `martens` for rapid transformation.",
        "",
        "This is closer to what materials scientists mean by seeing what *assembles naturally*: the analysis retains both expected mechanisms and adjacent or unexpected concepts. It also keeps irrelevant and malformed candidates visible rather than silently selecting only attractive examples.",
        "",
        "## Relationship to the predetermined analysis",
        "",
        f"At least one exact predeclared term appeared among the top three background-corrected candidates in {overlap['3']['families_with_exact_overlap']}/{overlap['3']['n_families']} families and among the top five in {overlap['5']['families_with_exact_overlap']}/{overlap['5']['n_families']} families. These are descriptive overlaps, not an independent hypothesis test, because this prompt suite has already been inspected.",
        "",
        "The two analyses answer different scientific questions:",
        "",
        "| analysis | question | strength | limitation |",
        "|---|---|---|---|",
        "| predeclared ranks | Is a specified physical concept readable? | controlled and quantitative | cannot reveal concepts we did not name |",
        "| open vocabulary | What stable words appear without a target list? | reveals neighboring and unexpected structure | requires filtering, background correction, and held-out confirmation |",
        "",
        "## Family-level candidate sets",
        "",
        "Asterisks mark exact predeclared-term overlaps after ranking. Support is the number of alternate phrasings, out of five, in which the candidate survived strict three-seed consensus.",
        "",
        "| mechanism family | top background-corrected candidates |",
        "|---|---|",
    ]
    for family, result in sorted(stats["families"].items()):
        cells = []
        for candidate in result["candidates"]:
            marker = "*" if candidate["exact_predeclared_overlap"] else ""
            cells.append(f"`{candidate['token']}`{marker} ({candidate['prompt_support']}/5)")
        lines.append(f"| {family} | {', '.join(cells)} |")
    lines.extend([
        "",
        "## Required held-out confirmation",
        "",
        "The current result is an exploratory discovery stage. For a paper-quality test, freeze this exact algorithm and apply it to newly written prompts that have not been viewed during method development. Materials experts should receive only the shuffled candidate sets and select the most compatible mechanism family. Accuracy should be compared with a shuffled-family null and with equivalent logit-lens candidate sets.",
        "",
        "The blinded candidate sheet and its separately stored answer key are provided below. They demonstrate the proposed rating format, but ratings on this already-inspected suite remain exploratory.",
        "",
        "## Important limitations",
        "",
        "- Each existing run retained only its top 12 whole-prompt candidates. The present result is therefore censored and not exhaustive.",
        "- A decoded token can be a word fragment (`martens`, `coales`) rather than a complete scientific term.",
        "- Stable language scaffolding can outrank physical vocabulary.",
        "- Seed agreement shows measurement reproducibility, not causal use, reasoning, or human-like understanding.",
        "- The displayed neighborhoods are not a literal chain of thought.",
        "",
        "## Artifacts",
        "",
        "- Machine-readable analysis: [`open-vocabulary-consensus_statistics.json`](open-vocabulary-consensus_statistics.json)",
        "- Every prompt-level consensus candidate: [`open-vocabulary-consensus_candidates.csv`](open-vocabulary-consensus_candidates.csv)",
        "- Blinded family-rating sets: [`OPEN_VOCABULARY_BLINDED_FAMILY_SETS.csv`](OPEN_VOCABULARY_BLINDED_FAMILY_SETS.csv)",
        "- Separate rating key: [`OPEN_VOCABULARY_BLINDED_FAMILY_KEY.csv`](OPEN_VOCABULARY_BLINDED_FAMILY_KEY.csv)",
        "- Vector figure: [`../figures/gemma4-paper-multiseed/open-vocabulary-consensus.pdf`](../figures/gemma4-paper-multiseed/open-vocabulary-consensus.pdf)",
        "- Reproduce: `python scripts/analyze_open_vocabulary_consensus.py`",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs=3,
        type=Path,
        default=DEFAULT_RUN_PATHS,
        metavar=("SEED0_JSON", "SEED1_JSON", "SEED2_JSON"),
        help="three paper-protocol run JSONs (default: existing seed0/1/2 files)",
    )
    args = parser.parse_args()
    run_paths = [path if path.is_absolute() else (Path.cwd() / path) for path in args.runs]
    runs = [json.loads(path.read_text()) for path in run_paths]
    slugs, indexes = validate(runs)
    stats = build_analysis(runs, slugs, indexes, run_paths)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    write_candidate_csv(stats)
    write_blinded_sets(stats)
    plot_results(stats)
    write_report(stats)
    print(f"wrote {STATS_PATH}")
    print(f"wrote {CANDIDATE_CSV}")
    print(f"wrote {BLIND_CSV}")
    print(f"wrote {KEY_CSV}")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {FIGURE_PATH.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
