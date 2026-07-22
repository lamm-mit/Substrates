#!/usr/bin/env python3
"""Build the complete LaTeX prompt-and-result inventory for the paper SI."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "prompts" / "materials-heldout-v1-preregistered.json"
PROMPT_CSV = ROOT / "experiments" / "materials-heldout-v1_prompt_results.csv"
CONCEPT_CSV = ROOT / "experiments" / "materials-heldout-v1_concept_results.csv"
CANDIDATE_CSV = ROOT / "experiments" / "materials-heldout-v1_open_vocabulary_candidates.csv"
OUTPUT = ROOT / "paper" / "heldout_prompt_inventory.tex"

FAMILY_TITLES = {
    "ductile": "Energy-absorbing dimpled separation",
    "boundary-attack": "Chromium-depleted interfacial attack",
    "cyclic": "Progressive damage under repeated loading",
    "cleavage": "Low-temperature cleavage",
    "high-temperature-deformation": "Slow deformation at elevated temperature",
    "particle-strengthening": "Hard-particle strengthening",
    "rapid-transformation": "Rapid diffusionless transformation",
    "line-defect-motion": "Line-defect motion",
    "notch-resistance": "Resistance to unstable notch growth",
    "hot-air-surface-layer": "High-temperature surface oxidation",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def code(value: object) -> str:
    return r"\texttt{" + tex(value) + "}"


def format_candidates(items: list[dict[str, str]]) -> str:
    top = sorted(items, key=lambda row: int(row["candidate_rank"]))[:5]
    return ", ".join(
        f"{code(row['candidate'])} ({float(row['consensus_score']):.1f})"
        for row in top
    )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    prompt_results = {row["slug"]: row for row in rows(PROMPT_CSV)}
    concept_results: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows(CONCEPT_CSV):
        concept_results[row["slug"]].append(row)
    candidates: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows(CANDIDATE_CSV):
        candidates[(row["slug"], row["method"])].append(row)

    prompts_by_family: dict[str, list[dict]] = defaultdict(list)
    family_order: list[str] = []
    for prompt in manifest["prompts"]:
        family = prompt["target_family"]
        if family not in prompts_by_family:
            family_order.append(family)
        prompts_by_family[family].append(prompt)

    out: list[str] = []
    running = 1
    for family in family_order:
        out.append(r"\section{" + tex(FAMILY_TITLES[family]) + "}")
        out.append("")
        for prompt in prompts_by_family[family]:
            slug = prompt["slug"]
            result = prompt_results[slug]
            concepts = sorted(concept_results[slug], key=lambda row: prompt["tracked"].index(row["concept"]))
            dropped = result["dropped_concepts"] or "none"
            continuation = result["generated_completion"].strip() or "<empty>"
            out.extend(
                [
                    rf"\subsection*{{S{running}. {code(slug)}}}",
                    r"\noindent\textbf{Exact prompt.} \emph{" + tex(prompt["text"]) + "}",
                    "",
                    r"\noindent\textbf{Declared before execution:} "
                    + ", ".join(code(term) for term in prompt["tracked"])
                    + r". \textbf{Tokenizer-dropped:} " + tex(dropped) + ".",
                    "",
                    r"\noindent\textbf{Leakage check and prompt endpoint:} one-token continuation "
                    + code(continuation)
                    + rf"; Jacobian AUC {float(result['j_auc_mean_seed']):.4f}; direct AUC {float(result['logit_auc']):.4f}; difference {float(result['delta_auc_mean_seed']):+.4f}.",
                    "",
                    r"\begin{center}",
                    r"\small",
                    r"\begin{tabular}{lrrrr}",
                    r"\toprule",
                    r"Concept & J seed 0 & J seed 1 & J seed 2 & Direct \\",
                    r"\midrule",
                ]
            )
            for row in concepts:
                out.append(
                    f"{code(row['concept'])} & {int(row['j_rank_seed0']):,} & {int(row['j_rank_seed1']):,} & {int(row['j_rank_seed2']):,} & {int(row['logit_rank']):,} \\\\"
                )
            out.extend(
                [
                    r"\bottomrule",
                    r"\end{tabular}",
                    r"\end{center}",
                    r"\noindent\textbf{Leading target-free Jacobian candidates (consensus score):} "
                    + format_candidates(candidates[(slug, "jacobian")])
                    + ".",
                    "",
                    r"\noindent\textbf{Leading target-free direct candidates (consensus score):} "
                    + format_candidates(candidates[(slug, "logit")])
                    + ".",
                    "",
                ]
            )
            running += 1
    OUTPUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUTPUT} with {running - 1} prompts")


if __name__ == "__main__":
    main()
