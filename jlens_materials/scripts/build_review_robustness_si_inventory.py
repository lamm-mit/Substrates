#!/usr/bin/env python3
"""Build SI-ready exact prompt and robustness-result inventories."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUTPUT = PAPER / "review_robustness_prompt_inventory.tex"
GRAPH_MANIFEST = (
    ROOT / "experiments" / "answer-code-binding-2026-07-17"
    / "prompt_manifest.json"
)
HELDOUT_MANIFEST = ROOT / "prompts" / "materials-heldout-v1-preregistered.json"
MULTITOKEN = (
    ROOT / "experiments" / "multitoken-sequence-robustness-2026-07-18"
)


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
        "–": "--",
        "—": "---",
        "×": r"$\times$",
        "≤": r"$\leq$",
        "≥": r"$\geq$",
    }
    return "".join(replacements.get(character, character) for character in text)


def tex_id(value: object) -> str:
    """Escape an identifier and permit line breaks after hyphens."""

    return (
        tex(value)
        .replace("-", r"-\allowbreak{}")
        .replace(r"\_", r"\_\allowbreak{}")
    )


def option_free_rows() -> list[str]:
    manifest = json.loads(GRAPH_MANIFEST.read_text())
    lines = [
        r"\subsection*{Exact natural question-end prompt inventory}",
        (
            "The following are the complete 72 user questions used for the "
            "natural question-end state capture. The runner used only the "
            r"\texttt{stem} field printed here. It added no answer choices, "
            "answer words, arbitrary code, response instruction, or "
            "checkpoint marker. Prompt IDs retain the source manifest's "
            r"\texttt{answer-code} suffix for provenance even though that "
            "suffix content was not used."
        ),
        "",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\LTpre}{0.4em}",
        r"\setlength{\LTpost}{0.4em}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.15\linewidth}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.10\linewidth}>{\raggedright\arraybackslash}p{0.08\linewidth}>{\raggedright\arraybackslash}p{0.46\linewidth}}",
        r"\caption{Complete option-free natural question-end prompt inventory.}\label{tab:s-option-free-prompts}\\",
        r"\toprule",
        r"Prompt ID & Mechanism & Variant & Input / outcome & Exact user question \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Prompt ID & Mechanism & Variant & Input / outcome & Exact user question \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for row in manifest["prompts"]:
        lines.append(
            " & ".join(
                [
                    r"\texttt{" + tex_id(row["prompt_id"]) + "}",
                    tex_id(row["family_id"]),
                    tex_id(row["variant"]),
                    (
                        tex(row["numeric_direction"])
                        + " / "
                        + tex(row["expected_outcome"])
                    ),
                    tex(row["stem"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    return lines


def multitoken_rows() -> list[str]:
    manifest = json.loads(HELDOUT_MANIFEST.read_text())
    prompt_map = {row["slug"]: row for row in manifest["prompts"]}
    prompt_scores = pd.read_csv(
        MULTITOKEN / "prompt_jacobian_direct_contrasts.csv"
    ).sort_values(["family", "slug"])
    stats = json.loads((MULTITOKEN / "statistics.json").read_text())
    tokenization = stats["tokenization"]
    lines = [
        r"\subsection*{Exact multi-token robustness inventory}",
        (
            "All ten originally excluded multi-token cases are printed below. "
            "The target and scientific contrast were fixed before sequence "
            "scoring. The Jacobian and direct columns are target-minus-contrast "
            "restricted sequence log-odds averaged over the registered "
            r"38--92\% depth band."
        ),
        "",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\LTpre}{0.4em}",
        r"\setlength{\LTpost}{0.4em}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.12\linewidth}>{\raggedright\arraybackslash}p{0.34\linewidth}>{\raggedright\arraybackslash}p{0.18\linewidth}rrr}",
        r"\caption{Complete multi-token prompt and sequence-score inventory.}\label{tab:s-multitoken-prompts}\\",
        r"\toprule",
        r"ID & Exact input description & Target / contrast and pieces & Jacobian & Direct & Difference \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"ID & Exact input description & Target / contrast and pieces & Jacobian & Direct & Difference \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for row in prompt_scores.itertuples(index=False):
        prompt = prompt_map[row.slug]
        tokens = tokenization[row.family]
        target = " + ".join(tokens["target_pieces"])
        contrast = " + ".join(tokens["contrast_pieces"])
        pair_text = (
            f"{row.family}: {prompt['tracked'][0] if row.family == 'cleavage' else 'martensite'}"
        )
        if row.family == "cleavage":
            pair_text = (
                f"transgranular ({target}) / intergranular ({contrast})"
            )
        else:
            pair_text = f"martensite ({target}) / bainite ({contrast})"
        lines.append(
            " & ".join(
                [
                    r"\texttt{" + tex_id(row.slug) + "}",
                    tex(prompt["text"]),
                    tex(pair_text),
                    f"{row.jacobian_band_sequence_margin:+.3f}",
                    f"{row.direct_band_sequence_margin:+.3f}",
                    f"{row.jacobian_minus_direct:+.3f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    return lines


def cross_mechanism_rows() -> list[str]:
    table = pd.read_csv(
        ROOT
        / "experiments"
        / "cross-mechanism-outcome-2026-07-18"
        / "mechanism_pair_metrics.csv"
    )
    jacobian = table[
        (table["method"] == "jacobian")
        & (table["opposite_response_orientation"].astype(bool))
    ].sort_values("family_pair")
    lines = [
        r"\subsection*{Complete counter-numeric mechanism-pair results}",
        (
            "These are all opposite-response mechanism pairs in the frozen "
            "cross-mechanism analysis. In this subset, the same physical "
            "outcome requires the opposite numerical direction."
        ),
        "",
        r"\begin{table}[ht!]",
        r"\centering",
        r"\scriptsize",
        r"\caption{All Jacobian counter-numeric mechanism-pair results.}",
        r"\label{tab:s-cross-mechanism-pairs}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Mechanism pair & AUC & Top-1 accuracy & Rankings \\",
        r"\midrule",
    ]
    for row in jacobian.itertuples(index=False):
        lines.append(
            f"{tex(row.family_pair)} & {row.pairwise_auc:.3f} & "
            f"{row.top1_accuracy:.3f} & {int(row.n_rankings)}"
            + r" \\"
        )
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    )
    return lines


def patching_rows() -> list[str]:
    directory = (
        ROOT
        / "experiments"
        / "cross-mechanism-activation-patching-2026-07-18"
    )
    subset = pd.read_csv(directory / "subset_statistics.csv")
    family = pd.read_csv(directory / "donor_family_effects.csv").sort_values(
        "donor_family"
    )
    names = {
        "all": "All mechanism pairs",
        "cross_vocabulary": "Different answer vocabularies",
        "opposite_orientation": "Opposite response orientations",
        "both": "Both controls",
    }
    lines = [
        r"\subsection*{Option-free cross-mechanism activation patching}",
        (
            "The 24 anchor rows in Table~"
            r"\ref{tab:s-option-free-prompts} served as both receivers and "
            "donors. Every receiver was patched with all four anchor donors "
            "from each of the other five mechanisms at layers 16, 24, 32, "
            "and 37, giving 1,920 retained interventions. Donor and receiver "
            "contained no answer list. The complete intervention table is "
            r"\texttt{experiments/cross-mechanism-activation-patching-"
            r"2026-07-18/all\_patch\_rows.csv}."
        ),
        "",
        r"\begin{table}[ht!]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{Frozen option-free activation-patching endpoints. "
            "Intervals resample unordered mechanism pairs. The pair-sign and "
            "structured columns are two-sided exact probabilities.}"
        ),
        r"\label{tab:s-option-free-patching-subsets}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Subset & Transfer & 95\% interval & Pair-sign $p$ & Structured $p$ & Pairs \\",
        r"\midrule",
    ]
    for row in subset.itertuples(index=False):
        lines.append(
            f"{tex(names[row.subset])} & {row.mean:+.3f} & "
            f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}] & "
            f"{row.exact_two_sided_p:.5f} & "
            f"{row.structured_exact_two_sided_p:.5f} & "
            f"{int(row.n_pairs)}"
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[ht!]",
            r"\centering",
            r"\scriptsize",
            (
                r"\caption{Activation-patching results by donor mechanism. "
                "Positive physical transfer means that positive-outcome "
                "donors increased the receiver's positive-minus-negative "
                "answer margin relative to negative-outcome donors.}"
            ),
            r"\label{tab:s-option-free-patching-families}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Donor mechanism & Physical outcome & Numerical direction & Positive ordered pairs \\",
            r"\midrule",
        ]
    )
    for row in family.itertuples(index=False):
        lines.append(
            f"{tex(row.donor_family)} & "
            f"{row.physical_outcome_contrast:+.3f} & "
            f"{row.numeric_direction_contrast:+.3f} & "
            f"{int(row.positive_ordered_pairs)}/{int(row.n_ordered_pairs)}"
            + r" \\"
        )
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    )
    return lines


def main() -> None:
    lines = [
        "% Generated by scripts/build_review_robustness_si_inventory.py.",
        "% Do not edit by hand; regenerate from frozen manifests and CSVs.",
        "",
    ]
    lines.extend(option_free_rows())
    lines.extend(multitoken_rows())
    lines.extend(cross_mechanism_rows())
    patch_directory = (
        ROOT
        / "experiments"
        / "cross-mechanism-activation-patching-2026-07-18"
    )
    if (patch_directory / "statistics.json").exists():
        lines.extend(patching_rows())
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
