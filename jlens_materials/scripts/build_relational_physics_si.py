#!/usr/bin/env python3
"""Build the portable SI inventory for the relational-physics benchmark."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "neutral-anchored-relational-physics-2026-07-18"
PAPER = ROOT / "paper"
DATA = PAPER / "relational_physics_data"
INVENTORY = PAPER / "relational_physics_inventory.tex"

DATA_FILES = [
    "PROTOCOL.md",
    "REPORT.md",
    "protocol.json",
    "prompt_manifest.json",
    "raw.json",
    "representations.npz",
    "statistics.json",
    "law_level_neutral_normalized.csv",
    "matched_pair_contrasts.csv",
    "output_head_law_level.csv",
    "word_tfidf_law_level.csv",
    "character_tfidf_law_level.csv",
]

FIGURE_FILES = [
    "neutral-anchored-relational-physics.pdf",
    "neutral-anchored-relational-physics.png",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def escape(value: object) -> str:
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
    return "".join(replacements.get(character, character) for character in text)


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def exact_prompt_examples(manifest: dict) -> list[dict]:
    wanted = [
        "current-voltage--a--case1--up--higher-first",
        "resistivity-wire-length-neutral--a--case1--up--higher-first",
        "capacitance-separation--b--case2--up--unchanged-first",
    ]
    lookup = {item["prompt_id"]: item for item in manifest["prompts"]}
    return [lookup[prompt_id] for prompt_id in wanted]


def build_inventory(manifest: dict, statistics: dict) -> str:
    laws = pd.read_csv(SOURCE / "law_level_neutral_normalized.csv")
    pairs = pd.read_csv(SOURCE / "matched_pair_contrasts.csv")
    law_meta = {item["law_id"]: item for item in manifest["laws"]}
    lines: list[str] = []
    add = lines.append

    add(r"\subsection*{S4E.1 Exact design, chronology, and prompt template}")
    add("")
    add(
        "The task was deliberately scaffolded, but the model emitted no "
        "intermediate rationale. The identical instruction asked it to infer "
        "the monotonic sign of the stated equation, infer whether the numerical "
        "control rose or fell, silently compose those signs, and output one "
        "allowed word. The layer-34 direction and state position were selected "
        "only on the 16-law development set. No state from this 60-law cohort "
        "was used to fit or rotate that direction. The large-cohort protocol, "
        "law list, prompt expansion, endpoints, success gates, bootstrap seed, "
        "and permutation seed were fingerprinted before any corresponding "
        "model output."
    )
    add("")
    add(
        "The exact prompt generator crossed every law with two algebraically "
        "equivalent equation surfaces, two material cases, numerical increase "
        "and decrease, and two answer orders. The following is the literal "
        "template; braces identify fields replaced by the frozen manifest."
    )
    add("")
    add(r"\begin{quote}\small")
    add(
        r"Apply the same two-stage physical test to every case. Stage 1: use "
        r"only the constitutive equation, not associations with material names, "
        r"to determine whether the response increases, decreases, or remains "
        r"unchanged as the control increases while all other quantities are "
        r"fixed. For \texttt{\{law name\}}, the response is "
        r"\texttt{\{response\}}, the control is \texttt{\{control\}}, and the "
        r"relation is \texttt{\{equation surface\}}. All stated coefficients "
        r"are positive. Complete this relation step silently. In "
        r"\texttt{\{material case\}}, \texttt{\{control\}} changes from "
        r"\texttt{\{start\}} to \texttt{\{end\}}. Determine the numerical-change "
        r"direction silently. Now compose the relation with the numerical "
        r"change to determine the direction of \texttt{\{response\}}. Reply "
        r"with exactly one lowercase word from this ordered set: "
        r"\texttt{\{answer order\}}."
    )
    add(r"\end{quote}")
    add("")
    add(
        "The SI bundle includes "
        r"\path{relational_physics_data/prompt_manifest.json}, a 960-entry "
        "ledger containing every fully expanded prompt exactly as sent to "
        "Gemma, not merely the template. It also retains all raw output rows "
        "and the captured 960-by-2,560 layer-34 state array."
    )
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(
        r"\caption{\textbf{Frozen neutral-anchored relational design.} "
        r"The ten calibration-neutral laws alone define the empirical center "
        r"and scale; the ten validation-neutral laws are untouched test cases.}"
    )
    add(r"\begin{tabular}{lr}")
    add(r"\toprule")
    add(r"Quantity & Frozen value \\")
    add(r"\midrule")
    dims = manifest["dimensions"]
    add(rf"Direct / inverse / neutral laws & {dims['n_direct']} / {dims['n_inverse']} / {dims['n_neutral']} \\")
    add(rf"Calibration / validation neutral laws & {dims['n_neutral_calibration']} / {dims['n_neutral_validation']} \\")
    add(rf"Exact prompts / matched comparisons & {dims['n_prompts']} / {dims['n_matched_pairs']} \\")
    add(r"Equation surfaces / cases / answer orders & 2 / 2 / 2 \\")
    add(r"Frozen state & final prompt token, layer 34 \\")
    add(r"Direction fitting & centroid difference on 16 earlier laws; no cohort refit \\")
    add(r"Neutral center & median of 10 calibration-neutral law contrasts \\")
    add(r"Robust scale & $1.4826\times$ median absolute deviation \\")
    add(r"Bootstrap / ordinal permutations & 50,000 / 100,000 \\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\subsection*{S4E.2 Three exact prompt examples}")
    add("")
    for index, prompt in enumerate(exact_prompt_examples(manifest), start=1):
        add(rf"\paragraph{{Example {index}: {escape(prompt['category'])} law.}}")
        add(r"\begin{quote}\small")
        add(escape(prompt["user"]))
        add(r"\end{quote}")
        add(
            rf"\noindent Prompt ID: \path{{{escape(prompt['prompt_id'])}}}; "
            rf"expected response: \texttt{{{escape(prompt['expected_answer'])}}}."
        )
        add("")

    add(r"\clearpage")
    add(r"\subsection*{S4E.3 Complete 60-law inventory}")
    add("")
    add(
        "Every row below contributes 16 exact prompts and eight matched "
        "numerical-reversal comparisons. Surface A states the relation explicitly; "
        "surface B is algebraically rearranged. The two cases supply the "
        "material wording and numerical endpoints. Calibration and validation "
        "refer only to the neutral-law role."
    )
    add("")
    add(r"\begingroup\tiny")
    add(r"\setlength{\tabcolsep}{2pt}")
    add(
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.10\linewidth}"
        r">{\raggedright\arraybackslash}p{0.065\linewidth}"
        r">{\raggedright\arraybackslash}p{0.085\linewidth}"
        r">{\raggedright\arraybackslash}p{0.14\linewidth}"
        r">{\raggedright\arraybackslash}p{0.17\linewidth}"
        r">{\raggedright\arraybackslash}p{0.17\linewidth}"
        r">{\raggedright\arraybackslash}p{0.12\linewidth}}"
    )
    add(r"\toprule")
    add(r"Law ID & Class & Domain & Response $\leftarrow$ control & Surface A & Surface B & Cases / neutral role \\")
    add(r"\midrule")
    add(r"\endhead")
    for law_id in laws.sort_values(["category", "domain", "law_id"])["law_id"]:
        item = law_meta[law_id]
        cases = "; ".join(
            f"{case[0]}: {case[1]} / {case[2]}" for case in item["cases"]
        )
        role = item.get("neutral_role") or "--"
        add(
            f"{escape(law_id)} & {escape(item['category'])} & "
            f"{escape(item['domain'])} & "
            f"{escape(item['response'])} $\\leftarrow$ {escape(item['control'])} & "
            f"{escape(item['formula_a'])} & {escape(item['formula_b'])} & "
            f"{escape(cases)}; role: {escape(role)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{longtable}")
    add(r"\endgroup")
    add("")

    add(r"\clearpage")
    add(r"\subsection*{S4E.4 Complete law-level results}")
    add("")
    add(
        "The score is the mean of eight matched up-minus-down hidden-state "
        "contrasts, centered on the calibration-neutral median and divided by "
        "the calibration-neutral robust scale. A positive value is direct-like; "
        "a negative value is inverse-like. Neutral-law values are not expected "
        "to equal zero individually."
    )
    add("")
    add(r"\begingroup\scriptsize")
    add(
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.27\linewidth}"
        r">{\raggedright\arraybackslash}p{0.10\linewidth}"
        r">{\raggedright\arraybackslash}p{0.18\linewidth}rrrr}"
    )
    add(r"\toprule")
    add(r"Law ID & Class & Domain & Mean $\Delta r$ & Surface A & Surface B & Robust $z$ \\")
    add(r"\midrule")
    add(r"\endhead")
    for _, row in laws.sort_values("neutral_z").iterrows():
        add(
            f"{escape(row['law_id'])} & {escape(row['category'])} & "
            f"{escape(row['domain'])} & {fmt(row['mean_contrast'], 5)} & "
            f"{fmt(row['surface_a_contrast'], 5)} & "
            f"{fmt(row['surface_b_contrast'], 5)} & "
            f"{fmt(row['neutral_z'], 3)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{longtable}")
    add(r"\endgroup")
    add("")

    add(r"\clearpage")
    add(r"\subsection*{S4E.5 Complete 480-comparison result inventory}")
    add("")
    add(
        "Each row is one matched comparison: the two prompts have the "
        "same law, equation surface, material case, and answer order, and differ "
        "only in whether the numerical control decreases or increases. "
        r"\(\Delta r=r(h_{\mathrm{up}})-r(h_{\mathrm{down}})\)."
    )
    add("")
    add(r"\begingroup\tiny")
    add(
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.34\linewidth}"
        r">{\raggedright\arraybackslash}p{0.09\linewidth}"
        r">{\raggedright\arraybackslash}p{0.08\linewidth}"
        r">{\raggedright\arraybackslash}p{0.16\linewidth}rr}"
    )
    add(r"\toprule")
    add(r"Law ID & Surface & Case & Answer order & $\Delta r$ & Robust $z$ \\")
    add(r"\midrule")
    add(r"\endhead")
    for _, row in pairs.sort_values(
        ["law_id", "surface", "case_index", "answer_order"]
    ).iterrows():
        add(
            f"{escape(row['law_id'])} & {escape(row['surface'])} & "
            f"{int(row['case_index']) + 1} & {escape(row['answer_order'])} & "
            f"{fmt(row['contrast'], 5)} & {fmt(row['neutral_z'], 3)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{longtable}")
    add(r"\endgroup")
    add("")

    add(r"\subsection*{S4E.6 Frozen endpoints and artifact integrity}")
    add("")
    primary = statistics["primary"]
    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{\textbf{Complete frozen endpoint verdict.}}")
    add(r"\begin{tabular}{lr}")
    add(r"\toprule")
    add(r"Endpoint & Result \\")
    add(r"\midrule")
    add(rf"Direct versus inverse AUC & {primary['direct_vs_inverse_auc']:.3f} \\")
    add(rf"Direct versus validation-neutral AUC & {primary['direct_vs_validation_neutral_auc']:.3f} \\")
    add(rf"Validation-neutral versus inverse AUC & {primary['validation_neutral_vs_inverse_auc']:.3f} \\")
    add(rf"Neutral-cut directional-law accuracy & {primary['correct_directional_laws']}/{primary['directional_laws_total']} ({100*primary['calibrated_direct_inverse_accuracy']:.1f}\\%) \\")
    add(rf"Explicit / rearranged direct--inverse AUC & {primary['surface_direct_inverse_auc']['a']:.3f} / {primary['surface_direct_inverse_auc']['b']:.3f} \\")
    add(rf"Ordinal Spearman $\rho$ / permutation $p$ & {primary['ordinal_spearman']:.3f} / {primary['ordinal_permutation_p']:.6f} \\")
    add(rf"Direct / neutral / inverse median robust $z$ & {primary['category_median_z']['direct']:.3f} / {primary['category_median_z']['neutral']:.3f} / {primary['category_median_z']['inverse']:.3f} \\")
    add(rf"All frozen criteria & {'PASS' if statistics['passed_all_frozen_criteria'] else 'FAIL'} \\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")
    add(r"\begin{table}[H]")
    add(r"\centering\tiny")
    add(
        r"\caption{\textbf{Portable relational-physics SI files.} "
        r"Hashes refer to the publication copies bundled with this "
        r"Supplementary Information.}"
    )
    add(r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.40\linewidth}>{\raggedright\arraybackslash}p{0.52\linewidth}}")
    add(r"\toprule")
    add(r"File & SHA-256 \\")
    add(r"\midrule")
    for name in DATA_FILES:
        copied = DATA / name
        add(f"{escape(name)} & \\texttt{{{sha256(copied)}}} \\\\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")

    return "\n".join(lines) + "\n"


def write_readme() -> None:
    rows = []
    for name in DATA_FILES:
        path = DATA / name
        rows.append(f"- `{name}` — SHA-256 `{sha256(path)}`")
    text = f"""# Neutral-anchored relational physics SI data

This directory is the portable data companion for SI Section S4E. It retains
all 960 exact prompts, all raw model-output rows, the complete 960-by-2,560
layer-34 state array, all 480 matched reversal comparisons, all 60 law aggregates,
the frozen protocol, and the final statistics.

The prompts are not reconstructed from prose: `prompt_manifest.json` contains
the exact full user string for every prompt. `raw.json` records the clean output
diagnostics in the same prompt order, and `representations.npz` contains
`prompt_ids`, `layer`, and `raw_states`.

Regenerate the analysis and publication figures from the repository root:

```bash
python scripts/analyze_neutral_anchored_relational_benchmark.py
python scripts/build_relational_physics_si.py
```

The first command refuses silently changed frozen inputs by checking the hashes
recorded in `protocol.json`. Rerunning Gemma itself requires the original
experiment directory and:

```bash
conda activate substrates-jlens
python scripts/run_neutral_anchored_relational_benchmark.py
```

## Integrity manifest

{chr(10).join(rows)}
"""
    (DATA / "README.md").write_text(text)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        shutil.copy2(SOURCE / name, DATA / name)
    for name in FIGURE_FILES:
        shutil.copy2(SOURCE / "figures" / name, PAPER / "figures" / name)
    manifest = json.loads((SOURCE / "prompt_manifest.json").read_text())
    statistics = json.loads((SOURCE / "statistics.json").read_text())
    write_readme()
    INVENTORY.write_text(build_inventory(manifest, statistics))
    print(f"wrote {INVENTORY}")
    print(f"copied {len(DATA_FILES)} data files to {DATA}")


if __name__ == "__main__":
    main()
