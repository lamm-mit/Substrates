#!/usr/bin/env python3
"""Build the paper's complete 50-prompt association inventory from the frozen manifest."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "prompts" / "materials-paper-v2-preregistered.json"
OUTPUT = ROOT / "paper" / "primary_prompt_inventory.tex"

FAMILY_LABELS = {
    "ductile": ("D", "Ductile dimpled failure"),
    "boundary-attack": ("BA", "Boundary-localized stainless attack"),
    "cyclic": ("CY", "Cyclic loading"),
    "cleavage": ("CL", "Low-temperature cleavage"),
    "high-temperature-deformation": ("HT", "High-temperature time-dependent deformation"),
    "particle-strengthening": ("PS", "Hard-particle strengthening"),
    "rapid-transformation": ("RT", "Rapid diffusionless transformation"),
    "line-defect-motion": ("LD", "Line-defect motion"),
    "notch-resistance": ("NR", "Notch resistance"),
    "hot-air-surface-layer": ("HA", "High-temperature surface reaction"),
}

TEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def tex_escape(value: str) -> str:
    return "".join(TEX_ESCAPES.get(character, character) for character in value)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    prompts = [prompt for prompt in manifest["prompts"] if prompt.get("shape") == "ASSOCIATION"]
    counts = Counter(prompt["category"] for prompt in prompts)
    if len(prompts) != 50 or set(counts) != set(FAMILY_LABELS) or any(count != 5 for count in counts.values()):
        raise ValueError(f"Expected ten five-item families and 50 prompts; found {dict(counts)}")

    lines = [
        "% Generated from prompts/materials-paper-v2-preregistered.json.",
        "% Regenerate with: python scripts/build_primary_prompt_appendix.py",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.055\textwidth}>{\raggedright\arraybackslash}p{0.67\textwidth}>{\raggedright\arraybackslash}p{0.205\textwidth}}",
        r"\caption{Complete primary association dataset: all 50 exact prompts and their predeclared candidate concepts. Each family contains five independent phrasings of the same physical mechanism. Asterisks mark declared multi-token labels retained in the manifest but dropped from direct single-token ranking.}\label{tab:all-primary-prompts}\\",
        r"\toprule",
        r"ID & Exact model input & Concepts searched for in the lens readout \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{3}{l}{\small\itshape Table \thetable\ continued from the previous page}\\",
        r"\toprule",
        r"ID & Exact model input & Concepts searched for in the lens readout \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{3}{r}{\small\itshape Continued on the next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    grouped = {family: [] for family in FAMILY_LABELS}
    for prompt in prompts:
        grouped[prompt["category"]].append(prompt)

    for family, (prefix, label) in FAMILY_LABELS.items():
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{tex_escape(label)}}} (five independent phrasings)}}\\*")
        for index, prompt in enumerate(grouped[family], start=1):
            concepts = []
            for concept in prompt["tracked"]:
                marker = r"$^{*}$" if concept in {"martensite", "transgranular"} else ""
                concepts.append(rf"\texttt{{{tex_escape(concept)}}}{marker}")
            lines.append(
                rf"{prefix}{index} & {tex_escape(prompt['text'])} & {', '.join(concepts)} \\"
            )
        lines.append(r"\addlinespace[0.45em]")

    lines.extend([r"\end{longtable}", ""])
    OUTPUT.write_text("\n".join(lines))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
