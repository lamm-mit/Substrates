#!/usr/bin/env python3
"""Freeze a second fresh-law test of relational physical contrasts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "relational-contrast-confirmation-2026-07-18"
MANIFEST = OUT / "prompt_manifest.json"
PROTOCOL = OUT / "protocol.json"
RUNNER = ROOT / "scripts" / "run_elicited_physics_abstraction.py"
DEVELOPMENT = ROOT / "experiments" / "elicited-physics-abstraction-2026-07-18"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


LAWS = [
    {
        "law_id": "hooke-strain-stress",
        "law_sign": 1,
        "law_name": "uniaxial Hooke response at fixed modulus",
        "response": "axial strain epsilon",
        "control": "axial stress sigma",
        "formula_a": "epsilon = sigma/E, with E fixed and positive",
        "formula_b": "epsilon E = sigma, with E fixed and positive",
        "cases": [
            ("a steel coupon in its elastic regime", "20 MPa", "200 MPa"),
            ("an elastic polymer specimen", "1 MPa", "10 MPa"),
        ],
    },
    {
        "law_id": "torsion-angle-torque",
        "law_sign": 1,
        "law_name": "elastic torsion at fixed geometry and modulus",
        "response": "twist angle theta",
        "control": "applied torque T_q",
        "formula_a": "theta = T_q L/(G J), with L, G, and J fixed",
        "formula_b": "theta/T_q = L/(G J), with L, G, and J fixed",
        "cases": [
            ("a circular steel shaft", "10 N m", "100 N m"),
            ("a composite tube in elastic torsion", "5 N m", "50 N m"),
        ],
    },
    {
        "law_id": "arrhenius-diffusivity-temperature",
        "law_sign": 1,
        "law_name": "Arrhenius diffusivity versus temperature",
        "response": "diffusivity D",
        "control": "absolute temperature T",
        "formula_a": "D = D_0 exp[-Q/(R T)], with Q and R positive",
        "formula_b": "ln(D/D_0) = -Q/(R T), with Q and R positive",
        "cases": [
            ("carbon diffusion in austenite", "700 K", "1100 K"),
            ("vacancy diffusion in a crystal", "500 K", "900 K"),
        ],
    },
    {
        "law_id": "radiative-power-temperature",
        "law_sign": 1,
        "law_name": "Stefan-Boltzmann emission at fixed area and emissivity",
        "response": "radiated power P_rad",
        "control": "absolute temperature T",
        "formula_a": "P_rad = C T^4",
        "formula_b": "P_rad/T^4 = C",
        "cases": [
            ("a ceramic surface at fixed emissivity", "400 K", "800 K"),
            ("an oxidized metal surface at fixed area", "500 K", "1000 K"),
        ],
    },
    {
        "law_id": "capacitance-permittivity",
        "law_sign": 1,
        "law_name": "parallel-plate capacitance at fixed geometry",
        "response": "capacitance C_p",
        "control": "dielectric permittivity epsilon_d",
        "formula_a": "C_p = epsilon_d A/d, with A and d fixed",
        "formula_b": "C_p/epsilon_d = A/d, with A and d fixed",
        "cases": [
            ("a ceramic dielectric capacitor", "2e-11 F m^-1", "2e-10 F m^-1"),
            ("a polymer dielectric stack", "3e-11 F m^-1", "9e-11 F m^-1"),
        ],
    },
    {
        "law_id": "magnetic-induction-field",
        "law_sign": 1,
        "law_name": "linear magnetic induction at fixed permeability",
        "response": "magnetic induction B",
        "control": "magnetic field H",
        "formula_a": "B = mu H, with mu fixed and positive",
        "formula_b": "B/H = mu, with mu fixed and positive",
        "cases": [
            ("a linear magnetic alloy below saturation", "100 A m^-1", "1000 A m^-1"),
            ("a paramagnetic solid in its linear regime", "50 A m^-1", "500 A m^-1"),
        ],
    },
    {
        "law_id": "axial-stiffness-length",
        "law_sign": -1,
        "law_name": "axial spring stiffness versus member length",
        "response": "axial stiffness k_ax",
        "control": "member length L",
        "formula_a": "k_ax = E A/L, with E and A fixed",
        "formula_b": "k_ax L = E A, with E and A fixed",
        "cases": [
            ("a steel tie of fixed cross-section", "0.5 m", "2 m"),
            ("a polymer fiber of fixed cross-section", "10 mm", "100 mm"),
        ],
    },
    {
        "law_id": "torsion-angle-modulus",
        "law_sign": -1,
        "law_name": "elastic twist versus shear modulus",
        "response": "twist angle theta",
        "control": "shear modulus G",
        "formula_a": "theta = C/G",
        "formula_b": "theta G = C",
        "cases": [
            ("a shaft at fixed torque and geometry", "2 GPa", "80 GPa"),
            ("a composite tube at fixed torque and geometry", "5 GPa", "50 GPa"),
        ],
    },
    {
        "law_id": "arrhenius-time-temperature",
        "law_sign": -1,
        "law_name": "Arrhenius process time at fixed transformation extent",
        "response": "required process time t_p",
        "control": "absolute temperature T",
        "formula_a": "t_p = C exp[Q/(R T)], with Q and R positive",
        "formula_b": "ln(t_p/C) = Q/(R T), with Q and R positive",
        "cases": [
            ("a thermally activated annealing step", "600 K", "1000 K"),
            ("a diffusion-limited treatment", "500 K", "900 K"),
        ],
    },
    {
        "law_id": "grain-growth-time-mobility",
        "law_sign": -1,
        "law_name": "grain-growth time at fixed target size",
        "response": "growth time t_g",
        "control": "grain-boundary mobility M",
        "formula_a": "t_g = C/M",
        "formula_b": "t_g M = C",
        "cases": [
            ("a polycrystal growing to a fixed grain size", "1e-15 m^4 J^-1 s^-1", "1e-13 m^4 J^-1 s^-1"),
            ("a ceramic reaching a fixed coarsening extent", "2e-16 m^4 J^-1 s^-1", "2e-14 m^4 J^-1 s^-1"),
        ],
    },
    {
        "law_id": "resistance-area",
        "law_sign": -1,
        "law_name": "electrical resistance versus conductor area",
        "response": "electrical resistance R_e",
        "control": "cross-sectional area A",
        "formula_a": "R_e = rho L/A, with rho and L fixed",
        "formula_b": "R_e A = rho L, with rho and L fixed",
        "cases": [
            ("a metal wire of fixed length", "0.1 mm^2", "1 mm^2"),
            ("a printed conductor of fixed length", "0.02 mm^2", "0.2 mm^2"),
        ],
    },
    {
        "law_id": "penetration-depth-absorption",
        "law_sign": -1,
        "law_name": "optical penetration depth versus absorption coefficient",
        "response": "penetration depth delta_p",
        "control": "absorption coefficient alpha_a",
        "formula_a": "delta_p = 1/alpha_a",
        "formula_b": "delta_p alpha_a = 1",
        "cases": [
            ("an absorbing semiconductor", "1e4 m^-1", "1e6 m^-1"),
            ("a pigmented polymer", "2e3 m^-1", "2e5 m^-1"),
        ],
    },
]


def make_prompt(law: dict, surface: str, case_index: int, numerical_sign: int, answer_order: str) -> dict:
    system, low, high = law["cases"][case_index]
    start, end = (low, high) if numerical_sign == 1 else (high, low)
    physical_sign = int(law["law_sign"]) * numerical_sign
    words = ("higher", "lower") if answer_order == "higher-first" else ("lower", "higher")
    prompt_id = (
        f"{law['law_id']}--{surface}--case{case_index + 1}--"
        f"{'up' if numerical_sign == 1 else 'down'}--{answer_order}"
    )
    user = (
        "Apply the same two-stage physical test to every case. Stage 1: use "
        "only the constitutive equation, not associations with material names, "
        "to determine the monotonic sign of the response with respect to the "
        "changed control while all other quantities are fixed. "
        f"For {law['law_name']}, the response is {law['response']}, the control "
        f"is {law['control']}, and the relation is {law[f'formula_{surface}']}. "
        "All stated coefficients are positive. Complete this relation-sign "
        f"step silently. § Stage 2: in {system}, {law['control']} changes from "
        f"{start} to {end}. Determine the sign of this numerical change "
        "silently. ¶ Now compose the relation sign with the numerical-change "
        f"sign to determine the direction of {law['response']}. † Reply with "
        "exactly one lowercase word from this ordered pair: "
        f"{words[0]}, {words[1]}."
    )
    return {
        "prompt_id": prompt_id,
        "law_id": law["law_id"],
        "law_name": law["law_name"],
        "split": "second-fresh-confirmation",
        "law_sign": int(law["law_sign"]),
        "surface": surface,
        "case_index": case_index,
        "system": system,
        "control": law["control"],
        "response": law["response"],
        "numeric_start": start,
        "numeric_end": end,
        "numerical_sign": numerical_sign,
        "physical_sign": physical_sign,
        "answer_order": answer_order,
        "expected_answer": "higher" if physical_sign == 1 else "lower",
        "user": user,
    }


def main() -> None:
    if (OUT / "raw.json").exists():
        raise RuntimeError("refusing to rebuild after outputs exist")
    prompts = [
        make_prompt(law, surface, case, numerical, order)
        for law in LAWS
        for surface in ("a", "b")
        for case in range(2)
        for numerical in (-1, 1)
        for order in ("higher-first", "lower-first")
    ]
    manifest = {
        "study_id": "relational-contrast-confirmation-2026-07-18",
        "markers": {"law": "§", "comparison": "¶", "decision": "†"},
        "laws": LAWS,
        "prompts": prompts,
        "dimensions": {"n_laws": len(LAWS), "n_prompts": len(prompts)},
    }
    write_json(MANIFEST, manifest)
    protocol = {
        "study_id": manifest["study_id"],
        "status": "frozen-before-second-fresh-output",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "model": "google/gemma-4-E4B-it",
        "model_revision": "a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
        "positions": ["law", "comparison", "decision", "final_prompt"],
        "inputs": {
            "prompt_manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": sha256(MANIFEST)},
            "runner": {"path": str(RUNNER.relative_to(ROOT)), "sha256": sha256(RUNNER)},
            "development_manifest": {"path": str((DEVELOPMENT / "prompt_manifest.json").relative_to(ROOT)), "sha256": sha256(DEVELOPMENT / "prompt_manifest.json")},
            "development_states": {"path": str((DEVELOPMENT / "representations.npz").relative_to(ROOT)), "sha256": sha256(DEVELOPMENT / "representations.npz")},
        },
        "frozen_lens": {
            "position": "final_prompt",
            "layer": 34,
            "state_normalization": "unit L2",
            "direction": "positive-minus-negative physical-outcome centroids over all 16 old development laws",
            "fresh_refit": "none",
        },
        "primary_relational_endpoint": (
            "For every matched surface/case/answer-order cell, subtract the "
            "frozen physical score for numerical-down from numerical-up. "
            "Average contrasts within law. Positive predicts a direct law and "
            "negative predicts an inverse law."
        ),
        "success_rule": {
            "law_orientation_auc_minimum": 0.90,
            "sign_accuracy_minimum": 0.80,
            "both_surface_auc_minimum": 0.80,
            "laws_total": len(LAWS),
        },
        "inference": {
            "exact_label_permutations": 924,
            "bootstrap_seed": 20260722,
            "bootstrap_resamples": 50000,
        },
        "guardrails": [
            "All 12 relations are disjoint from both earlier abstraction cohorts.",
            "No second-fresh label selects a layer, position, direction, or threshold.",
            "Matched subtraction cancels law wording, formula wording, material case, and answer order within each pair.",
            "Numerical direction is identical across direct and inverse laws; only its physical interpretation changes.",
            "Two algebraic surfaces, two cases, and two answer orders are retained per law.",
            "All outputs, including failures, are retained.",
        ],
    }
    write_json(PROTOCOL, protocol)
    (OUT / "PROTOCOL.md").write_text(
        "# Confirmatory relational physical contrast\n\n"
        "The frozen lens assigns a physical-outcome score to each prompt. For "
        "each law, matched prompts reverse only the numerical change. The sign "
        "of the resulting score difference estimates whether the constitutive "
        "law is direct or inverse. No new-law label is used to fit the lens.\n"
    )
    print(f"frozen {len(prompts)} prompts over {len(LAWS)} second-fresh laws")
    print(f"protocol sha256: {sha256(PROTOCOL)}")


if __name__ == "__main__":
    main()
