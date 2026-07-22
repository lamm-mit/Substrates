#!/usr/bin/env python3
"""Freeze a disjoint-law confirmation of the scaffolded physics direction.

No law below occurs in the development benchmark.  The primary layer,
position, decoder, and success rule are fixed from the completed development
run before this manifest is executed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "fresh-law-abstraction-confirmation-2026-07-18"
MANIFEST = OUT / "prompt_manifest.json"
PROTOCOL = OUT / "protocol.json"
RUNNER = ROOT / "scripts" / "run_elicited_physics_abstraction.py"
DEVELOPMENT = (
    ROOT
    / "experiments"
    / "elicited-physics-abstraction-2026-07-18"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


LAWS = [
    {
        "law_id": "linear-thermal-expansion",
        "law_sign": 1,
        "law_name": "linear thermal expansion",
        "response": "specimen length L",
        "control": "temperature T",
        "formula_a": "L = L_ref [1 + alpha (T - T_ref)], with alpha > 0",
        "formula_b": "(L - L_ref)/(T - T_ref) = alpha L_ref, with alpha > 0",
        "cases": [
            ("an aluminum coupon above its reference temperature", "300 K", "500 K"),
            ("a steel gauge above its reference temperature", "350 K", "650 K"),
        ],
    },
    {
        "law_id": "hertz-contact-radius",
        "law_sign": 1,
        "law_name": "Hertzian contact-radius scaling",
        "response": "contact radius a",
        "control": "normal load F",
        "formula_a": "a = (C F)^(1/3)",
        "formula_b": "a^3/F = C",
        "cases": [
            ("an elastic ceramic sphere on a flat", "10 N", "80 N"),
            ("a steel ball on an elastic half-space", "20 N", "160 N"),
        ],
    },
    {
        "law_id": "buckling-moment",
        "law_sign": 1,
        "law_name": "Euler buckling at fixed length",
        "response": "critical buckling load P_cr",
        "control": "second moment of area I",
        "formula_a": "P_cr = C I",
        "formula_b": "P_cr/I = C",
        "cases": [
            ("a fixed-length steel column", "2e-8 m^4", "8e-8 m^4"),
            ("a fixed-length composite strut", "1e-8 m^4", "9e-8 m^4"),
        ],
    },
    {
        "law_id": "elastic-wave-modulus",
        "law_sign": 1,
        "law_name": "longitudinal elastic-wave speed at fixed density",
        "response": "wave speed c",
        "control": "elastic modulus E",
        "formula_a": "c = sqrt(E/rho), with rho fixed",
        "formula_b": "c^2/E = 1/rho, with rho fixed",
        "cases": [
            ("a family of equal-density cellular solids", "5 GPa", "45 GPa"),
            ("a fixed-density model composite", "10 GPa", "90 GPa"),
        ],
    },
    {
        "law_id": "carrier-conductivity",
        "law_sign": 1,
        "law_name": "Drude conductivity at fixed mobility",
        "response": "electrical conductivity sigma_e",
        "control": "carrier density n",
        "formula_a": "sigma_e = q mu n, with q and mu fixed and positive",
        "formula_b": "sigma_e/n = q mu, with q and mu fixed and positive",
        "cases": [
            ("a doped semiconductor at fixed mobility", "1e21 m^-3", "1e23 m^-3"),
            ("a conducting oxide at fixed mobility", "2e20 m^-3", "2e22 m^-3"),
        ],
    },
    {
        "law_id": "thermal-diffusivity-conductivity",
        "law_sign": 1,
        "law_name": "thermal diffusivity at fixed heat capacity and density",
        "response": "thermal diffusivity alpha_th",
        "control": "thermal conductivity k",
        "formula_a": "alpha_th = k/(rho c_p), with rho and c_p fixed",
        "formula_b": "alpha_th/k = 1/(rho c_p), with rho and c_p fixed",
        "cases": [
            ("a fixed-density ceramic family", "2 W m^-1 K^-1", "18 W m^-1 K^-1"),
            ("a fixed-density polymer composite", "0.2 W m^-1 K^-1", "1.8 W m^-1 K^-1"),
        ],
    },
    {
        "law_id": "energy-release-stress",
        "law_sign": 1,
        "law_name": "elastic energy-release rate at fixed crack length",
        "response": "energy-release rate G",
        "control": "remote stress sigma",
        "formula_a": "G = C sigma^2",
        "formula_b": "sqrt(G)/sigma = sqrt(C)",
        "cases": [
            ("a cracked glass plate at fixed geometry", "10 MPa", "40 MPa"),
            ("a cracked polymer sheet at fixed geometry", "5 MPa", "25 MPa"),
        ],
    },
    {
        "law_id": "archard-sliding-distance",
        "law_sign": 1,
        "law_name": "Archard wear at fixed load and hardness",
        "response": "wear volume V",
        "control": "sliding distance s",
        "formula_a": "V = K W s/H, with K, W, and H fixed and positive",
        "formula_b": "V/s = K W/H, with K, W, and H fixed and positive",
        "cases": [
            ("a pin-on-disk steel contact", "10 m", "100 m"),
            ("a ceramic sliding contact", "20 m", "200 m"),
        ],
    },
    {
        "law_id": "thermal-resistance-conductivity",
        "law_sign": -1,
        "law_name": "steady one-dimensional thermal resistance",
        "response": "thermal resistance R_th",
        "control": "thermal conductivity k",
        "formula_a": "R_th = L/(k A), with L and A fixed",
        "formula_b": "R_th k = L/A, with L and A fixed",
        "cases": [
            ("a fixed-thickness insulation panel", "0.02 W m^-1 K^-1", "0.2 W m^-1 K^-1"),
            ("a fixed ceramic layer", "1 W m^-1 K^-1", "10 W m^-1 K^-1"),
        ],
    },
    {
        "law_id": "buckling-length",
        "law_sign": -1,
        "law_name": "Euler buckling versus member length",
        "response": "critical buckling load P_cr",
        "control": "column length L",
        "formula_a": "P_cr = C/L^2",
        "formula_b": "P_cr L^2 = C",
        "cases": [
            ("a steel column of fixed cross-section", "0.5 m", "2 m"),
            ("a composite strut of fixed cross-section", "0.4 m", "1.6 m"),
        ],
    },
    {
        "law_id": "stokes-einstein-radius",
        "law_sign": -1,
        "law_name": "Stokes-Einstein particle diffusion",
        "response": "particle diffusivity D_p",
        "control": "particle radius r",
        "formula_a": "D_p = C/r",
        "formula_b": "D_p r = C",
        "cases": [
            ("spherical nanoparticles in a fixed solvent", "2 nm", "20 nm"),
            ("colloidal spheres at fixed temperature and viscosity", "50 nm", "500 nm"),
        ],
    },
    {
        "law_id": "resistivity-conductivity",
        "law_sign": -1,
        "law_name": "electrical resistivity-conductivity reciprocity",
        "response": "electrical resistivity rho_e",
        "control": "electrical conductivity sigma_e",
        "formula_a": "rho_e = 1/sigma_e",
        "formula_b": "rho_e sigma_e = 1",
        "cases": [
            ("an isotropic conductor", "1e5 S m^-1", "1e7 S m^-1"),
            ("a conducting composite", "1e2 S m^-1", "1e4 S m^-1"),
        ],
    },
    {
        "law_id": "membrane-flux-thickness",
        "law_sign": -1,
        "law_name": "steady diffusion through a membrane",
        "response": "diffusive flux magnitude J",
        "control": "membrane thickness L",
        "formula_a": "J = D Delta_c/L, with D and Delta_c fixed and positive",
        "formula_b": "J L = D Delta_c, with D and Delta_c fixed and positive",
        "cases": [
            ("hydrogen transport through a metal membrane", "0.1 mm", "1 mm"),
            ("solute transport through a polymer film", "10 micrometers", "100 micrometers"),
        ],
    },
    {
        "law_id": "acoustic-density",
        "law_sign": -1,
        "law_name": "acoustic speed at fixed bulk modulus",
        "response": "acoustic wave speed c",
        "control": "mass density rho",
        "formula_a": "c = sqrt(K/rho), with K fixed",
        "formula_b": "c^2 rho = K, with K fixed",
        "cases": [
            ("a model medium at fixed bulk modulus", "500 kg m^-3", "2000 kg m^-3"),
            ("a porous-medium series at fixed effective bulk modulus", "300 kg m^-3", "1200 kg m^-3"),
        ],
    },
    {
        "law_id": "beam-deflection-moment",
        "law_sign": -1,
        "law_name": "elastic beam deflection versus section moment",
        "response": "tip deflection delta",
        "control": "second moment of area I",
        "formula_a": "delta = C/I",
        "formula_b": "delta I = C",
        "cases": [
            ("a cantilever at fixed load, length, and modulus", "1e-9 m^4", "1e-7 m^4"),
            ("a fixed-span beam at fixed load and modulus", "2e-8 m^4", "2e-6 m^4"),
        ],
    },
    {
        "law_id": "indentation-diagonal",
        "law_sign": -1,
        "law_name": "Vickers hardness relation at fixed load",
        "response": "measured hardness H_V",
        "control": "indent diagonal d",
        "formula_a": "H_V = C/d^2",
        "formula_b": "H_V d^2 = C",
        "cases": [
            ("a Vickers indentation at fixed applied load", "20 micrometers", "80 micrometers"),
            ("a microhardness test at fixed applied load", "10 micrometers", "50 micrometers"),
        ],
    },
]


def make_prompt(
    law: dict,
    surface: str,
    case_index: int,
    numerical_sign: int,
    answer_order: str,
) -> dict:
    system, low, high = law["cases"][case_index]
    start, end = (low, high) if numerical_sign == 1 else (high, low)
    physical_sign = int(law["law_sign"]) * numerical_sign
    expected = "higher" if physical_sign == 1 else "lower"
    ordered_words = (
        ("higher", "lower")
        if answer_order == "higher-first"
        else ("lower", "higher")
    )
    prompt_id = (
        f"{law['law_id']}--{surface}--case{case_index + 1}--"
        f"{'up' if numerical_sign == 1 else 'down'}--{answer_order}"
    )
    user = (
        "Apply the same two-stage physical test to every case. "
        "Stage 1: use only the constitutive equation, not associations with "
        "material names, to determine the monotonic sign of the response with "
        "respect to the changed control while all other quantities are fixed. "
        f"For {law['law_name']}, the response is {law['response']}, the "
        f"control is {law['control']}, and the relation is "
        f"{law[f'formula_{surface}']}. All stated coefficients are positive. "
        "Complete this relation-sign step silently. § "
        f"Stage 2: in {system}, {law['control']} changes from {start} to {end}. "
        "Determine the sign of this numerical change silently. ¶ "
        "Now compose the relation sign with the numerical-change sign to "
        f"determine the direction of {law['response']}. † "
        "Reply with exactly one lowercase word from this ordered pair: "
        f"{ordered_words[0]}, {ordered_words[1]}."
    )
    return {
        "prompt_id": prompt_id,
        "law_id": law["law_id"],
        "law_name": law["law_name"],
        "split": "fresh-confirmation",
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
        "presented_words": list(ordered_words),
        "expected_answer": expected,
        "user": user,
    }


def main() -> None:
    if (OUT / "raw.json").exists():
        raise RuntimeError("refusing to rebuild after fresh outputs exist")
    prompts = [
        make_prompt(law, surface, case_index, numerical_sign, answer_order)
        for law in LAWS
        for surface in ("a", "b")
        for case_index in range(2)
        for numerical_sign in (-1, 1)
        for answer_order in ("higher-first", "lower-first")
    ]
    manifest = {
        "study_id": "fresh-law-abstraction-confirmation-2026-07-18",
        "status": "fresh-law-confirmation",
        "markers": {"law": "§", "comparison": "¶", "decision": "†"},
        "laws": LAWS,
        "prompts": prompts,
        "dimensions": {
            "n_fresh_laws": len(LAWS),
            "n_direct_laws": sum(x["law_sign"] == 1 for x in LAWS),
            "n_inverse_laws": sum(x["law_sign"] == -1 for x in LAWS),
            "n_prompts": len(prompts),
        },
    }
    write_json(MANIFEST, manifest)
    protocol = {
        "study_id": manifest["study_id"],
        "status": "frozen-before-any-fresh-model-output",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "model": "google/gemma-4-E4B-it",
        "model_revision": "a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
        "freshness": (
            "All 16 constitutive relations are absent from the preceding "
            "abstract-physics composition and elicitation cohorts. No fresh "
            "model output was inspected before this protocol was written."
        ),
        "inputs": {
            "prompt_manifest": {
                "path": str(MANIFEST.relative_to(ROOT)),
                "sha256": sha256(MANIFEST),
            },
            "runner": {
                "path": str(RUNNER.relative_to(ROOT)),
                "sha256": sha256(RUNNER),
            },
            "development_manifest": {
                "path": str(
                    (DEVELOPMENT / "prompt_manifest.json").relative_to(ROOT)
                ),
                "sha256": sha256(DEVELOPMENT / "prompt_manifest.json"),
            },
            "development_states": {
                "path": str((DEVELOPMENT / "representations.npz").relative_to(ROOT)),
                "sha256": sha256(DEVELOPMENT / "representations.npz"),
            },
        },
        "positions": ["law", "comparison", "decision", "final_prompt"],
        "primary_decoder": {
            "position": "final_prompt",
            "layer": 34,
            "normalization": "unit L2 norm per state",
            "fit_data": "all 16 completed development laws",
            "fit_method": (
                "parameter-free physical-sign centroid difference; midpoint is "
                "the mean of the positive and negative centroids"
            ),
            "fresh_refit": "none",
        },
        "primary_endpoint": (
            "Mean of 16 fresh-law ROC AUC values for physical outcome. Each "
            "law AUC pools two algebraic surfaces, two material cases, two "
            "numerical directions, and two answer-word orders."
        ),
        "success_rule": {
            "mean_fresh_law_auc_minimum": 0.75,
            "bootstrap_95_lower_above": 0.50,
            "positive_laws_minimum": 13,
            "laws_total": 16,
            "behavior_accuracy_minimum": 0.80,
            "both_surface_mean_auc_minimum": 0.70,
            "both_answer_order_mean_auc_minimum": 0.70,
        },
        "secondary_frozen_endpoints": [
            "Per-law AUC separately for each algebraic surface.",
            "Per-law AUC separately for each answer-word order.",
            "Direct-law and inverse-law means.",
            "Behavioral accuracy by law, surface, and answer order.",
            "Word- and character-TF-IDF controls trained only on development text.",
            "Numerical-direction AUC of the frozen physical direction.",
        ],
        "inference": {
            "bootstrap_seed": 20260721,
            "bootstrap_resamples": 50000,
            "exact_test": "two-sided sign-flip test of per-law AUC minus 0.5",
        },
        "guardrails": [
            "No layer, position, decoder, or prompt suffix is selected on fresh laws.",
            "The physical label is the XOR/product of law sign and numerical sign.",
            "Direct and inverse laws are balanced.",
            "Answer-word order is counterbalanced within every law and case.",
            "Two algebraically equivalent formula surfaces are tested per law.",
            "The decoder is fitted only on completed old-law data.",
            "All prompts and outputs are retained, including failures.",
        ],
    }
    write_json(PROTOCOL, protocol)
    (OUT / "PROTOCOL.md").write_text(
        "# Fresh-law confirmation of an elicited physical-outcome direction\n\n"
        "This is the first confirmatory test of the scaffolded method. A "
        "parameter-free direction and layer 34 were fixed using the prior 16 "
        "laws. The direction is applied without refitting to 16 new laws. "
        "Within every new law, numerical increases and decreases reverse the "
        "physical outcome; across direct and inverse laws, the same numerical "
        "change also reverses its physical meaning. Answer order and algebraic "
        "surface are crossed as nuisance variables.\n"
    )
    print(f"frozen {len(prompts)} prompts over {len(LAWS)} fresh laws")
    print(f"protocol sha256: {sha256(PROTOCOL)}")


if __name__ == "__main__":
    main()
