#!/usr/bin/env python3
"""Freeze a large, diverse, neutral-anchored relational benchmark."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "neutral-anchored-relational-physics-2026-07-18"
MANIFEST = OUT / "prompt_manifest.json"
PROTOCOL = OUT / "protocol.json"
RUNNER = ROOT / "scripts" / "run_neutral_anchored_relational_benchmark.py"
DEVELOPMENT = ROOT / "experiments" / "elicited-physics-abstraction-2026-07-18"


def law(
    law_id: str,
    category: str,
    domain: str,
    law_name: str,
    response: str,
    control: str,
    formula_a: str,
    formula_b: str,
    case_1: tuple[str, str, str],
    case_2: tuple[str, str, str],
    *,
    neutral_role: str | None = None,
) -> dict:
    sign = {"direct": 1, "inverse": -1, "neutral": 0}[category]
    return {
        "law_id": law_id,
        "category": category,
        "domain": domain,
        "law_sign": sign,
        "law_name": law_name,
        "response": response,
        "control": control,
        "formula_a": formula_a,
        "formula_b": formula_b,
        "cases": [case_1, case_2],
        "neutral_role": neutral_role,
    }


LAWS = [
    # Twenty direct relations.
    law("shear-strain-stress", "direct", "mechanics", "linear shear response", "shear strain gamma", "shear stress tau", "gamma = tau/G, with G fixed and positive", "gamma G = tau, with G fixed and positive", ("an elastic aluminum element", "5 MPa", "50 MPa"), ("an elastic polymer layer", "0.2 MPa", "2 MPa")),
    law("axial-elongation-force", "direct", "mechanics", "axial extension at fixed geometry", "elongation delta_L", "axial force F", "delta_L = F L/(E A), with L, E, and A fixed", "delta_L/F = L/(E A), with L, E, and A fixed", ("a steel tie", "1 kN", "10 kN"), ("a polymer fiber", "10 N", "100 N")),
    law("spring-energy-displacement", "direct", "mechanics", "elastic spring energy", "stored energy U", "displacement x", "U = k x^2/2, with k fixed and positive", "2 U/x^2 = k, with k fixed and positive", ("a coil spring", "1 mm", "10 mm"), ("an elastic flexure", "0.2 mm", "2 mm")),
    law("stress-intensity-applied-stress", "direct", "fracture", "mode-I intensity at fixed crack geometry", "stress intensity K_I", "applied stress sigma", "K_I = C sigma", "K_I/sigma = C", ("a cracked aluminum panel", "10 MPa", "100 MPa"), ("a cracked glass plate", "2 MPa", "20 MPa")),
    law("paris-rate-delta-k", "direct", "fatigue", "Paris fatigue-crack growth", "crack-growth rate da_dN", "stress-intensity range Delta_K", "da_dN = C Delta_K^m, with m > 0", "da_dN/Delta_K^m = C, with m > 0", ("a cyclically loaded steel", "5 MPa sqrt(m)", "20 MPa sqrt(m)"), ("a cyclically loaded aluminum alloy", "2 MPa sqrt(m)", "12 MPa sqrt(m)")),
    law("pore-permeability-radius", "direct", "transport", "pore-radius permeability scaling", "permeability k_perm", "pore radius r_p", "k_perm = C r_p^2", "k_perm/r_p^2 = C", ("a porous ceramic", "0.1 micrometers", "1 micrometer"), ("a model sandstone", "2 micrometers", "20 micrometers")),
    law("heat-capacity-mass", "direct", "thermal", "total heat capacity", "total heat capacity C_tot", "sample mass m_s", "C_tot = c_p m_s, with c_p fixed", "C_tot/m_s = c_p, with c_p fixed", ("an aluminum specimen", "1 g", "100 g"), ("a polymer specimen", "2 g", "200 g")),
    law("thermal-energy-temperature", "direct", "thermal", "sensible heating at fixed mass", "added thermal energy Q_h", "temperature rise Delta_T", "Q_h = m c_p Delta_T, with m and c_p fixed", "Q_h/Delta_T = m c_p, with m and c_p fixed", ("a copper block", "5 K", "50 K"), ("a ceramic pellet", "10 K", "100 K")),
    law("current-voltage", "direct", "electrical", "Ohmic current at fixed resistance", "electric current I_e", "applied voltage V", "I_e = V/R, with R fixed and positive", "I_e R = V, with R fixed and positive", ("a metallic resistor", "1 V", "10 V"), ("a conducting composite", "0.5 V", "5 V")),
    law("joule-power-current", "direct", "electrical", "Joule heating at fixed resistance", "electrical power P_e", "electric current I_e", "P_e = R I_e^2, with R fixed and positive", "P_e/I_e^2 = R, with R fixed and positive", ("a resistive wire", "0.1 A", "1 A"), ("a thin-film heater", "0.02 A", "0.2 A")),
    law("hall-voltage-field", "direct", "electronic", "Hall voltage at fixed current and geometry", "Hall voltage V_H", "magnetic field B", "V_H = C B", "V_H/B = C", ("a semiconductor Hall bar", "0.1 T", "1 T"), ("a metallic Hall sensor", "0.05 T", "0.5 T")),
    law("magnetization-field", "direct", "magnetic", "linear magnetic susceptibility", "magnetization M", "magnetic field H", "M = chi H, with chi fixed and positive", "M/H = chi, with chi fixed and positive", ("a paramagnetic crystal", "100 A m^-1", "1000 A m^-1"), ("a linear magnetic composite", "50 A m^-1", "500 A m^-1")),
    law("absorbance-thickness", "direct", "optical", "Beer-Lambert absorbance", "optical absorbance A_opt", "absorber thickness L", "A_opt = alpha L, with alpha fixed and positive", "A_opt/L = alpha, with alpha fixed and positive", ("a dyed polymer film", "10 micrometers", "100 micrometers"), ("an absorbing semiconductor layer", "0.1 micrometers", "1 micrometer")),
    law("capillary-rise-surface-tension", "direct", "interfacial", "capillary rise at fixed tube radius", "capillary height h_c", "surface tension gamma_s", "h_c = C gamma_s", "h_c/gamma_s = C", ("a wetting liquid in a glass tube", "0.02 N m^-1", "0.08 N m^-1"), ("a molten metal in a fixed capillary", "0.2 N m^-1", "0.8 N m^-1")),
    law("nucleation-rate-sites", "direct", "phase transformation", "heterogeneous nucleation at fixed per-site rate", "total nucleation rate J_n", "active-site density N_s", "J_n = C N_s", "J_n/N_s = C", ("a solidifying alloy", "1e8 m^-3", "1e12 m^-3"), ("a crystallizing glass", "1e10 m^-3", "1e14 m^-3")),
    law("grain-size-time", "direct", "microstructure", "normal grain growth at fixed mobility", "grain size d_g", "annealing time t", "d_g^2 = d_0^2 + K t, with K > 0", "(d_g^2 - d_0^2)/t = K, with K > 0", ("an annealed steel", "1 hour", "16 hours"), ("a sintered ceramic", "0.5 hour", "8 hours")),
    law("creep-strain-time", "direct", "deformation", "steady creep accumulation", "creep strain epsilon_c", "exposure time t", "epsilon_c = eps_dot t, with eps_dot fixed and positive", "epsilon_c/t = eps_dot, with eps_dot fixed and positive", ("a superalloy at fixed stress and temperature", "10 hours", "1000 hours"), ("a polymer under fixed load", "1 hour", "100 hours")),
    law("corrosion-loss-time", "direct", "corrosion", "constant-rate corrosion mass loss", "mass loss m_loss", "exposure time t", "m_loss = C t", "m_loss/t = C", ("a steel coupon in a fixed environment", "1 day", "100 days"), ("a magnesium coupon in a fixed electrolyte", "2 hours", "200 hours")),
    law("pressure-drop-flow", "direct", "fluid mechanics", "linear pressure drop at fixed hydraulic resistance", "pressure drop Delta_p", "volumetric flow Q_v", "Delta_p = R_h Q_v, with R_h fixed and positive", "Delta_p/Q_v = R_h, with R_h fixed and positive", ("flow through a porous plug", "1 mL s^-1", "10 mL s^-1"), ("flow through a microchannel", "0.1 mL s^-1", "1 mL s^-1")),
    law("reaction-extent-time", "direct", "kinetics", "constant-rate reaction extent", "transformed fraction X", "reaction time t", "X = k t, with k fixed and 0 < X < 1", "X/t = k, with k fixed and 0 < X < 1", ("an early-stage solid-state reaction", "1 minute", "5 minutes"), ("an early-stage cure reaction", "2 minutes", "10 minutes")),
    # Twenty inverse relations.
    law("shear-strain-modulus", "inverse", "mechanics", "shear strain at fixed stress", "shear strain gamma", "shear modulus G", "gamma = tau/G, with tau fixed and positive", "gamma G = tau, with tau fixed and positive", ("an elastic layer at fixed shear stress", "1 GPa", "50 GPa"), ("a composite at fixed shear stress", "2 GPa", "20 GPa")),
    law("axial-elongation-area", "inverse", "mechanics", "axial extension versus section area", "elongation delta_L", "cross-sectional area A", "delta_L = F L/(E A), with F, L, and E fixed", "delta_L A = F L/E, with F, L, and E fixed", ("a tensile bar", "1 mm^2", "100 mm^2"), ("a polymer ligament", "0.1 mm^2", "10 mm^2")),
    law("frequency-mass", "inverse", "vibrations", "spring-mass natural frequency", "natural frequency f_n", "attached mass m", "f_n = C/sqrt(m)", "f_n^2 m = C^2", ("a laboratory oscillator", "0.1 kg", "10 kg"), ("a microresonator with an added mass", "1 microgram", "100 micrograms")),
    law("fatigue-life-delta-k", "inverse", "fatigue", "power-law fatigue life", "fatigue life N_f", "stress-intensity range Delta_K", "N_f = C/Delta_K^m, with m > 0", "N_f Delta_K^m = C, with m > 0", ("a cyclically loaded steel", "5 MPa sqrt(m)", "20 MPa sqrt(m)"), ("a cyclically loaded titanium alloy", "3 MPa sqrt(m)", "15 MPa sqrt(m)")),
    law("hydraulic-resistance-permeability", "inverse", "transport", "Darcy resistance at fixed geometry", "hydraulic resistance R_h", "permeability k_perm", "R_h = C/k_perm", "R_h k_perm = C", ("a porous ceramic filter", "1e-16 m^2", "1e-12 m^2"), ("a sandstone core", "1e-15 m^2", "1e-13 m^2")),
    law("diffusive-flux-tortuosity", "inverse", "transport", "effective diffusion flux at fixed gradient", "diffusive flux magnitude J", "tortuosity tau_t", "J = C/tau_t", "J tau_t = C", ("a porous electrode", "1.2", "4.8"), ("a polymer membrane", "1.5", "6.0")),
    law("diffusion-time-diffusivity-new", "inverse", "transport", "diffusive equilibration time", "equilibration time t_eq", "effective diffusivity D_eff", "t_eq = C/D_eff", "t_eq D_eff = C", ("a fixed porous slab", "1e-12 m^2 s^-1", "1e-9 m^2 s^-1"), ("a fixed coating", "1e-15 m^2 s^-1", "1e-11 m^2 s^-1")),
    law("thermal-time-coefficient", "inverse", "thermal", "lumped thermal time versus heat-transfer coefficient", "thermal time constant tau_th", "heat-transfer coefficient h", "tau_th = C/h", "tau_th h = C", ("a fixed metal body", "5 W m^-2 K^-1", "500 W m^-2 K^-1"), ("a fixed ceramic component", "10 W m^-2 K^-1", "1000 W m^-2 K^-1")),
    law("current-resistance", "inverse", "electrical", "Ohmic current at fixed voltage", "electric current I_e", "electrical resistance R", "I_e = V/R, with V fixed and positive", "I_e R = V, with V fixed and positive", ("a circuit at fixed voltage", "1 ohm", "100 ohm"), ("a sensor at fixed bias", "10 ohm", "1000 ohm")),
    law("capacitance-separation", "inverse", "electrical", "parallel-plate capacitance versus separation", "capacitance C_p", "plate separation d", "C_p = epsilon A/d, with epsilon and A fixed", "C_p d = epsilon A, with epsilon and A fixed", ("a ceramic capacitor", "1 micrometer", "100 micrometers"), ("an air-gap capacitor", "0.1 mm", "10 mm")),
    law("hall-coefficient-carriers", "inverse", "electronic", "single-carrier Hall coefficient", "Hall coefficient R_H", "carrier density n", "R_H = 1/(q n), with q fixed and positive", "R_H q n = 1, with q fixed and positive", ("an n-type semiconductor", "1e20 m^-3", "1e24 m^-3"), ("a conducting oxide", "1e22 m^-3", "1e26 m^-3")),
    law("skin-depth-frequency", "inverse", "electromagnetic", "electromagnetic skin depth", "skin depth delta_s", "frequency f", "delta_s = C/sqrt(f)", "delta_s^2 f = C^2", ("a copper conductor", "1 kHz", "1 MHz"), ("a magnetic alloy", "100 Hz", "100 kHz")),
    law("transmission-absorption", "inverse", "optical", "Beer-Lambert transmission", "transmitted intensity I_t", "absorption coefficient alpha", "I_t = I_0 exp(-alpha L), with I_0 and L fixed", "ln(I_t/I_0) = -alpha L, with I_0 and L fixed", ("a fixed optical film", "1e3 m^-1", "1e6 m^-1"), ("a fixed semiconductor layer", "1e4 m^-1", "1e7 m^-1")),
    law("capillary-rise-radius", "inverse", "interfacial", "capillary rise versus tube radius", "capillary height h_c", "tube radius r", "h_c = C/r", "h_c r = C", ("a wetting liquid in a glass tube", "10 micrometers", "1 mm"), ("a molten salt in a capillary", "50 micrometers", "5 mm")),
    law("nucleation-barrier-undercooling", "inverse", "phase transformation", "classical nucleation barrier scaling", "nucleation barrier Delta_G_star", "undercooling Delta_T", "Delta_G_star = C/Delta_T^2", "Delta_G_star Delta_T^2 = C", ("a solidifying pure metal", "5 K", "50 K"), ("a crystallizing melt", "10 K", "100 K")),
    law("boundary-area-grain-size", "inverse", "microstructure", "grain-boundary area density", "boundary area per volume S_v", "grain size d_g", "S_v = C/d_g", "S_v d_g = C", ("an equiaxed polycrystal", "1 micrometer", "100 micrometers"), ("a nanocrystalline metal", "20 nm", "2 micrometers")),
    law("rupture-life-stress", "inverse", "deformation", "power-law creep-rupture life", "rupture life t_r", "applied stress sigma", "t_r = C/sigma^n, with n > 0", "t_r sigma^n = C, with n > 0", ("a superalloy at fixed temperature", "100 MPa", "500 MPa"), ("a polymer at fixed temperature", "2 MPa", "20 MPa")),
    law("corrosion-resistance-defects", "inverse", "corrosion", "barrier resistance versus defect density", "polarization resistance R_p", "coating defect density n_d", "R_p = C/n_d", "R_p n_d = C", ("a coated steel surface", "1e2 m^-2", "1e6 m^-2"), ("a coated magnesium alloy", "1e3 m^-2", "1e7 m^-2")),
    law("pipe-drop-radius", "inverse", "fluid mechanics", "Poiseuille pressure drop versus radius", "pressure drop Delta_p", "pipe radius r", "Delta_p = C/r^4", "Delta_p r^4 = C", ("a capillary at fixed flow", "10 micrometers", "100 micrometers"), ("a tube at fixed flow", "0.5 mm", "5 mm")),
    law("settling-speed-viscosity", "inverse", "fluid mechanics", "Stokes settling speed at fixed particle size", "settling speed v_s", "fluid viscosity eta", "v_s = C/eta", "v_s eta = C", ("a particle in a Newtonian liquid", "1 mPa s", "100 mPa s"), ("a ceramic inclusion in a melt", "0.01 Pa s", "1 Pa s")),
    # Twenty neutral relations. First ten calibrate the empirical null; the
    # final ten validate it without contributing to centering or scaling.
    law("modulus-specimen-length", "neutral", "mechanics", "material modulus independent of specimen length", "Young modulus E", "specimen length L", "E = C", "E/L^0 = C", ("a homogeneous steel specimen", "10 mm", "1 m"), ("a homogeneous ceramic specimen", "1 mm", "100 mm"), neutral_role="calibration"),
    law("poisson-area", "neutral", "mechanics", "Poisson ratio independent of section area", "Poisson ratio nu", "cross-sectional area A", "nu = C", "nu/A^0 = C", ("a homogeneous aluminum specimen", "1 mm^2", "100 mm^2"), ("a homogeneous polymer specimen", "0.1 mm^2", "10 mm^2"), neutral_role="calibration"),
    law("yield-stress-gauge-length", "neutral", "mechanics", "local yield stress independent of gauge length", "yield stress sigma_y", "gauge length L_g", "sigma_y = C", "sigma_y/L_g^0 = C", ("a uniform steel coupon", "10 mm", "100 mm"), ("a uniform aluminum coupon", "5 mm", "50 mm"), neutral_role="calibration"),
    law("toughness-crack-length", "neutral", "fracture", "material toughness independent of test crack length", "fracture toughness K_IC", "initial crack length a", "K_IC = C", "K_IC/a^0 = C", ("a homogeneous ceramic under valid K-dominance", "0.1 mm", "1 mm"), ("a homogeneous alloy under valid K-dominance", "1 mm", "10 mm"), neutral_role="calibration"),
    law("burgers-density", "neutral", "crystal defects", "Burgers-vector magnitude independent of dislocation density", "Burgers-vector magnitude b", "dislocation density rho_d", "b = C", "b/rho_d^0 = C", ("one crystal structure at fixed composition", "1e10 m^-2", "1e15 m^-2"), ("a second fixed crystal structure", "1e9 m^-2", "1e14 m^-2"), neutral_role="calibration"),
    law("lattice-parameter-cell-count", "neutral", "crystallography", "lattice parameter independent of modeled cell count", "lattice parameter a_0", "number of unit cells N_c", "a_0 = C", "a_0/N_c^0 = C", ("a perfect crystal at fixed state", "10", "1000"), ("a second perfect crystal at fixed state", "20", "2000"), neutral_role="calibration"),
    law("diffusivity-slab-thickness-neutral", "neutral", "transport", "intrinsic diffusivity independent of slab thickness", "intrinsic diffusivity D", "slab thickness L", "D = C", "D/L^0 = C", ("one homogeneous material at fixed temperature", "10 micrometers", "1 mm"), ("a second homogeneous material at fixed temperature", "0.1 mm", "10 mm"), neutral_role="calibration"),
    law("conductivity-sample-area-neutral", "neutral", "thermal", "intrinsic conductivity independent of sample area", "thermal conductivity k", "sample area A", "k = C", "k/A^0 = C", ("a homogeneous ceramic at fixed temperature", "1 mm^2", "100 mm^2"), ("a homogeneous polymer at fixed temperature", "10 mm^2", "1000 mm^2"), neutral_role="calibration"),
    law("specific-heat-mass-neutral", "neutral", "thermal", "specific heat independent of sample mass", "specific heat c_p", "sample mass m_s", "c_p = C", "c_p/m_s^0 = C", ("a homogeneous metal at fixed temperature", "1 g", "1 kg"), ("a homogeneous polymer at fixed temperature", "0.1 g", "100 g"), neutral_role="calibration"),
    law("melting-temperature-mass", "neutral", "phase equilibrium", "melting temperature independent of macroscopic sample mass", "melting temperature T_m", "sample mass m_s", "T_m = C", "T_m/m_s^0 = C", ("a pure bulk metal", "1 mg", "1 kg"), ("a pure bulk ceramic", "10 mg", "10 kg"), neutral_role="calibration"),
    law("resistivity-wire-length-neutral", "neutral", "electrical", "intrinsic resistivity independent of wire length", "electrical resistivity rho_e", "wire length L", "rho_e = C", "rho_e/L^0 = C", ("a uniform copper wire at fixed temperature", "1 cm", "10 m"), ("a uniform alloy wire at fixed temperature", "2 cm", "20 m"), neutral_role="validation"),
    law("permittivity-capacitor-area-neutral", "neutral", "electrical", "intrinsic permittivity independent of electrode area", "dielectric permittivity epsilon_d", "electrode area A", "epsilon_d = C", "epsilon_d/A^0 = C", ("a homogeneous dielectric", "1 mm^2", "100 mm^2"), ("a second homogeneous dielectric", "10 mm^2", "1000 mm^2"), neutral_role="validation"),
    law("susceptibility-volume-neutral", "neutral", "magnetic", "intrinsic susceptibility independent of specimen volume", "magnetic susceptibility chi", "specimen volume V_s", "chi = C", "chi/V_s^0 = C", ("a homogeneous paramagnet", "1 mm^3", "1 cm^3"), ("a homogeneous diamagnet", "10 mm^3", "10 cm^3"), neutral_role="validation"),
    law("index-path-length-neutral", "neutral", "optical", "refractive index independent of optical path length", "refractive index n_r", "path length L", "n_r = C", "n_r/L^0 = C", ("a homogeneous glass at fixed wavelength", "0.1 mm", "10 cm"), ("a homogeneous polymer at fixed wavelength", "1 mm", "1 m"), neutral_role="validation"),
    law("surface-energy-area-neutral", "neutral", "interfacial", "specific surface energy independent of exposed area", "specific surface energy gamma_s", "exposed area A", "gamma_s = C", "gamma_s/A^0 = C", ("one clean crystal facet", "1 micrometer^2", "1 mm^2"), ("a second clean crystal facet", "10 micrometers^2", "10 mm^2"), neutral_role="validation"),
    law("boundary-energy-grain-size-neutral", "neutral", "microstructure", "specific boundary energy independent of grain size", "grain-boundary energy gamma_gb", "grain size d_g", "gamma_gb = C", "gamma_gb/d_g^0 = C", ("one boundary character at fixed chemistry", "100 nm", "100 micrometers"), ("a second boundary character at fixed chemistry", "1 micrometer", "1 mm"), neutral_role="validation"),
    law("hardness-specimen-width-neutral", "neutral", "mechanical properties", "bulk hardness independent of specimen width", "hardness H", "specimen width w", "H = C", "H/w^0 = C", ("a sufficiently large homogeneous steel sample", "2 mm", "20 mm"), ("a sufficiently large homogeneous ceramic sample", "1 mm", "10 mm"), neutral_role="validation"),
    law("expansion-coefficient-length-neutral", "neutral", "thermal", "expansion coefficient independent of specimen length", "thermal expansion coefficient alpha", "specimen length L", "alpha = C", "alpha/L^0 = C", ("a homogeneous alloy at fixed temperature", "1 mm", "1 m"), ("a homogeneous glass at fixed temperature", "2 mm", "2 m"), neutral_role="validation"),
    law("activation-energy-anneal-time-neutral", "neutral", "kinetics", "activation energy independent of measurement duration", "activation energy Q_a", "measurement duration t", "Q_a = C", "Q_a/t^0 = C", ("one fixed diffusion mechanism", "1 minute", "100 hours"), ("a second fixed reaction mechanism", "10 seconds", "10 hours"), neutral_role="validation"),
    law("atomic-mass-temperature-neutral", "neutral", "atomic physics", "atomic mass independent of temperature", "atomic mass m_a", "temperature T", "m_a = C", "m_a/T^0 = C", ("one isotope below ionization", "100 K", "1000 K"), ("a second isotope below ionization", "50 K", "500 K"), neutral_role="validation"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def make_prompt(
    item: dict,
    surface: str,
    case_index: int,
    numerical_sign: int,
    answer_order: str,
) -> dict:
    system, low, high = item["cases"][case_index]
    start, end = (low, high) if numerical_sign == 1 else (high, low)
    physical_sign = int(item["law_sign"]) * numerical_sign
    expected = {1: "higher", -1: "lower", 0: "unchanged"}[physical_sign]
    words = (
        ("higher", "lower", "unchanged")
        if answer_order == "higher-first"
        else ("unchanged", "lower", "higher")
    )
    prompt_id = (
        f"{item['law_id']}--{surface}--case{case_index + 1}--"
        f"{'up' if numerical_sign == 1 else 'down'}--{answer_order}"
    )
    user = (
        "Apply the same two-stage physical test to every case. Stage 1: use "
        "only the constitutive equation, not associations with material names, "
        "to determine whether the response increases, decreases, or remains "
        "unchanged as the control increases while all other quantities are "
        f"fixed. For {item['law_name']}, the response is {item['response']}, "
        f"the control is {item['control']}, and the relation is "
        f"{item[f'formula_{surface}']}. All stated coefficients are positive. "
        f"Complete this relation step silently. In {system}, "
        f"{item['control']} changes from {start} to {end}. Determine the "
        "numerical-change direction silently. Now compose the relation with "
        f"the numerical change to determine the direction of {item['response']}. "
        "Reply with exactly one lowercase word from this ordered set: "
        f"{words[0]}, {words[1]}, {words[2]}."
    )
    return {
        "prompt_id": prompt_id,
        "law_id": item["law_id"],
        "category": item["category"],
        "domain": item["domain"],
        "law_sign": int(item["law_sign"]),
        "neutral_role": item["neutral_role"],
        "surface": surface,
        "case_index": case_index,
        "system": system,
        "control": item["control"],
        "response": item["response"],
        "numeric_start": start,
        "numeric_end": end,
        "numerical_sign": numerical_sign,
        "physical_sign": physical_sign,
        "answer_order": answer_order,
        "expected_answer": expected,
        "user": user,
    }


def main() -> None:
    if (OUT / "raw.json").exists():
        raise RuntimeError("refusing to rebuild after model outputs exist")
    counts = {
        category: sum(x["category"] == category for x in LAWS)
        for category in ("direct", "inverse", "neutral")
    }
    if counts != {"direct": 20, "inverse": 20, "neutral": 20}:
        raise RuntimeError(f"unbalanced law inventory: {counts}")
    if len({x["law_id"] for x in LAWS}) != len(LAWS):
        raise RuntimeError("duplicate law id")
    prompts = [
        make_prompt(item, surface, case, numerical, order)
        for item in LAWS
        for surface in ("a", "b")
        for case in range(2)
        for numerical in (-1, 1)
        for order in ("higher-first", "unchanged-first")
    ]
    manifest = {
        "study_id": "neutral-anchored-relational-physics-2026-07-18",
        "laws": LAWS,
        "prompts": prompts,
        "dimensions": {
            "n_laws": len(LAWS),
            "n_direct": counts["direct"],
            "n_inverse": counts["inverse"],
            "n_neutral": counts["neutral"],
            "n_neutral_calibration": 10,
            "n_neutral_validation": 10,
            "n_prompts": len(prompts),
            "n_matched_pairs": len(prompts) // 2,
        },
    }
    write_json(MANIFEST, manifest)
    protocol = {
        "study_id": manifest["study_id"],
        "status": "frozen-before-large-cohort-output",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "model": "google/gemma-4-E4B-it",
        "model_revision": "a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
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
                "path": str((DEVELOPMENT / "prompt_manifest.json").relative_to(ROOT)),
                "sha256": sha256(DEVELOPMENT / "prompt_manifest.json"),
            },
            "development_states": {
                "path": str((DEVELOPMENT / "representations.npz").relative_to(ROOT)),
                "sha256": sha256(DEVELOPMENT / "representations.npz"),
            },
        },
        "frozen_lens": {
            "position": "final_prompt",
            "layer": 34,
            "state_normalization": "unit L2",
            "direction": "old-law positive-minus-negative physical-outcome centroids",
            "large_cohort_refit": "none",
        },
        "matched_contrast": (
            "Within each law/surface/case/answer-order cell, frozen physical "
            "score for numerical-up minus score for numerical-down."
        ),
        "neutral_normalization": {
            "center": "median law contrast over the 10 calibration-neutral laws",
            "scale": "1.4826 times MAD over the 10 calibration-neutral laws",
            "validation_neutrals": "the other 10 neutral laws are excluded from normalization",
            "reason": "defines an empirical physical null instead of assuming raw zero",
        },
        "primary_endpoints": {
            "direct_vs_inverse_auc": "threshold-free, 20 versus 20 laws",
            "direct_vs_validation_neutral_auc": "20 versus 10 laws",
            "validation_neutral_vs_inverse_auc": "10 versus 20 laws",
            "calibrated_sign_accuracy": "direct above and inverse below calibration-neutral median",
            "ordinal_spearman": "law sign -1/0/+1 versus neutral-normalized contrast, validation neutral only",
        },
        "success_rule": {
            "direct_inverse_auc_minimum": 0.90,
            "direct_neutral_auc_minimum": 0.80,
            "neutral_inverse_auc_minimum": 0.80,
            "calibrated_direct_inverse_accuracy_minimum": 0.80,
            "both_surface_direct_inverse_auc_minimum": 0.85,
            "ordinal_spearman_minimum": 0.60,
            "ordinal_permutation_p_maximum": 0.01,
        },
        "inference": {
            "bootstrap_seed": 20260723,
            "bootstrap_resamples": 50000,
            "permutation_seed": 20260724,
            "permutations": 100000,
        },
        "guardrails": [
            "The cohort has 20 direct, 20 inverse, and 20 neutral laws.",
            "The 60 laws span mechanics, fracture, fatigue, transport, thermal, electrical, magnetic, optical, interfacial, microstructural, corrosion, fluid, and kinetic domains.",
            "Only calibration-neutral laws define center and scale.",
            "Direct, inverse, and validation-neutral labels never alter the normalization.",
            "Every law crosses two equation surfaces, two cases, two numerical directions, and two answer orders.",
            "The layer and hidden direction remain frozen from old laws.",
            "All outputs, including failures, are retained.",
        ],
    }
    write_json(PROTOCOL, protocol)
    (OUT / "PROTOCOL.md").write_text(
        "# Neutral-anchored relational physics benchmark\n\n"
        "Sixty laws are balanced among direct, inverse, and physically neutral "
        "relations. Ten neutral laws define the empirical null center and robust "
        "scale; ten different neutral laws test that calibration. Direct and "
        "inverse outcomes are evaluated without refitting the old-law lens. "
        "Threshold-free AUC remains primary, while the neutral median supplies "
        "a physically interpretable cutoff for signed classification.\n"
    )
    print(f"frozen {len(prompts)} prompts over {len(LAWS)} laws: {counts}")
    print(f"protocol sha256: {sha256(PROTOCOL)}")


if __name__ == "__main__":
    main()
