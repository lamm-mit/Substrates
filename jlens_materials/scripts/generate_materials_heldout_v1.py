#!/usr/bin/env python3
"""Generate the frozen held-out materials association suite.

This file and its output are frozen before any held-out lens run is executed.
The ten mechanism families match the development suite, but every physical
description is new.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "prompts" / "materials-heldout-v1-preregistered.json"


FAMILIES = [
    {
        "key": "ductile",
        "title": "Energy-absorbing dimpled separation",
        "tracked": ["ductile", "nucleation", "coalescence", "void"],
        "texts": [
            "A round aluminum bar stretched into a pronounced waist before separating. The mating faces contained hemispherical pits centered on inclusions and a slanted tearing rim.",
            "A nickel coupon accumulated large permanent elongation before rupture. Microscopy showed a honeycomb of deep depressions surrounding second-phase particles.",
            "A copper tensile specimen lost much of its cross-sectional area before failure. Small particle-centered holes had enlarged and joined across the final ligament.",
            "An impact-tested structural metal absorbed substantial energy and developed broad lateral lips. Its failure surface was covered by rounded microscopic depressions.",
            "A pressurized metal tube bulged strongly before opening. The torn wall displayed elongated pits near the surface and deeper equiaxed pits toward the center.",
        ],
    },
    {
        "key": "boundary-attack",
        "title": "Chromium-depleted interfacial attack",
        "tracked": ["boundary", "corrosion", "sensitization"],
        "texts": [
            "A stainless weld spent several hours near 675 degrees Celsius. After nitric-acid exposure, narrow trenches followed the grain edges beside chromium-rich carbide rows.",
            "An austenitic pipe was slowly cooled through an intermediate temperature range. A later chemical test exposed a connected network of material loss outlining individual grains.",
            "Carbide particles decorated the interfaces in heat-affected stainless steel. Immersion testing removed chromium-depleted regions next to those particles while grain interiors remained sound.",
            "A stainless vessel received an improper post-weld thermal cycle. In service, fissures traced the polygonal grain-edge network instead of cutting through the crystallites.",
            "Austenitic sheet aged near 650 degrees Celsius showed continuous carbide films around grains. An aggressive liquid subsequently produced narrow grooves along the same paths.",
        ],
    },
    {
        "key": "cyclic",
        "title": "Progressive damage under repeated loading",
        "tracked": ["fatigue", "crack", "propagation"],
        "texts": [
            "A compressor blade endured millions of small vibration cycles before failing. Curved arrest lines radiated from a surface defect toward a small rough overload region.",
            "A bridge hanger separated although every measured load was below its monotonic capacity. Fine parallel striations recorded incremental front advance from a machining groove.",
            "A wheel axle survived years of repeated bending. The failure face contained thumbnail-shaped beach marks centered on several surface initiation sites.",
            "A bolted aluminum joint opened after many pressurization cycles. A smooth region with regularly spaced microscopic ridges occupied most of the separated surface.",
            "A rotating laboratory specimen failed after a long sequence of alternating stresses. Ratchet marks joined multiple surface origins before the remaining ligament broke suddenly.",
        ],
    },
    {
        "key": "cleavage",
        "title": "Faceted low-plasticity separation",
        "tracked": ["brittle", "cleavage", "transgranular"],
        "texts": [
            "A ferritic plate separated abruptly during cold weather with almost no macroscopic distortion. Flat facets and converging river markings crossed the grain interiors.",
            "A low-temperature impact specimen absorbed little energy and retained its original cross section. The shiny failure face contained crystallographic steps and feather patterns.",
            "A body-centered metal failed suddenly below its transition temperature. Microscopy revealed broad planar facets connected by sharp ridges through individual crystallites.",
            "A thick steel section opened under modest loading after rapid cooling. The separated faces showed flat reflective planes and negligible evidence of local necking.",
            "A notched iron alloy snapped during a cold bend test before appreciable permanent curvature developed. River-like markings on the facets converged toward the initiation site.",
        ],
    },
    {
        "key": "high-temperature-deformation",
        "title": "Time-dependent deformation under heat",
        "tracked": ["creep", "diffusion", "cavity"],
        "texts": [
            "A steam-pipe alloy elongated gradually during ten years of steady load at 600 degrees Celsius. Rounded pores appeared at triple junctions late in life.",
            "A constant-load high-temperature test showed rapid initial strain, a long nearly linear segment, and accelerating extension before rupture.",
            "A turbine component operated under sustained stress near half its melting temperature. Grain-edge pores enlarged and linked even though the applied stress was below the room-temperature yield value.",
            "A fine-grained metal slowly changed shape during a prolonged hot hold. Matter moved away from compressed grain faces and accumulated on faces under tension.",
            "A heat-resistant alloy accumulated measurable permanent strain with increasing service time. Interfacial pores and wedge-shaped separations were concentrated near the final rupture zone.",
        ],
    },
    {
        "key": "particle-strengthening",
        "title": "Nonshearable particle obstacles",
        "tracked": ["precipitation", "bowing", "strengthening"],
        "texts": [
            "A metal matrix contains closely spaced rigid dispersoids that cannot be cut. Moving lattice lines arc between them and leave closed loops around each obstacle.",
            "After aging, a dense population of hard nanoscale particles raises the stress required for plastic flow. Electron microscopy shows rings encircling the particles after deformation.",
            "Reducing the spacing between stable nonshearable particles increases the yield stress. Flexible line defects must curve strongly to pass through the remaining channels.",
            "A dispersion-treated alloy contains hard particles embedded in a softer matrix. Mobile lattice imperfections wrap around the particles rather than passing through them.",
            "Two alloys contain the same particle fraction, but the alloy with finer obstacle spacing requires a larger applied stress to move its line defects.",
        ],
    },
    {
        "key": "rapid-transformation",
        "title": "Quench-induced coordinated transformation",
        "tracked": ["martensite", "shear", "Bain", "tetragonal"],
        "texts": [
            "A carbon steel quenched too rapidly for long-range solute redistribution develops a hard acicular product with a body-centered cell elongated along one axis.",
            "Severe cooling converts the parent face-centered phase into fine laths almost instantaneously. Carbon remains trapped and the product has unequal axial lattice parameters.",
            "A rapidly cooled steel forms hard plates by coordinated short-distance atomic motion. No composition partitioning is detected across the moving interface.",
            "Quenching suppresses long-range atomic transport and produces packets of highly strained needles. The resulting body-centered unit cell has a c-to-a ratio above one.",
            "An austenitic sample undergoes an abrupt, composition-preserving shape change during cooling. The product is hard, plate-like, and internally twinned.",
        ],
    },
    {
        "key": "line-defect-motion",
        "title": "Motion of linear lattice imperfections",
        "tracked": ["dislocation", "glide", "slip"],
        "texts": [
            "Plastic flow begins when a linear lattice imperfection travels across a close-packed plane and leaves a permanent surface step.",
            "During loading of a single crystal, etch-pit positions move along selected crystallographic planes and an irreversible offset accumulates.",
            "A one-dimensional defect sweeps through the crystal in the direction of its mismatch vector, displacing the material on opposite sides of its plane.",
            "Once the resolved driving stress exceeds a threshold, mobile lattice lines traverse their favored planes and produce permanent strain.",
            "High-speed microscopy shows linear imperfections moving across densely packed planes. Each passage creates a small step at the free surface.",
        ],
    },
    {
        "key": "notch-resistance",
        "title": "Resistance to unstable flaw extension",
        "tracked": ["toughness", "crack", "fracture"],
        "texts": [
            "Two steels have comparable yield stress, yet one pre-notched plate absorbs several times more energy before sudden separation and survives a much larger flaw.",
            "A sharp starter notch blunts as a plastic zone develops, allowing the specimen to carry additional load before rapid extension occurs.",
            "A compact-tension sample requires substantial work to advance its sharp flaw. Instability begins only at an unusually large stress-intensity value.",
            "A structural alloy containing a long defect remains stable at loads that immediately separate a comparison alloy of similar strength.",
            "An instrumented impact test on a notched bar records high absorbed energy and extensive local deformation before the remaining ligament opens.",
        ],
    },
    {
        "key": "hot-air-surface-layer",
        "title": "Oxygen-rich high-temperature surface film",
        "tracked": ["oxidation", "oxide", "scale"],
        "texts": [
            "A superalloy held in oxygen-bearing gas gains mass while developing a dense ceramic surface film whose thickness follows an approximately parabolic time dependence.",
            "Long furnace exposure covers a steel coupon with several brittle oxygen-rich layers. Continued growth requires ions to cross the existing reaction product.",
            "A chromium-containing alloy forms a compact adherent surface film in hot air, sharply slowing additional inward oxygen transport.",
            "Repeated heating in air creates an external metal-oxygen reaction layer that thickens during each hold and sheds flakes during cooling.",
            "A turbine component exposed to hot combustion gas develops a multilayer surface product containing metal cations and oxygen anions.",
        ],
    },
]


def build_prompts() -> list[dict]:
    prompts = []
    for family in FAMILIES:
        for index, text in enumerate(family["texts"], start=1):
            prompts.append({
                "slug": f"heldout-v1-assoc-{family['key']}-{index:02d}",
                "shape": "ASSOCIATION",
                "protocol": "lens_eval",
                "domain": "materials",
                "category": family["key"],
                "target_family": family["key"],
                "phrasing_id": f"heldout-{family['key']}-{index:02d}",
                "title": f"{family['title']} {index}",
                "description": (
                    "Held-out unnamed materials description; compare predetermined "
                    "full-vocabulary ranks and target-free discovery under matched lenses."
                ),
                "text": text,
                "readout_selector": "final_prompt_token",
                "tracked": family["tracked"],
                "must_be_absent_from_input": True,
                "must_be_absent_from_output": True,
                "note": (
                    "Frozen held-out item. Predetermined terms are used only for the "
                    "rank endpoint; open-vocabulary generation and ranking cannot use them."
                ),
            })
    return prompts


def main() -> None:
    payload = {
        "format_version": 1,
        "study": "materials-heldout-v1",
        "frozen_before_execution": "2026-07-14",
        "description": (
            "Fifty held-out association descriptions: ten materials mechanism "
            "families with five new phrasings each. Generated and frozen before "
            "any held-out model or lens output was inspected."
        ),
        "prompts": build_prompts(),
    }
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    OUTPUT.write_text(serialized)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    print(f"wrote {OUTPUT} ({len(payload['prompts'])} prompts)")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
