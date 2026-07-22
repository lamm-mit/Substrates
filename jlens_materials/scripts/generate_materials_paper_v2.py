#!/usr/bin/env python3
"""Generate the frozen materials-paper-v2 evaluation prompt manifest.

The suite contains 50 independent unnamed association vignettes (ten
mechanism families, five wordings each) and 24 matched directed-modulation
triplets. The generator is deterministic and intentionally stores every final
prompt as ordinary JSON for inspection and archival.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "prompts" / "materials-paper-v2-preregistered.json"


ASSOCIATION_FAMILIES = [
    {
        "key": "ductile",
        "title": "Dimpled tensile failure",
        "tracked": ["ductile", "nucleation", "coalescence", "void"],
        "texts": [
            "A tensile coupon developed pronounced necking before separation. Its failure face contained rounded dimples centered on inclusions, with a cup-shaped center and a surrounding slanted lip.",
            "After substantial plastic elongation, a metal bar separated along a surface filled with microscopic pits around second-phase particles. Neighboring pits had joined into larger depressions.",
            "A pulled specimen narrowed strongly in its gauge section. Electron microscopy of the separated faces showed deep equiaxed dimples around particles and elongated dimples near the outer rim.",
            "The load-displacement curve showed extensive post-yield deformation before final separation. The mating surfaces displayed many rounded depressions that had linked across the section.",
            "A structural alloy absorbed considerable energy and formed a visible waist before failure. Its cup-and-cone surface was covered by particle-centered microscopic pits.",
        ],
    },
    {
        "key": "boundary-attack",
        "title": "Interface-localized stainless-steel attack",
        "tracked": ["boundary", "corrosion", "sensitization"],
        "texts": [
            "An austenitic stainless component remained near 650 degrees Celsius long enough for chromium-rich carbides to decorate grain interfaces. Chloride exposure then produced connected grooves along those interfaces while the interiors stayed intact.",
            "A welded stainless plate experienced a prolonged intermediate-temperature dwell. Subsequent acid testing revealed narrow continuous attack paths around each grain and chromium-depleted zones beside carbide particles.",
            "Metallography of heat-affected stainless steel showed carbide films at crystallite edges. After immersion in an aggressive aqueous solution, material loss followed those edges rather than crossing the grain interiors.",
            "A stainless tube cooled slowly through the carbide-forming temperature range. Service in chloride brine later produced a network of fissures and grooves tracing the outlines of individual grains.",
            "Following an intermediate-temperature heat treatment, austenitic steel contained chromium-rich particles at grain interfaces. Chemical exposure preferentially removed narrow regions next to those particles.",
        ],
    },
    {
        "key": "cyclic",
        "title": "Cyclic loading and beach-mark morphology",
        "tracked": ["fatigue", "crack", "propagation"],
        "texts": [
            "A rotating shaft survived many thousands of alternating load cycles before sudden separation. Concentric beach marks spread from a surface notch, and fine parallel ridges covered the smooth region.",
            "A component failed after repeated stress reversals far below its monotonic strength. The separated surface contained a smooth semicircular region advancing from a machining mark and a rough final overload zone.",
            "An aircraft bracket endured a long sequence of takeoff and landing loads. Microscopy showed evenly spaced ridges marking incremental advance from a bolt-hole corner.",
            "A steel axle broke after millions of modest load cycles. Ratchet marks identified several surface origins, while curved arrest bands recorded successive stages of front advance.",
            "A notched specimen was subjected to a sinusoidal load until separation. Most of the surface was smooth and striated, followed by a small rough region formed during the final overload.",
        ],
    },
    {
        "key": "cleavage",
        "title": "Faceted low-temperature failure",
        "tracked": ["brittle", "cleavage", "transgranular"],
        "texts": [
            "A body-centered alloy separated at low temperature with negligible necking. The failure face consisted of flat crystallographic facets connected by river-like markings.",
            "During impact testing below the transition temperature, a ferritic specimen absorbed little energy. Its separated surface was shiny and faceted, with steps running through individual grains.",
            "A thick steel plate failed suddenly in cold service without visible plastic distortion. Microscopy revealed flat planes and converging river patterns cutting across grain interiors.",
            "A low-temperature bend test produced abrupt separation after almost no permanent curvature. The mating faces displayed large flat facets and feather-like markings.",
            "A notched ferritic bar showed low impact energy and a bright granular-looking failure face. Microscopic steps passed directly through crystallites rather than around their edges.",
        ],
    },
    {
        "key": "high-temperature-deformation",
        "title": "Slow high-temperature deformation",
        "tracked": ["creep", "diffusion", "cavity"],
        "texts": [
            "A component held for months near half its melting temperature elongated slowly under a constant load well below its room-temperature yield strength. Small rounded holes accumulated at grain interfaces before rupture.",
            "Under steady tensile load at elevated temperature, a polycrystal first deformed rapidly, then at an almost constant slow rate, and finally accelerated as microscopic holes linked along grain edges.",
            "A turbine alloy exposed to sustained stress and heat developed time-dependent strain. Late in life, isolated pores at triple junctions enlarged and joined into an interfacial failure path.",
            "A fine-grained metal changed shape gradually during a long low-stress hold close to its melting point. Atoms moved from compressed interfaces toward tensile ones and the specimen elongated.",
            "Long-duration testing at high temperature produced a nearly linear strain-versus-time segment followed by rapid extension. Microscopy found linked pores along grain interfaces.",
        ],
    },
    {
        "key": "particle-strengthening",
        "title": "Hard-particle obstacle strengthening",
        "tracked": ["precipitation", "bowing", "strengthening"],
        "texts": [
            "A metal contains a dense dispersion of hard nonshearable nanoscale particles. Lattice line imperfections curve between neighboring particles and leave loops around them as load increases.",
            "Heat treatment creates many fine second-phase particles that resist cutting. Moving line defects must arc between the particles, and reducing their spacing raises the required stress.",
            "Electron microscopy after deformation shows loops encircling rigid particles within a metal matrix. The particles were too hard for moving lattice imperfections to pass through directly.",
            "An aged alloy gains yield strength from closely spaced stable particles. Under load, flexible line imperfections bend around each obstacle rather than slicing it.",
            "A fixed particle volume fraction is refined into a larger number of smaller nonshearable obstacles. The free span available to a moving line imperfection decreases markedly.",
        ],
    },
    {
        "key": "rapid-transformation",
        "title": "Rapid diffusionless steel transformation",
        "tracked": ["martensite", "shear", "Bain", "tetragonal"],
        "texts": [
            "Austenitic steel is cooled so quickly that long-range atomic redistribution cannot occur. A hard needle-like product forms by coordinated lattice motion, and its body-centered cell is elongated along one axis.",
            "Rapid quenching produces thin laths without allowing carbon to redistribute over long distances. The product lattice is body-centered but has unequal axial dimensions.",
            "A steel sample transforms almost instantaneously during severe cooling. Atoms move cooperatively over short distances, producing a hard plate-like phase with one stretched unit-cell axis.",
            "Suppressing atomic redistribution during cooling converts the parent face-centered phase into a hard acicular product. The new body-centered cell has a c-to-a ratio different from one.",
            "A quenched carbon steel contains packets of fine laths and high internal strain. Formation required a coordinated shape change rather than long-range solute transport.",
        ],
    },
    {
        "key": "line-defect-motion",
        "title": "Crystal line-imperfection motion",
        "tracked": ["dislocation", "glide", "slip"],
        "texts": [
            "Under resolved loading, a line imperfection moves across a close-packed plane in the direction of its mismatch vector, leaving a permanent offset at the free surface.",
            "Plastic strain begins when mobile line imperfections sweep across selected crystallographic planes and create small surface steps.",
            "A single crystal deforms by the passage of linear lattice imperfections along densely packed planes. Their motion produces an irreversible relative displacement of the crystal halves.",
            "Etch pits tracing mobile line imperfections shift across a crystal during loading. Each passage leaves a step whose direction follows the mismatch vector.",
            "Once the resolved driving force exceeds a critical value, a linear lattice defect travels on its favored plane and the specimen acquires permanent deformation.",
        ],
    },
    {
        "key": "notch-resistance",
        "title": "Resistance to notch-driven failure",
        "tracked": ["toughness", "crack", "fracture"],
        "texts": [
            "Two alloys have similar yield strength, but one notched specimen absorbs far more energy before unstable separation and tolerates a much larger flaw under the same remote load.",
            "A pre-notched plate continues to carry increasing load while a local damaged zone blunts the notch tip. Considerable work is required before rapid extension begins.",
            "An alloy with a sharp starter notch resists sudden extension under increasing tensile load. The critical stress-intensity value measured at instability is unusually high.",
            "During a compact-tension test, substantial energy is dissipated near the notch tip before the specimen separates. The material tolerates larger defects than a comparison alloy.",
            "A structural steel containing a sharp flaw remains stable under loads that cause immediate separation in a less damage-resistant steel. Its critical intensity threshold is larger.",
        ],
    },
    {
        "key": "hot-air-surface-layer",
        "title": "High-temperature environmental surface layer",
        "tracked": ["oxidation", "oxide", "scale"],
        "texts": [
            "A nickel alloy held in hot air develops an adherent multilayer surface film as metal atoms react with oxygen. Continued exposure thickens the film according to an approximately parabolic time law.",
            "During prolonged furnace exposure, a metal component gains mass and forms a dark brittle surface layer rich in oxygen and metal cations.",
            "A hot steel coupon exposed to air becomes covered by several ceramic-like surface layers. Growth slows as ions must travel through the existing film.",
            "Elevated-temperature air exposure produces a compact protective film on a chromium-containing alloy. The film limits further oxygen ingress by solid-state transport.",
            "A turbine component operated in oxygen-bearing gas develops a thick external reaction product that can spall during thermal cycling.",
        ],
    },
]


CARRIERS = [
    "The amber notebook rested beside the ceramic bowl.",
    "A silver spoon lay beneath the folded cloth.",
    "The green folder remained inside the wooden drawer.",
    "A small lantern stood near the closed window.",
    "The paper ticket waited under the glass jar.",
    "A blue scarf hung behind the narrow door.",
    "The brass compass sat beside the blank card.",
    "A white candle stood on the stone shelf.",
    "The red envelope remained beneath the old clock.",
    "A clay cup waited near the folded map.",
    "The black pencil lay across the yellow pad.",
    "A linen pouch rested beside the copper tray.",
    "The round mirror leaned against the gray box.",
    "A wooden token sat inside the shallow dish.",
    "The violet ribbon lay below the framed sketch.",
    "A clear bottle stood next to the woven basket.",
    "The bronze key rested above the sealed letter.",
    "A cotton glove lay beside the empty vase.",
    "The narrow ruler remained under the green book.",
    "A ceramic bird stood near the folded towel.",
    "The orange marker lay inside the metal tin.",
    "A glass bead rested beside the paper label.",
    "The brown wallet waited below the desk lamp.",
    "A steel clip lay across the plain postcard.",
]


FOCUS_FORMS = [
    "Concentrate on a dislocation pile-up and its stress concentration while you write it.",
    "Keep a dislocation pile-up and the stress it concentrates in mind while copying it.",
    "Maintain an internal focus on a dislocation pile-up and its local stress concentration as you write.",
    "Mentally attend to a dislocation pile-up and the concentrated stress near its head while reproducing it.",
    "Hold the idea of a dislocation pile-up and its stress concentration in mind during the writing task.",
    "Direct your attention to a dislocation pile-up and the stress concentration it creates while writing.",
    "While copying, actively think about a dislocation pile-up and its associated stress concentration.",
    "Keep your thoughts centered on a dislocation pile-up and concentrated stress as you produce the sentence.",
]


SUPPRESS_FORMS = [
    "Try not to think about a dislocation pile-up or its stress concentration while you write it.",
    "Keep your attention away from a dislocation pile-up and the stress it concentrates while copying it.",
    "Avoid internally focusing on a dislocation pile-up or its local stress concentration as you write.",
    "Mentally suppress thoughts of a dislocation pile-up and the concentrated stress near its head while reproducing it.",
    "Do not hold the idea of a dislocation pile-up or its stress concentration in mind during the writing task.",
    "Direct your attention away from a dislocation pile-up and the stress concentration it creates while writing.",
    "While copying, actively avoid thinking about a dislocation pile-up and its associated stress concentration.",
    "Keep your thoughts from dwelling on a dislocation pile-up or concentrated stress as you produce the sentence.",
]


def association_prompts() -> list[dict]:
    prompts = []
    for family in ASSOCIATION_FAMILIES:
        for index, text in enumerate(family["texts"], start=1):
            prompts.append({
                "slug": f"paper-v2-assoc-{family['key']}-{index:02d}",
                "shape": "ASSOCIATION",
                "protocol": "lens_eval",
                "domain": "materials",
                "category": family["key"],
                "title": f"{family['title']} {index}",
                "description": (
                    "Preregistered unnamed materials vignette; compare fixed-band "
                    "J-lens ranks with the matched logit-lens baseline."
                ),
                "text": text,
                "readout_selector": "final_prompt_token",
                "tracked": family["tracked"],
                "must_be_absent_from_input": True,
                "must_be_absent_from_output": True,
                "note": (
                    "Primary association-suite item. Inclusion requires tokenizer-level "
                    "input/output absence. The family-level endpoint is paired pass@k/AUC "
                    "for J-lens versus logit lens over the fixed 38--92% band."
                ),
            })
    return prompts


def modulation_prompts() -> list[dict]:
    prompts = []
    group = "paper-v2-pileup-directed-modulation"
    for index, carrier in enumerate(CARRIERS, start=1):
        form_index = (index - 1) % len(FOCUS_FORMS)
        carrier_id = f"carrier-{index:02d}"
        triplet_id = f"triplet-{index:02d}"
        common = {
            "shape": "MODULATION",
            "protocol": "directed_modulation",
            "domain": "materials",
            "description": "Preregistered matched directed-modulation triplet.",
            "assistant_prefill": carrier,
            "readout_selector": "assistant_response",
            "tracked": ["dislocation", "pile", "stress"],
            "must_be_absent_from_output": True,
            "control_group": group,
            "phrasing_id": triplet_id,
            "carrier_id": carrier_id,
            "target_family": "dislocation-pileup",
            "note": (
                "Primary paired endpoint: mean log10 best-rank contrast for focus "
                "versus suppress and neutral, computed within carrier and compared "
                "with the same contrast under the logit lens."
            ),
        }
        prompts.extend([
            {
                **common,
                "slug": f"paper-v2-mod-focus-{index:02d}",
                "title": f"Directed focus {index:02d}",
                "user": f"Write exactly \"{carrier}\" {FOCUS_FORMS[form_index]} Do not write anything else.",
                "condition": "focus",
            },
            {
                **common,
                "slug": f"paper-v2-mod-suppress-{index:02d}",
                "title": f"Directed suppression {index:02d}",
                "user": f"Write exactly \"{carrier}\" {SUPPRESS_FORMS[form_index]} Do not write anything else.",
                "condition": "suppress",
            },
            {
                **common,
                "slug": f"paper-v2-mod-control-{index:02d}",
                "title": f"Neutral carrier control {index:02d}",
                "user": f"Write exactly \"{carrier}\" Do not write anything else.",
                "condition": "control",
            },
        ])
    return prompts


def main() -> None:
    prompts = association_prompts() + modulation_prompts()
    payload = {
        "description": (
            "Frozen preregistered materials-paper-v2 suite: 50 independent "
            "association vignettes and 24 matched focus/suppress/neutral "
            "directed-modulation triplets. Generated deterministically by "
            "scripts/generate_materials_paper_v2.py."
        ),
        "prompts": prompts,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT} ({len(prompts)} prompts)")


if __name__ == "__main__":
    main()
