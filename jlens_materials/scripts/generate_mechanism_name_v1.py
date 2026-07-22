#!/usr/bin/env python3
"""Generate the frozen mechanism-versus-eponym evaluation.

The experiment asks whether Gemma's readable internal state preferentially
contains a physical carrier (for example, ``dislocation``) rather than the
surname attached to the law (for example, ``Schmid``).  Every prompt omits
both tracked words and is stored verbatim for later audit.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "prompts" / "materials-mechanism-name-v1.json"


FAMILIES = [
    {
        "key": "grain-size-strength",
        "title": "Grain-size strengthening",
        "mechanism": "boundary",
        "name": "Hall",
        "texts": [
            "Two specimens of the same alloy receive different heat treatments. The one with much smaller crystallites requires a larger applied stress before permanent deformation begins.",
            "A polycrystalline metal is processed until its average crystallite diameter is reduced by half. Its resistance to the onset of plastic flow increases.",
            "An ultrafine-grained steel yields at a higher stress than a coarse-grained specimen with the same composition and phase fractions.",
            "Repeated rolling and recrystallization create many small crystallites. Moving line defects encounter interfaces more frequently, and the measured yield stress rises.",
            "A metal with coarse crystallites is compared with an otherwise identical fine-crystallite version. The refined material better resists the start of irreversible strain.",
            "Thermomechanical processing reduces the characteristic crystal-region size without changing chemistry. The resulting sheet shows higher yield strength.",
            "A line defect moving through one crystallite is blocked at the interface with a differently oriented neighbor. Making the crystallites smaller increases the number of such obstacles.",
            "Two copper samples differ mainly in crystallite size. The sample containing more interfaces per unit volume begins plastic flow at the larger stress.",
            "Recrystallized grains are refined from tens of micrometers to only a few micrometers. The stress needed to initiate slip increases approximately with the inverse square root of size.",
            "A structural alloy is strengthened by subdividing it into smaller misoriented crystal regions, creating more barriers to the passage of moving line imperfections.",
        ],
    },
    {
        "key": "flaw-controlled-failure",
        "title": "Flaw-controlled brittle failure",
        "mechanism": "crack",
        "name": "Griffith",
        "texts": [
            "Two brittle plates carry the same remote tensile load, but one contains a surface flaw four times longer. The plate with the larger flaw becomes unstable at a lower applied stress.",
            "A glass specimen with a sharp internal slit fails well below its theoretical bond strength because the local intensity at the slit tip is strongly amplified.",
            "Doubling the length of a sharp flaw in an elastic ceramic reduces the remote stress required for unstable extension by roughly the square root of two.",
            "A brittle solid separates when the elastic energy released by extending a sharp defect exceeds the energy needed to create two new surfaces.",
            "A polished glass fiber is much stronger than a scratched one of the same diameter because the scratch concentrates tensile loading at its tip.",
            "An elastic plate containing a central slit is loaded in tension. Instability begins when the energy-release rate reaches the material's surface-formation resistance.",
            "A ceramic component survives a given tensile stress when its largest flaw is short, but fails after processing introduces a much longer sharp flaw.",
            "The nominal strength of a brittle specimen scales approximately with the inverse square root of the size of its dominant sharp defect.",
            "A sharp notch in a glass sheet locally magnifies stress, allowing separation to start even though the average applied stress is modest.",
            "A brittle plate with a pre-existing slit becomes unstable when additional extension lowers the total elastic-plus-surface energy of the system.",
        ],
    },
    {
        "key": "resolved-slip-threshold",
        "title": "Resolved slip threshold",
        "mechanism": "dislocation",
        "name": "Schmid",
        "texts": [
            "A single crystal is pulled along an axis that places 120 MPa of resolved shear on its favored slip system. The critical value is 80 MPa, so microscopic plastic motion begins.",
            "An applied uniaxial stress is projected onto a crystal's slip-plane normal and slip direction. Plastic flow starts when that resolved component exceeds a critical threshold.",
            "Two identically loaded crystals have different orientations. The crystal with the larger product of the two direction cosines activates its favored slip system first.",
            "A crystal remains elastic until the shear component acting along a permitted direction on a permitted plane reaches a material-specific critical value.",
            "Under tension, only part of the applied load drives shear on a close-packed system. Once this projected driving force is large enough, line imperfections begin to glide.",
            "Rotating a single crystal changes the axial stress required to initiate slip even though the critical resolved shear resistance is unchanged.",
            "A favorably oriented slip system reaches its critical shear value before any other system, producing the first permanent surface steps.",
            "The onset of plasticity in a single crystal is predicted by multiplying the applied tensile stress by geometric factors for the plane normal and slip direction.",
            "One crystal orientation converts a larger fraction of the applied normal stress into shear on an allowed system, so it yields before a less favorable orientation.",
            "When the resolved driving force on the most favorably oriented plane-direction pair exceeds its critical resistance, mobile line defects sweep across the crystal.",
        ],
    },
    {
        "key": "lattice-diffusion-creep",
        "title": "Lattice-diffusion creep",
        "mechanism": "vacancy",
        "name": "Herring",
        "texts": [
            "A fine polycrystal held at high temperature and low stress lengthens as atoms diffuse through crystal interiors from compressed faces toward tensile faces.",
            "At elevated temperature, atoms move through the lattice between oppositely loaded grain faces, producing slow elongation without extensive line-defect glide.",
            "A polycrystalline solid changes shape under a small sustained load because empty lattice sites migrate through grain interiors in the direction opposite atomic transport.",
            "Slow deformation at high homologous temperature is controlled by bulk atomic transport, and its rate increases strongly as crystallite size decreases.",
            "Material is transported through crystal interiors from interfaces under compression to interfaces under tension, gradually elongating the specimen.",
            "A fine-grained ceramic deforms under low stress near its melting temperature through diffusion across the volume of each crystallite rather than along interfaces.",
            "Atoms leave compressed grain faces, traverse the crystal lattice, and attach at tensile faces, leading to time-dependent shape change.",
            "The steady deformation rate varies inversely with the square of crystallite size and is controlled by bulk self-diffusion.",
            "A high-temperature specimen elongates as missing-site defects carry mass transport through the interiors of its constituent crystals.",
            "Under a small tensile load, bulk diffusion moves atoms toward faces normal to the loading direction, producing gradual axial extension.",
        ],
    },
    {
        "key": "diffusionless-lattice-change",
        "title": "Diffusionless lattice change",
        "mechanism": "tetragonal",
        "name": "Bain",
        "texts": [
            "Rapid cooling converts a face-centered parent lattice into a body-centered product whose vertical cell dimension differs from the two horizontal dimensions.",
            "A quenched carbon steel forms a hard phase by coordinated atomic motion. The product unit cell is stretched along one axis relative to the other two.",
            "Long-range solute redistribution is suppressed during cooling, leaving a body-centered cell with an axial ratio different from unity.",
            "The parent face-centered cell transforms through compression along one crystallographic axis and expansion along the other two, creating a distorted body-centered product.",
            "A rapid diffusionless transformation produces an acicular phase whose unit cell has two equal short axes and one longer axis.",
            "Carbon trapped during quenching distorts the body-centered product so that its c dimension is not equal to a and b.",
            "A coordinated lattice deformation maps the parent cubic arrangement into a body-centered cell elongated in one direction.",
            "The hard lath-shaped product of severe quenching has a c-to-a lattice ratio measurably greater than one.",
            "Atoms move cooperatively over short distances during a rapid transformation, generating a body-centered structure with unequal axial dimensions.",
            "X-ray peaks from the quenched product split because one unit-cell axis differs from the two equivalent perpendicular axes.",
        ],
    },
]


def build() -> dict:
    prompts = []
    for family in FAMILIES:
        for index, text in enumerate(family["texts"], start=1):
            prompts.append({
                "slug": f"mechanism-name-{family['key']}-{index:02d}",
                "shape": "ASSOCIATION",
                "protocol": "lens_eval",
                "domain": "materials",
                "category": family["key"],
                "title": f"{family['title']} {index}",
                "description": (
                    "Frozen mechanism-versus-eponym vignette; both tracked "
                    "words are absent from the input and generated output."
                ),
                "text": text,
                "readout_selector": "final_prompt_token",
                "tracked": [family["mechanism"], family["name"]],
                "must_be_absent_from_input": True,
                "must_be_absent_from_output": True,
                "note": (
                    f"Paired comparison of physical carrier '{family['mechanism']}' "
                    f"against eponym token '{family['name']}'. Primary endpoint is "
                    "the within-item log10 rank difference, summarized separately "
                    "for J-lens and logit lens."
                ),
            })
    return {
        "description": (
            "Frozen 50-item mechanism-versus-eponym materials evaluation: five "
            "principles, ten independent wordings each."
        ),
        "prompts": prompts,
    }


def main() -> None:
    OUTPUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
