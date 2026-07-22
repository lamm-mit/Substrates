# Copyright 2026.  Apache-2.0.
"""Domain prompt sets for applying the Jacobian lens to scientific reasoning.

This module ports the experiment *shapes* from "Verbalizable Representations
Form a Global Workspace in Language Models" (transformer-circuits.pub/2026/
workspace) onto two scientific domains:

    * fracture / solid mechanics
    * protein science

Each experiment shape asks a different question of the J-lens:

    MULTIHOP        Does an *unspoken intermediate concept* surface in the lens
                    before the answer token?  (paper: Mars -> color -> red;
                    (4+17)*2+7 -> 21 -> 42 -> 49)  We track a chain of
                    domain intermediates and watch the depth at which each
                    reaches the top of the lens.

    ASSOCIATION     A vignette evokes ONE concept but never names it.  Does the
                    concept appear in the lens at the final token?  (paper:
                    grief / Einstein / noir association eval.)

    RECOGNITION     Raw symbolic input (an amino-acid sequence, a stress-state
                    tensor) whose *meaning* the model must recover.  (paper:
                    "protein" / "fluor" / "green" five residues into a GFP
                    sequence.)  The domain analogue of the paper's protein and
                    bug-detection examples.

    MODULATION      Hold a target concept in mind while copying an unrelated
                    carrier sentence; check whether the target surfaces in the
                    lens over the response even though the surface text never
                    mentions it.  (paper: "concentrate on ocean creatures".)

    REPORT_SWAP     "Think of a {category}. Answer in one word."  The chosen
                    word is a verbal report; swapping its lens vector for a
                    sibling should change the reported word.  (paper: think of
                    a sport -> Soccer, swap -> Rugby.)

Each Prompt carries the metadata needed to score it without searching for a
favourable layer/position: a protocol, fixed readout selector or response span,
ordered latent intermediates and synonyms, clean/counterfactual answers, and
control-condition labels.  The JSON loader also accepts the public
``jacobian-lens`` evaluation and experiment schemas, which makes the same
runner usable for new Gemma mechanics/materials datasets.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    """One lens prompt plus everything the analysis needs to score it.

    Attributes:
        slug: Stable identifier (also the output-figure basename).
        shape: One of MULTIHOP / ASSOCIATION / RECOGNITION / MODULATION /
            REPORT_SWAP -- selects which figure(s) the runner produces.
        domain: "fracture" or "protein".
        title: Human-readable heading for the figure.
        description: One-line caption shown under the figure.
        text: Raw-text prompt.  Mutually exclusive with `user`.
        user / system / assistant_prefill: chat-mode fields, assembled with the
            tokenizer chat template (needed for instruct models like Gemma-IT).
        protocol: Paper protocol implemented by the item: ``lens_eval``,
            ``directed_modulation``, ``verbal_report``, ``probe_swap``, or
            ``exploratory``.
        readout_selector: Predetermined scoring location: ``before_answer``,
            ``final_prompt_token``, ``last_newline``, ``assistant_response``,
            ``explicit``, or exploratory-only ``all_prompt``.
        readout_at: Python index used only by the ``explicit`` selector.
        tracked: Ordered intermediate concept words to follow across layers.
            Order encodes the *expected* computational order (earliest first),
            so the runner can test the paper's "intermediates surface in the
            order they are computed" claim.
        synonyms: Optional map surface-word -> list of equivalent surface forms
            (e.g. "45" -> ["45", "forty-five"]); rank is the min over forms.
        answer / acceptable_answers: Expected clean continuation.  Quantitative
            items are scored only when the clean model is correct.
        must_be_absent_from_input/output: Enforce the paper's latent-
            intermediate controls at tokenizer level.
        condition / control_group: Directed-modulation contrast metadata.
        swap_from / swap_to / swap_answer: Causal intervention definition.
            ``swap_from`` may be omitted for verbal report, where the model's
            clean greedy choice is used dynamically.
        note: Free-text scientific rationale, surfaced in the analysis writeup.
    """

    slug: str
    shape: str
    domain: str
    title: str
    description: str
    protocol: str = "lens_eval"
    text: str | None = None
    user: str | None = None
    system: str | None = None
    assistant_prefill: str = ""
    readout_selector: str = "final_prompt_token"
    readout_at: int = -1
    tracked: tuple[str, ...] = ()
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    answer: str | None = None
    acceptable_answers: tuple[str, ...] = ()
    must_be_absent_from_input: bool = False
    must_be_absent_from_output: bool = False
    condition: str | None = None
    control_group: str | None = None
    phrasing_id: str | None = None
    carrier_id: str | None = None
    target_family: str | None = None
    category: str | None = None
    swap_from: str | None = None
    swap_to: str | None = None
    swap_answer: str | None = None
    candidates: tuple[str, ...] = ()
    note: str = ""


def resolve_text(p: Prompt, tokenizer) -> str:
    """Return the final prompt string, formatting chat-mode prompts with the
    tokenizer's chat template (matches jlens.examples.resolve_prompt)."""
    if p.user is None:
        if p.text is None:
            raise ValueError(f"prompt {p.slug!r} has neither text nor user")
        return p.text
    messages = []
    if p.system:
        messages.append({"role": "system", "content": p.system})
    messages.append({"role": "user", "content": p.user})
    if getattr(tokenizer, "chat_template", None) is None:
        # Base model with no chat template: assemble a plain instruction block
        # so the same prompt still runs (degraded — instruct behaviour needs an
        # -it checkpoint).
        head = (f"{p.system}\n\n" if p.system else "")
        return f"{head}{p.user}\n{p.assistant_prefill}"
    if p.assistant_prefill:
        messages.append({"role": "assistant", "content": p.assistant_prefill})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# --------------------------------------------------------------------------- #
# JSON <-> Prompt  (prompts are data, not code)
# --------------------------------------------------------------------------- #

_FIELDS = {f.name for f in fields(Prompt)}


def to_dict(p: Prompt) -> dict:
    """Prompt -> plain dict, dropping empty/default fields for tidy JSON."""
    d = asdict(p)
    for key in ("tracked", "acceptable_answers", "candidates"):
        d[key] = list(d[key])
    return {k: v for k, v in d.items() if v not in (None, "", [], {}, ())}


def from_dict(d: dict) -> Prompt:
    """Plain dict -> Prompt. Unknown keys are ignored (forward-compatible);
    ``tracked`` accepts a list; ``shape``/``domain``/``title`` fall back to
    sensible defaults so a minimal ``{"text": ...}`` is a valid prompt."""
    d = dict(d)
    if "prompt" in d and "text" not in d:
        d["text"] = d["prompt"]
    if "intermediates" in d and "tracked" not in d:
        d["tracked"] = d["intermediates"]
    if "intermediate" in d and "tracked" not in d:
        d["tracked"] = [d["intermediate"]]
    if "target" in d and "answer" not in d:
        d["answer"] = d["target"]
    if "swap_answer" in d or ("intermediate" in d and "swap_to" in d):
        d.setdefault("protocol", "probe_swap")
        d.setdefault("shape", "PROBE_SWAP")
    elif "phrasings" not in d:
        d.setdefault("protocol", "lens_eval")
    if d.get("protocol") in {"lens_eval", "probe_swap"}:
        d.setdefault("readout_selector", "before_answer")
        d.setdefault("must_be_absent_from_input", True)
        d.setdefault("must_be_absent_from_output", True)
    kw = {k: v for k, v in d.items() if k in _FIELDS}
    kw.setdefault("slug", d.get("slug") or d.get("name") or "adhoc")
    kw.setdefault("shape", "MULTIHOP")
    kw.setdefault("domain", d.get("domain", "custom"))
    kw.setdefault("title", d.get("title") or d.get("name") or kw["slug"])
    kw.setdefault("description", d.get("description", ""))
    for key in ("tracked", "acceptable_answers", "candidates"):
        if key in kw:
            kw[key] = tuple(kw[key])
    return Prompt(**kw)


def load_prompts(source: str) -> list[Prompt]:
    """Load prompts from JSON. ``source`` is one of:

    - a path to a ``.json`` file holding a list of prompt dicts, or a dict with
      a ``"prompts": [...]`` key;
    - a path to a directory — every ``*.json`` in it is loaded and concatenated;
    - a raw JSON string (a single dict, or a list of dicts) — for defining a
      prompt on the fly on the command line.
    """
    stripped = source.lstrip()
    if stripped[:1] in "{[":  # inline JSON string (not a path)
        data = json.loads(source)
        records = (data.get("prompts", data.get("items", data))
                   if isinstance(data, dict) else data)
        if isinstance(records, dict):
            records = [records]
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"prompt source not found: {source!r}")
        blobs = ([p] if p.is_file()
                 else sorted(x for x in p.iterdir() if x.suffix == ".json"))
        records = []
        for f in blobs:
            data = json.loads(f.read_text())
            rows = (data.get("prompts", data.get("items", data))
                    if isinstance(data, dict) else data)
            if not isinstance(rows, list):
                rows = [rows]
            lower = f.name.lower()
            defaults: dict[str, object] = {}
            bundled_legacy = (f.resolve().parent == (Path(__file__).resolve().parent /
                                                     "prompts") and
                              lower in {"fracture.json", "protein.json"})
            if bundled_legacy:
                defaults = {"protocol": "exploratory"}
            elif "lens-eval-association" in lower:
                defaults = {"shape": "ASSOCIATION",
                            "readout_selector": "final_prompt_token"}
            elif "lens-eval-poetry" in lower:
                defaults = {"shape": "POETRY", "readout_selector": "last_newline"}
            elif "lens-eval-typo" in lower:
                defaults = {"shape": "TYPO", "readout_selector": "final_prompt_token"}
            elif "lens-eval-multilingual" in lower:
                defaults = {"shape": "MULTILINGUAL", "readout_selector": "before_answer"}
            elif "lens-eval-order-ops" in lower:
                defaults = {"shape": "ORDER_OPS", "readout_selector": "before_answer"}
            elif "lens-eval-multihop" in lower:
                defaults = {"shape": "MULTIHOP", "readout_selector": "before_answer"}
            records += [{**defaults, **row} if isinstance(row, dict) else row
                        for row in rows]
    return [from_dict(r) for r in records]


def dump_prompts(prompts: list[Prompt], path: str) -> None:
    """Write ``prompts`` to a JSON file (used to export the built-in templates)."""
    Path(path).write_text(
        json.dumps({"prompts": [to_dict(p) for p in prompts]},
                   indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Fracture / solid mechanics
# --------------------------------------------------------------------------- #

FRACTURE: list[Prompt] = [
    Prompt(
        slug="fracture-griffith",
        shape="MULTIHOP",
        domain="fracture",
        title="Griffith criterion (multi-hop)",
        description=(
            "Infer the classical brittle-failure framework, then the flaw "
            "quantity that controls the critical load."
        ),
        text=(
            "A brittle elastic plate contains a sharp internal flaw. Its "
            "failure load is predicted by balancing the stored work released "
            "as the flaw advances against the cost of forming two fresh faces. "
            "With material properties fixed, the critical load is inversely "
            "proportional to the square root of the flaw's"
        ),
        readout_selector="before_answer",
        tracked=("Griffith", "crack"),
        answer="length",
        must_be_absent_from_input=True,
        must_be_absent_from_output=True,
        note=(
            "Griffith's result sigma_c ~ sqrt(2 E gamma / (pi a)). The answer "
            "token is 'length'; 'Griffith' and 'crack' are input-absent "
            "intermediates rather than words copied from the question."
        ),
    ),
    Prompt(
        slug="fracture-fatigue",
        shape="MULTIHOP",
        domain="fracture",
        title="Cyclic loading -> failure mode (multi-hop)",
        description=(
            "Repeated sub-yield reversals imply a damage mechanism before the "
            "question asks for its microscopic surface marking."
        ),
        text=(
            "A polished steel shaft undergoes millions of sub-yield load "
            "reversals. Microscopy shows one small increment of advance per "
            "reversal. The characteristic parallel markings left on the "
            "failure surface are called"
        ),
        readout_selector="before_answer",
        tracked=("fatigue",),
        answer="striations",
        must_be_absent_from_input=True,
        must_be_absent_from_output=True,
        note=(
            "The answer is 'striations'; the latent failure mechanism is "
            "'fatigue', which is absent from both the question and intended answer."
        ),
    ),
    Prompt(
        slug="fracture-toughness-chain",
        shape="MULTIHOP",
        domain="fracture",
        title="Stress-intensity factor chain (multi-hop)",
        description=(
            "Compare a computed crack-driving value to a critical material "
            "value, then predict the flaw response."
        ),
        text=(
            "For an edge-flawed plate under load, an engineer obtains "
            "K_I = 80 MPa sqrt(m). The measured critical material value is "
            "K_Ic = 50 MPa sqrt(m). Since the computed driving value exceeds "
            "the resistance, the flaw will"
        ),
        readout_selector="before_answer",
        tracked=("toughness", "unstable"),
        answer="grow",
        acceptable_answers=("propagate",),
        must_be_absent_from_input=True,
        must_be_absent_from_output=True,
        note=(
            "Multi-step like (4+17)*2+7: compute K, then compare to K_Ic, then "
            "conclude 'propagate'. The paper's claim is that these steps "
            "surface in the lens in computed order and at successively later "
            "layers."
        ),
    ),
    Prompt(
        slug="fracture-brittle-assoc",
        shape="ASSOCIATION",
        domain="fracture",
        title="Brittle fracture (association, unnamed)",
        description=(
            "A vignette of glass shattering. Does 'brittle' / 'fracture' rise "
            "in the lens at the final token though never named?"
        ),
        text=(
            "The technician tapped the cold pane once with the hammer. Without "
            "bending or warning it flew apart into a hundred glittering shards "
            "that skated across the floor, each edge sharp as the moment "
            "before."
        ),
        readout_selector="final_prompt_token",
        tracked=("brittle", "fracture"),
        must_be_absent_from_input=True,
        must_be_absent_from_output=True,
        note=(
            "Association eval (paper: grief/Einstein/noir). The passage evokes "
            "brittle fracture without the words 'brittle' or 'fracture'; a "
            "workspace representation of the concept should be readable at the "
            "closing token."
        ),
    ),
    Prompt(
        slug="fracture-ductile-assoc",
        shape="ASSOCIATION",
        domain="fracture",
        title="Ductile failure (association, unnamed)",
        description=(
            "A copper rod necking and drawing out. Contrast partner to the "
            "brittle vignette: expect 'ductile'/'plastic'/'neck'."
        ),
        text=(
            "As the load climbed the copper rod stopped springing back. It "
            "grew visibly thinner at one waist, stretching and drawing out like "
            "warm taffy, holding on far past the point anyone expected before "
            "it finally parted."
        ),
        readout_selector="final_prompt_token",
        tracked=("ductile", "plastic", "necking"),
        must_be_absent_from_input=True,
        must_be_absent_from_output=True,
        note=(
            "The matched control for fracture-brittle-assoc. Comparing the two "
            "tests whether the lens cleanly separates the two canonical failure "
            "regimes from prose alone."
        ),
    ),
    Prompt(
        slug="fracture-stress-recognition",
        shape="RECOGNITION",
        domain="fracture",
        title="Stress state from a tensor (recognition)",
        description=(
            "A raw Cauchy stress tensor with one large tensile eigenvalue. "
            "Does the lens read out 'tension'/'crack'/'mode'?"
        ),
        text=(
            "Consider the Cauchy stress tensor (MPa):\n"
            "[[420,   0,   0],\n"
            " [  0,  15,   0],\n"
            " [  0,   0,  10]]\n"
            "The dominant loading state this represents is"
        ),
        readout_selector="before_answer",
        tracked=("uniaxial", "opening"),
        answer="tension",
        must_be_absent_from_input=True,
        must_be_absent_from_output=True,
        note=(
            "Analogue of the paper reading 'protein' out of a raw sequence: "
            "here the model must parse a numeric tensor, recognise near-uniaxial "
            "tension, and (in fracture terms) associate it with Mode I opening."
        ),
    ),
    Prompt(
        slug="fracture-modulation",
        shape="MODULATION",
        protocol="directed_modulation",
        domain="fracture",
        title="Hold 'crack-tip plasticity' while copying (modulation)",
        description=(
            "Copy an unrelated sentence while concentrating on crack-tip "
            "plasticity. Does the concept surface over the carrier response?"
        ),
        user=(
            'Write "The lab notebook lay open on the bench." Concentrate on '
            "crack-tip plasticity while you write the sentence. Don't write "
            "anything else."
        ),
        assistant_prefill="The lab notebook lay open on the bench.",
        readout_selector="assistant_response",
        tracked=("crack", "plastic", "plasticity", "tip", "yield", "zone"),
        must_be_absent_from_output=True,
        condition="focus",
        control_group="fracture-plasticity-notebook",
        note=(
            "Directed-modulation protocol. Success = 'crack'/'plastic'/'zone' "
            "appear in the lens across the carrier sentence, whose surface text "
            "is about a notebook -- evidence the workspace holds an instructed "
            "concept independent of output."
        ),
    ),
    Prompt(
        slug="fracture-report-swap",
        shape="REPORT_SWAP",
        protocol="verbal_report",
        domain="fracture",
        title="Think of a failure mode -> swap (verbal report)",
        description=(
            "'Think of a failure mode. Answer in one word.' Swap the chosen "
            "mode's lens vector for a sibling and check the report changes."
        ),
        user="Think of a metal failure mode. Answer in one word.",
        assistant_prefill="",
        readout_selector="final_prompt_token",
        tracked=("fatigue", "creep", "corrosion", "fracture", "buckling"),
        swap_to="creep",
        candidates=("fatigue", "creep", "corrosion", "fracture", "buckling"),
        note=(
            "Verbal-report swap (paper: Soccer->Rugby). The siblings are all "
            "single-word failure modes; swapping the chosen word's J-lens "
            "coordinate for a sibling's should flip the one-word answer."
        ),
    ),
]

# Matched modulation controls use the identical teacher-forced carrier and
# target set. They are kept separate from the focus prompt so custom datasets
# can add many instruction phrasings while grouping trials by control_group.
FRACTURE[7:7] = [
    Prompt(
        slug="fracture-modulation-suppress",
        shape="MODULATION",
        protocol="directed_modulation",
        domain="fracture",
        title="Suppress crack-tip plasticity while copying",
        description="Matched suppress condition for the modulation contrast.",
        user=(
            'Write "The lab notebook lay open on the bench." Try not to think '
            "about crack-tip plasticity while you write it. Don't write anything else."
        ),
        assistant_prefill="The lab notebook lay open on the bench.",
        readout_selector="assistant_response",
        tracked=("crack", "plastic", "plasticity", "tip", "yield", "zone"),
        must_be_absent_from_output=True,
        condition="suppress",
        control_group="fracture-plasticity-notebook",
    ),
    Prompt(
        slug="fracture-modulation-control",
        shape="MODULATION",
        protocol="directed_modulation",
        domain="fracture",
        title="Copy without a side task",
        description="Matched no-instruction control for the modulation contrast.",
        user=(
            'Write "The lab notebook lay open on the bench." '
            "Don't write anything else."
        ),
        assistant_prefill="The lab notebook lay open on the bench.",
        readout_selector="assistant_response",
        tracked=("crack", "plastic", "plasticity", "tip", "yield", "zone"),
        must_be_absent_from_output=True,
        condition="control",
        control_group="fracture-plasticity-notebook",
    ),
]


# --------------------------------------------------------------------------- #
# Protein science
# --------------------------------------------------------------------------- #

PROTEIN: list[Prompt] = [
    Prompt(
        slug="protein-helix-multihop",
        shape="MULTIHOP",
        domain="protein",
        title="Backbone H-bonds -> secondary structure (multi-hop)",
        description=(
            "i to i+4 backbone hydrogen bonding. Track 'hydrogen'/'backbone' "
            "before 'helix'."
        ),
        text=(
            "In a folded protein, the carbonyl oxygen of one residue repeatedly "
            "pairs with the amide proton of the residue four positions later. "
            "The regular local structure produced by this i-to-i+4 pattern is "
            "the alpha"
        ),
        readout_selector="before_answer",
        tracked=("hydrogen", "backbone"),
        answer="helix",
        must_be_absent_from_input=True,
        must_be_absent_from_output=True,
        note=(
            "Two-hop: i,i+4 H-bond pattern -> alpha helix. 'hydrogen'/'bond'/"
            "'backbone' are the unspoken intermediates before the answer "
            "'helix'."
        ),
    ),
    Prompt(
        slug="protein-gfp-recognition",
        shape="RECOGNITION",
        protocol="exploratory",
        domain="protein",
        title="GFP sequence -> function (recognition)",
        description=(
            "The paper's own example, ported: a GFP N-terminal fragment. Watch "
            "'protein'/'fluor'/'green' appear a few residues in."
        ),
        text=(
            "What is this? MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
        ),
        readout_at=-1,
        tracked=("protein", "fluor", "green", "GFP", "sequence", "amino"),
        note=(
            "Direct port of the paper's Figure 3 protein example (GFP). Their "
            "lens reads 'protein' ~5 residues in, then 'fluor' and 'green'. "
            "Reproducing this on Gemma is the cleanest cross-model check that "
            "the fitted lens is behaving."
        ),
    ),
    Prompt(
        slug="protein-collagen-recognition",
        shape="RECOGNITION",
        protocol="exploratory",
        domain="protein",
        title="Gly-X-Y repeat -> collagen (recognition)",
        description=(
            "A (Gly-Pro-Hyp)_n-like repeat. Does the lens surface "
            "'collagen'/'triple'/'helix' from the motif alone?"
        ),
        text=(
            "What is this? GPPGPKGDPGPQGPRGPAGPPGPAGPQGPKGDPGPQGPRGPAGPP"
        ),
        readout_at=-1,
        tracked=("collagen", "triple", "helix", "glycine", "proline", "fibril"),
        note=(
            "Gly-X-Y (often Gly-Pro-Hyp) is the collagen signature. Whether the "
            "lens jumps from generic 'protein' to specific 'collagen' probes "
            "how much structured biochemical knowledge sits in the workspace."
        ),
    ),
    Prompt(
        slug="protein-transmembrane-recognition",
        shape="RECOGNITION",
        protocol="exploratory",
        domain="protein",
        title="Hydrophobic stretch -> transmembrane (recognition)",
        description=(
            "A ~20-residue hydrophobic run. Expect 'membrane'/'hydrophobic'/"
            "'transmembrane'/'helix'."
        ),
        text=(
            "What is this? KKPNGFILVGGVLLLGAAIIGGVMLFAYVVKKPN"
        ),
        readout_at=-1,
        tracked=("membrane", "hydrophobic", "transmembrane", "helix", "lipid"),
        note=(
            "A flanked poly-(L/I/V/A/G/F) stretch is the transmembrane-helix "
            "signature. Recognising it requires reading residue *chemistry*, "
            "not just 'this is a sequence'."
        ),
    ),
    Prompt(
        slug="protein-amyloid-assoc",
        shape="ASSOCIATION",
        domain="protein",
        title="Amyloid aggregation (association, unnamed)",
        description=(
            "A vignette of misfolding and fibril formation. Does "
            "'amyloid'/'aggregate'/'fibril' rise without being named?"
        ),
        text=(
            "Under the microscope the soluble molecules had abandoned their "
            "folds. They had stacked edge to edge into long stiff ribbons that "
            "clumped into tangled plaques, each strand locked into its "
            "neighbour in a rigid pleated sheet."
        ),
        readout_selector="final_prompt_token",
        tracked=("amyloid", "fibril", "aggregate"),
        must_be_absent_from_input=True,
        must_be_absent_from_output=True,
        note=(
            "Association eval. The passage describes cross-beta amyloid "
            "formation (misfolding -> stacked beta strands -> fibrils/plaques) "
            "without the diagnostic words; a workspace concept of 'amyloid' "
            "should be readable at the close."
        ),
    ),
    Prompt(
        slug="protein-fold-multihop",
        shape="MULTIHOP",
        domain="protein",
        title="Hydrophobic collapse -> tertiary fold (multi-hop)",
        description=(
            "Chain: sequence -> hydrophobic core -> native fold. Track "
            "'hydrophobic'/'core'/'fold' in order."
        ),
        text=(
            "A denatured protein is returned to water. Its nonpolar side chains "
            "avoid the surrounding liquid and pack together in the interior. "
            "This solvent-driven compaction helps the chain recover its "
            "functional three-dimensional"
        ),
        readout_selector="before_answer",
        tracked=("hydrophobic", "core", "collapse"),
        answer="structure",
        acceptable_answers=("fold",),
        must_be_absent_from_input=True,
        must_be_absent_from_output=True,
        note=(
            "Anfinsen-style hydrophobic-collapse chain: the intermediates "
            "'hydrophobic' and 'core' should precede the answer 'fold'/"
            "'structure' across layers."
        ),
    ),
    Prompt(
        slug="protein-modulation",
        shape="MODULATION",
        protocol="directed_modulation",
        domain="protein",
        title="Hold 'disulfide bond' while copying (modulation)",
        description=(
            "Copy an unrelated sentence while concentrating on disulfide "
            "bonds. Does 'disulfide'/'cysteine'/'bridge' surface?"
        ),
        user=(
            'Write "The rain tapped against the window all afternoon." '
            "Concentrate on disulfide bonds while you write the sentence. "
            "Don't write anything else."
        ),
        assistant_prefill="The rain tapped against the window all afternoon.",
        readout_selector="assistant_response",
        tracked=("disulfide", "cysteine", "bridge", "bond", "sulfur", "cross"),
        must_be_absent_from_output=True,
        condition="focus",
        control_group="protein-disulfide-rain",
        note=(
            "Directed modulation. The carrier is about rain; success is "
            "'disulfide'/'cysteine' appearing in the lens across it, showing "
            "the workspace holds an instructed biochemical concept off-output."
        ),
    ),
    Prompt(
        slug="protein-report-swap",
        shape="REPORT_SWAP",
        protocol="verbal_report",
        domain="protein",
        title="Think of an amino acid -> swap (verbal report)",
        description=(
            "'Think of an amino acid. Answer in one word.' Swap the chosen "
            "residue's lens vector for a sibling; check the report changes."
        ),
        user="Think of an amino acid. Answer in one word.",
        assistant_prefill="",
        readout_selector="final_prompt_token",
        tracked=("glycine", "alanine", "leucine", "serine", "proline"),
        swap_to="proline",
        candidates=("glycine", "alanine", "leucine", "serine", "proline"),
        note=(
            "Verbal-report swap. Amino-acid names are the 'sport' category "
            "analogue; the swap tests whether the reported residue is set by "
            "the workspace coordinate at the report position."
        ),
    ),
]

PROTEIN[7:7] = [
    Prompt(
        slug="protein-modulation-suppress",
        shape="MODULATION",
        protocol="directed_modulation",
        domain="protein",
        title="Suppress disulfide bonds while copying",
        description="Matched suppress condition for the protein modulation contrast.",
        user=(
            'Write "The rain tapped against the window all afternoon." Try not '
            "to think about disulfide bonds while you write it. Don't write anything else."
        ),
        assistant_prefill="The rain tapped against the window all afternoon.",
        readout_selector="assistant_response",
        tracked=("disulfide", "cysteine", "bridge", "bond", "sulfur", "cross"),
        must_be_absent_from_output=True,
        condition="suppress",
        control_group="protein-disulfide-rain",
    ),
    Prompt(
        slug="protein-modulation-control",
        shape="MODULATION",
        protocol="directed_modulation",
        domain="protein",
        title="Copy without a biochemical side task",
        description="Matched no-instruction control for the protein contrast.",
        user=(
            'Write "The rain tapped against the window all afternoon." '
            "Don't write anything else."
        ),
        assistant_prefill="The rain tapped against the window all afternoon.",
        readout_selector="assistant_response",
        tracked=("disulfide", "cysteine", "bridge", "bond", "sulfur", "cross"),
        must_be_absent_from_output=True,
        condition="control",
        control_group="protein-disulfide-rain",
    ),
]


ALL_PROMPTS: list[Prompt] = FRACTURE + PROTEIN


def by_slug(slug: str) -> Prompt:
    for p in ALL_PROMPTS:
        if p.slug == slug:
            return p
    raise KeyError(slug)


def by_shape(shape: str) -> list[Prompt]:
    return [p for p in ALL_PROMPTS if p.shape == shape]
