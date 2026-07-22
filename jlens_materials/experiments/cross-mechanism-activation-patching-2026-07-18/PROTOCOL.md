# Frozen option-free cross-mechanism activation patching

Frozen: `2026-07-17T17:50:28-04:00`, before any output from this patching
experiment was calculated.

## Why this experiment is needed

The earlier grain-size patching result is causal, but its donor relation and
donor answer are inseparable: refinement always implies `higher`, and
coarsening always implies `lower`. A full late-state transplant could
therefore transfer an answer decision rather than a grain-size relation.

This experiment removes explicit answer choices and crosses three factors:

1. donor physical outcome (property increases or decreases);
2. donor numerical direction (the measured input increases or decreases);
3. answer vocabulary (`higher/lower` or `greater/smaller`).

The six mechanisms are balanced: three have a direct response to increasing
the changed quantity and three have an inverse response. Donor outcome and
numerical direction can therefore be separated across mechanisms. Transfers
between the two answer vocabularies additionally cannot be explained by
copying one particular answer token.

## Exact cohort

Use only the exact `stem` field from all 24 `anchor` rows in
`experiments/answer-code-binding-2026-07-17/prompt_manifest.json`: four
material cases in each of six mechanisms. No answer choices, answer words,
arbitrary codes, response instruction, or checkpoint marker are appended.

Every receiver is paired with all four anchor donors from each of the other
five mechanisms. This gives 480 receiver--donor pairs. At registered layers
16, 24, 32, and 37, replace the receiver's final-prompt-position post-block
residual with the archived donor residual from the same layer. The resulting
1,920 patches use the frozen Gemma checkpoint and no Jacobian lens.

The output is measured using the receiver's own scientific answer pair. A
`higher/lower` donor can therefore be evaluated in a `greater/smaller`
receiver, and conversely.

## Frozen endpoint

Within each donor mechanism, two anchor cases have a positive physical
outcome and two have a negative outcome. For every ordered donor-to-receiver
mechanism pair, receiver case, and layer, calculate:

`mean patched receiver margin | positive-outcome donor`

minus

`mean patched receiver margin | negative-outcome donor`.

Average over receiver cases and the four frozen layers, then average the two
patching directions within each unordered mechanism pair. Positive means the
donor's physical outcome controls the receiver's answer margin.

Report the endpoint for:

- all 15 unordered mechanism pairs;
- nine cross-answer-vocabulary pairs;
- nine opposite-response-orientation pairs, for which physical outcome and
  numerical trend reverse relative to one another;
- five pairs satisfying both conditions.

The matched numerical-direction contrast and all four layer curves are
retained as controls.

## Inference and decision rule

Enumerate every sign flip over the relevant unordered mechanism-pair effects
and report a two-sided exact p-value. Report a 30,000-sample pair bootstrap
interval with seed `20260718`.

Strong evidence requires positive exact-test results at `p <= 0.05` overall,
across answer vocabularies, and across opposite response orientations, plus
positive transfer from at least five of six donor mechanisms. One or two
passing subset gates is partial evidence; failure of the overall endpoint or
all transfer subsets is no evidence.

## Interpretation guardrail

Cross-vocabulary option-free transfer would rule out simple transplantation
of a particular answer word. It would still be compatible with a general
positive-versus-negative answer-decision state, rather than a
mechanism-specific constitutive law. A negative result would require the
existing grain patching claim to be narrowed to its original constrained
cohort. This experiment is causal but is not an independent replication,
because the natural question-end cohort had already been inspected.

## Fingerprints

- Runner:
  `4402b2c6327c07c1a0ff6eda75c89bd57496f08e148d8f6d42a8ab3957e67a44`
- Exact prompt manifest:
  `9ec1bdc3d12960cdcb2538bd214a1771d9f45ce5d1243fa4a2ab2ff3f83d8de5`
- Archived option-free states:
  `2451d643f94934c2ea5ef73acd06dbaf1ea58f43065a724c9190423a5be9b9dc`
