# Frozen cross-mechanism physical-outcome test

Frozen: `2026-07-18T01:20:00-04:00`

## Status and motivation

This is a newly frozen analysis of the prospectively captured natural
question-end states. The within-mechanism option-free graph was inspected
before this protocol was written. No cross-mechanism outcome statistic had
been calculated.

The within-mechanism graph supplies the mechanism family and asks whether
nearby cases have the same physical outcome. Within any one of the six
families, however, the outcome is perfectly determined by whether the changed
quantity increases or decreases. A model could therefore pass by encoding
only numerical direction.

The six mechanisms form a balanced falsification:

- For crosslink density, dislocation density, and stiff-particle fraction, an
  increase in the changed variable increases the predicted property.
- For obstacle spacing, pearlite spacing, and porosity, an increase in the
  changed variable decreases the predicted property.

Consequently, two prompts can have the same physical consequence despite
opposite numerical directions. The critical test compares such cases across
different mechanisms.

## Scientific question

At the natural end of an option-free scientific question, does Gemma's state
organize different mechanisms by their physically predicted increase or
decrease rather than merely by whether the input number went up or down?

## Frozen cohort and representation

- The 72 natural question-end states from
  `option-free-question-end-2026-07-18`.
- Six mechanisms, four cases per mechanism, and three variants per case.
- No answer choices, answer words, arbitrary code, response instruction, or
  checkpoint marker.
- Primary representation: three-fit Jacobian cosine, averaged first across
  fits and then over the frozen 38--92% depth band.
- Direct decoder-basis and raw-state comparisons use identical states,
  layers, and queries.

## Cross-mechanism candidate queries

For every source prompt, each of the five other mechanism families, and each
of the three target surface variants, rank the four target-family cases.
Every query therefore contains exactly:

- two cases with the same harmonized physical outcome as the source;
- two cases with the opposite physical outcome.

This gives `72 x 5 x 3 = 1,080` rankings. No candidate shares the source
mechanism. The primary score is mean pairwise ROC--AUC across all rankings.
Top-1 same-outcome accuracy is secondary.

The harmonized outcome is positive when the expected answer equals the
family's registered positive answer (`higher` or `greater`) and negative when
it equals the registered negative answer (`lower` or `smaller`).

## Critical counter-numeric subset

The critical subset contains queries whose source and target mechanisms have
opposite response orientations: increasing the changed quantity raises the
property in one mechanism but lowers it in the other. In this subset, matching
physical outcome requires matching cases with opposite numerical directions.
A representation that records only `increase` versus `decrease` should score
below 0.5, not above it.

Report separately:

1. all 1,080 cross-mechanism rankings;
2. 648 opposite-orientation rankings;
3. 432 same-orientation rankings;
4. all 15 unordered mechanism pairs and all nine ordered surface-variant
   combinations per pair;
5. every registered layer.

Word and character TF--IDF of the option-free stems are lexical baselines.
Future answer order is a negative control. Numerical-direction similarity is a
deliberately nonphysical baseline and should reverse in the counter-numeric
subset.

## Exact mechanism-orientation null

Exactly three of the six mechanisms have a direct response orientation.
Enumerate all `choose(6,3) = 20` assignments of direct versus inverse
orientation. For each assignment, preserve every prompt, number, variant,
mechanism, state, and similarity, and derive the corresponding positive or
negative outcome from the prompt's numerical direction and assigned mechanism
orientation.

The observed physical assignment is one of these 20 possibilities. Exact
upper-tail probabilities include the observed assignment. The same exact
enumeration is applied separately to:

- overall AUC;
- counter-numeric AUC, with the opposite-orientation subset redefined under
  each assignment;
- top-1 accuracy;
- any best-layer statistic, using the maximum over all 25 layers.

Because the null has only 20 assignments, `p=0.05` is the smallest attainable
exact value and must be described as such.

## Frozen success rule

Strong cross-mechanism evidence requires:

1. overall Jacobian AUC above 0.5 with exact `p <= 0.05`;
2. counter-numeric Jacobian AUC above 0.5 with exact `p <= 0.05`;
3. at least six of nine opposite-orientation mechanism pairs have AUC above
   0.5.

If only the overall endpoint passes, the result is numerical-direction
compatible rather than physical-outcome evidence. If only the
counter-numeric endpoint passes, the result is partial. If neither passes,
there is no cross-mechanism evidence.

A Jacobian-specific claim additionally requires the paired
mechanism-pair-bootstrap 95% interval for Jacobian minus direct to exclude zero
positively. Otherwise the result is attributed to Gemma's state geometry.

## Inference and guardrails

- Report exact orientation-null probabilities.
- Report 30,000 fixed-seed bootstrap intervals over the 15 unordered
  mechanism pairs, retaining all nine surface-variant combinations inside
  each sampled pair.
- This is a post-hoc analysis of a previously inspected cohort, with endpoints
  frozen before their computation.
- A positive result supports cross-mechanism physical-outcome alignment at
  this prompt position. It does not prove causal use, broad quantitative
  reasoning, a literal chain of thought, or a complete materials ontology.
- Every prompt, mechanism pair, variant, layer, fit, and negative result is
  retained.

## Input fingerprints

- Natural question-end representations:
  `2451d643f94934c2ea5ef73acd06dbaf1ea58f43065a724c9190423a5be9b9dc`
- Natural question-end raw record:
  `f05ccdb9be78bcb452d99edc6955e10d5c6804e9fca5a5d67ad4713fff1497e8`
- Natural question-end protocol:
  `7875f33077694e5fd774ff1b82957a33051950912dd714e4b50c9cdd47a85243`
- Natural question-end analysis implementation before this test:
  `be84b4a62580c0cf669c70a16c189e00fc3196472880f2466d2e8fda68f902b8`
