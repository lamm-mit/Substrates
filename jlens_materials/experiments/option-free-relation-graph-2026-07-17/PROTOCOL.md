# Frozen option-free signed-relation graph

Frozen: `2026-07-17T23:55:00-04:00`

## Status and motivation

This is a newly frozen analysis of archived states, not a prospective model
experiment. The 72 prompts and representations were generated for the
prospectively frozen arbitrary answer-code study, and its registered endpoint
results were inspected before this graph analysis was designed. No
option-free graph statistic had been calculated when this protocol was
written.

The existing signed-relation graph reads the final state of a prompt that
already contains semantic answer choices such as `higher` and `lower`.
Consequently, that graph may partly reflect an intended answer rather than a
physical relation. The archived answer-code study also saved a `checkpoint`
state after the complete scientific question and the common words `Internal
checkpoint`, but before any answer words, A/B mapping, or response instruction
appears. That state cannot attend to the future answer scaffold.

## Scientific question

Before an answer vocabulary is supplied, do Gemma states place different
material cases with the same signed physical consequence nearer to one another
than cases with the opposite consequence?

## Cohort and frozen representations

- 72 prompts from six mechanisms.
- Four material cases per mechanism.
- Three surface variants per case: anchor, physics paraphrase, and
  near-verbatim lexical counterfactual.
- 25 registered layers.
- Raw residual states, direct decoder-basis states, and three independently
  fitted Jacobian decoder-basis states.
- Primary position: `checkpoint`, after the complete scientific question and
  before the answer mapping.
- Diagnostic comparison: `final_prompt`, after the arbitrary A/B mapping.

The primary Jacobian similarity is the mean of the three fit-specific cosine
similarity matrices, followed by averaging over the frozen 38--92% depth band.
The previously frozen 80--96% late band is secondary. Raw and direct
representations use the identical positions, layers, candidate sets, and
similarity calculation.

## Graph construction

The mechanism family is supplied and is not an inferred endpoint. For every
source prompt, select its nearest eligible target from each of the other two
surface-variant groups. Eligible targets:

1. belong to the same mechanism family;
2. use a different surface variant;
3. describe a different material-case triplet.

This gives 144 directed edges. A correct edge joins prompts with the same
registered physical outcome (`higher`/`lower` or `greater`/`smaller`).

## Primary and corroborating endpoints

1. **Primary graph precision:** fraction of the 144 checkpoint Jacobian-band
   edges preserving physical outcome.
2. **Full-candidate pairwise AUC:** for every source/target-variant query,
   compare every eligible same-outcome candidate with every eligible
   opposite-outcome candidate. This prevents one favorable nearest neighbor
   from determining the result.
3. **Family breadth:** graph precision and candidate AUC in each of the six
   mechanisms.
4. **Layer trajectory:** graph precision and candidate AUC at all 25 layers.
   Any best-layer statement is corrected against the maximum across all
   layers under each null assignment.
5. **Representation comparison:** Jacobian, direct decoder-basis, and raw
   states. A Jacobian-specific advantage is claimed only from paired
   family-level contrasts, not from separate significance tests.
6. **Scaffold comparison:** checkpoint versus final-prompt results within the
   same archived answer-code prompts. This is descriptive because position,
   suffix, and visible answer mapping change together.

Word and character TF--IDF of the option-free scientific stems are lexical
baselines. Future answer order is a negative control. Numerical change
direction is a labeled oracle, not a model baseline.

## Exact structured null

For each mechanism independently, choose two of its four material cases to
carry the positive anchor/paraphrase outcome and assign the other two the
negative outcome. The counterfactual outcome is always the opposite of its
case's anchor. Enumerating all six balanced assignments in each of six
families gives `6^6 = 46,656` exact structured-null labelings. This preserves:

- the four-case balance within every mechanism;
- the anchor/paraphrase agreement;
- the counterfactual sign reversal;
- the complete graph and all similarities.

The primary p-value is the upper-tail exact probability including the observed
or more extreme statistic. Candidate AUC and best-layer statistics use the
same exact assignments. Family uncertainty is also summarized by a
leave-one-family-out range and a fixed-seed two-stage family/case bootstrap.

## Frozen interpretation rule

- **Strong option-free evidence:** both primary graph precision and
  full-candidate AUC have exact structured-null `p <= 0.05`, candidate AUC is
  greater than 0.5, and at least four of six family AUCs exceed 0.5.
- **Partial evidence:** exactly one of graph precision or candidate AUC passes,
  or both pass but fewer than four families have AUC above 0.5.
- **No option-free evidence:** neither primary endpoint passes.

A Jacobian-specific claim additionally requires the two-stage 95% interval
for the paired Jacobian-minus-direct family contrast to exclude zero in the
positive direction. Otherwise the finding is attributed to Gemma's state
geometry rather than uniquely to the Jacobian transport.

## Guardrails

- This is a post-hoc analysis of previously generated and inspected arrays.
- The common words `Internal checkpoint` occur before the captured state, but
  answer words and mappings do not.
- A positive result supports option-free signed-relation organization within
  supplied mechanism families. It does not establish a complete materials
  ontology, a literal chain of thought, consciousness, or causal use.
- A null result will be retained and will narrow the existing final-state
  graph to answer-conditioned decision geometry.
- No prompt, layer, family, fit, edge, or negative result may be removed.

## Frozen input fingerprints

- Representations:
  `aa226a72f47c4758290627e00908f8e2a85a0de4c5d16ec41551e3da89092a63`
- Prompt manifest:
  `9ec1bdc3d12960cdcb2538bd214a1771d9f45ce5d1243fa4a2ab2ff3f83d8de5`
- Source answer-code protocol:
  `41b19f84e31cc123cc90e98ca6058cd269addf83f433c0c7a519cf1724df08c3`
- Reused graph implementation before this analysis:
  `4737adf18db3c2eba6fd4cf01da8f43c660abb910d49b757c890c0c2be65540e`
