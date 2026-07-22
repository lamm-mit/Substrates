# Frozen multi-token sequence robustness

Frozen: `2026-07-18T02:15:00-04:00`

## Status and motivation

This is a prospectively frozen new scoring run on two families from the
already inspected 50-prompt held-out cohort. It is a post-hoc robustness
analysis, not a replacement for the preregistered single-token endpoint.

The preflight excluded `transgranular` and `martensite` because neither is one
Gemma vocabulary token. Ranking only the first fragments (`trans` or `mart`)
would not test the intended technical concept.

## Scientific question

When the exact predeclared technical word contains multiple tokenizer pieces,
does an intermediate Jacobian readout favor that complete word over an
equal-piece, scientifically relevant alternative?

## Exact prompts and contrasts

All five frozen prompts in each affected family are retained:

- cleavage: `transgranular` versus `intergranular`;
- rapid transformation: `martensite` versus `bainite`.

Targets and contrasts are absent from every corresponding prompt. The
contrasts were selected before scoring because they are scientifically
meaningful alternatives and have exactly the same number of Gemma pieces as
their targets.

Frozen tokenizer resolution:

- `transgranular`: `[3849, 198597]` = `trans`, `granular`;
- `intergranular`: `[2266, 198597]` = `inter`, `granular`;
- `martensite`: `[26348, 832, 785]` = `mart`, `ens`, `ite`;
- `bainite`: `[236763, 662, 785]` = `b`, `ain`, `ite`.

## Sequence scoring

A Jacobian or direct lens reads a next-token distribution from one
intermediate state. It cannot legitimately score every piece of a word from
that same state.

For each candidate word:

1. read the first piece from the registered intermediate state using direct
   decoding or one of the three Jacobian transports;
2. append that piece to the unchanged prompt;
3. teacher-force all remaining pieces through the unchanged Gemma model;
4. add the first-piece logit and remaining-piece conditional log
   probabilities.

Target and contrast contain the same number of pieces. Their first-token
softmax denominator is shared within a prompt/layer/readout and cancels in the
target-minus-contrast difference. The resulting margin is therefore an exact
restricted sequence log-odds ratio even though an unnecessary full-vocabulary
normalization is not calculated for the first piece.

## Frozen endpoints

- Primary: target-minus-contrast sequence margin averaged over the registered
  38--92% depth band, first across layers and then across the five prompts in
  each family.
- Report every prompt, layer, fit, first-piece margin, continuation
  contribution, and full sequence margin.
- Compare the three-fit Jacobian ensemble with direct decoding.
- Report the three Jacobian fits separately.

Descriptive robustness passes if:

1. both family-level Jacobian sequence margins are positive;
2. at least 8/10 prompt-level Jacobian band margins are positive;
3. every Jacobian fit has a positive family-level margin in both families.

A Jacobian-specific descriptive gain is claimed only if the
Jacobian-minus-direct family mean is positive in both families. With two
families, no population p-value or broad multi-token generalization claim is
permitted.

## Guardrails

- This is a two-term targeted robustness analysis.
- It does not put multi-token terms into the original global vocabulary-rank
  endpoint.
- It does not imply that all multi-token scientific terminology is recovered.
- The scientifically relevant contrasts make this a harder specificity test
  than scoring arbitrary strings, but they do not exhaust all alternative
  phases or fracture paths.
- All prompts, layers, fits, and negative results are retained.

## Fingerprints

- Runner:
  `276374b814a347a40c53738ea71db90164e32ce33e1bb95aa2dacfe11f694ce1`
- Held-out manifest:
  `8c034cf33d287d379fddf842971914ec035a22b9e31d29f457258bf85c52e203`
- Lens seed 0:
  `d15ff55233c458f4289a7aac1b3f5c8e6441d0334a44a7b6fce03e447889aa99`
- Lens seed 1:
  `98bf7c7491c525df5ae9c9ac8040f450cce630dc8257a2ae062e6bdbf76980dd`
- Lens seed 2:
  `51930e2b8d751de78e66ed92fcf6c1724783a4f81f94d0b7021d2278aabe00e5`
