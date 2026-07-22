# Frozen option-free question-end robustness experiment

Frozen: `2026-07-18T00:35:00-04:00`

## Status

This protocol was frozen after the archived `checkpoint` graph failed its
option-free success rule. It is a prospectively frozen new model run but not
an independent replication: it reuses the same 72 scientific stems, model,
lenses, layers, candidate sets, labels, and graph endpoint. Its sole purpose is
to test whether the common words `Internal checkpoint` or the position on that
marker caused the negative result.

## Question

Does signed-relation topology appear at the natural end of a complete
scientific question when no answer choices, answer words, arbitrary code,
response instruction, or checkpoint marker are present?

## Prompt construction

For every row of the frozen 72-prompt answer-code manifest, use only the exact
`stem` field. Format that stem as a complete Gemma chat user turn. Do not append
or prepend any experimental language. Capture the final prompt-position
residual after the complete chat-formatted question.

All 72 stems are retained:

- six mechanisms;
- four material cases per mechanism;
- anchor, physics paraphrase, and near-verbatim lexical counterfactual for
  every case;
- balanced positive and negative physical outcomes.

## Representations

- Model: `google/gemma-4-E4B-it`, immutable revision
  `a4c2d58be94dda072b918d9db64ee85c8ed34e3f`.
- The same 25 registered source layers.
- Raw residual, direct decoder-basis, and each of three frozen Jacobian
  decoder-basis representations.
- `bfloat16` model inference; states archived as `float16`.
- Primary Jacobian similarity: average the three fit-specific cosine matrices
  at each layer, then average the frozen 38--92% depth band.

## Frozen analysis

Apply the graph construction, exact 46,656-assignment case-preserving null,
full-candidate pairwise AUC, family breadth, layer scan, and
Jacobian/direct/raw comparisons specified in
`experiments/option-free-relation-graph-2026-07-17/PROTOCOL.md`.

Primary endpoints:

1. 144-edge same-outcome precision.
2. Full-candidate pairwise AUC.

Strong evidence requires both exact structured-null `p <= 0.05`, AUC above
0.5, and at least four of six family AUCs above 0.5. Exactly one passing
endpoint, or insufficient breadth, is partial evidence. Neither passing is no
evidence. A Jacobian-specific claim additionally requires a positive paired
family-level 95% interval for Jacobian minus direct.

## Decision consequences

- If this natural question-end experiment is positive, the archived
  checkpoint failure is treated as position/marker sensitivity and both
  results are retained.
- If it also fails, the existing final-state signed-relation graph is
  explicitly described as answer-conditioned decision geometry rather than
  pre-scaffold signed-relation organization.

No layer, prompt, family, fit, or behavior is excluded. The model's leading
next tokens are archived descriptively but are not an inferential endpoint.

## Fingerprints

- Runner:
  `157ef2d9697c97f275beede0413ec7dc2b4162a11be324d506a3106d6665bfdd`
- Source manifest:
  `9ec1bdc3d12960cdcb2538bd214a1771d9f45ce5d1243fa4a2ab2ff3f83d8de5`
- Lens seed 0:
  `d15ff55233c458f4289a7aac1b3f5c8e6441d0334a44a7b6fce03e447889aa99`
- Lens seed 1:
  `98bf7c7491c525df5ae9c9ac8040f450cce630dc8257a2ae062e6bdbf76980dd`
- Lens seed 2:
  `51930e2b8d751de78e66ed92fcf6c1724783a4f81f94d0b7021d2278aabe00e5`
