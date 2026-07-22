# Materials held-out v1 preregistration

Frozen on 2026-07-14 before executing or inspecting any held-out lens output.

## Scientific purpose

This experiment tests whether the three paper-protocol Gemma-4 Jacobian lenses
generalize from the already inspected 50-prompt development suite to 50 new
materials descriptions. It deliberately separates two questions:

1. **Predetermined concepts:** does a physical term specified before execution
   become readable in the full vocabulary?
2. **Open-vocabulary discovery:** what input/output-absent tokens appear without
   providing a candidate list, and are those tokens more scientifically
   identifiable under the Jacobian lens than under direct unembedding?

## Frozen artifacts

- Generator: `scripts/generate_materials_heldout_v1.py`
- Manifest: `prompts/materials-heldout-v1-preregistered.json`
- Design: ten mechanism families by five new phrasings, 50 prompts total.
- Model: `google/gemma-4-E4B-it` at revision
  `a4c2d58be94dda072b918d9db64ee85c8ed34e3f`.
- Lenses: seeds 0, 1, and 2 from repository
  `lamm-mit/gemma4-jacobian-lenses` at revision
  `37ba15033a42e72bdfbc04815b3fbd37e516fd59`.
- Fixed depth band: 38--92%.
- Fixed readout position: final prompt token.
- Candidate retention: 32 fixed-position tokens per layer and 64 whole-prompt
  open-vocabulary candidates per lens.
- Clean generation: one greedy token, used for output-absence checking.

The frozen hashes, inserted before any held-out model or lens output was
inspected, are:

- Generator SHA-256:
  `22c2e8dcf0f8788db74e4458e9c1be48c2349af24a5f376f6aaad5736286d9f9`
- Manifest SHA-256:
  `8c034cf33d287d379fddf842971914ec035a22b9e31d29f457258bf85c52e203`

The tokenizer preflight is retained in
`experiments/materials-heldout-v1-preflight.json`. It found 50 unique prompts,
ten families with five prompts each, no exact development-prompt duplicates,
no tokenizer-resolved target leakage into any input, and a maximum pairwise
development/held-out word-5-gram Jaccard similarity of 0.0952. As registered,
`transgranular` and `martensite` do not resolve to a single Gemma token and are
documented but omitted from single-token rank endpoints.

## Inclusion and exclusions

- Every tokenizer-resolved predetermined concept must be absent from the input.
- The generated one-token continuation must not contain a tokenizer-resolved
  predetermined concept.
- Any prompt with a design violation is excluded identically for all lens seeds.
- Multi-token concepts remain documented but are excluded from single-token
  rank endpoints exactly as in the development study.
- No prompt, concept, band, filter, candidate cutoff, or primary endpoint may be
  changed after inspecting outputs.

## H1 Predetermined concept recovery

For each prompt and lens-fit seed, compute the best full-vocabulary Jacobian
rank for every valid predeclared concept across the fixed source-layer band.
Compute the matched direct-unembedding rank over the same layers, position, and
band.

The prompt-level endpoint is log-k pass-at-k AUC for
`k = {1, 2, 5, 10, 20, 50, 100}`. Average the three Jacobian lens seeds within
prompt before population inference. The primary contrast is mean Jacobian AUC
minus logit-lens AUC. Report a hierarchical bootstrap that resamples mechanism
families and then phrasings, plus an exact family-level sign-flip test.

Lens seeds are repeated measurements, not independent prompt samples.

## H2 Seed reproducibility

Compute pairwise Spearman correlations across all valid prompt-concept
full-vocabulary ranks for the three Jacobian lens fits. Report family-clustered
bootstrap intervals. This endpoint measures stability, not validity or causal
use.

## H3 Open-vocabulary family identification

Candidate generation cannot use the predetermined concept list. For each
prompt and lens method:

1. scan unrestricted top-1 decoded tokens across all prompt positions and
   sampled layers in the 38--92% band;
2. exclude tokens present in the input or generated continuation;
3. retain the top 64 candidates per seed;
4. retain a candidate at the prompt level only when it appears under all three
   lens-fit seeds;
5. remove the frozen target-agnostic English function-word list;
6. rank family candidates by the mean seed-consensus score multiplied by
   `log(50 / global prompt frequency)`, averaged over five phrasings.

Apply the identical procedure to the matched logit-lens readouts stored during
the same forward evaluations.

The primary semantic endpoint is blinded mechanism-family identification from
candidate words alone. Candidate sets are shuffled and lens identity is hidden.
At least three materials-science raters should classify all Jacobian and direct
sets among the ten frozen family labels. Report majority-vote accuracy,
individual accuracy, inter-rater agreement, a shuffled-label null, and paired
family-level Jacobian-minus-direct accuracy.

An automated language-model rater may be reported as a reproducible secondary
analysis but cannot be described as a human expert.

## H4 Relationship between controlled and discovered readouts

At the prompt-family level, correlate predetermined concept recovery with
open-vocabulary identification success using a family-blocked Spearman
bootstrap. A positive result would indicate that the controlled and discovery
views capture related structure. A null result would imply that they expose
different aspects of the representation.

## Interpretation boundaries

- Open-vocabulary outputs are semantic readouts, not a literal chain of thought.
- Seed consensus is a stability control, not evidence of human-like
  understanding.
- Predetermined-term overlap is annotated only after discovery ranking and is
  not a discovery endpoint.
- No causal or reasoning claim follows from ranking or classification alone.

## Registered execution pattern

From the `jlens_materials` directory, run once for each seed:

```bash
python run_lens.py \
  --model google/gemma-4-E4B-it \
  --model-revision a4c2d58be94dda072b918d9db64ee85c8ed34e3f \
  --lens lenses/lens_gemma4-e4b-it.paper.seedSEED.pt \
  --recipe paper --dtype bfloat16 \
  --tag gemma4-e4b-it-heldout-v1-seedSEED \
  --prompts prompts/materials-heldout-v1-preregistered.json \
  --shapes ASSOCIATION \
  --workspace-band 38,92 \
  --generation-max-new-tokens 1 \
  --layer-readout-top 32 \
  --surprising-top 64 \
  --open-vocab-logit-baseline
```

Any operational change needed solely for device compatibility must be recorded
and may not alter prompts, scoring, or endpoints.
