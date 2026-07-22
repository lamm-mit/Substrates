# Candidate non-steering analysis protocol

> Publication note: these retrospective analyses were later reported with
> their exploratory status in the Supplementary Information. The status below
> records when the protocol was frozen.

Frozen at 2026-07-16T22:10:18-04:00, before calculating any output from the
analyses below. These are retrospective candidate analyses of already known
paper data. They are not confirmatory endpoints and must not enter the
manuscript until reviewed.

## Fingerprinted inputs

- Held-out statistics: `5f3904...a54d`
- Latent-vector archive: `687bf1...791b0`
- Latent-vector metadata: `7b4d7c...671a`
- Held-out raw fits 0/1/2: `829778...d46e`, `6397e3...169c`,
  `ae7d9e...d1c8`

Full hashes are in `protocol.json`.

## Analysis A: target-free semantic classification after lexical removal

Each of 50 prompts is represented by its complete filtered three-fit consensus
candidate list from the frozen target-free analysis. Candidate weights equal
the stored consensus score times an inverse-document-frequency factor computed
only from the 40 training prompts in each fold. The vocabulary is likewise
defined from training prompts only. A cosine nearest-centroid classifier trains
on four phrasings per family and tests the fifth; the five existing phrasing
indices are the folds. No declared concept term is used.

Run both Jacobian and direct candidate features under three filters:

1. `all`: the paper's target-agnostic function-word filter only;
2. `prompt_exact_removed`: additionally remove a candidate appearing as an
   exact lowercase alphabetic word in that prompt;
3. `prompt_morphology_removed`: additionally remove exact matches and any
   candidate/input pair of length at least five for which either token begins
   with the other. This deliberately conservative rule removes simple variants
   such as `oxidation`/`oxide` only when the prefix condition actually holds.

Baselines are a training-fold TF-IDF bag of prompt words and a label shuffle.
Report accuracy, macro-F1, confusion matrices, paired correctness versus direct,
per-family accuracy, and 10,000 label permutations with seed 20260716. Each
permutation independently shuffles the ten balanced family labels within every
phrasing fold, preserving fold composition. The family label is the independent
inference unit for bootstrap intervals; 30,000 family resamples use the same
seed. This analysis asks whether target-free vocabulary carries mechanism
identity beyond exact words copied from the prompt. It does not prove that the
remaining candidates are causally used.

## Analysis B: materials-ontology representational similarity

At every registered layer, average the three L2-normalized Jacobian-transported
states and renormalize, then average the five prompt states within each family
and renormalize. Form the 45 pairwise cosine distances among ten family
centroids. Repeat for raw layer states. The target-layer and mean-input-embedding
baselines each provide one 45-entry distance vector.

Compare neural distances with two prespecified expert structures:

- `response_class`: families share a class when their principal response is
  fracture/damage, deformation/strengthening, transformation, or degradation;
- `multi_attribute`: Jaccard distance over the binary materials attributes in
  `protocol.json`, covering driving domain, structural carrier, and response.

Use Spearman correlation. For layerwise methods, the primary null randomly
permutes the mapping between ten family names and expert rows, recomputes all 25
layers, and retains the largest absolute correlation. Use all 10! mappings only
if computationally cheap; otherwise use 30,000 unique seeded permutations plus
the observed mapping. Single-vector baselines use the same mappings without a
layer maximum. Report layer curves, corrected p-values, and leave-one-family-out
correlations. Positive alignment means that geometrically similar mechanism
families share the frozen materials attributes; it does not show a discovered
or unique scientific ontology.

## Analysis C: scientific abstraction and layer onset

The 29 declared single-token concepts are assigned before calculation to three
coarse roles in `protocol.json`: physical entity/state, physical process or
mechanism, and response/property/mode. Within the matched 38--92% band stored
for both readouts, for every prompt-concept, fit, and readout, record best depth
and first depth with at least two consecutive sampled layers in the top 1,000.
Summarize fit-averaged prompt-concept units. Compare
roles using family-clustered bootstrap intervals and a family-blocked
permutation of role labels, retaining the maximum pairwise median-depth
difference across the three role pairs. Missing sustained events are reported
as missing and as a separate event rate; they are never assigned an artificial
late depth. The analysis is descriptive because word frequency and tokenizer
properties can affect onset.

## Decision rules

- Prefer Analysis A for the main paper only if prompt-decontaminated Jacobian
  vocabulary classifies substantially above the corrected null and its gain is
  not carried by one family.
- Prefer Analysis B only if alignment survives max-layer correction and
  leave-one-family-out sensitivity under both expert structures.
- Keep Analysis C in the Supplementary Information unless role ordering is
  consistent for Jacobian and direct readouts and event rates are adequate.
- Negative results are retained. No analysis will be tuned by dropping a family,
  changing the ontology, or changing lexical filters after output inspection.
