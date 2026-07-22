# Rigorous graph-topology extension

Frozen at `2026-07-17T18:44:02Z`, before computing any result in this
directory. The underlying representations and the earlier analyses of them
have already been inspected. This is therefore a prospectively specified
analysis of archived data, not a new confirmatory model run.

## Scientific question

Do Gemma's internal states form a reproducible graph organized by materials
mechanism after deliberately controlling for prompt wording, and do
target-free decoded words form a readable concept network? A positive result
must survive prompt-only baselines and cannot by itself establish causal use or
a literal chain of thought.

## Archived cohorts

### Unnamed-vignette cohort

- 50 prompts: ten mechanism families, five independently worded descriptions
  per family.
- 25 registered source layers and three independently fitted Jacobian lenses.
- Frozen workspace band: 38--92% depth.

### Disjoint signed-relation cohort

- 72 prompts from six mechanisms.
- Four material cases per mechanism, with an anchor, a physically equivalent
  paraphrase, and a near-verbatim reversed-relation counterfactual.
- This cohort was generated independently of the unnamed-vignette graph
  analysis. Its earlier registered late-window similarity replication failed;
  that negative result is retained.

## Analysis 1: cross-phrasing mechanism graph

For each unnamed vignette and each of the four other phrasing folds, select
exactly one nearest prompt in that fold. This produces 200 directed edges per
representation and prevents a prompt from satisfying the endpoint by linking
only to one favored phrasing style.

- Primary representation: mean pairwise cosine similarity over the frozen
  38--92% band after averaging the three Jacobian fits.
- Primary metric: fraction of the 200 directed cross-phrasing edges joining
  the same mechanism family.
- Baselines: raw residual states, final target state, mean input-token
  embedding, word TF--IDF, character 3--5-gram TF--IDF, target-free direct
  words, and target-free Jacobian words.
- Layerwise curves are secondary. Any reported layer maximum is corrected
  against the maximum of the same statistic across all 25 layers in every
  null permutation.

The primary null independently permutes the ten balanced mechanism labels
within each phrasing fold. Report plus-one one-sided p-values from 50,000
permutations. Report family-level values and exact family sign-flip tests for
paired Jacobian-minus-baseline contrasts; the ten families, not the 200 edges,
are the independent population units.

## Analysis 2: prompt-only residualization

At each layer, regress every off-diagonal pairwise state cosine on prompt-only
features without using mechanism labels:

1. word TF--IDF cosine;
2. character 3--5-gram TF--IDF cosine;
3. token-set Jaccard similarity;
4. absolute token-count difference;
5. same phrasing-fold indicator.

Construct the same cross-phrasing graph from the symmetric residual similarity
matrix. The primary residualized endpoint is the 38--92% band consensus graph;
layerwise results and a max-over-layer corrected permutation test are
secondary. A dyadic partial coefficient for same-mechanism membership, with
the same prompt-only covariates, is reported as a non-graph corroboration.

## Analysis 3: frozen lexical hard negatives

For each query and each other phrasing fold, the true same-mechanism target is
compared with the different-mechanism prompt having the largest average of
word and character TF--IDF cosine. The hard negative is selected without
consulting any model state.

Report the state-similarity margin between the true target and this lexical
competitor, averaged first within mechanism family. Report all 200 contrasts
and the subset in which the prompt-only score itself favors the wrong
mechanism. Use a family bootstrap and an exact family sign-flip test.

## Analysis 4: edge stability across layers and lens fits

For every permitted directed cross-phrasing pair, count the fraction of
registered band layers and lens fits in which that pair is selected. Evaluate
whether same-mechanism edges have larger stability than other edges using
ROC--AUC and the blocked label-permutation null. Compare the Jacobian ensemble
with raw-state stability. This endpoint tests repeated graph assembly rather
than one selected layer.

## Analysis 5: disjoint within-mechanism relation graph

Within each of the six disjoint mechanisms, connect every prompt to its nearest
prompt in each other surface variant, excluding the same material-case
triplet. This produces a graph that cannot earn credit merely by identifying
the mechanism. The endpoint is the fraction of directed edges joining prompts
with the same physically correct outcome direction.

Outcome labels are independently permuted within mechanism and surface
variant, preserving the balanced outcome counts. Report:

- the 38--92% band-consensus graph;
- the previously frozen 80--96% late window;
- layerwise curves with max-over-layer correction;
- raw, direct-decoder-basis, Jacobian, word TF--IDF, and character TF--IDF
  results.

Because this analysis is performed after the representations and prior
triplet results were inspected, it is a robustness/generalization analysis,
not a confirmatory replication.

## Analysis 6: target-free concept network

Use only the already archived three-lens-consensus candidate words. Remove
exact prompt words and conservative prefix-related variants, and retain the
existing frozen function-word filter. No predeclared mechanism term is used to
generate, filter, retain, or rank a word.

Create a prompt--word bipartite graph from consensus score times inverse
document frequency. For every word, test family concentration against the
same 50,000 blocked label permutations. Control the word-level false discovery
rate by Benjamini--Hochberg at 0.05. The visualization may show only
FDR-significant words and must display all ten families, including those with
no significant word.

## Robustness requirements

- Repeat cross-phrasing results for one, two, and three neighbors per target
  fold where meaningful, and for each lens fit separately.
- Report union and mutual undirected graph sensitivity for the primary
  similarity matrix.
- Report exact prompt and input fingerprints.
- Retain all families, prompts, layers, and negative results.
- Separate descriptive label-free community detection from the primary
  family-labeled evaluation.
- Do not claim that a graph proves consciousness, private prose, a literal
  reasoning trace, or causal use.
