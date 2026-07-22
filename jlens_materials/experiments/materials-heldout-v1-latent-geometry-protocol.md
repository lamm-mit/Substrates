# Frozen protocol for exploratory latent geometry

Status: frozen before extraction or inspection of the held-out residual
vectors. The prompt suite and lens outputs had already been analyzed, so this
is a post-hoc exploratory endpoint rather than a preregistered primary result.

## Inputs

- The 50 frozen held-out prompts and their fixed final-prompt score positions.
- Gemma-4-E4B-it at revision
  `a4c2d58be94dda072b918d9db64ee85c8ed34e3f`.
- The 25 registered source layers and three immutable paper lens checkpoints.
- The five leading target-free candidates per family and method, ranked before
  this geometry analysis.

## Vector definitions

For prompt `p`, source layer `l`, and lens seed `s`:

- `h[p,l]` is the contextual residual at the fixed score position.
- `z[s,p,l] = final_norm(J[s,l] h[p,l])` is the Jacobian-transported
  residual in the final decoding basis.
- `zbar[p,l]` is the arithmetic mean of the three transported vectors followed
  by L2 normalization.
- `e[w]` is the corresponding normalized Gemma unembedding vector. If a
  displayed word encodes to more than one token, its token rows are averaged
  before normalization and the token count is retained.
- The lexical baseline is the normalized mean input-token embedding for each
  prompt.

All full-dimensional analyses use L2-normalized 2,560-dimensional vectors.

## Quantitative analyses

1. At every source layer, perform leave-one-phrasing-out nearest-centroid
   classification of the ten mechanism families using `zbar`. Report accuracy
   over all 50 held-out predictions.
2. Apply the same classifier to raw normalized `h`, the lexical baseline, and
   the target-layer residual.
3. Compute the ratio of mean between-family cosine distance to mean
   within-family cosine distance at every layer.
4. Compute seed spread as mean pairwise cosine distance between the three
   transported vectors for each prompt and layer.
5. Compute descriptive alignment to the frozen family-specific target-free
   word vectors as the best within-family cosine similarity minus the best
   other-family similarity.
6. Use 5,000 family-label permutations. For the depth scan, compare the
   observed maximum classification accuracy with the permutation distribution
   of the maximum across all 25 layers (plus-one correction).

## Projection

- Concatenate the 1,250 seed-mean transported states, target-layer prompt
  states, and unique frozen word vectors.
- L2 normalize, reduce to 50 principal components without whitening, and fit a
  joint UMAP using cosine distance, `n_neighbors=30`, `min_dist=0.15`, and
  random seed `20260715`.
- Repeat UMAP with seeds 0, 1, 2, 3, and 4; report 15-neighbor trustworthiness.
- Retain a two-component PCA projection as a linear sensitivity display.
- UMAP coordinates are visual only. No p-value or classification result is
  computed in two dimensions.

## Main figure decision

The geometry figure remains in the main paper only if the full-dimensional
classification exceeds the max-over-layer permutation null and the qualitative
organization is stable across UMAP seeds or visible in the PCA sensitivity.
Otherwise it moves to the Supplementary Information and is reported as a
negative exploratory result.

## Interpretation boundary

Geometric alignment indicates organization of contextual representations. It
does not establish conscious exploration, a literal reasoning trajectory, or
causal use of the decoded words.
