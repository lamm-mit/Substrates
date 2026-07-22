# Frozen exact graph-partition follow-up

Frozen: `2026-07-17T21:55:00-04:00`

> Publication note: this follow-up was later reported in the paper's
> Supplementary Information. Statements below about manuscript scope preserve
> the protocol's status when it was frozen.

## Motivation and status

The frozen spectral density ablation found that the complete graded candidate
graph preserves substantially more relation structure than the registered
top-one sparsification. This follow-up was specified before inspecting any
exact balanced-partition or held-out-counterfactual partition result. It asks
whether the graded affinity graph has a globally coherent bipartition rather
than only above-chance edgewise ranks.

This is an exploratory graph-theory analysis. It remains outside the
manuscript. Physical same/different and numeric-direction same/different are
exactly identical in this prompt cohort, so any recovered community is a
direction-relation community and cannot independently prove physics.

## Study 6A: exact balanced affinity partition

For each mechanism and representation, average cosine similarities over the
frozen 38--92% band. Retain every registered different-case, cross-variant
candidate edge and symmetrize its nonnegative weight. Enumerate all unique
balanced 6-versus-6 bipartitions of the 12 nodes, fixing node zero in the first
community to remove complementary duplicates.

For each partition, score:

`mean observed affinity within communities - mean observed affinity between communities`.

Only eligible candidate entries enter either mean; structural zeros do not.
Choose the maximum-score partition with lexicographic tie breaking. Reveal
labels only after selection and report adjusted Rand index, normalized mutual
information, gauge-adjusted accuracy, best score, second-best score, and
margin.

## Study 6B: held-out-surface transfer

Fit an exact balanced 4-versus-4 partition using only the eight anchor and
physics-paraphrase nodes and their eligible cross-variant affinities. The four
lexical-counterfactual nodes are absent from fitting.

For each counterfactual node, compare its mean eligible affinity to each fitted
community and assign it to the community with larger mean; ties go to the first
community. Evaluate the resulting 12-node partition up to one global
community-name swap. This tests whether a relation assembled without the
counterfactual nodes extends to a new surface form.

## Nulls and controls

For 10,000 null sets per representation, independently permute weights within
each directed source-variant to target-variant block while preserving:

- the complete eligible graph;
- every mechanism, case, and surface-variant position;
- the weight multiset of each ordered variant block; and
- all missing-edge structure.

Recompute the exact partition after every shuffle. The primary statistics are
the six-mechanism mean adjusted Rand index for Studies 6A and 6B, each with a
plus-one upper-tail exact Monte Carlo p-value.

Repeat Jacobian, direct, and raw states. Strong dense-partition evidence
requires positive adjusted Rand index and gauge accuracy above 0.5 in all six
mechanisms plus `p <= 0.05` for both complete and held-out-surface endpoints.
A Jacobian-specific claim additionally requires corrected superiority to both
direct and raw controls in at least five of six mechanisms.

## Stability

For each observed mechanism, run 1,000 fixed-seed stratified edge bootstraps
within ordered variant blocks. Record partition co-assignment probability,
adjusted Rand index to the unbootstrapped partition, and the fraction of
bootstrap partitions identical up to complement. Bootstrap replicates are
stability diagnostics, not inferential units.

## Interpretation

Passing Study 6A but failing Study 6B means graded affinities contain a
retrospective partition that does not transfer to unseen phrasing. Passing
both without Jacobian specificity supports a general representation-level
direction relation. Failure against blockwise weight permutation means the
dense spectral result was not a coherent community.
