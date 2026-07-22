# Frozen spectral graph follow-up

Frozen: `2026-07-17T21:46:00-04:00`

> Publication note: this follow-up was later reported in the paper's
> Supplementary Information. Statements below about manuscript scope preserve
> the protocol's status when it was frozen.

## Motivation and status

Studies 3 and 4C showed that supervised GINs do not generalize absolute or
pairwise outcomes to a wholly unseen mechanism. This follow-up asks a
different, purely graph-theoretic question before inspecting any spectral
partition: does each mechanism graph itself contain an unlabeled two-community
partition corresponding to the positive/negative relation, up to an arbitrary
community-name swap?

The analysis is exploratory with frozen endpoints. It does not alter the
manuscript. The exact identity
`physical_relation == numeric_direction_relation` in the current prompts is
already known and remains a mandatory interpretive limit.

## Primary spectral endpoint

For each representation and mechanism, average cosine similarity over the
frozen 38--92% band and build the registered directed graph with one selected
edge per source into each other surface-variant group. Symmetrize by summing
the two directed weighted adjacencies. Negative weights, if any, are clipped
to zero. Compute the normalized graph Laplacian and its Fiedler vector. Split
the 12 nodes into exactly six and six at the stable rank midpoint; eigenvector
sign and tied ranks use node-index order.

No label enters graph construction or partitioning. After the partition is
fixed, reveal physical labels and report:

- adjusted Rand index (primary);
- normalized mutual information;
- gauge-adjusted accuracy, the better accuracy after allowing one global
  community-name swap; and
- ordinary accuracy under the deterministic eigenvector sign, as a diagnostic
  only.

## Constrained null

For each of 10,000 null sets and each mechanism, preserve 12 nodes, two
outgoing edges, target surface-variant groups, same-case exclusion, and the
observed 24-edge weight multiset. Randomize eligible targets and independently
permute weights. The primary null statistic is mean adjusted Rand index over
the six mechanisms. Use a plus-one upper-tail exact Monte Carlo p-value.

Strong spectral-community evidence requires positive adjusted Rand index and
gauge-adjusted accuracy above 0.5 in all six mechanisms, plus primary null
`p <= 0.05`.

## Representation and information controls

Repeat the complete endpoint and null for Jacobian, direct, and raw states. A
Jacobian-specific claim requires Jacobian-minus-direct and
Jacobian-minus-raw adjusted Rand index above zero in at least five of six
mechanisms and plus-one family sign-flip `p <= 0.05` for both contrasts.

The physical, numerical-direction, and prompt-derived relation partitions are
reported side by side. If physical and numerical partitions are identical,
community recovery is described as direction-structure recovery and not as
independent physical understanding.

## Frozen density and weight ablations

Repeat spectral partitioning for:

1. binary registered edges;
2. weighted top-one edge per other variant (primary);
3. weighted top-two edges per other variant; and
4. the complete weighted different-case, cross-variant candidate graph.

For the complete graph, use nonnegative cosine weights and no learned
temperature. These ablations determine whether graph sparsification discards
graded relational information. They are secondary, with Benjamini--Hochberg
correction across representation-by-ablation tests.

At every registered layer, also record the primary weighted partition and its
label-free normalized-Laplacian eigengap. Layer scans are descriptive;
mechanisms, not layers, are inferential units.

## Synthetic positive and negative controls

Generate 1,000 fixed-seed, balanced 12-node two-block graphs with the same
variant and case constraints.

- Positive control: eligible same-block edges receive a `+1.0` affinity offset
  before top-one selection.
- Negative control: affinities are independent of block identity.
- For both controls, independently flip all node labels in every graph with
  probability one half.

The spectral method must recover the positive-control partition up to a global
flip (mean adjusted Rand index at least 0.8), remain at chance for the negative
control (absolute mean adjusted Rand index at most 0.05), and show that
ordinary signed accuracy is not a stable endpoint under random graph-wide
flips. Failure invalidates spectral conclusions.

## Interpretation

A passing spectral endpoint with failed Jacobian specificity supports a
representation-wide relational community, not a Jacobian-specific scientific
graph. A passing endpoint that is identical to the numeric-direction
partition supports a direction-sensitive latent organization but cannot
separate physics from explicit comparative wording. A failed endpoint means
that above-chance edge-level AUC does not assemble into a recoverable global
community.
