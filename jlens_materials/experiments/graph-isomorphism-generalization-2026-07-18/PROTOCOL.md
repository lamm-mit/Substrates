# Frozen graph-isomorphism generalization study

Frozen: `2026-07-17T20:48:17-04:00`

> Publication note: this study was later reported in the paper's
> Supplementary Information. Statements below about manuscript scope preserve
> the protocol's status when it was frozen.

## Status and scope

This is a new post-hoc analysis of a previously executed, option-free model
run. The hidden-state arrays, 72 scientific questions, six mechanism families,
25 layers, three Jacobian fits, and physical labels were already inspected in
earlier work. No new Gemma output is generated. The graph-isomorphism,
cross-family mapping, and graph-neural-network endpoints below were specified
before any of those endpoints were calculated.

The study is exploratory with frozen tests. It can identify a transferable
structural regularity in this cohort, but it cannot by itself establish a
universal materials graph. The inferential unit is always the governing
mechanism or unordered mechanism pair, never a layer, node, edge, training
epoch, or random seed.

The manuscript and Supplementary Information are out of scope. All outputs
remain in this experiment directory until reviewed with the investigator.

## Frozen inputs

- Option-free final-question-token representations:
  `experiments/option-free-question-end-2026-07-18/representations.npz`
  (SHA-256
  `2451d643f94934c2ea5ef73acd06dbaf1ea58f43065a724c9190423a5be9b9dc`).
- Exact 72-prompt manifest:
  `experiments/late-physics-representation-replication-2026-07-17/prompt_manifest.json`
  (SHA-256
  `da49f9d57b870404beabb7401f40a677d90d0b93d49d58170a0ab9750268d634`).
- The frozen source layers and 38--92% primary depth band from the option-free
  study.
- Representations: the mean of the three fit-specific Jacobian cosine
  matrices, direct decoder-basis states, and raw residual states.

Every family contains four material cases and three surface variants per case:
anchor, physical paraphrase, and near-verbatim counterfactual. Anchor outcomes
are balanced two positive and two negative within every family; each
counterfactual reverses its case's physical outcome.

## Question

Do independently governed materials mechanisms instantiate a common relational
graph template that can be recovered without using mechanism names, prompt
text, physical labels, answer words, or numerical values during graph
alignment, and does that template generalize to a completely held-out
mechanism?

## Study 1: label-blind cross-mechanism graph matching

For each mechanism and representation, calculate cosine similarity at every
registered layer and in the frozen 38--92% band. For every unordered pair of
mechanisms, consider all 24 bijections between their four material cases.

### Fitting the case map

The case bijection is fitted using only anchor and physical-paraphrase nodes.
The objective is Pearson correlation between the two mechanisms' eligible
anchor-to-paraphrase similarity vectors after mapping. Only different-case
entries are eligible, matching the graph candidate rule. Physical outcomes,
numeric direction, case names, prompt words, answer words, and family identity
are unavailable to the fitting objective. Ties use lexicographic permutation
order.

### Held-out topology endpoint

After fitting, apply the case map without adjustment to the counterfactual
variant. The test vector contains all different-case
counterfactual-to-anchor and counterfactual-to-paraphrase similarities. The
held-out topology endpoint is their Pearson correlation between mechanisms.

Strong topology transfer requires:

1. positive median correlation across the 15 unordered mechanism pairs;
2. a two-sided exact 15-pair sign-flip `p <= 0.05`;
3. at least 10 of 15 pair correlations above zero; and
4. positive leave-one-mechanism-out means for at least five of six mechanisms.

### Physical-label endpoint

Only after the map is frozen are anchor physical labels revealed. The endpoint
is the fraction of the 60 mapped case pairs whose physical outcome agrees.
The structured null enumerates all `6^6 = 46,656` balanced assignments of two
positive anchor cases within each mechanism, while holding all fitted maps
fixed. The same calculation is repeated for numeric direction.

Strong evidence for a general physical template requires:

1. physical-label agreement above 0.5 with structured exact `p <= 0.05`;
2. at least 10 of 15 mechanism-pair agreements above 0.5;
3. at least six of the nine opposite-response-orientation pairs favor physical
   agreement over numeric-direction agreement; and
4. physical agreement minus numeric agreement has a two-sided exact
   nine-pair sign-flip `p <= 0.05`.

Topology transfer without physical-label transfer is evidence for shared graph
shape, not a shared scientific relation.

### Stability and method controls

Repeat the complete mapping at every layer and for each individual Jacobian
fit. Record permutation stability, pairwise fit agreement, and onset depth.
Apply the identical procedure to direct and raw states. A Jacobian-specific
claim additionally requires a positive Jacobian-minus-direct contrast in at
least five of six leave-one-mechanism-out means and an exact family sign-flip
`p <= 0.05`.

## Study 2: exact and approximate graph isomorphism

At every family and layer, build the frozen directed selected-edge graph:
12 nodes and two outgoing edges per node, one into each other surface-variant
group, with same-case candidates prohibited.

Test exact directed isomorphism between mechanism pairs under three node-label
schemes:

1. no node labels;
2. surface-variant labels only;
3. surface variant plus physical outcome, used only as a diagnostic after graph
   construction.

Report the fraction of exact matches; do not interpret failure of exact
isomorphism as failure of approximate structure.

Approximate similarity is measured with:

- Weisfeiler--Lehman subtree feature cosine for heights 0--3;
- sorted directed-degree and edge-reciprocity signatures;
- normalized Laplacian spectral distance after symmetrization; and
- the held-out graph-matching correlation from Study 1.

Null graphs preserve 12 nodes, the two-outgoing-edge rule, surface-variant
target groups, and the same-case exclusion while randomly selecting eligible
targets. Use 10,000 fixed-seed null graph sets. The primary approximate
endpoint is the frozen-band mean cross-mechanism Weisfeiler--Lehman similarity
with surface-variant labels, compared with the complete constrained-null
distribution. The other signatures are descriptive and false-discovery
corrected together.

## Study 3: whole-mechanism-held-out GIN

A graph isomorphism network is implemented directly in PyTorch; no external
graph-learning package is introduced.

### Graph and task

Each example is one 12-node family graph at one registered layer. The primary
node task predicts positive versus negative physical outcome. Every fold holds
out all 25 graphs and all 12 nodes from one mechanism; the other five
mechanisms are the only training data. No layer from the held-out mechanism may
enter training, model selection, normalization, PCA, or early stopping.

Two frozen input modes are tested:

1. **Topology only:** constant, three-way surface-variant one-hot, numeric
   increase/decrease sign, normalized depth, in-degree, out-degree, incoming
   strength, and outgoing strength.
2. **State assisted:** topology-only features plus 32 principal components of
   the decoder-basis state. PCA is fitted inside each fold using training
   mechanisms only. Jacobian uses the three-fit mean state; direct and raw use
   their matched states.

The primary architecture has three GIN message-passing blocks, hidden width
32, sum aggregation, ReLU, residual connections, 0.10 dropout, and a two-class
node head. Directed edges are made bidirectional for message passing while
retaining edge weight as a multiplier. Training uses Adam, learning rate
`0.01`, weight decay `0.0001`, at most 400 epochs, and training-family
leave-one-family-out validation for checkpoint choice. Seeds 0--19 are
ensembled only for algorithmic stability; seeds are not inferential units.
Node order is freshly permuted every epoch and at test time as an equivariance
audit.

### Frozen baselines and falsifications

- An MLP receives identical node features but no graph.
- A constrained edge-shuffle preserves node features, variant target groups,
  out-degree, and the graph's edge-weight multiset.
- Balanced physical labels are shuffled within every training family.
- Numeric-direction prediction is run as a competing target.
- A case-ID lookup, family ID, mechanism name, prompt text, answer word, and
  physical label are prohibited inputs.

The primary metric is ROC--AUC after averaging each node's logits across the
frozen 38--92% layers and 20 seeds. Accuracy and per-layer curves are
secondary. Strong whole-mechanism generalization requires:

1. AUC above 0.5 in all six held-out mechanisms;
2. GIN-minus-MLP and GIN-minus-edge-shuffle AUC positive in all six mechanisms;
3. plus-one exact one-sided family sign-flip `p <= 0.05` for both contrasts;
4. no loss of more than 0.02 AUC under the node-permutation audit; and
5. physical-outcome AUC exceeding numeric-direction AUC in at least four of six
   mechanisms.

State-assisted success without topology-only success is a nonlinear
representation result, not proof that graph topology alone is sufficient.
GIN performance without superiority to the identical-feature MLP is not a
graph-specific result.

## Multiplicity and interpretation

The three strong gates above are distinct:

- label-blind graph-template transfer;
- constrained-null structural isomorphism; and
- whole-mechanism-held-out GIN generalization.

No omnibus claim is made unless at least two gates pass and their required
controls pass. Benjamini--Hochberg correction at 0.05 is applied within each
family of secondary topology signatures, architectures, representations, and
layer scans. Primary frozen gates are reported regardless of sign.

A positive result supports generalization only across the six studied
mechanisms at the natural question boundary in one Gemma checkpoint. A negative
result is retained and narrows the graph claim. No graph edge is a reasoning
step, and no result is described as a mathematical proof of scientific
understanding.

## Required artifacts

Retain:

- this protocol and its SHA-256 hash;
- input hashes and environment;
- every fitted case permutation and all 24 candidate scores;
- all held-out topology vectors and correlations;
- complete 46,656-row structured-null summaries;
- every exact-isomorphism decision and approximate signature;
- constrained-null seed and statistics;
- every GIN fold, seed, checkpoint epoch, prediction, baseline, shuffle, and
  permutation-audit row;
- summary statistics, failed gates, warnings, and plots;
- a plain-language `RESULTS.md`;
- an exact artifact inventory and one-command rebuild path.
