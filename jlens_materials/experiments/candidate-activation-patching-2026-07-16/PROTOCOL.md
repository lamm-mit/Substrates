# Candidate counterfactual activation-patching protocol

> Publication note: this exploratory follow-up was later reported in the
> Supplementary Information. The status below records when the protocol was
> frozen.

Frozen at `2026-07-16T22:26:08-04:00`, before any activation-patching output
was calculated. This is a prospective analysis of an already inspected prompt
cohort, so it is an exploratory causal follow-up rather than a new confirmatory
test. It remains outside the manuscript until reviewed.

## Question

When a materials relation is reversed while material identity, grain sizes,
covariates, and answer words are held fixed, at which layers does transplanting
the donor prompt's internal state causally move the receiver toward the donor's
physical answer? Do layers with a readable Jacobian-lens relation also have a
large causal patching effect?

## Frozen inputs

- Prompt source: `experiments/relational-grain-steering-v4-preregistration.json`
  (`32d0ebc...e33`)
- The previously observed v4 raw output is fingerprinted (`818584...838`) to
  make clear that this cohort had already been inspected; it is not used by the
  runner.
- Model: `google/gemma-4-E4B-it` at revision
  `a4c2d58be94dda072b918d9db64ee85c8ed34e3f`
- Three independently fitted WikiText lenses, at the paths and hashes stored in
  `protocol.json`.

## Intervention

The 12 frozen grain-size conditions are expanded into both registered answer
orders, giving 24 receiver prompts. For each receiver, the exact final-prompt-
token residual after a registered transformer block is replaced by one of four
donor residuals at the same layer:

1. **matched reverse:** same material and answer order, reversed grain-size
   relation;
2. **cross-material reverse:** next material in a frozen cyclic order, reversed
   relation;
3. **cross-material same:** next material, same relation;
4. **order only:** same condition, opposite answer-word presentation order.

The first donor is the primary counterfactual. The second tests whether a
generic refinement/coarsening state transfers across materials. The last two
are controls that change wording or material without reversing the physical
relation. Full-state replacement is used rather than adding a tuned vector.
No Jacobian lens is used to perform the intervention.

Patching is performed at the same 25 layers registered for the paper:
`0, 2, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23, 24, 26, 28, 29, 31,
32, 34, 36, 37, 39`. Only the final prompt position is replaced. The answer
endpoint is the exact next-token log-odds `higher - lower`.

## Readout-to-causality bridge

For each clean receiver state and layer, calculate the `higher - lower` contrast
under (i) the three fitted Jacobian lenses and (ii) matched direct unembedding.
Within every material pair, relation separation is one half of the refinement
contrast minus the coarsening contrast, averaged over answer orders (and lens
fits for the Jacobian curve). This vocabulary is used only for measurement; it
does not define or select the transplanted state.

## Endpoints and inference

Let `s=+1` for a refinement receiver and `s=-1` for a coarsening receiver. For
every patch, the counterfactual-aligned shift is
`-s * (patched_log_odds - receiver_clean_log_odds)`. Positive values mean the
receiver moved toward the answer appropriate to the opposite relation.

The primary independent unit is the matched material pair after averaging two
relations and two answer orders. The primary scalar endpoint is the signed mean
counterfactual-aligned shift over the fixed 38--92% depth band. Report 30,000
pair-cluster bootstrap intervals (seed `20260716`) for:

- matched reverse;
- matched reverse minus order only;
- matched reverse minus cross-material same;
- cross-material reverse.

Layerwise curves and the peak layer are descriptive localization results, not
independently selected tests. Report a circular-shift sensitivity analysis for
the across-layer Spearman association between the aggregate patching curve and
the Jacobian/direct relation-separation curves; because only 25 ordered layers
exist, treat this as descriptive even if small.

The experiment is successful as a causal-localization candidate only if the
matched-reverse band mean and both registered control contrasts have 95%
pair-bootstrap intervals above zero. A transferable relational code is
supported if the cross-material-reverse interval is also above zero. All
negative and mixed results are retained.

## Guardrails

Activation patching identifies states sufficient to alter a constrained answer
under a matched intervention. It does not reveal private prose, prove a literal
chain of thought, establish that the patched state is necessary, or demonstrate
unrestricted materials understanding. Because the prompts were used in an
earlier steering study, a positive result would still require replication on a
new disjoint cohort.
