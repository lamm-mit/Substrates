# Frozen gauge-invariant graph follow-up

Frozen: `2026-07-17T21:20:50-04:00`

> Publication note: this follow-up was later reported in the paper's
> Supplementary Information. Statements below about manuscript scope preserve
> the protocol's status when it was frozen.

## Why this follow-up exists

The previously frozen whole-mechanism GIN endpoint failed after all 1,320
training runs completed.  In particular, a topology-only GIN did not predict
absolute physical outcome in a held-out mechanism, whereas it decoded the
explicit numerical increase/decrease direction perfectly.  This follow-up was
specified after that failure and before inspecting pairwise gauge-adjusted
label agreement, permutation-cycle consistency, per-mechanism relational AUC,
or relation-GIN results.

The failure suggests a precise identifiability problem.  Some mechanisms have
a direct response (increasing the input increases the output) and others have
an inverse response.  Multiplying every binary outcome in one mechanism by
minus one changes absolute signs but leaves all same-versus-different
relations unchanged.  We therefore test whether the transferable object is an
orientation-free relation graph rather than a universal absolute positive
label.  This is a finite `Z_2` gauge analogy, not a claim of a physical gauge
field.

This work is exploratory with frozen endpoints.  The manuscript remains out
of scope until the investigator reviews the complete positive and negative
results.

## Frozen inputs

The inputs, graph rule, 72 prompts, six mechanisms, 25 layers, 38--92% depth
band, three-fit Jacobian mean, direct states, raw states, and all hashes are
identical to `PROTOCOL.md`.  Studies 1--3 are not refitted or altered.

## Study 4A: gauge-adjusted alignment

Use every band-level case permutation fitted without labels in Study 1.  Extend
the case map to all three surface variants.  Reveal physical labels only after
mapping and calculate:

1. ordinary 12-node agreement;
2. gauge-adjusted agreement, the larger of ordinary agreement and agreement
   after flipping every target-mechanism label;
3. the same quantities for numerical direction; and
4. agreement with the fully prompt-derived response code
   `numeric_direction * counterfactual_variant_sign`.

The structured null enumerates all `6^6 = 46,656` balanced two-positive anchor
assignments while preserving the registered counterfactual reversal and fixed
maps.  It recomputes the complete 15-pair mean gauge agreement.  A physical
claim requires exact `p <= 0.05`, at least 10 of 15 perfect gauge-aligned
pairs, and performance above the numerical and prompt-derived controls.  If
the prompt-derived control is identical, the result is an identifiability
demonstration rather than evidence of additional learned physics.

## Study 4B: permutation cycle consistency and a global atlas

Treat each label-blind four-case map as an element of the permutation group
`S_4`.  Invert maps to obtain both directions.  For each of the 20 mechanism
triples, compose the three maps around the directed cycle.  Record:

- exact identity-cycle rate;
- fraction of the four case positions returned to themselves; and
- permutation distance from identity.

Compare the mean fixed-position fraction with 100,000 sets of 15 independent
uniform random `S_4` maps.  Also compare with a degree/margin-matched null that
resamples each pair from its 24 candidate maps with probability proportional
to `exp((score - maximum_score) / 0.05)`.

Construct a global atlas without labels by fixing the alphabetically first
mechanism to the identity and finding the exact maximum-total-score assignment
of one `S_4` element to each remaining mechanism.  All 15 pairwise candidate
score tables enter the objective.  Ties are lexicographic.  Test the atlas on
the frozen counterfactual similarity endpoint and then reveal labels.

Strong atlas evidence requires a cycle-consistency `p <= 0.05` against both
nulls, positive held-out counterfactual correlation for at least 10 of 15
pairs, and a label-blind atlas objective better than 95% of 10,000
pair-score-permuted atlases.  Repeat for Jacobian, direct, and raw states.  A
Jacobian-specific interpretation requires a corrected Jacobian advantage over
both controls.

## Study 4C: orientation-free relational decoding

The target for an ordered node pair is whether its two physical outcomes are
the same.  This target is unchanged by flipping every outcome within a
mechanism.  Use the registered cross-variant, different-case candidate
universe.

### Nonparametric endpoint

For every mechanism and layer, rank candidate pairs by cosine similarity and
compute physical-relation ROC--AUC.  Average scores over the frozen band before
computing the primary per-mechanism AUC.  Compare against:

- numerical same/different labels;
- the prompt-derived response-code relation;
- direct and raw representations;
- within-family label permutations preserving six positive and six negative
  node labels; and
- layerwise circular shifts as a depth control.

Strong nonparametric generalization requires physical-relation AUC above 0.5
in all six mechanisms, a one-sided exact family sign-flip `p <= 0.05`, and
physical AUC above both prompt-derived and raw-state controls in at least five
of six mechanisms with corrected family sign-flip `p <= 0.05`.

### Relation GIN

Train a three-block width-32 GIN under the same outer and inner mechanism
splits, optimizer, epochs, seeds, permutation audit, and graph construction as
Study 3.  The GIN produces a node embedding; a symmetric pair head receives
the elementwise product and absolute difference of two embeddings and predicts
same versus different physical outcome.  The input deliberately excludes
numeric direction, surface variant, prompt text, case ID, mechanism ID, answer
words, and physical labels.  It contains only a constant, normalized depth,
in/out degree, and in/out strength.  An identical-feature pair MLP, constrained
edge shuffle, balanced label shuffle, direct/raw representations, and numerical
relation target are controls.

Average pair logits across band layers and 20 seeds before inference.  Strong
relation-GIN evidence requires AUC above 0.5 in all six held-out mechanisms,
positive GIN-minus-pair-MLP and GIN-minus-edge-shuffle contrasts in all six,
plus-one one-sided family sign-flip `p <= 0.05` for both contrasts, and no more
than 0.02 AUC loss under node permutation.

## Interpretation gate

Three outcomes are distinguished:

1. **Absolute and relational transfer:** absolute Study 3 and relational Study
   4 both pass.
2. **Relational-only transfer:** Study 3 fails but at least two independent
   Study 4 gates pass with their prompt/raw controls.  The supported claim is
   that a relation survives mechanism-wide orientation changes.
3. **Form-only transfer:** topology/cycle endpoints pass but physical
   relational endpoints do not beat prompt-derived or raw-state controls.  The
   shared graph is then attributed to question form or generic representation
   geometry, not general scientific content.

No result is called a mathematical proof of model understanding.  Exact
finite permutation identities may be proved for this dataset; scientific
generalization remains bounded to the six mechanisms and one checkpoint.
