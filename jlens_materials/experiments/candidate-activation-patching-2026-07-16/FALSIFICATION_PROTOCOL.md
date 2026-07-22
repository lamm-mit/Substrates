# State-distance falsification protocol

Frozen at `2026-07-16T22:47:13-04:00`, after inspecting the primary activation-
patching output and before calculating any distance-matched control. This is a
post-hoc falsification study; it cannot upgrade the inspected cohort to a
confirmatory experiment.

## Confound being tested

In the frozen primary run, reversed-relation donor states were farther from the
receiver than same-relation and answer-order-only donor states. A large patch
effect could therefore reflect perturbation magnitude rather than the physical
direction encoded by the donor.

## Controls

For every receiver and registered layer, let `h` be the receiver state, `r` the
same-material reversed-relation donor, and `c` one of two controls:

- a different-material donor with the same physical relation;
- the identical condition with only answer-word order reversed.

Construct

`c_matched = h + ||r-h|| (c-h) / ||c-h||`.

Thus the falsification state has exactly the same Euclidean distance from the
receiver as the reversed-relation donor, but points along a relation-preserving
or order-only direction. Replace the final-prompt-token residual with this
state at the same 25 layers and score the same `higher - lower` log odds.

The independent unit, 38--92% band mean, counterfactual alignment, six-pair
bootstrap, seed, and 30,000 resamples are unchanged. The reverse-patch claim
survives this falsification only if its paired band-mean advantage over both
distance-matched controls has a 95% pair-bootstrap interval above zero. All
results are retained.
