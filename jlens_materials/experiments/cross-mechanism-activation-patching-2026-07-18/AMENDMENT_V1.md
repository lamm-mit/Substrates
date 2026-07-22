# Amendment v1: structured donor-label exact null

Frozen at `2026-07-17T18:16:50-04:00` while the model run was still
executing, before the raw archive was complete and before any patching
endpoint or figure was calculated or inspected.

The base protocol averages two patching directions within 15 unordered
mechanism pairs and uses exact pair-level sign flips. Those 15 graph edges
reuse the same six mechanism families. To prevent an optimistic
independence assumption, the final analysis must also pass a stricter
structured randomization.

Within each donor mechanism, the four anchor cases contain two positive and
two negative physical outcomes. The new null enumerates every choice of two
cases as positive and two as negative. There are six assignments per
mechanism and six mechanisms, giving exactly `6^6 = 46,656` joint
assignments. Every receiver, donor state, patch output, layer, vocabulary,
mechanism identity, and response-orientation label remains fixed.

The null recomputes all four frozen physical-outcome endpoints. Strong
evidence now requires both:

1. the original pair-sign exact `p <= 0.05`; and
2. the structured donor-label exact `p <= 0.05`.

This change is conservative and cannot turn a failed base-protocol gate into
a pass. It changes no prompt, donor, state, layer, model output, or scientific
intervention.

Fingerprints:

- base protocol:
  `a3d41ff944b9c602ce87fa55d6efcfb238353fa7cd4a2837c56d1344091c0f57`;
- frozen analysis implementation:
  `cfa4bb5fd1a38b5a1cce36ad6f9fbccc331c92a89aa764e8526d3a78d081b29d`.
