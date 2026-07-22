# Counterfactual activation-patching candidate

This frozen exploratory causal follow-up is reported in the Supplementary
Information. It uses the already inspected six-pair grain-size cohort and a
separately frozen state-distance falsification.

## Exact question

If the final-prompt-token state from a refinement description is transplanted
into the matched coarsening description, or vice versa, does the receiver move
toward the donor's physically appropriate answer? At which layers? Does the
layer profile align with relation information readable through the Jacobian
lens?

## Tracked source and generated outputs

- `PROTOCOL.md` and `protocol.json`: primary design frozen before output.
- `FALSIFICATION_PROTOCOL.md` and `falsification_protocol.json`: post-hoc
  state-distance control frozen before its output.

The runners generate `raw.json` (24 clean prompts, 2,400 readout rows, and
2,400 patch rows) and `distance_controls_raw.json` (1,200 norm-matched control
rows). The analyzer then creates statistics, row-level CSV files, result prose,
and figures. Generated outputs are not included in this source-only release.

The 24 exact formatted user prompts are generated mechanically from the 12
stems in
`../relational-grain-steering-v4-preregistration.json`, each evaluated in both
answer-word orders.

## Reproduce

From the `jlens_materials` directory:

```bash
python scripts/run_counterfactual_activation_patching.py \
  --device mps --dtype bfloat16

python scripts/run_activation_patching_distance_controls.py \
  --device mps --dtype bfloat16

MPLCONFIGDIR=/tmp/jlens-mplconfig \
  python scripts/analyze_counterfactual_activation_patching.py
```

Use `--resume` on either runner after an interrupted execution. On Linux,
replace `--device mps` with `--device cuda`.

## Headline result and limits

Across the frozen 38--92% band, the matched reversed-relation state shifted the
answer by `+6.249` higher-minus-lower log-odds units toward the donor's physical
answer (95% pair-bootstrap interval `+5.663` to `+6.795`) and was positive in
all 12 conditions. A reversed relation from a different material transferred
equally well. Relation-preserving and answer-order-only controls remained much
smaller even after their state differences were scaled to exactly match the
reverse donor's distance.

This shows causal sufficiency of a prompt state for a constrained relation. It
does not show necessity, identify a complete circuit, reveal a prose chain of
thought, or provide a confirmatory result on a new cohort. The paper therefore
treats it as an exploratory causal follow-up rather than a primary independent
confirmation.
