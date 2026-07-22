# Robustness-study index and integrity bundle

This directory indexes the robustness experiments reported in the paper and
Supplementary Information. It does not replace each study's frozen protocol.
It records which question each directory answers, which results were known
before later protocols were frozen, and which claims are permitted.

This public repository is source-only. Names such as `raw.json`,
`representations.npz`, `statistics.json`, CSV tables, validation records, and
figures below refer to outputs created by the listed runners and analyzers;
those generated files are not tracked.

Run the bundle integrity check with:

```bash
python scripts/validate_robustness_bundle.py
```

Rebuild every analysis, corrected figure, SI inventory, and audit from the
archived raw data without rerunning Gemma:

```bash
scripts/rebuild_review_robustness_outputs.sh
```

In the reported execution, `validation.json` recorded 25 of 25 passing bundle
checks and the causal-patching audit recorded 27 of 27. Re-running the integrity
check requires regenerating the complete upstream output bundle first.

## Chronology and outcomes

| Order | Study | New model run? | Frozen result | Claim permitted |
|---|---|---:|---|---|
| 1 | `option-free-relation-graph-2026-07-17` | No; archived checkpoint states | None | The common `Internal checkpoint` marker does not carry the previously reported signed-relation graph. |
| 2 | `option-free-question-end-2026-07-18` | Yes | Strong within-mechanism evidence | At the natural end of a question with no answer scaffold, within-mechanism neighborhoods share the expected property direction. This is Gemma-state evidence, not Jacobian-specific and not an independent replication. |
| 3 | `cross-mechanism-outcome-2026-07-18` | No; frozen analysis of study 2 | None | The within-mechanism result does not generalize when mechanisms reverse the link between numerical trend and physical outcome. |
| 4 | `multitoken-sequence-robustness-2026-07-18` | Yes | Frozen breadth gate failed | Complete `transgranular` and `martensite` sequences are favored on average, but only 7/10 prompt-level checks are positive. First-token fragments alone are misleading. |
| 5 | `cross-mechanism-activation-patching-2026-07-18` | Yes; causal | Partial | Aggregate and transfer-subset gates pass, but only 4/6 donor families are physically aligned; numerical direction transfers more strongly. |

Later rows were designed only after the earlier result was inspected. They
must not be described as independent replications of one another.

## 1. Option-free checkpoint graph

Question: does relation topology appear after the words `Internal
checkpoint`, before any answer code is supplied?

Primary Jacobian result:

- selected-edge precision `0.5208`, exact structured-null `p=0.8038`;
- all-candidate AUC `0.4861`, exact `p=0.6818`;
- two of six family AUCs above `0.5`;
- frozen verdict: no option-free evidence.

Exact inputs and outputs:

- protocol: `experiments/option-free-relation-graph-2026-07-17/protocol.json`;
- all 72 stems and constructed prompts:
  `experiments/option-free-relation-graph-2026-07-17/prompt_inventory.csv`;
- every selected edge and candidate:
  `all_selected_edges.csv` and `all_candidate_rankings.csv` in that directory;
- all 46,656 structured-null assignments:
  `primary_exact_nulls.npz`.

Reproduce:

```bash
python scripts/analyze_option_free_relation_graph.py
```

## 2. Natural question-end graph

Question: does topology appear at the end of the complete scientific question
when no answer choices, answer words, arbitrary code, response instruction,
or checkpoint marker is present?

Primary Jacobian result:

- selected-edge precision `0.6736`, exact structured-null `p=0.02195`;
- all-candidate AUC `0.6424`, exact `p=0.01235`;
- five of six family AUCs above `0.5`;
- no Jacobian-specific improvement over direct or raw states;
- frozen verdict: strong option-free within-mechanism evidence.

The same 72 scientific stems were already used in the checkpoint experiment.
This is a new, frozen positional robustness run, not an independent cohort.

Exact inputs and outputs:

- source manifest:
  `experiments/answer-code-binding-2026-07-17/prompt_manifest.json`;
- exact field used for every prompt: `stem`;
- protocol and position definition:
  `experiments/option-free-question-end-2026-07-18/PROTOCOL.md`;
- raw model metadata and clean next tokens: `raw.json`;
- complete raw, direct, and three-fit Jacobian states:
  `representations.npz`;
- every edge, candidate, family, and layer result: CSV files in the same
  directory.

Reproduce:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/run_option_free_question_end_states.py
python scripts/analyze_option_free_question_end.py
```

## 3. Cross-mechanism outcome falsification

Question: does the natural question-end state match the same physical outcome
across different mechanisms when the required numerical trend reverses?

For each of 72 sources, five other target mechanisms and three target
phrasings define 1,080 rankings per readout. Each ranking has exactly two
same-outcome and two opposite-outcome candidates. The critical 648 rankings
per readout cross mechanisms with opposite response orientations.

Primary Jacobian result:

- all cross-mechanism AUC `0.5134`, exact 20-assignment `p=0.10`;
- counter-numeric AUC `0.4707`, exact `p=0.10`;
- three of nine opposite-orientation mechanism pairs above `0.5`;
- direct and raw states behave similarly;
- frozen verdict: no cross-mechanism physical-outcome evidence.

The exact null has only 20 balanced assignments, so `0.05` is its smallest
attainable probability. This design limitation was fixed before calculation.

Exact data:

- all 10,800 method-query rows:
  `experiments/cross-mechanism-outcome-2026-07-18/all_query_rankings.csv`;
- all 150 method-pair rows: `mechanism_pair_metrics.csv`;
- all 20 exact assignments: `exact_orientation_nulls.json`;
- all layers: `layer_metrics.csv`.

Reproduce:

```bash
python scripts/analyze_cross_mechanism_outcome.py
```

## 4. Multi-token scientific terms

Question: does a complete multi-piece technical word outrank an equal-piece
scientific alternative?

Frozen contrasts:

- `transgranular` = `trans` + `granular` versus `intergranular` =
  `inter` + `granular`;
- `martensite` = `mart` + `ens` + `ite` versus `bainite` =
  `b` + `ain` + `ite`.

The lens supplies only the first-piece score at the intermediate state.
Remaining pieces are teacher-forced through unchanged Gemma. Equal piece
counts make target minus contrast an exact restricted sequence log-odds
ratio.

Results:

- cleavage: Jacobian `+1.093`, direct `-0.105`;
- rapid transformation: Jacobian `+6.149`, direct `+2.739`;
- every lens fit is positive at the family level;
- only 7/10 Jacobian prompt margins are positive versus the frozen 8/10
  requirement;
- frozen verdict: fail.

Exact prompts are the five `cleavage` and five `rapid-transformation` rows in
`prompts/materials-heldout-v1-preregistered.json`. Complete layer, prompt,
fit, first-piece, continuation, and sequence values are retained in
`experiments/multitoken-sequence-robustness-2026-07-18/`.

Reproduce model scores:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/run_multitoken_sequence_robustness.py \
  --device cpu \
  --local-model-snapshot /path/to/a4c2d58be94dda072b918d9db64ee85c8ed34e3f \
  --amendment experiments/multitoken-sequence-robustness-2026-07-18/protocol-amendment-v1.json
```

Rebuild the corrected plotting-only figure:

```bash
python scripts/plot_multitoken_sequence_robustness.py
```

## 5. Option-free cross-mechanism activation patching

Question: can a state captured before any answer list causally transfer its
physical outcome across mechanisms, different answer words, and reversed
input--property trends?

The experiment patches all 24 exact anchor questions with all four anchor
donors from each of the other five mechanisms at four frozen layers. No
Jacobian lens performs the patch. The complete design contains 1,920
interventions.

Frozen result:

- all-pair physical-outcome transfer `+0.385`; pair-sign exact
  `p=0.00897`, structured donor-label exact `p=0.03194`;
- across `higher/lower` versus `greater/smaller`: `+0.491`,
  pair-sign `p=0.03125`, structured `p=0.01363`;
- across reversed numerical response orientations: `+0.382`,
  pair-sign `p=0.003906`, structured `p=0.02816`;
- all five pairs satisfying both controls are positive, but the pair-sign
  test is limited to `p=0.0625`;
- only 4/6 donor mechanisms are positive, below the frozen 5/6 breadth gate;
- numerical-direction transfer is stronger overall (`+0.571`) than
  physical-outcome transfer;
- Orowan and porosity donors move receivers in the wrong physical direction;
- frozen verdict: partial evidence.

The cross-vocabulary result rules out copying one particular answer token.
The stronger numerical-direction contrast and failed mechanism breadth rule
out a universal physical-relation interpretation. The defensible description
is **option-free causal transfer of a late numerical/decision feature with
partial physical-outcome alignment**.

Exact artifacts:

- all 1,920 interventions: `all_patch_rows.csv`;
- all receiver/layer contrasts: `receiver_layer_contrasts.csv`;
- all ordered and unordered mechanism-pair effects:
  `ordered_mechanism_pair_effects.csv` and
  `unordered_mechanism_pair_effects.csv`;
- all 46,656 structured-null assignments:
  `primary_exact_donor_label_nulls.npz`;
- frozen amendment: `protocol-amendment-v1.json`;
- independent audit: `validation.json`, 27/27 checks passed.

Reproduce:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/run_cross_mechanism_activation_patching.py \
  --device cpu \
  --dtype bfloat16 \
  --local-model-snapshot /path/to/a4c2d58be94dda072b918d9db64ee85c8ed34e3f
python scripts/analyze_cross_mechanism_activation_patching.py
python scripts/audit_cross_mechanism_activation_patching.py
python scripts/plot_cross_mechanism_activation_patching.py
```

## Prompt and data inventory represented in the SI

The Supplementary Information records:

1. all 72 natural question stems, IDs, mechanisms, variants, numerical
   directions, and expected physical outcomes;
2. all ten multi-token prompts, tokenizer pieces, exact scientific contrasts,
   prompt-level scores, and family summaries;
3. the complete mechanism-pair and subset summaries for cross-mechanism
   geometry and patching.

The machine-readable SI archive should include the full candidate and patch
tables rather than typesetting thousands of rows. Every SI table must point to
the exact CSV and SHA-256 entry in `validation.json`. No row may be removed
because it is negative or visually inconvenient.

## Interpretation boundaries

- A vocabulary readout is not a literal chain of thought.
- Similar geometry in raw, direct, and Jacobian states is evidence about
  Gemma, not a Jacobian-specific gain.
- Within-mechanism relation topology is not a universal property-direction
  coordinate.
- Full-state patching is causal sufficiency, but the semantic content of the
  transplanted state requires factorial controls.
- The multi-token test contains only two terms and cannot support a universal
  tokenizer-robustness claim.
- Later frozen analyses of already inspected cohorts are transparent post-hoc
  tests, not preregistration or independent replication.
