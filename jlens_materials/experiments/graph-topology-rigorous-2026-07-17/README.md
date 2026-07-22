# Rigorous graph-topology experiment

This directory contains the frozen source for the graph analyses reported in
the paper and Supplementary Information. The analysis consumes regenerated or
separately obtained Gemma representation arrays; those arrays and all derived
tables and figures are not bundled in this source-only release.

## What was tested

1. **Cross-phrasing mechanism organization.** Each of 50 prompts retrieves one
   neighbor from each of the four other phrasing folds. The graph therefore
   cannot succeed by matching a prompt to another prompt written in the same
   template.
2. **Lexical-confound resistance.** State similarity is residualized against
   word and character TF–IDF, token overlap, prompt length, and phrasing fold.
   A separate hard-negative test asks whether the state prefers the correct
   mechanism when prompt-only similarity prefers a wrong mechanism.
3. **Disjoint signed-relation organization.** In 72 prompts covering six new
   mechanisms, each prompt retrieves across surface variants within a
   mechanism while the same material case is excluded. An edge is correct
   when the two prompts imply the same physical increase/decrease outcome.
4. **Falsification and robustness.** Answer order, reversed counterfactual
   wording, exact case-preserving label assignments, full candidate rankings,
   leave-one-mechanism-out analyses, lens seeds, directed/undirected graphs,
   raw states, direct decoding, and lexical baselines are all retained.

## Generated figures

- `figures/mechanism-graph-evidence.*` — cross-phrasing mechanism graph,
  prompt-residual graph, hard lexical negatives, and the selected-edge matrix.
- `figures/relation-graph-falsification.*` — layer emergence, method controls,
  family results, and the surface-variant/answer-order falsification matrix.
- `figures/relation-ranking-robustness.*` — exact 46,656-assignment null and
  full-candidate ranking AUC with leave-one-mechanism-out ranges.

## Evidence and audit trail

- `PROTOCOL.md` and `protocol.json` — frozen primary analysis.
- `protocol-amendment-v1.json` — answer-order and surface-form falsification.
- `protocol-amendment-v2.json` — full-ranking and exact-null extension.
- `protocol-amendment-v2-correction.json` — transparent correction after the
  first v2 execution halted on the registered counterfactual sign reversal.
- The analyzer generates the numerical results, complete statistics, selected
  edges, candidate rankings, validation records, and figures named above.

Only the frozen protocols and amendments are tracked here. Generated outputs
must be recreated before running the audit script.

## Reproduce

From the `jlens_materials` directory:

```bash
python scripts/analyze_graph_topology_rigorous.py
python scripts/audit_graph_topology_rigorous.py
```

The analysis itself invokes no model or hosted API, but it requires the held-out
and replication representation arrays produced by the upstream workflows.

## Claim boundary

The strongest supported statement is that scientific mechanism and signed
relation information is organized in model-state geometry beyond the tested
lexical baselines. Raw states and direct-decoder states perform similarly to
the Jacobian representation, so these graph results are not evidence of a
Jacobian-specific advantage. They are descriptive, not causal, and do not
license claims about consciousness, a literal chain of thought, or unrestricted
materials understanding.
