# Experiment index

This directory contains the frozen protocols, amendments, preregistrations,
and exact prompt manifests used in the paper and Supplementary Information.
Historical status fields record when a protocol was frozen; some studies that
were initially labeled candidate or out of manuscript scope were later reported
in the final paper or SI.

This public repository is source-only. Files named `raw.json`,
`representations.npz`, `statistics.json`, result tables, and figures are
generated outputs and are intentionally not tracked. Run model-facing scripts
before analyzers that consume those files. All commands below assume:

```bash
cd jlens_materials
```

## Core readout and steering studies

| study | frozen source | primary scripts |
| --- | --- | --- |
| Held-out materials readout | `materials-heldout-v1-preregistration.md`, `../prompts/materials-heldout-v1-preregistered.json` | `generate_materials_heldout_v1.py`, `validate_materials_heldout_v1.py`, `analyze_materials_heldout_v1.py` |
| Broad mechanism steering | `semantic-steering-v3-preregistration.json` | `run_mechanism_steering.py --study broad-screen`, `analyze_semantic_steering_v3.py` |
| Prospective grain steering | `relational-grain-steering-v4-preregistration.json` | `run_mechanism_steering.py --study prospective-grain`, `analyze_relational_grain_steering_v4.py` |
| Counterfactual activation patching | `candidate-activation-patching-2026-07-16/` | `run_counterfactual_activation_patching.py`, `run_activation_patching_distance_controls.py`, `analyze_counterfactual_activation_patching.py` |

The held-out readout runner is `jlens-run`; the registered command pattern is
in `materials-heldout-v1-preregistration.md`. Lens checkpoints are not bundled.

## Lexical and answer-scaffold studies

| study | frozen source | execution order |
| --- | --- | --- |
| Lexical-adversarial discovery | `lexical-adversarial-representation-2026-07-17/` | `run_lexical_adversarial_representation.py`, then `analyze_lexical_adversarial_representation.py` |
| Disjoint late replication | `late-physics-representation-replication-2026-07-17/` | the same runner with that directory's protocol, then `analyze_late_physics_replication.py` |
| Arbitrary answer-code binding | `answer-code-binding-2026-07-17/` | `run_answer_code_binding.py`, then `analyze_answer_code_binding.py` |
| Answer-scaffold audit | `answer-scaffold-audit-2026-07-17/` | regenerate both lexical cohorts, then `analyze_answer_scaffold_audit.py` |

## Graph and robustness studies

| study | frozen source | primary scripts |
| --- | --- | --- |
| Mechanism and relation topology | `graph-topology-rigorous-2026-07-17/` | `analyze_graph_topology_rigorous.py`, `audit_graph_topology_rigorous.py` |
| Option-free checkpoint graph | `option-free-relation-graph-2026-07-17/` | `analyze_option_free_relation_graph.py` |
| Natural question-end graph | `option-free-question-end-2026-07-18/` | `run_option_free_question_end_states.py`, `analyze_option_free_question_end.py` |
| Cross-mechanism outcome test | `cross-mechanism-outcome-2026-07-18/` | `analyze_cross_mechanism_outcome.py` |
| Graph identifiability/generalization | `graph-isomorphism-generalization-2026-07-18/` | graph mapping, GIN, spectral, and exact-partition scripts documented in `PROTOCOL.md` |
| Cross-mechanism activation patching | `cross-mechanism-activation-patching-2026-07-18/` | runner, analyzer, audit, and plot scripts with matching names |
| Multi-token robustness | `multitoken-sequence-robustness-2026-07-18/` | `run_multitoken_sequence_robustness.py`, then `plot_multitoken_sequence_robustness.py` |

`review-robustness-audit-2026-07-18/README.md` gives the chronological claim
boundaries and the full regeneration order. Archived-state analyses require the
state arrays created by their upstream runners.

## Relational constitutive benchmark

The final neutral-anchored benchmark is in
`neutral-anchored-relational-physics-2026-07-18/`. Its frozen direction uses
the development cohort in `elicited-physics-abstraction-2026-07-18/`; the
disjoint confirmation is in `relational-contrast-confirmation-2026-07-18/`.

```bash
python scripts/run_elicited_physics_abstraction.py --device cpu
python scripts/run_neutral_anchored_relational_benchmark.py
python scripts/analyze_neutral_anchored_relational_benchmark.py
```

The registered neutral-benchmark runner targets Apple MPS. Other model runners
accept `--device cpu`, `--device cuda`, or `--device mps` as appropriate. Model
runs are compute-intensive.
