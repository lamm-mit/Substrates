# Disjoint replication of the late physical-equivalence transition

This prospectively frozen, disjoint cohort is reported in the paper and
Supplementary Information. The tracked directory contains the exact prompt
manifest and frozen protocol; raw states, statistics, results, and figures are
generated locally.

From the `jlens_materials` directory:

```bash
python scripts/run_lexical_adversarial_representation.py \
  --protocol experiments/late-physics-representation-replication-2026-07-17/protocol.json \
  --output experiments/late-physics-representation-replication-2026-07-17/raw.json \
  --states-output experiments/late-physics-representation-replication-2026-07-17/representations.npz \
  --device mps --dtype bfloat16
python scripts/analyze_late_physics_replication.py
```

Use `--device cuda` or `--device cpu` on other systems.
