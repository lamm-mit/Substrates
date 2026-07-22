# Answer-scaffold audit

This post-hoc audit is reported in the Supplementary Information. It performs
no new model forward pass: it compares stored states from the lexical-
adversarial discovery and disjoint replication cohorts before and after the
answer scaffold.

Regenerate both upstream cohorts first, then run from `jlens_materials`:

```bash
python scripts/analyze_answer_scaffold_audit.py
```

The script writes the statistics, result narrative, and figure; those generated
files are not included in this source-only release.
