# Lexical-adversarial materials representation study

This frozen discovery cohort is reported in the paper and Supplementary
Information. The protocol and exact prompt manifest were checksum-locked before
the model run. The tracked directory contains those source inputs; raw states,
statistics, result prose, and figures are generated locally.

From the `jlens_materials` directory:

```bash
python scripts/run_lexical_adversarial_representation.py --device mps --dtype bfloat16
python scripts/analyze_lexical_adversarial_representation.py
```

Use `--device cuda` or `--device cpu` on other systems.
