# Arbitrary answer-code binding falsification

This prospectively frozen negative control is reported in the Supplementary
Information. It tests whether the late lexical-adversarial signal survives an
arbitrary A/B answer mapping; the registered manipulation check failed, so the
result is treated as inconclusive rather than as evidence for or against a
physical representation.

Tracked source files are `PROTOCOL.md`, `protocol.json`,
`protocol-amendment-v1.json`, and the exact 72-prompt `prompt_manifest.json`.
Raw outputs, statistics, result prose, and figures are generated and are not
included in this source-only release.

From the `jlens_materials` directory:

```bash
python scripts/run_answer_code_binding.py --device mps --dtype bfloat16 --chunk-size 12
python scripts/analyze_answer_code_binding.py
```

Use `--device cuda` or `--device cpu` on other systems.
