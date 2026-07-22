# Prompt and experiment data

`jlens-run --prompts PATH` accepts either this toolkit's `{"prompts": [...]}`
schema or the released Jacobian Lens `{"items": [...]}` evaluation/probe-swap
schema. A directory loads every JSON file in lexical order.

For a paper-protocol lens-quality item, provide:

```json
{
  "slug": "stable-id",
  "shape": "MULTIHOP",
  "protocol": "lens_eval",
  "domain": "mechanics",
  "text": "The question ending immediately before its answer is",
  "readout_selector": "before_answer",
  "tracked": ["latent-intermediate"],
  "synonyms": {"latent-intermediate": ["alternative"]},
  "answer": "answer",
  "must_be_absent_from_input": true,
  "must_be_absent_from_output": true
}
```

The available fixed selectors are `before_answer`, `final_prompt_token`,
`last_newline`, `assistant_response`, and `explicit`. `all_prompt` exists only
for exploratory visualization and is rejected by strict protocol mode.

For `probe_swap`, also provide `swap_from`, `swap_to`, and `swap_answer`.
For `verbal_report`, omit `swap_from`: the clean greedy answer is selected
dynamically. Directed-modulation trials use the identical teacher-forced
carrier/targets across `focus`, `suppress`, and `control`, connected by a shared
`control_group`.
For paper-scale modulation, also provide stable `phrasing_id`, `carrier_id`,
and `target_family` values; strict paper runs require 24 distinct phrasings.

[`mechanics-paper-example.json`](mechanics-paper-example.json) demonstrates the
lens-evaluation, directed-modulation, probe-swap, and verbal-report variants.
It is deliberately too small for quantitative reporting;
create at least 50 independent items per evaluation distribution and multiple
instruction phrasings/trials for modulation.

[`materials-qualitative-pack.json`](materials-qualitative-pack.json) provides
14 materials-science case studies covering Hall--Petch strengthening, Orowan
bypass, martensitic transformation, Schmid slip, diffusional creep, dislocation
character, diffraction recognition, unnamed associations, matched modulation,
and causal swaps. Run it with `--recipe demo --min-items-per-shape 50` for close
inspection without accidentally labeling the small sample quantitative. Use
`--generation-max-new-tokens 1` when auditing the input/output-absence controls.

`fracture.json` and `protein.json` are retained as legacy visualization sets.
The authoritative built-ins in `domain_prompts.py` include stricter scoring and
matched modulation controls; use the mechanics example as the starting point
for new paper-protocol datasets.
