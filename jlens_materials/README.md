# Technical reference

This directory contains the flat Python modules, prompt manifests, experiment
scripts, and vendored Jacobian Lens runtime used by Substrates. Install the
repository from its root so the console commands and top-level module imports
resolve consistently.

## Core workflow

1. `run_lens.py` loads a Hugging Face causal language model, fits or validates a
   Jacobian lens, evaluates prompts at fixed positions, and writes figures plus
   a JSON run record.
2. `analyze.py` creates an offline summary or optional model-assisted analysis.
3. `report.py` and `report_latex.py` turn the run and analysis records into
   Markdown or LaTeX/PDF reports.
4. `swap.py` and `auto_swap.py` run causal coordinate interventions.

Installed command names are:

| command | purpose |
| --- | --- |
| `jlens-run` | fit/apply a lens and create a run record |
| `jlens-analyze` | analyze a run, optionally offline |
| `jlens-report-md` | build a Markdown report |
| `jlens-report` | build a LaTeX report and optionally a PDF |
| `jlens-compare` | compare run records |
| `jlens-animate` | render a thought-stream animation |
| `jlens-swap` | run one causal swap |
| `jlens-auto-swap` | run registered swap batches |
| `jlens-artifacts` | synchronize explicitly selected artifacts with a Hub dataset |

Use `COMMAND --help` for the complete arguments.

## Recipes

`--recipe demo` is intended for smoke tests and qualitative exploration. It
permits the eight-record built-in fitting corpus and small evaluation sets.

`--recipe paper` enforces the quantitative protocol:

- 1,000 independent fitting records of length 128;
- penultimate target layer and 25 evenly spaced source/report layers;
- strict fixed-position prompt checks and no skipped layers;
- at least 50 independent items per reported distribution;
- a content-addressed lens provenance sidecar.

Paper mode rejects unverified lenses, the built-in corpus, relaxed protocol
checks, reduced sample thresholds, and exploratory legacy prompt records.

## Fitting corpora

`--corpus wikitext` resolves to `Salesforce/wikitext`, subset
`wikitext-103-raw-v1`, training split. A Hugging Face dataset ID or a local
TXT/JSON/JSONL file can be supplied instead. Use `--corpus-revision` to pin a
Hub dataset and preserve the resulting metadata sidecar.

## Prompts

`--prompts` accepts one JSON file or a directory of JSON files. New quantitative
items should declare a stable slug, protocol, domain, text or chat fields,
fixed readout selector, tracked concepts, expected answer where applicable,
and input/output absence requirements. See [`prompts/README.md`](prompts/README.md)
and `mechanics-paper-example.json` for the schema.

The fixed selectors are `before_answer`, `final_prompt_token`, `last_newline`,
`assistant_response`, and `explicit`. `all_prompt` is exploratory and rejected
by strict protocol mode.

## Outputs

Default generated paths are:

```text
jlens_materials/lenses/       fitted weights and provenance sidecars
jlens_materials/runs/         numeric run and analysis records
jlens_materials/figures/      plots and animations
jlens_materials/reports/      Markdown, LaTeX, and PDF reports
jlens_materials/experiments/  frozen protocols plus generated study artifacts
```

The lens, run, figure, and report paths are ignored because they can be large,
may contain model outputs, and must be reviewed separately before publication.
Within `experiments/`, protocol and prompt sources are tracked while raw model
outputs, state arrays, statistics, tables, and rendered figures are ignored.

## Credentials and remote artifacts

Hugging Face authentication uses `HF_TOKEN` or `hf auth login`. Optional hosted
analysis uses `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. The code reads these from
the environment and does not put them in run JSON or lens metadata.

Lens weights can be stored as a private Hugging Face model repository through
the `--hf-lens-*` options. `jlens-artifacts` separately handles selected run
artifacts in a dataset repository. Neither remote operation occurs unless it is
explicitly requested.

## Experiment scripts

The files under `scripts/` encode prompt generation, frozen-protocol execution,
analysis, null tests, robustness audits, and figure construction for the study.
They intentionally do not bundle raw observations or model checkpoints. Run
them from this directory after creating the prerequisite run records described
by the corresponding protocol. From the repository root, install the
`experiments` extra for pandas, SciPy, scikit-learn, NetworkX, and UMAP:

```bash
python -m pip install -e ".[experiments]"
```

The [experiment index](experiments/README.md) distinguishes model runners from
analyzers that require generated raw/state files.
