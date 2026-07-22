#!/usr/bin/env bash
# End-to-end Jacobian-lens pipeline using the active Python environment.
#   fit/apply lens -> figures + runs/<model>.json
#   -> LLM analysis (Anthropic Opus 4.8 OR OpenAI GPT-5.5)
#   -> LaTeX report compiled to PDF
#
# Usage:
#   ./pipeline.sh                                   # Qwen demo, Claude Opus 4.8
#   ./pipeline.sh --provider openai                 # ... GPT-5.5 for the analysis
#   ./pipeline.sh --model google/gemma-4-E4B-it --tag gemma4-e4b-it --provider anthropic
#   ./pipeline.sh --offline                         # skip the LLM (template analysis)
#   MODEL=gpt2 ./pipeline.sh --offline              # quick GPT-2 sanity run
#
# Env:
#   PYTHON        override the interpreter (default: python3/python on PATH)
#   MODEL         HF model id (default Qwen/Qwen2.5-0.5B-Instruct)
#   SHAPES        experiment shapes (default all five)
#   ANTHROPIC_API_KEY / OPENAI_API_KEY   provider credentials for the analysis
set -euo pipefail
cd "$(dirname "$0")"

# Interpreter: honor $PYTHON, else the active python3/python on PATH. Portable
# across macOS and Linux — no hardcoded env path. After activating any conda or
# virtual environment this resolves to that environment's python; override with
# `PYTHON=/path/to/python ./pipeline.sh` if needed.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON" ]]; then
  echo "no python found on PATH; activate your env or set PYTHON=/path/to/python" >&2
  exit 1
fi
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_REVISION=""
SHAPES="${SHAPES:-MULTIHOP,ASSOCIATION,RECOGNITION,MODULATION,REPORT_SWAP}"
PROVIDER="anthropic"
ANALYZE_MODEL=""
OFFLINE=""
RECIPE="demo"      # paper = 1000 x 128, penultimate target, strict protocol
NFIT=""            # empty uses the recipe default
CORPUS="wikitext"
WORKSPACE_BAND="38,92"
PROMPTS=""
MIN_ITEMS=""
SWAPS="--swaps"   # run causal swaps for REPORT_SWAP prompts by default
TAG_OVERRIDE=""
DTYPE="float32"   # model weight dtype; bfloat16 ~halves memory on GPU
DIMBATCH="64"     # backward-pass batch during --fit; lower cuts fit peak memory
PUSH_HF=""        # if set to <user>/<repo>, upload all artifacts to a HF dataset

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider) PROVIDER="$2"; shift 2;;
    --analysis-model) ANALYZE_MODEL="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --model-revision) MODEL_REVISION="$2"; shift 2;;
    --shapes) SHAPES="$2"; shift 2;;
    --recipe) RECIPE="$2"; shift 2;;
    --n-fit) NFIT="$2"; shift 2;;
    --corpus) CORPUS="$2"; shift 2;;
    --workspace-band) WORKSPACE_BAND="$2"; shift 2;;
    --prompts) PROMPTS="$2"; shift 2;;
    --min-items-per-shape) MIN_ITEMS="$2"; shift 2;;
    --tag) TAG_OVERRIDE="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --dim-batch) DIMBATCH="$2"; shift 2;;
    --push-hf) PUSH_HF="$2"; shift 2;;
    --offline) OFFLINE="--offline"; shift;;
    --no-swaps) SWAPS=""; shift;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

echo ">> interpreter : $PYTHON"
"$PYTHON" -c "import torch,transformers; print('   torch',torch.__version__,'| transformers',transformers.__version__)"
echo ">> model       : $MODEL  (dtype $DTYPE)"
echo ">> protocol    : $RECIPE  (corpus $CORPUS; band $WORKSPACE_BAND)"
echo ">> analysis    : $PROVIDER ${ANALYZE_MODEL:-<default>} $OFFLINE"

# tag comes from run_lens.arch_tag so pipeline.sh and run_lens.py always agree
# (matters for multimodal Gemma, where config.architectures[0] != the causal
# class). --tag overrides it, e.g. to keep gemma -pt and -it distinct.
if [[ -n "$TAG_OVERRIDE" ]]; then
  LENS_TAG="$TAG_OVERRIDE"
else
  LENS_TAG="$("$PYTHON" -c "import run_lens, sys; print(run_lens.arch_tag(sys.argv[1], revision=(sys.argv[2] or None)))" "$MODEL" "$MODEL_REVISION")"
fi
LENS="lens_${LENS_TAG}.${RECIPE}.pt"
RUN="runs/${LENS_TAG}.json"
FIT_ARGS=(--recipe "$RECIPE" --corpus "$CORPUS" --workspace-band "$WORKSPACE_BAND")
[[ -n "$NFIT" ]] && FIT_ARGS+=(--n-fit "$NFIT")
[[ -n "$MODEL_REVISION" ]] && FIT_ARGS+=(--model-revision "$MODEL_REVISION")
[[ -n "$PROMPTS" ]] && FIT_ARGS+=(--prompts "$PROMPTS")
[[ -n "$MIN_ITEMS" ]] && FIT_ARGS+=(--min-items-per-shape "$MIN_ITEMS")

echo ">> step 1/3: lens + figures ${SWAPS:+(+ causal swaps)}"
if [[ -f "$LENS" ]]; then
  "$PYTHON" run_lens.py --model "$MODEL" --lens "$LENS" --tag "$LENS_TAG" \
    --dtype "$DTYPE" --shapes "$SHAPES" \
    ${FIT_ARGS[@]+"${FIT_ARGS[@]}"} $SWAPS
else
  "$PYTHON" run_lens.py --model "$MODEL" --fit --tag "$LENS_TAG" \
    --dtype "$DTYPE" --dim-batch "$DIMBATCH" --shapes "$SHAPES" \
    ${FIT_ARGS[@]+"${FIT_ARGS[@]}"} $SWAPS
fi

echo ">> step 2/3: LLM analysis"
AMODEL_ARG=()
[[ -n "$ANALYZE_MODEL" ]] && AMODEL_ARG=(--model "$ANALYZE_MODEL")
# ${arr[@]+...} guards empty-array expansion under `set -u` on macOS bash 3.2
"$PYTHON" analyze.py --run "$RUN" --provider "$PROVIDER" \
  ${AMODEL_ARG[@]+"${AMODEL_ARG[@]}"} $OFFLINE

echo ">> step 3/3: LaTeX report -> PDF"
"$PYTHON" report_latex.py --run "$RUN"

if [[ -n "$PUSH_HF" ]]; then
  echo ">> uploading artifacts to HF dataset: $PUSH_HF"
  "$PYTHON" artifacts.py push --tag "$LENS_TAG" --repo "$PUSH_HF"
fi

echo ">> done. PDF: reports/${LENS_TAG}_report.pdf"
