#!/usr/bin/env bash
# Rebuild the robustness tables and figures from archived raw data.
# This script does not rerun Gemma or alter any frozen protocol.

set -euo pipefail

cd "$(dirname "$0")/.."

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/jlens-materials-mpl}"
mkdir -p "$MPLCONFIGDIR"

python scripts/analyze_option_free_relation_graph.py
python scripts/analyze_option_free_question_end.py
python scripts/analyze_cross_mechanism_outcome.py
python scripts/plot_multitoken_sequence_robustness.py
python scripts/plot_relation_graph_robustness.py

if [[ -f experiments/cross-mechanism-activation-patching-2026-07-18/raw.json ]]; then
  python scripts/analyze_cross_mechanism_activation_patching.py
  python scripts/audit_cross_mechanism_activation_patching.py
  python scripts/plot_cross_mechanism_activation_patching.py
fi

python scripts/build_review_robustness_si_inventory.py
python scripts/validate_robustness_bundle.py

echo "Robustness analyses, figures, SI inventory, and validation rebuilt."
