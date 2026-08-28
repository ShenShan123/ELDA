#!/usr/bin/env bash
set -euo pipefail

ELDA_ROOT="${ELDA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ELDA_PAPER_ROOT="${ELDA_PAPER_ROOT:-$ELDA_ROOT/paper}"
ELDA_PYTHON="${ELDA_PYTHON:-python}"
ELDA_REPRO_SCOPE="${ELDA_REPRO_SCOPE:-all}"

export ELDA_ROOT ELDA_PAPER_ROOT
cd "$ELDA_ROOT"

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "missing required directory: $1" >&2
    exit 2
  fi
}

require_dir "$ELDA_PAPER_ROOT/results/source_net_v61/topology_safe_mask_n1024_best7epoch/attempts"
require_dir "$ELDA_PAPER_ROOT/results/source_net_v61/d2_policy_main_n1024"
require_dir "$ELDA_PAPER_ROOT/results/source_net_v61/decoder_ablation_best7epoch_final_20260706"
require_dir "$ELDA_PAPER_ROOT/results/source_net_v61/final_representation_ablation_parallel"

run_script() {
  echo "[ELDA reproducibility] $1"
  "$ELDA_PYTHON" "$ELDA_PAPER_ROOT/scripts/$1"
}

# Attempt-only and frozen-report tables.
run_script audit_table3_and_ablation_tables.py
run_script summarize_random100_medium_scaffold_experiment.py
run_script audit_appendix_e3_e4_frozen_details.py

if [[ "$ELDA_REPRO_SCOPE" == "core" ]]; then
  exit 0
fi

: "${ELDA_DATA_ROOT:?set ELDA_DATA_ROOT to the V6.1 dataset root}"
: "${ELDA_V5_DATA_ROOT:?set ELDA_V5_DATA_ROOT to the V5 projection dataset root}"
: "${ELDA_COMMON_DATA_ROOT:?set ELDA_COMMON_DATA_ROOT to the common graph dataset root}"
export ELDA_DATA_ROOT ELDA_V5_DATA_ROOT ELDA_COMMON_DATA_ROOT

# Main graph/endpoint table.
run_script compute_table1_canonical_novelty.py
run_script compute_unified_dedup11390_reference_metrics.py
run_script recompute_v61_policy_native_metrics.py
run_script compute_v61_endpoint_fidelity_table.py

# Appendix tables.
run_script compare_logic_usefulness_test_baselines.py
run_script compute_appendix_size_stratified_and_cell_inventory.py
run_script compute_functional_category_design_scale_table.py

echo "[ELDA reproducibility] completed scope=$ELDA_REPRO_SCOPE"
