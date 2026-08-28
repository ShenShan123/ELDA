#!/usr/bin/env bash
set -euo pipefail

if [[ "${START_SOURCE_NET_V61_TRAINING:-0}" != "1" ]]; then
  echo "Refusing to start V6.1 training: set START_SOURCE_NET_V61_TRAINING=1 explicitly."
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ELDA_DATA_ROOT:?set ELDA_DATA_ROOT to the materialized V6.1 dataset root}"
PY="${ELDA_PYTHON:-python}"
cd "$PROJECT_ROOT"

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "v61_dataset_manifest.json").read_text())
expected = {"train": 170940, "val": 9396, "test": 11574}
if manifest["counts"] != expected:
    raise SystemExit(f"clean split count mismatch: {manifest['counts']} != {expected}")
if manifest["truncation"] != "disabled":
    raise SystemExit("V6.1 truncation must be disabled")
print("V6.1 dataset guard passed:", manifest["counts"])
PY

exec "$PY" train.py \
  experiment=circuit_source_net_v61_clean_compact_load \
  final_test_num_samples=0 \
  "datamodule.root=$ROOT" \
  "datamodule.cell_mapping_path=$ROOT/mapping_v61.txt" \
  "$@"
