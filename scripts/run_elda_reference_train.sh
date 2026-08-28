#!/usr/bin/env bash
set -euo pipefail

if [[ "${START_ELDA_TRAINING:-0}" != "1" ]]; then
  echo "Refusing to start ELDA training: set START_ELDA_TRAINING=1 explicitly."
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ELDA_DATA_ROOT:?set ELDA_DATA_ROOT to the materialized ELDA dataset root}"
PY="${ELDA_PYTHON:-python}"
cd "$PROJECT_ROOT"

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "elda_dataset_manifest.json").read_text())
expected = {"train": 170940, "val": 9396, "test": 11574}
if manifest["counts"] != expected:
    raise SystemExit(f"clean split count mismatch: {manifest['counts']} != {expected}")
if manifest["truncation"] != "disabled":
    raise SystemExit("ELDA truncation must be disabled")
print("ELDA dataset guard passed:", manifest["counts"])
PY

exec "$PY" train.py \
  experiment=elda_reference \
  final_test_num_samples=0 \
  "datamodule.root=$ROOT" \
  "datamodule.cell_mapping_path=$ROOT/mapping.txt" \
  "$@"
