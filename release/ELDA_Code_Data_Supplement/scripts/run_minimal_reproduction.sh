#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" - <<'PY'
import importlib
required = ("torch", "torch_geometric", "transformers", "networkx")
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing Python dependencies:\n  " + "\n  ".join(missing))
print("Python dependency check: PASS")
PY

exec "$PYTHON_BIN" "$ROOT/scripts/run_pipeline.py" "$@"
