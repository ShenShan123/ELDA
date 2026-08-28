#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

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
