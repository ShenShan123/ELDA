#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT"
exec "$PYTHON_BIN" -m pytest -q -p no:cacheprovider "$ROOT/tests"
