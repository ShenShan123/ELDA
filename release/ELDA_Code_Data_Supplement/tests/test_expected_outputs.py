from __future__ import annotations

import subprocess
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_minimal_reproduction_matches_expected(tmp_path):
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_minimal_reproduction.sh"),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHON_BIN": sys.executable},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert "PASS" in completed.stdout
