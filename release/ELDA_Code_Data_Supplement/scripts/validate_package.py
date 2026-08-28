#!/usr/bin/env python3
"""Static release checks independent of the scientific smoke test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REQUIRED = (
    "README.md",
    "MANIFEST.md",
    "environment.yml",
    "requirements.txt",
    "configs/paper_training.yaml",
    "configs/paper_generation.yaml",
    "data_sample/raw_subcircuit.json",
    "expected_outputs/validity_report.json",
    "scripts/run_minimal_reproduction.sh",
    "tests/test_source_demand.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-mb", type=float, default=50.0)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures = []
    for name in REQUIRED:
        if not (root / name).is_file():
            failures.append(f"missing required file: {name}")
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            failures.append(f"symlink is not allowed: {path.relative_to(root)}")
        elif path.is_file():
            total += path.stat().st_size
        try:
            path.relative_to(root)
        except ValueError:
            failures.append(f"path escapes root: {path}")
    if total > args.max_mb * 1024 * 1024:
        failures.append(f"uncompressed size exceeds {args.max_mb:.1f} MB")
    for script in (root / "scripts").glob("*.sh"):
        if not os.access(script, os.X_OK):
            failures.append(f"shell script is not executable: {script.name}")
    if failures:
        print("Package validation: FAIL")
        for item in failures:
            print(f"  {item}")
        raise SystemExit(1)
    print(f"Package validation: PASS ({total / 1024 / 1024:.3f} MiB uncompressed)")


if __name__ == "__main__":
    main()
