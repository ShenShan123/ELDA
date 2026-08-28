#!/usr/bin/env python3
"""Audit the lightweight GitHub source view before commit or upload."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
REQUIRED = (
    "README.md",
    "ARTIFACTS.md",
    "LICENSE",
    "pyproject.toml",
    "environment.yaml",
    "elda/datamodules/data/circuit_source_net_v61_tokenizer.py",
    "configs/experiment/circuit_source_net_v61_clean_compact_load.yaml",
    "data_manifests/v61/split_summary.json",
    "release/ELDA_Code_Data_Supplement/data_sample/paper_attempt_payload.json",
)
TEXT_SUFFIXES = {
    ".cfg", ".csv", ".json", ".md", ".py", ".pyx", ".sh", ".toml",
    ".txt", ".yaml", ".yml",
}


def publication_paths() -> list[Path]:
    command = [
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"
    ]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    return sorted(
        ROOT / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    )


def main() -> int:
    errors: list[str] = []
    paths = publication_paths()
    path_set = {path.relative_to(ROOT).as_posix() for path in paths}
    for required in REQUIRED:
        if required not in path_set:
            errors.append(f"required publication file is absent: {required}")

    total = 0
    home_pattern = re.compile(r"/home/[A-Za-z0-9._-]+/")
    secret_pattern = re.compile(
        r"(?:ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
    )
    legacy_import = re.compile(r"(?:^|\s)(?:from|import)\s+autograph(?:\.|\s|$)", re.M)
    for path in paths:
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f"symlink is not allowed in source release: {relative}")
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 10 MiB source limit: {relative} ({size} bytes)")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if home_pattern.search(text):
            errors.append(f"author-machine absolute path: {relative}")
        if secret_pattern.search(text):
            errors.append(f"credential-like material: {relative}")
        if legacy_import.search(text):
            errors.append(f"legacy implementation import: {relative}")

    if total > MAX_TOTAL_BYTES:
        errors.append(f"source view exceeds 25 MiB: {total} bytes")

    for path in (path for path in paths if path.suffix == ".py"):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError) as exc:
            errors.append(f"Python compilation failed: {path.relative_to(ROOT)}: {exc}")

    if errors:
        print("ELDA GitHub release audit: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        "ELDA GitHub release audit: PASS "
        f"({len(paths)} files, {total / 1024 / 1024:.2f} MiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
