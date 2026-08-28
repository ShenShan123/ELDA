#!/usr/bin/env python3
"""Audit that the ELDA implementation uses only its public package name."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY = "auto" + "graph"
OLD_ABSOLUTE_ROOT = "/" + "home" + "/yuany/work/" + "Auto" + "Graph"
TEXT_SUFFIXES = {
    ".py", ".pyx", ".yaml", ".yml", ".toml", ".sh", ".json", ".txt", ".csv"
}
CORE_PATHS = (
    ROOT / "elda",
    ROOT / "circuit_kahypar",
    ROOT / "configs",
    ROOT / "scripts",
    ROOT / "tools",
    ROOT / "tests",
    ROOT / "skills",
    ROOT / "paper" / "scripts",
    ROOT / "train.py",
    ROOT / "test.py",
    ROOT / "test_conditional_generation.py",
    ROOT / "environment.yaml",
    ROOT / "pyproject.toml",
)


def iter_text_files(path: Path):
    if path.is_file():
        if path.suffix in TEXT_SUFFIXES:
            yield path
        return
    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.suffix in TEXT_SUFFIXES:
            yield candidate


def contains_bytes(path: Path, needle: bytes) -> bool:
    overlap = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            window = overlap + chunk.lower()
            if needle in window:
                return True
            overlap = window[-max(len(needle) - 1, 0) :]
    return False


def audit(*, check_checkpoint_payloads: bool = True) -> list[str]:
    errors: list[str] = []
    if (ROOT / LEGACY).exists():
        errors.append("legacy top-level Python package directory exists")

    import_pattern = re.compile(
        rf"(?:^|\s)(?:from|import)\s+{re.escape(LEGACY)}(?:\.|\s|$)", re.MULTILINE
    )
    target_pattern = re.compile(rf"_target_\s*:\s*{re.escape(LEGACY)}\.")
    for root in CORE_PATHS:
        for path in iter_text_files(root):
            text = path.read_text(encoding="utf-8", errors="replace")
            if import_pattern.search(text):
                errors.append(f"legacy implementation import: {path.relative_to(ROOT)}")
            if target_pattern.search(text):
                errors.append(f"legacy Hydra target: {path.relative_to(ROOT)}")
            if OLD_ABSOLUTE_ROOT in text:
                errors.append(f"legacy absolute project path: {path.relative_to(ROOT)}")

    for path in (ROOT / "logs" / "train").rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".yaml", ".yml", ".json", ".txt", ".csv"}:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if LEGACY in text:
                errors.append(f"legacy name in retained training metadata: {path.relative_to(ROOT)}")
        elif (
            check_checkpoint_payloads
            and path.suffix == ".ckpt"
            and contains_bytes(path, LEGACY.encode("ascii"))
        ):
            errors.append(f"legacy name embedded in checkpoint: {path.relative_to(ROOT)}")

    migration = ROOT / "release" / "checkpoint_namespace_migration.json"
    if not migration.is_file():
        errors.append("missing checkpoint namespace-migration manifest")
    else:
        payload = json.loads(migration.read_text(encoding="utf-8"))
        for row in payload.get("checkpoints", []):
            if row.get("tensor_bytes_unchanged") is not True:
                errors.append(f"checkpoint tensor equivalence not verified: {row.get('path')}")
    return errors


def main() -> int:
    errors = audit()
    print(
        json.dumps(
            {
                "schema": "elda_public_namespace_audit_v1",
                "status": "pass" if not errors else "fail",
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
