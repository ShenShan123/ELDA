#!/usr/bin/env python3
"""Fail on private paths, identities, credentials, caches, or escaping links."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".sh", ".json", ".jsonl", ".csv", ".yaml",
    ".yml", ".toml", ".v", ".log", ".cfg", ".ini",
}
PATTERNS = {
    "absolute private path": re.compile(
        r"(?<![A-Za-z0-9_.-])(?:/(?:home|Users|data)/|[A-Za-z]:\\\\)", re.I
    ),
    "private host/user marker": re.compile(r"(?:memlab|yuany)", re.I),
    "credential assignment": re.compile(
        r"(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*['\"]?[^\s<]+",
        re.I,
    ),
    "email address": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
    "publication-status marker": re.compile(
        r"\b(?:submitted\s+to|under\s+review|"
        r"submission\s+(?:id|number)|paper\s+id)\b",
        re.I,
    ),
}
FORBIDDEN_NAMES = {".git", "__pycache__", ".pytest_cache", ".DS_Store"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if rel == Path("scripts/audit_privacy.py") or rel.parts[:1] == ("outputs",):
            continue
        if any(part in FORBIDDEN_NAMES for part in rel.parts):
            failures.append(f"forbidden path: {rel}")
        if path.is_symlink():
            target = path.resolve()
            try:
                target.relative_to(root)
            except ValueError:
                failures.append(f"escaping symlink: {rel} -> {target}")
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                # Third-party license notices may retain official public URLs,
                # but no private identity or email is needed in this package.
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{label}: {rel}:{line}")
    if failures:
        print("Privacy audit: FAIL")
        for item in failures:
            print(f"  {item}")
        raise SystemExit(1)
    print(f"Privacy audit: PASS ({root})")


if __name__ == "__main__":
    main()
