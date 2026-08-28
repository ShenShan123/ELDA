#!/usr/bin/env python3
"""Verify that the executable supplement is released only under ELDA identity."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TEXT_SUFFIXES = {
    ".py", ".sh", ".yaml", ".yml", ".json", ".csv", ".toml", ".md", ".txt", ".v"
}
EXECUTABLE_SCOPES = ("src", "scripts", "configs", "tests", "data_sample", "expected_outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures = []
    comparison_mentions = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(root)
        if rel == Path("scripts/audit_release_identity.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\bpanda\b", text, flags=re.I):
            failures.append(f"obsolete method name in {rel}")
        if re.search(r"autograph", text, flags=re.I):
            if rel.parts and rel.parts[0] in EXECUTABLE_SCOPES:
                failures.append(f"comparison name leaked into executable scope: {rel}")
            else:
                comparison_mentions.append(str(rel))
        if rel.suffix == ".py" and re.search(r"(?:from|import)\s+autograph\b", text):
            failures.append(f"obsolete Python import in {rel}")
    if not (root / "src" / "elda" / "__init__.py").is_file():
        failures.append("missing ELDA package namespace")
    if failures:
        print("Release identity audit: FAIL")
        for item in failures:
            print(f"  {item}")
        raise SystemExit(1)
    print("Release identity audit: PASS")
    print("Method/package identity: ELDA / elda")
    print("Comparison-name files:")
    for name in sorted(set(comparison_mentions)):
        print(f"  {name}")


if __name__ == "__main__":
    main()
