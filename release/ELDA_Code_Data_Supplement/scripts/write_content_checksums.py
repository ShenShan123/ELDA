#!/usr/bin/env python3
"""Write deterministic SHA-256 checksums for all packaged files."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "FILE_SHA256SUMS.txt"


def main():
    lines = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if (
            rel.startswith("outputs/")
            or "/__pycache__/" in f"/{rel}"
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} content checksums")


if __name__ == "__main__":
    main()
