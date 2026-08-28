#!/usr/bin/env python3
"""Verify the frozen ELDA publication artifacts using the standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from audit_public_namespace import audit as audit_public_namespace


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
ATTEMPT_RE = re.compile(r"attempt_(\d{4})$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_attempts(errors: list[str], artifact_root: Path) -> dict[str, int]:
    manifest_path = RELEASE / "final_attempts_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_count = int(manifest["attempts_per_run"])
    expected_ids = set(range(expected_count))
    counts: dict[str, int] = {}

    for run in manifest["runs"]:
        run_id = run["id"]
        attempts_root = artifact_root / run["path"]
        if not attempts_root.is_dir():
            errors.append(f"{run_id}: missing attempts directory: {run['path']}")
            continue
        observed_ids = {
            int(match.group(1))
            for path in attempts_root.iterdir()
            if path.is_dir() and (match := ATTEMPT_RE.fullmatch(path.name))
        }
        counts[run_id] = len(observed_ids)
        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)
            extra = sorted(observed_ids - expected_ids)
            errors.append(
                f"{run_id}: expected attempt_0000..attempt_1023; "
                f"count={len(observed_ids)}, missing={missing[:8]}, extra={extra[:8]}"
            )

    for relative in manifest["required_table_artifacts"]:
        if not (artifact_root / relative).is_file():
            errors.append(f"missing table artifact: {relative}")
    return counts


def check_checkpoints(errors: list[str], artifact_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for line in (RELEASE / "checkpoints.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = artifact_root / relative
        if not path.is_file():
            errors.append(f"missing checkpoint: {relative}")
            continue
        actual = sha256(path)
        observed[relative] = actual
        if actual != expected:
            errors.append(
                f"checkpoint hash mismatch: {relative}: expected={expected}, actual={actual}"
            )
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-checkpoint-hashes",
        action="store_true",
        help="perform the fast attempt/table audit without reading all checkpoint bytes",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT,
        help=(
            "root containing checkpoints, paper/results, and paper/reports; "
            "defaults to the source checkout"
        ),
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.expanduser().resolve()

    errors: list[str] = []
    errors.extend(
        audit_public_namespace(
            check_checkpoint_payloads=not args.skip_checkpoint_hashes
        )
    )
    counts = check_attempts(errors, artifact_root)
    checkpoint_hashes = (
        {}
        if args.skip_checkpoint_hashes
        else check_checkpoints(errors, artifact_root)
    )
    result = {
        "schema": "elda_release_verification_v1",
        "status": "pass" if not errors else "fail",
        "source_root": str(ROOT),
        "artifact_root": str(artifact_root),
        "attempt_counts": counts,
        "checkpoint_hashes_checked": len(checkpoint_hashes),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
