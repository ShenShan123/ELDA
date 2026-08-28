#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from elda_paths import PAPER_ROOT

ROOT = PAPER_ROOT
OUT = ROOT / "reports/final_baseline/phase10_8_endpoint_fidelity_main_table"
REFERENCE_CACHE = OUT / "native_elda_reference_rows_dedup11390.jsonl"
METRICS_OUT = OUT / "elda_policy_native_graph_metrics.json"
TRAIN_FINGERPRINTS = (
    ROOT
    / "reports/final_baseline/phase10_6_main_graph_quality_table"
    / "elda_train_clean_decoded_view_fingerprints.txt"
)
METHODS = {
    "ELDA_Unconstrained": ROOT / "results/elda/controls/unconstrained_lm/attempts",
    "ELDA_Syntax": ROOT / "results/elda/controls/syntax_only_lm/attempts",
    "Frequency sampler": ROOT / "results/elda/controls/field_frequency/attempts",
    "ELDA": ROOT / "results/elda/reference/attempts",
}

sys.path.insert(0, str(ROOT / "scripts"))
from compute_unified_dedup11390_reference_metrics import (  # noqa: E402
    dedup_reference_paths,
    structural_hash,
)
from compute_dedup_reference_sensitivity import (  # noqa: E402
    inspect_generated,
    inspect_reference,
    summarize,
)


def load_or_build_reference() -> list[dict[str, Any]]:
    if REFERENCE_CACHE.is_file():
        rows = [
            json.loads(line)
            for line in REFERENCE_CACHE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            row["fanout_hist"] = {
                tuple(key.split("|", 1)): value
                for key, value in row["fanout_hist"].items()
            }
        return rows
    paths = dedup_reference_paths()
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(inspect_reference, paths, chunksize=16))
    if len(rows) != 11390:
        raise RuntimeError(f"expected 11390 reference rows, got {len(rows)}")
    serializable_rows = []
    for row in rows:
        encoded = dict(row)
        encoded["fanout_hist"] = {
            "|".join(key): value for key, value in row["fanout_hist"].items()
        }
        serializable_rows.append(encoded)
    REFERENCE_CACHE.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in serializable_rows) + "\n",
        encoding="utf-8",
    )
    return rows


def report_rows(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for report_path in sorted(root.glob("attempt_*/report.json")):
        rows.append((report_path.parent, json.loads(report_path.read_text(encoding="utf-8"))))
    if len(rows) != 1024:
        raise RuntimeError(f"expected 1024 reports under {root}, got {len(rows)}")
    return rows


def native_object_complete(row: dict[str, Any]) -> bool:
    return bool(
        row.get("token_decode_valid")
        and row.get("section_complete")
        and row.get("EOS_validity")
        and row.get("cell_table_valid")
        and row.get("demand_table_valid")
        and row.get("source_table_valid")
        and row.get("source_load_full_coverage")
        and row.get("demand_exactly_once")
        and int(row.get("source_budget_violation") or 0) == 0
        and int(row.get("same_cell_same_net_reuse") or 0) == 0
    )


def rate(rows: list[tuple[Path, dict[str, Any]]], predicate) -> float:
    return sum(bool(predicate(path, row)) for path, row in rows) / len(rows)


def pipeline_summary(root: Path) -> dict[str, Any]:
    rows = report_rows(root)
    failures = Counter(str(row.get("failure_stage") or "success") for _, row in rows)
    dominant = failures.most_common(1)[0][0]
    return {
        "attempts": len(rows),
        "graph_valid_rate": rate(rows, lambda _p, r: r.get("token_decode_valid")),
        "cell_type_complete_rate": rate(rows, lambda _p, r: r.get("cell_table_valid")),
        "pin_identity_rate": rate(
            rows, lambda _p, r: r.get("demand_table_valid") and r.get("source_table_valid")
        ),
        "source_load_assignment_rate": rate(
            rows,
            lambda _p, r: r.get("source_load_full_coverage") and r.get("demand_exactly_once"),
        ),
        "native_object_complete_rate": rate(rows, lambda _p, r: native_object_complete(r)),
        "strict_valid_rate": rate(rows, lambda _p, r: r.get("strict_pass")),
        "raw_verilog_rate": rate(
            rows, lambda path, _r: (path / "candidate.verify_only.v").is_file()
        ),
        "yosys_check_rate": rate(rows, lambda _p, r: r.get("yosys_valid")),
        "main_failure": dominant,
        "failure_stage_breakdown": dict(failures.most_common()),
    }


def generated_pairs(root: Path) -> list[tuple[str, str]]:
    pairs = []
    for attempt in sorted(root.glob("attempt_*")):
        graph = attempt / "decoded_graph.pt"
        payload = attempt / "payload.json"
        if graph.is_file() and payload.is_file():
            pairs.append((str(graph), str(payload)))
    return pairs


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_distribution_table(rows: list[dict[str, Any]]) -> None:
    fields = [
        "method", "native_object_n", "degree_MMD", "norm_components_W1", "lcc_W1", "fanout_TV"
    ]
    labels = [
        "Method", "Native object n", "Degree MMD ↓", "Rel. Components W1 ↓", "LCC W1 ↓", "Fanout TV ↓"
    ]
    with (OUT / "table1b_elda_native_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# ELDA native distribution comparison",
        "",
        "| " + " | ".join(labels) + " |",
        "|" + "---|" * len(labels),
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row[field]) for field in fields) + " |")
    lines.extend([
        "",
        "All four metrics use native ELDA decoded graphs/payloads and the same 11,390 "
        "development-deduplicated source-clean test reference. Distribution metrics are "
        "conditional on native-object-complete candidates; `Native object n` must be reported.",
        "",
        "Fanout TV is exact TV over `P(source kind, fanout bin)` and counts only load-demand endpoints.",
    ])
    (OUT / "table1b_elda_native_distribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pipeline_table(rows: list[dict[str, Any]]) -> None:
    fields = [
        "method", "graph_valid_rate", "cell_type_complete_rate", "pin_identity_rate",
        "source_load_assignment_rate", "native_object_complete_rate", "strict_valid_rate",
        "raw_verilog_rate", "yosys_check_rate", "main_failure",
    ]
    labels = [
        "Method", "Graph valid ↑", "Cell-type complete ↑", "Pin identity ↑",
        "Source/load assignment ↑", "Native object complete ↑", "Strict valid ↑",
        "Raw Verilog ↑", "Yosys check ↑", "Main failure",
    ]
    with (OUT / "table3_elda_native_object_pipeline.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Native ELDA object and materialization pipeline",
        "",
        "| " + " | ".join(labels) + " |",
        "|" + "---|" * len(labels),
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row[field]) for field in fields) + " |")
    lines.extend([
        "",
        "All rates use all 1,024 attempts as the denominator. Graph valid means strict ELDA token "
        "parse/decode. Native object complete additionally requires complete sections/EOS, valid "
        "CELL/DEMAND/SOURCE tables, full and exactly-once demand assignment, zero source-budget "
        "violation, and zero same-source reuse within a load cell.",
        "",
        "Raw Verilog is verify-only materialization with no repair/fallback. `Yosys check` uses the "
        "stored no-repair Yosys result; the invoked flow also performs synthesis and final check.",
    ])
    (OUT / "table3_elda_native_object_pipeline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = load_or_build_reference()
    train_fingerprints = {
        value.strip()
        for value in TRAIN_FINGERPRINTS.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    distribution_rows = []
    pipeline_rows = []
    artifact: dict[str, Any] = {
        "protocol": {
            "reference": "native ELDA development-deduplicated source-clean test set",
            "reference_count": len(reference),
            "attempt_denominator": 1024,
            "distribution_population": "native-object-complete payload/decoded-graph pairs",
        },
        "methods": {},
    }
    for method, root in METHODS.items():
        pairs = generated_pairs(root)
        with ProcessPoolExecutor(max_workers=8) as pool:
            generated = list(pool.map(inspect_generated, pairs, chunksize=16))
        metrics = summarize(reference, generated, "Native ELDA development-dedup-11,390")
        pipeline = pipeline_summary(root)
        signatures = [
            structural_hash(
                int(row["feature"]["node_count"]),
                row["feature"].get("labels"),
                row["feature"]["edges"],
            )
            for row in generated
        ]
        unique_signatures = set(signatures)
        novel_count = sum(
            signature not in train_fingerprints for signature in signatures
        )
        novel_unique_count = len(unique_signatures - train_fingerprints)
        diversity = {
            "valid_fingerprint_count": len(signatures),
            "unique_valid_fingerprint_count": len(unique_signatures),
            "train_fingerprint_count": len(train_fingerprints),
            "train_exact_copy_count": len(signatures) - novel_count,
            "novel_unique_count": novel_unique_count,
            "unique_rate_over_valid": (
                len(unique_signatures) / len(signatures)
                if signatures
                else None
            ),
            "novelty_rate_over_valid": (
                novel_count / len(signatures) if signatures else None
            ),
            "VUN_rate_over_attempts": novel_unique_count / 1024,
        }
        artifact["methods"][method] = {
            "distribution": metrics,
            "pipeline": pipeline,
            "diversity": diversity,
        }
        distribution_rows.append({
            "method": method,
            "native_object_n": len(generated),
            "degree_MMD": metrics["degree_MMD"],
            "norm_components_W1": metrics["norm_components_W1"],
            "lcc_W1": metrics["lcc_W1"],
            "fanout_TV": metrics["fanout_TV"],
        })
        pipeline_rows.append({"method": method, **pipeline})
    METRICS_OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_distribution_table(distribution_rows)
    write_pipeline_table(pipeline_rows)


if __name__ == "__main__":
    main()
