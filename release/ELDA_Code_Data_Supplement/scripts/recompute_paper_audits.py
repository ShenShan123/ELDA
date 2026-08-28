#!/usr/bin/env python3
"""Recompute compact arithmetic audits from included frozen paper records."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper_results"


def rows(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    checks = []
    for row in rows("table_e3_repair_components.csv"):
        repr_sum = sum(float(row[key]) for key in ("role", "type", "pin", "assign"))
        struct_sum = sum(float(row[key]) for key in ("syn_in", "syn_out", "stub", "drop"))
        checks.append(
            {
                "check": f"E3 component closure: {row['method']}",
                "pass": abs(repr_sum - float(row["r_repr"])) < 1e-9
                and abs(struct_sum - float(row["r_struct"])) < 1e-9,
                "recomputed_r_repr": repr_sum,
                "recomputed_r_struct": struct_sum,
            }
        )
    for row in rows("table_e4b_assembly_components.csv"):
        component_keys = [
            key for key in row
            if key.endswith("_mean") and key not in {"a_asm_mean"}
        ]
        total = sum(float(row[key]) for key in component_keys)
        checks.append(
            {
                "check": f"E4 assembly component closure: {row['method']}",
                "pass": abs(total - float(row["a_asm_mean"])) < 1e-9,
                "recomputed_a_asm_mean": total,
            }
        )
    stats = rows("dataset_statistics.csv")
    split_rows = stats[:-1]
    total = stats[-1]
    checks.append(
        {
            "check": "dataset split-count closure",
            "pass": sum(int(row["Gate-level subcircuits"]) for row in split_rows)
            == int(total["Gate-level subcircuits"]),
            "recomputed_total": sum(int(row["Gate-level subcircuits"]) for row in split_rows),
        }
    )
    graph_rows = rows("table1_graph_quality.csv")
    checks.append(
        {
            "check": "main graph table has seven frozen methods",
            "pass": len(graph_rows) == 7,
            "row_count": len(graph_rows),
        }
    )
    result = {
        "scope": "Arithmetic closure over included compact frozen records; raw 1024-attempt and full-corpus artifacts are not included.",
        "all_pass": all(row["pass"] for row in checks),
        "checks": checks,
    }
    output = RESULTS / "recomputed_compact_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
