#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from elda_paths import PAPER_ROOT

ROOT = PAPER_ROOT
OUT = ROOT / "reports/final_baseline/phase10_8_endpoint_fidelity_main_table"
REPAIR_ROOT = (
    ROOT
    / "reports/final_baseline/deterministic_repair_burden_20260726_multiout"
)
NEUTRAL_SUMMARY = (
    ROOT
    / "reports/final_baseline/neutral_raw_netlist_audit_20260630"
    / "neutral_raw_netlist_baseline_summary.json"
)
NATIVE_PIPELINE = OUT / "table3_elda_native_object_pipeline.csv"
REPAIR_SUMMARY = REPAIR_ROOT / "deterministic_repair_burden_summary.json"
INCIDENCE_SUMMARY = REPAIR_ROOT / "incidence_retention_summary.csv"
REPRESENTATION_SUMMARY = (
    ROOT
    / "results/elda/representation_ablation"
    / "summary_current/representation_ablation_current_summary.json"
)
DECODER_SUMMARY = (
    ROOT
    / "results/elda/constraint_ablation"
    / "summary_final/decoder_ablation_final_summary.json"
)

GENERIC_ATTEMPTS = {
    "AutoGraph": REPAIR_ROOT / "models/autograph/attempts.json",
    "G2PT": REPAIR_ROOT / "models/g2pt/attempts.json",
    "DiGress": REPAIR_ROOT / "models/digress/attempts.json",
}
REPR_FIELDS = (
    "node_role_imputation_count",
    "cell_type_imputation_count",
    "pin_assignment_count",
    "source_load_assignment_count",
)
STRUCT_FIELDS = (
    "synthetic_net_count",
    "dropped_incidence_count",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_demands(row: dict[str, Any]) -> int:
    if row.get("required_demand_count") is not None:
        return int(row["required_demand_count"])
    return max(
        0,
        int(row.get("pin_assignment_count", 0))
        - int(row.get("cell_count", 0)),
    )


def aggregate_attempts(path: Path) -> dict[str, float]:
    rows = load_json(path)
    if len(rows) != 1024:
        raise RuntimeError(f"{path}: expected 1024 attempts, got {len(rows)}")
    repr_total = 0
    struct_total = 0
    repair_total = 0
    demand_total = 0
    observed = 0
    for row in rows:
        if row.get("repair_op_count") is None:
            continue
        row_repr = sum(int(row.get(field, 0) or 0) for field in REPR_FIELDS)
        row_struct = sum(
            int(row.get(field, 0) or 0) for field in STRUCT_FIELDS
        )
        row_total = int(row["repair_op_count"])
        if row_repr + row_struct != row_total:
            raise RuntimeError(
                f"{path}: attempt {row.get('sample_id')} does not reconcile"
            )
        repr_total += row_repr
        struct_total += row_struct
        repair_total += row_total
        demand_total += required_demands(row)
        observed += 1
    return {
        "attempt_count": float(len(rows)),
        "observed_count": float(observed),
        "r_repr": repr_total / len(rows),
        "r_struct": struct_total / len(rows),
        "r_total": repair_total / len(rows),
        "repair_per_100_demands": (
            100.0 * repair_total / demand_total if demand_total else 0.0
        ),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_table3(rows: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "object_complete",
        "native_verilog",
        "elda_valid",
        "native_yosys",
        "r_repr",
        "r_struct",
        "repair_per_100_demands",
        "incidence_retention",
        "final_yosys",
    ]
    write_csv(OUT / "table3_native_materializability_repair_burden.csv", fields, rows)
    lines = [
        "# Table 3. Native materializability and repair burden over 1,024 one-shot attempts",
        "",
        "| Method | Object complete ↑ | Verilog emitted ↑ | ELDA-valid ↑ | "
        "Native Yosys ↑ | R_repr ↓ | R_struct ↓ | Repair/100D ↓ | "
        "Inc. retention ↑ | Final Yosys ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['object_complete']:.4f} | "
            f"{row['native_verilog']:.4f} | {row['elda_valid']:.4f} | "
            f"{row['native_yosys']:.4f} | {row['r_repr']:.4f} | "
            f"{row['r_struct']:.4f} | "
            f"{row['repair_per_100_demands']:.4f} | "
            f"{row['incidence_retention']:.4f} | "
            f"{row['final_yosys']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`R_repr` includes node-role, cell-type, pin, and source/load "
            "assignment reconstruction. `R_struct` includes synthetic-net "
            "creation and dropped incidence. `Repair/100D` is the micro-average "
            "number of repair audit operations per 100 realized Liberty input-pin "
            "demands. Final Yosys is measured after deterministic repair.",
        ]
    )
    (OUT / "table3_native_materializability_repair_burden.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_simple_table(
    path_stem: str,
    headers: list[str],
    fields: list[str],
    rows: list[dict[str, Any]],
) -> None:
    write_csv(OUT / f"{path_stem}.csv", fields, rows)
    lines = [
        f"# {headers[0]}",
        "",
        "| " + " | ".join(headers[1:]) + " |",
        "|" + "---|" * (len(headers) - 1),
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    (OUT / f"{path_stem}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    neutral = {
        row["model"]: row for row in load_json(NEUTRAL_SUMMARY)["models"]
    }
    native = {row["method"]: row for row in read_csv(NATIVE_PIPELINE)}
    repair = {
        row["model"].split(" (", 1)[0]: row
        for row in load_json(REPAIR_SUMMARY)["models"]
    }
    incidence = {
        row["Method"]: float(row["Incidence retention micro"])
        for row in read_csv(INCIDENCE_SUMMARY)
    }

    table3_rows: list[dict[str, Any]] = []
    raw_aggregates: dict[str, Any] = {}
    for method in ("AutoGraph", "G2PT", "DiGress"):
        agg = aggregate_attempts(GENERIC_ATTEMPTS[method])
        raw_aggregates[method] = agg
        summary = repair[method]
        for field in ("repair_ops_per_100_demands",):
            if abs(
                agg["repair_per_100_demands"] - float(summary[field])
            ) > 1e-12:
                raise RuntimeError(f"{method}: {field} mismatch")
        if abs(agg["r_total"] - float(summary["repair_op_count_per_attempt"])) > 1e-12:
            raise RuntimeError(f"{method}: repair total mismatch")
        table3_rows.append(
            {
                "method": method,
                "object_complete": float(
                    neutral[method]["canonical_ir_success_rate"]
                ),
                "native_verilog": float(
                    neutral[method]["raw_netlist_materialized_rate"]
                ),
                "elda_valid": 0.0,
                "native_yosys": float(
                    neutral[method]["yosys_check_pass_rate"]
                ),
                "r_repr": agg["r_repr"],
                "r_struct": agg["r_struct"],
                "repair_per_100_demands": agg[
                    "repair_per_100_demands"
                ],
                "incidence_retention": incidence[method],
                "final_yosys": float(
                    summary["repaired_yosys_check_pass_rate"]
                ),
                "fanout_tv": float(
                    summary["repaired_source_type_fanout_TV"]
                ),
            }
        )
    elda_native = native["ELDA"]
    elda_repair = repair["ELDA"]
    table3_rows.append(
        {
            "method": "ELDA",
            "object_complete": float(
                elda_native["native_object_complete_rate"]
            ),
            "native_verilog": float(elda_native["raw_verilog_rate"]),
            "elda_valid": float(elda_native["strict_valid_rate"]),
            "native_yosys": float(elda_native["yosys_check_rate"]),
            "r_repr": 0.0,
            "r_struct": 0.0,
            "repair_per_100_demands": 0.0,
            "incidence_retention": 1.0,
            "final_yosys": float(
                elda_repair["repaired_yosys_check_pass_rate"]
            ),
            "fanout_tv": float(
                elda_repair["repaired_source_type_fanout_TV"]
            ),
        }
    )
    write_table3(table3_rows)

    representation = {
        row["variant"]: row
        for row in load_json(REPRESENTATION_SUMMARY)["rows"]
    }
    representation_rows = []
    removed_labels = {
        "R0 Full ELDA": "None",
        "R1 No boundary identity": "Boundary endpoint identity",
        "R2 No per-source budget": "Fanout bucket and per-source capacity",
        "R3 No full-load assignment": "Source-to-demand assignment",
        "R4 Cell-level demand": "Liberty input-pin identity",
    }
    for variant in (
        "R0 Full ELDA",
        "R1 No boundary identity",
        "R2 No per-source budget",
        "R3 No full-load assignment",
        "R4 Cell-level demand",
    ):
        row = representation[variant]
        representation_rows.append(
            {
                "variant": "ELDA" if variant.startswith("R0 ") else variant,
                "removed_information": removed_labels[variant],
                "pin_slot_recoverable": float(
                    row["pin_slot_recoverable_rate"]
                ),
                "full_assignment_recoverable": float(
                    row["full_load_recoverable_rate"]
                ),
                "elda_valid": float(row["structural_strict_rate"]),
            }
        )
    write_simple_table(
        "table4_representation_ablation_compact",
        [
            "Representation ablation",
            "Variant",
            "Removed information",
            "Pin-slot recoverable ↑",
            "Full-assignment recoverable ↑",
            "ELDA-valid ↑",
        ],
        [
            "variant",
            "removed_information",
            "pin_slot_recoverable",
            "full_assignment_recoverable",
            "elda_valid",
        ],
        representation_rows,
    )

    decoder = {
        row["id"]: row for row in load_json(DECODER_SUMMARY)["rows"]
    }
    decoder_rows = []
    for publication_id, source_id, disabled in (
        ("ELDA", "D0", "None"),
        ("A1", "D3", "Assignment-budget mask"),
        ("A2", "D5", "Completion-gated EOS"),
    ):
        row = decoder[source_id]
        decoder_rows.append(
            {
                "id": publication_id,
                "disabled_constraint": disabled,
                "assignment_complete": float(
                    row["full_load_coverage_rate"]
                ),
                "elda_valid": float(row["elda_valid_rate"]),
                "source_id": source_id,
            }
        )
    write_simple_table(
        "table5_decoder_ablation_compact",
        [
            "Decoder ablation",
            "ID",
            "Disabled constraint",
            "Assignment complete ↑",
            "ELDA-valid ↑",
        ],
        [
            "id",
            "disabled_constraint",
            "assignment_complete",
            "elda_valid",
        ],
        decoder_rows,
    )

    inputs = [
        NEUTRAL_SUMMARY,
        NATIVE_PIPELINE,
        REPAIR_SUMMARY,
        INCIDENCE_SUMMARY,
        REPRESENTATION_SUMMARY,
        DECODER_SUMMARY,
        *GENERIC_ATTEMPTS.values(),
    ]
    audit = {
        "status": "pass",
        "attempt_denominator": 1024,
        "table3_rows": table3_rows,
        "representation_ablation_rows": representation_rows,
        "decoder_ablation_rows": decoder_rows,
        "raw_repair_aggregates": raw_aggregates,
        "definitions": {
            "R_repr": list(REPR_FIELDS),
            "R_struct": list(STRUCT_FIELDS),
            "Repair_per_100D": (
                "100 * sum(R_repr + R_struct) / sum(required demands)"
            ),
            "Fanout_TV": "TV over P(source type, fanout bin)",
        },
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in inputs
        },
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "table3_and_ablation_reproducibility_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUT / "table3_and_ablation_reproducibility_audit.json")


if __name__ == "__main__":
    main()
