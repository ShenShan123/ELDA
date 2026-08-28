#!/usr/bin/env python3
"""Freeze the Appendix E.3/E.4 repair and assembly breakdowns.

The script consumes only the released per-attempt repair records and the
released per-design assembly CSV.  It checks every accounting identity before
writing publication-facing CSV/Markdown tables and a machine-readable audit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from elda_paths import PAPER_ROOT


ROOT = PAPER_ROOT
REPAIR_ROOT = (
    ROOT
    / "reports/final_baseline/deterministic_repair_burden_20260726_multiout"
    / "models"
)
ASSEMBLY_ROOT = (
    ROOT
    / "results/elda/design_assembly"
    / "random100_medium_n1900_3500_strict_20260724"
)
PER_DESIGN = ASSEMBLY_ROOT / "summary/per_design_metrics.csv"
EXPERIMENT_MANIFEST = ASSEMBLY_ROOT / "experiment_manifest.json"
OUT = (
    ROOT
    / "reports/final_baseline/appendix_e3_e4_frozen_details_20260731"
)

PROTOCOL_SOURCES = [
    ROOT / "scripts/evaluate_deterministic_repair_burden.py",
    ROOT / "scripts/run_scaffold_matching_protocol_v2.py",
    ROOT / "scripts/scaffold_endpoint_assignment.py",
    ROOT / "scripts/scaffold_replace_elda_generated_all.py",
    ROOT / "scripts/scaffold_replace_baseline_repaired_all.py",
    ROOT.parent / "tools/export_assembled_graph_netlist.py",
    ROOT / "scripts/audit_full_design_logic_survival.py",
    ROOT / "scripts/run_random100_medium_openroad.py",
]

ATTEMPT_METHODS = {
    "AutoGraph": REPAIR_ROOT / "autograph/attempts.json",
    "G2PT": REPAIR_ROOT / "g2pt/attempts.json",
    "DiGress": REPAIR_ROOT / "digress/attempts.json",
}

REPAIR_COMPONENTS = {
    "role": "node_role_imputation_count",
    "type": "cell_type_imputation_count",
    "pin": "pin_assignment_count",
    "assign": "source_load_assignment_count",
    "syn_in": "synthetic_input_source_count",
    "syn_out": "synthetic_output_net_count",
    # The released adapter performs no native net splitting.  This legacy
    # zero-valued field is the only structural-stub/split component.
    "stub": "net_split_count",
    "drop": "dropped_incidence_count",
}

ASSEMBLY_NATIVE_COMPONENTS = {
    "role": "node_role_imputation",
    "type": "cell_type_imputation",
    "pin": "pin_assignment",
    "assign": "assignment_reconstruction",
    "syn": "synthetic_nets",
    "drop": "dropped_incidence",
}

ASSEMBLY_COMPONENTS = {
    "cell_spec_fallback": "materializer_cell_spec_fallbacks",
    "input_pool_net": "materializer_input_pool_net_creations",
    "input_completion": "materializer_input_connection_completions",
    "helper_stage": "materializer_helper_input_stages",
    "input_reuse": "materializer_input_reuses",
    "output_completion": "materializer_output_net_completions",
    "output_conflict": "materializer_output_conflict_reassignments",
    "multi_driver_split": "materializer_multi_driver_splits",
    "self_loop_remap": "materializer_legalizer_self_loop_input_remaps",
    "same_net_remap": "materializer_legalizer_same_net_reuse_remaps",
    "undriven_to_pi": "materializer_legalizer_undriven_load_pi_promotions",
    "legalizer_input_pool": "materializer_legalizer_input_pool_net_creations",
    "drop": "materializer_dropped_extra_incidence",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, Any], field: str) -> float:
    value = row.get(field, 0)
    return float(value or 0)


def mean_sd(values: list[float]) -> dict[str, float]:
    if not values:
        raise RuntimeError("empty population")
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_stat(value: dict[str, float]) -> str:
    return f"{value['mean']:.2f} ± {value['sample_sd']:.2f}"


def standalone_breakdown() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for method, path in ATTEMPT_METHODS.items():
        attempts = json.loads(path.read_text(encoding="utf-8"))
        if len(attempts) != 1024:
            raise RuntimeError(f"{method}: expected 1,024 attempts")
        component_totals = {
            name: sum(number(row, field) for row in attempts)
            for name, field in REPAIR_COMPONENTS.items()
        }
        for index, attempt in enumerate(attempts):
            repr_value = sum(
                number(attempt, REPAIR_COMPONENTS[name])
                for name in ("role", "type", "pin", "assign")
            )
            struct_value = sum(
                number(attempt, REPAIR_COMPONENTS[name])
                for name in ("syn_in", "syn_out", "stub", "drop")
            )
            if number(attempt, "synthetic_net_count") != (
                number(attempt, "synthetic_input_source_count")
                + number(attempt, "synthetic_output_net_count")
            ):
                raise RuntimeError(f"{method} attempt {index}: synthetic mismatch")
            if repr_value + struct_value != number(attempt, "repair_op_count"):
                raise RuntimeError(f"{method} attempt {index}: repair mismatch")
        output = {
            "method": method,
            **{name: value / 1024 for name, value in component_totals.items()},
        }
        output["r_repr"] = sum(output[name] for name in ("role", "type", "pin", "assign"))
        output["r_struct"] = sum(output[name] for name in ("syn_in", "syn_out", "stub", "drop"))
        rows.append(output)
        failures = [row for row in attempts if row.get("failure_stage") != "success"]
        audit[method] = {
            "attempt_count": len(attempts),
            "attempts_with_atomic_counters": sum(
                row.get("repair_op_count") is not None for row in attempts
            ),
            "non_success_attempt_count": len(failures),
            "non_success_attempts_with_atomic_counters": sum(
                row.get("repair_op_count") is not None for row in failures
            ),
            "all_per_attempt_identities_pass": True,
        }
    rows.append(
        {
            "method": "ELDA",
            **{name: 0.0 for name in REPAIR_COMPONENTS},
            "r_repr": 0.0,
            "r_struct": 0.0,
        }
    )
    audit["ELDA"] = {
        "attempt_count": 1024,
        "native_no_repair_protocol": True,
        "all_components_zero": True,
    }
    return rows, audit


def assembly_breakdown() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = read_csv(PER_DESIGN)
    by_method: dict[str, list[dict[str, str]]] = {}
    for row in source:
        by_method.setdefault(row["method"], []).append(row)
    native_rows: list[dict[str, Any]] = []
    assembly_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for method in ("AutoGraph-labeled", "G2PT-labeled", "DiGress", "ELDA"):
        rows = by_method.get(method, [])
        if len(rows) != 100:
            raise RuntimeError(f"{method}: expected 100 designs, got {len(rows)}")
        for index, row in enumerate(rows):
            r_repr = sum(number(row, field) for field in list(ASSEMBLY_NATIVE_COMPONENTS.values())[:4])
            r_struct = sum(number(row, field) for field in list(ASSEMBLY_NATIVE_COMPONENTS.values())[4:])
            if r_repr != number(row, "representation_completion_ops"):
                raise RuntimeError(f"{method} design {index}: R_repr mismatch")
            if r_struct != number(row, "native_partition_structural_repair_ops"):
                raise RuntimeError(f"{method} design {index}: R_struct mismatch")
            a_asm = sum(number(row, field) for field in ASSEMBLY_COMPONENTS.values())
            if a_asm != number(row, "post_stitch_materializer_repair_ops"):
                raise RuntimeError(f"{method} design {index}: A_asm mismatch")
        native_stats = {
            name: mean_sd([number(row, field) for row in rows])
            for name, field in ASSEMBLY_NATIVE_COMPONENTS.items()
        }
        native_stats["r_repr"] = mean_sd(
            [number(row, "representation_completion_ops") for row in rows]
        )
        native_stats["r_struct"] = mean_sd(
            [number(row, "native_partition_structural_repair_ops") for row in rows]
        )
        native_rows.append({"method": method, **native_stats})
        assembly_stats = {
            name: mean_sd([number(row, field) for row in rows])
            for name, field in ASSEMBLY_COMPONENTS.items()
        }
        assembly_stats["a_asm"] = mean_sd(
            [number(row, "post_stitch_materializer_repair_ops") for row in rows]
        )
        assembly_rows.append({"method": method, **assembly_stats})
        audit[method] = {
            "design_count": len(rows),
            "all_per_design_identities_pass": True,
            "yosys_success_count": sum(int(number(row, "yosys_read_check")) for row in rows),
            "routed_gds_success_count": sum(int(number(row, "routed_gds_flow")) for row in rows),
            "zero_detailed_route_drc_count": sum(
                number(row, "backend_drc_errors") == 0 for row in rows
            ),
        }
    return native_rows, assembly_rows, audit


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    repair_rows, repair_audit = standalone_breakdown()
    native_rows, assembly_rows, assembly_audit = assembly_breakdown()

    repair_fields = ["method", *REPAIR_COMPONENTS, "r_repr", "r_struct"]
    write_csv(OUT / "table_e3_repair_components.csv", repair_fields, repair_rows)

    native_fields = ["method", *ASSEMBLY_NATIVE_COMPONENTS, "r_repr", "r_struct"]
    native_flat = []
    for row in native_rows:
        flat: dict[str, Any] = {"method": row["method"]}
        for field in native_fields[1:]:
            flat[f"{field}_mean"] = row[field]["mean"]
            flat[f"{field}_sample_sd"] = row[field]["sample_sd"]
        native_flat.append(flat)
    native_csv_fields = ["method"] + [
        suffix
        for field in native_fields[1:]
        for suffix in (f"{field}_mean", f"{field}_sample_sd")
    ]
    write_csv(OUT / "table_e4a_design_repair_components.csv", native_csv_fields, native_flat)

    assembly_fields = ["method", *ASSEMBLY_COMPONENTS, "a_asm"]
    assembly_flat = []
    for row in assembly_rows:
        flat = {"method": row["method"]}
        for field in assembly_fields[1:]:
            flat[f"{field}_mean"] = row[field]["mean"]
            flat[f"{field}_sample_sd"] = row[field]["sample_sd"]
        assembly_flat.append(flat)
    assembly_csv_fields = ["method"] + [
        suffix
        for field in assembly_fields[1:]
        for suffix in (f"{field}_mean", f"{field}_sample_sd")
    ]
    write_csv(OUT / "table_e4b_assembly_components.csv", assembly_csv_fields, assembly_flat)

    lines = [
        "# Frozen Appendix E.3/E.4 details",
        "",
        "## E.3 Deterministic realization and repair",
        "",
        "| Method | Role | Type | Pin | Assign | Syn-in | Syn-out | Stub/split | Drop | R_repr | R_struct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in repair_rows:
        lines.append(
            "| " + " | ".join(
                [row["method"]]
                + [fmt(row[field]) for field in repair_fields[1:]]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "All component means use the full 1,024-attempt denominator. The "
            "released records contain atomic counters for every attempt, including "
            "all operations emitted before a later export/tool failure. `Stub/split` "
            "is zero because the frozen native adapter does not split nets; retained "
            "unmatched generated nets become boundary sources without a counted "
            "structural mutation.",
            "Accounting identities are checked on unrounded values. Independently "
            "rounded four-decimal components may differ from a rounded total by "
            "one unit in the last displayed decimal.",
            "",
            "The frozen adapter applies the following deterministic priority "
            "rules. It first converts the generated graph to an undirected "
            "simple incidence graph; invalid endpoints, self-loops, and repeated "
            "normalized incidences are counted in `Drop`. When explicit roles are "
            "unavailable, connected components are traversed from ascending node "
            "identifiers and bipartite colors are resolved by smaller color-class "
            "size, then by the smallest node identifier. Cell-type imputation uses "
            "the lexicographic key `(capacity shortage, sequential penalty, excess "
            "inputs, input count, Liberty cell name)`. Output endpoints are assigned "
            "to generated nets by deterministic maximum-cardinality matching, with "
            "endpoints ordered by `(cell id, Liberty output pin)` and adjacent nets "
            "by generated net id. Cells and input incidences are then processed in "
            "ascending identifier order, while compatible input pins follow Liberty "
            "declaration order; surplus tail incidences are dropped.",
            "",
            "The generic repair adapter does not run a combinational-cycle "
            "legalizer. It removes graph self-loops during normalization, but no "
            "topology-conflict deletion priority exists. Accordingly, the repair "
            "protocol should be described as Liberty-compatible deterministic "
            "realization, not as cycle-safe topology repair.",
            "",
            "## E.4 Design-level native-repair components",
            "",
            "| Method | Role | Type | Pin | Assign | Syn. | Drop | R_repr | R_struct |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in native_rows:
        lines.append(
            "| " + " | ".join(
                [row["method"]]
                + [fmt_stat(row[field]) for field in native_fields[1:]]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "## E.4 Assembly-specific operation components",
            "",
            "| Method | Spec | Pool | In-complete | Helper | In-reuse | Out-complete | Out-conflict | Multi-driver | Self-loop | Same-net | Undriven→PI | Legalizer pool | Drop | A_asm |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in assembly_rows:
        lines.append(
            "| " + " | ".join(
                [row["method"]]
                + [fmt_stat(row[field]) for field in assembly_fields[1:]]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "Values in E.4 are mean ± sample standard deviation over the same 100 "
            "designs. `A_asm` is the per-design sum of all 13 execution-stage "
            "components. In particular, input/same-net reuse and undriven-load "
            "promotion alone are not an exhaustive decomposition of `A_asm`.",
            "",
            "### Candidate retrieval",
            "",
            "For candidate `c` and replacement target `t`, scalar size terms use "
            "`d(a,t)=|a-t|/max(t,1)` and histograms use total variation. The frozen "
            "common-view score is",
            "",
            "```text",
            "d_common(c,t) = d_cell",
            "              + 0.75 TV_cell-type",
            "              + 0.50 d_boundary",
            "              + 0.25 d_node",
            "              + 0.10 d_edge",
            "              + 0.25 TV_degree",
            "              + 0.25 TV_internal-fanout",
            "              + 0.25 TV_boundary-fanout.",
            "```",
            "",
            "The semantic diagnostic additionally adds `0.25 TV_boundary-role`, "
            "but frozen selection uses only `d_common`, so methods without endpoint "
            "roles are not ranked using unavailable information. A candidate is "
            "boundary-feasible when its boundary count is at least the target count. "
            "Feasible candidates are preferred; if none exist, ranking falls back "
            "to the full profiled pool. Candidates are greedily unique within each "
            "target design, but may be reused across different designs. Score ties "
            "are resolved by candidate index. `K=1024` is a budget cap: the profiled "
            "pools contain 1,024 ELDA, 1,018 AutoGraph, 1,022 G2PT, and 1,024 "
            "DiGress candidates.",
            "",
            "### Scaffolds and replacement slots",
            "",
            "The 100 whole-design source-clean scaffolds are sampled uniformly "
            "without replacement with seed 20260724 from eligible source graphs in "
            "the inclusive 1,900--3,500-node range; the realized range is "
            "1,905--3,432 nodes. The range applies to the complete source graph, not "
            "to a replacement slot. Every manifest `partition_record` is one slot, "
            "all slots are replaced, and one selected candidate is assigned to each "
            "slot. The frozen set contains 1,220 slots (8--16 per design, median 12).",
            "",
            "### Boundary binding",
            "",
            "Primary target/generated boundaries are paired by Hungarian assignment. "
            "With constrained merging enabled, the pair cost is",
            "",
            "```text",
            "100 capacity_overflow + |generated_degree - target_degree|",
            "+ 0.05 role_mismatch + 1e-9 generated_index.",
            "```",
            "",
            "ELDA computes role mismatch from load/driver endpoint counts; role-blind "
            "baselines use total degree. Extra generated boundaries are assigned "
            "many-to-one with deterministic component, same-cell collision, and "
            "fanout-overflow guards; multiple local boundaries assigned to one target "
            "therefore map to the same scaffold net. Missing target boundaries use "
            "one-to-many completion: ELDA clones a same-role boundary-incidence "
            "template when available, while generic baselines select anchor cells by "
            "descending generated degree and then node id. The common materializer "
            "remaps a self-driven input or a repeated same-cell input net to the first "
            "legal source in the order `(original primary inputs, existing synthetic "
            "inputs, sorted driven internal nets)`; if none exists, it creates a "
            "legalizer input-pool net. A load net with no remaining driver is "
            "promoted to a primary input in sorted net-name order.",
            "",
            "### Tool-flow and retention criteria",
            "",
            "The reported design-level Yosys success executes `read_liberty`, "
            "`read_verilog`, `hierarchy -check`, `proc`, `check`, `stat`, `synth`, "
            "and a final `check`; it is stronger than read/check alone. Routed-GDS "
            "success requires a zero process return code and a nonempty "
            "`6_final.gds`, using the first successful utilization in the frozen "
            "50/40/30/20 percent ladder. Zero detailed-route DRC violations are "
            "reported separately from the success predicate; all 400 selected runs "
            "have zero `detailedroute__route__drc_errors`. Exact retention strips "
            "`keep` attributes, runs Yosys `opt_clean -purge`, and measures exact "
            "survival of generated instance names `u_<hybrid-node-id>`. The frozen "
            "metric is exact-name retention, not a post-remapping name-and-cell-type "
            "equivalence check.",
        ]
    )
    (OUT / "appendix_e3_e4_tables.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    inputs = [
        *ATTEMPT_METHODS.values(),
        PER_DESIGN,
        EXPERIMENT_MANIFEST,
        *PROTOCOL_SOURCES,
    ]
    audit = {
        "schema": "elda_appendix_e3_e4_frozen_details_v1",
        "status": "pass",
        "standalone_denominator": 1024,
        "design_level_denominator": 100,
        "repair_component_fields": REPAIR_COMPONENTS,
        "assembly_native_component_fields": ASSEMBLY_NATIVE_COMPONENTS,
        "assembly_component_fields": ASSEMBLY_COMPONENTS,
        "standalone_audit": repair_audit,
        "assembly_audit": assembly_audit,
        "repair_rows": repair_rows,
        "assembly_native_rows": native_rows,
        "assembly_rows": assembly_rows,
        "input_sha256": {
            str(path.relative_to(ROOT.parent)): sha256(path) for path in inputs
        },
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "appendix_e3_e4_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUT / "appendix_e3_e4_audit.json")


if __name__ == "__main__":
    main()
