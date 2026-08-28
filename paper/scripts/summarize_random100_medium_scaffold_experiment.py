#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from repair_operation_accounting import (
    POST_STITCH_ACCOUNTING_PROTOCOL,
    post_stitch_operation_breakdown,
)

from elda_paths import PAPER_ROOT

ROOT = PAPER_ROOT
DEFAULT_EXPERIMENT = ROOT / (
    "results/elda/design_assembly/"
    "random100_medium_n1900_3500_strict_20260724"
)
REPAIR_ROOT = ROOT / (
    "reports/final_baseline/deterministic_repair_burden_20260630/models"
)
METHODS = (
    ("AutoGraph-labeled", "AutoGraph", "autograph"),
    ("G2PT-labeled", "G2PT", "g2pt"),
    ("DiGress", "DiGress", "digress"),
    ("ELDA", "ELDA", "elda"),
)
_ORFS_VALUE = os.environ.get("ELDA_ORFS_ROOT") or os.environ.get(
    "OPENROAD_FLOW_ROOT"
)
ORFS = Path(_ORFS_VALUE).expanduser().resolve() if _ORFS_VALUE else None
DIEAREA_RE = re.compile(
    r"DIEAREA\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+"
    r"\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;"
)


def load_openroad_metrics(design_dir: Path, method: str) -> dict[str, Any]:
    result = {
        "routed_gds_flow": None,
        "backend_core_utilization": None,
        "backend_elapsed_seconds": None,
        "backend_gds_bytes": None,
        "backend_die_area_um2": None,
        "backend_drc_errors": None,
        "backend_wirelength": None,
        "backend_vias": None,
        "backend_tns": None,
        "backend_wns": None,
        "backend_worst_slack": None,
    }
    status_path = design_dir / "openroad" / method / "openroad_status.json"
    if not status_path.is_file():
        return result
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result
    result["routed_gds_flow"] = int(status.get("status") == "complete")
    selected = status.get("selected_attempt")
    if not isinstance(selected, dict):
        return result
    result["backend_core_utilization"] = selected.get("core_utilization")
    result["backend_elapsed_seconds"] = status.get("elapsed_seconds")
    result["backend_gds_bytes"] = selected.get("gds_bytes")
    top = str(selected["top"])
    variant = str(selected["flow_variant"])
    if ORFS is None:
        return result
    route_json = ORFS / f"logs/nangate45/{top}/{variant}/5_2_route.json"
    final_def = ORFS / f"results/nangate45/{top}/{variant}/6_final.def"
    finish = ORFS / f"reports/nangate45/{top}/{variant}/6_finish.rpt"
    if route_json.is_file():
        route = json.loads(route_json.read_text(encoding="utf-8"))
        result["backend_drc_errors"] = route.get(
            "detailedroute__route__drc_errors"
        )
        result["backend_wirelength"] = route.get(
            "detailedroute__route__wirelength"
        )
        result["backend_vias"] = route.get("detailedroute__route__vias")
    if final_def.is_file():
        match = DIEAREA_RE.search(final_def.read_text(encoding="utf-8"))
        if match:
            x1, y1, x2, y2 = (int(value) for value in match.groups())
            result["backend_die_area_um2"] = (
                (x2 - x1) * (y2 - y1) / (2000.0 * 2000.0)
            )
    if finish.is_file():
        text = finish.read_text(encoding="utf-8")
        for field, pattern in (
            ("backend_tns", r"\btns\s+(-?\d+(?:\.\d+)?)"),
            ("backend_wns", r"\bwns\s+(-?\d+(?:\.\d+)?)"),
            (
                "backend_worst_slack",
                r"\bworst slack\s+(-?\d+(?:\.\d+)?)",
            ),
        ):
            match = re.search(pattern, text)
            if match:
                result[field] = float(match.group(1))
    return result


def load_survival_metrics(design_dir: Path, method: str) -> dict[str, Any]:
    result = {
        "no_keep_exact_instance_retention": None,
        "observable_state_rate": None,
        "generated_state_cells": None,
        "natural_output_lineage_available": False,
    }
    report_path = design_dir / "survival" / method / "no_keep_survival.json"
    if not report_path.is_file():
        return result
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result
    if report.get("status") != "complete":
        return result
    for key in result:
        result[key] = report.get(key)
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaffold_metadata_restoration_audit(experiment: dict[str, Any]) -> dict[str, Any]:
    """Audit common retained-scaffold annotation recovery across all methods."""
    metric_fields = (
        "edge_count",
        "role_known_edge_count",
        "pin_known_edge_count",
        "missing_metadata_edge_count",
        "ambiguous_collapsed_pin_edge_count",
        "inout_collapsed_edge_count",
    )
    report_locations = (
        (
            "ELDA",
            Path("common_k1024/elda/scaffold_generated_replace_all_report.json"),
        ),
        (
            "AutoGraph-labeled",
            Path(
                "common_k1024/baselines/autograph/"
                "baseline_repaired_scaffold_report.json"
            ),
        ),
        (
            "G2PT-labeled",
            Path(
                "common_k1024/baselines/g2pt/"
                "baseline_repaired_scaffold_report.json"
            ),
        ),
        (
            "DiGress",
            Path(
                "common_k1024/baselines/digress/"
                "baseline_repaired_scaffold_report.json"
            ),
        ),
    )
    rows = []
    missing_reports = []
    per_design_signatures: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    unique_design_metadata: dict[str, dict[str, Any]] = {}
    for design in experiment["designs"]:
        design_id = str(design["design_id"])
        design_dir = Path(design["design_dir"])
        for method, relative_path in report_locations:
            path = design_dir / relative_path
            if not path.is_file():
                missing_reports.append(
                    {"design_id": design_id, "method": method, "path": str(path)}
                )
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            metadata = dict(
                report.get("retained_scaffold_metadata_reconstruction", {})
            )
            row = {
                "design_id": design_id,
                "method": method,
                "report_path": str(path),
                "graph_match_ok": bool(metadata.get("graph_match_ok")),
                **{
                    field: int(metadata.get(field, 0) or 0)
                    for field in metric_fields
                },
            }
            rows.append(row)
            signature = tuple(row[field] for field in metric_fields)
            per_design_signatures[design_id].add(signature)
            unique_design_metadata.setdefault(design_id, row)

    total = {
        field: sum(int(row[field]) for row in rows)
        for field in metric_fields
    }
    unique_total = {
        field: sum(
            int(row[field]) for row in unique_design_metadata.values()
        )
        for field in metric_fields
    }
    return {
        "protocol": "retained_scaffold_metadata_restoration_audit_v1",
        "scope": (
            "retained source-scaffold edges only; generated candidate objects "
            "are not inputs to metadata restoration"
        ),
        "operation": (
            "attach role and Liberty-pin annotations to existing source edges "
            "after exact graph-contract matching"
        ),
        "repair_classification": "common preprocessing; excluded from repair burden",
        "topology_mutation": {
            "nodes_added_or_removed": 0,
            "edges_added_or_removed": 0,
            "nets_created": 0,
            "driver_connectivity_changes": 0,
            "required_pin_completions": 0,
            "evidence": (
                "restore_source_edge_metadata returns annotation tensors and a "
                "report only; it does not return or mutate a graph"
            ),
        },
        "expected_report_count": len(experiment["designs"]) * len(report_locations),
        "observed_report_count": len(rows),
        "missing_reports": missing_reports,
        "design_count": len(experiment["designs"]),
        "graph_match_ok_count": sum(
            int(row["graph_match_ok"]) for row in rows
        ),
        "cross_method_metadata_mismatch_design_count": sum(
            len(signatures) != 1
            for signatures in per_design_signatures.values()
        ),
        "all_method_call_totals": total,
        "unique_design_totals": unique_total,
        "ambiguity_policy": (
            "when the exact contract exposes both load and driver candidates "
            "for one incidence, deterministically retain the first sorted load "
            "pin and record the collapse"
        ),
        "per_call": rows,
    }


def load_attempts() -> dict[str, list[dict[str, Any]]]:
    return {
        model: json.loads((REPAIR_ROOT / model / "attempts.json").read_text(encoding="utf-8"))
        for model in ("autograph", "g2pt", "digress")
    }


def selected_repair(
    selection: dict[str, Any],
    selection_key: str,
    attempts: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    if selection_key == "elda":
        return {
            "node_role_imputation": 0,
            "cell_type_imputation": 0,
            "pin_assignment": 0,
            "assignment_reconstruction": 0,
            "synthetic_nets": 0,
            "dropped_incidence": 0,
            "retained_incidence": 0,
        }
    totals = defaultdict(int)
    source = attempts[selection_key]
    for selected in selection["models"][selection_key]["selected"]:
        row = source[int(selected["candidate_index"])]
        totals["node_role_imputation"] += int(
            row.get("node_role_imputation_count", 0) or 0
        )
        totals["cell_type_imputation"] += int(
            row.get("cell_type_imputation_count", 0) or 0
        )
        totals["pin_assignment"] += int(row.get("pin_assignment_count", 0) or 0)
        totals["assignment_reconstruction"] += int(
            row.get("source_load_assignment_count", 0) or 0
        )
        totals["synthetic_nets"] += int(row.get("synthetic_net_count", 0) or 0)
        totals["dropped_incidence"] += int(
            row.get("dropped_incidence_count", 0) or 0
        )
        totals["retained_incidence"] += int(
            row.get("retained_incidence_count", 0) or 0
        )
        totals["repair_record_failure"] += int(
            not bool(row.get("repair_success", False))
        )
    return dict(totals)


def mean_sd(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    return {
        "n": len(clean),
        "mean": statistics.mean(clean) if clean else None,
        "sample_std": statistics.stdev(clean) if len(clean) > 1 else (0.0 if clean else None),
    }


def format_stat(stat: dict[str, Any], digits: int = 4) -> str:
    if stat["mean"] is None:
        return "N/A"
    return f"{stat['mean']:.{digits}f} ± {stat['sample_std']:.{digits}f}"


def distribution_w1_with_paired_bootstrap(
    generated: list[float],
    reference: list[float],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20_260_729,
) -> dict[str, Any]:
    """Return empirical 1-D W1 and paired-design bootstrap uncertainty."""
    generated_array = np.asarray(generated, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if generated_array.shape != reference_array.shape or generated_array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "sample_std": None,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "uncertainty": "paired_bootstrap_standard_deviation",
        }
    point = float(
        np.mean(
            np.abs(
                np.sort(generated_array)
                - np.sort(reference_array)
            )
        )
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        generated_array.size,
        size=(bootstrap_samples, generated_array.size),
    )
    bootstrap = np.mean(
        np.abs(
            np.sort(generated_array[indices], axis=1)
            - np.sort(reference_array[indices], axis=1)
        ),
        axis=1,
    )
    return {
        "n": int(generated_array.size),
        "mean": point,
        "sample_std": float(np.std(bootstrap, ddof=1)),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "uncertainty": "paired_bootstrap_standard_deviation",
    }


def relative_distribution_w1_with_paired_bootstrap(
    generated: list[float],
    reference: list[float],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20_260_729,
) -> dict[str, Any]:
    """Return W1(component counts)/mean(reference) and paired uncertainty."""
    generated_array = np.asarray(generated, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if generated_array.shape != reference_array.shape or generated_array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "sample_std": None,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "uncertainty": "paired_bootstrap_standard_deviation",
        }

    def relative_w1(left: np.ndarray, right: np.ndarray) -> float:
        raw_w1 = float(np.mean(np.abs(np.sort(left) - np.sort(right))))
        return raw_w1 / max(float(np.mean(right)), 1.0)

    point = relative_w1(generated_array, reference_array)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        generated_array.size,
        size=(bootstrap_samples, generated_array.size),
    )
    bootstrap = np.asarray(
        [
            relative_w1(generated_array[index], reference_array[index])
            for index in indices
        ],
        dtype=np.float64,
    )
    return {
        "n": int(generated_array.size),
        "mean": point,
        "sample_std": float(np.std(bootstrap, ddof=1)),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "uncertainty": "paired_bootstrap_standard_deviation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()

    experiment = json.loads(
        (args.experiment_root / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    attempts = load_attempts()
    per_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incomplete = []
    comparison_protocol_mismatches = []
    row_protocol_mismatches = []
    standalone_report_protocol_mismatches = []
    standalone_report_count = 0
    for design in experiment["designs"]:
        design_dir = Path(design["design_dir"])
        comparison_path = design_dir / "final_table/constrained_scaffold_comparison.json"
        if not comparison_path.is_file():
            incomplete.append(str(design["design_id"]))
            continue
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if (
            comparison.get("post_stitch_accounting_protocol")
            != POST_STITCH_ACCOUNTING_PROTOCOL
        ):
            comparison_protocol_mismatches.append(str(comparison_path))
        comparison_by_method = {row["method"]: row for row in comparison["rows"]}
        selection = json.loads(
            Path(design["selection_manifest"]).read_text(encoding="utf-8")
        )
        for display, comparison_key, selection_key in METHODS:
            row = comparison_by_method[comparison_key]
            if (
                row.get("materializer_repair_accounting_protocol")
                != POST_STITCH_ACCOUNTING_PROTOCOL
                or row["export"].get("materializer_repair_accounting_protocol")
                != POST_STITCH_ACCOUNTING_PROTOCOL
            ):
                row_protocol_mismatches.append(
                    f"{design['design_id']}:{comparison_key}"
                )
            standalone_report = (
                Path(row["export"]["verilog_path"]).parent
                / "assembled_practical_export_report.json"
            )
            if standalone_report.is_file():
                standalone_report_count += 1
                standalone_payload = json.loads(
                    standalone_report.read_text(encoding="utf-8")
                )
                if (
                    standalone_payload.get(
                        "materializer_repair_accounting_protocol"
                    )
                    != POST_STITCH_ACCOUNTING_PROTOCOL
                ):
                    standalone_report_protocol_mismatches.append(
                        str(standalone_report)
                    )
            repair = selected_repair(selection, selection_key, attempts)
            representation = (
                repair["node_role_imputation"]
                + repair["cell_type_imputation"]
                + repair["pin_assignment"]
                + repair["assignment_reconstruction"]
            )
            native_structural = (
                repair["synthetic_nets"]
                + repair["dropped_incidence"]
            )
            issues = row["export"].get("materializer_issue_counter", {})
            boundary = row["boundary"]
            materializer_breakdown = post_stitch_operation_breakdown(
                issues,
                row["export"].get("verilog_metadata", {}),
                explicit_pin_overlay_complete=(selection_key == "elda"),
            )
            materializer_completion = materializer_breakdown["completion_ops"]
            materializer_dropped = materializer_breakdown[
                "dropped_extra_incidence"
            ]
            materializer = materializer_breakdown["total_ops"]
            native_object_repair = representation + native_structural
            composite_semantic_burden = native_object_repair + materializer
            demand_count = int(
                row["native_semantic"]["required_input_pin_slot_count"]
            )
            if selection_key == "elda":
                incidence_retention = 1.0
            else:
                original_incidence = (
                    repair["retained_incidence"] + repair["dropped_incidence"]
                )
                incidence_retention = (
                    repair["retained_incidence"] / original_incidence
                    if original_incidence
                    else None
                )
            backend = load_openroad_metrics(design_dir, selection_key)
            survival = load_survival_metrics(design_dir, selection_key)
            per_method[display].append(
                {
                    "design_id": design["design_id"],
                    "sample_index": int(design["sample_index"]),
                    "source_nodes": int(design["nodes"]),
                    **repair,
                    "representation_completion_ops": representation,
                    "native_partition_structural_repair_ops": native_structural,
                    "native_object_repair_ops": native_object_repair,
                    "assembly_existing_input_net_reuse_ops": (
                        materializer_breakdown["input_reuses"]
                    ),
                    "post_stitch_materializer_completion_ops": (
                        materializer_completion
                    ),
                    "post_stitch_materializer_dropped_incidence": (
                        materializer_dropped
                    ),
                    "post_stitch_materializer_repair_ops": materializer,
                    "post_stitch_accounting_protocol": (
                        POST_STITCH_ACCOUNTING_PROTOCOL
                    ),
                    "post_stitch_materializer_repair_breakdown": (
                        materializer_breakdown
                    ),
                    **{
                        f"materializer_{key}": value
                        for key, value in materializer_breakdown.items()
                        if key not in {"completion_ops", "total_ops"}
                    },
                    "diagnostic_composite_semantic_burden": (
                        composite_semantic_burden
                    ),
                    "required_demand_count": demand_count,
                    "native_repair_ops_per_100_demands": (
                        100.0 * native_object_repair / demand_count
                        if demand_count
                        else None
                    ),
                    "assembly_adaptation_ops_per_100_demands": (
                        100.0 * materializer / demand_count
                        if demand_count
                        else None
                    ),
                    "diagnostic_composite_burden_per_100_demands": (
                        100.0 * composite_semantic_burden / demand_count
                        if demand_count
                        else None
                    ),
                    "incidence_retention": incidence_retention,
                    "boundary_adapter_ops_excluded": int(row["boundary_adapter_ops"]),
                    "yosys_read_check": int(row["yosys_read"] and row["yosys_check"]),
                    "yosys_synth": int(row["yosys_synth"]),
                    "assembled_components": int(row["post_assembly"]["component_count"]),
                    "reference_components": int(
                        comparison["reference"]["component_count"]
                    ),
                    "assembled_lcc_ratio": float(row["post_assembly"]["lcc_ratio"]),
                    "reference_lcc_ratio": float(comparison["reference"]["lcc_ratio"]),
                    "degree_TV": float(row["post_assembly"]["degree_TV"]),
                    "net_incidence_TV": float(
                        row["post_assembly"]["net_incidence_degree_TV"]
                    ),
                    "components_W1": float(
                        row["post_assembly"]["relative_component_distance"]
                    ),
                    "lcc_W1": float(row["post_assembly"]["lcc_W1"]),
                    "lcc_ratio_abs_error": float(row["post_assembly"]["lcc_W1"]),
                    "missing_boundary": int(boundary["missing_boundary_count"]),
                    "extra_boundary": int(boundary["extra_boundary_count"]),
                    "many_to_one_boundary_merge": int(
                        boundary["many_to_one_boundary_merge_count"]
                    ),
                    "one_to_many_boundary_split": int(
                        boundary["one_to_many_boundary_split_count"]
                    ),
                    "interface_dropped_incidence": int(
                        boundary["dropped_incidence_count"]
                    ),
                    **backend,
                    **survival,
                }
            )

    expected_design_count = len(experiment["designs"])
    row_count_mismatches = {
        display: len(per_method[display])
        for display, _, _ in METHODS
        if len(per_method[display]) != expected_design_count
    }
    duplicate_designs = {}
    native_equation_mismatches = []
    composite_equation_mismatches = []
    materializer_sum_mismatches = []
    main_metric_missing = []
    required_main_fields = (
        "representation_completion_ops",
        "native_partition_structural_repair_ops",
        "native_object_repair_ops",
        "post_stitch_materializer_repair_ops",
        "yosys_read_check",
        "routed_gds_flow",
        "assembled_components",
        "reference_components",
        "assembled_lcc_ratio",
        "reference_lcc_ratio",
        "no_keep_exact_instance_retention",
    )
    for display, _, _ in METHODS:
        rows = per_method[display]
        design_counts = Counter(row["design_id"] for row in rows)
        duplicates = sorted(
            design_id
            for design_id, count in design_counts.items()
            if count != 1
        )
        if duplicates:
            duplicate_designs[display] = duplicates
        for row in rows:
            key = f"{display}:{row['design_id']}"
            missing_fields = [
                field
                for field in required_main_fields
                if row.get(field) is None
            ]
            if missing_fields:
                main_metric_missing.append(
                    {"record": key, "fields": missing_fields}
                )
            if row["native_object_repair_ops"] != (
                row["representation_completion_ops"]
                + row["native_partition_structural_repair_ops"]
            ):
                native_equation_mismatches.append(key)
            if row["diagnostic_composite_semantic_burden"] != (
                row["native_object_repair_ops"]
                + row["post_stitch_materializer_repair_ops"]
            ):
                composite_equation_mismatches.append(key)
            breakdown = row["post_stitch_materializer_repair_breakdown"]
            effective_sum = sum(
                int(value)
                for field, value in breakdown.items()
                if field not in {"completion_ops", "total_ops"}
                and not field.startswith("diagnostic_")
            )
            if effective_sum != row["post_stitch_materializer_repair_ops"]:
                materializer_sum_mismatches.append(key)

    out = args.experiment_root / "summary"
    validation = {
        "status": "pass",
        "summary_protocol": "random100_medium_scaffold_summary_v2",
        "post_stitch_accounting_protocol": POST_STITCH_ACCOUNTING_PROTOCOL,
        "statistics": (
            "mean plus sample standard deviation over distinct designs"
        ),
        "interface_adaptation_included_in_repair": False,
        "expected_design_count_per_method": expected_design_count,
        "rows_per_method": {
            display: len(per_method[display])
            for display, _, _ in METHODS
        },
        "comparison_file_count": expected_design_count - len(incomplete),
        "comparison_row_count": sum(len(rows) for rows in per_method.values()),
        "standalone_export_report_count": standalone_report_count,
        "expected_standalone_export_report_count": (
            expected_design_count * len(METHODS)
        ),
        "row_count_mismatches": row_count_mismatches,
        "duplicate_designs": duplicate_designs,
        "main_metric_missing_count": len(main_metric_missing),
        "comparison_protocol_mismatch_count": len(
            comparison_protocol_mismatches
        ),
        "row_protocol_mismatch_count": len(row_protocol_mismatches),
        "standalone_report_protocol_mismatch_count": len(
            standalone_report_protocol_mismatches
        ),
        "native_equation_mismatch_count": len(native_equation_mismatches),
        "diagnostic_composite_equation_mismatch_count": len(
            composite_equation_mismatches
        ),
        "materializer_sum_mismatch_count": len(
            materializer_sum_mismatches
        ),
        "equations": {
            "native": "R_total_native = R_repr + R_struct_native",
            "diagnostic_composite": (
                "composite_semantic_burden = R_total_native + A_asm_total; "
                "not a sequential repair trace"
            ),
            "materializer": (
                "A_asm_total = sum(effective atomic post-stitch mutations)"
            ),
        },
        "sha256": {
            "experiment_manifest": sha256_file(
                args.experiment_root / "experiment_manifest.json"
            ),
            "repair_operation_accounting": sha256_file(
                Path(__file__).with_name("repair_operation_accounting.py")
            ),
            "summarizer": sha256_file(Path(__file__)),
        },
    }
    validation_failures = (
        incomplete
        or row_count_mismatches
        or duplicate_designs
        or main_metric_missing
        or standalone_report_count != expected_design_count * len(METHODS)
        or comparison_protocol_mismatches
        or row_protocol_mismatches
        or standalone_report_protocol_mismatches
        or native_equation_mismatches
        or composite_equation_mismatches
        or materializer_sum_mismatches
    )
    if validation_failures:
        validation["status"] = "fail"
    write_json(out / "accounting_validation.json", validation)
    if validation_failures:
        raise RuntimeError(
            "Repair-accounting validation failed; see "
            f"{out / 'accounting_validation.json'}"
        )

    metric_names = [
        "representation_completion_ops",
        "native_partition_structural_repair_ops",
        "native_object_repair_ops",
        "assembly_existing_input_net_reuse_ops",
        "diagnostic_composite_semantic_burden",
        "required_demand_count",
        "native_repair_ops_per_100_demands",
        "assembly_adaptation_ops_per_100_demands",
        "diagnostic_composite_burden_per_100_demands",
        "incidence_retention",
        "yosys_read_check",
        "routed_gds_flow",
        "assembled_components",
        "assembled_lcc_ratio",
        "reference_lcc_ratio",
        "no_keep_exact_instance_retention",
        "observable_state_rate",
        "cell_type_imputation",
        "node_role_imputation",
        "pin_assignment",
        "assignment_reconstruction",
        "synthetic_nets",
        "dropped_incidence",
        "degree_TV",
        "net_incidence_TV",
        "components_W1",
        "lcc_W1",
        "lcc_ratio_abs_error",
        "boundary_adapter_ops_excluded",
        "post_stitch_materializer_completion_ops",
        "post_stitch_materializer_dropped_incidence",
        "post_stitch_materializer_repair_ops",
        "materializer_cell_spec_fallbacks",
        "materializer_input_pool_net_creations",
        "materializer_input_connection_completions",
        "materializer_helper_input_stages",
        "materializer_input_reuses",
        "materializer_output_net_completions",
        "materializer_output_conflict_reassignments",
        "materializer_multi_driver_splits",
        "materializer_dropped_extra_incidence",
        "materializer_legalizer_self_loop_input_remaps",
        "materializer_legalizer_same_net_reuse_remaps",
        "materializer_legalizer_undriven_load_pi_promotions",
        "materializer_legalizer_input_pool_net_creations",
        "materializer_diagnostic_heuristic_input_reuse_proposals",
        "materializer_diagnostic_heuristic_extra_incidence_proposals",
        "missing_boundary",
        "extra_boundary",
        "many_to_one_boundary_merge",
        "one_to_many_boundary_split",
        "interface_dropped_incidence",
        "backend_core_utilization",
        "backend_elapsed_seconds",
        "backend_gds_bytes",
        "backend_die_area_um2",
        "backend_drc_errors",
        "backend_wirelength",
        "backend_vias",
        "backend_tns",
        "backend_wns",
        "backend_worst_slack",
        "generated_state_cells",
    ]
    aggregates = {}
    for display, _, _ in METHODS:
        rows = per_method[display]
        aggregates[display] = {
            "design_count": len(rows),
            **{
                metric: mean_sd(
                    [row[metric] for row in rows if row.get(metric) is not None]
                )
                for metric in metric_names
            },
        }
        aggregates[display]["assembled_lcc_ratio_distribution_W1"] = (
            distribution_w1_with_paired_bootstrap(
                [row["assembled_lcc_ratio"] for row in rows],
                [row["reference_lcc_ratio"] for row in rows],
            )
        )
        aggregates[display]["assembled_relative_components_distribution_W1"] = (
            relative_distribution_w1_with_paired_bootstrap(
                [row["assembled_components"] for row in rows],
                [row["reference_components"] for row in rows],
            )
        )

    out.mkdir(parents=True, exist_ok=True)
    metadata_audit = scaffold_metadata_restoration_audit(experiment)
    write_json(
        out / "scaffold_metadata_restoration_audit.json",
        metadata_audit,
    )
    metadata_unique = metadata_audit["unique_design_totals"]
    metadata_lines = [
        "# Retained Scaffold Metadata Restoration Audit",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Designs | {metadata_audit['design_count']} |",
        f"| Method/design calls | {metadata_audit['observed_report_count']} |",
        f"| Exact graph-contract matches | {metadata_audit['graph_match_ok_count']} |",
        (
            "| Cross-method metadata mismatches | "
            f"{metadata_audit['cross_method_metadata_mismatch_design_count']} |"
        ),
        f"| Unique source scaffold edges | {metadata_unique['edge_count']} |",
        (
            "| Edges with recovered role | "
            f"{metadata_unique['role_known_edge_count']} |"
        ),
        (
            "| Edges with recovered pin | "
            f"{metadata_unique['pin_known_edge_count']} |"
        ),
        (
            "| Missing metadata edges | "
            f"{metadata_unique['missing_metadata_edge_count']} |"
        ),
        (
            "| Ambiguous collapsed incidences | "
            f"{metadata_unique['ambiguous_collapsed_pin_edge_count']} |"
        ),
        (
            "| Inout collapsed incidences | "
            f"{metadata_unique['inout_collapsed_edge_count']} |"
        ),
        "",
        "This common step annotates existing retained-scaffold edges only. It "
        "adds or removes no nodes or edges, creates no nets, changes no driver "
        "connectivity, and completes no generated-object required pins.",
        "Ambiguous/inout collapses remain explicit diagnostics and are not "
        "silently treated as exact semantic recovery.",
    ]
    (
        out / "scaffold_metadata_restoration_audit.md"
    ).write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    payload = {
        "protocol": "random100_medium_scaffold_summary_v2",
        "post_stitch_accounting_protocol": POST_STITCH_ACCOUNTING_PROTOCOL,
        "requested_design_count": 100,
        "completed_design_count": len(experiment["designs"]) - len(incomplete),
        "incomplete_designs": incomplete,
        "statistics": "mean and sample standard deviation over distinct designs",
        "repair_policy": {
            "representation_completion": (
                "node-role imputation + cell-type imputation + pin assignment + "
                "source/load assignment reconstruction on selected native partitions"
            ),
            "native_structural_repair": (
                "native-partition synthetic nets + native-partition dropped incidence"
            ),
            "native_total_repair": (
                "representation completion + native structural repair"
            ),
            "assembly_adaptation": (
                "effective atomic post-stitch materializer/legalizer mutations; "
                "diagnostic aliases and overridden ELDA heuristic proposals excluded"
            ),
            "diagnostic_composite_semantic_burden": (
                "native total repair + assembly adaptation from two independent "
                "experimental paths; retained only as a diagnostic and not "
                "reported as a sequential total"
            ),
            "normalized_burdens": (
                "native total repair and assembly adaptation are separately divided "
                "by the required input-pin demand count of the selected partitions"
            ),
            "incidence_retention": (
                "retained / (retained + dropped) native candidate incidence under "
                "deterministic repair; ELDA is 1 because no native incidence repair "
                "is applied"
            ),
            "materializer_counting": (
                "input-pool connections are counted once; diagnostic pre-repair "
                "deficits are excluded; actual output reassignment, multi-driver "
                "split, cell-spec fallback, dropped-extra-incidence, input "
                "legalization remap, undriven-load promotion, and legalization "
                "input-pool creation operations are included"
            ),
            "explicit_pin_overlay": (
                "For ELDA, pre-overlay heuristic input-reuse and extra-incidence "
                "counters are diagnostics, because the complete native pin_to_net "
                "map is applied afterward. Only final legalizer mutations count. "
                "Generic baselines lack this pin map, so those counters remain "
                "effective materializer operations."
            ),
        },
        "aggregates": aggregates,
        "per_design": dict(per_method),
    }
    write_json(out / "random100_scaffold_metrics.json", payload)

    main_columns = [
        ("Method", None),
        ("R_repr ↓", "representation_completion_ops"),
        ("R_struct^native ↓", "native_partition_structural_repair_ops"),
        ("R_total^native ↓", "native_object_repair_ops"),
        ("A_asm^total ↓", "post_stitch_materializer_repair_ops"),
        ("Yosys ↑", "yosys_read_check"),
        ("Routed-GDS flow ↑", "routed_gds_flow"),
        (
            "Rel. Components W1 ↓",
            "assembled_relative_components_distribution_W1",
        ),
        (
            "Design-level LCC-ratio W1 ↓",
            "assembled_lcc_ratio_distribution_W1",
        ),
        ("Exact-instance retention ↑", "no_keep_exact_instance_retention"),
    ]
    lines = [
        "| " + " | ".join(name for name, _ in main_columns) + " |",
        "|" + "|".join("---" if key is None else "---:" for _, key in main_columns) + "|",
    ]
    for display, _, _ in METHODS:
        stats = aggregates[display]
        lines.append(
            "| "
            + " | ".join(
                display if key is None else format_stat(stats[key])
                for _, key in main_columns
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Values are mean ± sample standard deviation across distinct designs, "
            "except Rel. Components W1 and Design-level LCC-ratio W1, whose "
            "uncertainties are paired-design bootstrap standard deviations.",
            "R_repr is native representation completion. R_struct^native is native "
            "synthetic-net creation plus native dropped incidence. R_total^native is "
            "their sum.",
            "A_asm^total counts atomic post-stitch materializer/legalizer "
            "mutations with alias counters removed. One endpoint may undergo "
            "more than one sequential mutation. Boundary/scaffold interface "
            "adaptation is excluded.",
            "Its semantic fallback, completion, reuse, conflict-resolution, and "
            "dropped-incidence subcategories are reported in "
            "table_materializer_operation_breakdown_mean_sd.md.",
            "All reported main-table metrics have n=100 per method; per-metric n is "
            "recorded in random100_scaffold_metrics.json.",
            "Design-level LCC-ratio W1 uses the same empirical one-dimensional "
            "Wasserstein definition as the subcircuit main table, but compares "
            "the 100 post-assembly LCC ratios with those of the corresponding 100 "
            "original target designs. Its standard deviation uses 10,000 paired "
            "bootstrap resamples with seed 20260729.",
            "Rel. Components W1 is W1 between the 100 post-assembly and matched "
            "original-design component-count distributions, divided by the mean "
            "original-design component count. Its standard deviation uses the "
            "same 10,000 paired bootstrap resamples.",
            f"Accounting protocol: {POST_STITCH_ACCOUNTING_PROTOCOL}; frozen hashes "
            "and invariant checks are recorded in accounting_validation.json.",
        ]
    )
    (out / "table_main_mean_sd.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "table_main_mean_sd.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([name for name, _ in main_columns])
        for display, _, _ in METHODS:
            stats = aggregates[display]
            writer.writerow(
                [
                    display if key is None else format_stat(stats[key])
                    for _, key in main_columns
                ]
            )

    detail_columns = [
        ("Method", None),
        ("Node-role imp. ↓", "node_role_imputation"),
        ("Type imp. ↓", "cell_type_imputation"),
        ("Pin assign. ↓", "pin_assignment"),
        ("Assignment recon. ↓", "assignment_reconstruction"),
        ("Synthetic nets ↓", "synthetic_nets"),
        ("Native dropped inc. ↓", "dropped_incidence"),
        ("Mat. completion ↓", "post_stitch_materializer_completion_ops"),
        ("Mat. dropped inc. ↓", "post_stitch_materializer_dropped_incidence"),
        ("Materializer total ↓", "post_stitch_materializer_repair_ops"),
        ("Degree TV ↓", "degree_TV"),
        ("Net-incidence TV ↓", "net_incidence_TV"),
        ("Components W1 ↓", "components_W1"),
        ("Paired LCC-ratio error ↓", "lcc_ratio_abs_error"),
    ]
    lines = [
        "| " + " | ".join(name for name, _ in detail_columns) + " |",
        "|" + "|".join("---" if key is None else "---:" for _, key in detail_columns) + "|",
    ]
    for display, _, _ in METHODS:
        stats = aggregates[display]
        lines.append(
            "| "
            + " | ".join(
                display if key is None else format_stat(stats[key])
                for _, key in detail_columns
            )
            + " |"
        )
    (out / "table_detailed_mean_sd.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    repair_columns = [
        ("Method", None),
        ("Node-role imp. ↓", "node_role_imputation"),
        ("Type imp. ↓", "cell_type_imputation"),
        ("Pin assign. ↓", "pin_assignment"),
        ("Assignment recon. ↓", "assignment_reconstruction"),
        ("Synthetic nets ↓", "synthetic_nets"),
        ("Native dropped inc. ↓", "dropped_incidence"),
        ("R_repr ↓", "representation_completion_ops"),
        ("R_struct^native ↓", "native_partition_structural_repair_ops"),
        ("R_total^native ↓", "native_object_repair_ops"),
        ("A_asm^total ↓", "post_stitch_materializer_repair_ops"),
        ("R_total^native / 100D ↓", "native_repair_ops_per_100_demands"),
        (
            "A_asm^total / 100D ↓",
            "assembly_adaptation_ops_per_100_demands",
        ),
        ("Inc. retention ↑", "incidence_retention"),
        ("Final Yosys ↑", "yosys_read_check"),
    ]
    lines = [
        "| " + " | ".join(name for name, _ in repair_columns) + " |",
        "|" + "|".join("---" if key is None else "---:" for _, key in repair_columns) + "|",
    ]
    for display, _, _ in METHODS:
        stats = aggregates[display]
        lines.append(
            "| "
            + " | ".join(
                display if key is None else format_stat(stats[key])
                for _, key in repair_columns
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Values are mean ± sample standard deviation over 100 distinct designs.",
            "R_repr is native representation completion. R_struct^native contains "
            "only native synthetic-net creation and native dropped incidence. "
            "R_total^native is their sum. A_asm^total contains only effective "
            "post-stitch materializer/legalizer mutations.",
            "R_total^native and A_asm^total come from independent experimental "
            "paths and are therefore not added into a published total.",
            "D is the required input-pin demand count of the selected generated partitions.",
            "The two /100D columns normalize native repair and assembly adaptation "
            "separately. Boundary/scaffold interface adaptation is excluded.",
            "Incidence retention is measured before scaffold adaptation.",
            "Native net splitting is omitted: the frozen deterministic repair algorithm "
            "never performs that operation. Materializer split operations, if any, are "
            "included in Mat. completion.",
        ]
    )
    (out / "table_repair_fidelity_mean_sd.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (out / "table_final_repair_statistics_mean_sd.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    with (out / "table_repair_fidelity_mean_sd.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([name for name, _ in repair_columns])
        for display, _, _ in METHODS:
            stats = aggregates[display]
            writer.writerow(
                [
                    display if key is None else format_stat(stats[key])
                    for _, key in repair_columns
                ]
            )
    with (out / "table_final_repair_statistics_mean_sd.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([name for name, _ in repair_columns])
        for display, _, _ in METHODS:
            stats = aggregates[display]
            writer.writerow(
                [
                    display if key is None else format_stat(stats[key])
                    for _, key in repair_columns
                ]
            )

    materializer_columns = [
        ("Method", None),
        ("Cell-spec fallback ↓", "materializer_cell_spec_fallbacks"),
        ("Input-pool nets ↓", "materializer_input_pool_net_creations"),
        (
            "Input connection completion ↓",
            "materializer_input_connection_completions",
        ),
        ("Helper stages ↓", "materializer_helper_input_stages"),
        ("Input reuse ↓", "materializer_input_reuses"),
        ("Output-net completion ↓", "materializer_output_net_completions"),
        (
            "Output conflict reassignment ↓",
            "materializer_output_conflict_reassignments",
        ),
        ("Multi-driver split ↓", "materializer_multi_driver_splits"),
        (
            "Self-loop input remap ↓",
            "materializer_legalizer_self_loop_input_remaps",
        ),
        (
            "Same-net reuse remap ↓",
            "materializer_legalizer_same_net_reuse_remaps",
        ),
        (
            "Undriven load to PI ↓",
            "materializer_legalizer_undriven_load_pi_promotions",
        ),
        (
            "Legalizer input-pool nets ↓",
            "materializer_legalizer_input_pool_net_creations",
        ),
        ("Dropped extra incidence ↓", "materializer_dropped_extra_incidence"),
        (
            "Heuristic reuse proposals (diag.)",
            "materializer_diagnostic_heuristic_input_reuse_proposals",
        ),
        (
            "Heuristic extra-incidence proposals (diag.)",
            "materializer_diagnostic_heuristic_extra_incidence_proposals",
        ),
        ("Materializer total ↓", "post_stitch_materializer_repair_ops"),
    ]
    lines = [
        "| " + " | ".join(name for name, _ in materializer_columns) + " |",
        "|"
        + "|".join(
            "---" if key is None else "---:" for _, key in materializer_columns
        )
        + "|",
    ]
    for display, _, _ in METHODS:
        stats = aggregates[display]
        lines.append(
            "| "
            + " | ".join(
                display if key is None else format_stat(stats[key])
                for _, key in materializer_columns
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Values are mean ± sample standard deviation over 100 distinct designs.",
            "These columns are atomic execution-stage operation counts with alias "
            "counters removed. One endpoint can appear in more than one column when "
            "it undergoes sequential mutations. Diagnostic deficits and post-repair "
            "quality warnings are not counted as repair operations.",
            "For ELDA, heuristic reuse/extra-incidence proposals are shown only as "
            "diagnostics because its complete explicit pin map overrides them before "
            "Verilog rendering. Final legalization remaps remain counted.",
        ]
    )
    (out / "table_materializer_operation_breakdown_mean_sd.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    backend_columns = [
        ("Method", None),
        ("Routed GDS ↑", "routed_gds_flow"),
        ("Chosen core util. (%) ↑", "backend_core_utilization"),
        ("Die area (um^2) ↓", "backend_die_area_um2"),
        ("DRC errors ↓", "backend_drc_errors"),
        ("Wirelength ↓", "backend_wirelength"),
        ("Vias ↓", "backend_vias"),
        ("WNS ↑", "backend_wns"),
        ("TNS ↑", "backend_tns"),
        ("Worst slack ↑", "backend_worst_slack"),
        ("Backend time (s) ↓", "backend_elapsed_seconds"),
    ]
    lines = [
        "| " + " | ".join(name for name, _ in backend_columns) + " |",
        "|" + "|".join("---" if key is None else "---:" for _, key in backend_columns) + "|",
    ]
    for display, _, _ in METHODS:
        stats = aggregates[display]
        lines.append(
            "| "
            + " | ".join(
                display if key is None else format_stat(stats[key])
                for _, key in backend_columns
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Values are mean ± sample standard deviation over completed backend jobs.",
            "The selected attempt is the highest successful utilization in the fixed "
            "50%, 40%, 30%, 20% ladder; no oversized fixed DIE_AREA is used.",
        ]
    )
    (out / "table_backend_mean_sd.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    utilization_values = (50, 40, 30, 20)
    lines = [
        "| Method | 50% | 40% | 30% | 20% | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for display, _, _ in METHODS:
        counts = Counter(
            int(row["backend_core_utilization"])
            for row in per_method[display]
            if row.get("backend_core_utilization") is not None
        )
        lines.append(
            f"| {display} | "
            + " | ".join(str(counts[value]) for value in utilization_values)
            + f" | {sum(counts.values())} |"
        )
    lines.extend(
        [
            "",
            "Each row records the highest successful utilization selected from the "
            "fixed 50%, 40%, 30%, 20% ladder.",
        ]
    )
    (out / "table_backend_utilization_distribution.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with (out / "per_design_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["method"] + list(next(iter(per_method.values()))[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for display, _, _ in METHODS:
            for row in per_method[display]:
                writer.writerow({"method": display, **row})
    print(out / "table_main_mean_sd.md")


if __name__ == "__main__":
    main()
