#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import networkx as nx
import torch

from elda_paths import PAPER_ROOT

ROOT = PAPER_ROOT
OUT = ROOT / "reports/final_baseline/phase10_8_endpoint_fidelity_main_table"
CACHE = OUT / "reference_endpoint_profile_dedup11390_v1.json"
NATIVE_GRAPH_METRICS = OUT / "elda_policy_native_graph_metrics.json"
GENERIC_GRAPH_METRICS = (
    ROOT
    / "reports/final_baseline/phase10_6_main_graph_quality_table"
    / "unified_dedup11390_reference_metrics.json"
)
GENERIC_NOVELTY_REPORT = (
    ROOT
    / "reports/final_baseline/phase10_2_final_common_tables/reports"
    / "table1_canonical_novelty_report.json"
)
ATTEMPT_ROOTS = {
    "Frequency sampler": ROOT / "results/elda/controls/field_frequency/attempts",
    "ELDA_Unconstrained": ROOT / "results/elda/controls/unconstrained_lm/attempts",
    "ELDA_Syntax": ROOT / "results/elda/controls/syntax_only_lm/attempts",
    "ELDA": ROOT / "results/elda/reference/attempts",
}

sys.path.insert(0, str(ROOT / "scripts"))
from compute_unified_dedup11390_reference_metrics import (  # noqa: E402
    dedup_reference_paths,
    worker_tokenizer,
)


FANOUT_BINS = (
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 4, "3-4"),
    (5, 8, "5-8"),
    (9, 16, "9-16"),
    (17, None, "17+"),
)
SEQUENTIAL_PREFIXES = ("DFF", "SDFF", "LATCH", "DLH", "DLL")


GENERIC_METHODS = (
    (
        "AutoGraph-labeled",
        "AutoGraph-labeled-llama-s-e20-best",
    ),
    ("G2PT-labeled", "G2PT-labeled-e25-best"),
    ("DiGress-full", "DiGress-full"),
)


def fanout_bin(value: int) -> str:
    for lower, upper, label in FANOUT_BINS:
        if value >= lower and (upper is None or value <= upper):
            return label
    raise ValueError(value)


def source_kind(source: dict[str, Any]) -> str:
    kind = str(source.get("source_kind") or source.get("endpoint_type") or "UNKNOWN").upper()
    role = str(source.get("source_role") or "").upper()
    if "BOUNDARY" in kind or "BOUNDARY" in role:
        return "boundary_source"
    if "CONST" in kind or "CONST" in role:
        return "constant_source"
    if "CELL_OUTPUT" in kind or "OUTPUT" in role:
        return "cell_output_source"
    return "unknown_source"


def demand_ids(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for chunk in record.get("chunks", []) or []:
        values.extend(str(value) for value in (chunk.get("demand_ids", []) or []))
    if not values:
        values.extend(str(value) for value in (record.get("demand_ids", []) or []))
    return values


def combinational_depths(payload: dict[str, Any]) -> tuple[Counter[str], bool]:
    cell_type = {
        str(cell["cell_id"]): str(cell["cell_type"])
        for cell in payload.get("cells", [])
    }
    combinational = {
        cell_id
        for cell_id, name in cell_type.items()
        if not name.startswith(SEQUENTIAL_PREFIXES)
    }
    demands = {
        str(demand["demand_id"]): demand
        for demand in payload.get("demands", [])
    }
    sources = {
        str(source["source_id"]): source
        for source in payload.get("sources", [])
    }
    graph = nx.DiGraph()
    graph.add_nodes_from(combinational)
    for record in payload.get("source_nets", []):
        source = sources.get(str(record.get("source_id")), {})
        driver = source.get("source_cell_id")
        if driver not in combinational:
            continue
        for demand_id in demand_ids(record):
            load = demands.get(demand_id, {}).get("load_cell_id")
            if load in combinational:
                graph.add_edge(str(driver), str(load))

    if not graph:
        return Counter(), True
    acyclic = nx.is_directed_acyclic_graph(graph)
    condensed = nx.condensation(graph)
    depth: dict[int, int] = {}
    for node in nx.topological_sort(condensed):
        predecessors = list(condensed.predecessors(node))
        depth[node] = 0 if not predecessors else 1 + max(depth[p] for p in predecessors)
    histogram: Counter[str] = Counter()
    for node, data in condensed.nodes(data=True):
        histogram[str(depth[node])] += len(data.get("members", []))
    return histogram, acyclic


def inspect_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cells = payload.get("cells", [])
    demands = payload.get("demands", [])
    sources = payload.get("sources", [])
    source_nets = payload.get("source_nets", [])
    cell_types = {str(cell["cell_id"]): str(cell["cell_type"]) for cell in cells}
    demand_by_id = {str(item["demand_id"]): item for item in demands}
    source_by_id = {str(item["source_id"]): item for item in sources}
    assignments: dict[str, list[str]] = {source_id: [] for source_id in source_by_id}
    for record in source_nets:
        assignments.setdefault(str(record.get("source_id")), []).extend(demand_ids(record))

    input_pin_hist = Counter(str(item.get("load_pin", "UNKNOWN")) for item in demands)
    cell_pin_hist = Counter(
        f"{cell_types.get(str(item.get('load_cell_id')), 'UNKNOWN')}|{item.get('load_pin', 'UNKNOWN')}"
        for item in demands
    )
    source_kind_hist: Counter[str] = Counter()
    source_fanout_hist: Counter[str] = Counter()
    source_load_pin_hist: Counter[str] = Counter()
    assigned_demand_count = 0
    constant_demand_count = 0
    zero_load_source_count = 0
    boundary_source_count = 0
    source_load_cell_group_count = 0
    same_source_multi_pin_group_count = 0
    output_pins_by_cell: dict[str, set[str]] = {}

    for source_id, source in source_by_id.items():
        kind = source_kind(source)
        assigned = assignments.get(source_id, [])
        pins_by_load_cell: dict[str, set[str]] = {}
        source_kind_hist[kind] += 1
        source_fanout_hist[f"{kind}|{fanout_bin(len(assigned))}"] += 1
        zero_load_source_count += int(not assigned)
        boundary_source_count += int(kind == "boundary_source")
        if kind == "cell_output_source" and source.get("source_cell_id") is not None:
            output_pins_by_cell.setdefault(str(source["source_cell_id"]), set()).add(
                str(source.get("source_pin", "UNKNOWN"))
            )
        for demand_id in assigned:
            demand = demand_by_id.get(demand_id)
            if demand is None:
                continue
            assigned_demand_count += 1
            constant_demand_count += int(kind == "constant_source")
            source_load_pin_hist[f"{kind}|{demand.get('load_pin', 'UNKNOWN')}"] += 1
            pins_by_load_cell.setdefault(str(demand.get("load_cell_id")), set()).add(
                str(demand.get("load_pin", "UNKNOWN"))
            )
        source_load_cell_group_count += len(pins_by_load_cell)
        same_source_multi_pin_group_count += sum(
            len(pins) >= 2 for pins in pins_by_load_cell.values()
        )

    pin_specs = worker_tokenizer().pin_specs
    multi_output_slot_count = 0
    multi_output_emitted_slot_count = 0
    multi_output_cell_count = 0
    multi_output_cell_source_covered_count = 0
    for cell_id, cell_name in cell_types.items():
        spec = pin_specs.get(cell_name)
        required_outputs = set(spec.outputs) if spec is not None else set()
        if len(required_outputs) < 2:
            continue
        multi_output_cell_count += 1
        multi_output_slot_count += len(required_outputs)
        emitted = output_pins_by_cell.get(cell_id, set()) & required_outputs
        multi_output_emitted_slot_count += len(emitted)
        multi_output_cell_source_covered_count += int(
            len(emitted) >= int(spec.min_required_outputs)
        )

    depth_hist, cycle_free = combinational_depths(payload)
    return {
        "input_pin_hist": dict(input_pin_hist),
        "source_kind_hist": dict(source_kind_hist),
        "source_fanout_hist": dict(source_fanout_hist),
        "source_load_pin_hist": dict(source_load_pin_hist),
        "cell_pin_hist": dict(cell_pin_hist),
        "depth_hist": dict(depth_hist),
        "source_count": len(sources),
        "boundary_source_count": boundary_source_count,
        "assigned_demand_count": assigned_demand_count,
        "constant_demand_count": constant_demand_count,
        "zero_load_source_count": zero_load_source_count,
        "multi_output_slot_count": multi_output_slot_count,
        "multi_output_emitted_slot_count": multi_output_emitted_slot_count,
        "multi_output_cell_count": multi_output_cell_count,
        "multi_output_cell_source_covered_count": multi_output_cell_source_covered_count,
        "source_load_cell_group_count": source_load_cell_group_count,
        "same_source_multi_pin_group_count": same_source_multi_pin_group_count,
        "cycle_free": cycle_free,
    }


def inspect_reference(path_text: str) -> dict[str, Any]:
    graph = torch.load(path_text, map_location="cpu", weights_only=False)
    payload = worker_tokenizer()._serialize_payload(graph)
    return inspect_payload(payload)


def merge_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    histogram_fields = (
        "input_pin_hist",
        "source_kind_hist",
        "source_fanout_hist",
        "source_load_pin_hist",
        "cell_pin_hist",
        "depth_hist",
    )
    count_fields = (
        "source_count",
        "boundary_source_count",
        "assigned_demand_count",
        "constant_demand_count",
        "zero_load_source_count",
        "multi_output_slot_count",
        "multi_output_emitted_slot_count",
        "multi_output_cell_count",
        "multi_output_cell_source_covered_count",
        "source_load_cell_group_count",
        "same_source_multi_pin_group_count",
    )
    merged: dict[str, Any] = {field: Counter() for field in histogram_fields}
    merged.update({field: 0 for field in count_fields})
    merged["sample_count"] = len(rows)
    merged["cycle_free_count"] = 0
    for row in rows:
        for field in histogram_fields:
            merged[field].update(row[field])
        for field in count_fields:
            merged[field] += int(row[field])
        merged["cycle_free_count"] += int(row["cycle_free"])
    for field in histogram_fields:
        merged[field] = dict(sorted(merged[field].items()))
    return merged


def tv(left: dict[str, int], right: dict[str, int]) -> float | None:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        return None
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
        for key in keys
    )


def histogram_w1(left: dict[str, int], right: dict[str, int]) -> float | None:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        return None
    keys = sorted({int(key) for key in left} | {int(key) for key in right})
    cdf_left = cdf_right = 0.0
    distance = 0.0
    previous = keys[0]
    for key in keys:
        distance += abs(cdf_left - cdf_right) * (key - previous)
        cdf_left += left.get(str(key), 0) / left_total
        cdf_right += right.get(str(key), 0) / right_total
        previous = key
    return distance


def ratio(profile: dict[str, Any], numerator: str, denominator: str) -> float | None:
    total = int(profile[denominator])
    return int(profile[numerator]) / total if total else None


def abs_delta(left: float | None, right: float | None) -> float | None:
    return abs(left - right) if left is not None and right is not None else None


def compare(reference: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    ref_boundary = ratio(reference, "boundary_source_count", "source_count")
    gen_boundary = ratio(generated, "boundary_source_count", "source_count")
    ref_constant = ratio(reference, "constant_demand_count", "assigned_demand_count")
    gen_constant = ratio(generated, "constant_demand_count", "assigned_demand_count")
    ref_zero = ratio(reference, "zero_load_source_count", "source_count")
    gen_zero = ratio(generated, "zero_load_source_count", "source_count")
    ref_multi = ratio(reference, "multi_output_emitted_slot_count", "multi_output_slot_count")
    gen_multi = ratio(generated, "multi_output_emitted_slot_count", "multi_output_slot_count")
    ref_multi_coverage = ratio(
        reference, "multi_output_cell_source_covered_count", "multi_output_cell_count"
    )
    gen_multi_coverage = ratio(
        generated, "multi_output_cell_source_covered_count", "multi_output_cell_count"
    )
    ref_same_source_multi_pin = ratio(
        reference, "same_source_multi_pin_group_count", "source_load_cell_group_count"
    )
    gen_same_source_multi_pin = ratio(
        generated, "same_source_multi_pin_group_count", "source_load_cell_group_count"
    )
    return {
        "endpoint_sample_count": generated["sample_count"],
        "endpoint_coverage": generated["sample_count"] / 1024,
        "input_pin_hist_TV": tv(reference["input_pin_hist"], generated["input_pin_hist"]),
        "source_kind_TV": tv(reference["source_kind_hist"], generated["source_kind_hist"]),
        "boundary_source_ratio": gen_boundary,
        "boundary_source_ratio_reference": ref_boundary,
        "boundary_source_ratio_abs_error": abs_delta(gen_boundary, ref_boundary),
        "constant_driven_demand_ratio": gen_constant,
        "constant_driven_demand_ratio_reference": ref_constant,
        "constant_driven_demand_ratio_abs_error": abs_delta(gen_constant, ref_constant),
        "source_fanout_TV": tv(reference["source_fanout_hist"], generated["source_fanout_hist"]),
        "source_kind_load_pin_joint_TV": tv(
            reference["source_load_pin_hist"], generated["source_load_pin_hist"]
        ),
        "cell_type_input_pin_usage_TV": tv(
            reference["cell_pin_hist"], generated["cell_pin_hist"]
        ),
        "zero_load_source_ratio": gen_zero,
        "zero_load_source_ratio_reference": ref_zero,
        "zero_load_source_ratio_abs_error": abs_delta(gen_zero, ref_zero),
        "multi_output_source_slot_ratio": gen_multi,
        "multi_output_source_slot_ratio_reference": ref_multi,
        "multi_output_source_slot_ratio_abs_error": abs_delta(gen_multi, ref_multi),
        "multi_output_cell_source_coverage": gen_multi_coverage,
        "multi_output_cell_source_coverage_reference": ref_multi_coverage,
        "same_source_multi_pin_ratio": gen_same_source_multi_pin,
        "same_source_multi_pin_ratio_reference": ref_same_source_multi_pin,
        "same_source_multi_pin_ratio_abs_error": abs_delta(
            gen_same_source_multi_pin, ref_same_source_multi_pin
        ),
        "combinational_depth_W1": histogram_w1(reference["depth_hist"], generated["depth_hist"]),
        "combinational_cycle_free_rate": (
            generated["cycle_free_count"] / generated["sample_count"]
            if generated["sample_count"]
            else None
        ),
    }


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_graph_rows() -> list[dict[str, Any]]:
    generic_metrics = json.loads(
        GENERIC_GRAPH_METRICS.read_text(encoding="utf-8")
    )
    novelty_report = json.loads(
        GENERIC_NOVELTY_REPORT.read_text(encoding="utf-8")
    )
    metrics_by_method = {
        row["method"]: row for row in generic_metrics["metrics"]
    }
    novelty_by_method = {
        row["method_name"]: row for row in novelty_report["methods"]
    }
    novelty_names = {
        "AutoGraph-labeled-llama-s-e20-best": "AutoGraph",
        "G2PT-labeled-e25-best": "G2PT-full-e20-best",
        "DiGress-full": "DiGress-full",
    }
    rows: list[dict[str, Any]] = []
    for display, source in GENERIC_METHODS:
        distribution = metrics_by_method[source]
        novelty = novelty_by_method[novelty_names[source]]
        rows.append(
            {
                "group": "Generic graph",
                "method": display,
                "graph_valid": (
                    int(novelty["valid_fingerprint_count"]) / 1024
                ),
                "unique": float(novelty["unique_rate_over_valid"]),
                "novelty": float(novelty["canonical_novelty"]),
                "vun": float(novelty["VUN_rate_over_1024_attempts"]),
                "degree_mmd": float(distribution["degree_MMD"]),
                "rel_components_w1": float(
                    distribution["relative_components_W1"]
                ),
                "lcc_w1": float(distribution["lcc_W1"]),
                "fanout_tv": float(distribution["fanout_TV"]),
            }
        )

    native = json.loads(NATIVE_GRAPH_METRICS.read_text(encoding="utf-8"))
    for method in ATTEMPT_ROOTS:
        record = native["methods"][method]
        distribution = record["distribution"]
        pipeline = record["pipeline"]
        diversity = record["diversity"]
        rows.append(
            {
            "group": "ELDA object",
                "method": method,
                "graph_valid": float(
                    pipeline["native_object_complete_rate"]
                ),
                "unique": float(diversity["unique_rate_over_valid"]),
                "novelty": float(diversity["novelty_rate_over_valid"]),
                "vun": float(diversity["VUN_rate_over_attempts"]),
                "degree_mmd": float(distribution["degree_MMD"]),
                "rel_components_w1": float(
                    distribution["norm_components_W1"]
                ),
                "lcc_w1": float(distribution["lcc_W1"]),
                "fanout_tv": float(distribution["fanout_TV"]),
            }
        )
    return rows


def write_graph_table(graph_rows: list[dict[str, Any]]) -> None:
    fields = [
        "group", "method", "graph_valid", "unique", "novelty", "vun",
        "degree_mmd", "rel_components_w1", "lcc_w1", "fanout_tv",
    ]
    labels = [
        "Group", "Method", "Graph valid ↑", "Unique ↑", "Novelty ↑", "V.U.N. ↑",
        "Degree MMD ↓", "Rel. Components W1 ↓", "LCC W1 ↓", "Fanout TV ↓",
    ]
    with (OUT / "table1_graph_quality.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(graph_rows)
    lines = ["# Main graph-quality comparison", "", "| " + " | ".join(labels) + " |", "|" + "---|" * len(labels)]
    for row in graph_rows:
        lines.append("| " + " | ".join(format_value(row[field]) for field in fields) + " |")
    lines.extend([
        "",
        "Graph valid in this cross-model table requires a canonical graph artifact suitable for "
        "distribution evaluation; for ELDA rows it equals native-object-complete coverage. It "
        "does not imply strict or netlist validity. Unique and Novelty use canonical fingerprints; "
        "V.U.N. is the fraction of all 1024 attempts that are simultaneously valid, unique, and novel.",
        "",
        "All ELDA-object rows use native ELDA decoded graphs/payloads and the 11,390 "
        "development-deduplicated reference population for the four distribution columns. Their Graph valid/V.U.N. "
        "rates require a native-object-complete graph; parser-stage rates are reported separately "
        "in the native object pipeline table. Generic-graph rows retain their unified graph view, "
        "and their Fanout TV is an incidence proxy rather than exact source/load fanout.",
    ])
    (OUT / "table1_graph_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_table(
    graph_rows: list[dict[str, Any]],
    endpoint_metrics: dict[str, dict[str, Any]],
) -> None:
    fields = [
        "method",
        "graph_valid",
        "unique",
        "novelty",
        "vun",
        "degree_mmd",
        "rel_components_w1",
        "lcc_w1",
        "fanout_tv",
        "source_kind_tv",
        "source_pin_tv",
        "depth_w1",
    ]
    labels = [
        "Method",
        "Graph valid ↑",
        "Unique ↑",
        "Novelty ↑",
        "V.U.N. ↑",
        "Deg. MMD ↓",
        "Rel. Com. W1 ↓",
        "LCC W1 ↓",
        "Fanout TV ↓",
        "Src.-kind TV ↓",
        "Src.-pin TV ↓",
        "Depth W1 ↓",
    ]
    rows = []
    for graph_row in graph_rows:
        endpoint = endpoint_metrics.get(graph_row["method"], {})
        rows.append(
            {
                **graph_row,
                "source_kind_tv": endpoint.get("source_kind_TV"),
                "source_pin_tv": endpoint.get(
                    "source_kind_load_pin_joint_TV"
                ),
                "depth_w1": endpoint.get("combinational_depth_W1"),
            }
        )
    with (OUT / "table_main_graph_endpoint_quality.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                field: (
                    ""
                    if row.get(field) is None
                    else (
                        f"{row[field]:.6f}"
                        if isinstance(row.get(field), float)
                        else row.get(field)
                    )
                )
                for field in fields
            }
            for row in rows
        )
    lines = [
        "# Main graph and endpoint-quality comparison",
        "",
        "| " + " | ".join(labels) + " |",
        "|" + "---|" * len(labels),
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                "-"
                if row.get(field) is None
                else format_value(row.get(field))
                for field in fields
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Generic graph baselines have no native source, demand, or pin "
            "identity; endpoint-level columns are therefore not applicable.",
            "",
            "ELDA-object distribution and endpoint metrics use native ELDA "
            "objects against the 11,390-subcircuit development-deduplicated source-clean "
            "test reference. Generic rows use the frozen unified graph view. "
            "All validity and V.U.N. rates use 1,024 attempts.",
            "",
            "Frequency sampler has 944 valid canonical objects and no duplicate "
            "canonical fingerprints, hence Unique = 1 and V.U.N. = 944/1024. "
            "Endpoint/distribution values for ELDA_Unconstrained and ELDA_Syntax "
            "are conditional on only 2 and 159 complete native objects.",
        ]
    )
    (OUT / "table_main_graph_endpoint_quality.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_endpoint_table(metrics: dict[str, dict[str, Any]]) -> None:
    fields = [
        "method", "endpoint_sample_count", "endpoint_coverage", "input_pin_hist_TV",
        "source_kind_TV", "boundary_source_ratio_abs_error",
        "constant_driven_demand_ratio_abs_error", "source_fanout_TV",
        "source_kind_load_pin_joint_TV", "same_source_multi_pin_ratio_abs_error",
        "cell_type_input_pin_usage_TV",
        "zero_load_source_ratio_abs_error", "multi_output_cell_source_coverage",
        "combinational_depth_W1",
    ]
    labels = [
        "Method", "Endpoint n", "Endpoint cov. ↑", "Input-pin TV ↓", "Source-kind TV ↓",
        "Boundary ratio Δ ↓", "Constant-demand ratio Δ ↓", "Source fanout TV ↓",
        "Source-kind × load-pin TV ↓", "Same-source multi-pin ratio Δ ↓",
        "Cell-type × input-pin TV ↓",
        "Zero-load source ratio Δ ↓", "Multi-output cell/source cov. ↑", "Comb. depth W1 ↓",
    ]
    rows = []
    for method in ATTEMPT_ROOTS:
        rows.append({"method": method, **metrics[method]})
    with (OUT / "table2_endpoint_fidelity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Native endpoint-level fidelity", "", "| " + " | ".join(labels) + " |", "|" + "---|" * len(labels)]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(field)) for field in fields) + " |")
    lines.extend([
        "",
        "All distributions use the 11,390 source-clean test subcircuits remaining after exact "
        "training/validation endpoint-sequence removal. Histograms are pooled over canonical ELDA objects. Ratio columns "
        "marked Δ are absolute errors from the reference ratio.",
        "",
        "Endpoint n/cov. is mandatory context: unconstrained LM has only 2 canonical payloads, "
        "so its distribution values are diagnostic and statistically unreliable. Generic graph "
        "baselines are excluded because their native outputs do not contain pin, source, demand, "
        "or boundary identity; deterministic repair metrics remain a separate appendix.",
        "",
        "Combinational depth is the longest-path depth after sequential cells are treated as "
        "cutpoints and cyclic combinational regions are condensed into SCCs. The multi-output "
        "coverage metric requires each Liberty multi-output cell to expose at least its declared "
        "`min_required_outputs`. Optional output-slot ratio and its reference delta remain in JSON.",
        "",
        "Same-source multi-pin groups demands by `(source identity, load cell)` and detects "
        "patterns such as `.A(n1), .B(n1)` that disappear under cell-level edge merging.",
    ])
    (OUT / "table2_endpoint_fidelity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if CACHE.is_file():
        reference = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        paths = dedup_reference_paths()
        with ProcessPoolExecutor(max_workers=16) as pool:
            rows = list(pool.map(inspect_reference, paths, chunksize=16))
        reference = merge_profiles(rows)
        CACHE.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    generated_profiles: dict[str, dict[str, Any]] = {}
    endpoint_metrics: dict[str, dict[str, Any]] = {}
    for method, attempts in ATTEMPT_ROOTS.items():
        payload_paths = sorted(attempts.glob("attempt_*/payload.json"))
        rows = [inspect_payload(json.loads(path.read_text(encoding="utf-8"))) for path in payload_paths]
        generated_profiles[method] = merge_profiles(rows)
        endpoint_metrics[method] = compare(reference, generated_profiles[method])

    artifact = {
        "protocol": {
            "reference": "11390 source-clean test subcircuits after exact training/validation endpoint-sequence removal",
            "attempt_denominator": 1024,
            "endpoint_population": "canonical payload.json objects only; no repair or fallback",
            "histogram_aggregation": "pooled endpoint/source/cell counts",
            "ratio_columns": "absolute error from the development-dedup11390 reference ratio",
            "source_fanout": "TV over P(source_kind, fanout_bin), load endpoints only",
        },
        "reference_profile": reference,
        "generated_profiles": generated_profiles,
        "endpoint_metrics": endpoint_metrics,
        "graph_table": build_graph_rows(),
    }
    (OUT / "endpoint_fidelity_metrics.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    graph_rows = artifact["graph_table"]
    for row in graph_rows:
        endpoint = endpoint_metrics.get(row["method"])
        if endpoint is not None and abs(
            float(row["fanout_tv"])
            - float(endpoint["source_fanout_TV"])
        ) > 1e-12:
            raise RuntimeError(
                f"fanout mismatch for {row['method']}: "
                f"{row['fanout_tv']} vs {endpoint['source_fanout_TV']}"
            )
    validation = {
        "status": "pass",
        "reference_count": int(reference["sample_count"]),
        "attempt_denominator": 1024,
        "payload_counts": {
            method: int(profile["sample_count"])
            for method, profile in generated_profiles.items()
        },
        "native_graph_metrics_source": str(NATIVE_GRAPH_METRICS.relative_to(ROOT)),
        "generic_graph_metrics_source": str(GENERIC_GRAPH_METRICS.relative_to(ROOT)),
        "generic_novelty_source": str(GENERIC_NOVELTY_REPORT.relative_to(ROOT)),
        "fanout_cross_table_mismatch_count": 0,
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                ROOT / "scripts/recompute_policy_native_metrics.py",
                ROOT / "scripts/compute_table1_canonical_novelty.py",
                NATIVE_GRAPH_METRICS,
                GENERIC_GRAPH_METRICS,
                GENERIC_NOVELTY_REPORT,
                CACHE,
            )
        },
    }
    if validation["reference_count"] != 11390:
        raise RuntimeError(
            f"expected 11390 references, got {validation['reference_count']}"
        )
    (OUT / "table_reproducibility_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_graph_table(graph_rows)
    write_endpoint_table(endpoint_metrics)
    write_combined_table(graph_rows, endpoint_metrics)


if __name__ == "__main__":
    main()
