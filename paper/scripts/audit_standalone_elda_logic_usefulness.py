#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter, deque
from pathlib import Path
from statistics import mean, median
from typing import Any

from elda_paths import PAPER_ROOT

ROOT = PAPER_ROOT
ATTEMPTS = ROOT / "results/elda/reference/attempts"
OUT = ROOT / "reports/final_baseline/phase10_9_standalone_logic_usefulness"
SEQUENTIAL_PREFIXES = ("DFF", "SDFF", "LATCH", "DLH", "DLL")


def demand_ids(record: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for chunk in record.get("chunks", []) or []:
        ids.extend(str(x) for x in chunk.get("demand_ids", []) or [])
    ids.extend(str(x) for x in record.get("demand_ids", []) or [])
    return ids


def source_is_constant(source: dict[str, Any]) -> bool:
    text = " ".join(str(source.get(k, "")) for k in ("source_kind", "source_role", "endpoint_type"))
    return "CONST" in text.upper()


def source_is_cell_output(source: dict[str, Any]) -> bool:
    text = " ".join(str(source.get(k, "")) for k in ("source_kind", "source_role", "endpoint_type"))
    return "CELL_OUTPUT" in text.upper() or "OUTPUT_PIN" in text.upper()


def is_sequential(cell_type: str) -> bool:
    upper = cell_type.upper()
    return upper.startswith(SEQUENTIAL_PREFIXES)


def parse_last_yosys_cell_count(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    counts = [int(m.group(1)) for m in re.finditer(r"Number of cells:\s+(\d+)", log_path.read_text(errors="ignore"))]
    return counts[-1] if counts else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * pct / 100.0))
    return values[idx]


def audit_payload(payload: dict[str, Any], yosys_log: Path | None = None) -> dict[str, Any]:
    cells = {str(c["cell_id"]): str(c["cell_type"]) for c in payload.get("cells", [])}
    demands = {str(d["demand_id"]): d for d in payload.get("demands", [])}
    sources = {str(s["source_id"]): s for s in payload.get("sources", [])}
    source_loads: dict[str, list[str]] = {}
    for record in payload.get("source_nets", []) or []:
        source_id = str(record.get("source_id"))
        source_loads.setdefault(source_id, []).extend(demand_ids(record))

    predecessors: dict[str, set[str]] = {cell_id: set() for cell_id in cells}
    successors: dict[str, set[str]] = {cell_id: set() for cell_id in cells}
    input_driver_sources: dict[str, list[str]] = {cell_id: [] for cell_id in cells}
    for source_id, source in sources.items():
        driver = source.get("source_cell_id")
        for demand_id in source_loads.get(source_id, []):
            demand = demands.get(str(demand_id))
            if not demand:
                continue
            load = str(demand.get("load_cell_id"))
            if load not in cells:
                continue
            input_driver_sources[load].append(source_id)
            if driver in cells:
                predecessors[load].add(str(driver))
                successors[str(driver)].add(load)

    boundary_output_cells = {
        str(source.get("source_cell_id"))
        for source_id, source in sources.items()
        if source_is_cell_output(source)
        and source.get("source_cell_id") in cells
        and not source_loads.get(source_id)
    }
    sequential_sink_cells = {
        cell_id
        for cell_id, cell_type in cells.items()
        if is_sequential(cell_type) and input_driver_sources.get(cell_id)
    }

    def backward_cone(seeds: set[str]) -> set[str]:
        seen = set(seeds)
        queue = deque(seeds)
        while queue:
            cell = queue.popleft()
            for pred in predecessors.get(cell, set()):
                if pred not in seen:
                    seen.add(pred)
                    queue.append(pred)
        return seen

    obs_boundary = backward_cone(boundary_output_cells)
    obs_boundary_or_seq = backward_cone(boundary_output_cells | sequential_sink_cells)

    # Conservative transitive constant-only audit. Sequential outputs are not treated as
    # constant-producing even if their data input is constant, because state semantics are absent.
    constant_output: dict[str, bool] = {cell_id: False for cell_id in cells}
    changed = True
    while changed:
        changed = False
        for cell_id, cell_type in cells.items():
            if is_sequential(cell_type) or constant_output[cell_id]:
                continue
            driver_ids = input_driver_sources.get(cell_id, [])
            if not driver_ids:
                continue
            all_const = True
            for source_id in driver_ids:
                source = sources.get(source_id, {})
                driver = source.get("source_cell_id")
                if source_is_constant(source):
                    continue
                if driver in cells and constant_output.get(str(driver), False):
                    continue
                all_const = False
                break
            if all_const:
                constant_output[cell_id] = True
                changed = True

    constant_cells = {cell for cell, flag in constant_output.items() if flag}
    constant_output_partition = bool(boundary_output_cells) and boundary_output_cells <= constant_cells

    # Combinational depth with sequential cells treated as cut points.
    comb_cells = {cell for cell, cell_type in cells.items() if not is_sequential(cell_type)}
    indeg = {cell: 0 for cell in comb_cells}
    comb_succ = {cell: set() for cell in comb_cells}
    for src in comb_cells:
        for dst in successors.get(src, set()):
            if dst in comb_cells:
                comb_succ[src].add(dst)
                indeg[dst] += 1
    queue = deque([cell for cell, deg in indeg.items() if deg == 0])
    depth = {cell: 0 for cell in comb_cells}
    visited = 0
    while queue:
        cell = queue.popleft()
        visited += 1
        for dst in comb_succ.get(cell, set()):
            depth[dst] = max(depth[dst], depth[cell] + 1)
            indeg[dst] -= 1
            if indeg[dst] == 0:
                queue.append(dst)
    cycle_in_comb = visited != len(comb_cells)
    max_depth = max(depth.values(), default=0)

    yosys_cells = parse_last_yosys_cell_count(yosys_log) if yosys_log else None
    cell_count = len(cells)
    return {
        "cell_count": cell_count,
        "boundary_output_cell_count": len(boundary_output_cells),
        "sequential_sink_cell_count": len(sequential_sink_cells),
        "observable_boundary_cell_count": len(obs_boundary),
        "observable_boundary_or_seq_cell_count": len(obs_boundary_or_seq),
        "observable_boundary_ratio": len(obs_boundary) / max(cell_count, 1),
        "observable_boundary_or_seq_ratio": len(obs_boundary_or_seq) / max(cell_count, 1),
        "constant_only_cell_count": len(constant_cells),
        "constant_only_cell_ratio": len(constant_cells) / max(cell_count, 1),
        "constant_output_partition": constant_output_partition,
        "max_comb_depth": max_depth,
        "mean_comb_cell_depth": sum(depth.values()) / max(len(depth), 1),
        "comb_depth_ge_1_partition": max_depth >= 1,
        "comb_depth_ge_2_partition": max_depth >= 2,
        "comb_depth_ge_3_partition": max_depth >= 3,
        "cycle_in_comb": cycle_in_comb,
        "yosys_blackbox_cell_count": yosys_cells,
        "yosys_blackbox_retained_ratio": (
            yosys_cells / cell_count if yosys_cells is not None and cell_count else None
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for attempt_dir in sorted(ATTEMPTS.glob("attempt_*")):
        payload_path = attempt_dir / "payload.json"
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text())
        row = audit_payload(payload, attempt_dir / "yosys_flow.log")
        row["attempt"] = attempt_dir.name
        rows.append(row)

    csv_path = OUT / "standalone_elda_logic_usefulness_attempts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()) if rows else ["attempt"])
        writer.writeheader()
        writer.writerows(rows)

    total_cells = sum(int(r["cell_count"]) for r in rows)
    summary = {
        "attempt_root": str(ATTEMPTS),
        "attempts": len(rows),
        "total_cells": total_cells,
        "observable_definition": (
            "Boundary-observable cells are generated cells in the backward cone of "
            "standalone exported module outputs, approximated by CELL_OUTPUT sources "
            "with no internal source-to-demand loads. The boundary-or-sequential variant "
            "also treats generated sequential cells with input demands as observable sinks."
        ),
        "synth_retention_caveat": (
            "The retained ratio is parsed from the existing Yosys verification/synth logs. "
            "The candidate Verilog declares standard cells as black boxes, so this is a "
            "blackbox reachability/cleanup check rather than a semantic constant-propagation proof."
        ),
        "observable_boundary_ratio_micro": sum(int(r["observable_boundary_cell_count"]) for r in rows)
        / max(total_cells, 1),
        "observable_boundary_ratio_mean": mean(float(r["observable_boundary_ratio"]) for r in rows),
        "observable_boundary_ratio_median": median(float(r["observable_boundary_ratio"]) for r in rows),
        "observable_boundary_or_seq_ratio_micro": sum(
            int(r["observable_boundary_or_seq_cell_count"]) for r in rows
        )
        / max(total_cells, 1),
        "observable_boundary_or_seq_ratio_mean": mean(
            float(r["observable_boundary_or_seq_ratio"]) for r in rows
        ),
        "constant_only_cell_ratio_micro": sum(int(r["constant_only_cell_count"]) for r in rows)
        / max(total_cells, 1),
        "constant_only_cell_ratio_mean": mean(float(r["constant_only_cell_ratio"]) for r in rows),
        "constant_output_partition_rate": sum(bool(r["constant_output_partition"]) for r in rows)
        / max(len(rows), 1),
        "nonzero_comb_depth_partition_rate": sum(bool(r["comb_depth_ge_1_partition"]) for r in rows)
        / max(len(rows), 1),
        "comb_depth_ge_2_partition_rate": sum(bool(r["comb_depth_ge_2_partition"]) for r in rows)
        / max(len(rows), 1),
        "comb_depth_ge_3_partition_rate": sum(bool(r["comb_depth_ge_3_partition"]) for r in rows)
        / max(len(rows), 1),
        "max_comb_depth_mean": mean(float(r["max_comb_depth"]) for r in rows),
        "max_comb_depth_median": median(float(r["max_comb_depth"]) for r in rows),
        "max_comb_depth_p95": percentile([float(r["max_comb_depth"]) for r in rows], 95),
        "mean_comb_cell_depth": mean(float(r["mean_comb_cell_depth"]) for r in rows),
        "comb_cycle_attempt_rate": sum(bool(r["cycle_in_comb"]) for r in rows) / max(len(rows), 1),
    }
    retained = [float(r["yosys_blackbox_retained_ratio"]) for r in rows if r["yosys_blackbox_retained_ratio"] is not None]
    if retained:
        summary["yosys_blackbox_retained_ratio_micro"] = sum(
            int(r["yosys_blackbox_cell_count"] or 0) for r in rows
        ) / max(total_cells, 1)
        summary["yosys_blackbox_retained_ratio_mean"] = mean(retained)
        summary["yosys_blackbox_retained_ratio_median"] = median(retained)
        summary["yosys_blackbox_retained_attempts"] = len(retained)

    summary_path = OUT / "standalone_elda_logic_usefulness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_path = OUT / "standalone_elda_logic_usefulness_summary.md"
    md = [
        "# Standalone ELDA logic-usefulness audit",
        "",
        f"Attempt root: `{ATTEMPTS}`",
        "",
        "All ratios are computed over standalone freely generated ELDA candidates from the final selected checkpoint topology-safe n=1024 run.",
        "",
        "| Metric | Value | Notes |",
        "|---|---:|---|",
        f"| Observable-cone ratio, boundary outputs ↑ | {summary['observable_boundary_ratio_micro']:.4f} | Micro over generated cells; backward cone of exported module outputs. |",
        f"| Observable-cone ratio, boundary outputs or sequential sinks ↑ | {summary['observable_boundary_or_seq_ratio_micro']:.4f} | Adds generated sequential cells with input demands as state-observable sinks. |",
        f"| Yosys blackbox retained cell ratio ↑ | {summary.get('yosys_blackbox_retained_ratio_micro', float('nan')):.4f} | Existing verification/synth logs; blackbox standard-cell caveat applies. |",
        f"| Constant-only cell ratio ↓ | {summary['constant_only_cell_ratio_micro']:.4f} | Conservative transitive constant-source audit. |",
        f"| Constant-output partition rate ↓ | {summary['constant_output_partition_rate']:.4f} | All exported boundary-output cells are constant-only. |",
        f"| Nonzero comb-depth partition ratio ↑ | {summary['nonzero_comb_depth_partition_rate']:.4f} | At least one internal comb→comb edge. |",
        f"| Comb-depth ≥2 partition ratio ↑ | {summary['comb_depth_ge_2_partition_rate']:.4f} | Stronger non-triviality check. |",
        f"| Comb-depth ≥3 partition ratio ↑ | {summary['comb_depth_ge_3_partition_rate']:.4f} | Stronger non-triviality check. |",
        f"| Max comb depth, median | {summary['max_comb_depth_median']:.1f} | Sequential cells treated as cut points. |",
        f"| Max comb depth, p95 | {summary['max_comb_depth_p95']:.1f} | Sequential cells treated as cut points. |",
        "",
        "Caveat: the retained-cell metric is not a semantic optimization proof because candidate Verilog declares Liberty cells as black boxes. A true semantic retained-cell audit requires whitebox Liberty-function models or an equivalent mapped-cell semantics pass.",
        "",
        f"Machine-readable summary: `{summary_path}`",
        f"Per-attempt CSV: `{csv_path}`",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(summary_path)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
