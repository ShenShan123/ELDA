#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from collections import deque
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from elda_paths import PAPER_ROOT

ROOT = PAPER_ROOT
OUT = ROOT / "reports/final_baseline/phase10_9_standalone_logic_usefulness"
REPAIR_ROOT = ROOT / "reports/final_baseline/deterministic_repair_burden_20260726_multiout/models"
ELDA_ATTEMPTS = ROOT / "results/elda/reference/attempts"
SEQUENTIAL_PREFIXES = ("DFF", "SDFF", "LATCH", "DLH", "DLL")

sys.path.insert(0, str(ROOT / "scripts"))
from audit_standalone_elda_logic_usefulness import audit_payload  # noqa: E402
from compute_unified_dedup11390_reference_metrics import (  # noqa: E402
    dedup_reference_paths,
    worker_tokenizer,
)


def is_sequential(cell_type: str) -> bool:
    return cell_type.upper().startswith(SEQUENTIAL_PREFIXES)


def parse_last_yosys_cell_count(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    counts = [int(m.group(1)) for m in re.finditer(r"Number of cells:\s+(\d+)", log_path.read_text(errors="ignore"))]
    return counts[-1] if counts else None


def split_net_expr(expr: str) -> list[str]:
    expr = expr.strip()
    if not expr or re.fullmatch(r"1'[bB][01xzXZ]", expr) or expr in {"1'b0", "1'b1", "1'h0", "1'h1"}:
        return [expr] if expr else []
    m = re.fullmatch(r"\{(.+)\}", expr, flags=re.S)
    if not m:
        return [expr]
    parts: list[str] = []
    depth = 0
    start = 0
    inner = m.group(1)
    for idx, ch in enumerate(inner):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.extend(split_net_expr(inner[start:idx]))
            start = idx + 1
    parts.extend(split_net_expr(inner[start:]))
    return parts


def parse_verilog_graph(verilog: Path, yosys_log: Path | None = None) -> dict[str, Any]:
    text = verilog.read_text(errors="ignore")
    text = re.sub(r"//.*", "", text)
    modules = list(re.finditer(r"module\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\((.*?)\);\s*(.*?)endmodule", text, flags=re.S))
    if not modules:
        raise ValueError(f"No modules found in {verilog}")

    module_dirs: dict[str, dict[str, str]] = {}
    for mod in modules:
        name, _ports, body = mod.group(1), mod.group(2), mod.group(3)
        dirs: dict[str, str] = {}
        for direction in ("input", "output", "inout"):
            for decl in re.finditer(rf"\b{direction}\b\s+([^;]+);", body):
                cleaned = re.sub(r"\[[^\]]+\]", " ", decl.group(1))
                cleaned = cleaned.replace("wire", " ").replace("reg", " ").replace("logic", " ")
                for token in re.split(r"[,\\s]+", cleaned):
                    token = token.strip()
                    if token:
                        dirs[token] = direction
        module_dirs[name] = dirs

    top = modules[-1]
    top_name, _top_ports, body = top.group(1), top.group(2), top.group(3)
    top_dirs = module_dirs.get(top_name, {})
    output_nets = {name for name, direction in top_dirs.items() if direction == "output"}

    cells: dict[str, str] = {}
    predecessors: dict[str, set[str]] = {}
    successors: dict[str, set[str]] = {}
    input_driver_nets: dict[str, list[str]] = {}
    net_drivers: dict[str, set[str]] = {}

    instance_re = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\((.*?)\);\s*$",
        flags=re.M | re.S,
    )
    instances = []
    for match in instance_re.finditer(body):
        cell_type, inst, conn_blob = match.group(1), match.group(2), match.group(3)
        if cell_type not in module_dirs or cell_type == top_name:
            continue
        conns = re.findall(r"\.([A-Za-z_][A-Za-z0-9_$]*)\s*\(([^()]*)\)", conn_blob)
        instances.append((cell_type, inst, conns))
        cells[inst] = cell_type
        predecessors.setdefault(inst, set())
        successors.setdefault(inst, set())
        input_driver_nets.setdefault(inst, [])
        dirs = module_dirs[cell_type]
        for pin, expr in conns:
            for net in split_net_expr(expr):
                if dirs.get(pin) == "output":
                    net_drivers.setdefault(net, set()).add(inst)

    for cell_type, inst, conns in instances:
        dirs = module_dirs[cell_type]
        for pin, expr in conns:
            if dirs.get(pin) != "input":
                continue
            for net in split_net_expr(expr):
                input_driver_nets[inst].append(net)
                for driver in net_drivers.get(net, set()):
                    if driver != inst:
                        predecessors[inst].add(driver)
                        successors.setdefault(driver, set()).add(inst)

    boundary_output_cells: set[str] = set()
    for net in output_nets:
        boundary_output_cells.update(net_drivers.get(net, set()))
    sequential_sink_cells = {
        inst for inst, cell_type in cells.items() if is_sequential(cell_type) and input_driver_nets.get(inst)
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

    constant_output = {inst: False for inst in cells}
    changed = True
    while changed:
        changed = False
        for inst, cell_type in cells.items():
            if is_sequential(cell_type) or constant_output[inst]:
                continue
            nets = input_driver_nets.get(inst, [])
            if not nets:
                continue
            all_const = True
            for net in nets:
                if re.fullmatch(r"1'[bB][01]", net):
                    continue
                drivers = net_drivers.get(net, set())
                if drivers and all(constant_output.get(driver, False) for driver in drivers):
                    continue
                all_const = False
                break
            if all_const:
                constant_output[inst] = True
                changed = True
    constant_cells = {inst for inst, flag in constant_output.items() if flag}

    comb_cells = {inst for inst, typ in cells.items() if not is_sequential(typ)}
    indeg = {inst: 0 for inst in comb_cells}
    comb_succ = {inst: set() for inst in comb_cells}
    for src in comb_cells:
        for dst in successors.get(src, set()):
            if dst in comb_cells:
                comb_succ[src].add(dst)
                indeg[dst] += 1
    queue = deque([inst for inst, deg in indeg.items() if deg == 0])
    depth = {inst: 0 for inst in comb_cells}
    visited = 0
    while queue:
        inst = queue.popleft()
        visited += 1
        for dst in comb_succ.get(inst, set()):
            depth[dst] = max(depth[dst], depth[inst] + 1)
            indeg[dst] -= 1
            if indeg[dst] == 0:
                queue.append(dst)

    cell_count = len(cells)
    max_depth = max(depth.values(), default=0)
    yosys_cells = parse_last_yosys_cell_count(yosys_log) if yosys_log else None
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
        "constant_output_partition": bool(boundary_output_cells) and boundary_output_cells <= constant_cells,
        "max_comb_depth": max_depth,
        "mean_comb_cell_depth": sum(depth.values()) / max(len(depth), 1),
        "comb_depth_ge_1_partition": max_depth >= 1,
        "comb_depth_ge_2_partition": max_depth >= 2,
        "comb_depth_ge_3_partition": max_depth >= 3,
        "cycle_in_comb": visited != len(comb_cells),
        "yosys_blackbox_cell_count": yosys_cells,
        "yosys_blackbox_retained_ratio": (
            yosys_cells / cell_count if yosys_cells is not None and cell_count else None
        ),
    }


def summarize_rows(name: str, rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    total_cells = sum(int(r["cell_count"]) for r in rows)
    retained = [float(r["yosys_blackbox_retained_ratio"]) for r in rows if r.get("yosys_blackbox_retained_ratio") is not None]
    result = {
        "method": name,
        "source": source,
        "attempts_or_partitions": len(rows),
        "total_cells": total_cells,
        "observable_boundary_ratio_micro": sum(int(r["observable_boundary_cell_count"]) for r in rows) / max(total_cells, 1),
        "observable_boundary_or_seq_ratio_micro": sum(int(r["observable_boundary_or_seq_cell_count"]) for r in rows) / max(total_cells, 1),
        "constant_only_cell_ratio_micro": sum(int(r["constant_only_cell_count"]) for r in rows) / max(total_cells, 1),
        "constant_output_partition_rate": sum(bool(r["constant_output_partition"]) for r in rows) / max(len(rows), 1),
        "nonzero_comb_depth_partition_rate": sum(bool(r["comb_depth_ge_1_partition"]) for r in rows) / max(len(rows), 1),
        "comb_depth_ge_2_partition_rate": sum(bool(r["comb_depth_ge_2_partition"]) for r in rows) / max(len(rows), 1),
        "comb_depth_ge_3_partition_rate": sum(bool(r["comb_depth_ge_3_partition"]) for r in rows) / max(len(rows), 1),
        "max_comb_depth_median": median(float(r["max_comb_depth"]) for r in rows),
        "max_comb_depth_mean": mean(float(r["max_comb_depth"]) for r in rows),
        "comb_cycle_attempt_rate": sum(bool(r["cycle_in_comb"]) for r in rows) / max(len(rows), 1),
    }
    if retained:
        result["yosys_blackbox_retained_ratio_micro"] = sum(int(r["yosys_blackbox_cell_count"] or 0) for r in rows) / max(total_cells, 1)
        result["yosys_blackbox_retained_ratio_mean"] = mean(retained)
    else:
        result["yosys_blackbox_retained_ratio_micro"] = None
        result["yosys_blackbox_retained_ratio_mean"] = None
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    method_rows: dict[str, list[dict[str, Any]]] = {}

    test_rows: list[dict[str, Any]] = []
    tokenizer = worker_tokenizer()
    for path_text in dedup_reference_paths():
        graph = torch.load(path_text, map_location="cpu", weights_only=False)
        payload = tokenizer._serialize_payload(graph)
        test_rows.append(audit_payload(payload))
    method_rows["Test reference"] = test_rows

    elda_rows = []
    for attempt in sorted(ELDA_ATTEMPTS.glob("attempt_*")):
        payload = attempt / "payload.json"
        if payload.exists():
            elda_rows.append(audit_payload(json.loads(payload.read_text()), attempt / "yosys_flow.log"))
    method_rows["ELDA"] = elda_rows

    model_names = {
        "AutoGraph repaired": "autograph",
        "G2PT repaired": "g2pt",
        "DiGress repaired": "digress",
    }
    for label, folder in model_names.items():
        rows = []
        for attempt in sorted((REPAIR_ROOT / folder / "attempts").glob("attempt_*")):
            verilog = attempt / "candidate.repaired.v"
            if verilog.exists():
                rows.append(parse_verilog_graph(verilog, attempt / "yosys_synth.log"))
        method_rows[label] = rows

    summaries = [
        summarize_rows("Test reference", method_rows["Test reference"], "source-clean oracle payload, train/validation-dedup test n=11390"),
        summarize_rows("AutoGraph repaired", method_rows["AutoGraph repaired"], "deterministic repaired standalone Verilog n=1024"),
        summarize_rows("G2PT repaired", method_rows["G2PT repaired"], "deterministic repaired standalone Verilog n=1024"),
        summarize_rows("DiGress repaired", method_rows["DiGress repaired"], "deterministic repaired standalone Verilog n=1024"),
        summarize_rows("ELDA", method_rows["ELDA"], "native standalone ELDA payload/verilog n=1024"),
    ]

    summary_json = OUT / "logic_usefulness_test_baseline_comparison.json"
    summary_json.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = OUT / "logic_usefulness_test_baseline_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    md_path = OUT / "logic_usefulness_test_baseline_comparison.md"
    headers = [
        "Method",
        r"(N_{\mathrm{eval}})",
        "Cell-level PO cone ↑",
        "Cell-level PO+seq cone ↑",
        "Cell-level retained ↑",
        "Cell-level const-only ↓",
        "Partition-level const-output ↓",
        "Partition-level nonzero depth",
        "Partition-level depth≥2",
        "Partition-level depth≥3",
        "Partition-level median max depth",
    ]
    lines = [
        "# Logic-usefulness comparison against test and repaired baselines",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        retained = item["yosys_blackbox_retained_ratio_micro"]
        method = item["method"]
        is_elda = method == "ELDA"
        values = [
            f"{item['attempts_or_partitions']:,}",
            f"{item['observable_boundary_ratio_micro']:.4f}",
            f"{item['observable_boundary_or_seq_ratio_micro']:.4f}",
            "N/A" if retained is None else f"{retained:.4f}",
            f"{item['constant_only_cell_ratio_micro']:.4f}",
            f"{item['constant_output_partition_rate']:.4f}",
            f"{item['nonzero_comb_depth_partition_rate']:.4f}",
            f"{item['comb_depth_ge_2_partition_rate']:.4f}",
            f"{item['comb_depth_ge_3_partition_rate']:.4f}",
            f"{item['max_comb_depth_median']:.1f}",
        ]
        if is_elda:
            method = f"**{method}**"
            # Bold the ELDA population, observable/retained ratios, and
            # reference-fidelity depth diagnostics selected for publication.
            for index in (0, 1, 2, 3, 6, 7, 8, 9):
                values[index] = f"**{values[index]}**"
        lines.append(
            "| "
            + " | ".join([method, *values])
            + " |"
        )
    lines += [
        "",
        r"Definitions: Cell-level ratios use generated/test cells as the denominator. Partition-level ratios are computed over the \(N_{\mathrm{eval}}\) evaluable standalone partitions reported in the table. `PO cone` is the cell fraction in backward cones of standalone exported primary-output/boundary-output endpoints. `PO+seq cone` additionally treats sequential cells with input demands as observable state sinks. Depth statistics are computed per partition with sequential cells treated as cut points; they are reference-fidelity diagnostics and are not monotonic objectives. Baseline rows are computed after deterministic repair because native generic graph outputs do not contain pin/source/demand endpoint semantics. `Cell-level retained` is parsed from existing blackbox-cell Yosys logs and is a reachability/cleanup check, not a whitebox Liberty-function constant-propagation proof.",
        "",
        "Reproduce:",
        "",
        "```bash",
        "cd /path/to/ELDA",
        "python paper/scripts/compare_logic_usefulness_test_baselines.py",
        "```",
        "",
        f"JSON: `{summary_json}`",
        f"CSV: `{csv_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    publication_csv_path = OUT / "logic_usefulness_test_baseline_comparison_publication.csv"
    with publication_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Method",
                "N_eval",
                "Cell-level PO cone",
                "Cell-level PO+seq cone",
                "Cell-level retained",
                "Cell-level const-only",
                "Partition-level const-output",
                "Partition-level nonzero depth",
                "Partition-level depth>=2",
                "Partition-level depth>=3",
                "Partition-level median max depth",
            ]
        )
        for item in summaries:
            retained = item["yosys_blackbox_retained_ratio_micro"]
            writer.writerow(
                [
                    item["method"],
                    item["attempts_or_partitions"],
                    f"{item['observable_boundary_ratio_micro']:.4f}",
                    f"{item['observable_boundary_or_seq_ratio_micro']:.4f}",
                    "N/A" if retained is None else f"{retained:.4f}",
                    f"{item['constant_only_cell_ratio_micro']:.4f}",
                    f"{item['constant_output_partition_rate']:.4f}",
                    f"{item['nonzero_comb_depth_partition_rate']:.4f}",
                    f"{item['comb_depth_ge_2_partition_rate']:.4f}",
                    f"{item['comb_depth_ge_3_partition_rate']:.4f}",
                    f"{item['max_comb_depth_median']:.1f}",
                ]
            )

    latex_path = OUT / "logic_usefulness_test_baseline_comparison.tex"
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Standalone logic-usefulness diagnostics. Cell-level ratios use generated/test cells as the denominator. Partition-level ratios use the $N_{\mathrm{eval}}$ evaluable standalone partitions. Depth statistics are reference-fidelity diagnostics rather than monotonic objectives.}",
        r"\label{tab:logic-usefulness-appendix}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Method & $N_{\mathrm{eval}}$ & \multicolumn{4}{c}{Cell-level} & \multicolumn{5}{c}{Partition-level} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-11}",
        r"& & PO cone $\uparrow$ & PO+seq cone $\uparrow$ & Retained $\uparrow$ & Const-only $\downarrow$ & Const-output $\downarrow$ & Nonzero depth & Depth$\geq$2 & Depth$\geq$3 & Median max depth \\",
        r"\midrule",
    ]
    for item in summaries:
        retained = item["yosys_blackbox_retained_ratio_micro"]
        latex_lines.append(
            " & ".join(
                [
                    item["method"].replace("_", r"\_"),
                    str(item["attempts_or_partitions"]),
                    f"{item['observable_boundary_ratio_micro']:.4f}",
                    f"{item['observable_boundary_or_seq_ratio_micro']:.4f}",
                    "N/A" if retained is None else f"{retained:.4f}",
                    f"{item['constant_only_cell_ratio_micro']:.4f}",
                    f"{item['constant_output_partition_rate']:.4f}",
                    f"{item['nonzero_comb_depth_partition_rate']:.4f}",
                    f"{item['comb_depth_ge_2_partition_rate']:.4f}",
                    f"{item['comb_depth_ge_3_partition_rate']:.4f}",
                    f"{item['max_comb_depth_median']:.1f}",
                ]
            )
            + r" \\"
        )
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    print(summary_json)
    print(csv_path)
    print(publication_csv_path)
    print(md_path)
    print(latex_path)


if __name__ == "__main__":
    main()
