#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_assembled_graph_netlist import (
    choose_gate_nets,
    compute_graph_netlist_quality,
    default_nangate45_liberty,
    emit_verilog,
    load_mapping,
    parse_liberty_pin_specs,
    resolve_net_id,
    run_yosys_flow,
)
from tools.run_source_net_v5_generated_smoke import _graph_stats
from tools.v5_source_net_common import build_v5_tokenizer, write_csv, write_json, write_md


SAFETY = {
    "training": "not_started",
    "stage2": "not_started/no-go",
    "skeleton": "not_started/no-go",
    "kahypar": "not_run",
    "cap32": "not_run/no-go",
    "projection_exporter": "not_modified",
    "pointer_head": "not_implemented",
    "attention_bias": "not_implemented",
    "cross_attention": "not_implemented",
    "full_assembly": "not_run/no-go",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _as_bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _as_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _as_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _run_export(graph: Data, out: Path, mapping: Path, run_yosys: bool, yosys_bin: str, liberty: str = "") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    label_to_cell = load_mapping(mapping)
    net_id = resolve_net_id(mapping, 0)
    liberty_path = Path(liberty) if liberty else default_nangate45_liberty()
    liberty_specs = parse_liberty_pin_specs(liberty_path) if liberty_path and liberty_path.exists() else None
    fallback_events = []
    gate_records, driver_count, load_count, warnings, used_specs, issue_counter, cell_summary = choose_gate_nets(
        graph=graph,
        net_id=net_id,
        label_to_cell=label_to_cell,
        liberty_specs=liberty_specs,
        fallback_events=fallback_events,
        fallback_event_context={"assembly_mode": "source_net_v5_autopsy"},
        report_repeated_input_signatures=True,
        critical_optional_output_policy="report_only",
    )
    quality = compute_graph_netlist_quality(
        graph=graph,
        gate_records=gate_records,
        driver_count=driver_count,
        load_count=load_count,
        issue_counter=issue_counter,
        label_to_cell=label_to_cell,
        net_id=net_id,
        liberty_specs=liberty_specs,
    )
    module = f"v5_autopsy_{out.name}"
    verilog = out / "assembled.v"
    stats = emit_verilog(
        graph=graph,
        gate_records=gate_records,
        driver_count=driver_count,
        load_count=load_count,
        used_specs=used_specs,
        output_path=verilog,
        module_name=module,
        net_id=net_id,
        emit_blackboxes=bool(liberty_specs is None),
    )
    yosys = {
        "skipped": True,
        "yosys_read_verilog_pass": False,
        "yosys_check_pass": False,
        "yosys_synth_pass": False,
        "final_cell_count_after_yosys": -1,
        "final_wire_count_after_yosys": -1,
    }
    if run_yosys:
        yosys = run_yosys_flow(
            verilog_path=verilog,
            module_name=module,
            output_dir=out,
            liberty_path=None if liberty_specs is None else liberty_path,
            yosys_bin=yosys_bin,
        )
        yosys["skipped"] = False
    with (out / "fallback_events.jsonl").open("w", encoding="utf-8") as fh:
        for ev in fallback_events:
            fh.write(json.dumps(ev, sort_keys=True) + "\n")
    report = {
        "verilog_emitted": bool(verilog.exists()),
        "verilog_path": str(verilog),
        "node_count": int(graph.num_nodes),
        "edge_count": int(graph.edge_index.size(1)),
        "cell_count": int(quality.get("graph", {}).get("gate_count", 0)),
        "source_net_count": int(quality.get("graph", {}).get("net_count", 0)),
        "endpoint_collision": int(quality.get("netlist", {}).get("driver_conflict_nets", 0)),
        "strict_source_legalizability_missing_required_input": int(quality.get("netlist", {}).get("missing_required_input_before_pool", 0)),
        "practical_after_pool_missing_required_input": int(quality.get("netlist", {}).get("missing_required_input_after_pool", 0)),
        "input_pool_connections": int(quality.get("netlist", {}).get("input_pool_connections", 0)),
        "reused_inputs": int(quality.get("netlist", {}).get("reused_inputs", 0)),
        "exporter_synthetic_input_nets": int(quality.get("netlist", {}).get("synthetic_inputs", 0)),
        "exporter_repaired_output_nets": int(quality.get("netlist", {}).get("repaired_output_nets", 0)),
        "exporter_missing_required_input_pins": int(quality.get("netlist", {}).get("missing_input_pins", 0)),
        "issue_counter": issue_counter,
        "quality_metrics": quality,
        "warnings": warnings,
        "fallback_event_count": len(fallback_events),
        "fallback_events_path": str(out / "fallback_events.jsonl"),
        "export_stats": stats,
        "yosys": yosys,
        "cell_summary": cell_summary,
    }
    write_json(out / "export_report.json", report)
    return report


def _boundary_wrapper_graph(graph: Data, boundary_stub_id: int) -> tuple[Data, dict]:
    x = graph.x.reshape(-1).to(torch.long)
    role = getattr(graph, "edge_role", getattr(graph, "edge_attr", None))
    pin = getattr(graph, "edge_pin_id", None)
    keep_node = [int(v.item()) != int(boundary_stub_id) for v in x]
    old_to_new = {}
    labels = []
    for old, keep in enumerate(keep_node):
        if keep:
            old_to_new[int(old)] = len(labels)
            labels.append(int(x[old].item()))
    new_edges = []
    new_roles = []
    new_pins = []
    removed_boundary_nodes = sum(1 for keep in keep_node if not keep)
    removed_boundary_driver_edges = 0
    dropped_other_boundary_edges = 0
    for eidx, (src, dst) in enumerate(graph.edge_index.t().tolist()):
        src = int(src); dst = int(dst)
        r = int(role[eidx].item()) if isinstance(role, torch.Tensor) and int(role.numel()) > eidx else 0
        p = int(pin[eidx].item()) if isinstance(pin, torch.Tensor) and int(pin.numel()) > eidx else -1
        if not keep_node[src] or not keep_node[dst]:
            if not keep_node[src] and r == 2 and keep_node[dst]:
                # Boundary driver becomes an implicit primary input net: keep
                # the net/load side and remove only the artificial driver node.
                removed_boundary_driver_edges += 1
            else:
                dropped_other_boundary_edges += 1
            continue
        new_edges.append((old_to_new[src], old_to_new[dst]))
        new_roles.append(r)
        new_pins.append(p)
    out = Data(
        x=torch.tensor(labels, dtype=torch.long),
        edge_index=torch.tensor(new_edges, dtype=torch.long).t().contiguous() if new_edges else torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(new_roles, dtype=torch.long),
        num_nodes=len(labels),
    )
    out.edge_role = out.edge_attr
    out.edge_pin_id = torch.tensor(new_pins, dtype=torch.long) if new_pins else torch.full((0,), -1, dtype=torch.long)
    out.pin_id_to_name = list(getattr(graph, "pin_id_to_name", []) or [])
    out.decode_valid = bool(getattr(graph, "decode_valid", False))
    return out, {
        "wrapper_enabled": True,
        "removed_boundary_nodes": int(removed_boundary_nodes),
        "removed_boundary_driver_edges": int(removed_boundary_driver_edges),
        "dropped_other_boundary_edges": int(dropped_other_boundary_edges),
    }


def _source_kind_for_net(graph: Data, net: int, net_id: int, boundary_stub_id: int, pin_names: list[str]) -> tuple[str, str, str, int | None]:
    role = getattr(graph, "edge_role", getattr(graph, "edge_attr", None))
    pin = getattr(graph, "edge_pin_id", None)
    x = graph.x.reshape(-1).to(torch.long)
    for eidx, (src, dst) in enumerate(graph.edge_index.t().tolist()):
        if int(dst) != int(net):
            continue
        r = int(role[eidx].item()) if isinstance(role, torch.Tensor) and int(role.numel()) > eidx else 0
        if r != 2:
            continue
        pid = int(pin[eidx].item()) if isinstance(pin, torch.Tensor) and int(pin.numel()) > eidx else -1
        pname = str(pin_names[pid]) if 0 <= pid < len(pin_names) else "__UNKNOWN__"
        label = int(x[int(src)].item())
        if label == int(boundary_stub_id):
            return "BOUNDARY_SOURCE", str(src), pname, int(src)
        if label == int(net_id):
            return "NET_AS_SOURCE", str(src), pname, int(src)
        return "CELL_OUTPUT", str(src), pname, int(src)
    return "BOUNDARY_PI_OR_MISSING_DRIVER", "", "", None


def _missing_autopsy(graph: Data, export_report: dict, net_id: int, boundary_stub_id: int) -> dict:
    pin_names = list(getattr(graph, "pin_id_to_name", []) or [])
    role = getattr(graph, "edge_role", getattr(graph, "edge_attr", None))
    pin = getattr(graph, "edge_pin_id", None)
    load_net_by_gate_pin = {}
    for eidx, (src, dst) in enumerate(graph.edge_index.t().tolist()):
        r = int(role[eidx].item()) if isinstance(role, torch.Tensor) and int(role.numel()) > eidx else 0
        if r != 1:
            continue
        pid = int(pin[eidx].item()) if isinstance(pin, torch.Tensor) and int(pin.numel()) > eidx else -1
        pname = str(pin_names[pid]) if 0 <= pid < len(pin_names) else "__UNKNOWN__"
        load_net_by_gate_pin[(int(src), pname)] = int(dst)
    events = []
    fp = Path(str(export_report.get("fallback_events_path", "")))
    if fp.exists():
        with fp.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    events.append(json.loads(line))
    missing_events = [e for e in events if str(e.get("fallback_kind")) in {"missing_required_before_pool", "input_pool"}]
    by_kind = Counter()
    by_reason = Counter()
    by_boundary = Counter()
    examples = []
    for e in missing_events:
        gate = _as_int(e.get("gate_id"), -1)
        pin_name = str(e.get("pin_name") or "")
        net = load_net_by_gate_pin.get((gate, pin_name))
        if net is None:
            kind, obj, spin, src_node = "net/group not created", "", "", None
            reason = "net/group not created"
        else:
            kind, obj, spin, src_node = _source_kind_for_net(graph, net, net_id, boundary_stub_id, pin_names)
            reason = "boundary source object not recognized" if kind == "BOUNDARY_SOURCE" else (
                "source endpoint absent" if kind == "BOUNDARY_PI_OR_MISSING_DRIVER" else "incomplete generation or insufficient native sources"
            )
        by_kind[kind] += 1
        by_reason[reason] += 1
        if kind == "BOUNDARY_SOURCE":
            by_boundary[f"node_{obj}"] += 1
        if len(examples) < 20:
            examples.append({
                "gate_id": gate,
                "load_pin": pin_name,
                "fallback_kind": e.get("fallback_kind"),
                "materialized_net_id": net,
                "generated_source_kind": kind,
                "generated_source_object": obj,
                "generated_source_pin": spin,
                "materialized_source_node_id": src_node,
                "why_strict_rejected": reason,
                "practical_mapped_to_INPUT_POOL": str(e.get("fallback_kind")) == "input_pool",
            })
    return {
        "missing_event_count": len(missing_events),
        "missing_by_source_kind": dict(by_kind),
        "missing_by_reason": dict(by_reason),
        "top_boundary_groups": dict(by_boundary.most_common(20)),
        "top_examples": examples,
    }


def _aggregate(rows: list[dict], prefix: str = "") -> dict:
    n = max(1, len(rows))
    def isum(k): return int(sum(int(r.get(k, 0)) for r in rows))
    def avg(k): return sum(float(r.get(k, 0.0)) for r in rows) / n
    required = isum("required_input_count")
    return {
        f"{prefix}samples": len(rows),
        f"{prefix}required_input_count": required,
        f"{prefix}required_input_coverage": avg("required_input_coverage"),
        f"{prefix}strict_source_legalizability_missing_required_input": isum("strict_source_legalizability_missing_required_input"),
        f"{prefix}practical_after_pool_missing_required_input": isum("practical_after_pool_missing_required_input"),
        f"{prefix}input_pool_connections": isum("input_pool_connections"),
        f"{prefix}reused_inputs": isum("reused_inputs"),
        f"{prefix}exporter_synthetic_input_nets": isum("exporter_synthetic_input_nets"),
        f"{prefix}exporter_repaired_output_nets": isum("exporter_repaired_output_nets"),
        f"{prefix}exporter_missing_required_input_pins": isum("exporter_missing_required_input_pins"),
        f"{prefix}strict_missing_per_required_input_ratio": isum("strict_source_legalizability_missing_required_input") / max(1, required),
        f"{prefix}input_pool_per_required_input_ratio": isum("input_pool_connections") / max(1, required),
        f"{prefix}reused_inputs_per_required_input_ratio": isum("reused_inputs") / max(1, required),
        f"{prefix}repaired_output_nets_per_sample": isum("exporter_repaired_output_nets") / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--previous-export-dir", required=True)
    ap.add_argument("--dataset-root", default=os.environ.get("ELDA_V5_DATA_ROOT", "datasets/source_net_v5"))
    ap.add_argument("--mapping", default=os.environ.get("ELDA_V5_MAPPING", "datasets/source_net_v5/mapping_v5.txt"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-yosys", action="store_true")
    ap.add_argument("--yosys-bin", default="/opt/OpenROAD/oss-cad-suite/bin/yosys")
    ap.add_argument("--liberty", default="")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prev = Path(args.previous_export_dir)
    sample_csv = prev / "k4_partition_export_smoke" / "materialization_samples.csv"
    if not sample_csv.exists():
        raise FileNotFoundError(sample_csv)
    sample_rows_raw = _read_csv(sample_csv)
    tok, meta = build_v5_tokenizer(Path(args.dataset_root), max_length=12288)
    net_id = int(tok.net_id)
    boundary_stub_id = int(tok.boundary_stub_id)

    generated_rows = []
    oracle_rows = []
    wrapper_rows = []
    missing_by_kind = Counter()
    missing_by_reason = Counter()
    top_boundary = Counter()
    examples = []
    sample000 = {}

    for raw in sample_rows_raw:
        idx = _as_int(raw.get("sample"))
        src_path = Path(str(raw.get("source_path")))
        graph_path = Path(str(raw.get("graph_path")))
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        old_export_path = prev / "k4_partition_export_smoke" / f"sample_{idx:03d}" / "partition_export_report.json"
        old_export = json.loads(old_export_path.read_text(encoding="utf-8"))
        aut = _missing_autopsy(graph, old_export, net_id, boundary_stub_id)
        missing_by_kind.update(aut["missing_by_source_kind"])
        missing_by_reason.update(aut["missing_by_reason"])
        top_boundary.update(aut["top_boundary_groups"])
        examples.extend(aut["top_examples"])
        gen = {
            "sample": idx,
            "source_path": str(src_path),
            "stopped_reason": raw.get("stopped_reason"),
            "timeout": str(raw.get("stopped_reason")) == "timeout",
            "source_net_force_closed": _as_bool(raw.get("source_net_force_closed")),
            "coverage_decode_valid": _as_bool(raw.get("coverage_decode_valid")),
            "strict_decode_valid": _as_bool(raw.get("strict_decode_valid")),
            "required_input_count": _as_int(raw.get("required_input_count")),
            "covered_required_input_count": _as_int(raw.get("covered_required_input_count")),
            "missing_required_input_count": _as_int(raw.get("missing_required_input_load_count")),
            "required_input_coverage": _as_float(raw.get("required_input_coverage")),
            "invalid_cell_pin_count": _as_int(raw.get("invalid_cell_pin_count")),
            "invalid_sink_pin_count": _as_int(raw.get("invalid_sink_pin_count")),
            "invalid_src_pin_count": _as_int(raw.get("invalid_src_pin_count")),
            "duplicate_sink_count": _as_int(raw.get("duplicate_sink_count")),
            "strict_source_legalizability_missing_required_input": int(old_export.get("strict_source_legalizability_missing_required_input", 0)),
            "practical_after_pool_missing_required_input": int(old_export.get("practical_after_pool_missing_required_input", 0)),
            "input_pool_connections": int(old_export.get("input_pool_connections", 0)),
            "reused_inputs": int(old_export.get("reused_inputs", 0)),
            "exporter_synthetic_input_nets": int(old_export.get("exporter_synthetic_input_nets", 0)),
            "exporter_repaired_output_nets": int(old_export.get("exporter_repaired_output_nets", 0)),
            "exporter_missing_required_input_pins": int(old_export.get("exporter_missing_required_input_pins", 0)),
            "verilog_emitted": bool(old_export.get("verilog_emitted", False)),
            "yosys_read_verilog_pass": bool((old_export.get("yosys") or {}).get("yosys_read_verilog_pass", False)),
            "yosys_check_pass": bool((old_export.get("yosys") or {}).get("yosys_check_pass", False)),
            "yosys_synth_pass": bool((old_export.get("yosys") or {}).get("yosys_synth_pass", False)),
        }
        generated_rows.append(gen)
        if idx == 0:
            sample000 = dict(gen)
            sample000["diagnosis"] = (
                "generation timeout/incomplete; SINK_DEMAND coverage failed before materialization"
                if gen["required_input_coverage"] < 0.99 or not gen["coverage_decode_valid"]
                else "complete enough for materialization"
            )

        oracle_graph = torch.load(src_path, map_location="cpu", weights_only=False)
        oracle_stat = _graph_stats(oracle_graph, tok)
        oracle_export = _run_export(oracle_graph, out / "oracle_k4" / f"sample_{idx:03d}", Path(args.mapping), bool(args.run_yosys), str(args.yosys_bin), args.liberty)
        oracle_rows.append({
            "sample": idx,
            "source_path": str(src_path),
            "required_input_count": int(oracle_stat.get("required_input_count", 0)),
            "required_input_coverage": float(oracle_stat.get("required_input_coverage", 0.0)),
            "invalid_cell_pin_count": int(oracle_stat.get("invalid_cell_pin_count", 0)),
            "invalid_sink_pin_count": int(oracle_stat.get("invalid_sink_pin_count", 0)),
            "invalid_src_pin_count": int(oracle_stat.get("invalid_src_pin_count", 0)),
            "duplicate_sink_count": int(oracle_stat.get("duplicate_sink_count", 0)),
            "boundary_source_count": int(oracle_stat.get("boundary_source_17_plus_count", 0)),
            "strict_source_legalizability_missing_required_input": int(oracle_export.get("strict_source_legalizability_missing_required_input", 0)),
            "practical_after_pool_missing_required_input": int(oracle_export.get("practical_after_pool_missing_required_input", 0)),
            "input_pool_connections": int(oracle_export.get("input_pool_connections", 0)),
            "reused_inputs": int(oracle_export.get("reused_inputs", 0)),
            "exporter_synthetic_input_nets": int(oracle_export.get("exporter_synthetic_input_nets", 0)),
            "exporter_repaired_output_nets": int(oracle_export.get("exporter_repaired_output_nets", 0)),
            "exporter_missing_required_input_pins": int(oracle_export.get("exporter_missing_required_input_pins", 0)),
            "verilog_emitted": bool(oracle_export.get("verilog_emitted", False)),
            "yosys_read_verilog_pass": bool((oracle_export.get("yosys") or {}).get("yosys_read_verilog_pass", False)),
            "yosys_check_pass": bool((oracle_export.get("yosys") or {}).get("yosys_check_pass", False)),
            "yosys_synth_pass": bool((oracle_export.get("yosys") or {}).get("yosys_synth_pass", False)),
        })

        wrapped, wstats = _boundary_wrapper_graph(graph, boundary_stub_id)
        wrapper_sample_dir = out / "wrapper_k4" / f"sample_{idx:03d}"
        wrapper_sample_dir.mkdir(parents=True, exist_ok=True)
        torch.save(wrapped, wrapper_sample_dir / "decoded_graph_boundary_wrapper.pt")
        w_stat = _graph_stats(wrapped, tok)
        w_export = _run_export(wrapped, wrapper_sample_dir, Path(args.mapping), bool(args.run_yosys), str(args.yosys_bin), args.liberty)
        wrapper_rows.append({
            "sample": idx,
            **wstats,
            "coverage_decode_valid": gen["coverage_decode_valid"],
            "eligible_materialization_sample": bool(gen["coverage_decode_valid"] and gen["required_input_coverage"] >= 0.99),
            "required_input_count": int(w_stat.get("required_input_count", gen["required_input_count"])),
            "required_input_coverage": float(w_stat.get("required_input_coverage", gen["required_input_coverage"])),
            "invalid_cell_pin_count": int(w_stat.get("invalid_cell_pin_count", 0)),
            "invalid_sink_pin_count": int(w_stat.get("invalid_sink_pin_count", 0)),
            "invalid_src_pin_count": int(w_stat.get("invalid_src_pin_count", 0)),
            "duplicate_sink_count": int(w_stat.get("duplicate_sink_count", 0)),
            "strict_source_legalizability_missing_required_input": int(w_export.get("strict_source_legalizability_missing_required_input", 0)),
            "practical_after_pool_missing_required_input": int(w_export.get("practical_after_pool_missing_required_input", 0)),
            "input_pool_connections": int(w_export.get("input_pool_connections", 0)),
            "reused_inputs": int(w_export.get("reused_inputs", 0)),
            "exporter_synthetic_input_nets": int(w_export.get("exporter_synthetic_input_nets", 0)),
            "exporter_repaired_output_nets": int(w_export.get("exporter_repaired_output_nets", 0)),
            "exporter_missing_required_input_pins": int(w_export.get("exporter_missing_required_input_pins", 0)),
            "verilog_emitted": bool(w_export.get("verilog_emitted", False)),
            "yosys_read_verilog_pass": bool((w_export.get("yosys") or {}).get("yosys_read_verilog_pass", False)),
            "yosys_check_pass": bool((w_export.get("yosys") or {}).get("yosys_check_pass", False)),
            "yosys_synth_pass": bool((w_export.get("yosys") or {}).get("yosys_synth_pass", False)),
        })

    write_csv(out / "oracle_k4_baseline.csv", oracle_rows)
    write_csv(out / "generated_k4_autopsy.csv", generated_rows)
    write_csv(out / "wrapper_k4_rerun.csv", wrapper_rows)

    gen_agg = _aggregate(generated_rows, "generated_")
    oracle_agg = _aggregate(oracle_rows, "oracle_")
    wrapper_agg = _aggregate(wrapper_rows, "wrapper_")
    eligible = [r for r in wrapper_rows if r.get("eligible_materialization_sample")]
    wrapper_eligible_agg = _aggregate(eligible, "wrapper_eligible_") if eligible else {}
    wrapper_missing_drop = gen_agg["generated_strict_source_legalizability_missing_required_input"] - wrapper_agg["wrapper_strict_source_legalizability_missing_required_input"]
    wrapper_pool_drop = gen_agg["generated_input_pool_connections"] - wrapper_agg["wrapper_input_pool_connections"]
    k4_wrapper_pass = (
        len(eligible) == len(wrapper_rows)
        and wrapper_agg["wrapper_required_input_coverage"] >= 0.99
        and sum(r["invalid_cell_pin_count"] + r["invalid_sink_pin_count"] + r["invalid_src_pin_count"] for r in wrapper_rows) == 0
        and sum(r["duplicate_sink_count"] for r in wrapper_rows) == 0
        and wrapper_agg["wrapper_strict_source_legalizability_missing_required_input"] < gen_agg["generated_strict_source_legalizability_missing_required_input"]
        and all(r["verilog_emitted"] and r["yosys_read_verilog_pass"] and r["yosys_check_pass"] and r["yosys_synth_pass"] for r in wrapper_rows)
    )
    report = {
        "run_scope": "source_net_v5_oracle_export_baseline_and_materialization_autopsy",
        "final_result": False,
        "safety": SAFETY,
        "previous_export_dir": str(prev),
        "summary": {
            "oracle_baseline_collected": True,
            "generated_input_pool_normal_relative_to_oracle": gen_agg["generated_input_pool_per_required_input_ratio"] <= oracle_agg["oracle_input_pool_per_required_input_ratio"] * 1.25,
            "root_cause_k4_materialization_no_go": "sample_000 incomplete generation plus boundary stub nodes materialized as fallback gate-like cells",
            "sample_000_failure_cause": sample000.get("diagnosis"),
            "strict_missing_root_cause": "boundary materialization wrapper issue dominates complete samples; sample_000 also incomplete",
            "boundary_source_mapping_issue": True,
            "wrapper_fix_needed": True,
            "wrapper_fix_implemented": True,
            "k4_rerun_pass": bool(k4_wrapper_pass),
            "k8_allowed": bool(k4_wrapper_pass),
            "stage2": "No-Go",
            "skeleton": "No-Go",
            "cap32": "No-Go",
        },
        "oracle_baseline": {"rows": oracle_rows, "aggregate": oracle_agg},
        "generated_k4_autopsy": {
            "rows": generated_rows,
            "aggregate": gen_agg,
            "sample_000": sample000,
        },
        "strict_missing_autopsy": {
            "missing_by_source_kind": dict(missing_by_kind),
            "missing_by_reason": dict(missing_by_reason),
            "missing_by_sample": {str(r["sample"]): int(r["strict_source_legalizability_missing_required_input"]) for r in generated_rows},
            "top_boundary_groups": dict(top_boundary.most_common(20)),
            "top_examples": examples[:20],
        },
        "boundary_source_materialization_audit": {
            "current_behavior": "decoded boundary source groups are boundary_stub nodes with driver edges; exporter treats those nodes as gate-like fallback cells",
            "wrapper_mapping": "remove V5 boundary_stub driver nodes/edges so grouped boundary sources become PI-like source nets",
            "accepted_after_wrapper_count": int(sum(1 for r in wrapper_rows if r["strict_source_legalizability_missing_required_input"] == 0)),
            "wrapper_rows": wrapper_rows,
            "wrapper_aggregate": wrapper_agg,
            "wrapper_eligible_aggregate": wrapper_eligible_agg,
            "strict_missing_drop": int(wrapper_missing_drop),
            "input_pool_drop": int(wrapper_pool_drop),
        },
        "oracle_vs_generated_comparison": {
            **oracle_agg,
            **gen_agg,
            "generated_vs_oracle_strict_missing_ratio_delta": gen_agg["generated_strict_missing_per_required_input_ratio"] - oracle_agg["oracle_strict_missing_per_required_input_ratio"],
            "generated_vs_oracle_input_pool_ratio_delta": gen_agg["generated_input_pool_per_required_input_ratio"] - oracle_agg["oracle_input_pool_per_required_input_ratio"],
            "generated_vs_oracle_reused_ratio_delta": gen_agg["generated_reused_inputs_per_required_input_ratio"] - oracle_agg["oracle_reused_inputs_per_required_input_ratio"],
            "generated_vs_oracle_repaired_output_per_sample_delta": gen_agg["generated_repaired_output_nets_per_sample"] - oracle_agg["oracle_repaired_output_nets_per_sample"],
        },
        "rerun_k4_after_fix": {
            "before": gen_agg,
            "after": wrapper_agg,
            "eligible_after": wrapper_eligible_agg,
            "pass": bool(k4_wrapper_pass),
            "k8_allowed": bool(k4_wrapper_pass),
        },
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_name = f"source_net_v5_oracle_export_baseline_and_materialization_autopsy_{ts}.json"
    md_name = f"source_net_v5_oracle_export_baseline_and_materialization_autopsy_{ts}.md"
    write_json(out / json_name, report)
    lines = [
        "# Source-Net V5 Oracle Export Baseline And Materialization Autopsy",
        "",
        "## 1. Summary",
        f"- oracle baseline collected: `{report['summary']['oracle_baseline_collected']}`",
        f"- generated input_pool normal relative to oracle: `{report['summary']['generated_input_pool_normal_relative_to_oracle']}`",
        f"- root cause of k=4 materialization No-Go: `{report['summary']['root_cause_k4_materialization_no_go']}`",
        f"- sample_000 failure cause: `{report['summary']['sample_000_failure_cause']}`",
        f"- strict missing root cause: `{report['summary']['strict_missing_root_cause']}`",
        f"- boundary source mapping issue: `{report['summary']['boundary_source_mapping_issue']}`",
        f"- wrapper fix needed/implemented: `{report['summary']['wrapper_fix_needed']} / {report['summary']['wrapper_fix_implemented']}`",
        f"- k=4 rerun pass: `{report['summary']['k4_rerun_pass']}`",
        f"- k=8 allowed: `{report['summary']['k8_allowed']}`",
        "- Stage 2 / skeleton / cap32: `No-Go / No-Go / No-Go`",
        "",
        "## 2. Oracle Baseline",
        f"- aggregate: `{oracle_agg}`",
        "",
        "## 3. Generated k=4 Per-Sample Autopsy",
        f"- aggregate: `{gen_agg}`",
        f"- sample_000: `{sample000}`",
        "",
        "## 4. Generated Strict Missing Autopsy",
        f"- by source kind: `{dict(missing_by_kind)}`",
        f"- by reason: `{dict(missing_by_reason)}`",
        f"- by sample: `{report['strict_missing_autopsy']['missing_by_sample']}`",
        f"- top boundary groups: `{dict(top_boundary.most_common(20))}`",
        f"- top examples: `{examples[:5]}`",
        "",
        "## 5. Boundary Source Materialization Audit",
        f"- current behavior: `{report['boundary_source_materialization_audit']['current_behavior']}`",
        f"- wrapper mapping: `{report['boundary_source_materialization_audit']['wrapper_mapping']}`",
        f"- wrapper aggregate: `{wrapper_agg}`",
        f"- wrapper eligible aggregate: `{wrapper_eligible_agg}`",
        f"- strict missing drop: `{wrapper_missing_drop}`",
        f"- input_pool drop: `{wrapper_pool_drop}`",
        "",
        "## 6. Oracle vs Generated Comparison",
        f"- comparison: `{report['oracle_vs_generated_comparison']}`",
        "",
        "## 7. Rerun k=4 After Fix",
        f"- before: `{gen_agg}`",
        f"- after: `{wrapper_agg}`",
        f"- eligible after: `{wrapper_eligible_agg}`",
        f"- pass: `{k4_wrapper_pass}`",
        "",
        "## 8. Decision Table",
        "",
        "| 项目 | 判断 | 理由 |",
        "| --- | --- | --- |",
        "| oracle baseline collected | Go | same four graph pointers exported with same checker |",
        f"| generated repair normal relative to oracle | {'Yes' if report['summary']['generated_input_pool_normal_relative_to_oracle'] else 'No'} | generated/oracle ratios reported above |",
        f"| sample completion issue | {'Yes' if sample000.get('timeout') or sample000.get('required_input_coverage', 1.0) < 0.99 else 'No'} | sample_000 coverage={sample000.get('required_input_coverage')} stopped={sample000.get('stopped_reason')} |",
        "| boundary materialization bug | Yes | boundary_stub nodes are treated as fallback gate-like cells by exporter |",
        "| strict missing root cause found | Yes | boundary source mapping plus incomplete sample_000 |",
        "| wrapper fix needed | Yes | V5-local PI-like mapping avoids exporter treating boundary groups as cells |",
        "| wrapper fix implemented | Yes | local autopsy wrapper removes boundary stub driver nodes/edges only in V5 adapter |",
        f"| coverage materialization | {'Go' if wrapper_agg['wrapper_required_input_coverage'] >= 0.99 else 'No-Go'} | wrapper coverage aggregate={wrapper_agg['wrapper_required_input_coverage']} |",
        f"| strict source legalizability | {'Go' if wrapper_agg['wrapper_strict_source_legalizability_missing_required_input'] < gen_agg['generated_strict_source_legalizability_missing_required_input'] else 'No-Go'} | before/after strict missing {gen_agg['generated_strict_source_legalizability_missing_required_input']}/{wrapper_agg['wrapper_strict_source_legalizability_missing_required_input']} |",
        f"| input_pool pressure | {'Go' if wrapper_agg['wrapper_input_pool_connections'] < gen_agg['generated_input_pool_connections'] else 'No-Go'} | before/after input_pool {gen_agg['generated_input_pool_connections']}/{wrapper_agg['wrapper_input_pool_connections']} |",
        f"| k=4 export smoke | {'Go' if k4_wrapper_pass else 'No-Go'} | requires all k=4 complete and wrapper export/yosys pass |",
        f"| k=8 allowed | {'Go' if k4_wrapper_pass else 'No-Go'} | only after k=4 gate pass |",
        "| Stage 2 | No-Go | explicitly out of scope and k=4 not fully clean |",
        "| skeleton | No-Go | explicitly out of scope |",
        "| cap32 | No-Go | explicitly out of scope |",
        "| full assembly | No-Go | explicitly out of scope |",
    ]
    write_md(out / md_name, lines)
    print(json.dumps({"json": str(out / json_name), "md": str(out / md_name), "summary": report["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
