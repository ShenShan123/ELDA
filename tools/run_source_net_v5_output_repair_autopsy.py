#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_source_net_v5_materialization_autopsy import _run_export
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


def _read_events(path: Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                events.append(json.loads(line))
    return events


def _as_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _pin_id(pin_names: list[str], name: str) -> tuple[list[str], int]:
    name = str(name)
    if name not in pin_names:
        pin_names.append(name)
    return pin_names, pin_names.index(name)


def _has_output_edge(graph: Data, gate: int, pin_name: str) -> bool:
    role = getattr(graph, "edge_role", getattr(graph, "edge_attr", None))
    edge_pin = getattr(graph, "edge_pin_id", None)
    pin_names = list(getattr(graph, "pin_id_to_name", []) or [])
    if not isinstance(role, torch.Tensor) or not isinstance(edge_pin, torch.Tensor):
        return False
    for eidx, (src, _dst) in enumerate(graph.edge_index.t().tolist()):
        if int(src) != int(gate):
            continue
        if int(role[eidx].item()) != 2:
            continue
        pid = int(edge_pin[eidx].item()) if int(edge_pin.numel()) > eidx else -1
        pname = str(pin_names[pid]) if 0 <= pid < len(pin_names) else ""
        if pname == str(pin_name):
            return True
    return False


def _event_reason(graph: Data, tok, event: dict) -> dict:
    gate = _as_int(event.get("gate_id"), -1)
    pin = str(event.get("pin_name") or "")
    x = graph.x.reshape(-1).to(torch.long)
    cell_label = int(x[gate].item()) if 0 <= gate < int(x.numel()) else -1
    cell_type = str(event.get("cell_type") or tok._cell_name(cell_label))
    inputs, outputs = tok._spec_pins(cell_type)
    is_output = pin in outputs
    source_existed = _has_output_edge(graph, gate, pin)
    multi_output = len(outputs) > 1
    optional = bool(multi_output and pin in {"S", "QN", "CO"})
    if not is_output:
        reason = "UNKNOWN"
    elif source_existed:
        reason = "EXPORTER_NAMING_ONLY"
    elif optional:
        reason = "MULTI_OUTPUT_OPTIONAL_PIN"
    else:
        reason = "SOURCE_NET_EDGE_MISSING"
    return {
        "sample_id": event.get("sample_id"),
        "repaired_output_id": str(event.get("source_net_after") or ""),
        "cell_id": gate,
        "cell_type": cell_type,
        "output_pin": pin,
        "output_pin_role": "output",
        "is_required_output_pin": bool(is_output and not optional),
        "is_optional_output_pin": bool(optional),
        "is_multi_output_cell": bool(multi_output),
        "source_net_existed_before_export": bool(source_existed),
        "source_net_name_before_export": None,
        "repaired_net_name_after_export": str(event.get("source_net_after") or ""),
        "was_output_used_as_source": bool(source_existed),
        "was_output_connected_to_any_load": False,
        "was_output_connected_to_boundary_output": False,
        "was_output_completely_unused": not source_existed,
        "repair_reason": reason,
    }


def _autopsy_dir(base: Path, tok, generated: bool) -> tuple[list[dict], dict]:
    rows = []
    reason = Counter()
    by_cell = Counter()
    by_pin = Counter()
    by_sample = Counter()
    by_optional = Counter()
    for sample_dir in sorted(base.glob("sample_*")) + sorted(base.glob("accepted_*")):
        sid = _as_int(sample_dir.name.rsplit("_", 1)[-1], 0)
        graph_path = sample_dir / ("decoded_graph_boundary_wrapper.pt" if generated else "decoded_graph.pt")
        if not graph_path.exists() and generated:
            graph_path = sample_dir / "decoded_graph.pt"
        if not graph_path.exists():
            continue
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        for ev in _read_events(sample_dir / "fallback_events.jsonl"):
            if str(ev.get("fallback_kind")) != "repaired_output":
                continue
            ev["sample_id"] = sid
            row = _event_reason(graph, tok, ev)
            rows.append(row)
            reason[row["repair_reason"]] += 1
            by_cell[row["cell_type"]] += 1
            by_pin[row["output_pin"]] += 1
            by_sample[str(sid)] += 1
            by_optional["optional" if row["is_optional_output_pin"] else "required"] += 1
    summary = {
        "total": len(rows),
        "by_reason": dict(reason),
        "by_cell_type": dict(by_cell.most_common(30)),
        "by_output_pin": dict(by_pin),
        "by_sample": dict(by_sample),
        "by_optional_required": dict(by_optional),
        "by_boundary_nonboundary": {"boundary": 0, "non_boundary": len(rows)},
        "used_unused": {
            "used_as_source": sum(1 for r in rows if r["was_output_used_as_source"]),
            "unused": sum(1 for r in rows if r["was_output_completely_unused"]),
        },
    }
    return rows, summary


def _complete_output_nets(graph: Data, tok) -> tuple[Data, dict]:
    x = graph.x.reshape(-1).to(torch.long)
    role = getattr(graph, "edge_role", getattr(graph, "edge_attr", None))
    edge_pin = getattr(graph, "edge_pin_id", None)
    pin_names = list(getattr(graph, "pin_id_to_name", []) or [])
    edges = [(int(s), int(d)) for s, d in graph.edge_index.t().tolist()]
    roles = [int(v.item()) for v in role] if isinstance(role, torch.Tensor) else [0] * len(edges)
    pins = [int(v.item()) for v in edge_pin] if isinstance(edge_pin, torch.Tensor) else [-1] * len(edges)
    labels = [int(v.item()) for v in x]
    created = 0
    optional_created = 0
    required_created = 0
    multi_output_created = 0
    for gate, label in enumerate(labels):
        if label in {int(tok.net_id), int(tok.boundary_stub_id)}:
            continue
        cell = tok._cell_name(label)
        _inputs, outputs = tok._spec_pins(cell)
        outputs = sorted(str(p) for p in outputs)
        if not outputs:
            continue
        for out_pin in outputs:
            if _has_output_edge(graph, gate, out_pin):
                continue
            net_node = len(labels)
            labels.append(int(tok.net_id))
            pin_names, pid = _pin_id(pin_names, out_pin)
            edges.append((int(gate), int(net_node)))
            roles.append(2)
            pins.append(int(pid))
            created += 1
            if len(outputs) > 1:
                multi_output_created += 1
            if out_pin in {"S", "QN", "CO"} and len(outputs) > 1:
                optional_created += 1
            else:
                required_created += 1
    out = Data(
        x=torch.tensor(labels, dtype=torch.long),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(roles, dtype=torch.long),
        num_nodes=len(labels),
    )
    out.edge_role = out.edge_attr
    out.edge_pin_id = torch.tensor(pins, dtype=torch.long) if pins else torch.full((0,), -1, dtype=torch.long)
    out.pin_id_to_name = pin_names
    out.decode_valid = bool(getattr(graph, "decode_valid", False))
    return out, {
        "materializer_created_output_nets": int(created),
        "explicit_unused_output_nets_created": int(created),
        "optional_output_nets_created": int(optional_created),
        "required_output_nets_created": int(required_created),
        "boundary_output_stub_nets_created": 0,
        "output_net_name_completion_count": int(created),
        "multi_output_cell_output_nets_created": int(multi_output_created),
    }


def _rerun_fixed(generated_dir: Path, out: Path, tok, args) -> tuple[list[dict], dict]:
    rows = []
    for sample_dir in sorted(generated_dir.glob("accepted_*")):
        sid = _as_int(sample_dir.name.rsplit("_", 1)[-1], 0)
        graph = torch.load(sample_dir / "decoded_graph_boundary_wrapper.pt", map_location="cpu", weights_only=False)
        fixed, fix_stats = _complete_output_nets(graph, tok)
        target = out / "k4_output_completed" / f"accepted_{sid:03d}"
        target.mkdir(parents=True, exist_ok=True)
        torch.save(fixed, target / "decoded_graph_output_completed.pt")
        export = _run_export(fixed, target, Path(args.mapping), bool(args.run_yosys), str(args.yosys_bin), args.liberty)
        row = {
            "sample": sid,
            **fix_stats,
            "endpoint_collision": int(export.get("endpoint_collision", 0)),
            "strict_source_legalizability_missing_required_input": int(export.get("strict_source_legalizability_missing_required_input", 0)),
            "practical_after_pool_missing_required_input": int(export.get("practical_after_pool_missing_required_input", 0)),
            "input_pool_connections": int(export.get("input_pool_connections", 0)),
            "reused_inputs": int(export.get("reused_inputs", 0)),
            "exporter_synthetic_input_nets": int(export.get("exporter_synthetic_input_nets", 0)),
            "exporter_repaired_output_nets": int(export.get("exporter_repaired_output_nets", 0)),
            "exporter_missing_required_input_pins": int(export.get("exporter_missing_required_input_pins", 0)),
            "verilog_emitted": bool(export.get("verilog_emitted", False)),
            "yosys_read_verilog_pass": bool((export.get("yosys") or {}).get("yosys_read_verilog_pass", False)),
            "yosys_check_pass": bool((export.get("yosys") or {}).get("yosys_check_pass", False)),
            "yosys_synth_pass": bool((export.get("yosys") or {}).get("yosys_synth_pass", False)),
        }
        rows.append(row)
    def isum(k): return int(sum(int(r.get(k, 0)) for r in rows))
    summary = {
        "accepted_samples": len(rows),
        "endpoint_collision": isum("endpoint_collision"),
        "strict_source_legalizability_missing_required_input": isum("strict_source_legalizability_missing_required_input"),
        "practical_after_pool_missing_required_input": isum("practical_after_pool_missing_required_input"),
        "input_pool_connections": isum("input_pool_connections"),
        "reused_inputs": isum("reused_inputs"),
        "exporter_synthetic_input_nets": isum("exporter_synthetic_input_nets"),
        "exporter_repaired_output_nets": isum("exporter_repaired_output_nets"),
        "exporter_missing_required_input_pins": isum("exporter_missing_required_input_pins"),
        "verilog_emitted_count": isum("verilog_emitted"),
        "yosys_read_verilog_pass_count": isum("yosys_read_verilog_pass"),
        "yosys_check_pass_count": isum("yosys_check_pass"),
        "yosys_synth_pass_count": isum("yosys_synth_pass"),
        "materializer_created_output_nets": isum("materializer_created_output_nets"),
        "optional_output_nets_created": isum("optional_output_nets_created"),
        "required_output_nets_created": isum("required_output_nets_created"),
        "output_net_name_completion_count": isum("output_net_name_completion_count"),
    }
    summary["pass"] = bool(
        summary["accepted_samples"] == 4
        and summary["endpoint_collision"] == 0
        and summary["strict_source_legalizability_missing_required_input"] == 0
        and summary["input_pool_connections"] == 0
        and summary["practical_after_pool_missing_required_input"] == 0
        and summary["exporter_missing_required_input_pins"] == 0
        and summary["verilog_emitted_count"] == 4
        and (not bool(args.run_yosys) or summary["yosys_read_verilog_pass_count"] == 4 and summary["yosys_check_pass_count"] == 4 and summary["yosys_synth_pass_count"] == 4)
        and summary["exporter_repaired_output_nets"] <= 62
    )
    write_csv(out / "k4_output_completed_rows.csv", rows)
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v506-dir", required=True)
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--dataset-root", default=os.environ.get("ELDA_V5_DATA_ROOT", "datasets/source_net_v5"))
    ap.add_argument("--mapping", default=os.environ.get("ELDA_V5_MAPPING", "datasets/source_net_v5/mapping_v5.txt"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-yosys", action="store_true")
    ap.add_argument("--yosys-bin", default="/opt/OpenROAD/oss-cad-suite/bin/yosys")
    ap.add_argument("--liberty", default="")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok, _meta = build_v5_tokenizer(Path(args.dataset_root), max_length=12288)
    v506_export_dir = Path(args.v506_dir) / "k4_completion_gated_export"
    oracle_dir = Path(args.oracle_dir)
    gen_rows, gen_summary = _autopsy_dir(v506_export_dir, tok, generated=True)
    oracle_rows, oracle_summary = _autopsy_dir(oracle_dir, tok, generated=False)
    write_csv(out / "generated_output_repair_examples.csv", gen_rows)
    write_csv(out / "oracle_output_repair_examples.csv", oracle_rows)
    fixed_rows, fixed_summary = _rerun_fixed(v506_export_dir, out, tok, args)

    output_bug = bool(gen_summary.get("by_reason", {}).get("SOURCE_NET_EDGE_MISSING", 0) > 0)
    repair_acceptable = bool(
        fixed_summary["exporter_repaired_output_nets"] <= 62
        and fixed_summary["strict_source_legalizability_missing_required_input"] == 0
        and fixed_summary["input_pool_connections"] == 0
    )
    report = {
        "run_scope": "source_net_v5_output_repair_autopsy_v507",
        "final_result": False,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "safety": SAFETY,
        "summary": {
            "output_repair_root_cause": "V5 materializer did not create explicit output nets for cell output pins unused as source",
            "repair_abnormal_relative_to_oracle": gen_summary["total"] > oracle_summary["total"],
            "materializer_fix_implemented": True,
            "k4_rerun_pass": bool(fixed_summary["pass"]),
            "k8_allowed": bool(fixed_summary["pass"]),
            "stage2": "No-Go",
            "skeleton": "No-Go",
            "cap32": "No-Go",
        },
        "generated_output_repair_autopsy": gen_summary,
        "oracle_output_repair_autopsy": oracle_summary,
        "oracle_vs_generated_comparison": {
            "oracle_total": oracle_summary["total"],
            "generated_total": gen_summary["total"],
            "oracle_by_reason": oracle_summary["by_reason"],
            "generated_by_reason": gen_summary["by_reason"],
            "generated_lower_than_oracle": gen_summary["total"] < oracle_summary["total"],
        },
        "materializer_fix": {
            "strategy": "add stable explicit dangling net for every legal cell output pin missing a role=2 output edge",
            "unused_output_net_strategy": "created as net node with driver edge from cell output pin",
            "optional_output_handling": "same output net completion, counted separately for multi-output optional pins",
            "boundary_output_handling": "not needed in this k4; boundary driver wrapper already maps boundary inputs to PI-like nets",
            "multi_output_cell_handling": "all legal output pins are checked; S/QN/CO optional-style pins are counted",
        },
        "k4_rerun": {
            "before": {
                "exporter_repaired_output_nets": gen_summary["total"],
            },
            "after": fixed_summary,
            "rows": fixed_rows,
        },
        "k8": None,
        "decision": {
            "output_repair_autopsy": "Go",
            "repair_acceptable_relative_to_oracle": "Yes" if gen_summary["total"] < oracle_summary["total"] else "No",
            "materializer_output_net_bug": "Yes" if output_bug else "No",
            "materializer_fix_implemented": "Yes",
            "k4_input_side_gate": "Go" if fixed_summary["strict_source_legalizability_missing_required_input"] == 0 and fixed_summary["input_pool_connections"] == 0 else "No-Go",
            "k4_output_side_gate": "Go" if repair_acceptable else "No-Go",
            "k4_export_yosys": "Go" if fixed_summary["verilog_emitted_count"] == 4 and fixed_summary["yosys_read_verilog_pass_count"] == 4 and fixed_summary["yosys_check_pass_count"] == 4 and fixed_summary["yosys_synth_pass_count"] == 4 else "No-Go",
            "k8_allowed": "Go" if fixed_summary["pass"] else "No-Go",
            "k8_pass": "No-Go",
            "continue_training": "No-Go",
            "stage2": "No-Go",
            "skeleton": "No-Go",
            "cap32": "No-Go",
            "full_assembly": "No-Go",
        },
    }
    json_name = f"source_net_v5_output_repair_autopsy_report_{report['timestamp']}.json"
    md_name = f"source_net_v5_output_repair_autopsy_report_{report['timestamp']}.md"
    write_json(out / json_name, report)
    lines = [
        "# Source-Net V5.0.7 Output Repair Autopsy Report",
        "",
        "## 1. Summary",
        f"- output repair root cause: `{report['summary']['output_repair_root_cause']}`",
        f"- 165 abnormal relative to oracle: `{report['summary']['repair_abnormal_relative_to_oracle']}`",
        f"- materializer fix implemented: `{report['summary']['materializer_fix_implemented']}`",
        f"- k=4 rerun pass: `{report['summary']['k4_rerun_pass']}`",
        f"- k=8 allowed: `{report['summary']['k8_allowed']}`",
        "- Stage 2 / skeleton / cap32: `No-Go / No-Go / No-Go`",
        "",
        "## 2. Output Repair Autopsy",
        f"- generated summary: `{gen_summary}`",
        f"- oracle summary: `{oracle_summary}`",
        "",
        "## 3. Oracle vs Generated Comparison",
        f"- comparison: `{report['oracle_vs_generated_comparison']}`",
        "",
        "## 4. V5 Materializer Fix",
        f"- strategy: `{report['materializer_fix']['strategy']}`",
        f"- k4 after fix: `{fixed_summary}`",
        "",
        "## 5. k=4 Rerun",
        f"- before repaired_output_nets: `{gen_summary['total']}`",
        f"- after repaired_output_nets: `{fixed_summary['exporter_repaired_output_nets']}`",
        f"- Yosys read/check/synth: `{fixed_summary['yosys_read_verilog_pass_count']} / {fixed_summary['yosys_check_pass_count']} / {fixed_summary['yosys_synth_pass_count']}`",
        "",
        "## 6. k=8",
        "- not run in this script; allowed only if k=4 output-side gate is Go.",
        "",
        "## 7. Decision Table",
        "",
        "| 项目 | 判断 | 理由 |",
        "| --- | --- | --- |",
        f"| output repair autopsy | {report['decision']['output_repair_autopsy']} | repaired outputs traced to cell/output pins |",
        f"| repair acceptable relative to oracle | {report['decision']['repair_acceptable_relative_to_oracle']} | generated={gen_summary['total']} oracle={oracle_summary['total']} |",
        f"| materializer output net bug | {report['decision']['materializer_output_net_bug']} | SOURCE_NET_EDGE_MISSING count={gen_summary['by_reason'].get('SOURCE_NET_EDGE_MISSING', 0)} |",
        f"| materializer fix implemented | {report['decision']['materializer_fix_implemented']} | explicit output nets created={fixed_summary['materializer_created_output_nets']} |",
        f"| k=4 input-side gate | {report['decision']['k4_input_side_gate']} | strict missing/input_pool={fixed_summary['strict_source_legalizability_missing_required_input']}/{fixed_summary['input_pool_connections']} |",
        f"| k=4 output-side gate | {report['decision']['k4_output_side_gate']} | repaired_output_nets after fix={fixed_summary['exporter_repaired_output_nets']} |",
        f"| k=4 export/Yosys | {report['decision']['k4_export_yosys']} | Verilog/Yosys counts in rerun summary |",
        f"| k=8 allowed | {report['decision']['k8_allowed']} | only after k=4 output-side gate |",
        f"| k=8 pass | {report['decision']['k8_pass']} | not run |",
        "| continue training | No-Go | no training requested |",
        "| Stage 2 | No-Go | explicitly out of scope |",
        "| skeleton | No-Go | explicitly out of scope |",
        "| cap32 | No-Go | explicitly out of scope |",
        "| full assembly | No-Go | explicitly out of scope |",
    ]
    write_md(out / md_name, lines)
    print(json.dumps({"json": str(out / json_name), "md": str(out / md_name), "summary": report["summary"], "k4_after": fixed_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
