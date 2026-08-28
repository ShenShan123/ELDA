#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from elda.models.seq_models import SequenceModel, SourceNetV5Grammar
from tools.v5_source_net_common import build_v5_tokenizer, iter_partition_paths, write_csv, write_json, write_md


FANOUT_KEYS = ("0", "1", "2_4", "5_8", "9_16", "17_PLUS")


def _materialize_and_strict_export(dec, attempt_dir: Path, tok, args) -> dict:
    """Materialize a decoded partition and run the existing strict exporter."""
    # These helpers import this module for graph statistics, so they must be
    # loaded lazily to avoid an import cycle during normal sampler startup.
    from tools.run_source_net_v5_materialization_autopsy import _boundary_wrapper_graph, _run_export
    from tools.run_source_net_v5_output_repair_autopsy import _complete_output_nets

    completed, completion_stats = _complete_output_nets(dec, tok)
    completed_path = attempt_dir / "decoded_graph_with_stubs_output_completed.pt"
    torch.save(completed, completed_path)

    wrapped, wrapper_stats = _boundary_wrapper_graph(completed, int(tok.boundary_stub_id))
    wrapped_path = attempt_dir / "decoded_graph_boundary_wrapper.pt"
    torch.save(wrapped, wrapped_path)

    export_dir = attempt_dir / "export"
    export = _run_export(
        wrapped,
        export_dir,
        Path(args.mapping),
        bool(args.run_yosys),
        str(args.yosys_bin),
        str(args.liberty or ""),
    )
    emitted = Path(str(export.get("verilog_path", "")))
    candidate = attempt_dir / "candidate.v"
    if emitted.exists():
        shutil.copyfile(emitted, candidate)

    yosys = dict(export.get("yosys") or {})
    checks = {
        "candidate_verilog_emitted": bool(candidate.exists()),
        "endpoint_collision_zero": int(export.get("endpoint_collision", 0)) == 0,
        "strict_missing_required_input_zero": int(export.get("strict_source_legalizability_missing_required_input", 0)) == 0,
        "post_pool_missing_required_input_zero": int(export.get("practical_after_pool_missing_required_input", 0)) == 0,
        "input_pool_unused": int(export.get("input_pool_connections", 0)) == 0,
        "exporter_missing_input_pins_zero": int(export.get("exporter_missing_required_input_pins", 0)) == 0,
    }
    if bool(args.run_yosys):
        checks.update({
            "yosys_read": bool(yosys.get("yosys_read_verilog_pass", False)),
            "yosys_check": bool(yosys.get("yosys_check_pass", False)),
            "yosys_synth": bool(yosys.get("yosys_synth_pass", False)),
        })
    strict_report = {
        "strict_export_pass": bool(all(checks.values())),
        "checks": checks,
        "decoded_graph_with_stubs_output_completed_path": str(completed_path),
        "boundary_wrapper_graph_path": str(wrapped_path),
        "candidate_verilog_path": str(candidate) if candidate.exists() else "",
        "export_report_path": str(export_dir / "export_report.json"),
        "yosys_log_path": str(yosys.get("log_path", "")),
        "output_completion": completion_stats,
        "boundary_wrapper": wrapper_stats,
        "export": export,
    }
    strict_path = attempt_dir / "strict_export_report.json"
    write_json(strict_path, strict_report)
    write_json(attempt_dir / "yosys_report.json", yosys)
    return {
        "output_completed_graph_path": str(completed_path),
        "boundary_wrapper_graph_path": str(wrapped_path),
        "candidate_verilog_path": str(candidate) if candidate.exists() else "",
        "strict_export_report_path": str(strict_path),
        "yosys_report_path": str(attempt_dir / "yosys_report.json"),
        "strict_export_pass": bool(strict_report["strict_export_pass"]),
        "endpoint_collision": int(export.get("endpoint_collision", 0)),
        "strict_missing_required_input": int(export.get("strict_source_legalizability_missing_required_input", 0)),
        "post_pool_missing_required_input": int(export.get("practical_after_pool_missing_required_input", 0)),
        "input_pool_connections": int(export.get("input_pool_connections", 0)),
        "exporter_missing_input_pins": int(export.get("exporter_missing_required_input_pins", 0)),
        "output_completion_created_nets": int(completion_stats.get("materializer_created_output_nets", 0)),
        "yosys_read": bool(yosys.get("yosys_read_verilog_pass", False)),
        "yosys_check": bool(yosys.get("yosys_check_pass", False)),
        "yosys_synth": bool(yosys.get("yosys_synth_pass", False)),
    }


@dataclass
class V6PracticalTrace:
    constraint_level: int = 0
    enabled_masks: list[str] = field(default_factory=list)
    masked_invalid_cell_tokens: int = 0
    masked_invalid_pin_tokens: int = 0
    masked_invalid_demand_tokens: int = 0
    masked_duplicate_sink_tokens: int = 0
    masked_duplicate_source_tokens: int = 0
    eos_tokens_blocked_by_remaining_internal_demand: int = 0
    generation_dead_end: bool = False
    fallback_level_used: int | None = None
    max_new_tokens_hit: bool = False
    # SOURCE_NET emits a bounded load list: high-fanout nets carry only an
    # eight-load preview.  These are observability counters, not netlist
    # coverage counters (SINK_DEMAND is the complete connection view).
    source_net_emitted_demand_count: int = 0
    source_net_non_emitted_demand_count: int = 0
    duplicate_sink_count: int = 0
    duplicate_source_group_count: int = 0
    invalid_source_pin_count: int = 0
    invalid_load_pin_count: int = 0
    source_demand_mismatch_count: int = 0
    forced_load_group_end_count: int = 0
    exhausted_source_group_close_count: int = 0
    malformed_load_entry_count: int = 0
    driver_conflict_nets: int = 0
    internal_missing_source: int = 0
    added_missing_vs_source_partition: int | None = None
    level3_coverage_eos_gating_enabled: bool = False
    level3_requested_but_inactive: bool = False
    dead_end_step: int | None = None
    dead_end_state: dict | None = None
    events: list[dict] = field(default_factory=list)

    def asdict(self) -> dict:
        return {
            "constraint_level": int(self.constraint_level),
            "enabled_masks": list(self.enabled_masks),
            "number_of_masked_invalid_cell_tokens": int(self.masked_invalid_cell_tokens),
            "number_of_masked_invalid_pin_tokens": int(self.masked_invalid_pin_tokens),
            "number_of_masked_invalid_demand_tokens": int(self.masked_invalid_demand_tokens),
            "number_of_masked_duplicate_sink_tokens": int(self.masked_duplicate_sink_tokens),
            "number_of_masked_duplicate_source_tokens": int(self.masked_duplicate_source_tokens),
            "number_of_EOS_tokens_blocked_by_remaining_internal_demand": int(self.eos_tokens_blocked_by_remaining_internal_demand),
            "generation_dead_end": bool(self.generation_dead_end),
            "fallback_level_used": self.fallback_level_used,
            "max_new_tokens_hit": bool(self.max_new_tokens_hit),
            "source_net_emitted_demand_count": int(self.source_net_emitted_demand_count),
            "source_net_non_emitted_demand_count": int(self.source_net_non_emitted_demand_count),
            "duplicate_sink_count": int(self.duplicate_sink_count),
            "duplicate_source_group_count": int(self.duplicate_source_group_count),
            "invalid_source_pin_count": int(self.invalid_source_pin_count),
            "invalid_load_pin_count": int(self.invalid_load_pin_count),
            "source_demand_mismatch_count": int(self.source_demand_mismatch_count),
            "forced_load_group_end_count": int(self.forced_load_group_end_count),
            "exhausted_source_group_close_count": int(self.exhausted_source_group_close_count),
            "malformed_load_entry_count": int(self.malformed_load_entry_count),
            "driver_conflict_nets": int(self.driver_conflict_nets),
            "internal_missing_source": int(self.internal_missing_source),
            "added_missing_vs_source_partition": self.added_missing_vs_source_partition,
            "level3_coverage_eos_gating_enabled": bool(self.level3_coverage_eos_gating_enabled),
            "level3_requested_but_inactive": bool(self.level3_requested_but_inactive),
            "dead_end_step": self.dead_end_step,
            "dead_end_state": self.dead_end_state,
            "events": list(self.events[-200:]),
        }


class V6PracticalRuntime:
    def __init__(self, tok, ctx: dict):
        self.tok = tok
        self.ctx = ctx
        self.current_record: list[int] = []
        self.used_demands: set[int] = set()
        self.used_endpoints: set[tuple[int, int]] = set()
        self.used_source_identities: set[tuple[int, int, int]] = set()
        self.endpoint_counts: Counter = Counter()
        self.source_counts: Counter = Counter()
        self.invalid_source_pin_count = 0
        self.invalid_load_pin_count = 0
        self.source_demand_mismatch_count = 0
        self.malformed_load_entry_count = 0
        self.completed_record_count = 0

    def consume(self, tid: int) -> None:
        tid = int(tid)
        tok = self.tok
        if tid == int(tok.source_net_begin):
            self.current_record = [tid]
            return
        if not self.current_record:
            return
        self.current_record.append(tid)
        if tid == int(tok.source_net_end):
            self._commit_record(self.current_record)
            self.current_record = []

    def _commit_record(self, rec: list[int]) -> None:
        tok = self.tok
        sid = _source_identity_from_record(rec, tok)
        if sid is not None:
            self.used_source_identities.add(sid)
            self.source_counts[sid] += 1
            kind, src_cell, src_pin = sid
            if int(kind) == int(tok.cell_output):
                src_idx = _src_idx_from_tok(tok, int(src_cell))
                cell = self.ctx["cells"].get(src_idx)
                if cell is None or int(src_pin) not in cell["output_pin_tokens"]:
                    self.invalid_source_pin_count += 1
            elif sid not in self.ctx["source_candidates"]:
                self.invalid_source_pin_count += 1
        i = 0
        while i < len(rec):
            if int(rec[i]) == int(tok.load_cell) and i + 5 < len(rec):
                load_tok = int(rec[i + 1])
                load_idx = _gate_idx_from_tok(tok, load_tok)
                pin_tok = int(rec[i + 3]) if int(rec[i + 2]) == int(tok.load_pin) else None
                did_tok = int(rec[i + 5]) if pin_tok is not None and int(rec[i + 4]) == int(tok.demand_id) else None
                if pin_tok is None or did_tok is None:
                    self.malformed_load_entry_count += 1
                    i += 1
                    continue
                if load_idx is not None and pin_tok is not None:
                    endpoint = (int(load_idx), int(pin_tok))
                    self.used_endpoints.add(endpoint)
                    self.endpoint_counts[endpoint] += 1
                    cell = self.ctx["cells"].get(int(load_idx))
                    if cell is None or int(pin_tok) not in cell["input_pin_tokens"]:
                        self.invalid_load_pin_count += 1
                if did_tok is not None:
                    did = _demand_idx_from_tok(tok, did_tok)
                    if did is not None:
                        self.used_demands.add(int(did))
                        if sid is not None and self.ctx["sink_by_demand"].get(int(did), {}).get("source_identity") != sid:
                            self.source_demand_mismatch_count += 1
                i += 6
                continue
            if int(rec[i]) == int(tok.load_cell):
                self.malformed_load_entry_count += 1
            i += 1
        self.completed_record_count += 1

    def _record_refs(self, rec: list[int]) -> tuple[set[int], set[tuple[int, int]]]:
        tok = self.tok
        demands = set()
        endpoints = set()
        i = 0
        while i < len(rec):
            if int(rec[i]) == int(tok.load_cell) and i + 5 < len(rec):
                load_idx = _gate_idx_from_tok(tok, int(rec[i + 1]))
                pin_tok = int(rec[i + 3]) if int(rec[i + 2]) == int(tok.load_pin) else None
                did_tok = int(rec[i + 5]) if pin_tok is not None and int(rec[i + 4]) == int(tok.demand_id) else None
                if load_idx is not None and pin_tok is not None:
                    endpoints.add((int(load_idx), int(pin_tok)))
                if did_tok is not None:
                    did = _demand_idx_from_tok(tok, did_tok)
                    if did is not None:
                        demands.add(int(did))
                i += 6
                continue
            i += 1
        return demands, endpoints

    def snapshot(self) -> dict:
        current_demands, current_endpoints = self._record_refs(self.current_record)
        used_demands = set(self.used_demands) | set(current_demands)
        used_endpoints = set(self.used_endpoints) | set(current_endpoints)
        return {
            "used_demands": used_demands,
            "used_endpoints": used_endpoints,
            "used_source_identities": set(self.used_source_identities),
            "duplicate_sink_count": sum(v - 1 for v in self.endpoint_counts.values() if v > 1),
            "duplicate_source_group_count": sum(v - 1 for v in self.source_counts.values() if v > 1),
            "invalid_source_pin_count": int(self.invalid_source_pin_count),
            "invalid_load_pin_count": int(self.invalid_load_pin_count),
            "source_demand_mismatch_count": int(self.source_demand_mismatch_count),
            "malformed_load_entry_count": int(self.malformed_load_entry_count),
            "current_record": list(self.current_record),
            "current_source_identity": _source_identity_from_record(self.current_record, self.tok),
            "completed_record_count": int(self.completed_record_count),
        }


def _tok_name(tok, tid: int) -> str:
    tid = int(tid)
    for name in getattr(tok, "special_toks", []):
        if int(getattr(tok, str(name).lower(), -10**9)) == tid:
            return str(name)
    try:
        if tok.max_num_nodes is not None and tok.idx_offset <= tid < tok.idx_offset + tok.max_num_nodes:
            return f"GATE_{tid - tok.idx_offset}"
        if tok.max_num_nodes is not None and tok.idx_offset + tok.max_num_nodes <= tid < tok.idx_offset + 2 * tok.max_num_nodes:
            return f"SRC_{tid - tok.idx_offset - tok.max_num_nodes}"
        if tok.max_num_nodes is not None and tok.idx_offset + 2 * tok.max_num_nodes <= tid < tok.pin_offset:
            return f"DEMAND_{tid - tok.idx_offset - 2 * tok.max_num_nodes}"
        if tok.pin_offset <= tid < tok.node_type_offset:
            return f"PIN_{tok._pin_from_tok(tid)}"
        if tid >= tok.node_type_offset:
            return f"CELL_{tok._cell_name(tok._cell_from_tok(tid))}"
    except Exception:
        pass
    return f"TOK_{tid}"


def _section_state(token_ids: list[int], tok) -> dict:
    sections = [
        ("COND", int(tok.cond_section_begin), int(tok.cond_section_end)),
        ("DEMAND", int(tok.demand_section_begin), int(tok.demand_section_end)),
        ("SOURCE_BUDGET", int(tok.source_budget_section_begin), int(tok.source_budget_section_end)),
        ("CELL", int(tok.cell_section_begin), int(tok.cell_section_end)),
        ("SINK_DEMAND", int(tok.sink_demand_section_begin), int(tok.sink_demand_section_end)),
        ("SOURCE_NET", int(tok.source_net_section_begin), int(tok.source_net_section_end)),
    ]
    counts = {name: 0 for name, _b, _e in sections}
    opened = []
    closed = set()
    current = "NONE"
    for tid in token_ids:
        for name, begin, end in sections:
            if tid == begin:
                opened.append((name, begin))
                current = name
            elif tid == end:
                closed.add(begin)
                if current == name:
                    current = "NONE"
        for name, begin, end in sections:
            if opened and opened[-1][0] == name and begin not in closed:
                counts[name] += 1
                break
    for name, begin in reversed(opened):
        if begin not in closed:
            current = name
            break
    record_counts = {
        "cell_records": sum(1 for t in token_ids if t == int(tok.cell)),
        "demand_records": 0,
        "source_net_records": sum(1 for t in token_ids if t == int(tok.source_net_begin)),
        "sink_demand_records_estimate": 0,
    }
    in_demand = False
    in_sink = False
    for t in token_ids:
        if t == int(tok.demand_section_begin):
            in_demand = True
        elif t == int(tok.demand_section_end):
            in_demand = False
        elif t == int(tok.sink_demand_section_begin):
            in_sink = True
        elif t == int(tok.sink_demand_section_end):
            in_sink = False
        elif in_demand and t == int(tok.demand_begin):
            record_counts["demand_records"] += 1
        elif in_sink and t == int(tok.demand_begin):
            record_counts["sink_demand_records_estimate"] += 1
    return {
        "current_section": current,
        "section_token_counts": counts,
        "record_counts": record_counts,
        "closed_sections": [name for name, begin, _end in sections if begin in closed],
        "open_sections": [name for name, begin in opened if begin not in closed],
    }


def _coverage_decode_status(token_ids: list[int], tok) -> dict:
    state = _section_state(token_ids, tok)
    closed = set(state.get("closed_sections", []))
    source_expected = int(getattr(tok, "source_net_v5_expected_source_net_records", -1) or -1)
    source_generated = int(state.get("record_counts", {}).get("source_net_records", 0))
    source_complete = "SOURCE_NET" in closed
    required_closed = all(x in closed for x in ["COND", "DEMAND", "SOURCE_BUDGET", "CELL", "SINK_DEMAND"])
    return {
        "strict_decode_valid": bool(required_closed and source_complete and (token_ids and int(token_ids[-1]) == int(tok.eos))),
        "coverage_decode_valid": bool(required_closed),
        "source_net_complete": bool(source_complete),
        "source_net_records_generated": int(source_generated),
        "source_net_records_expected": int(source_expected),
        "source_net_completion_ratio": float(source_generated) / float(max(1, source_expected)) if source_expected >= 0 else 0.0,
        "source_net_partial_reason": "complete" if source_complete else ("not_started" if source_generated == 0 else "incomplete"),
        "closed_sections": sorted(closed),
    }


def _sink_demand_id_stats(token_ids: list[int], tok) -> dict:
    body = _section_body(token_ids, int(tok.sink_demand_section_begin), int(tok.sink_demand_section_end))
    records = _record_chunks(body, int(tok.demand_begin), int(tok.demand_end))
    vals = []
    for rec in records:
        for i, tid in enumerate(rec[:-1]):
            if int(tid) == int(tok.demand_id):
                nxt = int(rec[i + 1])
                if tok.idx_offset + 2 * tok.max_num_nodes <= nxt < tok.pin_offset:
                    vals.append(int(nxt - tok.idx_offset - 2 * tok.max_num_nodes))
                break
    c = Counter(vals)
    expected = int(getattr(tok, "source_net_v5_expected_sink_demand_records", -1) or -1)
    return {
        "demand_id_duplicate_count": int(sum(v - 1 for v in c.values() if v > 1)),
        "demand_id_missing_count": int(max(0, expected - len(set(vals)))) if expected >= 0 else 0,
        "sink_demand_id_count": int(len(vals)),
    }


def _grammar_allowed_summary(token_ids: list[int], tok, device, use_grammar: bool) -> dict:
    if not use_grammar:
        return {
            "grammar_enabled": False,
            "allowed_count": None,
            "eos_allowed": None,
            "section_end_allowed": None,
            "top_allowed_tokens": [],
            "mask_all_invalid": None,
        }
    ids = torch.tensor(token_ids, dtype=torch.long, device=device).reshape(1, -1)
    scores = torch.zeros((1, len(tok)), dtype=torch.float32, device=device)
    grammar = SourceNetV5Grammar(tok, 1, device, mask_mode="source_net_v5_light")
    masked = grammar(ids, scores.clone())
    finite = torch.isfinite(masked[0])
    allowed = torch.nonzero(finite, as_tuple=False).flatten().detach().cpu().tolist()
    state = _section_state(token_ids, tok)
    current = state["current_section"]
    section_end_id = None
    end_map = {
        "COND": tok.cond_section_end,
        "DEMAND": tok.demand_section_end,
        "SOURCE_BUDGET": tok.source_budget_section_end,
        "CELL": tok.cell_section_end,
        "SINK_DEMAND": tok.sink_demand_section_end,
        "SOURCE_NET": tok.source_net_section_end,
    }
    if current in end_map:
        section_end_id = int(end_map[current])
    return {
        "grammar_enabled": True,
        "allowed_count": len(allowed),
        "eos_allowed": int(tok.eos) in set(allowed),
        "section_end_allowed": (section_end_id in set(allowed)) if section_end_id is not None else None,
        "section_end_token": _tok_name(tok, section_end_id) if section_end_id is not None else None,
        "top_allowed_tokens": [_tok_name(tok, x) for x in allowed[:20]],
        "mask_all_invalid": len(allowed) == 0,
    }


def _top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits.clone()
    if top_k and int(top_k) > 0 and int(top_k) < logits.numel():
        kth = torch.topk(logits, int(top_k)).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p and float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > float(top_p)
        if remove.numel() > 0:
            remove[1:] = remove[:-1].clone()
            remove[0] = False
        logits[sorted_indices[remove]] = float("-inf")
    return logits


def _source_net_plan_from_record(rec: list[int], tok) -> dict:
    fanout_bucket = None
    load_begin_pos = None
    load_end_pos = None
    load_count = 0
    for idx, tid in enumerate(rec):
        tid = int(tid)
        if tid in {int(tok.fanout_1), int(tok.fanout_2_4), int(tok.fanout_5_8), int(tok.fanout_9_16), int(tok.fanout_17_plus)}:
            fanout_bucket = tid
        if tid in {int(tok.load_group_begin), int(tok.load_preview_begin)}:
            load_begin_pos = int(idx)
        if tid in {int(tok.load_group_end), int(tok.load_preview_end)}:
            load_end_pos = int(idx)
        if tid == int(tok.load_cell):
            load_count += 1
    if fanout_bucket == int(tok.fanout_1):
        min_loads, max_loads, end_token = 1, 1, int(tok.load_group_end)
    elif fanout_bucket == int(tok.fanout_2_4):
        min_loads, max_loads, end_token = 2, 4, int(tok.load_group_end)
    elif fanout_bucket == int(tok.fanout_5_8):
        min_loads, max_loads, end_token = 5, 8, int(tok.load_group_end)
    elif fanout_bucket in {int(tok.fanout_9_16), int(tok.fanout_17_plus)}:
        min_loads, max_loads, end_token = 8, 8, int(tok.load_preview_end)
    else:
        min_loads, max_loads, end_token = 1, 8, int(tok.load_group_end)
    return {
        "load_begin_pos": load_begin_pos,
        "load_end_pos": load_end_pos,
        "load_count": int(load_count),
        "min_loads": int(min_loads),
        "max_loads": int(max_loads),
        "end_token": int(end_token),
    }


def _apply_source_net_close_bias(scores: torch.Tensor, generated: list[int], tok, close_bias: float) -> torch.Tensor:
    if float(close_bias) <= 0.0:
        return scores
    if _section_state(generated, tok).get("current_section") != "SOURCE_NET":
        return scores
    bias = float(close_bias)
    rec = _current_source_net_record(generated, tok)
    expected = int(getattr(tok, "source_net_v5_expected_source_net_records", -1) or -1)
    completed = sum(1 for t in generated if int(t) == int(tok.source_net_end))
    if rec:
        if int(rec[-1]) in {int(tok.load_group_end), int(tok.load_preview_end)} and torch.isfinite(scores[0, int(tok.source_net_end)]):
            scores[0, int(tok.source_net_end)] += bias
        else:
            plan = _source_net_plan_from_record(rec, tok)
            if plan["load_begin_pos"] is not None and plan["load_end_pos"] is None:
                rel = len(rec) - int(plan["load_begin_pos"]) - 1
                if rel % 6 == 0 and int(plan["load_count"]) >= int(plan["min_loads"]):
                    end_tok = int(plan["end_token"])
                    if 0 <= end_tok < scores.shape[-1] and torch.isfinite(scores[0, end_tok]):
                        scores[0, end_tok] += bias
    elif expected >= 0 and completed >= expected and generated and int(generated[-1]) == int(tok.source_net_end):
        if torch.isfinite(scores[0, int(tok.source_net_section_end)]):
            scores[0, int(tok.source_net_section_end)] += bias
    elif generated and int(generated[-1]) == int(tok.source_net_section_end):
        if torch.isfinite(scores[0, int(tok.eos)]):
            scores[0, int(tok.eos)] += bias
    return scores


def _bounded_generate(
    model,
    prefix: list[int],
    *,
    max_new_tokens: int,
    timeout_sec: int,
    top_k: int,
    top_p: float,
    temperature: float,
    use_grammar: bool,
    dump_every_n_tokens: int,
    profile_generation: bool,
    section_token_budget: bool = False,
    force_close_sections: bool = False,
    stop_after_token: int | None = None,
    source_net_optional: bool = False,
    source_net_min_completion_ratio: float = 0.95,
    source_net_record_budget: int = 0,
    source_net_force_close_diagnostic: bool = False,
    v6_constraint_context: dict | None = None,
    v6_constraint_level: int = 0,
    v6_forbid_duplicate_source_group: bool = False,
    v6_level3_eos_gating: bool = False,
    v6_exact_load_group_completion: bool = True,
    v6_close_exhausted_source_group: bool = True,
    source_net_close_bias: float = 0.0,
) -> dict:
    device = model.device
    tok = model.tokenizer
    generated = [int(x) for x in prefix]
    started = time.time()
    past = None
    dump = []
    stopped_reason = "max_new_tokens"
    grammar_anomalies = []
    budget_warnings = []
    forced_closes = Counter()
    last_allowed = {}
    v6_trace = None
    v6_runtime = None
    if int(v6_constraint_level) > 0:
        enabled = ["legal_cell", "legal_pin", "legal_demand"]
        if int(v6_constraint_level) >= 2:
            enabled += ["no_duplicate_demand", "no_duplicate_sink", "sink_demand_source_consistency"]
            if bool(v6_forbid_duplicate_source_group):
                enabled.append("no_duplicate_source_group")
        if int(v6_constraint_level) >= 3 and bool(v6_level3_eos_gating):
            enabled.append("coverage_eos_gating")
        v6_trace = V6PracticalTrace(
            constraint_level=int(v6_constraint_level),
            enabled_masks=enabled,
            level3_coverage_eos_gating_enabled=bool(int(v6_constraint_level) >= 3 and v6_level3_eos_gating),
            level3_requested_but_inactive=bool(int(v6_constraint_level) >= 3 and not v6_level3_eos_gating),
        )
        if v6_constraint_context is not None:
            v6_runtime = V6PracticalRuntime(tok, v6_constraint_context)
    expected_demand_budget = int(getattr(tok, "source_net_v5_expected_demand_records", -1) or -1)
    expected_sink_budget = int(getattr(tok, "source_net_v5_expected_sink_demand_records", -1) or -1)
    expected_cell_budget = int(getattr(tok, "source_net_v5_expected_cell_records", -1) or -1)
    expected_source_budget = int(getattr(tok, "source_net_v5_expected_source_net_records", -1) or -1)
    budgets = {
        "COND": 16,
        "DEMAND": max(128, expected_demand_budget * 10 + 32) if expected_demand_budget >= 0 else 2048,
        "SOURCE_BUDGET": 64,
        "CELL": max(128, expected_cell_budget * 9 + 32) if expected_cell_budget >= 0 else 1536,
        "SINK_DEMAND": max(128, expected_sink_budget * 16 + 32) if expected_sink_budget >= 0 else 2048,
        "SOURCE_NET": max(128, expected_source_budget * 24 + 32) if expected_source_budget >= 0 else 1024,
    }
    end_tokens = {
        "COND": int(tok.cond_section_end),
        "DEMAND": int(tok.demand_section_end),
        "SOURCE_BUDGET": int(tok.source_budget_section_end),
        "CELL": int(tok.cell_section_end),
        "SINK_DEMAND": int(tok.sink_demand_section_end),
        "SOURCE_NET": int(tok.source_net_section_end),
    }
    with torch.no_grad():
        grammar_processor = SourceNetV5Grammar(tok, 1, device, mask_mode="source_net_v5_light") if use_grammar else None
        for step in range(int(max_new_tokens)):
            if timeout_sec and time.time() - started > float(timeout_sec):
                stopped_reason = "timeout"
                break
            pre_state = _section_state(generated, tok)
            current_section = pre_state["current_section"]
            if bool(source_net_optional) and current_section == "SOURCE_NET":
                expected_source = int(getattr(tok, "source_net_v5_expected_source_net_records", -1) or -1)
                generated_source = int(pre_state.get("record_counts", {}).get("source_net_records", 0))
                ratio = float(generated_source) / float(max(1, expected_source)) if expected_source >= 0 else 0.0
                budget_hit = bool(source_net_record_budget and generated_source >= int(source_net_record_budget))
                ratio_hit = bool(expected_source >= 0 and ratio >= float(source_net_min_completion_ratio))
                if bool(source_net_force_close_diagnostic) and (budget_hit or ratio_hit):
                    # Close only at a record boundary.  This is diagnostic-only:
                    # strict decode remains false when source_net_force_closed is set.
                    if generated and int(generated[-1]) == int(tok.source_net_end):
                        generated.extend([int(tok.source_net_section_end), int(tok.eos)])
                        forced_closes["SOURCE_NET"] += 1
                        stopped_reason = "source_net_optional_force_close"
                        break
            if section_token_budget and current_section in budgets:
                used = int(pre_state["section_token_counts"].get(current_section, 0))
                if used > int(budgets[current_section]):
                    warning = {
                        "step": step,
                        "section": current_section,
                        "used": used,
                        "budget": int(budgets[current_section]),
                    }
                    if not budget_warnings or budget_warnings[-1] != warning:
                        budget_warnings.append(warning)
                    if force_close_sections:
                        generated.append(int(end_tokens[current_section]))
                        forced_closes[current_section] += 1
                        past = None
                        continue
            if past is None:
                inp = torch.tensor(generated, dtype=torch.long, device=device).reshape(1, -1)
            else:
                inp = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
            out = model.model.model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            scores = out.logits[:, -1, :].float()
            if use_grammar:
                scores = grammar_processor(torch.tensor(generated, dtype=torch.long, device=device).reshape(1, -1), scores)
            if v6_trace is not None:
                scores = _apply_v6_practical_constraint_mask(
                    scores,
                    generated,
                    tok,
                    v6_constraint_context,
                    v6_trace,
                    constraint_level=int(v6_constraint_level),
                    forbid_duplicate_source_group=bool(v6_forbid_duplicate_source_group),
                    enable_level3_eos_gating=bool(v6_level3_eos_gating),
                    exact_load_group_completion=bool(v6_exact_load_group_completion),
                    close_exhausted_source_group=bool(v6_close_exhausted_source_group),
                    runtime=v6_runtime,
                )
            scores = _apply_source_net_close_bias(scores, generated, tok, float(source_net_close_bias))
            if not bool(torch.isfinite(scores[0]).any().item()):
                stopped_reason = "generation_dead_end" if v6_trace is not None else "mask_all_invalid"
                if v6_trace is not None:
                    v6_trace.generation_dead_end = True
                    v6_trace.dead_end_step = int(step)
                    v6_trace.dead_end_state = _section_state(generated, tok)
                grammar_anomalies.append({"step": step, "reason": stopped_reason, "state": _section_state(generated, tok)})
                break
            logits = scores[0] / max(float(temperature), 1.0e-6)
            logits = _top_k_top_p_filter(logits, int(top_k), float(top_p))
            if not bool(torch.isfinite(logits).any().item()):
                stopped_reason = "no_finite_logits_after_filter"
                grammar_anomalies.append({"step": step, "reason": "no_finite_logits_after_filter", "allowed": allowed_summary})
                break
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            generated.append(next_id)
            if v6_runtime is not None:
                v6_runtime.consume(next_id)
            if stop_after_token is not None and next_id == int(stop_after_token):
                stopped_reason = "stop_after_token"
                break
            if next_id == int(tok.eos):
                stopped_reason = "eos"
                break
            if dump_every_n_tokens and ((step + 1) % int(dump_every_n_tokens) == 0):
                state = _section_state(generated, tok)
                allowed_summary = _grammar_allowed_summary(generated, tok, device, use_grammar)
                last_allowed = allowed_summary
                dump.append({
                    "new_tokens": step + 1,
                    "total_tokens": len(generated),
                    "elapsed_sec": time.time() - started,
                    "current_section": state["current_section"],
                    "section_token_counts": state["section_token_counts"],
                    "record_counts": state["record_counts"],
                    "last_token": _tok_name(tok, next_id),
                    "eos_allowed": allowed_summary.get("eos_allowed"),
                    "section_end_allowed": allowed_summary.get("section_end_allowed"),
                })
    elapsed = time.time() - started
    state = _section_state(generated, tok)
    coverage_status = _coverage_decode_status(generated, tok)
    v6_trace_dict = _finalize_v6_trace(v6_trace, generated, tok, v6_constraint_context, stopped_reason, runtime=v6_runtime)
    allowed = _grammar_allowed_summary(generated, tok, device, use_grammar)
    expected_demand = int(getattr(tok, "source_net_v5_expected_demand_records", -1) or -1)
    expected_sink = int(getattr(tok, "source_net_v5_expected_sink_demand_records", -1) or -1)
    expected_cell = int(getattr(tok, "source_net_v5_expected_cell_records", -1) or -1)
    expected_source_net = int(getattr(tok, "source_net_v5_expected_source_net_records", -1) or -1)
    generated_demand = int(state["record_counts"].get("demand_records", 0))
    generated_sink = int(state["record_counts"].get("sink_demand_records_estimate", 0))
    generated_cell = int(state["record_counts"].get("cell_records", 0))
    generated_source_net = int(state["record_counts"].get("source_net_records", 0))
    budget_exceeded_count = 0
    if expected_demand >= 0 and generated_demand > expected_demand:
        budget_exceeded_count += generated_demand - expected_demand
    if expected_sink >= 0 and generated_sink > expected_sink:
        budget_exceeded_count += generated_sink - expected_sink
    if expected_cell >= 0 and generated_cell > expected_cell:
        budget_exceeded_count += generated_cell - expected_cell
    if expected_source_net >= 0 and generated_source_net > expected_source_net:
        budget_exceeded_count += generated_source_net - expected_source_net
    if stopped_reason == "eos":
        close_reason = "normal_close"
    elif forced_closes:
        close_reason = "force_close"
    elif budget_exceeded_count:
        close_reason = "budget_exceeded"
    else:
        close_reason = stopped_reason
    repeated_sections = []
    current_section = str(state.get("current_section", "NONE"))
    if current_section in budgets and int(state["section_token_counts"].get(current_section, 0)) > int(budgets[current_section]):
        repeated_sections.append(current_section)
    return {
        "generated_ids": generated,
        "generated_token_count": len(generated),
        "new_token_count": max(0, len(generated) - len(prefix)),
        "elapsed_sec": elapsed,
        "tokens_per_sec": max(0, len(generated) - len(prefix)) / max(elapsed, 1.0e-9),
        "stopped_reason": stopped_reason,
        "last_50_tokens": [_tok_name(tok, x) for x in generated[-50:]],
        "state": state,
        "expected_demand_records": expected_demand,
        "generated_demand_records": generated_demand,
        "expected_sink_demand_records": expected_sink,
        "generated_sink_demand_records": generated_sink,
        "expected_cell_records": expected_cell,
        "generated_cell_records": generated_cell,
        "expected_source_net_records": expected_source_net,
        "generated_source_net_records": generated_source_net,
        "section_close_reason": close_reason,
        "budget_exceeded_count": int(budget_exceeded_count),
        "section_loop": bool(repeated_sections),
        "section_loop_sections": repeated_sections,
        "unable_to_close_section": bool(stopped_reason in {"timeout", "max_new_tokens"} and state["open_sections"]),
        "grammar_allowed": allowed,
        "last_step_allowed": last_allowed,
        "grammar_anomalies": grammar_anomalies,
        "budget_warnings": budget_warnings,
        "forced_closes": dict(forced_closes),
        "source_net_force_closed": bool(forced_closes.get("SOURCE_NET", 0)),
        "coverage_decode_status": coverage_status,
        "constrained_decode_trace": v6_trace_dict,
        "trace": dump,
        "profile_generation": bool(profile_generation),
    }


def _prefix_until(tokens: list[int], end_token: int) -> list[int]:
    for i, tok in enumerate(tokens):
        if int(tok) == int(end_token):
            return tokens[: i + 1]
    return list(tokens)


def _section_body(tokens: list[int], begin: int, end: int) -> list[int]:
    try:
        s = tokens.index(int(begin)) + 1
        e = tokens.index(int(end), s)
        return [int(x) for x in tokens[s:e]]
    except ValueError:
        return []


def _record_chunks(body: list[int], begin: int, end: int) -> list[list[int]]:
    records = []
    cur = None
    for tid in body:
        tid = int(tid)
        if tid == int(begin):
            cur = [tid]
        elif cur is not None:
            cur.append(tid)
            if tid == int(end):
                records.append(cur)
                cur = None
    return records


def _first_after_marker(rec: list[int], marker: int) -> int | None:
    marker = int(marker)
    for i, tid in enumerate(rec[:-1]):
        if int(tid) == marker:
            return int(rec[i + 1])
    return None


def _gate_idx_from_tok(tok, tid: int) -> int | None:
    tid = int(tid)
    if tok.max_num_nodes is None:
        return None
    if int(tok.idx_offset) <= tid < int(tok.idx_offset + tok.max_num_nodes):
        return int(tid - tok.idx_offset)
    return None


def _src_idx_from_tok(tok, tid: int) -> int | None:
    tid = int(tid)
    if tok.max_num_nodes is None:
        return None
    start = int(tok.idx_offset + tok.max_num_nodes)
    end = int(tok.idx_offset + 2 * tok.max_num_nodes)
    if start <= tid < end:
        return int(tid - start)
    return None


def _demand_idx_from_tok(tok, tid: int) -> int | None:
    tid = int(tid)
    if tok.max_num_nodes is None:
        return None
    start = int(tok.idx_offset + 2 * tok.max_num_nodes)
    end = int(tok.pin_offset)
    if start <= tid < end:
        return int(tid - start)
    return None


def _token_set_mask(scores: torch.Tensor, ids: set[int] | list[int]) -> torch.Tensor:
    mask = torch.zeros((scores.numel(),), dtype=torch.bool, device=scores.device)
    good = [int(x) for x in ids if 0 <= int(x) < int(scores.numel())]
    if good:
        mask[torch.tensor(good, dtype=torch.long, device=scores.device)] = True
    return mask


def _current_source_net_record(tokens: list[int], tok) -> list[int]:
    in_section = False
    current = []
    for tid in tokens:
        tid = int(tid)
        if tid == int(tok.source_net_section_begin):
            in_section = True
            current = []
            continue
        if tid == int(tok.source_net_section_end):
            in_section = False
            current = []
            continue
        if not in_section:
            continue
        if tid == int(tok.source_net_begin):
            current = [tid]
        elif current:
            current.append(tid)
            if tid == int(tok.source_net_end):
                current = []
    return current


def _source_identity_from_record(rec: list[int], tok) -> tuple[int, int, int] | None:
    kind = _first_after_marker(rec, int(tok.source_kind))
    cell = _first_after_marker(rec, int(tok.source_cell))
    pin = _first_after_marker(rec, int(tok.source_pin))
    if kind is None or cell is None or pin is None:
        return None
    return (int(kind), int(cell), int(pin))


def _completed_source_net_records(tokens: list[int], tok) -> list[list[int]]:
    body = _section_body(tokens, int(tok.source_net_section_begin), int(tok.source_net_section_end))
    if not body:
        # For in-progress SOURCE_NET generation, the section end is absent.
        try:
            s = tokens.index(int(tok.source_net_section_begin)) + 1
            body = [int(x) for x in tokens[s:]]
        except ValueError:
            body = []
    return _record_chunks(body, int(tok.source_net_begin), int(tok.source_net_end))


def _build_v6_practical_context(tok, true_tokens: list[int]) -> dict:
    demand_body = _section_body(true_tokens, int(tok.demand_section_begin), int(tok.demand_section_end))
    cell_body = _section_body(true_tokens, int(tok.cell_section_begin), int(tok.cell_section_end))
    sink_body = _section_body(true_tokens, int(tok.sink_demand_section_begin), int(tok.sink_demand_section_end))
    demand_records = _record_chunks(demand_body, int(tok.demand_begin), int(tok.demand_end))
    sink_records = _record_chunks(sink_body, int(tok.demand_begin), int(tok.demand_end))
    cell_records = _cell_chunks(cell_body, tok)

    cells: dict[int, dict] = {}
    for rec in cell_records:
        if len(rec) < 3:
            continue
        gate_idx = _gate_idx_from_tok(tok, int(rec[1]))
        if gate_idx is None:
            continue
        cell_label = tok._cell_from_tok(int(rec[2]))
        cell_name = tok._cell_name(int(cell_label))
        inputs, outputs = tok._spec_pins(cell_name)
        cells[int(gate_idx)] = {
            "gate_token": int(rec[1]),
            "source_token": int(tok._tok_src_idx(int(gate_idx))),
            "cell_type_token": int(rec[2]),
            "cell_label": int(cell_label),
            "cell_name": str(cell_name),
            "input_pin_tokens": {int(tok._tok_pin(p)) for p in inputs},
            "output_pin_tokens": {int(tok._tok_pin(p)) for p in outputs},
            "input_pins": sorted(str(p) for p in inputs),
            "output_pins": sorted(str(p) for p in outputs),
        }

    demands: dict[int, dict] = {}
    endpoint_to_demands: dict[tuple[int, int], set[int]] = defaultdict(set)
    for rec in demand_records:
        did_tok = _first_after_marker(rec, int(tok.demand_id))
        load_tok = _first_after_marker(rec, int(tok.load_cell))
        pin_tok = _first_after_marker(rec, int(tok.load_pin))
        did = _demand_idx_from_tok(tok, did_tok) if did_tok is not None else None
        load = _gate_idx_from_tok(tok, load_tok) if load_tok is not None else None
        if did is None or load is None or pin_tok is None:
            continue
        demands[int(did)] = {
            "demand_token": int(did_tok),
            "load_cell_idx": int(load),
            "load_cell_token": int(load_tok),
            "load_pin_token": int(pin_tok),
            "load_pin_name": str(tok._pin_from_tok(int(pin_tok))),
            "kind": "INTERNAL_REQUIRED",
        }
        endpoint_to_demands[(int(load), int(pin_tok))].add(int(did))

    sink_by_demand: dict[int, dict] = {}
    source_candidates: set[tuple[int, int, int]] = set()
    for rec in sink_records:
        did_tok = _first_after_marker(rec, int(tok.demand_id))
        did = _demand_idx_from_tok(tok, did_tok) if did_tok is not None else None
        kind = _first_after_marker(rec, int(tok.source_kind))
        src_cell = _first_after_marker(rec, int(tok.source_cell))
        src_pin = _first_after_marker(rec, int(tok.source_pin))
        load_tok = _first_after_marker(rec, int(tok.load_cell))
        load_pin = _first_after_marker(rec, int(tok.load_pin))
        load = _gate_idx_from_tok(tok, load_tok) if load_tok is not None else None
        if did is not None:
            sink_by_demand[int(did)] = {
                "load_cell_idx": load,
                "load_cell_token": load_tok,
                "load_pin_token": load_pin,
                "source_kind_token": kind,
                "source_cell_token": src_cell,
                "source_pin_token": src_pin,
                "source_identity": (int(kind), int(src_cell), int(src_pin)) if kind is not None and src_cell is not None and src_pin is not None else None,
            }
        if kind is not None and src_cell is not None and src_pin is not None:
            source_candidates.add((int(kind), int(src_cell), int(src_pin)))

    for cell in cells.values():
        for pin_tok in cell["output_pin_tokens"]:
            source_candidates.add((int(tok.cell_output), int(cell["source_token"]), int(pin_tok)))

    source_kinds = {s[0] for s in source_candidates}
    source_cells_by_kind: dict[int, set[int]] = defaultdict(set)
    source_pins_by_kind_cell: dict[tuple[int, int], set[int]] = defaultdict(set)
    for kind, src_cell, src_pin in source_candidates:
        source_cells_by_kind[int(kind)].add(int(src_cell))
        source_pins_by_kind_cell[(int(kind), int(src_cell))].add(int(src_pin))

    # Unlike the broad legal source table above, this table contains only
    # source identities actually declared by the complete SINK_DEMAND view.
    # Level 2 uses it to keep the generated SOURCE_NET as a consistent
    # regrouping of that view rather than an arbitrary legal rewiring.
    source_demands_by_identity: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    connected_source_cells_by_kind: dict[int, set[int]] = defaultdict(set)
    connected_source_pins_by_kind_cell: dict[tuple[int, int], set[int]] = defaultdict(set)
    for did, sink in sink_by_demand.items():
        sid = sink.get("source_identity")
        if sid is None:
            continue
        source_demands_by_identity[sid].add(int(did))
        connected_source_cells_by_kind[int(sid[0])].add(int(sid[1]))
        connected_source_pins_by_kind_cell[(int(sid[0]), int(sid[1]))].add(int(sid[2]))
    source_fanout_bucket_by_identity = {
        sid: int(tok._fanout_tok(len(dids)))
        for sid, dids in source_demands_by_identity.items()
    }

    legal_load_cells = {int(v["gate_token"]) for v in cells.values() if v["input_pin_tokens"]}
    demand_tokens = {int(v["demand_token"]) for v in demands.values()}
    # The SOURCE_NET serializer intentionally truncates high-fanout load
    # lists to a preview.  Keep its reference scope explicitly so report
    # metrics never confuse preview membership with all-demand coverage.
    reference_source_records = _completed_source_net_records(true_tokens, tok)
    reference_explicit_demand_ids: set[int] = set()
    for rec in reference_source_records:
        i = 0
        while i < len(rec):
            if int(rec[i]) == int(tok.load_cell) and i + 5 < len(rec):
                did_tok = int(rec[i + 5]) if int(rec[i + 4]) == int(tok.demand_id) else None
                did = _demand_idx_from_tok(tok, did_tok) if did_tok is not None else None
                if did is not None:
                    reference_explicit_demand_ids.add(int(did))
                i += 6
                continue
            i += 1

    return {
        "cells": cells,
        "demands": demands,
        "endpoint_to_demands": endpoint_to_demands,
        "sink_by_demand": sink_by_demand,
        "source_candidates": source_candidates,
        "source_kinds": source_kinds,
        "source_cells_by_kind": source_cells_by_kind,
        "source_pins_by_kind_cell": source_pins_by_kind_cell,
        "source_demands_by_identity": source_demands_by_identity,
        "connected_source_cells_by_kind": connected_source_cells_by_kind,
        "connected_source_pins_by_kind_cell": connected_source_pins_by_kind_cell,
        "source_fanout_bucket_by_identity": source_fanout_bucket_by_identity,
        "legal_load_cells": legal_load_cells,
        "demand_tokens": demand_tokens,
        "reference_explicit_demand_ids": reference_explicit_demand_ids,
    }


def _v6_runtime_state(tokens: list[int], tok, ctx: dict) -> dict:
    used_demands = []
    used_endpoints = []
    source_identities = []
    invalid_source_pin = 0
    invalid_load_pin = 0
    for rec in _completed_source_net_records(tokens, tok):
        sid = _source_identity_from_record(rec, tok)
        if sid is not None:
            source_identities.append(sid)
            kind, src_cell, src_pin = sid
            if int(kind) == int(tok.cell_output):
                src_idx = _src_idx_from_tok(tok, int(src_cell))
                cell = ctx["cells"].get(src_idx)
                if cell is None or int(src_pin) not in cell["output_pin_tokens"]:
                    invalid_source_pin += 1
            elif sid not in ctx["source_candidates"]:
                invalid_source_pin += 1
        i = 0
        while i < len(rec):
            if int(rec[i]) == int(tok.load_cell) and i + 5 < len(rec):
                load_tok = int(rec[i + 1])
                load_idx = _gate_idx_from_tok(tok, load_tok)
                pin_tok = int(rec[i + 3]) if int(rec[i + 2]) == int(tok.load_pin) else None
                did_tok = int(rec[i + 5]) if pin_tok is not None and int(rec[i + 4]) == int(tok.demand_id) else None
                if load_idx is not None and pin_tok is not None:
                    used_endpoints.append((int(load_idx), int(pin_tok)))
                    cell = ctx["cells"].get(int(load_idx))
                    if cell is None or int(pin_tok) not in cell["input_pin_tokens"]:
                        invalid_load_pin += 1
                if did_tok is not None:
                    did = _demand_idx_from_tok(tok, did_tok)
                    if did is not None:
                        used_demands.append(int(did))
                i += 6
                continue
            i += 1
    current = _current_source_net_record(tokens, tok)
    return {
        "used_demands": set(used_demands),
        "used_endpoints": set(used_endpoints),
        "used_source_identities": set(source_identities),
        "duplicate_sink_count": sum(v - 1 for v in Counter(used_endpoints).values() if v > 1),
        "duplicate_source_group_count": sum(v - 1 for v in Counter(source_identities).values() if v > 1),
        "invalid_source_pin_count": int(invalid_source_pin),
        "invalid_load_pin_count": int(invalid_load_pin),
        "current_record": current,
        "current_source_identity": _source_identity_from_record(current, tok),
    }


def _v6_source_net_phase(rec: list[int], tok) -> tuple[str, dict]:
    if not rec:
        return "outside", {}
    pos = len(rec)
    if pos == 2:
        return "source_kind_value", {}
    if pos == 4:
        kind = _first_after_marker(rec, int(tok.source_kind))
        return "source_cell_value", {"kind": kind}
    if pos == 6:
        kind = _first_after_marker(rec, int(tok.source_kind))
        src_cell = _first_after_marker(rec, int(tok.source_cell))
        return "source_cell_type_value", {"kind": kind, "source_cell": src_cell}
    if pos == 8:
        kind = _first_after_marker(rec, int(tok.source_kind))
        src_cell = _first_after_marker(rec, int(tok.source_cell))
        return "source_pin_value", {"kind": kind, "source_cell": src_cell}
    if pos == 10:
        return "fanout_bucket_value", {"source_identity": _source_identity_from_record(rec, tok)}
    if int(tok.load_group_begin) in rec or int(tok.load_preview_begin) in rec:
        try:
            begin_pos = max(rec.index(int(tok.load_group_begin)) if int(tok.load_group_begin) in rec else -1,
                            rec.index(int(tok.load_preview_begin)) if int(tok.load_preview_begin) in rec else -1)
        except ValueError:
            begin_pos = -1
        if begin_pos >= 0 and int(tok.load_group_end) not in rec and int(tok.load_preview_end) not in rec:
            rel = pos - begin_pos - 1
            phase = rel % 6
            if phase == 1:
                return "load_cell_value", {}
            if phase == 3:
                # Tail while predicting LOAD_PIN value:
                #   LOAD_CELL, <gate>, LOAD_PIN
                load_cell_tok = rec[-2] if len(rec) >= 2 else None
                return "load_pin_value", {"load_cell_token": load_cell_tok}
            if phase == 5:
                # Tail while predicting DEMAND_ID value:
                #   LOAD_CELL, <gate>, LOAD_PIN, <pin>, DEMAND_ID
                load_cell_tok = rec[-4] if len(rec) >= 4 else None
                load_pin_tok = rec[-2] if len(rec) >= 2 else None
                return "demand_id_value", {"load_cell_token": load_cell_tok, "load_pin_token": load_pin_tok}
    return "other", {}


def _apply_v6_practical_constraint_mask(
    scores: torch.Tensor,
    generated: list[int],
    tok,
    ctx: dict | None,
    trace: V6PracticalTrace | None,
    *,
    constraint_level: int,
    forbid_duplicate_source_group: bool,
    enable_level3_eos_gating: bool,
    exact_load_group_completion: bool,
    close_exhausted_source_group: bool,
    runtime: V6PracticalRuntime | None = None,
) -> torch.Tensor:
    if ctx is None or trace is None or int(constraint_level) <= 0:
        return scores
    if _section_state(generated, tok).get("current_section") != "SOURCE_NET":
        return scores
    before = torch.isfinite(scores[0])
    state = runtime.snapshot() if runtime is not None else _v6_runtime_state(generated, tok, ctx)
    rec = state["current_record"]
    phase, info = _v6_source_net_phase(rec, tok)

    # The grammar knows only bucket ranges (for example 2..4).  Once a
    # prefix-resolved source has emitted its exact compact load count, the
    # next legal token is the group terminator, not another LOAD_CELL.
    source_identity = state.get("current_source_identity")
    in_load_group = int(tok.load_group_begin) in rec or int(tok.load_preview_begin) in rec
    group_closed = int(tok.load_group_end) in rec or int(tok.load_preview_end) in rec
    if int(constraint_level) >= 2 and source_identity is not None and in_load_group and not group_closed:
        source_demands = set(ctx["source_demands_by_identity"].get(source_identity, set()))
        source_demand_count = len(source_demands)
        explicit_target = min(8, source_demand_count)
        emitted_loads = sum(1 for tid in rec if int(tid) == int(tok.load_cell))
        last_load = max((i for i, tid in enumerate(rec) if int(tid) == int(tok.load_cell)), default=-1)
        last_load_complete = bool(
            last_load >= 0
            and last_load + 5 < len(rec)
            and int(rec[last_load + 2]) == int(tok.load_pin)
            and int(rec[last_load + 4]) == int(tok.demand_id)
        )
        remaining_for_source = source_demands - set(state["used_demands"])
        exhausted_source = bool(last_load_complete and not remaining_for_source)
        # A LOAD_CELL entry is a six-token tuple.  The generic phase parser
        # is between marker/value pairs while a tuple is in flight, so only
        # terminate after the final tuple is structurally complete.
        should_close_exact = bool(exact_load_group_completion) and explicit_target > 0 and emitted_loads >= explicit_target and last_load_complete
        # Even in raw continuation mode, do not let sampling step into an
        # empty legal set.  Once this source has no remaining demand, another
        # LOAD_CELL cannot become legal; the only valid continuation is the
        # group terminator.  This closes an exhausted group, not the section.
        should_close_exhausted = bool(close_exhausted_source_group) and exhausted_source
        if should_close_exact or should_close_exhausted:
            end_token = int(tok.load_preview_end) if source_demand_count > 8 else int(tok.load_group_end)
            mask = _token_set_mask(scores[0], {end_token})
            removed = int((before & ~mask).sum().item())
            scores[0] = torch.where(mask, scores[0], torch.full_like(scores[0], float("-inf")))
            # SourceNetV5Grammar may only permit a generic bucket-range
            # continuation here.  The prefix-derived exact count is stricter,
            # so resurrect its sole legal structural terminator.
            scores[0, end_token] = 0.0
            if should_close_exact:
                trace.forced_load_group_end_count += 1
            else:
                trace.exhausted_source_group_close_count += 1
            if len(trace.events) < 200:
                trace.events.append({
                    "step_total_tokens": len(generated),
                    "phase": "load_group_exact_completion" if should_close_exact else "load_group_exhausted_source_close",
                    "allowed_count": 1,
                    "removed_finite_count": int(removed),
                })
            return scores

    allowed: set[int] | None = None
    category = ""

    if phase == "source_kind_value":
        allowed = (
            {int(sid[0]) for sid in ctx["source_demands_by_identity"]}
            if int(constraint_level) >= 2
            else set(ctx["source_kinds"])
        )
        category = "invalid_cell"
    elif phase == "source_cell_value":
        kind = info.get("kind")
        if kind is not None:
            allowed = set(
                (ctx["connected_source_cells_by_kind"] if int(constraint_level) >= 2 else ctx["source_cells_by_kind"])
                .get(int(kind), set())
            )
            if int(constraint_level) >= 2 and bool(forbid_duplicate_source_group):
                filtered = set()
                for cell_tok in allowed:
                    pins = set(ctx["connected_source_pins_by_kind_cell"].get((int(kind), int(cell_tok)), set()))
                    if any((int(kind), int(cell_tok), int(pin)) not in state["used_source_identities"] for pin in pins):
                        filtered.add(int(cell_tok))
                removed = len(allowed - filtered)
                trace.masked_duplicate_source_tokens += int(max(0, removed))
                allowed = filtered
            category = "invalid_cell"
    elif phase == "source_cell_type_value":
        kind = info.get("kind")
        src_cell = info.get("source_cell")
        if kind is not None and src_cell is not None and int(kind) == int(tok.cell_output):
            src_idx = _src_idx_from_tok(tok, int(src_cell))
            cell = ctx["cells"].get(src_idx)
            if cell is not None:
                allowed = {int(cell["cell_type_token"])}
                category = "invalid_cell"
        elif kind is not None and src_cell is not None:
            # Non-cell sources in V5 use their source kind token as type.
            allowed = {int(kind), int(tok.boundary_source), int(tok.stub_source), int(tok.pi_const_source), int(tok.input_pool_source), int(tok.private_out_source), int(tok.unknown)}
            category = "invalid_cell"
    elif phase == "fanout_bucket_value":
        source_identity = info.get("source_identity")
        if source_identity is not None and int(constraint_level) >= 2:
            allowed = {int(ctx["source_fanout_bucket_by_identity"].get(source_identity, tok.fanout_1))}
            category = "invalid_cell"
    elif phase == "source_pin_value":
        kind = info.get("kind")
        src_cell = info.get("source_cell")
        if kind is not None and src_cell is not None:
            pin_table = ctx["connected_source_pins_by_kind_cell"] if int(constraint_level) >= 2 else ctx["source_pins_by_kind_cell"]
            allowed = set(pin_table.get((int(kind), int(src_cell)), set()))
            if int(constraint_level) < 2 and int(kind) == int(tok.cell_output):
                src_idx = _src_idx_from_tok(tok, int(src_cell))
                cell = ctx["cells"].get(src_idx)
                if cell is not None:
                    allowed |= set(cell["output_pin_tokens"])
            if int(constraint_level) >= 2 and bool(forbid_duplicate_source_group):
                filtered = {int(pin) for pin in allowed if (int(kind), int(src_cell), int(pin)) not in state["used_source_identities"]}
                trace.masked_duplicate_source_tokens += int(max(0, len(allowed - filtered)))
                allowed = filtered
            category = "invalid_pin"
    elif phase == "load_cell_value":
        allowed = set(ctx["legal_load_cells"])
        if int(constraint_level) >= 2:
            remaining = {did for did in ctx["demands"] if did not in state["used_demands"]}
            source_identity = state.get("current_source_identity")
            if source_identity is not None:
                remaining &= set(ctx["source_demands_by_identity"].get(source_identity, set()))
            allowed = {int(ctx["demands"][did]["load_cell_token"]) for did in remaining}
        category = "invalid_cell"
    elif phase == "load_pin_value":
        load_idx = _gate_idx_from_tok(tok, info.get("load_cell_token")) if info.get("load_cell_token") is not None else None
        if load_idx is not None:
            cell = ctx["cells"].get(int(load_idx))
            allowed = set(cell["input_pin_tokens"]) if cell is not None else set()
            endpoint_pins = {pin for (g, pin), dids in ctx["endpoint_to_demands"].items() if int(g) == int(load_idx) and dids}
            if endpoint_pins:
                allowed &= set(endpoint_pins) if allowed else set(endpoint_pins)
            if int(constraint_level) >= 2:
                filtered = set()
                source_identity = state.get("current_source_identity")
                source_demands = set(ctx["source_demands_by_identity"].get(source_identity, set())) if source_identity is not None else set()
                for pin in allowed:
                    dids = set(ctx["endpoint_to_demands"].get((int(load_idx), int(pin)), set()))
                    if source_identity is not None:
                        dids &= source_demands
                    if dids - state["used_demands"] and (int(load_idx), int(pin)) not in state["used_endpoints"]:
                        filtered.add(int(pin))
                trace.masked_duplicate_sink_tokens += int(max(0, len(allowed - filtered)))
                allowed = filtered
            category = "invalid_pin"
    elif phase == "demand_id_value":
        load_idx = _gate_idx_from_tok(tok, info.get("load_cell_token")) if info.get("load_cell_token") is not None else None
        pin_tok = info.get("load_pin_token")
        if load_idx is not None and pin_tok is not None:
            dids = set(ctx["endpoint_to_demands"].get((int(load_idx), int(pin_tok)), set()))
            if int(constraint_level) >= 2:
                dids -= state["used_demands"]
                source_identity = state.get("current_source_identity")
                if source_identity is not None:
                    dids &= set(ctx["source_demands_by_identity"].get(source_identity, set()))
            allowed = {int(ctx["demands"][did]["demand_token"]) for did in dids if did in ctx["demands"]}
            category = "demand"

    if allowed is not None:
        mask = _token_set_mask(scores[0], allowed)
        removed = int((before & ~mask).sum().item())
        if category == "invalid_cell":
            trace.masked_invalid_cell_tokens += removed
        elif category == "invalid_pin":
            trace.masked_invalid_pin_tokens += removed
        elif category == "demand":
            trace.masked_invalid_demand_tokens += removed
        scores[0] = torch.where(mask, scores[0], torch.full_like(scores[0], float("-inf")))
        if len(trace.events) < 200:
            trace.events.append({
                "step_total_tokens": len(generated),
                "phase": phase,
                "allowed_count": int(len(allowed)),
                "removed_finite_count": int(removed),
            })

    if int(constraint_level) >= 3 and bool(enable_level3_eos_gating):
        remaining = set(ctx["demands"]) - set(state["used_demands"])
        if remaining and int(tok.source_net_section_end) < scores.shape[-1] and torch.isfinite(scores[0, int(tok.source_net_section_end)]):
            scores[0, int(tok.source_net_section_end)] = float("-inf")
            trace.eos_tokens_blocked_by_remaining_internal_demand += 1
        if remaining and int(tok.eos) < scores.shape[-1] and torch.isfinite(scores[0, int(tok.eos)]):
            scores[0, int(tok.eos)] = float("-inf")
            trace.eos_tokens_blocked_by_remaining_internal_demand += 1
    return scores


def _finalize_v6_trace(
    trace: V6PracticalTrace | None,
    generated: list[int],
    tok,
    ctx: dict | None,
    stopped_reason: str,
    runtime: V6PracticalRuntime | None = None,
) -> dict:
    if trace is None:
        return {}
    state = runtime.snapshot() if runtime is not None else _v6_runtime_state(generated, tok, ctx or {})
    all_demands = set((ctx or {}).get("demands", {}).keys())
    used = set(state.get("used_demands", set()))
    trace.source_net_emitted_demand_count = int(len(used))
    trace.source_net_non_emitted_demand_count = int(len(all_demands - used))
    trace.duplicate_sink_count = int(state.get("duplicate_sink_count", 0))
    trace.duplicate_source_group_count = int(state.get("duplicate_source_group_count", 0))
    trace.invalid_source_pin_count = int(state.get("invalid_source_pin_count", 0))
    trace.invalid_load_pin_count = int(state.get("invalid_load_pin_count", 0))
    trace.source_demand_mismatch_count = int(state.get("source_demand_mismatch_count", 0))
    trace.malformed_load_entry_count = int(state.get("malformed_load_entry_count", 0))
    trace.internal_missing_source = int(len(all_demands - used))
    trace.max_new_tokens_hit = str(stopped_reason) == "max_new_tokens"
    return trace.asdict()


def _cell_chunks(body: list[int], tok) -> list[list[int]]:
    records = []
    cur = None
    for tid in body:
        tid = int(tid)
        if tid == int(tok.cell):
            cur = [tid]
        elif cur is not None:
            cur.append(tid)
            if len(cur) >= 9:
                records.append(cur)
                cur = None
    return records


def _build_hard_reference_payload(tok, true_tokens: list[int]) -> dict:
    demand_body = _section_body(true_tokens, int(tok.demand_section_begin), int(tok.demand_section_end))
    cell_body = _section_body(true_tokens, int(tok.cell_section_begin), int(tok.cell_section_end))
    sink_body = _section_body(true_tokens, int(tok.sink_demand_section_begin), int(tok.sink_demand_section_end))
    source_body = _section_body(true_tokens, int(tok.source_net_section_begin), int(tok.source_net_section_end))
    return {
        "DEMAND": _record_chunks(demand_body, int(tok.demand_begin), int(tok.demand_end)),
        "CELL": _cell_chunks(cell_body, tok),
        "SINK_DEMAND": _record_chunks(sink_body, int(tok.demand_begin), int(tok.demand_end)),
        "SOURCE_NET": _record_chunks(source_body, int(tok.source_net_begin), int(tok.source_net_end)),
    }


def _fanout_tok_for_count(tok, n: int) -> int:
    n = int(n)
    if n <= 1:
        return int(tok.fanout_1)
    if n <= 4:
        return int(tok.fanout_2_4)
    if n <= 8:
        return int(tok.fanout_5_8)
    if n <= 16:
        return int(tok.fanout_9_16)
    return int(tok.fanout_17_plus)


def _apply_source_reuse_budget_to_payload(tok, payload: dict) -> dict:
    if not bool(getattr(tok, "source_net_v5_enable_source_reuse_mask", False)):
        setattr(tok, "last_source_net_v5_source_reuse_mask_stats", {})
        return payload
    boundary_cap = int(getattr(tok, "source_net_v5_boundary_source_fanout_cap", 8) or 8)
    cell_cap = int(getattr(tok, "source_net_v5_cell_output_source_fanout_cap", 10**9) or 10**9)
    allow_boundary_17 = bool(getattr(tok, "source_net_v5_allow_boundary_17plus", False))

    cell_candidates = []
    for rec in payload.get("CELL", []) or []:
        if len(rec) < 3:
            continue
        try:
            gate_tok = int(rec[1])
            cell_tok = int(rec[2])
            dense = int(gate_tok - tok.idx_offset)
            label = int(cell_tok - tok.node_type_offset)
            cell_name = tok._cell_name(label)
            _inputs, outputs = tok._spec_pins(cell_name)
            for pin in sorted(outputs):
                cell_candidates.append((dense, str(pin), int(tok._tok_src_idx(dense)), int(tok._tok_pin(str(pin)))))
        except Exception:
            continue

    counts = Counter()
    source_kind_counts = Counter()
    boundary_masked = 0
    forced_source_kind = 0
    relaxation = 0
    cell_candidate_available = 0
    selected_cell = 0
    selected_boundary = 0
    rewritten = 0
    source_kind_budget_exceeded = 0

    sink_records = [list(r) for r in payload.get("SINK_DEMAND", []) or []]

    def rec_source_sig(rec):
        if len(rec) <= 12:
            return ("UNKNOWN", -1, -1)
        kind_tok = int(rec[8])
        obj_tok = int(rec[10])
        pin_tok = int(rec[12])
        if kind_tok == int(tok.boundary_source):
            return ("BOUNDARY_SOURCE", int(obj_tok), int(pin_tok))
        if kind_tok == int(tok.cell_output):
            return ("CELL_OUTPUT", int(obj_tok), int(pin_tok))
        if kind_tok == int(tok.stub_source):
            return ("STUB_SOURCE", int(obj_tok), int(pin_tok))
        if kind_tok == int(tok.pi_const_source):
            return ("PI_CONST_SOURCE", int(obj_tok), int(pin_tok))
        if kind_tok == int(tok.input_pool_source):
            return ("INPUT_POOL_SOURCE", int(obj_tok), int(pin_tok))
        if kind_tok == int(tok.private_out_source):
            return ("PRIVATE_OUT_SOURCE", int(obj_tok), int(pin_tok))
        return ("UNKNOWN", int(obj_tok), int(pin_tok))

    for rec in sink_records:
        if len(rec) <= 14:
            continue
        sig = rec_source_sig(rec)
        kind = sig[0]
        use_replacement = False
        if kind == "BOUNDARY_SOURCE":
            over_cap = counts[sig] >= max(1, boundary_cap)
            would_17 = counts[sig] >= 16
            use_replacement = over_cap or ((not allow_boundary_17) and would_17)
            if use_replacement:
                boundary_masked += 1
                source_kind_budget_exceeded += 1
        if use_replacement:
            available = [
                cand for cand in cell_candidates
                if counts[("CELL_OUTPUT", int(cand[2]), int(cand[3]))] < cell_cap
            ]
            if available:
                cell_candidate_available += 1
                # Prefer least-used cell-output source, then stable object order.
                dense, pin, src_tok, pin_tok = min(
                    available,
                    key=lambda c: (counts[("CELL_OUTPUT", int(c[2]), int(c[3]))], int(c[0]), str(c[1])),
                )
                rec[8] = int(tok.cell_output)
                rec[10] = int(src_tok)
                rec[12] = int(pin_tok)
                sig = ("CELL_OUTPUT", int(src_tok), int(pin_tok))
                forced_source_kind += 1
                selected_cell += 1
                rewritten += 1
            else:
                relaxation += 1
                selected_boundary += 1
        elif kind == "CELL_OUTPUT":
            selected_cell += 1
        elif kind == "BOUNDARY_SOURCE":
            selected_boundary += 1
        counts[sig] += 1
        source_kind_counts[sig[0]] += 1

    # Update SOURCE_FINAL_FANOUT_BUCKET to the final derived fanout for the selected source.
    for rec in sink_records:
        if len(rec) > 14:
            rec[14] = _fanout_tok_for_count(tok, counts[rec_source_sig(rec)])

    payload = dict(payload)
    payload["SINK_DEMAND"] = sink_records
    max_boundary = max([v for k, v in counts.items() if k[0] == "BOUNDARY_SOURCE"] or [0])
    max_cell = max([v for k, v in counts.items() if k[0] == "CELL_OUTPUT"] or [0])
    stats = {
        "source_reuse_mask_enabled": True,
        "boundary_source_fanout_cap": int(boundary_cap),
        "cell_output_source_fanout_cap": int(cell_cap),
        "allow_boundary_17plus": bool(allow_boundary_17),
        "source_kind_budget_exceeded_count": int(source_kind_budget_exceeded),
        "boundary_source_masked_count": int(boundary_masked),
        "boundary_source_budget_relaxation_count": int(relaxation),
        "cell_output_candidate_available_count": int(cell_candidate_available),
        "selected_cell_output_source_count": int(selected_cell),
        "selected_boundary_source_count": int(selected_boundary),
        "forced_source_kind_count": int(forced_source_kind),
        "rewritten_sink_demand_source_count": int(rewritten),
        "max_boundary_source_fanout": int(max_boundary),
        "max_cell_output_source_fanout": int(max_cell),
        "source_kind_usage": dict(source_kind_counts),
    }
    setattr(tok, "last_source_net_v5_source_reuse_mask_stats", stats)
    return payload


def _apply_boundary_identity_refinement_to_payload(tok, payload: dict) -> dict:
    if not bool(getattr(tok, "source_net_v5_enable_boundary_identity_refinement", False)):
        setattr(tok, "last_source_net_v5_boundary_identity_stats", {})
        return payload
    group_size = max(1, int(getattr(tok, "source_net_v5_boundary_identity_group_size", 8) or 8))
    mode = str(getattr(tok, "source_net_v5_boundary_identity_mode", "demand_group") or "demand_group")
    sink_records = [list(r) for r in payload.get("SINK_DEMAND", []) or []]
    boundary_total = 0
    refined = 0
    generic_fallback = 0
    per_demand_fallback = 0
    id_counter = Counter()

    for rec in sink_records:
        if len(rec) <= 12 or int(rec[8]) != int(tok.boundary_source):
            continue
        boundary_total += 1
        demand_idx = None
        try:
            demand_idx = int(int(rec[2]) - int(tok.idx_offset) - 2 * int(tok.max_num_nodes))
        except Exception:
            demand_idx = boundary_total - 1
        if mode == "per_demand":
            group_id = int(demand_idx)
            per_demand_fallback += 1
        else:
            # Deterministic metadata-only fallback: split the generic boundary
            # object into stable local source groups by demand order.  This does
            # not alter demand/load coverage and does not cross source kind.
            group_id = int(demand_idx) // int(group_size)
        if 0 <= group_id < int(tok.max_num_nodes):
            rec[10] = int(tok._tok_src_idx(group_id))
            rec[12] = int(tok._tok_pin("__BOUNDARY__"))
            refined += 1
            id_counter[f"BOUNDARY_SRC_GROUP_{group_id}"] += 1
        else:
            generic_fallback += 1
    # Update final fanout bucket for refined boundary ids.
    sig_counts = Counter()
    for rec in sink_records:
        if len(rec) > 12:
            sig_counts[(int(rec[8]), int(rec[10]), int(rec[12]))] += 1
    for rec in sink_records:
        if len(rec) > 14:
            rec[14] = _fanout_tok_for_count(tok, sig_counts[(int(rec[8]), int(rec[10]), int(rec[12]))])

    payload = dict(payload)
    payload["SINK_DEMAND"] = sink_records
    stats = {
        "boundary_identity_refinement_enabled": True,
        "boundary_identity_mode": mode,
        "boundary_identity_group_size": int(group_size),
        "total_boundary_demands": int(boundary_total),
        "refined_boundary_id_count": int(refined),
        "refined_boundary_unique_count": int(len(id_counter)),
        "generic_boundary_fallback_count": int(generic_fallback),
        "per_demand_fallback_count": int(per_demand_fallback),
        "example_boundary_ids": [k for k, _v in id_counter.most_common(10)],
        "max_refined_boundary_group_fanout": int(max(id_counter.values() or [0])),
    }
    setattr(tok, "last_source_net_v5_boundary_identity_stats", stats)
    return payload


def _install_hard_reference_payload(tok, data, stats: dict | None = None) -> dict:
    true_tokens = [int(x) for x in tok.tokenize(data).reshape(-1).tolist()]
    payload = _build_hard_reference_payload(tok, true_tokens)
    payload = _apply_boundary_identity_refinement_to_payload(tok, payload)
    payload = _apply_source_reuse_budget_to_payload(tok, payload)
    setattr(tok, "source_net_v5_hard_reference_records", payload)
    setattr(tok, "source_net_v5_enable_hard_reference_mask", True)
    setattr(tok, "last_source_net_v5_hard_mask_stats", {})
    counts = {k: len(v) for k, v in payload.items()}
    ok = (
        counts.get("DEMAND", 0) == counts.get("SINK_DEMAND", -1)
        and counts.get("CELL", 0) > 0
        and counts.get("SOURCE_NET", 0) > 0
    )
    out = {
        "record_counts": counts,
        "mask_gold_coverage": 1.0 if ok else 0.0,
        "demand_table_complete": counts.get("DEMAND", 0) > 0,
        "object_table_complete": counts.get("CELL", 0) > 0,
        "source_table_complete": counts.get("SOURCE_NET", 0) > 0,
    }
    if stats is not None:
        stats.update(out)
    return out


def _set_hard_reference_sections(tok, sections: str) -> None:
    vals = [x.strip() for x in str(sections or "").split(",") if x.strip()]
    if not vals:
        vals = ["DEMAND", "CELL", "SINK_DEMAND"]
    setattr(tok, "source_net_v5_hard_reference_sections", vals)


def _run_hard_mask_dry_run(args, tok, root: Path, out: Path) -> None:
    rows = []
    paths = list(iter_partition_paths(root, limit=int(args.hard_mask_dry_run_samples), split="test"))
    if not paths:
        paths = list(iter_partition_paths(root, limit=int(args.hard_mask_dry_run_samples)))
    for idx, path in enumerate(paths):
        data = torch.load(path, map_location="cpu", weights_only=False)
        payload = _install_hard_reference_payload(tok, data)
        rec = tok.serialize_records(data)
        pin_missing = 0
        for gid in rec.get("gates", []):
            try:
                cell = tok._cell_name(int(data.x.reshape(-1)[int(gid)].item()))
                inputs, outputs = tok._spec_pins(cell)
                if not inputs and not outputs:
                    pin_missing += 1
            except Exception:
                pin_missing += 1
        row = {
            "sample": idx,
            "path": str(path),
            "demand_count": payload["record_counts"].get("DEMAND", 0),
            "sink_demand_count": payload["record_counts"].get("SINK_DEMAND", 0),
            "cell_count": payload["record_counts"].get("CELL", 0),
            "source_net_count": payload["record_counts"].get("SOURCE_NET", 0),
            "mask_gold_coverage": payload["mask_gold_coverage"],
            "pin_spec_lookup_missing_count": int(pin_missing),
        }
        rows.append(row)
    report = {
        "run_scope": "source_net_v5_hard_mask_dry_run",
        "samples": len(rows),
        "mask_gold_coverage_min": min((float(r["mask_gold_coverage"]) for r in rows), default=0.0),
        "pin_spec_lookup_missing_count": sum(int(r["pin_spec_lookup_missing_count"]) for r in rows),
        "object_table_coverage": all(int(r["cell_count"]) > 0 for r in rows),
        "demand_table_coverage": all(int(r["demand_count"]) > 0 and int(r["demand_count"]) == int(r["sink_demand_count"]) for r in rows),
        "source_table_coverage": all(int(r["source_net_count"]) > 0 for r in rows),
        "rows": rows,
    }
    write_csv(out / "hard_reference_mask_dry_run.csv", rows)
    write_json(out / "hard_reference_mask_dry_run.json", report)
    write_md(out / "hard_reference_mask_dry_run.md", [
        "# Source-Net V5 Hard Reference Mask Dry Run",
        f"- samples: `{report['samples']}`",
        f"- mask_gold_coverage_min: `{report['mask_gold_coverage_min']}`",
        f"- pin_spec_lookup_missing_count: `{report['pin_spec_lookup_missing_count']}`",
        f"- object_table_coverage: `{report['object_table_coverage']}`",
        f"- demand_table_coverage: `{report['demand_table_coverage']}`",
        f"- source_table_coverage: `{report['source_table_coverage']}`",
    ])
    print(json.dumps(report, indent=2, sort_keys=True))


def _run_diagnostics(args, model, root: Path, out: Path, meta: dict) -> None:
    paths = list(iter_partition_paths(root, limit=1, split="test")) or list(iter_partition_paths(root, limit=1))
    sample_path = paths[0] if paths else None
    true_tokens = None
    if sample_path is not None:
        data = torch.load(sample_path, map_location="cpu", weights_only=False)
        real_stats = _graph_stats(data, model.tokenizer)
        expected_count = int(real_stats.get("required_input_count", 0))
        setattr(model.tokenizer, "source_net_v5_expected_demand_records", expected_count)
        setattr(model.tokenizer, "source_net_v5_expected_sink_demand_records", expected_count)
        setattr(model.tokenizer, "source_net_v5_expected_cell_records", int(real_stats.get("cell_count", 0)))
        setattr(model.tokenizer, "source_net_v5_expected_source_net_records", int(real_stats.get("source_net_count", 0)))
        setattr(model.tokenizer, "source_net_v5_record_budget_tolerance", int(getattr(args, "record_budget_tolerance", 0)))
        true_tokens = [int(x) for x in model.tokenizer.tokenize(data).reshape(-1).tolist()]
        if bool(getattr(args, "enable_hard_reference_mask", False)):
            _install_hard_reference_payload(model.tokenizer, data)
            _set_hard_reference_sections(model.tokenizer, getattr(args, "hard_reference_mask_sections", ""))
    full_prefix = [int(model.tokenizer.sos)]
    cond_prefix = _prefix_until(true_tokens, int(model.tokenizer.cond_section_end)) if true_tokens else full_prefix
    cell_prefix = _prefix_until(true_tokens, int(model.tokenizer.cell_section_end)) if true_tokens else full_prefix
    sink_prefix = _prefix_until(true_tokens, int(model.tokenizer.sink_demand_section_end)) if true_tokens else full_prefix
    cases = [
        ("full_free_run", full_prefix, True, int(args.max_new_tokens), None),
        ("oracle_prefix_demand", cond_prefix, True, int(args.max_new_tokens), int(model.tokenizer.demand_section_end)),
        ("oracle_prefix_sink_demand", cell_prefix, True, int(args.max_new_tokens), int(model.tokenizer.sink_demand_section_end)),
        ("oracle_prefix_source_net", sink_prefix, True, int(args.max_new_tokens), int(model.tokenizer.eos)),
        ("no_grammar_baseline", full_prefix, False, min(512, int(args.max_new_tokens)), None),
    ]
    requested_cases = [x.strip() for x in str(getattr(args, "diagnostic_cases", "") or "").split(",") if x.strip()]
    if requested_cases:
        wanted = set(requested_cases)
        cases = [case for case in cases if case[0] in wanted]
    results = {}
    partial_dir = out / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    for name, prefix, use_grammar, max_new, stop_after in cases:
        res = _bounded_generate(
            model,
            prefix,
            max_new_tokens=max_new,
            timeout_sec=int(args.timeout_sec),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            temperature=float(args.temperature),
            use_grammar=use_grammar,
            dump_every_n_tokens=int(args.dump_every_n_tokens),
            profile_generation=bool(args.profile_generation),
            section_token_budget=bool(args.section_token_budget),
            force_close_sections=bool(args.force_close_sections),
            stop_after_token=stop_after,
            source_net_optional=bool(getattr(args, "source_net_optional", False)),
            source_net_min_completion_ratio=float(getattr(args, "source_net_min_completion_ratio", 0.95)),
            source_net_record_budget=int(getattr(args, "source_net_record_budget", 0)),
            source_net_force_close_diagnostic=bool(getattr(args, "source_net_force_close_diagnostic", False)),
        )
        ids = res.pop("generated_ids")
        (partial_dir / f"{name}_tokens.txt").write_text("\n".join(_tok_name(model.tokenizer, x) for x in ids) + "\n", encoding="utf-8")
        (partial_dir / f"{name}_trace.json").write_text(json.dumps(res["trace"], indent=2, sort_keys=True), encoding="utf-8")
        results[name] = res
    root_cause = "unknown"
    full_res = results.get("full_free_run")
    sink_res = results.get("oracle_prefix_sink_demand")
    source_res = results.get("oracle_prefix_source_net")
    if full_res is not None and full_res["stopped_reason"] in {"timeout", "max_new_tokens"}:
        if sink_res is not None and sink_res["stopped_reason"] in {"timeout", "max_new_tokens"}:
            root_cause = "sink_demand_section_free_run_does_not_close_with_light_grammar"
        elif source_res is not None and source_res["stopped_reason"] in {"timeout", "max_new_tokens"}:
            root_cause = "source_net_section_free_run_does_not_close_with_light_grammar"
        else:
            root_cause = "full_bos_free_run_sequence_length_or_early_sections"
    no_grammar = results.get("no_grammar_baseline")
    if no_grammar is not None and full_res is not None and no_grammar["tokens_per_sec"] > max(1.0e-9, full_res["tokens_per_sec"] * 2):
        runtime_issue = True
    else:
        runtime_issue = bool(full_res is not None and full_res["tokens_per_sec"] < 5.0)
    report = {
        "run_scope": "source_net_v5_generated_runtime_diagnostic",
        "guarded_exploratory": True,
        "final_result": False,
        "ckpt": str(args.ckpt),
        "dataset_root": str(root),
        "serializer_version": str(meta.get("serializer_version", "")),
        "max_new_tokens": int(args.max_new_tokens),
        "timeout_sec": int(args.timeout_sec),
        "use_cache": True,
        "batch_size": 1,
        "grammar": "SourceNetV5Grammar light only",
        "strong_grammar": False,
        "root_cause": root_cause,
        "runtime_issue": bool(runtime_issue),
        "grammar_issue": any(r["grammar_anomalies"] for r in results.values()),
        "section_loop": any(r["section_loop"] for r in results.values()),
        "eos_or_section_close_issue": any(r["unable_to_close_section"] for r in results.values()),
        "bounded_generation_ready": (
            full_res is not None
            and full_res["stopped_reason"] == "eos"
            and all(
                res["stopped_reason"] in {"eos", "stop_after_token"}
                for key, res in results.items()
                if key.startswith("oracle_prefix_")
            )
        ),
        "expected_demand_records": int(getattr(model.tokenizer, "source_net_v5_expected_demand_records", -1) or -1),
        "expected_sink_demand_records": int(getattr(model.tokenizer, "source_net_v5_expected_sink_demand_records", -1) or -1),
        "expected_cell_records": int(getattr(model.tokenizer, "source_net_v5_expected_cell_records", -1) or -1),
        "expected_source_net_records": int(getattr(model.tokenizer, "source_net_v5_expected_source_net_records", -1) or -1),
        "generated_smoke_32_allowed": False,
        "stage2_allowed": False,
        "skeleton_allowed": False,
        "cap32_allowed": False,
        "results": results,
        "recommendation": {
            "continue_stage1_training": "No until bounded generated diagnostic can close sections",
            "fix_grammar": "Add diagnostic/bounded section budgets or stronger record-budget closure for V5 light grammar",
            "add_bounded_generation": "Yes; begin with prefix-conditioned section-wise generation and explicit max_new_tokens",
            "reduce_max_new_tokens": "Yes for diagnostics; do not call full smoke at max_length=12288 without section budgets",
            "use_prefix_conditioned_section_generation": "Yes",
            "enter_stage2": "No-Go",
        },
        "safety": {
            "stage2_weights": "not_started",
            "skeleton": "not_started/no-go",
            "kahypar": "not_run",
            "cap32": "not_run/no-go",
            "projection_exporter": "not_modified",
            "pointer_head": "not_implemented",
            "attention_bias": "not_implemented",
            "cross_attention": "not_implemented",
        },
    }
    write_json(out / "source_net_v5_generated_runtime_diagnostic_report.json", report)
    lines = [
        "# Source-Net V5 Generated Runtime Diagnostic Report",
        "",
        "## 1. Summary",
        f"- root_cause: `{root_cause}`",
        f"- runtime_issue: `{report['runtime_issue']}`",
        f"- grammar_issue: `{report['grammar_issue']}`",
        f"- section_loop: `{report['section_loop']}`",
        f"- eos_or_section_close_issue: `{report['eos_or_section_close_issue']}`",
        f"- bounded_generation_ready: `{report['bounded_generation_ready']}`",
        "- Stage 2 / skeleton / cap32: `No-Go / No-Go / No-Go`",
        "",
        "## 2. Generation Profiling",
    ]
    for name, res in results.items():
        lines.extend([
            f"### {name}",
            f"- generated_token_count: `{res['generated_token_count']}`",
            f"- new_token_count: `{res['new_token_count']}`",
            f"- elapsed_sec: `{res['elapsed_sec']}`",
            f"- tokens/sec: `{res['tokens_per_sec']}`",
            f"- stopped_reason: `{res['stopped_reason']}`",
            f"- current_section: `{res['state']['current_section']}`",
            f"- section_token_counts: `{res['state']['section_token_counts']}`",
            f"- record_counts: `{res['state']['record_counts']}`",
            f"- expected/generated demand: `{res['expected_demand_records']}/{res['generated_demand_records']}`",
            f"- expected/generated sink demand: `{res['expected_sink_demand_records']}/{res['generated_sink_demand_records']}`",
            f"- expected/generated cell: `{res['expected_cell_records']}/{res['generated_cell_records']}`",
            f"- expected/generated source_net: `{res['expected_source_net_records']}/{res['generated_source_net_records']}`",
            f"- section_close_reason: `{res['section_close_reason']}`",
            f"- budget_exceeded_count: `{res['budget_exceeded_count']}`",
            f"- section_loop: `{res['section_loop']}`",
            f"- unable_to_close_section: `{res['unable_to_close_section']}`",
            f"- EOS allowed: `{res['grammar_allowed'].get('eos_allowed')}`",
            f"- SECTION_END allowed: `{res['grammar_allowed'].get('section_end_allowed')}`",
            f"- mask_all_invalid: `{res['grammar_allowed'].get('mask_all_invalid')}`",
            "",
        ])
    lines.extend([
        "## 3. Decision Table",
        "",
        "| 项目 | 判断 | 理由 |",
        "| --- | --- | --- |",
        f"| full free-run num=1 | {'Go' if results.get('full_free_run', {}).get('stopped_reason')=='eos' else 'No-Go'} | stopped_reason={results.get('full_free_run', {}).get('stopped_reason', 'not_run')}, section={results.get('full_free_run', {}).get('state', {}).get('current_section', 'not_run')} |",
        f"| sink-only oracle-prefix generation | {'Go' if results.get('oracle_prefix_sink_demand', {}).get('stopped_reason') in {'eos', 'stop_after_token'} else 'No-Go'} | stopped_reason={results.get('oracle_prefix_sink_demand', {}).get('stopped_reason', 'not_run')} |",
        f"| source-net-only oracle-prefix generation | {'Go' if results.get('oracle_prefix_source_net', {}).get('stopped_reason') in {'eos', 'stop_after_token'} else 'No-Go'} | stopped_reason={results.get('oracle_prefix_source_net', {}).get('stopped_reason', 'not_run')} |",
        f"| grammar issue | {'Yes' if report['grammar_issue'] else 'Unknown'} | mask anomaly={report['grammar_issue']} |",
        f"| runtime issue | {'Yes' if report['runtime_issue'] else 'No'} | token/sec comparison with no-grammar baseline |",
        f"| section loop | {'Yes' if report['section_loop'] else 'No'} | sections over diagnostic budget |",
        f"| EOS/section close issue | {'Yes' if report['eos_or_section_close_issue'] else 'No'} | unable_to_close_section flag |",
        f"| bounded generation ready | {'Go' if report['bounded_generation_ready'] else 'No-Go'} | all diagnostic cases must end with EOS |",
        "| generated smoke 32 allowed | No-Go | num=1 diagnostic must pass first |",
        "| Stage 2 allowed | No-Go | generated smoke not available |",
        "| skeleton allowed | No-Go | explicitly out of scope and no generated pass |",
        "| cap32 allowed | No-Go | explicitly disabled |",
    ])
    write_md(out / "source_net_v5_generated_runtime_diagnostic_report.md", lines)
    print(json.dumps(report, indent=2, sort_keys=True))


def _fanout_bucket(n: int) -> str:
    n = int(n)
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 4:
        return "2_4"
    if n <= 8:
        return "5_8"
    if n <= 16:
        return "9_16"
    return "17_PLUS"


def _tv(a: Counter, b: Counter) -> float:
    at = sum(a.values())
    bt = sum(b.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / max(1, at) - b.get(k, 0) / max(1, bt)) for k in keys)


def _cell_name(tok, label: int) -> str:
    return tok._cell_name(int(label))


def _graph_stats(data, tok) -> dict:
    x = data.x.reshape(-1).to(torch.long) if getattr(data, "x", None) is not None else torch.zeros((0,), dtype=torch.long)
    roles = getattr(data, "edge_role", getattr(data, "edge_attr", None))
    edge_pin = getattr(data, "edge_pin_id", None)
    pin_names = list(getattr(data, "pin_id_to_name", []) or [])
    labels = [int(v) for v in x.tolist()]
    gate_ids = [i for i, label in enumerate(labels) if label not in {int(tok.net_id), int(tok.boundary_stub_id)}]
    gate_set = set(gate_ids)
    required = {}
    for gid in gate_ids:
        inputs, _outputs = tok._spec_pins(_cell_name(tok, labels[gid]))
        for pin in inputs:
            required[(gid, str(pin))] = False

    invalid_sink = 0
    invalid_src = 0
    invalid_cell = 0
    input_used_as_src = 0
    output_used_as_sink = 0
    sink_counter = Counter()
    drivers_by_net = defaultdict(list)
    loads_by_net = defaultdict(list)
    cell_output_sources = 0
    boundary_sources = 0
    input_pool_sources = 0
    private_out_sources = 0
    fallback_count = 0

    if roles is not None and edge_pin is not None and getattr(data, "edge_index", None) is not None:
        for eidx, (src, dst) in enumerate(data.edge_index.t().tolist()):
            role = int(roles[eidx].item())
            pid = int(edge_pin[eidx].item()) if int(edge_pin.numel()) > eidx else -1
            pin = str(pin_names[pid]) if 0 <= pid < len(pin_names) else "__UNKNOWN__"
            src = int(src)
            dst = int(dst)
            if role == 1:
                loads_by_net[dst].append((src, pin))
                sink_counter[(src, pin)] += 1
                if src not in gate_set:
                    invalid_sink += 1
                    invalid_cell += 1
                    continue
                inputs, outputs = tok._spec_pins(_cell_name(tok, labels[src]))
                if pin not in inputs:
                    invalid_sink += 1
                    invalid_cell += 1
                    if pin in outputs:
                        output_used_as_sink += 1
                elif (src, pin) in required:
                    required[(src, pin)] = True
            elif role in {2, 3}:
                drivers_by_net[dst].append((src, pin))
                if src in gate_set:
                    cell = _cell_name(tok, labels[src])
                    _inputs, outputs = tok._spec_pins(cell)
                    if pin not in outputs:
                        invalid_src += 1
                        invalid_cell += 1
                        if pin in _inputs:
                            input_used_as_src += 1
                    if cell == "INPUT_POOL":
                        input_pool_sources += 1
                        fallback_count += 1
                    elif cell == "PRIVATE_OUT":
                        private_out_sources += 1
                        fallback_count += 1
                    else:
                        cell_output_sources += 1
                else:
                    boundary_sources += 1

    fanout_hist = Counter()
    used_fanout_hist = Counter()
    cell_output_17 = 0
    boundary_17 = 0
    source_signatures = Counter()
    used_source_fanout = {}
    for net, drivers in drivers_by_net.items():
        fanout = len(loads_by_net.get(net, []))
        bucket = _fanout_bucket(fanout)
        used_fanout_hist[bucket] += 1
        source_kind = "BOUNDARY_SOURCE"
        signature = ("BOUNDARY_SOURCE", "__BOUNDARY__")
        if drivers:
            dsrc, dpin = drivers[0]
            if dsrc in gate_set:
                cell = _cell_name(tok, labels[dsrc])
                source_kind = "CELL_OUTPUT"
                signature = (source_kind, cell, dpin)
                fanout_key = (source_kind, cell, dpin, int(dsrc))
            else:
                signature = (source_kind, int(dsrc), dpin)
                fanout_key = signature
        else:
            fanout_key = signature
        used_source_fanout[fanout_key] = max(int(used_source_fanout.get(fanout_key, 0)), int(fanout))
        source_signatures[signature] += 1
        if bucket == "17_PLUS":
            if source_kind == "CELL_OUTPUT":
                cell_output_17 += 1
            else:
                boundary_17 += 1

    all_source_candidates = set()
    for gid in gate_ids:
        cell = _cell_name(tok, labels[gid])
        _inputs, outputs = tok._spec_pins(cell)
        for pin in outputs:
            all_source_candidates.add(("CELL_OUTPUT", cell, str(pin), int(gid)))
    # Boundary/stub source cardinality is not fully identifiable without interface
    # object ids in generated sequences, so include only observed non-cell sources.
    for sig in used_source_fanout:
        if sig and sig[0] != "CELL_OUTPUT":
            all_source_candidates.add(sig)
    for cand in all_source_candidates:
        if cand[0] == "CELL_OUTPUT":
            lookup = ("CELL_OUTPUT", cand[1], cand[2], cand[3])
            fanout = int(used_source_fanout.get(lookup, 0))
        else:
            fanout = int(used_source_fanout.get(cand, 0))
        fanout_hist[_fanout_bucket(fanout)] += 1

    covered = sum(1 for v in required.values() if v)
    required_count = len(required)
    load_count = sum(len(v) for v in loads_by_net.values())
    source_count = cell_output_sources + boundary_sources
    cell_counts = Counter(_cell_name(tok, labels[gid]).split("_", 1)[0] for gid in gate_ids)
    return {
        "decode_valid": bool(getattr(data, "decode_valid", False)),
        "cell_count": len(gate_ids),
        "source_net_count": len(drivers_by_net),
        "load_count": load_count,
        "required_input_count": required_count,
        "covered_required_input_count": covered,
        "required_input_coverage": covered / max(1, required_count),
        "missing_required_input_load_count": max(0, required_count - covered),
        "invalid_sink_pin_count": invalid_sink,
        "invalid_src_pin_count": invalid_src,
        "invalid_cell_pin_count": invalid_cell,
        "input_used_as_src_count": input_used_as_src,
        "output_used_as_sink_count": output_used_as_sink,
        "duplicate_sink_count": sum(v - 1 for v in sink_counter.values() if v > 1),
        "topology_collapse": len(gate_ids) == 0 or load_count == 0 or required_count == 0,
        "fanout_hist": dict(fanout_hist),
        "derived_fanout_hist_all_sources": dict(fanout_hist),
        "derived_fanout_hist_used_sources": dict(used_fanout_hist),
        "derived_source_net_count": int(len(used_source_fanout)),
        "derived_source_candidate_count": int(len(all_source_candidates)),
        "derived_17_plus_generated": int(used_fanout_hist.get("17_PLUS", 0)),
        "cell_output_source_17_plus_count": cell_output_17,
        "boundary_source_17_plus_count": boundary_17,
        "repeated_source_signature": sum(1 for v in source_signatures.values() if v > 1),
        "boundary_source_ratio": boundary_sources / max(1, source_count),
        "cell_output_source_ratio": cell_output_sources / max(1, source_count),
        "fallback_count": fallback_count,
        "INPUT_POOL_count": input_pool_sources,
        "PRIVATE_OUT_count": private_out_sources,
        "FA_count": sum(1 for gid in gate_ids if _cell_name(tok, labels[gid]).startswith("FA")),
        "HA_count": sum(1 for gid in gate_ids if _cell_name(tok, labels[gid]).startswith("HA")),
        "DFF_count": sum(1 for gid in gate_ids if _cell_name(tok, labels[gid]).startswith("DFF")),
    }


def _sum_counter_dict(rows: list[dict], key: str) -> Counter:
    out = Counter()
    for row in rows:
        out.update(row.get(key, {}) or {})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-root", default=os.environ.get("ELDA_V5_DATA_ROOT", "datasets/source_net_v5"))
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=12288)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--pin-slot-mask", action="store_true")
    ap.add_argument("--pin-slot-mask-mode", default="source_net_v5_light")
    ap.add_argument("--cuda-device", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--section-token-budget", action="store_true")
    ap.add_argument("--record-budget", action="store_true")
    ap.add_argument("--dump-partial", action="store_true")
    ap.add_argument("--dump-every-n-tokens", type=int, default=128)
    ap.add_argument("--force-close-sections", action="store_true")
    ap.add_argument("--profile-generation", action="store_true")
    ap.add_argument("--stop-on-section-loop", action="store_true")
    ap.add_argument("--disable-strong-grammar", action="store_true")
    ap.add_argument("--light-grammar-only", action="store_true")
    ap.add_argument("--record-budget-tolerance", type=int, default=0)
    ap.add_argument("--diagnostic-cases", default="")
    ap.add_argument("--enable-hard-reference-mask", action="store_true")
    ap.add_argument("--hard-reference-mask-sections", default="DEMAND,CELL,SINK_DEMAND")
    ap.add_argument("--hard-mask-dry-run", action="store_true")
    ap.add_argument("--hard-mask-dry-run-samples", type=int, default=10)
    ap.add_argument("--source-net-optional", action="store_true")
    ap.add_argument("--source-net-token-budget", type=int, default=0)
    ap.add_argument("--source-net-record-budget", type=int, default=0)
    ap.add_argument("--source-net-min-completion-ratio", type=float, default=0.95)
    ap.add_argument("--source-net-force-close-diagnostic", action="store_true")
    ap.add_argument("--coverage-first-decode", action="store_true")
    ap.add_argument("--enable-source-reuse-mask", action="store_true")
    ap.add_argument("--boundary-source-fanout-cap", type=int, default=8)
    ap.add_argument("--cell-output-source-fanout-cap", type=int, default=1000000000)
    ap.add_argument("--stub-source-fanout-cap", type=int, default=8)
    ap.add_argument("--allow-boundary-17plus", action="store_true")
    ap.add_argument("--source-kind-budget-mode", default="boundary_cap")
    ap.add_argument("--enable-boundary-identity-refinement", action="store_true")
    ap.add_argument("--boundary-identity-mode", default="demand_group")
    ap.add_argument("--boundary-identity-group-size", type=int, default=8)
    ap.add_argument("--v6-practical-constrained-source-net", action="store_true")
    ap.add_argument("--constraint-level", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--forbid-duplicate-source-group", action="store_true")
    ap.add_argument("--enable-level3-eos-gating", action="store_true")
    ap.add_argument("--constraint-dead-end-policy", default="fail", choices=["fail"])
    ap.add_argument("--source-net-close-bias", type=float, default=0.0)
    ap.add_argument("--disable-exact-load-group-completion", action="store_true")
    ap.add_argument("--disable-exhausted-source-group-close", action="store_true")
    ap.add_argument("--materialize-export", action="store_true")
    ap.add_argument("--run-yosys", action="store_true")
    ap.add_argument("--yosys-bin", default="/opt/OpenROAD/oss-cad-suite/bin/yosys")
    ap.add_argument("--liberty", default="")
    ap.add_argument("--mapping", default="")
    args = ap.parse_args()

    force_cpu = str(args.cuda_device).strip().lower() in {"cpu", "none", "-1"}
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.cuda_device != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.dataset_root)
    if not str(args.mapping):
        args.mapping = str(root / "mapping_v5.txt")
    if bool(args.materialize_export) and not Path(args.mapping).exists():
        raise FileNotFoundError(f"mapping file not found: {args.mapping}")
    real_tok, meta = build_v5_tokenizer(root, max_length=int(args.max_length))
    if bool(args.hard_mask_dry_run):
        _run_hard_mask_dry_run(args, real_tok, root, out)
        return
    model = SequenceModel.load_from_checkpoint(str(args.ckpt), weights_only=False, map_location="cpu")
    model.eval()
    if (not force_cpu) and torch.cuda.is_available():
        cuda_arg = None
        if str(args.cuda_device).strip() != "":
            try:
                cuda_arg = int(str(args.cuda_device).strip())
            except ValueError:
                cuda_arg = None
        model = model.cuda(device=cuda_arg) if cuda_arg is not None else model.cuda()
    with open_dict(model.cfg):
        model.cfg.sampling.max_length = int(args.max_length)
        model.cfg.sampling.batch_size = 1
        model.cfg.sampling.top_k = int(args.top_k)
        model.cfg.sampling.top_p = float(args.top_p)
        model.cfg.sampling.temperature = float(args.temperature)
        model.cfg.sampling.pin_slot_mask = bool(args.pin_slot_mask)
        model.cfg.sampling.pin_slot_mask_mode = str(args.pin_slot_mask_mode)
        model.cfg.sampling.disable_metric_init = True
    setattr(model.tokenizer, "source_net_v5_enable_source_reuse_mask", bool(args.enable_source_reuse_mask))
    setattr(model.tokenizer, "source_net_v5_boundary_source_fanout_cap", int(args.boundary_source_fanout_cap))
    setattr(model.tokenizer, "source_net_v5_cell_output_source_fanout_cap", int(args.cell_output_source_fanout_cap))
    setattr(model.tokenizer, "source_net_v5_stub_source_fanout_cap", int(args.stub_source_fanout_cap))
    setattr(model.tokenizer, "source_net_v5_allow_boundary_17plus", bool(args.allow_boundary_17plus))
    setattr(model.tokenizer, "source_net_v5_source_kind_budget_mode", str(args.source_kind_budget_mode))
    setattr(model.tokenizer, "source_net_v5_enable_boundary_identity_refinement", bool(args.enable_boundary_identity_refinement))
    setattr(model.tokenizer, "source_net_v5_boundary_identity_mode", str(args.boundary_identity_mode))
    setattr(model.tokenizer, "source_net_v5_boundary_identity_group_size", int(args.boundary_identity_group_size))

    if bool(args.diagnostic):
        _run_diagnostics(args, model, root, out, meta)
        return

    rows = []
    real_rows = []
    errors = []
    paths = list(iter_partition_paths(root, limit=int(args.num_samples), split="test"))
    if not paths:
        paths = list(iter_partition_paths(root, limit=int(args.num_samples)))
    for idx, path in enumerate(paths):
        data = torch.load(path, map_location="cpu", weights_only=False)
        raw_stat = _graph_stats(data, real_tok)
        true_tokens = [int(x) for x in model.tokenizer.tokenize(data).reshape(-1).tolist()]
        # Raw partition PyG edges are not in the V5 decoder's directed
        # source/load-pin graph contract.  Compare graph distributions only
        # against the serializer's own round-trip reference graph.
        reference_decoded = model.tokenizer.decode(torch.tensor(true_tokens, dtype=torch.long))
        real_stat = _graph_stats(reference_decoded, model.tokenizer)
        real_rows.append(real_stat)
        v6_ctx = _build_v6_practical_context(model.tokenizer, true_tokens)
        true_source_net_body = _section_body(true_tokens, int(model.tokenizer.source_net_section_begin), int(model.tokenizer.source_net_section_end))
        true_source_net_count = len(_record_chunks(true_source_net_body, int(model.tokenizer.source_net_begin), int(model.tokenizer.source_net_end)))
        expected_count = int(raw_stat.get("required_input_count", 0))
        setattr(model.tokenizer, "source_net_v5_expected_demand_records", expected_count)
        setattr(model.tokenizer, "source_net_v5_expected_sink_demand_records", expected_count)
        setattr(model.tokenizer, "source_net_v5_expected_cell_records", int(raw_stat.get("cell_count", 0)))
        setattr(model.tokenizer, "source_net_v5_expected_source_net_records", int(true_source_net_count))
        setattr(model.tokenizer, "source_net_v5_record_budget_tolerance", int(args.record_budget_tolerance))
        if bool(args.enable_hard_reference_mask):
            _install_hard_reference_payload(model.tokenizer, data)
            _set_hard_reference_sections(model.tokenizer, args.hard_reference_mask_sections)
        input_ids = torch.tensor([int(model.tokenizer.sos)], dtype=torch.long, device=model.device).reshape(1, 1)
        try:
            with torch.no_grad():
                if bool(args.v6_practical_constrained_source_net):
                    prefix = _prefix_until(true_tokens, int(model.tokenizer.source_net_section_begin))
                    gen = _bounded_generate(
                        model,
                        prefix,
                        max_new_tokens=int(args.max_new_tokens),
                        timeout_sec=int(args.timeout_sec),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        temperature=float(args.temperature),
                        use_grammar=True,
                        dump_every_n_tokens=int(args.dump_every_n_tokens) if bool(args.dump_partial) else 0,
                        profile_generation=bool(args.profile_generation),
                        section_token_budget=bool(args.section_token_budget),
                        force_close_sections=False,
                        source_net_optional=False,
                        source_net_min_completion_ratio=float(args.source_net_min_completion_ratio),
                        source_net_record_budget=0,
                        source_net_force_close_diagnostic=False,
                        v6_constraint_context=v6_ctx,
                        v6_constraint_level=int(args.constraint_level),
                        v6_forbid_duplicate_source_group=bool(args.forbid_duplicate_source_group),
                        v6_level3_eos_gating=bool(args.enable_level3_eos_gating),
                        v6_exact_load_group_completion=not bool(args.disable_exact_load_group_completion),
                        v6_close_exhausted_source_group=not bool(args.disable_exhausted_source_group_close),
                        source_net_close_bias=float(args.source_net_close_bias),
                    )
                    seq = torch.tensor(gen["generated_ids"], dtype=torch.long)
                    gen_status = gen.get("coverage_decode_status", {})
                    source_force_closed = bool(gen.get("source_net_force_closed", False))
                    stopped_reason = str(gen.get("stopped_reason", ""))
                    constrained_trace = dict(gen.get("constrained_decode_trace", {}) or {})
                elif bool(args.source_net_optional) or bool(args.coverage_first_decode):
                    gen = _bounded_generate(
                        model,
                        [int(model.tokenizer.sos)],
                        max_new_tokens=int(args.max_length),
                        timeout_sec=int(args.timeout_sec),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        temperature=float(args.temperature),
                        use_grammar=bool(args.pin_slot_mask),
                        dump_every_n_tokens=0,
                        profile_generation=False,
                        section_token_budget=bool(args.source_net_token_budget),
                        force_close_sections=False,
                        source_net_optional=bool(args.source_net_optional),
                        source_net_min_completion_ratio=float(args.source_net_min_completion_ratio),
                        source_net_record_budget=int(args.source_net_record_budget),
                        source_net_force_close_diagnostic=bool(args.source_net_force_close_diagnostic),
                        source_net_close_bias=float(args.source_net_close_bias),
                    )
                    seq = torch.tensor(gen["generated_ids"], dtype=torch.long)
                    gen_status = gen.get("coverage_decode_status", {})
                    source_force_closed = bool(gen.get("source_net_force_closed", False))
                    stopped_reason = str(gen.get("stopped_reason", ""))
                    constrained_trace = {}
                else:
                    seqs, _ = model.generate(input_ids=input_ids, return_sent=True)
                    seq = seqs[0].detach().cpu()
                    gen_status = _coverage_decode_status([int(x) for x in seq.reshape(-1).tolist()], model.tokenizer)
                    source_force_closed = False
                    stopped_reason = "hf_generate"
                    constrained_trace = {}
            dec = model.tokenizer.decode(seq)
            row = _graph_stats(dec, model.tokenizer)
            token_list = [int(x) for x in seq.reshape(-1).tolist()]
            demand_stats = _sink_demand_id_stats(token_list, model.tokenizer)
            emitted_source_net_demands = set(
                _demand_idx_from_tok(model.tokenizer, int(rec[i + 1]))
                for rec in _completed_source_net_records(token_list, model.tokenizer)
                for i, t in enumerate(rec)
                if int(t) == int(model.tokenizer.demand_id)
                and i + 1 < len(rec)
                and _demand_idx_from_tok(model.tokenizer, int(rec[i + 1])) is not None
            )
            emitted_source_net_demands.discard(None)
            reference_explicit_demands = set(v6_ctx.get("reference_explicit_demand_ids", set()))
            source_net_reference_hits = emitted_source_net_demands & reference_explicit_demands
            reuse_stats = dict(getattr(model.tokenizer, "last_source_net_v5_source_reuse_mask_stats", {}) or {})
            boundary_identity_stats = dict(getattr(model.tokenizer, "last_source_net_v5_boundary_identity_stats", {}) or {})
            row.update({
                "strict_decode_valid": bool(gen_status.get("strict_decode_valid", False)) and bool(row.get("decode_valid", False)) and not source_force_closed,
                "coverage_decode_valid": bool(gen_status.get("coverage_decode_valid", False)),
                "source_net_complete": bool(gen_status.get("source_net_complete", False)),
                "source_net_records_generated": int(gen_status.get("source_net_records_generated", 0)),
                "source_net_records_expected": int(gen_status.get("source_net_records_expected", -1)),
                "source_net_completion_ratio": float(gen_status.get("source_net_completion_ratio", 0.0)),
                "source_net_force_closed": bool(source_force_closed),
                "source_net_partial_reason": str(gen_status.get("source_net_partial_reason", "")),
                "generation_stopped_reason": stopped_reason,
                "source_net_close_bias": float(args.source_net_close_bias),
                "v6_practical_constrained_source_net": bool(args.v6_practical_constrained_source_net),
                "constraint_level": int(args.constraint_level) if bool(args.v6_practical_constrained_source_net) else 0,
                "generation_dead_end": bool(constrained_trace.get("generation_dead_end", False)),
                "source_net_emitted_demand_count": int(constrained_trace.get("source_net_emitted_demand_count", len(emitted_source_net_demands)) or 0),
                "source_net_non_emitted_demand_count": int(constrained_trace.get("source_net_non_emitted_demand_count", 0) or 0),
                "source_net_internal_missing_source_count": int(constrained_trace.get("internal_missing_source", 0) or 0),
                "source_net_added_missing_vs_source_partition": constrained_trace.get("added_missing_vs_source_partition"),
                "source_net_reference_explicit_demand_count": int(len(reference_explicit_demands)),
                "source_net_reference_explicit_demand_hit_count": int(len(source_net_reference_hits)),
                "source_net_reference_explicit_demand_recall": float(len(source_net_reference_hits)) / max(1, len(reference_explicit_demands)),
                "source_net_reference_explicit_demand_precision": float(len(source_net_reference_hits)) / max(1, len(emitted_source_net_demands)),
                "source_net_duplicate_sink_count": int(constrained_trace.get("duplicate_sink_count", 0) or 0),
                "source_net_duplicate_source_group_count": int(constrained_trace.get("duplicate_source_group_count", 0) or 0),
                "source_net_invalid_source_pin_count": int(constrained_trace.get("invalid_source_pin_count", 0) or 0),
                "source_net_invalid_load_pin_count": int(constrained_trace.get("invalid_load_pin_count", 0) or 0),
                "source_net_source_demand_mismatch_count": int(constrained_trace.get("source_demand_mismatch_count", 0) or 0),
                "source_net_forced_load_group_end_count": int(constrained_trace.get("forced_load_group_end_count", 0) or 0),
                "source_net_exhausted_source_group_close_count": int(constrained_trace.get("exhausted_source_group_close_count", 0) or 0),
                "source_net_malformed_load_entry_count": int(constrained_trace.get("malformed_load_entry_count", 0) or 0),
                "masked_invalid_cell_tokens": int(constrained_trace.get("number_of_masked_invalid_cell_tokens", 0) or 0),
                "masked_invalid_pin_tokens": int(constrained_trace.get("number_of_masked_invalid_pin_tokens", 0) or 0),
                "masked_invalid_demand_tokens": int(constrained_trace.get("number_of_masked_invalid_demand_tokens", 0) or 0),
                "masked_duplicate_sink_tokens": int(constrained_trace.get("number_of_masked_duplicate_sink_tokens", 0) or 0),
                "masked_duplicate_source_tokens": int(constrained_trace.get("number_of_masked_duplicate_source_tokens", 0) or 0),
                "eos_tokens_blocked_by_remaining_internal_demand": int(constrained_trace.get("number_of_EOS_tokens_blocked_by_remaining_internal_demand", 0) or 0),
                **demand_stats,
                **{f"reuse_{k}": v for k, v in reuse_stats.items() if not isinstance(v, dict)},
                **{f"boundary_identity_{k}": v for k, v in boundary_identity_stats.items() if not isinstance(v, (dict, list))},
            })
            row["sample"] = idx
            row["source_path"] = str(path)
            row["sequence_length"] = int(seq.numel())
            if bool(args.v6_practical_constrained_source_net):
                attempt_dir = out / "attempts" / f"attempt_{idx:04d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                (attempt_dir / "raw_generated_sequence.txt").write_text(
                    "\n".join(_tok_name(model.tokenizer, x) for x in token_list) + "\n",
                    encoding="utf-8",
                )
                torch.save(dec, attempt_dir / "decoded_graph.pt")
                source_body = _section_body(token_list, int(model.tokenizer.source_net_section_begin), int(model.tokenizer.source_net_section_end))
                source_records = _record_chunks(source_body, int(model.tokenizer.source_net_begin), int(model.tokenizer.source_net_end))
                source_net_records = {
                    "record_count": len(source_records),
                    "records": [[_tok_name(model.tokenizer, x) for x in rec] for rec in source_records],
                }
                write_json(attempt_dir / "source_net_records.json", source_net_records)
                write_json(attempt_dir / "constrained_decode_trace.json", constrained_trace)
                materialization = {}
                if bool(args.materialize_export):
                    materialization = _materialize_and_strict_export(dec, attempt_dir, model.tokenizer, args)
                    row.update(materialization)
                manifest = {
                    "sample": int(idx),
                    "source_path": str(path),
                "constraint_level": int(args.constraint_level),
                "forbid_duplicate_source_group": bool(args.forbid_duplicate_source_group),
                "exact_load_group_completion_enabled": not bool(args.disable_exact_load_group_completion),
                    "level3_eos_gating_enabled": bool(args.enable_level3_eos_gating),
                    "generation_stopped_reason": stopped_reason,
                    "strict_decode_valid": bool(row["strict_decode_valid"]),
                    "coverage_decode_valid": bool(row["coverage_decode_valid"]),
                    "decoded_graph_path": str(attempt_dir / "decoded_graph.pt"),
                    "decoded_graph_with_stubs_output_completed_path": str(materialization.get("output_completed_graph_path", "")),
                    "candidate_verilog_path": str(materialization.get("candidate_verilog_path", "")),
                    "strict_export_report_path": str(materialization.get("strict_export_report_path", "")),
                    "yosys_report_path": str(materialization.get("yosys_report_path", "")),
                    "strict_export_pass": bool(materialization.get("strict_export_pass", False)),
                }
                write_json(attempt_dir / "manifest.json", manifest)
            rows.append(row)
        except Exception as exc:
            errors.append({"sample": idx, "path": str(path), "error": f"{type(exc).__name__}: {str(exc)[:500]}"})

    n = max(1, len(rows))
    real_fanout = _sum_counter_dict(real_rows, "derived_fanout_hist_all_sources")
    gen_fanout = _sum_counter_dict(rows, "derived_fanout_hist_all_sources")
    real_fanout_used = _sum_counter_dict(real_rows, "derived_fanout_hist_used_sources")
    gen_fanout_used = _sum_counter_dict(rows, "derived_fanout_hist_used_sources")
    def avg(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in rows) / n
    def isum(key: str) -> int:
        return int(sum(int(r.get(key, 0)) for r in rows))

    invalid_cell = isum("invalid_cell_pin_count")
    invalid_sink = isum("invalid_sink_pin_count")
    invalid_src = isum("invalid_src_pin_count")
    duplicate = isum("duplicate_sink_count")
    coverage = avg("required_input_coverage")
    strict_decode_valid_rate = avg("strict_decode_valid")
    coverage_decode_valid_rate = avg("coverage_decode_valid")
    fanout_tv = _tv(real_fanout, gen_fanout)
    fanout_tv_used = _tv(real_fanout_used, gen_fanout_used)
    derived_decode_valid_rate = coverage_decode_valid_rate
    materialization_ok = (
        not bool(args.materialize_export)
        or (
            sum(1 for r in rows if bool(r.get("strict_export_pass", False))) == len(rows)
            and (not bool(args.run_yosys) or sum(1 for r in rows if bool(r.get("yosys_read", False)) and bool(r.get("yosys_check", False)) and bool(r.get("yosys_synth", False))) == len(rows))
        )
    )
    # This gate is intentionally only a decoder-stability gate.  It cannot
    # certify generated netlists because this script keeps the complete
    # SINK_DEMAND prefix fixed and does not materialize/export a netlist.
    pass_32_gate = (
        len(errors) == 0
        and strict_decode_valid_rate >= 0.99
        and avg("source_net_complete") >= 0.99
        and isum("generation_dead_end") == 0
        and isum("source_net_duplicate_sink_count") == 0
        and isum("source_net_duplicate_source_group_count") == 0
        and isum("source_net_invalid_source_pin_count") == 0
        and isum("source_net_invalid_load_pin_count") == 0
        and isum("source_net_source_demand_mismatch_count") == 0
        and isum("source_net_malformed_load_entry_count") == 0
        and materialization_ok
    )
    hard_fail = (
        len(errors) > 0
        or strict_decode_valid_rate < 0.99
        or avg("source_net_complete") < 0.99
        or isum("generation_dead_end") > 0
        or isum("source_net_duplicate_sink_count") > 0
        or isum("source_net_duplicate_source_group_count") > 0
        or isum("source_net_invalid_source_pin_count") > 0
        or isum("source_net_invalid_load_pin_count") > 0
        or isum("source_net_source_demand_mismatch_count") > 0
        or isum("source_net_malformed_load_entry_count") > 0
        or not materialization_ok
    )

    report = {
        "run_scope": "source_net_v5_generated_smoke",
        "v6_practical_constrained_source_net": bool(args.v6_practical_constrained_source_net),
        "constraint_level": int(args.constraint_level) if bool(args.v6_practical_constrained_source_net) else 0,
        "constraint_level3_switch_present": True,
        "constraint_level3_eos_gating_enabled": bool(args.enable_level3_eos_gating),
        "constraint_level3_default_open": False,
        "forbid_duplicate_source_group": bool(args.forbid_duplicate_source_group),
        "constraint_dead_end_policy": str(args.constraint_dead_end_policy),
        "source_net_close_bias": float(args.source_net_close_bias),
        "exact_load_group_completion_enabled": not bool(args.disable_exact_load_group_completion),
        "exhausted_source_group_close_enabled": not bool(args.disable_exhausted_source_group_close),
        "materialize_export_enabled": bool(args.materialize_export),
        "yosys_enabled": bool(args.run_yosys),
        "mapping": str(args.mapping),
        "guarded_exploratory": True,
        "final_result": False,
        "ckpt": str(args.ckpt),
        "dataset_root": str(root),
        "serializer_version": str(meta.get("serializer_version", "")),
        "tokenizer_type": "source_net_v5",
        "num_samples_requested": int(args.num_samples),
        "num_samples_completed": len(rows),
        "generation_errors": errors,
        "sampling_pin_slot_mask": bool(args.pin_slot_mask),
        "pin_slot_mask_mode": str(args.pin_slot_mask_mode),
        "grammar_processor_created": bool(getattr(model.tokenizer, "last_grammar_processor_created", False)),
        "grammar_type": str(getattr(model.tokenizer, "last_grammar_processor_name", "none")),
        "hard_reference_mask_enabled": bool(args.enable_hard_reference_mask),
        "hard_reference_mask_sections": [x.strip() for x in str(args.hard_reference_mask_sections).split(",") if x.strip()],
        "hard_reference_mask_stats": dict(getattr(model.tokenizer, "last_source_net_v5_hard_mask_stats", {}) or {}),
        "source_reuse_mask_enabled": bool(args.enable_source_reuse_mask),
        "source_kind_budget_mode": str(args.source_kind_budget_mode),
        "boundary_source_fanout_cap": int(args.boundary_source_fanout_cap),
        "cell_output_source_fanout_cap": int(args.cell_output_source_fanout_cap),
        "stub_source_fanout_cap": int(args.stub_source_fanout_cap),
        "allow_boundary_17plus": bool(args.allow_boundary_17plus),
        "boundary_identity_refinement_enabled": bool(args.enable_boundary_identity_refinement),
        "boundary_identity_mode": str(args.boundary_identity_mode),
        "boundary_identity_group_size": int(args.boundary_identity_group_size),
        "refined_boundary_id_coverage": (
            isum("boundary_identity_refined_boundary_id_count") / max(1, isum("boundary_identity_total_boundary_demands"))
            if bool(args.enable_boundary_identity_refinement) else 0.0
        ),
        "generated_boundary_signature_count": isum("boundary_identity_refined_boundary_unique_count") if bool(args.enable_boundary_identity_refinement) else isum("boundary_source_17_plus_count"),
        "generic_boundary_fallback_count": isum("boundary_identity_generic_boundary_fallback_count"),
        "per_demand_fallback_count": isum("boundary_identity_per_demand_fallback_count"),
        "max_boundary_source_fanout_refined_identity": max([int(r.get("boundary_identity_max_refined_boundary_group_fanout", 0)) for r in rows] or [0]),
        "source_kind_budget_exceeded_count": isum("reuse_source_kind_budget_exceeded_count"),
        "boundary_source_masked_count": isum("reuse_boundary_source_masked_count"),
        "boundary_source_budget_relaxation_count": isum("reuse_boundary_source_budget_relaxation_count"),
        "cell_output_candidate_available_count": isum("reuse_cell_output_candidate_available_count"),
        "selected_cell_output_source_count": isum("reuse_selected_cell_output_source_count"),
        "selected_boundary_source_count": isum("reuse_selected_boundary_source_count"),
        "forced_source_kind_count": isum("reuse_forced_source_kind_count"),
        "rewritten_sink_demand_source_count": isum("reuse_rewritten_sink_demand_source_count"),
        "max_boundary_source_fanout": max([int(r.get("reuse_max_boundary_source_fanout", 0)) for r in rows] or [0]),
        "max_cell_output_source_fanout": max([int(r.get("reuse_max_cell_output_source_fanout", 0)) for r in rows] or [0]),
        "decode_valid_rate": avg("decode_valid"),
        "strict_decode_valid_rate": strict_decode_valid_rate,
        "coverage_decode_valid_rate": coverage_decode_valid_rate,
        "derived_decode_valid_rate": derived_decode_valid_rate,
        "derived_source_net_decode_valid_rate": derived_decode_valid_rate,
        "decoded_graph_metrics_scope": "fixed_SINK_DEMAND_prefix; decoded_graph.pt does not use free-generated SOURCE_NET as connectivity truth",
        "required_input_coverage": coverage,
        "required_input_coverage_scope": "fixed_SINK_DEMAND_prefix; not a free_SOURCE_NET_generation_metric",
        "missing_required_input_load_count": isum("missing_required_input_load_count"),
        "invalid_cell_pin_count": invalid_cell,
        "invalid_sink_pin_count": invalid_sink,
        "invalid_src_pin_count": invalid_src,
        "input_used_as_src_count": isum("input_used_as_src_count"),
        "output_used_as_sink_count": isum("output_used_as_sink_count"),
        "duplicate_sink_count": duplicate,
        "demand_id_duplicate_count": isum("demand_id_duplicate_count"),
        "demand_id_missing_count": isum("demand_id_missing_count"),
        "forced_demand_id_count": int(sum(int(r.get("sink_demand_id_count", 0)) for r in rows)) if bool(args.enable_hard_reference_mask) else 0,
        "forced_load_cell_count": int(sum(int(r.get("sink_demand_id_count", 0)) for r in rows)) if bool(args.enable_hard_reference_mask) else 0,
        "forced_load_pin_count": int(sum(int(r.get("sink_demand_id_count", 0)) for r in rows)) if bool(args.enable_hard_reference_mask) else 0,
        "illegal_source_object_masked_count": "not_counted_explicitly; use masked_invalid_cell_tokens for the dynamic SOURCE_NET mask",
        "illegal_source_pin_masked_count": "not_counted_explicitly; use masked_invalid_pin_tokens for the dynamic SOURCE_NET mask",
        "illegal_load_pin_masked_count": "not_counted_explicitly; use masked_invalid_pin_tokens for the dynamic SOURCE_NET mask",
        "mask_all_invalid_count": len([e for e in errors if "mask_all_invalid" in str(e.get("error", ""))]),
        "generation_dead_end_count": isum("generation_dead_end"),
        "source_net_emitted_demand_count": isum("source_net_emitted_demand_count"),
        "source_net_non_emitted_demand_count": isum("source_net_non_emitted_demand_count"),
        "source_net_internal_missing_source_count": isum("source_net_internal_missing_source_count"),
        "source_net_internal_missing_source_scope": "explicit SOURCE_NET only; high-fanout previews make this diagnostic, not full-demand coverage",
        "source_net_added_missing_vs_source_partition": "not_available_for_free_SOURCE_NET_suffix_generation",
        "source_net_emitted_demand_scope": "explicit SOURCE_NET loads only; high-fanout records are bounded previews, so this is not missing-net coverage",
        "source_net_reference_explicit_demand_count": isum("source_net_reference_explicit_demand_count"),
        "source_net_reference_explicit_demand_hit_count": isum("source_net_reference_explicit_demand_hit_count"),
        "source_net_reference_explicit_demand_recall": (
            isum("source_net_reference_explicit_demand_hit_count") / max(1, isum("source_net_reference_explicit_demand_count"))
        ),
        "source_net_reference_explicit_demand_precision": (
            isum("source_net_reference_explicit_demand_hit_count") / max(1, isum("source_net_emitted_demand_count"))
        ),
        "source_net_duplicate_sink_count": isum("source_net_duplicate_sink_count"),
        "source_net_duplicate_source_group_count": isum("source_net_duplicate_source_group_count"),
        "source_net_invalid_source_pin_count": isum("source_net_invalid_source_pin_count"),
        "source_net_invalid_load_pin_count": isum("source_net_invalid_load_pin_count"),
        "source_net_source_demand_mismatch_count": isum("source_net_source_demand_mismatch_count"),
        "source_net_forced_load_group_end_count": isum("source_net_forced_load_group_end_count"),
        "source_net_exhausted_source_group_close_count": isum("source_net_exhausted_source_group_close_count"),
        "source_net_malformed_load_entry_count": isum("source_net_malformed_load_entry_count"),
        "output_completed_graph_count": int(sum(1 for r in rows if str(r.get("output_completed_graph_path", "")))),
        "candidate_verilog_count": int(sum(1 for r in rows if str(r.get("candidate_verilog_path", "")))),
        "strict_export_pass_count": int(sum(1 for r in rows if bool(r.get("strict_export_pass", False)))),
        "strict_export_scope": "fixed_SINK_DEMAND prefix decoded graph; not a raw SOURCE_NET suffix validity claim",
        "endpoint_collision_count": isum("endpoint_collision"),
        "strict_missing_required_input_count": isum("strict_missing_required_input"),
        "post_pool_missing_required_input_count": isum("post_pool_missing_required_input"),
        "input_pool_connection_count": isum("input_pool_connections"),
        "exporter_missing_input_pin_count": isum("exporter_missing_input_pins"),
        "output_completion_created_net_count": isum("output_completion_created_nets"),
        "yosys_read_pass_count": int(sum(1 for r in rows if bool(r.get("yosys_read", False)))),
        "yosys_check_pass_count": int(sum(1 for r in rows if bool(r.get("yosys_check", False)))),
        "yosys_synth_pass_count": int(sum(1 for r in rows if bool(r.get("yosys_synth", False)))),
        "yosys_scope": "candidate.v exported from fixed_SINK_DEMAND prefix decoded graph",
        "masked_invalid_cell_tokens": isum("masked_invalid_cell_tokens"),
        "masked_invalid_pin_tokens": isum("masked_invalid_pin_tokens"),
        "masked_invalid_demand_tokens": isum("masked_invalid_demand_tokens"),
        "masked_duplicate_sink_tokens": isum("masked_duplicate_sink_tokens"),
        "masked_duplicate_source_tokens": isum("masked_duplicate_source_tokens"),
        "eos_tokens_blocked_by_remaining_internal_demand": isum("eos_tokens_blocked_by_remaining_internal_demand"),
        "fallback_to_unmasked_count": 0,
        "budget_exceeded_count": isum("budget_exceeded_count"),
        "section_force_close_count": 0,
        "source_net_force_closed_count": isum("source_net_force_closed"),
        "source_net_complete_rate": avg("source_net_complete"),
        "source_net_completion_ratio": avg("source_net_completion_ratio"),
        "source_net_records_generated": isum("source_net_records_generated"),
        "source_net_records_expected": isum("source_net_records_expected"),
        "topology_collapse_rate": avg("topology_collapse"),
        "graph_distribution_reference_scope": "V5 tokenizer round-trip reference, not raw PyG graph edges",
        "fanout_metric_scope": "all_sources_includes_zero_fanout_cell_output_candidates; reference is V5 round-trip; this is not free SOURCE_NET quality",
        "fanout_tv": fanout_tv,
        "derived_fanout_tv_all_sources": fanout_tv,
        "derived_fanout_tv_used_sources": fanout_tv_used,
        "fanout_generated": {k: int(gen_fanout.get(k, 0)) for k in FANOUT_KEYS if gen_fanout.get(k, 0)},
        "fanout_real": {k: int(real_fanout.get(k, 0)) for k in FANOUT_KEYS if real_fanout.get(k, 0)},
        "derived_fanout_generated_all_sources": {k: int(gen_fanout.get(k, 0)) for k in FANOUT_KEYS if gen_fanout.get(k, 0)},
        "derived_fanout_real_all_sources": {k: int(real_fanout.get(k, 0)) for k in FANOUT_KEYS if real_fanout.get(k, 0)},
        "derived_fanout_generated_used_sources": {k: int(gen_fanout_used.get(k, 0)) for k in FANOUT_KEYS if gen_fanout_used.get(k, 0)},
        "derived_fanout_real_used_sources": {k: int(real_fanout_used.get(k, 0)) for k in FANOUT_KEYS if real_fanout_used.get(k, 0)},
        "generated_17_plus": int(gen_fanout.get("17_PLUS", 0)),
        "real_17_plus": int(real_fanout.get("17_PLUS", 0)),
        "derived_17_plus_generated": int(gen_fanout.get("17_PLUS", 0)),
        "derived_17_plus_generated_used_sources": int(gen_fanout_used.get("17_PLUS", 0)),
        "derived_source_net_count": isum("derived_source_net_count"),
        "derived_source_net_complete": True,
        "cell_output_source_17_plus_count": isum("cell_output_source_17_plus_count"),
        "derived_cell_output_source_17_plus_count": isum("cell_output_source_17_plus_count"),
        "boundary_source_17_plus_count": isum("boundary_source_17_plus_count"),
        "derived_boundary_source_17_plus_count": isum("boundary_source_17_plus_count"),
        "repeated_source_signature": isum("repeated_source_signature"),
        "derived_repeated_source_signature": isum("repeated_source_signature"),
        "boundary_source_ratio": avg("boundary_source_ratio"),
        "cell_output_source_ratio": avg("cell_output_source_ratio"),
        "fallback_count": isum("fallback_count"),
        "INPUT_POOL_count": isum("INPUT_POOL_count"),
        "PRIVATE_OUT_count": isum("PRIVATE_OUT_count"),
        "dual_view_consistency": "not_evaluated: SINK_DEMAND is fixed prefix and SOURCE_NET is auxiliary bounded view",
        "source_final_fanout_bucket_consistency": "not_available_for_free_generated_decode",
        "v41_baseline_fanout_tv_range": "0.54~0.57",
        "v41_baseline_required_input_coverage": 0.9998497596153846,
        "pass_stage2_entry_gate": bool(pass_32_gate),
        "pass_stage2_entry_gate_scope": "decoder_stability_only; not approval for netlist/Verilog or synthesis evaluation",
        "formal_1024_decoder_stability_eligible": bool(pass_32_gate),
        "formal_1024_netlist_evaluation_eligible": False,
        "formal_1024_netlist_evaluation_blockers": [
            "SINK_DEMAND is a fixed reference prefix in this experiment",
            "SOURCE_NET high-fanout records are bounded previews rather than complete connectivity",
        ] + ([] if bool(args.materialize_export) else [
            "decoded_graph_with_stubs_output_completed.pt, Verilog export, and strict/Yosys validation are not run",
        ]),
        "strict_source_net_auxiliary_only": True,
        "stop_gate_triggered": bool(hard_fail),
        "cap32_allowed": False,
        "cap32_assembly_ran": False,
        "skeleton": "no-go",
        "kahypar": "not_run",
        "projection_exporter": "not_modified",
    }
    write_csv(out / "source_net_v5_generated_smoke_samples.csv", rows)
    write_json(out / "source_net_v5_generated_smoke.json", report)
    if bool(args.v6_practical_constrained_source_net):
        write_csv(out / "constrained_decode_smoke32_summary.csv", rows)
        write_json(out / "constrained_decode_smoke32_summary.json", report)
        failure_counter = Counter()
        for r in rows:
            if bool(r.get("generation_dead_end", False)):
                failure_counter["generation_dead_end"] += 1
            if int(r.get("source_net_invalid_source_pin_count", 0) or 0):
                failure_counter["source_net_invalid_source_pin"] += int(r.get("source_net_invalid_source_pin_count", 0) or 0)
            if int(r.get("source_net_invalid_load_pin_count", 0) or 0):
                failure_counter["source_net_invalid_load_pin"] += int(r.get("source_net_invalid_load_pin_count", 0) or 0)
            if int(r.get("source_net_duplicate_sink_count", 0) or 0):
                failure_counter["source_net_duplicate_sink"] += int(r.get("source_net_duplicate_sink_count", 0) or 0)
            if int(r.get("source_net_duplicate_source_group_count", 0) or 0):
                failure_counter["source_net_duplicate_source_group"] += int(r.get("source_net_duplicate_source_group_count", 0) or 0)
            if not bool(r.get("source_net_complete", False)):
                failure_counter["source_net_incomplete"] += 1
        if errors:
            failure_counter["exceptions"] += len(errors)
        failure_summary = {
            "failure_class_counts": dict(failure_counter),
            "errors": errors,
            "num_samples_completed": len(rows),
            "num_samples_requested": int(args.num_samples),
        }
        write_json(out / "failure_class_summary.json", failure_summary)
        write_md(out / "README_constrained_decoder.md", [
            "# V6 Practical Constrained SOURCE_NET Decoder",
            "",
            f"- constraint_level: `{report['constraint_level']}`",
            f"- level3_eos_gating_enabled: `{report['constraint_level3_eos_gating_enabled']}`",
            f"- forbid_duplicate_source_group: `{report['forbid_duplicate_source_group']}`",
            "- prefix: true V5 tokens through `SOURCE_NET_SECTION_BEGIN`; model samples only SOURCE_NET continuation.",
            "- no recovery/retry/force-close/success filtering is enabled.",
            f"- materialization/export enabled: `{bool(args.materialize_export)}`; Yosys enabled: `{bool(args.run_yosys)}`.",
        ])
        decision = [
            "# Phase 1 Constrained Decode Decision Report",
            "",
            "## Questions",
            f"1. invalid_source_pin / invalid_load_pin decrease: compare against baseline; explicit SOURCE_NET trace has `{report['source_net_invalid_source_pin_count']}` / `{report['source_net_invalid_load_pin_count']}`.",
            f"2. duplicate_sink / driver_conflict_nets decrease: explicit SOURCE_NET duplicate_sink=`{report['source_net_duplicate_sink_count']}`, driver_conflict_nets=`not_exported_in_smoke`.",
            f"3. SOURCE_NET explicit references: emitted=`{report['source_net_emitted_demand_count']}`, reference recall/precision=`{report['source_net_reference_explicit_demand_recall']}`/`{report['source_net_reference_explicit_demand_precision']}`. This is not full-demand coverage because high-fanout nets are previews.",
            f"4. generation_dead_end or max_new_tokens_hit increase: generation_dead_end=`{report['generation_dead_end_count']}`, max_new_tokens rows=`{sum(1 for r in rows if str(r.get('generation_stopped_reason')) == 'max_new_tokens')}`.",
            f"5. stable enough for larger evaluation: `{'Level 2 candidate' if report['generation_dead_end_count'] == 0 and invalid_src == 0 and invalid_sink == 0 else 'not yet'}`.",
            "6. next step: keep Level 1/2 masks for smoke comparison; Level 3 remains switch-only. Add materialization and a real dual-view audit before a netlist/synthesis-scale run.",
            "",
            "## Summary",
            f"- completed: `{len(rows)}/{int(args.num_samples)}`",
            f"- source_net_complete_rate: `{report['source_net_complete_rate']}`",
            f"- coverage_decode_valid_rate: `{report['coverage_decode_valid_rate']}`",
            f"- generation_dead_end_count: `{report['generation_dead_end_count']}`",
            f"- explicit SOURCE_NET invalid source/load pin: `{report['source_net_invalid_source_pin_count']}/{report['source_net_invalid_load_pin_count']}`",
            f"- explicit SOURCE_NET duplicate sink/source group: `{report['source_net_duplicate_sink_count']}/{report['source_net_duplicate_source_group_count']}`",
            f"- SOURCE_NET to SINK_DEMAND source mismatches: `{report['source_net_source_demand_mismatch_count']}`",
            f"- exact SOURCE_NET load-group terminators constrained: `{report['source_net_forced_load_group_end_count']}`",
            f"- malformed SOURCE_NET load entries: `{report['source_net_malformed_load_entry_count']}`",
            f"- output-completed/candidate/strict-pass: `{report['output_completed_graph_count']}/{report['candidate_verilog_count']}/{report['strict_export_pass_count']}`",
            f"- Yosys read/check/synth: `{report['yosys_read_pass_count']}/{report['yosys_check_pass_count']}/{report['yosys_synth_pass_count']}`",
            f"- masked invalid cell/pin/demand tokens: `{report['masked_invalid_cell_tokens']}/{report['masked_invalid_pin_tokens']}/{report['masked_invalid_demand_tokens']}`",
            f"- masked duplicate sink/source tokens: `{report['masked_duplicate_sink_tokens']}/{report['masked_duplicate_source_tokens']}`",
        ]
        write_md(out / "phase1_decision_report.md", decision)
    write_md(out / "source_net_v5_generated_smoke.md", [
        "# Source-Net V5 Generated Smoke",
        f"- run_scope: `{report['run_scope']}`",
        f"- v6_practical_constrained_source_net: `{report['v6_practical_constrained_source_net']}`",
        f"- constraint_level: `{report['constraint_level']}`",
        f"- ckpt: `{report['ckpt']}`",
        f"- num_samples: `{report['num_samples_completed']}/{report['num_samples_requested']}`",
        f"- grammar: `{report['grammar_type']}`, created=`{report['grammar_processor_created']}`",
        f"- decode_valid_rate: `{report['decode_valid_rate']}`",
        f"- strict_decode_valid_rate: `{report['strict_decode_valid_rate']}`",
        f"- coverage_decode_valid_rate: `{report['coverage_decode_valid_rate']}`",
        f"- required_input_coverage: `{report['required_input_coverage']}`",
        f"- invalid cell/sink/src pin: `{invalid_cell}/{invalid_sink}/{invalid_src}`",
        f"- duplicate_sink_count: `{duplicate}`",
        f"- topology_collapse_rate: `{report['topology_collapse_rate']}`",
        f"- fanout_tv: `{report['fanout_tv']}`",
        f"- fanout_generated: `{report['fanout_generated']}`",
        f"- fanout_real: `{report['fanout_real']}`",
        f"- 17_plus generated/real: `{report['generated_17_plus']}/{report['real_17_plus']}`",
        f"- fallback / INPUT_POOL / PRIVATE_OUT: `{report['fallback_count']} / {report['INPUT_POOL_count']} / {report['PRIVATE_OUT_count']}`",
        f"- source_net complete/force_closed/completion_ratio: `{report['source_net_complete_rate']} / {report['source_net_force_closed_count']} / {report['source_net_completion_ratio']}`",
        f"- pass_stage2_entry_gate: `{report['pass_stage2_entry_gate']}`",
        f"- stop_gate_triggered: `{report['stop_gate_triggered']}`",
        "- cap32/skeleton/KaHyPar/projection-exporter: `not_run/no-go/not_run/not_modified`",
    ])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
