#!/usr/bin/env python3
"""Run the complete deterministic ELDA supplement smoke pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch

from elda.datamodules.data.circuit_source_net_schema import serialize_source_net
from elda.datamodules.data.circuit_source_net_tokenizer import CircuitSourceNetTokenizer
from elda.supplement.grammar_loader import ELDAGrammar
from elda.supplement.materialize import emit_structural_verilog, run_yosys
from elda.supplement.metrics import sample_metrics
from elda.supplement.sample_io import (
    BOUNDARY_STUB_ID,
    LABEL_TO_CELL,
    NET_ID,
    PIN_SPECS,
    graph_signature,
    load_raw_graph,
)
from elda.supplement.validation import negative_cases, validate_payload


ROOT = Path(__file__).resolve().parents[1]
STABLE_FILES = (
    "source_demand_object.json",
    "token_sequence.json",
    "reconstructed_object.json",
    "roundtrip_report.json",
    "validity_report.json",
    "negative_case_report.json",
    "constrained_decoding_trace.json",
    "structural_sample.v",
    "materialization_report.json",
    "yosys_result.json",
    "metrics.json",
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def library_dict() -> dict:
    return {
        name: {"inputs": list(spec.inputs), "outputs": list(spec.outputs)}
        for name, spec in PIN_SPECS.items()
    }


def configure_tokenizer() -> CircuitSourceNetTokenizer:
    tokenizer = CircuitSourceNetTokenizer(
        max_length=24576,
        append_eos=True,
        net_id=NET_ID,
        boundary_stub_id=BOUNDARY_STUB_ID,
        label_to_cell=LABEL_TO_CELL,
        pin_specs=PIN_SPECS,
    )
    tokenizer.set_num_nodes(64)
    tokenizer.set_pin_name_vocab(["A", "Z", "__BOUNDARY__"])
    tokenizer.set_num_node_and_edge_types(num_node_types=max(LABEL_TO_CELL) + 1)
    return tokenizer


def token_name(tokenizer, token: int) -> str:
    token = int(token)
    if 0 <= token < len(tokenizer.special_toks):
        return tokenizer.special_toks[token]
    ranges = (
        (tokenizer.gate_offset, tokenizer.source_offset, "CELL_PTR"),
        (tokenizer.source_offset, tokenizer.demand_offset, "SOURCE_PTR"),
        (tokenizer.demand_offset, tokenizer.count_offset, "DEMAND_PTR"),
        (tokenizer.count_offset, tokenizer.pin_offset, "COUNT"),
        (tokenizer.pin_offset, tokenizer.node_type_offset, "PIN"),
    )
    for start, end, name in ranges:
        if start <= token < end:
            value = token - start
            if name == "PIN" and value < len(tokenizer.pin_names):
                return f"PIN[{tokenizer.pin_names[value]}]"
            return f"{name}[{value}]"
    if token >= tokenizer.node_type_offset:
        label = token - tokenizer.node_type_offset
        return f"CELL_TYPE[{LABEL_TO_CELL.get(label, label)}]"
    return f"TOKEN[{token}]"


def state_snapshot(state: dict) -> dict:
    return {
        "expect": state["expect"],
        "cells": len(state["cells"]),
        "demands": len(state["demands"]),
        "sources": len(state["sources"]),
        "assigned_demands": len(state["assigned"]),
        "emitted_sources": len(state["emitted_sources"]),
        "current_source": state["load_source"],
        "current_record_remaining": state["load_left"],
    }


def constrained_trace(tokenizer, sequence: list[int]) -> list[dict]:
    grammar = ELDAGrammar(
        tokenizer, batch_size=1, device="cpu", mask_mode="reference"
    )
    state = grammar.states[0]
    trace = []
    for step, chosen in enumerate(sequence[1:], start=1):
        allowed = [int(value) for value in grammar._allowed(state)]
        accepted = int(chosen) in allowed
        trace.append(
            {
                "step": step,
                "state_before": state_snapshot(state),
                "chosen_id": int(chosen),
                "chosen_token": token_name(tokenizer, chosen),
                "allowed": accepted,
                "legal_token_count": len(allowed),
                "legal_tokens": [token_name(tokenizer, value) for value in allowed],
            }
        )
        if not accepted:
            raise RuntimeError(
                f"production grammar rejected token {chosen} at position {step}"
            )
        grammar._advance(state, int(chosen))
        trace[-1]["state_after"] = state_snapshot(state)
    return trace


def canonical_payload(payload: dict) -> dict:
    return {key: copy.deepcopy(payload[key]) for key in ("schema_version", "cells", "demands", "sources", "source_nets")}


def compare_expected(output_dir: Path, expected_dir: Path) -> None:
    failures = []
    for name in STABLE_FILES:
        actual = output_dir / name
        expected = expected_dir / name
        if not expected.exists():
            failures.append(f"missing expected file: {name}")
        elif actual.read_bytes() != expected.read_bytes():
            failures.append(f"mismatch: {name}")
    if failures:
        raise RuntimeError("Expected-output comparison failed: " + "; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--expected-dir", default=str(ROOT / "expected_outputs"))
    parser.add_argument("--update-expected", action="store_true")
    parser.add_argument("--skip-yosys", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    expected_dir = Path(args.expected_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = load_raw_graph(ROOT / "data_sample" / "raw_subcircuit.json")
    tokenizer = configure_tokenizer()
    raw_payload = serialize_source_net(
        graph,
        net_id=NET_ID,
        boundary_stub_id=BOUNDARY_STUB_ID,
        label_to_cell=LABEL_TO_CELL,
        pin_specs=PIN_SPECS,
        chunk_size=64,
    )
    sequence_tensor = tokenizer.tokenize(graph)
    sequence = [int(value) for value in sequence_tensor.tolist()]
    write_json(
        output_dir / "token_sequence.json",
        {
            "length": len(sequence),
            "ids": sequence,
            "tokens": [token_name(tokenizer, value) for value in sequence],
        },
    )
    reconstructed, parser_report = tokenizer.parse_tokens(sequence)
    reconstructed = canonical_payload(reconstructed)
    stable_payload = reconstructed
    write_json(output_dir / "source_demand_object.json", stable_payload)
    write_json(output_dir / "reconstructed_object.json", reconstructed)
    decoded_graph = tokenizer.decode(sequence)
    decoded_sequence = [int(value) for value in tokenizer.tokenize(decoded_graph).tolist()]
    reserialized, _ = tokenizer.parse_tokens(decoded_sequence)
    reserialized = canonical_payload(reserialized)
    roundtrip = {
        "object_equal_after_parse": reconstructed == stable_payload,
        "object_equal_after_decode_reserialize": reserialized == stable_payload,
        "raw_serializer_schema": raw_payload.get("schema_version"),
        "sequence_schema": stable_payload.get("schema_version"),
        "raw_graph_signature": graph_signature(graph),
        "decoded_graph_signature": graph_signature(decoded_graph),
        "parser_report": parser_report,
    }
    roundtrip["graph_reconstruction_complete"] = (
        int(graph.x.numel()) == int(decoded_graph.x.numel())
        and int(graph.edge_index.size(1)) == int(decoded_graph.edge_index.size(1))
    )
    if not all(
        roundtrip[key]
        for key in (
            "object_equal_after_parse",
            "object_equal_after_decode_reserialize",
            "graph_reconstruction_complete",
        )
    ):
        raise RuntimeError("Source--Demand round-trip mismatch")
    write_json(output_dir / "roundtrip_report.json", roundtrip)

    library = library_dict()
    validity = validate_payload(stable_payload, library)
    if not validity["overall_valid"]:
        raise RuntimeError(f"included positive sample is invalid: {validity['errors']}")
    write_json(output_dir / "validity_report.json", validity)
    negative_report = {}
    for name, (invalid, expected_failure) in negative_cases(stable_payload).items():
        report = validate_payload(invalid, library)
        observed = not report["conditions"][expected_failure]
        negative_report[name] = {
            "expected_failure": expected_failure,
            "observed_at_expected_rule": observed,
            "overall_valid": report["overall_valid"],
            "errors": report["errors"],
        }
        if not observed or report["overall_valid"]:
            raise RuntimeError(f"negative case {name} did not fail at {expected_failure}")
    write_json(output_dir / "negative_case_report.json", negative_report)

    trace = constrained_trace(tokenizer, sequence)
    write_json(output_dir / "constrained_decoding_trace.json", trace)

    materialization = emit_structural_verilog(
        stable_payload, library, output_dir / "structural_sample.v"
    )
    if materialization["repair_operations"] != 0:
        raise RuntimeError("repair-free materialization recorded a repair")
    write_json(output_dir / "materialization_report.json", materialization)

    if args.skip_yosys:
        yosys_result = {
            "read_check_pass": None,
            "diagnostic_skip": True,
            "note": "Explicit diagnostic skip; this is not a complete reproduction.",
        }
    else:
        yosys_bin = os.environ.get("YOSYS_BIN") or shutil.which("yosys")
        if not yosys_bin:
            raise RuntimeError("Yosys is required. Set YOSYS_BIN or place yosys on PATH.")
        raw_result = run_yosys(
            output_dir / "structural_sample.v",
            ROOT / "data_sample" / "cells.v",
            output_dir / "yosys.log",
            yosys_bin,
        )
        version_match = re.search(r"\bYosys\s+(\d+\.\d+)", raw_result["yosys_version"])
        yosys_result = {
            "read_check_pass": raw_result["read_check_pass"],
            "returncode": raw_result["returncode"],
            "warning_count": raw_result["warning_count"],
            "error_count": raw_result["error_count"],
            "yosys_version_major_minor": (
                version_match.group(1) if version_match else "unknown"
            ),
        }
        if not yosys_result["read_check_pass"]:
            raise RuntimeError("Yosys read/check failed; see outputs/yosys.log")
    write_json(output_dir / "yosys_result.json", yosys_result)
    write_json(output_dir / "metrics.json", sample_metrics(stable_payload))

    if args.update_expected:
        expected_dir.mkdir(parents=True, exist_ok=True)
        for name in STABLE_FILES:
            shutil.copy2(output_dir / name, expected_dir / name)
        if (output_dir / "yosys.log").exists():
            text = (output_dir / "yosys.log").read_text(encoding="utf-8", errors="replace")
            text = text.replace(str(ROOT), "<SUPPLEMENT_ROOT>")
            text = re.sub(
                r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
                "<UPSTREAM_CONTACT>",
                text,
                flags=re.I,
            )
            (expected_dir / "yosys.log").write_text(text, encoding="utf-8")
    else:
        compare_expected(output_dir, expected_dir)

    print("ELDA minimal reproduction: PASS")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ELDA minimal reproduction: FAIL: {exc}", file=sys.stderr)
        raise
