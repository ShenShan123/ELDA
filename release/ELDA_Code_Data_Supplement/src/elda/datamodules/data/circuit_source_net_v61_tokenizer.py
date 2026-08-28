from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

import torch

from .batch_converter import BatchConverter
from .circuit_source_net_v61_schema import decode_v61, serialize_v61


class CircuitSourceNetV61Tokenizer:
    serializer_version = "source_net_v6_1_clean_compact_load_v1"
    tokenizer_version = "source_net_v6_1_clean_compact_load_v1"
    include_target_profile = False

    def __init__(
        self,
        *,
        dataset_names=None,
        max_length=24576,
        truncation_length=None,
        append_eos=True,
        net_id: int = 0,
        boundary_stub_id: int = -1,
        label_to_cell: Optional[Dict[int, str]] = None,
        pin_specs: Optional[Dict[str, object]] = None,
        **_kwargs,
    ):
        self.dataset_names = list(dataset_names or [])
        self.max_length = int(max_length)
        self.truncation_length = None
        self.append_eos = bool(append_eos)
        self.net_id = int(net_id)
        self.boundary_stub_id = int(boundary_stub_id)
        self.label_to_cell = {int(key): str(value) for key, value in dict(label_to_cell or {}).items()}
        self.cell_to_label = {value: key for key, value in self.label_to_cell.items()}
        self.pin_specs = dict(pin_specs or {})
        self.labeled_graph = False
        self.is_source_net_v61_tokenizer = True
        self.is_source_net_v5_tokenizer = False
        self.last_pointer_payload: dict = {}
        self.last_overlength = False
        self.max_num_nodes: Optional[int] = None
        self.pin_names: List[str] = []
        self.pin_name_to_id: Dict[str, int] = {}
        self.num_node_types = 0

        names = [
            "SOS", "EOS", "PAD", "NONE",
            "CELL_SECTION_BEGIN", "CELL_SECTION_END", "CELL_BEGIN", "CELL_END",
            "DEMAND_SECTION_BEGIN", "DEMAND_SECTION_END", "DEMAND_BEGIN", "DEMAND_END",
            "SOURCE_SECTION_BEGIN", "SOURCE_SECTION_END", "SOURCE_BEGIN", "SOURCE_END",
            "SOURCE_NET_SECTION_BEGIN", "SOURCE_NET_SECTION_END",
            "SOURCE_LOAD_BEGIN", "SOURCE_LOAD_END",
            "CELL_ID", "CELL_TYPE", "DEMAND_ID", "LOAD_CELL", "LOAD_PIN",
            "SOURCE_ID", "SOURCE_KIND", "SOURCE_CELL", "SOURCE_PIN", "BOUNDARY_PIN_ID",
            "SOURCE_ROLE", "FANOUT_BUCKET", "MAX_FANOUT", "USED_FANOUT",
            "REMAINING_FANOUT", "COUNT_TOKEN",
            "CELL_OUTPUT", "BOUNDARY_SOURCE", "CONSTANT_SOURCE",
            "OUTPUT_PIN", "BOUNDARY_PIN", "CONST_PIN",
            "FANOUT_0", "FANOUT_1", "FANOUT_2_4", "FANOUT_5_8",
            "FANOUT_9_16", "FANOUT_17_PLUS",
        ]
        self.special_toks = names
        for index, name in enumerate(names):
            setattr(self, name.lower(), index)
        self.sos = self.special_toks.index("SOS")
        self.eos = self.special_toks.index("EOS")
        self.pad = self.special_toks.index("PAD")
        self.idx_offset = len(self.special_toks)
        self.gate_offset = None
        self.source_offset = None
        self.demand_offset = None
        self.count_offset = None
        self.pin_offset = None
        self.node_type_offset = None

    def batch_converter(self):
        return BatchConverter(self, None)

    def __call__(self, data):
        return self.tokenize(data)

    def set_num_nodes(self, max_num_nodes):
        value = int(max_num_nodes)
        self.max_num_nodes = value if self.max_num_nodes is None else max(self.max_num_nodes, value)
        self._refresh_offsets()

    def set_pin_name_vocab(self, pin_names: List[str]) -> None:
        self.pin_names = sorted(set(str(name) for name in pin_names) | {"__BOUNDARY__", "__UNKNOWN__"})
        self.pin_name_to_id = {name: index for index, name in enumerate(self.pin_names)}
        self._refresh_offsets()

    def set_num_node_and_edge_types(self, num_node_types=0, num_edge_types=0):
        self.num_node_types = int(num_node_types)
        self._refresh_offsets()

    def _refresh_offsets(self):
        if self.max_num_nodes is None:
            return
        self.gate_offset = self.idx_offset
        self.source_offset = self.gate_offset + self.max_num_nodes
        self.demand_offset = self.source_offset + self.max_num_nodes
        self.count_offset = self.demand_offset + self.max_num_nodes
        self.pin_offset = self.count_offset + self.max_num_nodes + 1
        self.node_type_offset = self.pin_offset + len(self.pin_names)

    def __len__(self):
        if self.node_type_offset is None:
            raise ValueError("V6.1 tokenizer is not initialized")
        return self.node_type_offset + max(1, self.num_node_types)

    def _bounded(self, value: int, name: str) -> int:
        value = int(value)
        if self.max_num_nodes is None or not (0 <= value <= self.max_num_nodes):
            raise ValueError(f"{name}={value} exceeds V6.1 pointer/count range")
        return value

    def _gate(self, value: int) -> int:
        return self.gate_offset + self._bounded(value, "gate")

    def _source(self, value: int) -> int:
        return self.source_offset + self._bounded(value, "source")

    def _demand(self, value: int) -> int:
        return self.demand_offset + self._bounded(value, "demand")

    def _count(self, value: int) -> int:
        return self.count_offset + self._bounded(value, "count")

    def _pin(self, value: str) -> int:
        return self.pin_offset + self.pin_name_to_id.get(str(value), self.pin_name_to_id["__UNKNOWN__"])

    def _cell_type(self, value: str) -> int:
        return self.node_type_offset + int(self.cell_to_label[value])

    def _static(self, value: str) -> int:
        aliases = {
            "CELL_OUTPUT": "CELL_OUTPUT",
            "BOUNDARY_SOURCE": "BOUNDARY_SOURCE",
            "PI_CONST_SOURCE": "CONSTANT_SOURCE",
            "OUTPUT_PIN": "OUTPUT_PIN",
            "BOUNDARY_PIN": "BOUNDARY_PIN",
            "CONST_PIN": "CONST_PIN",
        }
        name = aliases.get(str(value), str(value))
        try:
            return self.special_toks.index(name)
        except ValueError as exc:
            raise ValueError(f"unsupported V6.1 static token: {value}") from exc

    def _pointer_value(self, token: int, offset: int, kind: str) -> int:
        value = int(token) - int(offset)
        if self.max_num_nodes is None or not (0 <= value <= self.max_num_nodes):
            raise ValueError(f"invalid {kind} pointer token: {token}")
        return value

    def _pin_name(self, token: int) -> str:
        value = int(token) - int(self.pin_offset)
        if not (0 <= value < len(self.pin_names)):
            raise ValueError(f"invalid pin token: {token}")
        return self.pin_names[value]

    def _cell_name(self, token: int) -> str:
        value = int(token) - int(self.node_type_offset)
        if value not in self.label_to_cell:
            raise ValueError(f"invalid cell-type token: {token}")
        return self.label_to_cell[value]

    def parse_tokens(self, sequence) -> tuple[dict, dict]:
        """Strictly parse one production compact-load token sequence."""
        if torch.is_tensor(sequence):
            tokens = [int(value) for value in sequence.detach().cpu().reshape(-1).tolist()]
        else:
            tokens = [int(value) for value in sequence]
        while tokens and tokens[-1] == self.pad:
            tokens.pop()

        cursor = 0

        def take(expected=None):
            nonlocal cursor
            if cursor >= len(tokens):
                raise ValueError(f"unexpected end of sequence at token {cursor}")
            value = tokens[cursor]
            cursor += 1
            if expected is not None and value != int(expected):
                raise ValueError(
                    f"token {cursor - 1}: expected {int(expected)}, got {value}"
                )
            return value

        def gate_ptr():
            return self._pointer_value(take(), self.gate_offset, "cell")

        def source_ptr():
            return self._pointer_value(take(), self.source_offset, "source")

        def demand_ptr():
            return self._pointer_value(take(), self.demand_offset, "demand")

        def count_value():
            return self._pointer_value(take(), self.count_offset, "count")

        take(self.sos)
        control_profile = None
        if self.include_target_profile:
            take(self.target_profile_begin)
            take(self.profile_mode)
            mode = self.special_toks[take()]
            if mode == "PROFILE_FREE":
                control_profile = {"profile_mode": mode}
            elif mode == "PROFILE_COARSE":
                fields = (
                    ("cell_count_bin", self.cell_count_bin),
                    ("demand_count_bin", self.demand_count_bin),
                    ("source_count_bin", self.source_count_bin),
                    ("seq_ratio_bin", self.seq_ratio_bin),
                    ("pin_complexity_bin", self.pin_complexity_bin),
                )
                control_profile = {"profile_mode": mode}
                for key, tag in fields:
                    take(tag)
                    control_profile[key] = self.special_toks[take()]
                take(self.complex_cell_flags)
                for key, tag in (
                    ("rare_cell_enriched", self.rare_cell_enriched),
                    ("has_mux", self.has_mux),
                    ("has_aoi_oai", self.has_aoi_oai),
                    ("has_multi_output_cell", self.has_multi_output_cell),
                ):
                    take(tag)
                    value = self.special_toks[take()]
                    if value not in {"BOOL_FALSE", "BOOL_TRUE"}:
                        raise ValueError(f"invalid V6.2 boolean profile token: {value}")
                    control_profile[key] = value == "BOOL_TRUE"
            else:
                raise ValueError(f"invalid V6.2 profile mode: {mode}")
            take(self.target_profile_end)
        take(self.cell_section_begin)
        cells = []
        while cursor < len(tokens) and tokens[cursor] == self.cell_begin:
            take(self.cell_begin)
            take(self.cell_id)
            cell_id = gate_ptr()
            take(self.cell_type)
            cell_type = self._cell_name(take())
            take(self.cell_end)
            if cell_id != len(cells):
                raise ValueError(f"non-canonical cell pointer {cell_id}, expected {len(cells)}")
            cells.append({
                "cell_id": f"CELL_{cell_id}",
                "cell_dense_id": cell_id,
                "cell_type": cell_type,
            })
        take(self.cell_section_end)

        take(self.demand_section_begin)
        demands = []
        while cursor < len(tokens) and tokens[cursor] == self.demand_begin:
            take(self.demand_begin)
            take(self.demand_id)
            demand_id = demand_ptr()
            take(self.load_cell)
            load_cell = gate_ptr()
            take(self.load_pin)
            load_pin = self._pin_name(take())
            take(self.demand_end)
            if demand_id != len(demands):
                raise ValueError(
                    f"non-canonical demand pointer {demand_id}, expected {len(demands)}"
                )
            if load_cell >= len(cells):
                raise ValueError(f"demand {demand_id} references unknown cell {load_cell}")
            demands.append({
                "demand_id": f"DEMAND_{demand_id}",
                "load_cell_id": f"CELL_{load_cell}",
                "load_pin": load_pin,
            })
        take(self.demand_section_end)

        take(self.source_section_begin)
        sources = []
        static_names = {index: name for index, name in enumerate(self.special_toks)}
        while cursor < len(tokens) and tokens[cursor] == self.source_begin:
            take(self.source_begin)
            take(self.source_id)
            source_id = source_ptr()
            take(self.source_kind)
            source_kind = static_names.get(take())
            if source_kind not in {"CELL_OUTPUT", "BOUNDARY_SOURCE", "CONSTANT_SOURCE"}:
                raise ValueError(f"invalid source kind for source {source_id}: {source_kind}")
            take(self.source_cell)
            source_cell_token = take()
            source_cell = None
            if source_cell_token != self.none:
                source_cell = self._pointer_value(
                    source_cell_token, self.gate_offset, "source-cell"
                )
                if source_cell >= len(cells):
                    raise ValueError(
                        f"source {source_id} references unknown cell {source_cell}"
                    )
            take(self.source_pin)
            source_pin = self._pin_name(take())
            take(self.boundary_pin_id)
            boundary_token = take()
            boundary_pin = None
            if boundary_token != self.none:
                boundary_pin = self._pointer_value(
                    boundary_token, self.source_offset, "boundary-source"
                )
            take(self.source_role)
            source_role = static_names.get(take())
            if source_role not in {"OUTPUT_PIN", "BOUNDARY_PIN", "CONST_PIN"}:
                raise ValueError(f"invalid source role for source {source_id}: {source_role}")
            take(self.fanout_bucket)
            fanout_bucket = static_names.get(take())
            if fanout_bucket not in {
                "FANOUT_0", "FANOUT_1", "FANOUT_2_4", "FANOUT_5_8",
                "FANOUT_9_16", "FANOUT_17_PLUS",
            }:
                raise ValueError(f"invalid fanout bucket for source {source_id}")
            take(self.max_fanout)
            max_fanout = count_value()
            take(self.used_fanout)
            used_fanout = count_value()
            take(self.remaining_fanout)
            remaining_fanout = count_value()
            take(self.source_end)
            if source_id != len(sources):
                raise ValueError(
                    f"non-canonical source pointer {source_id}, expected {len(sources)}"
                )
            parsed_source = {
                "source_id": f"SOURCE_{source_id}",
                "source_kind": source_kind,
                "source_cell_id": None if source_cell is None else f"CELL_{source_cell}",
                "source_pin": source_pin,
                "boundary_pin_id": (
                    None if boundary_pin is None else f"BOUNDARY_PIN_{boundary_pin}"
                ),
                "source_role": source_role,
                "fanout_bucket": fanout_bucket,
                "max_fanout": max_fanout,
                "used_fanout": used_fanout,
                "remaining_fanout": remaining_fanout,
                "endpoint_type": (
                    "BOUNDARY_STUB" if source_kind == "BOUNDARY_SOURCE" else "NET"
                ),
            }
            sources.append(parsed_source)
        take(self.source_section_end)

        take(self.source_net_section_begin)
        source_nets = []
        seen_sources = set()
        emitted_demands = []
        same_cell_reuse = 0
        while cursor < len(tokens) and tokens[cursor] == self.source_load_begin:
            take(self.source_load_begin)
            source_id = source_ptr()
            if source_id >= len(sources):
                raise ValueError(f"source load references unknown source {source_id}")
            if source_id in seen_sources:
                raise ValueError(f"duplicate source load record for source {source_id}")
            seen_sources.add(source_id)
            take(self.count_token)
            demand_count = count_value()
            demand_ids = [demand_ptr() for _ in range(demand_count)]
            take(self.source_load_end)
            unknown = [value for value in demand_ids if value >= len(demands)]
            if unknown:
                raise ValueError(f"source {source_id} references unknown demands {unknown[:8]}")
            load_cells = [demands[value]["load_cell_id"] for value in demand_ids]
            same_cell_reuse += sum(
                max(0, count - 1) for count in Counter(load_cells).values()
            )
            emitted_demands.extend(demand_ids)
            source_nets.append({
                "source_id": f"SOURCE_{source_id}",
                "fanout": len(demand_ids),
                "chunks": [{
                    "source_id": f"SOURCE_{source_id}",
                    "chunk_id": 0,
                    "demand_ids": [f"DEMAND_{value}" for value in demand_ids],
                }],
            })
        take(self.source_net_section_end)
        take(self.eos)
        if cursor != len(tokens):
            raise ValueError(f"tokens remain after EOS: {len(tokens) - cursor}")

        emitted_counts = Counter(emitted_demands)
        expected_demands = set(range(len(demands)))
        actual_demands = set(emitted_demands)
        source_fanouts = {
            int(row["source_id"].rsplit("_", 1)[-1]): int(row["fanout"])
            for row in source_nets
        }
        budget_violations = sum(
            source_fanouts.get(index, 0) > int(source["max_fanout"])
            for index, source in enumerate(sources)
        )
        source_load_full_coverage = (
            seen_sources == set(range(len(sources)))
            and actual_demands == expected_demands
            and all(count == 1 for count in emitted_counts.values())
        )
        payload = {
            "schema_version": self.serializer_version,
            "cells": cells,
            "demands": demands,
            "sources": sources,
            "source_nets": source_nets,
        }
        if self.include_target_profile:
            payload["control_profile"] = control_profile
        realized_profile = (
            self.realized_control_profile(payload)
            if self.include_target_profile else None
        )
        profile_consistent = (
            not self.include_target_profile
            or control_profile.get("profile_mode") == "PROFILE_FREE"
            or self.control_profile_matches(payload, control_profile)
        )
        report = {
            "token_decode_valid": True,
            "section_complete": True,
            "cell_table_valid": bool(cells),
            "demand_table_valid": bool(demands),
            "source_table_valid": bool(sources),
            "source_load_full_coverage": source_load_full_coverage,
            "demand_exactly_once": (
                actual_demands == expected_demands
                and all(count == 1 for count in emitted_counts.values())
            ),
            "source_budget_violation": budget_violations,
            "same_cell_same_net_reuse": same_cell_reuse,
            "EOS_validity": True,
            "cell_count": len(cells),
            "demand_count": len(demands),
            "source_count": len(sources),
            "source_load_count": len(source_nets),
            "target_profile_consistent": profile_consistent,
            "control_profile_adherent": profile_consistent,
            **({
                "requested_profile": control_profile,
                "realized_profile": realized_profile,
                "profile_field_adherence": {
                    key: realized_profile.get(key) == value
                    for key, value in control_profile.items()
                    if key != "profile_mode"
                },
            } if self.include_target_profile else {}),
        }
        return payload, report

    def audit_tokens(self, sequence) -> dict:
        try:
            _payload, report = self.parse_tokens(sequence)
            return report
        except Exception as exc:
            return {
                "token_decode_valid": False,
                "section_complete": False,
                "cell_table_valid": False,
                "demand_table_valid": False,
                "source_table_valid": False,
                "source_load_full_coverage": False,
                "demand_exactly_once": False,
                "source_budget_violation": None,
                "same_cell_same_net_reuse": None,
                "EOS_validity": False,
                "error": str(exc),
            }

    def decode(self, sequence):
        payload, report = self.parse_tokens(sequence)
        if not report["source_load_full_coverage"]:
            raise ValueError("V6.1 decode requires full source-load coverage")
        if report["source_budget_violation"]:
            raise ValueError("V6.1 decode rejected source budget violation")
        if report["same_cell_same_net_reuse"]:
            raise ValueError("V6.1 decode rejected same-cell same-net input reuse")
        return decode_v61(
            payload,
            net_id=self.net_id,
            boundary_stub_id=self.boundary_stub_id,
            cell_to_label=self.cell_to_label,
        )

    def _serialize_payload(self, data):
        return serialize_v61(
            data,
            net_id=self.net_id,
            boundary_stub_id=self.boundary_stub_id,
            label_to_cell=self.label_to_cell,
            pin_specs=self.pin_specs,
            chunk_size=max(1, self.max_num_nodes),
        )

    @staticmethod
    def _numeric_id(value: str) -> int:
        return int(str(value).rsplit("_", 1)[-1])

    def tokenize(self, data):
        if self.node_type_offset is None:
            raise ValueError("V6.1 tokenizer offsets are not initialized")
        payload = self._serialize_payload(data)
        source_index = {
            source["source_id"]: index for index, source in enumerate(payload["sources"])
        }
        demand_index = {
            demand["demand_id"]: index for index, demand in enumerate(payload["demands"])
        }
        tokens = [self.sos]
        if self.include_target_profile:
            profile = payload["control_profile"]
            tokens.extend([self.target_profile_begin, self.profile_mode,
                           self._static(profile["profile_mode"])])
            if profile["profile_mode"] == "PROFILE_COARSE":
                for key, tag in (
                    ("cell_count_bin", self.cell_count_bin),
                    ("demand_count_bin", self.demand_count_bin),
                    ("source_count_bin", self.source_count_bin),
                    ("seq_ratio_bin", self.seq_ratio_bin),
                    ("pin_complexity_bin", self.pin_complexity_bin),
                ):
                    tokens.extend([tag, self._static(profile[key])])
                tokens.append(self.complex_cell_flags)
                for key, tag in (
                    ("rare_cell_enriched", self.rare_cell_enriched),
                    ("has_mux", self.has_mux),
                    ("has_aoi_oai", self.has_aoi_oai),
                    ("has_multi_output_cell", self.has_multi_output_cell),
                ):
                    tokens.extend([tag, self._static("BOOL_TRUE" if profile[key] else "BOOL_FALSE")])
            tokens.append(self.target_profile_end)
        tokens.append(self.cell_section_begin)
        for cell in payload["cells"]:
            tokens.extend([
                self.cell_begin, self.cell_id, self._gate(cell["cell_dense_id"]),
                self.cell_type, self._cell_type(cell["cell_type"]), self.cell_end,
            ])
        tokens.extend([self.cell_section_end, self.demand_section_begin])
        for demand in payload["demands"]:
            tokens.extend([
                self.demand_begin, self.demand_id, self._demand(demand_index[demand["demand_id"]]),
                self.load_cell, self._gate(self._numeric_id(demand["load_cell_id"])),
                self.load_pin, self._pin(demand["load_pin"]), self.demand_end,
            ])
        tokens.extend([self.demand_section_end, self.source_section_begin])
        for source in payload["sources"]:
            source_ptr = self._source(source_index[source["source_id"]])
            source_cell = (
                self.none
                if source["source_cell_id"] is None
                else self._gate(self._numeric_id(source["source_cell_id"]))
            )
            boundary_ptr = source_ptr if source["boundary_pin_id"] else self.none
            source_tokens = [
                self.source_begin, self.source_id, source_ptr,
                self.source_kind, self._static(source["source_kind"]),
                self.source_cell, source_cell,
                self.source_pin, self._pin(source["source_pin"]),
                self.boundary_pin_id, boundary_ptr,
                self.source_role, self._static(source["source_role"]),
            ]
            source_tokens.extend([
                self.fanout_bucket, self._static(source["fanout_bucket"]),
                self.max_fanout, self._count(source["max_fanout"]),
                self.used_fanout, self._count(source["used_fanout"]),
                self.remaining_fanout, self._count(source["remaining_fanout"]),
                self.source_end,
            ])
            tokens.extend(source_tokens)
        tokens.extend([self.source_section_end, self.source_net_section_begin])
        for source_net in payload["source_nets"]:
            demand_ids = [
                demand_id
                for chunk in source_net["chunks"]
                for demand_id in chunk["demand_ids"]
            ]
            tokens.extend([
                self.source_load_begin,
                self._source(source_index[source_net["source_id"]]),
                self.count_token, self._count(len(demand_ids)),
            ])
            tokens.extend(self._demand(demand_index[demand_id]) for demand_id in demand_ids)
            tokens.append(self.source_load_end)
        tokens.append(self.source_net_section_end)
        if self.append_eos:
            tokens.append(self.eos)
        sequence = torch.tensor(tokens, dtype=torch.long)
        self.last_overlength = bool(self.max_length > 0 and int(sequence.numel()) > self.max_length)
        if self.last_overlength:
            raise RuntimeError(
                f"V6.1 no-silent-truncation violation: length={int(sequence.numel())} "
                f"exceeds max_length={self.max_length}"
            )
        self.last_pointer_payload = {
            "schema_version": payload["schema_version"],
            "object_table": {
                "cell_count": len(payload["cells"]),
                "demand_count": len(payload["demands"]),
                "source_count": len(payload["sources"]),
            },
            **({"control_profile": payload["control_profile"]}
               if self.include_target_profile else {}),
            "invariants": payload["invariants"],
            "sequence_length": int(sequence.numel()),
        }
        return sequence
