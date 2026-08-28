from __future__ import annotations

from collections import Counter

import torch

from .circuit_source_net_tokenizer import CircuitSourceNetTokenizer
from .circuit_source_net_schema import decode_source_net, serialize_source_net, count_bucket


class _CircuitSourceNetAblationTokenizer(CircuitSourceNetTokenizer):
    """Independent, versioned token schemas for the R1-R4 representation ablations."""

    representation_variant = ""
    information_sufficient_for_strict_decode = True

    def _tokens_from_payload(self, payload) -> torch.Tensor:
        source_index = {
            source["source_id"]: index for index, source in enumerate(payload["sources"])
        }
        demand_index = {
            demand["demand_id"]: index for index, demand in enumerate(payload["demands"])
        }
        tokens = [self.sos, self.cell_section_begin]
        for cell in payload["cells"]:
            tokens.extend([
                self.cell_begin, self.cell_id, self._gate(cell["cell_dense_id"]),
                self.cell_type, self._cell_type(cell["cell_type"]), self.cell_end,
            ])
        tokens.extend([self.cell_section_end, self.demand_section_begin])
        for demand in payload["demands"]:
            tokens.extend([
                self.demand_begin, self.demand_id,
                self._demand(demand_index[demand["demand_id"]]),
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
            boundary_ptr = source_ptr if source.get("boundary_pin_id") else self.none
            tokens.extend([
                self.source_begin, self.source_id, source_ptr,
                self.source_kind, self._static(source["source_kind"]),
                self.source_cell, source_cell,
                self.source_pin, self._pin(source["source_pin"]),
                self.boundary_pin_id, boundary_ptr,
                self.source_role, self._static(source["source_role"]),
                self.fanout_bucket, self._static(source["fanout_bucket"]),
                self.max_fanout, self._count(source["max_fanout"]),
                self.used_fanout, self._count(source["used_fanout"]),
                self.remaining_fanout, self._count(source["remaining_fanout"]),
                self.source_end,
            ])
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
            tokens.extend(self._demand(demand_index[value]) for value in demand_ids)
            tokens.append(self.source_load_end)
        tokens.extend([self.source_net_section_end, self.eos])
        return torch.tensor(tokens, dtype=torch.long)

    def _strip_variant_fields(self, sequence: torch.Tensor) -> torch.Tensor:
        tokens = [int(value) for value in sequence.tolist()]
        if self.representation_variant == "r1_no_boundary_identity":
            for index, token in enumerate(tokens[:-1]):
                if token == self.boundary_pin_id:
                    tokens[index + 1] = self.none
        elif self.representation_variant == "r2_no_per_source_budget":
            budget_tags = {
                self.fanout_bucket, self.max_fanout,
                self.used_fanout, self.remaining_fanout,
            }
            result = []
            index = 0
            while index < len(tokens):
                if tokens[index] in budget_tags:
                    index += 2
                else:
                    result.append(tokens[index])
                    index += 1
            tokens = result
        elif self.representation_variant == "r3_no_full_load_assignment":
            begin = tokens.index(self.source_net_section_begin)
            end = tokens.index(self.source_net_section_end, begin + 1)
            tokens = tokens[:begin + 1] + tokens[end:]
        elif self.representation_variant == "r4_cell_level_demand":
            result = []
            index = 0
            in_demands = False
            while index < len(tokens):
                token = tokens[index]
                if token == self.demand_section_begin:
                    in_demands = True
                elif token == self.demand_section_end:
                    in_demands = False
                if in_demands and token == self.load_pin:
                    index += 2
                    continue
                result.append(token)
                index += 1
            tokens = result
        else:
            raise ValueError(f"unknown ELDA representation variant: {self.representation_variant}")
        return torch.tensor(tokens, dtype=sequence.dtype)

    def _inflate_for_r0_parser(self, sequence) -> torch.Tensor:
        tokens = (
            [int(value) for value in sequence.detach().cpu().reshape(-1).tolist()]
            if torch.is_tensor(sequence)
            else [int(value) for value in sequence]
        )
        if self.representation_variant in {
            "r1_no_boundary_identity",
            "r3_no_full_load_assignment",
        }:
            return torch.tensor(tokens, dtype=torch.long)
        if self.representation_variant == "r4_cell_level_demand":
            result = []
            in_demands = False
            for token in tokens:
                if token == self.demand_section_begin:
                    in_demands = True
                elif token == self.demand_section_end:
                    in_demands = False
                if in_demands and token == self.demand_end:
                    result.extend([
                        self.load_pin,
                        self.pin_offset + self.pin_name_to_id["__UNKNOWN__"],
                    ])
                result.append(token)
            return torch.tensor(result, dtype=torch.long)

        # R2: the full-load table still exposes realized fanout, but the source
        # table itself carries no per-source budget field.
        load_counts = {}
        try:
            begin = tokens.index(self.source_net_section_begin)
            end = tokens.index(self.source_net_section_end, begin + 1)
            index = begin + 1
            while index < end:
                if tokens[index] != self.source_load_begin:
                    raise ValueError("malformed R2 source-load table")
                source_id = self._pointer_value(
                    tokens[index + 1], self.source_offset, "source"
                )
                if tokens[index + 2] != self.count_token:
                    raise ValueError("malformed R2 source-load count")
                count = self._pointer_value(
                    tokens[index + 3], self.count_offset, "count"
                )
                load_counts[source_id] = count
                index += 5 + count
        except ValueError:
            load_counts = {}

        result = []
        source_ordinal = 0
        for token in tokens:
            if token == self.source_end:
                count = int(load_counts.get(source_ordinal, 0))
                if count <= 0:
                    bucket = self.fanout_0
                elif count == 1:
                    bucket = self.fanout_1
                elif count <= 4:
                    bucket = self.fanout_2_4
                elif count <= 8:
                    bucket = self.fanout_5_8
                elif count <= 16:
                    bucket = self.fanout_9_16
                else:
                    bucket = self.fanout_17_plus
                result.extend([
                    self.fanout_bucket, bucket,
                    self.max_fanout, self._count(count),
                    self.used_fanout, self._count(count),
                    self.remaining_fanout, self._count(0),
                ])
                source_ordinal += 1
            result.append(token)
        return torch.tensor(result, dtype=torch.long)

    def tokenize(self, data):
        r0 = super().tokenize(data)
        sequence = self._strip_variant_fields(r0)
        self.last_overlength = bool(
            self.max_length > 0 and int(sequence.numel()) > self.max_length
        )
        if self.last_overlength:
            raise RuntimeError(
                f"{self.tokenizer_version} no-silent-truncation violation: "
                f"length={int(sequence.numel())} exceeds max_length={self.max_length}"
            )
        self.last_pointer_payload.update({
            "schema_version": self.serializer_version,
            "tokenizer_version": self.tokenizer_version,
            "representation_variant": self.representation_variant,
            "information_sufficient_for_strict_decode": (
                self.information_sufficient_for_strict_decode
            ),
            "sequence_length": int(sequence.numel()),
        })
        return sequence

    def parse_tokens(self, sequence):
        payload, report = super().parse_tokens(self._inflate_for_r0_parser(sequence))
        payload["schema_version"] = self.serializer_version
        payload["representation_variant"] = self.representation_variant
        report["representation_variant"] = self.representation_variant
        report["information_sufficient_for_strict_decode"] = (
            self.information_sufficient_for_strict_decode
        )
        if self.representation_variant == "r2_no_per_source_budget":
            report["source_budget_encoded"] = False
        if self.representation_variant == "r3_no_full_load_assignment":
            report["source_load_full_coverage"] = False
            report["demand_exactly_once"] = False
        if self.representation_variant == "r4_cell_level_demand":
            report["pin_slot_identity_encoded"] = False
        return payload, report

    def decode(self, sequence):
        if not self.information_sufficient_for_strict_decode:
            raise ValueError(
                f"{self.tokenizer_version} intentionally lacks information "
                "required for repair-free strict graph decode"
            )
        payload, report = self.parse_tokens(sequence)
        if not report["source_load_full_coverage"]:
            raise ValueError(f"{self.tokenizer_version} requires full source-load coverage")
        if report["source_budget_violation"]:
            raise ValueError(f"{self.tokenizer_version} rejected source budget violation")
        if (
            report["same_cell_same_net_reuse"]
            and self.representation_variant != "r1_no_boundary_identity"
        ):
            raise ValueError(f"{self.tokenizer_version} rejected same-cell same-net reuse")
        return decode_source_net(
            payload,
            net_id=self.net_id,
            boundary_stub_id=self.boundary_stub_id,
            cell_to_label=self.cell_to_label,
        )


class CircuitSourceNetR1NoBoundaryIdentityTokenizer(
    _CircuitSourceNetAblationTokenizer
):
    serializer_version = "elda_r1_no_boundary_identity_v1"
    tokenizer_version = serializer_version
    representation_variant = "r1_no_boundary_identity"

    def tokenize(self, data):
        payload = serialize_source_net(
            data,
            net_id=self.net_id,
            boundary_stub_id=self.boundary_stub_id,
            label_to_cell=self.label_to_cell,
            pin_specs=self.pin_specs,
            chunk_size=max(1, self.max_num_nodes),
        )
        boundary_ids = {
            source["source_id"]
            for source in payload["sources"]
            if source["source_kind"] == "BOUNDARY_SOURCE"
        }
        if boundary_ids:
            generic_id = "GENERIC_BOUNDARY_SOURCE"
            generic_demands = []
            retained_nets = []
            for source_net in payload["source_nets"]:
                demand_ids = [
                    value
                    for chunk in source_net["chunks"]
                    for value in chunk["demand_ids"]
                ]
                if source_net["source_id"] in boundary_ids:
                    generic_demands.extend(demand_ids)
                else:
                    retained_nets.append(source_net)
            for demand in payload["demands"]:
                if demand.get("source_id") in boundary_ids:
                    demand["source_id"] = generic_id
            retained_sources = [
                source for source in payload["sources"]
                if source["source_id"] not in boundary_ids
            ]
            fanout = len(generic_demands)
            retained_sources.append({
                "source_id": generic_id,
                "source_kind": "BOUNDARY_SOURCE",
                "source_cell_id": None,
                "source_pin": "__BOUNDARY__",
                "boundary_pin_id": None,
                "source_role": "BOUNDARY_PIN",
                "endpoint_type": "BOUNDARY_STUB",
                "fanout_bucket": count_bucket(fanout),
                "max_fanout": fanout,
                "used_fanout": fanout,
                "remaining_fanout": 0,
            })
            retained_nets.append({
                "source_id": generic_id,
                "fanout": fanout,
                "chunks": [{
                    "source_id": generic_id,
                    "chunk_id": 0,
                    "demand_ids": generic_demands,
                }],
            })
            payload["sources"] = retained_sources
            payload["source_nets"] = retained_nets
        sequence = self._tokens_from_payload(payload)
        self.last_overlength = bool(
            self.max_length > 0 and int(sequence.numel()) > self.max_length
        )
        if self.last_overlength:
            raise RuntimeError(
                f"{self.tokenizer_version} no-silent-truncation violation: "
                f"length={int(sequence.numel())} exceeds max_length={self.max_length}"
            )
        self.last_pointer_payload = {
            "schema_version": self.serializer_version,
            "tokenizer_version": self.tokenizer_version,
            "representation_variant": self.representation_variant,
            "information_sufficient_for_strict_decode": True,
            "object_table": {
                "cell_count": len(payload["cells"]),
                "demand_count": len(payload["demands"]),
                "source_count": len(payload["sources"]),
            },
            "sequence_length": int(sequence.numel()),
        }
        return sequence


class CircuitSourceNetR2NoPerSourceBudgetTokenizer(
    _CircuitSourceNetAblationTokenizer
):
    serializer_version = "elda_r2_no_per_source_budget_v1"
    tokenizer_version = serializer_version
    representation_variant = "r2_no_per_source_budget"


class CircuitSourceNetR3NoFullLoadAssignmentTokenizer(
    _CircuitSourceNetAblationTokenizer
):
    serializer_version = "elda_r3_no_full_load_assignment_v1"
    tokenizer_version = serializer_version
    representation_variant = "r3_no_full_load_assignment"
    information_sufficient_for_strict_decode = False


class CircuitSourceNetR4CellLevelDemandTokenizer(
    _CircuitSourceNetAblationTokenizer
):
    serializer_version = "elda_r4_cell_level_demand_v1"
    tokenizer_version = serializer_version
    representation_variant = "r4_cell_level_demand"
    information_sufficient_for_strict_decode = False
