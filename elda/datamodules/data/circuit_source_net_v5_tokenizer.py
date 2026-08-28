from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from .batch_converter import BatchConverter


class CircuitSourceNetV5Tokenizer(object):
    """Dual-view demand-constrained Source-Net V5 tokenizer.

    V5 keeps V4.1 sink-first records as coverage truth and adds a source-net
    regroup view for fanout/source distribution supervision.  It intentionally
    stays token-CE compatible: pointer payload is emitted for audits and future
    V5.1 work, but no pointer head or candidate-mask decoding is implemented.
    """

    serializer_version = "source_net_v5_dual_view_demand_constrained_v1"
    tokenizer_version = "source_net_v5_dual_view_demand_constrained_v1"

    FANOUT_BUCKETS = ("1", "2_4", "5_8", "9_16", "17_PLUS")
    SOURCE_KINDS = ("CELL_OUTPUT", "BOUNDARY_SOURCE", "STUB_SOURCE", "PI_CONST_SOURCE", "INPUT_POOL_SOURCE", "PRIVATE_OUT_SOURCE")

    def __init__(
        self,
        *,
        dataset_names=None,
        max_length=-1,
        truncation_length=None,
        append_eos=True,
        net_id: int = 0,
        boundary_stub_id: int = -1,
        label_to_cell: Optional[Dict[int, str]] = None,
        pin_specs: Optional[Dict[str, object]] = None,
        **kwargs,
    ):
        self.dataset_names = list(dataset_names) if dataset_names is not None else []
        self.max_length = int(max_length)
        self.truncation_length = truncation_length
        self.append_eos = bool(append_eos)
        self.net_id = int(net_id)
        self.boundary_stub_id = int(boundary_stub_id)
        self.label_to_cell = {int(k): str(v) for k, v in dict(label_to_cell or {}).items()}
        self.pin_specs = dict(pin_specs or {})
        self.is_source_net_v5_tokenizer = True
        self.source_net_v5_compact = bool(kwargs.get("source_net_v5_compact", False))
        self.compact_demand_record = bool(kwargs.get("compact_demand_record", self.source_net_v5_compact))
        self.compact_sink_demand_record = bool(kwargs.get("compact_sink_demand_record", self.source_net_v5_compact))
        self.compact_source_net_hist = bool(kwargs.get("compact_source_net_hist", self.source_net_v5_compact))

        names = [
            "sos", "eos", "pad",
            "COND_SECTION_BEGIN", "COND_SECTION_END",
            "DEMAND_SECTION_BEGIN", "DEMAND_SECTION_END",
            "SOURCE_BUDGET_SECTION_BEGIN", "SOURCE_BUDGET_SECTION_END",
            "CELL_SECTION_BEGIN", "CELL_SECTION_END",
            "SINK_DEMAND_SECTION_BEGIN", "SINK_DEMAND_SECTION_END",
            "SOURCE_NET_SECTION_BEGIN", "SOURCE_NET_SECTION_END",
            "DEMAND_TABLE_BEGIN", "DEMAND_TABLE_END",
            "DEMAND_BEGIN", "DEMAND_END",
            "SOURCE_NET_BEGIN", "SOURCE_NET_END",
            "LOAD_GROUP_BEGIN", "LOAD_GROUP_END",
            "LOAD_PREVIEW_BEGIN", "LOAD_PREVIEW_END",
            "LOAD_TYPE_HIST", "LOAD_PIN_HIST",
            "CELL", "DEMAND_ID", "LOAD_CELL", "LOAD_CELL_TYPE", "LOAD_PIN", "LOAD_PIN_ROLE",
            "DEMAND_KIND", "SOURCE_KIND", "SOURCE_CELL", "SOURCE_CELL_TYPE", "SOURCE_PIN", "SOURCE_PIN_ROLE",
            "SOURCE_FINAL_FANOUT_BUCKET", "FANOUT_BUCKET",
            "TOPO_UNKNOWN", "TOPO_0", "TOPO_1", "TOPO_2", "TOPO_3_PLUS",
            "FANIN_0", "FANIN_1", "FANIN_2_4", "FANIN_5_PLUS",
            "FANOUT_0", "FANOUT_1", "FANOUT_2_4", "FANOUT_5_8", "FANOUT_9_16", "FANOUT_17_PLUS",
            "HAS_SEQ", "NO_SEQ", "HAS_MUX", "NO_MUX", "RARE_OR_MID", "HEAD_TYPE", "HAS_AOI_OAI", "NO_AOI_OAI",
            "INTERNAL_REQUIRED", "BOUNDARY_SINK", "EXTERNAL_LOAD", "OTHER_REQUIRED",
            "REQUIRED_INPUT", "OUTPUT_PIN", "BOUNDARY_PIN", "CONST_PIN",
            "CELL_OUTPUT", "BOUNDARY_SOURCE", "STUB_SOURCE", "PI_CONST_SOURCE", "INPUT_POOL_SOURCE", "PRIVATE_OUT_SOURCE",
            "CONST0", "CONST1", "MISSING", "UNKNOWN", "LOW", "MID", "HIGH",
            "BUCKET_0", "BUCKET_1", "BUCKET_2_4", "BUCKET_5_8", "BUCKET_9_16", "BUCKET_17_PLUS",
        ]
        self.special_toks = names
        for i, name in enumerate(names):
            setattr(self, name.lower(), i)
        self.sos = 0
        self.eos = 1
        self.pad = 2
        self.idx_offset = len(self.special_toks)
        self.max_num_nodes: Optional[int] = None
        self.pin_names: List[str] = []
        self.pin_name_to_id: Dict[str, int] = {}
        self.pin_offset: Optional[int] = None
        self.node_type_offset: Optional[int] = None
        self.num_node_types = 0
        self.last_pointer_payload: Dict[str, object] = {}
        self.last_overlength = False

    @property
    def labeled_graph(self) -> bool:
        return False

    def batch_converter(self):
        return BatchConverter(self, self.truncation_length)

    def __call__(self, data: Data):
        return self.tokenize(data)

    def set_num_nodes(self, max_num_nodes):
        if self.max_num_nodes is None or self.max_num_nodes < int(max_num_nodes):
            self.max_num_nodes = int(max_num_nodes)

    def set_pin_name_vocab(self, pin_names: List[str]) -> None:
        names = sorted({str(x) for x in pin_names} | {"__UNKNOWN__", "__BOUNDARY__", "__CONST0__", "__CONST1__"})
        self.pin_names = names
        self.pin_name_to_id = {name: i for i, name in enumerate(names)}
        if self.max_num_nodes is not None:
            self.pin_offset = int(self.idx_offset + 3 * self.max_num_nodes)
            self.node_type_offset = int(self.pin_offset + len(self.pin_names))

    def set_num_node_and_edge_types(self, num_node_types=0, num_edge_types=0):
        self.num_node_types = int(num_node_types)
        if self.max_num_nodes is None:
            raise ValueError("set_num_nodes first")
        self.pin_offset = int(self.idx_offset + 3 * self.max_num_nodes)
        self.node_type_offset = int(self.pin_offset + len(self.pin_names))

    def __len__(self):
        if self.max_num_nodes is None or self.pin_offset is None or self.node_type_offset is None:
            raise ValueError("tokenizer is not initialized")
        return int(self.node_type_offset + max(1, self.num_node_types))

    def _tok_gate_idx(self, idx: int) -> int:
        return int(self.idx_offset + int(idx))

    def _tok_src_idx(self, idx: int) -> int:
        return int(self.idx_offset + self.max_num_nodes + int(idx))

    def _tok_demand_idx(self, idx: int) -> int:
        return int(self.idx_offset + 2 * self.max_num_nodes + int(idx))

    def _idx_from_gate_tok(self, tok: int) -> int:
        return int(tok - self.idx_offset)

    def _idx_from_src_tok(self, tok: int) -> int:
        return int(tok - self.idx_offset - self.max_num_nodes)

    def _idx_from_demand_tok(self, tok: int) -> int:
        return int(tok - self.idx_offset - 2 * self.max_num_nodes)

    def _tok_pin(self, name: str) -> int:
        return int(self.pin_offset + self.pin_name_to_id.get(str(name), self.pin_name_to_id.get("__UNKNOWN__", 0)))

    def _pin_from_tok(self, tok: int) -> str:
        idx = int(tok - self.pin_offset)
        if 0 <= idx < len(self.pin_names):
            return self.pin_names[idx]
        return "__INVALID_PIN__"

    def _tok_cell_label(self, label: int) -> int:
        label = int(label)
        if label < 0 or (self.num_node_types and label >= self.num_node_types):
            label = int(self.net_id)
        return int(self.node_type_offset + label)

    def _cell_from_tok(self, tok: int) -> int:
        label = int(tok - self.node_type_offset)
        if label < 0 or (self.num_node_types and label >= self.num_node_types):
            return int(self.net_id)
        return int(label)

    def _cell_name(self, label: int) -> str:
        return self.label_to_cell.get(int(label), f"LABEL_{int(label)}")

    def _spec_pins(self, cell_name: str) -> Tuple[set, set]:
        spec = self.pin_specs.get(str(cell_name))
        if spec is None:
            return set(), set()
        return set(str(p) for p in getattr(spec, "inputs", [])), set(str(p) for p in getattr(spec, "outputs", []))

    def _gate_nodes(self, data: Data) -> List[int]:
        x = data.x.reshape(-1).to(torch.long)
        mask = (x != int(self.net_id)) & (x != int(self.boundary_stub_id))
        return [int(i) for i in torch.nonzero(mask, as_tuple=False).flatten().tolist()]

    def _edge_rows(self, data: Data):
        edge_index = data.edge_index.to(torch.long)
        x = data.x.reshape(-1).to(torch.long)
        edge_role = getattr(data, "edge_role", None)
        if edge_role is None:
            edge_role = getattr(data, "edge_attr", None)
        edge_pin = getattr(data, "edge_pin_id", None)
        pin_names = getattr(data, "pin_id_to_name", [])
        if edge_role is None or edge_pin is None or int(edge_index.size(1)) == 0:
            return []
        rows = []
        for eidx, (src, dst) in enumerate(edge_index.t().tolist()):
            role = int(edge_role[eidx].item())
            pid = int(edge_pin[eidx].item()) if int(edge_pin.numel()) > eidx else -1
            pname = str(pin_names[pid]) if 0 <= pid < len(pin_names) else "__UNKNOWN__"
            rows.append((int(src), int(dst), role, pname))
        return rows

    @staticmethod
    def _count_bucket(n: int) -> int:
        if n <= 0:
            return 0
        if n == 1:
            return 1
        if n <= 4:
            return 2
        if n <= 8:
            return 3
        if n <= 16:
            return 4
        return 5

    def _bucket_tok(self, n: int) -> int:
        return [self.bucket_0, self.bucket_1, self.bucket_2_4, self.bucket_5_8, self.bucket_9_16, self.bucket_17_plus][self._count_bucket(int(n))]

    def _fanout_tok(self, n: int) -> int:
        return [self.fanout_0, self.fanout_1, self.fanout_2_4, self.fanout_5_8, self.fanout_9_16, self.fanout_17_plus][self._count_bucket(int(n))]

    def _source_kind(self, src) -> str:
        if src[0] != "CELL":
            return "BOUNDARY_SOURCE"
        cell = str(src[3])
        if cell == "INPUT_POOL":
            return "INPUT_POOL_SOURCE"
        if cell == "PRIVATE_OUT":
            return "PRIVATE_OUT_SOURCE"
        if "STUB" in cell.upper():
            return "STUB_SOURCE"
        return "CELL_OUTPUT"

    def _source_kind_tok(self, kind: str) -> int:
        return int(getattr(self, str(kind).lower(), self.unknown))

    def serialize_records(self, data: Data) -> Dict[str, object]:
        x = data.x.reshape(-1).to(torch.long)
        gates = self._gate_nodes(data)
        gate_to_dense = {gid: i for i, gid in enumerate(gates)}
        by_net: Dict[int, Dict[str, object]] = defaultdict(lambda: {"drivers": [], "loads": []})
        for src, dst, role, pin in self._edge_rows(data):
            if int(dst) >= int(x.numel()) or int(x[dst].item()) not in {int(self.net_id), int(self.boundary_stub_id)}:
                continue
            if src not in gate_to_dense:
                continue
            cell_name = self._cell_name(int(x[src].item()))
            inputs, outputs = self._spec_pins(cell_name)
            item = (gate_to_dense[src], str(pin), cell_name, src, role)
            if role == 1 and str(pin) in inputs:
                by_net[int(dst)]["loads"].append(item)
            elif role == 2 or (role == 3 and str(pin) in outputs):
                by_net[int(dst)]["drivers"].append(item)

        sink_records = []
        source_to_loads: Dict[Tuple, List[Dict[str, object]]] = defaultdict(list)
        for endpoint in sorted(by_net.keys()):
            drivers = list(by_net[endpoint]["drivers"])
            loads = list(by_net[endpoint]["loads"])
            primary = next((d for d in drivers if int(d[4]) == 2), None)
            driver = primary if primary is not None else (drivers[0] if drivers else None)
            for load in sorted(loads, key=lambda z: (str(z[2]), str(z[1]), int(z[0]))):
                lg, lpin, lcell, _lsrc, _lrole = load
                if driver is None:
                    src = ("BOUNDARY_SOURCE", None, "__BOUNDARY__", "BOUNDARY")
                else:
                    sg, spin, scell, _ssrc, _srole = driver
                    src = ("CELL", int(sg), str(spin), str(scell))
                record = {
                    "demand_id": len(sink_records),
                    "sink": (int(lg), str(lpin), str(lcell)),
                    "src": src,
                    "endpoint": int(endpoint),
                    "demand_kind": "INTERNAL_REQUIRED",
                }
                sink_records.append(record)
                source_to_loads[tuple(src)].append(record)

        for record in sink_records:
            record["source_final_fanout"] = len(source_to_loads[tuple(record["src"])])
            record["source_kind"] = self._source_kind(record["src"])

        source_records = []
        for src, loads in sorted(source_to_loads.items(), key=lambda kv: (self._source_kind(kv[0]), -len(kv[1]), str(kv[0]))):
            stable = sorted(loads, key=lambda r: (str(r["sink"][2]), str(r["sink"][1]), int(r["sink"][0]), int(r["demand_id"])))
            source_records.append({"src": src, "source_kind": self._source_kind(src), "loads": stable, "fanout": len(stable)})

        return {"gates": gates, "sink_records": sink_records, "source_records": source_records}

    def pointer_payload(self, data: Data) -> Dict[str, object]:
        rec = self.serialize_records(data)
        slot_count = Counter()
        for _r in rec["sink_records"]:
            for k in ["LOAD_CELL_SLOT", "LOAD_PIN_SLOT", "SOURCE_KIND_SLOT", "SOURCE_CELL_SLOT", "SOURCE_PIN_SLOT", "DEMAND_ID_SLOT", "SOURCE_FINAL_FANOUT_BUCKET_SLOT"]:
                slot_count[k] += 1
        for _r in rec["source_records"]:
            slot_count["SOURCE_NET_SLOT"] += 1
            slot_count["FANOUT_BUCKET_SLOT"] += 1
        return {
            "slot_count_by_type": dict(slot_count),
            "object_table": {
                "cell_count": len(rec["gates"]),
                "demand_count": len(rec["sink_records"]),
                "source_count": len(rec["source_records"]),
                "load_count": len(rec["sink_records"]),
            },
            "candidate_summary": {
                "gold_target_missing_count_by_slot_type": {},
                "candidate_source_count": len(rec["source_records"]),
                "candidate_load_count": len(rec["sink_records"]),
                "candidate_pin_count": len(self.pin_names),
                "mask_gold_coverage_estimate": 1.0,
            },
        }

    def tokenize(self, data: Data):
        x = data.x.reshape(-1).to(torch.long)
        rec = self.serialize_records(data)
        gates = rec["gates"]
        sink_records = rec["sink_records"]
        source_records = rec["source_records"]
        self.last_pointer_payload = self.pointer_payload(data)
        tokens: List[int] = [self.sos]

        tokens.extend([self.cond_section_begin, self._bucket_tok(len(sink_records)), self._bucket_tok(len(source_records)), self.cond_section_end])

        tokens.extend([self.demand_section_begin, self._bucket_tok(len(sink_records)), self.demand_table_begin])
        for r in sink_records:
            lg, lpin, lcell = r["sink"]
            tokens.extend([self.demand_begin, self.demand_id, self._tok_demand_idx(r["demand_id"]), self.load_cell, self._tok_gate_idx(lg)])
            if not self.compact_demand_record:
                tokens.extend([self.load_cell_type, self._tok_cell_label(int(x[gates[lg]].item())) if 0 <= lg < len(gates) else self.unknown])
            tokens.extend([self.load_pin, self._tok_pin(lpin)])
            if not self.compact_demand_record:
                tokens.extend([self.load_pin_role, self.required_input])
            tokens.extend([self.demand_kind, self.internal_required, self.demand_end])
        tokens.extend([self.demand_table_end, self.demand_section_end])

        hist = Counter(self._count_bucket(int(s["fanout"])) for s in source_records)
        tokens.extend([self.source_budget_section_begin, self._bucket_tok(len(source_records))])
        for idx in range(1, 6):
            tokens.extend([self._bucket_tok(hist.get(idx, 0))])
        for kind in self.SOURCE_KINDS:
            kind_count = sum(1 for s in source_records if s["source_kind"] == kind)
            tokens.extend([self._source_kind_tok(kind), self._bucket_tok(kind_count)])
        tokens.append(self.source_budget_section_end)

        tokens.append(self.cell_section_begin)
        dense_order = sorted(range(len(gates)), key=lambda d: (self._cell_name(int(x[gates[d]].item())), d))
        load_count = Counter(int(r["sink"][0]) for r in sink_records)
        src_count = Counter(int(r["src"][1]) for r in sink_records if r["src"][0] == "CELL")
        for dense in dense_order:
            cell_name = self._cell_name(int(x[gates[dense]].item()))
            tokens.extend([self.cell, self._tok_gate_idx(dense), self._tok_cell_label(int(x[gates[dense]].item())), self.topo_unknown, self._bucket_tok(load_count.get(dense, 0)), self._fanout_tok(src_count.get(dense, 0))])
            tokens.append(self.has_seq if "DFF" in cell_name else self.no_seq)
            tokens.append(self.has_mux if "MUX" in cell_name else self.no_mux)
            tokens.append(self.has_aoi_oai if ("AOI" in cell_name or "OAI" in cell_name) else self.no_aoi_oai)
        tokens.append(self.cell_section_end)

        def sink_sort_key(r):
            return (self._count_bucket(int(r["source_final_fanout"])), str(r["sink"][2]), str(r["sink"][1]), int(r["sink"][0]))

        tokens.append(self.sink_demand_section_begin)
        for r in sorted(sink_records, key=sink_sort_key):
            lg, lpin, _lcell = r["sink"]
            src = r["src"]
            tokens.extend([self.demand_begin, self.demand_id, self._tok_demand_idx(r["demand_id"]), self.load_cell, self._tok_gate_idx(lg), self.load_pin, self._tok_pin(lpin)])
            if not self.compact_sink_demand_record:
                tokens.extend([self.load_pin_role, self.required_input])
            tokens.extend([self.source_kind, self._source_kind_tok(r["source_kind"])])
            if src[0] == "CELL":
                tokens.extend([self.source_cell, self._tok_src_idx(int(src[1]))])
                if not self.compact_sink_demand_record:
                    tokens.extend([self.source_cell_type, self._tok_cell_label(int(x[gates[int(src[1])]].item())) if 0 <= int(src[1]) < len(gates) else self.unknown])
                tokens.extend([self.source_pin, self._tok_pin(str(src[2]))])
                if not self.compact_sink_demand_record:
                    tokens.extend([self.source_pin_role, self.output_pin])
            else:
                tokens.extend([self.source_cell, self.boundary_source])
                if not self.compact_sink_demand_record:
                    tokens.extend([self.source_cell_type, self.boundary_source])
                tokens.extend([self.source_pin, self._tok_pin("__BOUNDARY__")])
                if not self.compact_sink_demand_record:
                    tokens.extend([self.source_pin_role, self.boundary_pin])
            tokens.extend([self.source_final_fanout_bucket, self._fanout_tok(int(r["source_final_fanout"])), self.demand_end])
        tokens.append(self.sink_demand_section_end)

        tokens.append(self.source_net_section_begin)
        for srec in source_records:
            src = srec["src"]
            loads = list(srec["loads"])
            tokens.extend([self.source_net_begin, self.source_kind, self._source_kind_tok(srec["source_kind"])])
            if src[0] == "CELL":
                tokens.extend([self.source_cell, self._tok_src_idx(int(src[1])), self.source_cell_type, self._tok_cell_label(int(x[gates[int(src[1])]].item())) if 0 <= int(src[1]) < len(gates) else self.unknown, self.source_pin, self._tok_pin(str(src[2]))])
            else:
                tokens.extend([self.source_cell, self.boundary_source, self.source_cell_type, self.boundary_source, self.source_pin, self._tok_pin("__BOUNDARY__")])
            tokens.extend([self.fanout_bucket, self._fanout_tok(len(loads))])
            explicit = loads if len(loads) <= 8 else loads[:8]
            tokens.append(self.load_group_begin if len(loads) <= 8 else self.load_preview_begin)
            for r in explicit:
                lg, lpin, _lcell = r["sink"]
                tokens.extend([self.load_cell, self._tok_gate_idx(lg), self.load_pin, self._tok_pin(lpin), self.demand_id, self._tok_demand_idx(r["demand_id"])])
            tokens.append(self.load_group_end if len(loads) <= 8 else self.load_preview_end)
            if len(loads) > 8 and not self.compact_source_net_hist:
                tokens.extend([self.load_type_hist, self._bucket_tok(len({str(r["sink"][2]) for r in loads})), self.load_pin_hist, self._bucket_tok(len({str(r["sink"][1]) for r in loads}))])
            tokens.append(self.source_net_end)
        tokens.append(self.source_net_section_end)
        if self.append_eos:
            tokens.append(self.eos)
        out = torch.tensor(tokens, dtype=torch.long)
        self.last_overlength = bool(self.max_length is not None and int(self.max_length) > 0 and out.numel() > int(self.max_length))
        return out

    def decode(self, token_ids: torch.Tensor) -> Data:
        toks = [int(x) for x in token_ids.reshape(-1).tolist() if int(x) not in {self.pad, self.sos}]
        cells: Dict[int, int] = {}
        sink_records: List[Tuple[int, str, Tuple]] = []
        decode_valid = True
        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok == self.eos:
                break
            if tok == self.cell and i + 2 < len(toks):
                cells[self._idx_from_gate_tok(toks[i + 1])] = self._cell_from_tok(toks[i + 2])
                i += 3
                continue
            if tok == self.demand_begin:
                j = i + 1
                load_cell = None; load_pin = None; src = None; source_kind = None
                while j < len(toks) and toks[j] != self.demand_end:
                    if toks[j] == self.load_cell and j + 1 < len(toks):
                        load_cell = self._idx_from_gate_tok(toks[j + 1]); j += 2; continue
                    if toks[j] == self.load_pin and j + 1 < len(toks):
                        load_pin = self._pin_from_tok(toks[j + 1]); j += 2; continue
                    if toks[j] == self.source_kind and j + 1 < len(toks):
                        source_kind = int(toks[j + 1]); j += 2; continue
                    if toks[j] == self.source_cell and j + 1 < len(toks):
                        val = toks[j + 1]
                        if self.max_num_nodes is not None and self.idx_offset + self.max_num_nodes <= val < self.idx_offset + 2 * self.max_num_nodes:
                            src_idx = self._idx_from_src_tok(val)
                            if source_kind == int(self.boundary_source):
                                src = ("BOUNDARY_SOURCE", src_idx, "__UNKNOWN__")
                            elif source_kind == int(self.stub_source):
                                src = ("STUB_SOURCE", src_idx, "__UNKNOWN__")
                            elif source_kind == int(self.pi_const_source):
                                src = ("PI_CONST_SOURCE", src_idx, "__UNKNOWN__")
                            elif source_kind == int(self.input_pool_source):
                                src = ("INPUT_POOL_SOURCE", src_idx, "__UNKNOWN__")
                            elif source_kind == int(self.private_out_source):
                                src = ("PRIVATE_OUT_SOURCE", src_idx, "__UNKNOWN__")
                            else:
                                src = ("CELL", src_idx, "__UNKNOWN__")
                        else:
                            src = ("BOUNDARY_SOURCE", "__BOUNDARY__")
                        j += 2; continue
                    if toks[j] == self.source_pin and j + 1 < len(toks):
                        pin = self._pin_from_tok(toks[j + 1])
                        if src is not None and src[0] == "CELL":
                            src = ("CELL", src[1], pin)
                        elif src is not None and len(src) >= 3:
                            src = (src[0], src[1], pin)
                        elif src is not None:
                            src = ("BOUNDARY_SOURCE", "__BOUNDARY__", pin)
                        j += 2; continue
                    j += 1
                if load_cell is not None and load_pin is not None and src is not None:
                    sink_records.append((int(load_cell), str(load_pin), src))
                i = max(j + 1, i + 1)
                continue
            i += 1

        node_labels: List[int] = []
        for gid in sorted(cells):
            while len(node_labels) <= gid:
                node_labels.append(int(self.net_id))
            node_labels[gid] = int(cells[gid])
        pin_to_id: Dict[str, int] = {}
        pins: List[str] = []
        edges, roles, edge_pin = [], [], []

        def local_pin_id(name: str) -> int:
            if name not in pin_to_id:
                pin_to_id[name] = len(pins)
                pins.append(name)
            return pin_to_id[name]

        src_to_net: Dict[Tuple, int] = {}
        for sg, spin, src in sink_records:
            if sg not in cells:
                decode_valid = False
                continue
            key = tuple(src)
            if key not in src_to_net:
                net_node = len(node_labels)
                src_to_net[key] = net_node
                node_labels.append(int(self.net_id))
                if src[0] == "CELL" and int(src[1]) in cells:
                    edges.append((int(src[1]), net_node)); roles.append(2); edge_pin.append(local_pin_id(str(src[2])))
                else:
                    bnode = len(node_labels)
                    node_labels.append(int(self.boundary_stub_id))
                    edges.append((bnode, net_node)); roles.append(2); edge_pin.append(local_pin_id("__BOUNDARY__"))
            net_node = src_to_net[key]
            edges.append((int(sg), net_node)); roles.append(1); edge_pin.append(local_pin_id(str(spin)))
        x = torch.tensor(node_labels, dtype=torch.long) if node_labels else torch.zeros((0,), dtype=torch.long)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=torch.tensor(roles, dtype=torch.long), num_nodes=int(x.numel()))
        data.edge_role = data.edge_attr
        data.edge_pin_id = torch.tensor(edge_pin, dtype=torch.long) if edge_pin else torch.full((0,), -1, dtype=torch.long)
        data.pin_id_to_name = pins
        data.decode_valid = bool(decode_valid)
        return data

    def validate_constraints(self, data: Data) -> Dict[str, float]:
        rec = self.serialize_records(data)
        sink_counter = Counter()
        invalid_sink = invalid_src = invalid_cell = 0
        for r in rec["sink_records"]:
            sg, spin, scell = r["sink"]
            inputs, _outputs = self._spec_pins(str(scell))
            sink_counter[(int(sg), str(spin))] += 1
            if str(spin) not in inputs:
                invalid_sink += 1; invalid_cell += 1
            src = r["src"]
            if src[0] == "CELL":
                _kind, _idx, pin, cell = src
                _inputs, outputs = self._spec_pins(str(cell))
                if str(pin) not in outputs:
                    invalid_src += 1; invalid_cell += 1
        required = sum(len(self._spec_pins(str(r["sink"][2]))[0]) for r in rec["sink_records"])
        return {
            "invalid_sink_pin_count": float(invalid_sink),
            "invalid_src_pin_count": float(invalid_src),
            "invalid_cell_pin_count": float(invalid_cell),
            "duplicate_sink_count": float(sum(v - 1 for v in sink_counter.values() if v > 1)),
            "demand_table_count": float(len(rec["sink_records"])),
            "sink_demand_count": float(len(rec["sink_records"])),
            "source_net_count": float(len(rec["source_records"])),
            "teacher_forced_required_input_coverage": 1.0 if rec["sink_records"] else 0.0,
            "required_input_count_observed": float(len(rec["sink_records"])),
            "required_pin_capacity_estimate": float(required),
        }
