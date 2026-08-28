from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from .batch_converter import BatchConverter


CRITICAL_OUTPUTS = {("FA_X1", "S"), ("HA_X1", "S"), ("DFF_X1", "QN")}


class CircuitSourceNetV41SinkFirstTokenizer(object):
    """Sink-first / coverage-first Source-Net V4.1 tokenizer.

    V4 was source-net-first: NET -> DRIVER -> LOAD*. This V4.1 schema emits
    one SINK record per observed input load and groups equal SRC records during
    decode. Required-input coverage therefore becomes the primary sequence
    modeling task instead of an emergent fanout side effect.
    """

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
        self.is_source_net_v41_tokenizer = True

        names = [
            "sos",
            "eos",
            "pad",
            "CONTRACT_START",
            "CONTRACT_END",
            "GC_BUCKET_0",
            "GC_BUCKET_64",
            "GC_BUCKET_96",
            "GC_BUCKET_128",
            "GC_BUCKET_160",
            "GC_BUCKET_224",
            "CUT_EDGE_BUCKET_0",
            "CUT_EDGE_BUCKET_1_4",
            "CUT_EDGE_BUCKET_5_16",
            "CUT_EDGE_BUCKET_17_32",
            "CUT_EDGE_BUCKET_33_PLUS",
            "SOURCE_DIVERSITY_BUCKET_LOW",
            "SOURCE_DIVERSITY_BUCKET_MID",
            "SOURCE_DIVERSITY_BUCKET_HIGH",
            "FANOUT_PROFILE_BUCKET_LOW",
            "FANOUT_PROFILE_BUCKET_MID",
            "FANOUT_PROFILE_BUCKET_HIGH",
            "CRITICAL_OUTPUT_BUCKET_NONE",
            "CRITICAL_OUTPUT_BUCKET_LOW",
            "CRITICAL_OUTPUT_BUCKET_HIGH",
            "POOL_RISK_BUCKET_LOW",
            "POOL_RISK_BUCKET_MID",
            "POOL_RISK_BUCKET_HIGH",
            "CELL_SECTION_START",
            "CELL",
            "CELL_SECTION_END",
            "SINK_CONNECTION_SECTION_START",
            "SINK",
            "SRC",
            "BOUNDARY_IN",
            "EXTERNAL_SOURCE",
            "CONST0",
            "CONST1",
            "SINK_CONNECTION_SECTION_END",
            "PIN_UNKNOWN",
            "CELL_UNKNOWN",
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
        self.num_node_types: int = 0

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
            self.pin_offset = int(self.idx_offset + 2 * self.max_num_nodes)
            self.node_type_offset = int(self.pin_offset + len(self.pin_names))

    def set_num_node_and_edge_types(self, num_node_types=0, num_edge_types=0):
        self.num_node_types = int(num_node_types)
        if self.max_num_nodes is None:
            raise ValueError("set_num_nodes first")
        self.pin_offset = int(self.idx_offset + 2 * self.max_num_nodes)
        self.node_type_offset = int(self.pin_offset + len(self.pin_names))

    def __len__(self):
        if self.max_num_nodes is None or self.pin_offset is None or self.node_type_offset is None:
            raise ValueError("tokenizer is not initialized")
        return int(self.node_type_offset + max(1, self.num_node_types))

    def _tok_gate_idx(self, idx: int) -> int:
        return int(self.idx_offset + int(idx))

    def _tok_src_idx(self, idx: int) -> int:
        if self.max_num_nodes is None:
            raise ValueError("tokenizer not initialized")
        return int(self.idx_offset + self.max_num_nodes + int(idx))

    def _idx_from_gate_tok(self, tok: int) -> int:
        return int(tok - self.idx_offset)

    def _idx_from_src_tok(self, tok: int) -> int:
        if self.max_num_nodes is None:
            return -1
        return int(tok - self.idx_offset - self.max_num_nodes)

    def _tok_pin(self, name: str) -> int:
        if self.pin_offset is None:
            raise ValueError("pin vocab not initialized")
        return int(self.pin_offset + self.pin_name_to_id.get(str(name), self.pin_name_to_id.get("__UNKNOWN__", 0)))

    def _pin_from_tok(self, tok: int) -> str:
        if self.pin_offset is None:
            return "__INVALID_PIN__"
        idx = int(tok - self.pin_offset)
        if 0 <= idx < len(self.pin_names):
            return self.pin_names[idx]
        return "__INVALID_PIN__"

    def _tok_cell_label(self, label: int) -> int:
        if self.node_type_offset is None:
            raise ValueError("node type vocab not initialized")
        label = int(label)
        if label < 0 or (self.num_node_types and label >= self.num_node_types):
            label = int(self.net_id)
        return int(self.node_type_offset + label)

    def _cell_from_tok(self, tok: int) -> int:
        if self.node_type_offset is None:
            return int(self.net_id)
        label = int(tok - self.node_type_offset)
        if label < 0 or (self.num_node_types and label >= self.num_node_types):
            return int(self.net_id)
        return int(label)

    def _gate_nodes(self, data: Data) -> List[int]:
        x = data.x.reshape(-1).to(torch.long)
        mask = (x != int(self.net_id)) & (x != int(self.boundary_stub_id))
        return [int(i) for i in torch.nonzero(mask, as_tuple=False).flatten().tolist()]

    def _cell_name(self, label: int) -> str:
        return self.label_to_cell.get(int(label), f"LABEL_{int(label)}")

    def _spec_pins(self, cell_name: str) -> Tuple[set, set]:
        spec = self.pin_specs.get(str(cell_name))
        if spec is None:
            return set(), set()
        return set(str(p) for p in getattr(spec, "inputs", [])), set(str(p) for p in getattr(spec, "outputs", []))

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
        used_outputs: Dict[int, set] = defaultdict(set)
        for src, _dst, role, pin in rows:
            if src >= int(x.numel()):
                continue
            cell_name = self._cell_name(int(x[src].item()))
            _inputs, outputs = self._spec_pins(cell_name)
            if role == 2 and str(pin) in outputs:
                used_outputs[int(src)].add(str(pin))
        fixed = []
        for src, dst, role, pin in rows:
            pin = str(pin)
            if src < int(x.numel()) and role == 3:
                cell_name = self._cell_name(int(x[src].item()))
                _inputs, outputs = self._spec_pins(cell_name)
                if pin not in outputs:
                    remaining = sorted(outputs - used_outputs.get(int(src), set()))
                    if len(remaining) == 1:
                        pin = remaining[0]
            fixed.append((src, dst, role, pin))
        return fixed

    def _gc_bucket_token(self, n: int) -> int:
        if n <= 64:
            return self.gc_bucket_64
        if n <= 96:
            return self.gc_bucket_96
        if n <= 128:
            return self.gc_bucket_128
        if n <= 160:
            return self.gc_bucket_160
        return self.gc_bucket_224

    def _fanout_profile_token(self, fanouts: List[int]) -> int:
        if not fanouts:
            return self.fanout_profile_bucket_low
        p95 = sorted(fanouts)[min(len(fanouts) - 1, int(0.95 * (len(fanouts) - 1)))]
        if p95 >= 9:
            return self.fanout_profile_bucket_high
        if p95 >= 2:
            return self.fanout_profile_bucket_mid
        return self.fanout_profile_bucket_low

    def serialize_records(self, data: Data) -> Dict[str, object]:
        x = data.x.reshape(-1).to(torch.long)
        gates = self._gate_nodes(data)
        gate_to_dense = {gid: i for i, gid in enumerate(gates)}
        by_net: Dict[int, Dict[str, object]] = defaultdict(lambda: {"drivers": [], "loads": []})
        for src, dst, role, pin in self._edge_rows(data):
            if int(dst) >= int(x.numel()):
                continue
            if int(x[dst].item()) not in {int(self.net_id), int(self.boundary_stub_id)}:
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
        fanouts = []
        for endpoint in sorted(by_net.keys()):
            drivers = list(by_net[endpoint]["drivers"])
            loads = list(by_net[endpoint]["loads"])
            primary = next((d for d in drivers if int(d[4]) == 2), None)
            driver = primary if primary is not None else (drivers[0] if drivers else None)
            if loads:
                fanouts.append(len(loads))
            for load in loads:
                lg, lpin, lcell, _lsrc, _lrole = load
                if driver is None:
                    src = ("BOUNDARY_IN", None, "__BOUNDARY__")
                else:
                    sg, spin, scell, _ssrc, _srole = driver
                    src = ("CELL", int(sg), str(spin), str(scell))
                sink_records.append(
                    {
                        "sink": (int(lg), str(lpin), str(lcell)),
                        "src": src,
                        "endpoint": int(endpoint),
                    }
                )
        return {"gates": gates, "sink_records": sink_records, "fanouts": fanouts}

    def tokenize(self, data: Data):
        if self.max_num_nodes is None or self.node_type_offset is None:
            raise ValueError("tokenizer not initialized")
        x = data.x.reshape(-1).to(torch.long)
        rec = self.serialize_records(data)
        gates: List[int] = rec["gates"]
        sink_records: List[Dict[str, object]] = rec["sink_records"]
        fanouts: List[int] = rec["fanouts"]
        tokens: List[int] = [self.sos, self.contract_start, self._gc_bucket_token(len(gates))]
        tokens.append(self.cut_edge_bucket_0)
        diversity = len({tuple(r["src"]) for r in sink_records}) / max(1, len(gates))
        tokens.append(self.source_diversity_bucket_high if diversity >= 0.5 else (self.source_diversity_bucket_mid if diversity >= 0.2 else self.source_diversity_bucket_low))
        tokens.append(self._fanout_profile_token(fanouts))
        critical_count = sum(1 for r in sink_records if tuple(r["src"][-2:]) in CRITICAL_OUTPUTS)
        tokens.append(self.critical_output_bucket_high if critical_count >= 2 else (self.critical_output_bucket_low if critical_count else self.critical_output_bucket_none))
        tokens.extend([self.pool_risk_bucket_low, self.contract_end])

        tokens.append(self.cell_section_start)
        for dense, gid in enumerate(gates):
            tokens.extend([self.cell, self._tok_gate_idx(dense), self._tok_cell_label(int(x[gid].item()))])
        tokens.append(self.cell_section_end)

        tokens.append(self.sink_connection_section_start)
        for record in sink_records:
            sg, spin, _scell = record["sink"]
            tokens.extend([self.sink, self._tok_gate_idx(int(sg)), self._tok_pin(str(spin)), self.src])
            src = record["src"]
            if src[0] == "CELL":
                tokens.extend([self._tok_src_idx(int(src[1])), self._tok_pin(str(src[2]))])
            else:
                tokens.extend([self.boundary_in, self._tok_pin("__BOUNDARY__")])
        tokens.append(self.sink_connection_section_end)
        if self.append_eos:
            tokens.append(self.eos)
        out = torch.tensor(tokens, dtype=torch.long)
        if self.max_length is not None and int(self.max_length) > 0:
            out = out[: int(self.max_length)]
        return out

    def decode(self, token_ids: torch.Tensor) -> Data:
        t = token_ids.reshape(-1).to(torch.long)
        toks = [int(x) for x in t.tolist() if int(x) not in {int(self.pad), int(self.sos)}]
        cells: Dict[int, int] = {}
        sink_records: List[Tuple[int, str, Tuple]] = []
        decode_valid = True
        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok == int(self.eos):
                break
            if tok == int(self.cell) and i + 2 < len(toks):
                gid = self._idx_from_gate_tok(toks[i + 1])
                cells[int(gid)] = self._cell_from_tok(toks[i + 2])
                i += 3
                continue
            if tok == int(self.sink) and i + 5 < len(toks):
                sg = self._idx_from_gate_tok(toks[i + 1])
                spin = self._pin_from_tok(toks[i + 2])
                if toks[i + 3] != int(self.src):
                    decode_valid = False
                    i += 1
                    continue
                src_kind = toks[i + 4]
                src_pin = self._pin_from_tok(toks[i + 5])
                if src_kind == int(self.boundary_in):
                    src = ("BOUNDARY_IN", src_pin)
                elif self.max_num_nodes is not None and int(self.idx_offset + self.max_num_nodes) <= src_kind < int(self.pin_offset):
                    src = ("CELL", self._idx_from_src_tok(src_kind), src_pin)
                elif src_kind == int(self.const0):
                    src = ("CONST0", src_pin)
                elif src_kind == int(self.const1):
                    src = ("CONST1", src_pin)
                else:
                    decode_valid = False
                    src = ("BOUNDARY_IN", "__BOUNDARY__")
                sink_records.append((int(sg), str(spin), src))
                i += 6
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
                    edges.append((int(src[1]), net_node))
                    roles.append(2)
                    edge_pin.append(local_pin_id(str(src[2])))
                else:
                    bnode = len(node_labels)
                    node_labels.append(int(self.boundary_stub_id))
                    edges.append((bnode, net_node))
                    roles.append(2)
                    edge_pin.append(local_pin_id("__BOUNDARY__"))
            net_node = src_to_net[key]
            edges.append((int(sg), net_node))
            roles.append(1)
            edge_pin.append(local_pin_id(str(spin)))
        x = torch.tensor(node_labels, dtype=torch.long) if node_labels else torch.zeros((0,), dtype=torch.long)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=torch.tensor(roles, dtype=torch.long), num_nodes=int(x.numel()))
        data.edge_role = data.edge_attr
        data.edge_pin_id = torch.tensor(edge_pin, dtype=torch.long) if edge_pin else torch.full((0,), -1, dtype=torch.long)
        data.pin_id_to_name = pins
        data.decode_valid = bool(decode_valid)
        return data

    def validate_sink_constraints(self, data: Data) -> Dict[str, float]:
        x = data.x.reshape(-1).to(torch.long)
        gates = self._gate_nodes(data)
        rec = self.serialize_records(data)
        sink_counter = Counter()
        src_counter = Counter()
        invalid_sink = invalid_src = invalid_cell = 0
        output_used_as_sink = input_used_as_src = 0
        for record in rec["sink_records"]:
            sg, spin, scell = record["sink"]
            inputs, outputs = self._spec_pins(str(scell))
            sink_counter[(int(sg), str(spin))] += 1
            if str(spin) not in inputs:
                invalid_sink += 1
                invalid_cell += 1
                if str(spin) in outputs:
                    output_used_as_sink += 1
            src = record["src"]
            if src[0] == "CELL":
                _kind, gidx, pin, cell = src
                inputs, outputs = self._spec_pins(str(cell))
                src_counter[(int(gidx), str(pin))] += 1
                if str(pin) not in outputs:
                    invalid_src += 1
                    invalid_cell += 1
                    if str(pin) in inputs:
                        input_used_as_src += 1
        required = 0
        covered = 0
        missing = 0
        fa = ha = dff = 0
        fa_s = ha_s = dff_qn = 0
        for gid in gates:
            cell = self._cell_name(int(x[gid].item()))
            inputs, _outputs = self._spec_pins(cell)
            for pin in inputs:
                required += 1
                if sink_counter.get((int(gates.index(gid)), str(pin)), 0) > 0:
                    covered += 1
                else:
                    missing += 1
            dense = int(gates.index(gid))
            if cell == "FA_X1":
                fa += 1
                fa_s += int(src_counter.get((dense, "S"), 0) > 0)
            elif cell == "HA_X1":
                ha += 1
                ha_s += int(src_counter.get((dense, "S"), 0) > 0)
            elif cell == "DFF_X1":
                dff += 1
                dff_qn += int(src_counter.get((dense, "QN"), 0) > 0)
        return {
            "invalid_sink_pin_count": float(invalid_sink),
            "invalid_src_pin_count": float(invalid_src),
            "invalid_cell_pin_count": float(invalid_cell),
            "output_used_as_sink_count": float(output_used_as_sink),
            "input_used_as_src_count": float(input_used_as_src),
            "duplicate_sink_count": float(sum(v - 1 for v in sink_counter.values() if v > 1)),
            "required_input_count": float(required),
            "covered_required_input_count": float(covered),
            "missing_required_input_load_count": float(missing),
            "required_input_coverage": float(covered / max(1, required)),
            "FA_X1_count": float(fa),
            "HA_X1_count": float(ha),
            "DFF_X1_count": float(dff),
            "FA_S_connected_rate": float(fa_s / max(1, fa)),
            "HA_S_connected_rate": float(ha_s / max(1, ha)),
            "DFF_QN_connected_rate": float(dff_qn / max(1, dff)),
        }
