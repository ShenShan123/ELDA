from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from .batch_converter import BatchConverter


CRITICAL_ARITH_OUTPUTS = {("FA_X1", "S"), ("HA_X1", "S")}


class CircuitSourceNetV4Tokenizer(object):
    """Source-net-first tokenizer for CIRCUIT_SOURCE_NET_PARTITION_V4_1.

    The representation separates cell inventory from driver-load net records so
    the model has to emit source organization instead of relying on practical
    exporter fallback to recover missing connectivity.
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
        self.is_source_net_v4_tokenizer = True

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
            "FANOUT_BUCKET_0",
            "FANOUT_BUCKET_1",
            "FANOUT_BUCKET_2_4",
            "FANOUT_BUCKET_5_8",
            "FANOUT_BUCKET_9_16",
            "FANOUT_BUCKET_17_PLUS",
            "CRITICAL_OUTPUT_BUCKET_NONE",
            "CRITICAL_OUTPUT_BUCKET_LOW",
            "CRITICAL_OUTPUT_BUCKET_HIGH",
            "POOL_RISK_BUCKET_LOW",
            "POOL_RISK_BUCKET_MID",
            "POOL_RISK_BUCKET_HIGH",
            "CELL_SECTION_START",
            "CELL",
            "CELL_SECTION_END",
            "NET_SECTION_START",
            "NET",
            "DRIVER",
            "PI_DRIVER",
            "BOUNDARY_DRIVER",
            "LOAD",
            "END_NET",
            "NET_SECTION_END",
            "SOURCE_ROLE_ARITH",
            "SOURCE_ROLE_CLOCK",
            "SOURCE_ROLE_NAND",
            "SOURCE_ROLE_DATA",
            "SOURCE_ROLE_MIXED",
            "SOURCE_ROLE_BOUNDARY",
            "CRITICAL_OUTPUT",
            "OPTIONAL_OUTPUT_CRITICAL",
            "OPTIONAL_OUTPUT_NORMAL",
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

    def set_num_nodes(self, max_num_nodes):
        if self.max_num_nodes is None or self.max_num_nodes < int(max_num_nodes):
            self.max_num_nodes = int(max_num_nodes)

    def set_pin_name_vocab(self, pin_names: List[str]) -> None:
        names = sorted({str(x) for x in pin_names} | {"__UNKNOWN__"})
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

    def batch_converter(self):
        return BatchConverter(self, self.truncation_length)

    def __call__(self, data: Data):
        return self.tokenize(data)

    def _tok_gate_idx(self, idx: int) -> int:
        return int(self.idx_offset + int(idx))

    def _tok_net_idx(self, idx: int) -> int:
        if self.max_num_nodes is None:
            raise ValueError("tokenizer not initialized")
        return int(self.idx_offset + self.max_num_nodes + int(idx))

    def _idx_from_gate_tok(self, tok: int) -> int:
        return int(tok - self.idx_offset)

    def _idx_from_net_tok(self, tok: int) -> int:
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
        # Older partition graphs encode optional/excess outputs as role=3 but
        # often lose the exact pin id. Recover the pin only when Liberty makes
        # it unambiguous, e.g. FA/HA have CO as the required output and S as the
        # sole remaining output. Ambiguous role=3 records stay unknown and are
        # not treated as normal V4 source-net drivers.
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
        rows = fixed
        return rows

    def _fanout_bucket_token(self, n: int) -> int:
        if n <= 0:
            return self.fanout_bucket_0
        if n == 1:
            return self.fanout_bucket_1
        if n <= 4:
            return self.fanout_bucket_2_4
        if n <= 8:
            return self.fanout_bucket_5_8
        if n <= 16:
            return self.fanout_bucket_9_16
        return self.fanout_bucket_17_plus

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

    def _source_role_token(self, loads: List[Tuple[int, str]]) -> int:
        pins = {pin for _, pin in loads}
        has_clock = "CK" in pins
        has_arith = bool(pins & {"A", "B", "CI"})
        has_nand = bool(pins & {"A1", "A2", "A3", "A4"})
        classes = sum(bool(x) for x in [has_clock, has_arith, has_nand])
        if classes > 1:
            return self.source_role_mixed
        if has_clock:
            return self.source_role_clock
        if has_arith:
            return self.source_role_arith
        if has_nand:
            return self.source_role_nand
        return self.source_role_data

    def serialize_records(self, data: Data) -> Dict[str, object]:
        x = data.x.reshape(-1).to(torch.long)
        gates = self._gate_nodes(data)
        gate_to_dense = {gid: i for i, gid in enumerate(gates)}
        edge_rows = self._edge_rows(data)
        net_map: Dict[int, Dict[str, object]] = defaultdict(lambda: {"drivers": [], "loads": []})
        for src, dst, role, pin in edge_rows:
            if src not in gate_to_dense:
                continue
            if int(dst) >= int(x.numel()):
                continue
            dst_label = int(x[dst].item())
            if dst_label not in {int(self.net_id), int(self.boundary_stub_id)}:
                continue
            cell_label = int(x[src].item())
            cell_name = self._cell_name(cell_label)
            item = (gate_to_dense[src], pin, cell_name, src, role)
            inputs, outputs = self._spec_pins(cell_name)
            if role == 2 or (role == 3 and str(pin) in outputs):
                net_map[int(dst)]["drivers"].append(item)
            elif role == 1:
                net_map[int(dst)]["loads"].append(item)

        no_driver = 0
        multi_driver = 0
        records = []
        for dense_nid, endpoint in enumerate(sorted(net_map.keys())):
            drivers = list(net_map[endpoint]["drivers"])
            loads = [(int(g), str(pin)) for g, pin, _cell, _src, _role in net_map[endpoint]["loads"]]
            if not loads and not drivers:
                continue
            if not drivers:
                no_driver += 1
            if len(drivers) > 1:
                multi_driver += 1
            primary = next((d for d in drivers if int(d[4]) == 2), None)
            driver = primary if primary is not None else (drivers[0] if drivers else None)
            role_tok = self._source_role_token(loads) if loads else self.source_role_data
            records.append(
                {
                    "endpoint": int(endpoint),
                    "dense_net": int(dense_nid),
                    "driver": driver,
                    "drivers": drivers,
                    "loads": loads,
                    "role_token": role_tok,
                    "fanout": len(loads),
                    "critical": bool(driver and (str(driver[2]), str(driver[1])) in CRITICAL_ARITH_OUTPUTS),
                    "optional_sidecar": False,
                    "boundary": bool(int(x[endpoint].item()) == int(self.boundary_stub_id)),
                }
            )
            # Preserve legal optional outputs without creating multi-driver V4
            # source nets. In V2.1 graphs FA/HA.S and DFF.QN are role=3 edges
            # and can share an endpoint with a normal role=2 driver. They are
            # output-coverage facts, not additional source-net drivers.
            for extra in drivers:
                if extra is driver or int(extra[4]) != 3:
                    continue
                if (str(extra[2]), str(extra[1])) in CRITICAL_ARITH_OUTPUTS or str(extra[1]) in self._spec_pins(str(extra[2]))[1]:
                    records.append(
                        {
                            "endpoint": int(endpoint),
                            "dense_net": int(len(records)),
                            "driver": extra,
                            "drivers": [extra],
                            "loads": [],
                            "role_token": self.source_role_data,
                            "fanout": 0,
                            "critical": bool((str(extra[2]), str(extra[1])) in CRITICAL_ARITH_OUTPUTS),
                            "optional_sidecar": True,
                            "boundary": bool(int(x[endpoint].item()) == int(self.boundary_stub_id)),
                        }
                    )
        return {"gates": gates, "records": records, "no_driver": no_driver, "multi_driver": multi_driver}

    def tokenize(self, data: Data):
        if self.max_num_nodes is None or self.node_type_offset is None:
            raise ValueError("tokenizer not initialized")
        x = data.x.reshape(-1).to(torch.long)
        rec = self.serialize_records(data)
        gates: List[int] = rec["gates"]
        records: List[Dict[str, object]] = rec["records"]
        tokens: List[int] = [self.sos, self.contract_start, self._gc_bucket_token(len(gates))]
        cut_bucket = int(getattr(data, "schema_target_driver_stub_bucket", 0))
        tokens.append(self.cut_edge_bucket_1_4 if cut_bucket > 0 else self.cut_edge_bucket_0)
        unique_sources = sum(1 for r in records if r.get("driver") is not None)
        diversity = unique_sources / max(1, len(gates))
        tokens.append(self.source_diversity_bucket_high if diversity >= 0.5 else (self.source_diversity_bucket_mid if diversity >= 0.2 else self.source_diversity_bucket_low))
        critical_count = sum(1 for r in records if r.get("critical"))
        tokens.append(self.critical_output_bucket_high if critical_count >= 2 else (self.critical_output_bucket_low if critical_count else self.critical_output_bucket_none))
        tokens.extend([self.pool_risk_bucket_low, self.contract_end])

        tokens.append(self.cell_section_start)
        for dense, gid in enumerate(gates):
            tokens.extend([self.cell, self._tok_gate_idx(dense), self._tok_cell_label(int(x[gid].item()))])
        tokens.append(self.cell_section_end)

        tokens.append(self.net_section_start)
        for rec_i, record in enumerate(records):
            tokens.extend([self.net, self._tok_net_idx(rec_i), self._fanout_bucket_token(int(record["fanout"])), int(record["role_token"])])
            if record.get("critical"):
                tokens.append(self.critical_output)
            if record.get("optional_sidecar"):
                tokens.append(self.optional_output_critical if record.get("critical") else self.optional_output_normal)
            driver = record.get("driver")
            if driver is None:
                tokens.extend([self.boundary_driver, self._tok_net_idx(rec_i)])
            else:
                gidx, pin, _cell_name, _src, _role = driver
                tokens.extend([self.driver, self._tok_gate_idx(int(gidx)), self._tok_pin(str(pin))])
            for gidx, pin in record["loads"]:
                tokens.extend([self.load, self._tok_gate_idx(int(gidx)), self._tok_pin(str(pin))])
            tokens.append(self.end_net)
        tokens.append(self.net_section_end)
        if self.append_eos:
            tokens.append(self.eos)
        out = torch.tensor(tokens, dtype=torch.long)
        if self.max_length is not None and int(self.max_length) > 0:
            out = out[: int(self.max_length)]
        return out

    def decode(self, token_ids: torch.Tensor) -> Data:
        t = token_ids.reshape(-1).to(torch.long)
        t = t[(t != int(self.pad)) & (t != int(self.sos))]
        cells: Dict[int, int] = {}
        nets: List[Dict[str, object]] = []
        pos = 0
        decode_valid = True
        while pos < int(t.numel()):
            tok = int(t[pos].item())
            if tok == self.eos:
                break
            if tok == self.cell and pos + 2 < int(t.numel()):
                gid = self._idx_from_gate_tok(int(t[pos + 1].item()))
                cells[int(gid)] = self._cell_from_tok(int(t[pos + 2].item()))
                pos += 3
                continue
            if tok == self.net:
                record = {"driver": None, "loads": [], "boundary_driver": False}
                pos += 1
                while pos < int(t.numel()):
                    ntok = int(t[pos].item())
                    if ntok in {self.end_net, self.eos}:
                        pos += 1
                        break
                    if ntok == self.driver and pos + 2 < int(t.numel()):
                        record["driver"] = (self._idx_from_gate_tok(int(t[pos + 1].item())), self._pin_from_tok(int(t[pos + 2].item())))
                        pos += 3
                        continue
                    if ntok in {self.boundary_driver, self.pi_driver}:
                        record["driver"] = None
                        record["boundary_driver"] = True
                        pos += 2 if pos + 1 < int(t.numel()) else 1
                        continue
                    if ntok == self.load and pos + 2 < int(t.numel()):
                        record["loads"].append((self._idx_from_gate_tok(int(t[pos + 1].item())), self._pin_from_tok(int(t[pos + 2].item()))))
                        pos += 3
                        continue
                    pos += 1
                nets.append(record)
                continue
            pos += 1

        node_labels: List[int] = []
        for gid in sorted(cells):
            while len(node_labels) <= gid:
                node_labels.append(int(self.net_id))
            node_labels[gid] = int(cells[gid])
        net_base = len(node_labels)
        pin_to_id: Dict[str, int] = {}
        pins: List[str] = []
        edges, roles, edge_pin = [], [], []

        def local_pin_id(name: str) -> int:
            if name not in pin_to_id:
                pin_to_id[name] = len(pins)
                pins.append(name)
            return pin_to_id[name]

        for ni, record in enumerate(nets):
            net_node = net_base + ni
            node_labels.append(int(self.net_id))
            driver = record.get("driver")
            if driver is not None and int(driver[0]) in cells:
                edges.append((int(driver[0]), net_node))
                roles.append(2)
                edge_pin.append(local_pin_id(str(driver[1])))
            elif bool(record.get("boundary_driver", False)):
                boundary_node = len(node_labels)
                node_labels.append(int(self.boundary_stub_id))
                edges.append((boundary_node, net_node))
                roles.append(2)
                edge_pin.append(local_pin_id("__BOUNDARY__"))
            else:
                decode_valid = False
            for gidx, pin in record.get("loads", []):
                if int(gidx) not in cells:
                    decode_valid = False
                    continue
                edges.append((int(gidx), net_node))
                roles.append(1)
                edge_pin.append(local_pin_id(str(pin)))
        x = torch.tensor(node_labels, dtype=torch.long) if node_labels else torch.zeros((0,), dtype=torch.long)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=torch.tensor(roles, dtype=torch.long), num_nodes=int(x.numel()))
        data.edge_role = data.edge_attr
        data.edge_pin_id = torch.tensor(edge_pin, dtype=torch.long) if edge_pin else torch.full((0,), -1, dtype=torch.long)
        data.pin_id_to_name = pins
        data.decode_valid = bool(decode_valid)
        return data

    def source_structure_stats(self, data: Data) -> Dict[str, float]:
        rec = self.serialize_records(data)
        fanouts = [int(r["fanout"]) for r in rec["records"]]
        repeated = Counter()
        for r in rec["records"]:
            for gidx, pin in r["loads"]:
                repeated[(gidx, pin)] += 1
        critical = sum(1 for r in rec["records"] if r.get("critical"))
        return {
            "gate_count": float(len(rec["gates"])),
            "net_count": float(len(rec["records"])),
            "no_driver_net_count": float(rec["no_driver"]),
            "multi_driver_net_count": float(rec["multi_driver"]),
            "max_source_fanout": float(max(fanouts) if fanouts else 0),
            "critical_arith_output_net_count": float(critical),
            "repeated_input_signature_proxy": float(sum(1 for v in repeated.values() if v > 1)),
        }

    def _spec_pins(self, cell_name: str) -> Tuple[set, set]:
        spec = self.pin_specs.get(str(cell_name))
        if spec is None:
            return set(), set()
        return set(str(p) for p in getattr(spec, "inputs", [])), set(str(p) for p in getattr(spec, "outputs", []))

    def validate_pin_constraints(self, data: Data) -> Dict[str, float]:
        """Validate source-net records against exact Liberty cell pin specs."""
        x = data.x.reshape(-1).to(torch.long)
        gate_nodes = self._gate_nodes(data)
        edge_rows = self._edge_rows(data)
        input_loads = Counter()
        output_drivers = Counter()
        invalid_driver_pin_count = 0
        invalid_load_pin_count = 0
        invalid_cell_pin_count = 0
        output_used_as_load_count = 0
        input_used_as_driver_count = 0
        unresolved_optional_output_count = 0

        for src, _dst, role, pin in edge_rows:
            if src >= int(x.numel()):
                continue
            label = int(x[src].item())
            if label in {int(self.net_id), int(self.boundary_stub_id)}:
                continue
            cell_name = self._cell_name(label)
            inputs, outputs = self._spec_pins(cell_name)
            pin = str(pin)
            if role == 2 or (role == 3 and pin in outputs):
                output_drivers[(int(src), pin)] += 1
                if pin not in outputs:
                    invalid_driver_pin_count += 1
                    invalid_cell_pin_count += 1
                    if pin in inputs:
                        input_used_as_driver_count += 1
            elif role == 1:
                input_loads[(int(src), pin)] += 1
                if pin not in inputs:
                    invalid_load_pin_count += 1
                    invalid_cell_pin_count += 1
                    if pin in outputs:
                        output_used_as_load_count += 1
            elif role == 3:
                if pin in inputs:
                    invalid_driver_pin_count += 1
                    invalid_cell_pin_count += 1
                    input_used_as_driver_count += 1
                else:
                    unresolved_optional_output_count += 1

        missing_required_input_load_count = 0
        duplicate_input_load_count = 0
        fa_total = ha_total = dff_total = 0
        fa_s_connected = ha_s_connected = dff_qn_connected = 0
        critical_optional_output_missing_count = 0
        for gid in gate_nodes:
            cell_name = self._cell_name(int(x[gid].item()))
            inputs, outputs = self._spec_pins(cell_name)
            for pin in inputs:
                if input_loads.get((int(gid), str(pin)), 0) <= 0:
                    missing_required_input_load_count += 1
            for pin in inputs:
                n = input_loads.get((int(gid), str(pin)), 0)
                if n > 1:
                    duplicate_input_load_count += int(n - 1)
            if cell_name == "FA_X1":
                fa_total += 1
                if output_drivers.get((int(gid), "S"), 0) > 0:
                    fa_s_connected += 1
                else:
                    critical_optional_output_missing_count += 1
            elif cell_name == "HA_X1":
                ha_total += 1
                if output_drivers.get((int(gid), "S"), 0) > 0:
                    ha_s_connected += 1
                else:
                    critical_optional_output_missing_count += 1
            elif cell_name == "DFF_X1":
                dff_total += 1
                if output_drivers.get((int(gid), "QN"), 0) > 0:
                    dff_qn_connected += 1

        return {
            "invalid_driver_pin_count": float(invalid_driver_pin_count),
            "invalid_load_pin_count": float(invalid_load_pin_count),
            "invalid_cell_pin_count": float(invalid_cell_pin_count),
            "output_used_as_load_count": float(output_used_as_load_count),
            "input_used_as_driver_count": float(input_used_as_driver_count),
            "unresolved_optional_output_count": float(unresolved_optional_output_count),
            "missing_required_input_load_count": float(missing_required_input_load_count),
            "duplicate_input_load_count": float(duplicate_input_load_count),
            "critical_optional_output_missing_count": float(critical_optional_output_missing_count),
            "FA_S_connected_rate": float(fa_s_connected / fa_total) if fa_total else 0.0,
            "HA_S_connected_rate": float(ha_s_connected / ha_total) if ha_total else 0.0,
            "DFF_QN_connected_rate": float(dff_qn_connected / dff_total) if dff_total else 0.0,
            "FA_X1_count": float(fa_total),
            "HA_X1_count": float(ha_total),
            "DFF_X1_count": float(dff_total),
        }
