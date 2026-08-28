from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from .batch_converter import BatchConverter


@dataclass(frozen=True)
class PinSlotSpec:
    inputs: List[str]
    outputs: List[str]
    min_required_outputs: int


class CircuitPinSlotTokenizer(object):
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
        pin_specs: Optional[Dict[str, PinSlotSpec]] = None,
        **kwargs,
    ):
        self.dataset_names = list(dataset_names) if dataset_names is not None else []
        self.max_length = int(max_length)
        self.truncation_length = truncation_length
        self.append_eos = bool(append_eos)
        self.net_id = int(net_id)
        self.boundary_stub_id = int(boundary_stub_id)
        self.label_to_cell = {int(k): str(v) for k, v in dict(label_to_cell).items()} if label_to_cell is not None else {}
        self.pin_specs = dict(pin_specs) if pin_specs is not None else {}

        self.sos = 0
        self.eos = 1
        self.pad = 2
        self.gate = 3
        self.end_gate = 4
        self.pin_in = 5
        self.pin_out = 6
        self.pin_opt_out = 7
        self.pin_skip = 8
        self.pin_excess = 9
        self.pin_unknown = 10
        self.ep_net = 11
        self.ep_stub = 12
        self.ep_pi = 13
        self.ep_const0 = 14
        self.ep_const1 = 15
        self.ep_input_pool = 16
        self.ep_private_out = 17
        self.special_toks = [
            'sos',
            'eos',
            'pad',
            'GATE',
            'END_GATE',
            'PIN_IN',
            'PIN_OUT',
            'PIN_OPT_OUT',
            'PIN_SKIP',
            'PIN_EXCESS',
            'PIN_UNKNOWN',
            'NET',
            'STUB',
            'PI',
            'CONST0',
            'CONST1',
            'INPUT_POOL',
            'PRIVATE_OUT',
        ]

        self.idx_offset = len(self.special_toks)
        self.max_num_nodes: Optional[int] = None
        self.num_node_types: int = 0
        self.pin_names: List[str] = []
        self.pin_name_to_id: Dict[str, int] = {}
        self.pin_offset: Optional[int] = None
        self.node_type_offset: Optional[int] = None

    @property
    def labeled_graph(self) -> bool:
        return False

    def set_num_nodes(self, max_num_nodes):
        if (self.max_num_nodes is None) or (self.max_num_nodes < int(max_num_nodes)):
            self.max_num_nodes = int(max_num_nodes)

    def set_num_node_and_edge_types(self, num_node_types=0, num_edge_types=0):
        self.num_node_types = int(num_node_types)
        if self.max_num_nodes is None:
            raise ValueError('set_num_nodes first')
        self.pin_offset = int(self.idx_offset + self.max_num_nodes)
        self.node_type_offset = int(self.pin_offset + len(self.pin_names))

    def set_pin_name_vocab(self, pin_names: List[str]) -> None:
        pin_names = [str(n) for n in pin_names]
        self.pin_names = list(pin_names)
        self.pin_name_to_id = {str(n): int(i) for i, n in enumerate(self.pin_names)}
        if self.max_num_nodes is not None:
            self.pin_offset = int(self.idx_offset + self.max_num_nodes)
            self.node_type_offset = int(self.pin_offset + len(self.pin_names))

    def __len__(self):
        if self.max_num_nodes is None or self.pin_offset is None or self.node_type_offset is None:
            raise ValueError('tokenizer is not initialized')
        return int(self.node_type_offset + self.num_node_types)

    def batch_converter(self):
        return BatchConverter(self, self.truncation_length)

    def _tok_idx(self, idx: int) -> int:
        return int(self.idx_offset + int(idx))

    def _tok_pin(self, name: str) -> int:
        if self.pin_offset is None:
            raise ValueError('pin vocab not initialized')
        pid = self.pin_name_to_id.get(str(name))
        if pid is None:
            raise KeyError(f'unknown pin name: {name}')
        return int(self.pin_offset + int(pid))

    def _tok_cell_label(self, label: int) -> int:
        if self.node_type_offset is None:
            raise ValueError('node types not initialized')
        return int(self.node_type_offset + int(label))

    def _pin_from_tok(self, tok: int) -> str:
        if self.pin_offset is None:
            raise ValueError('pin vocab not initialized')
        idx = int(tok - self.pin_offset)
        if idx < 0 or idx >= len(self.pin_names):
            return '__INVALID_PIN__'
        return str(self.pin_names[idx])

    def _cell_from_tok(self, tok: int) -> int:
        if self.node_type_offset is None:
            raise ValueError('node types not initialized')
        label = int(tok - self.node_type_offset)
        if label < 0 or label >= int(self.num_node_types):
            return int(self.net_id)
        return int(label)

    def _idx_from_tok(self, tok: int) -> int:
        idx = int(tok - self.idx_offset)
        if self.max_num_nodes is not None and (idx < 0 or idx >= int(self.max_num_nodes)):
            return -1
        return int(idx)

    def _gate_nodes(self, data: Data) -> List[int]:
        x = data.x.reshape(-1).to(torch.long)
        gate = (x != int(self.net_id)) & (x != int(self.boundary_stub_id))
        return [int(i) for i in torch.nonzero(gate, as_tuple=False).flatten().tolist()]

    def _build_pin_to_endpoint(self, data: Data, gate_idx: int) -> Dict[str, Tuple[str, int]]:
        x = data.x.reshape(-1).to(torch.long)
        edge_index = data.edge_index.to(torch.long)
        edge_role = getattr(data, 'edge_role', None)
        if edge_role is None:
            edge_role = getattr(data, 'edge_attr', None)
        edge_pin = getattr(data, 'edge_pin_id', None)
        pin_id_to_name = getattr(data, 'pin_id_to_name', None)
        syn = getattr(data, 'edge_is_synthetic_endpoint', None)

        out: Dict[str, Tuple[str, int]] = {}
        if edge_pin is None or pin_id_to_name is None:
            return out
        if edge_role is None:
            return out
        if int(edge_pin.numel()) != int(edge_index.size(1)):
            return out
        if int(edge_role.numel()) != int(edge_index.size(1)):
            return out

        for eidx, (src, dst) in enumerate(edge_index.t().tolist()):
            if int(src) != int(gate_idx):
                continue
            if syn is not None and int(syn.numel()) == int(edge_index.size(1)):
                if int(syn[int(eidx)].item()) != 0:
                    continue
            pid = int(edge_pin[int(eidx)].item())
            if pid < 0 or pid >= len(pin_id_to_name):
                continue
            pname = str(pin_id_to_name[pid])
            if pname in out:
                continue
            endpoint = int(dst)
            if int(x[endpoint].item()) == int(self.boundary_stub_id):
                out[pname] = ('STUB', endpoint)
            elif int(x[endpoint].item()) == int(self.net_id):
                out[pname] = ('NET', endpoint)
        return out

    def _incident_edges(self, data: Data, gate_idx: int):
        edge_index = data.edge_index.to(torch.long)
        edge_role = getattr(data, 'edge_role', None)
        if edge_role is None:
            edge_role = getattr(data, 'edge_attr', None)
        edge_pin = getattr(data, 'edge_pin_id', None)
        pin_id_to_name = getattr(data, 'pin_id_to_name', None)
        out = []
        if edge_role is None or int(edge_role.numel()) != int(edge_index.size(1)):
            return out
        for eidx, (src, dst) in enumerate(edge_index.t().tolist()):
            if int(src) != int(gate_idx):
                continue
            role = int(edge_role[int(eidx)].item())
            pname = None
            if edge_pin is not None and pin_id_to_name is not None and int(edge_pin.numel()) == int(edge_index.size(1)):
                pid = int(edge_pin[int(eidx)].item())
                if 0 <= pid < len(pin_id_to_name):
                    pname = str(pin_id_to_name[pid])
            out.append((int(src), int(dst), int(role), pname))
        return out

    def __call__(self, data: Data):
        return self.tokenize(data)

    def tokenize(self, data: Data):
        if self.max_num_nodes is None or self.node_type_offset is None:
            raise ValueError('tokenizer not initialized')

        x = data.x.reshape(-1).to(torch.long)
        syn_base = int(getattr(data, 'synthetic_endpoint_base', int(x.numel())))
        syn_count = int(getattr(data, 'synthetic_endpoint_count', 0))
        tokens: List[int] = [int(self.sos)]

        for gate_idx in self._gate_nodes(data):
            cell_label = int(x[gate_idx].item())
            cell_name = self.label_to_cell.get(int(cell_label))
            spec = None
            if cell_name is not None:
                spec = self.pin_specs.get(str(cell_name))
            if spec is None:
                spec = PinSlotSpec(inputs=[], outputs=[], min_required_outputs=0)

            pin_to_ep = self._build_pin_to_endpoint(data, gate_idx)
            assigned = set()

            tokens.extend([int(self.gate), self._tok_idx(gate_idx), self._tok_cell_label(cell_label)])

            for pname in spec.inputs:
                ep = pin_to_ep.get(str(pname))
                tokens.append(int(self.pin_in))
                tokens.append(self._tok_pin(str(pname)))
                if ep is None:
                    tokens.append(int(self.ep_input_pool))
                    if syn_count > 0:
                        pid = int(self.pin_name_to_id.get(str(pname), 0))
                        endpoint = int(syn_base + (pid % max(1, syn_count)))
                        tokens.append(self._tok_idx(endpoint))
                    else:
                        tokens.append(self._tok_idx(int(gate_idx)))
                else:
                    ep_type, ep_id = ep
                    assigned.add((int(gate_idx), int(ep_id)))
                    if ep_type == 'STUB':
                        tokens.append(int(self.ep_stub))
                        tokens.append(self._tok_idx(ep_id))
                    else:
                        tokens.append(int(self.ep_net))
                        tokens.append(self._tok_idx(ep_id))

            required_out = int(max(0, spec.min_required_outputs))
            out_pins = list(spec.outputs)
            min_out_pins = out_pins[:required_out]
            opt_out_pins = out_pins[required_out:]

            for pname in min_out_pins:
                ep = pin_to_ep.get(str(pname))
                tokens.append(int(self.pin_out))
                tokens.append(self._tok_pin(str(pname)))
                if ep is None:
                    tokens.append(int(self.ep_private_out))
                    if syn_count > 0:
                        pid = int(self.pin_name_to_id.get(str(pname), 0))
                        endpoint = int(syn_base + (pid % max(1, syn_count)))
                        tokens.append(self._tok_idx(endpoint))
                    else:
                        tokens.append(self._tok_idx(int(gate_idx)))
                else:
                    ep_type, ep_id = ep
                    assigned.add((int(gate_idx), int(ep_id)))
                    if ep_type == 'STUB':
                        tokens.append(int(self.ep_stub))
                        tokens.append(self._tok_idx(ep_id))
                    else:
                        tokens.append(int(self.ep_net))
                        tokens.append(self._tok_idx(ep_id))

            for pname in opt_out_pins:
                ep = pin_to_ep.get(str(pname))
                if ep is None:
                    tokens.append(int(self.pin_skip))
                    tokens.append(self._tok_pin(str(pname)))
                else:
                    ep_type, ep_id = ep
                    assigned.add((int(gate_idx), int(ep_id)))
                    tokens.append(int(self.pin_opt_out))
                    tokens.append(self._tok_pin(str(pname)))
                    if ep_type == 'STUB':
                        tokens.append(int(self.ep_stub))
                        tokens.append(self._tok_idx(ep_id))
                    else:
                        tokens.append(int(self.ep_net))
                        tokens.append(self._tok_idx(ep_id))

            extras = []
            for src, dst, role, pname in self._incident_edges(data, gate_idx):
                if (int(src), int(dst)) in assigned:
                    continue
                if role not in {0, 3}:
                    continue
                extras.append((int(dst), str(pname) if pname is not None else '', int(role)))
            extras = sorted(extras)
            for dst, pname, role in extras:
                tok_role = int(self.pin_excess) if int(role) == 3 else int(self.pin_unknown)
                tokens.append(tok_role)
                tokens.append(self._tok_pin(pname if pname else ('__EXCESS__' if int(role) == 3 else '__UNKNOWN__')))
                if int(data.x.reshape(-1)[dst].item()) == int(self.boundary_stub_id):
                    tokens.append(int(self.ep_stub))
                    tokens.append(self._tok_idx(dst))
                else:
                    tokens.append(int(self.ep_net))
                    tokens.append(self._tok_idx(dst))

            tokens.append(int(self.end_gate))

        if self.append_eos:
            tokens.append(int(self.eos))

        out = torch.tensor(tokens, dtype=torch.long)
        if self.max_length is not None and int(self.max_length) > 0:
            out = out[: int(self.max_length)]
        return out

    def decode(self, token_ids: torch.Tensor) -> Data:
        if self.max_num_nodes is None or self.pin_offset is None or self.node_type_offset is None:
            raise ValueError('tokenizer not initialized')
        t = token_ids.reshape(-1).to(torch.long)
        t = t[(t != int(self.pad)) & (t != int(self.sos))]
        pos = 0
        x_labels: Dict[int, int] = {}
        edges: List[Tuple[int, int]] = []
        roles: List[int] = []
        pin_names: List[str] = []
        pin_name_to_local: Dict[str, int] = {}
        edge_pin: List[int] = []
        synthetic_edge: List[int] = []
        edge_stub_role: List[int] = []
        edge_stub_owner_gate_id: List[int] = []
        edge_stub_pin_role: List[int] = []
        edge_stub_pin_id: List[int] = []
        edge_endpoint_type: List[int] = []

        synthetic_nodes = set()

        max_node = -1

        decode_valid = True
        decode_incomplete_gate = False
        decode_eos_before_gate_completion = False
        decode_invalid_token = False

        def pin_id(name: str) -> int:
            if name not in pin_name_to_local:
                pin_name_to_local[name] = len(pin_names)
                pin_names.append(name)
            return int(pin_name_to_local[name])

        def ensure_node(idx: int, label: int) -> None:
            nonlocal max_node, decode_valid
            if int(idx) < 0:
                return
            max_node = max(max_node, int(idx))
            existing = x_labels.get(int(idx))
            if existing is not None and int(existing) != int(label):
                decode_valid_flags['endpoint_id_collision'] = True
                decode_valid = False
                return
            x_labels[int(idx)] = int(label)

        decode_valid_flags = {
            'endpoint_id_collision': False,
        }

        def append_edge(gate_idx: int, endpoint_idx: int, role: int, pname: str, is_syn: int, ep_type: int, source_tok: int) -> None:
            pid = pin_id(pname)
            edges.append((int(gate_idx), int(endpoint_idx)))
            roles.append(int(role))
            edge_pin.append(int(pid))
            synthetic_edge.append(int(is_syn))
            edge_endpoint_type.append(int(ep_type))
            stub_role = 0
            if int(ep_type) == int(self.ep_stub):
                if int(role) == 1:
                    stub_role = 1  # load_stub
                elif int(source_tok) == int(self.pin_opt_out):
                    stub_role = 3  # optional_driver_stub
                elif int(role) == 2:
                    stub_role = 2  # driver_stub
                elif int(role) == 3:
                    stub_role = 4  # unknown/excess stub
                else:
                    stub_role = 4
            edge_stub_role.append(int(stub_role))
            edge_stub_owner_gate_id.append(int(gate_idx) if int(stub_role) != 0 else -1)
            edge_stub_pin_role.append(int(role) if int(stub_role) != 0 else 0)
            edge_stub_pin_id.append(int(pid) if int(stub_role) != 0 else -1)
            if int(is_syn) != 0:
                synthetic_nodes.add(int(endpoint_idx))

        while pos < int(t.numel()):
            tok = int(t[pos].item())
            if tok == int(self.eos):
                break
            if tok != int(self.gate):
                pos += 1
                continue
            if pos + 2 >= int(t.numel()):
                decode_valid = False
                decode_incomplete_gate = True
                break
            gate_idx = self._idx_from_tok(int(t[pos + 1].item()))
            cell_label = self._cell_from_tok(int(t[pos + 2].item()))
            if int(gate_idx) < 0:
                decode_valid = False
                decode_invalid_token = True
                pos += 1
                continue
            if int(t[pos + 2].item()) < int(self.node_type_offset) or int(t[pos + 2].item()) >= int(self.node_type_offset + self.num_node_types):
                decode_valid = False
                decode_invalid_token = True
            ensure_node(gate_idx, cell_label)
            pos += 3
            gate_complete = False
            while pos < int(t.numel()):
                tok = int(t[pos].item())
                if tok == int(self.eos):
                    decode_valid = False
                    decode_eos_before_gate_completion = True
                    gate_complete = False
                    pos = int(t.numel())
                    break
                if tok == int(self.end_gate):
                    pos += 1
                    gate_complete = True
                    break
                if tok == int(self.pin_in) or tok == int(self.pin_out) or tok == int(self.pin_opt_out):
                    if pos + 3 >= int(t.numel()):
                        pos = int(t.numel())
                        break
                    pin_tok = int(t[pos + 1].item())
                    pname = self._pin_from_tok(pin_tok)
                    if pname == '__INVALID_PIN__':
                        decode_valid = False
                        decode_invalid_token = True
                    ep_type = int(t[pos + 2].item())
                    ep_val = int(t[pos + 3].item())
                    role = 1 if tok == int(self.pin_in) else 2

                    endpoint_idx = None
                    is_syn = 0
                    if ep_type == int(self.ep_net):
                        endpoint_idx = self._idx_from_tok(ep_val)
                        ensure_node(endpoint_idx, int(self.net_id))
                    elif ep_type == int(self.ep_stub):
                        endpoint_idx = self._idx_from_tok(ep_val)
                        ensure_node(endpoint_idx, int(self.boundary_stub_id))
                    elif ep_type == int(self.ep_pi):
                        endpoint_idx = self._idx_from_tok(ep_val)
                        ensure_node(endpoint_idx, int(self.net_id))
                    elif ep_type == int(self.ep_private_out):
                        endpoint_idx = self._idx_from_tok(ep_val)
                        ensure_node(endpoint_idx, int(self.net_id))
                        is_syn = 1
                    elif ep_type == int(self.ep_input_pool):
                        endpoint_idx = self._idx_from_tok(ep_val)
                        ensure_node(endpoint_idx, int(self.net_id))
                        is_syn = 1
                    elif ep_type == int(self.ep_const0):
                        endpoint_idx = self._idx_from_tok(ep_val)
                        ensure_node(endpoint_idx, int(self.net_id))
                        is_syn = 1
                    elif ep_type == int(self.ep_const1):
                        endpoint_idx = self._idx_from_tok(ep_val)
                        ensure_node(endpoint_idx, int(self.net_id))
                        is_syn = 1

                    if endpoint_idx is not None and int(endpoint_idx) >= 0:
                        append_edge(gate_idx, endpoint_idx, role, pname, is_syn, ep_type, tok)
                    pos += 4
                    continue
                if tok == int(self.pin_excess) or tok == int(self.pin_unknown):
                    if pos + 3 >= int(t.numel()):
                        pos = int(t.numel())
                        break
                    pin_tok = int(t[pos + 1].item())
                    pname = self._pin_from_tok(pin_tok)
                    if pname == '__INVALID_PIN__':
                        decode_valid = False
                        decode_invalid_token = True
                    ep_type = int(t[pos + 2].item())
                    ep_val = int(t[pos + 3].item())
                    role = 3 if tok == int(self.pin_excess) else 0

                    endpoint_idx = None
                    if ep_type == int(self.ep_net):
                        endpoint_idx = self._idx_from_tok(ep_val)
                        ensure_node(endpoint_idx, int(self.net_id))
                    elif ep_type == int(self.ep_stub):
                        endpoint_idx = self._idx_from_tok(ep_val)
                        ensure_node(endpoint_idx, int(self.boundary_stub_id))
                    if endpoint_idx is not None and int(endpoint_idx) >= 0:
                        append_edge(gate_idx, endpoint_idx, role, pname, 0, ep_type, tok)
                    pos += 4
                    continue
                if tok == int(self.pin_skip):
                    pos += 2
                    continue
                pos += 1

            if not gate_complete:
                decode_valid = False
                decode_incomplete_gate = True
                break

        num_nodes = int(max_node + 1) if max_node >= 0 else 0
        x = torch.full((num_nodes,), int(self.net_id), dtype=torch.long)
        for idx, label in x_labels.items():
            if 0 <= int(idx) < num_nodes:
                x[int(idx)] = int(label)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(roles, dtype=torch.long) if roles else torch.zeros((0,), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        data.edge_role = edge_attr
        data.edge_pin_id = torch.tensor(edge_pin, dtype=torch.long) if edge_pin else torch.full((0,), -1, dtype=torch.long)
        data.pin_id_to_name = list(pin_names)
        data.edge_is_synthetic_endpoint = torch.tensor(synthetic_edge, dtype=torch.long) if synthetic_edge else torch.zeros((0,), dtype=torch.long)
        data.edge_endpoint_type = torch.tensor(edge_endpoint_type, dtype=torch.long) if edge_endpoint_type else torch.zeros((0,), dtype=torch.long)
        data.edge_stub_role = torch.tensor(edge_stub_role, dtype=torch.long) if edge_stub_role else torch.zeros((0,), dtype=torch.long)
        data.stub_owner_gate_id = torch.tensor(edge_stub_owner_gate_id, dtype=torch.long) if edge_stub_owner_gate_id else torch.full((0,), -1, dtype=torch.long)
        data.stub_pin_role = torch.tensor(edge_stub_pin_role, dtype=torch.long) if edge_stub_pin_role else torch.zeros((0,), dtype=torch.long)
        data.stub_pin_id = torch.tensor(edge_stub_pin_id, dtype=torch.long) if edge_stub_pin_id else torch.full((0,), -1, dtype=torch.long)
        if synthetic_nodes:
            base = int(min(synthetic_nodes))
            count = int(max(synthetic_nodes) - base + 1)
            data.synthetic_endpoint_base = int(base)
            data.synthetic_endpoint_count = int(count)
        data.decode_valid = bool(decode_valid)
        data.decode_incomplete_gate = bool(decode_incomplete_gate)
        data.decode_eos_before_gate_completion = bool(decode_eos_before_gate_completion)
        data.decode_invalid_token = bool(decode_invalid_token)
        data.decode_endpoint_id_collision = bool(decode_valid_flags['endpoint_id_collision'])
        return data
