from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from .circuit_pin_slot_tokenizer import CircuitPinSlotTokenizer, PinSlotSpec


class CircuitPinSlotV21Tokenizer(CircuitPinSlotTokenizer):
    """V2.1 pin-slot tokenizer with explicit stub roles and namespaced endpoint ids."""

    is_pin_slot_tokenizer = True
    pin_slot_version = "v2.1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        self.ep_load_stub = 12
        self.ep_driver_stub = 13
        self.ep_optional_driver_stub = 14
        self.ep_unknown_stub = 15
        self.ep_pi = 16
        self.ep_const0 = 17
        self.ep_const1 = 18
        self.ep_input_pool = 19
        self.ep_private_out = 20
        self.special_toks = [
            "sos",
            "eos",
            "pad",
            "GATE",
            "END_GATE",
            "PIN_IN",
            "PIN_OUT",
            "PIN_OPT_OUT",
            "PIN_SKIP",
            "PIN_EXCESS",
            "PIN_UNKNOWN",
            "NET",
            "LOAD_STUB",
            "DRIVER_STUB",
            "OPTIONAL_DRIVER_STUB",
            "UNKNOWN_STUB",
            "PI",
            "CONST0",
            "CONST1",
            "INPUT_POOL",
            "PRIVATE_OUT",
        ]
        self.idx_offset = len(self.special_toks)
        self.stub_endpoint_tokens = {
            int(self.ep_load_stub),
            int(self.ep_driver_stub),
            int(self.ep_optional_driver_stub),
            int(self.ep_unknown_stub),
        }
        self.driver_stub_endpoint_tokens = {
            int(self.ep_driver_stub),
            int(self.ep_optional_driver_stub),
        }
        self.load_stub_endpoint_tokens = {int(self.ep_load_stub)}
        self.ep_stub = int(self.ep_load_stub)  # compatibility for old callers
        self.net_idx_offset = None
        self.load_stub_idx_offset = None
        self.driver_stub_idx_offset = None
        self.unknown_stub_idx_offset = None

    def set_num_node_and_edge_types(self, num_node_types=0, num_edge_types=0):
        self.num_node_types = int(num_node_types)
        if self.max_num_nodes is None:
            raise ValueError("set_num_nodes first")
        n = int(self.max_num_nodes)
        self.net_idx_offset = int(self.idx_offset)
        self.load_stub_idx_offset = int(self.net_idx_offset + n)
        self.driver_stub_idx_offset = int(self.load_stub_idx_offset + n)
        self.unknown_stub_idx_offset = int(self.driver_stub_idx_offset + n)
        self.pin_offset = int(self.unknown_stub_idx_offset + n)
        self.node_type_offset = int(self.pin_offset + len(self.pin_names))

    def set_pin_name_vocab(self, pin_names: List[str]) -> None:
        super().set_pin_name_vocab(pin_names)
        if self.max_num_nodes is not None:
            self.set_num_node_and_edge_types(self.num_node_types)

    def _namespace_offset_for_endpoint(self, ep_tok: int) -> int:
        if self.net_idx_offset is None:
            raise ValueError("tokenizer not initialized")
        ep_tok = int(ep_tok)
        if ep_tok == int(self.ep_load_stub):
            return int(self.load_stub_idx_offset)
        if ep_tok in {int(self.ep_driver_stub), int(self.ep_optional_driver_stub)}:
            return int(self.driver_stub_idx_offset)
        if ep_tok == int(self.ep_unknown_stub):
            return int(self.unknown_stub_idx_offset)
        return int(self.net_idx_offset)

    def _tok_endpoint_idx(self, ep_tok: int, idx: int) -> int:
        return int(self._namespace_offset_for_endpoint(ep_tok) + int(idx))

    def idx_value_from_token(self, tok: int, ep_tok: Optional[int] = None) -> int:
        if self.max_num_nodes is None:
            return -1
        if ep_tok is not None:
            idx = int(tok) - int(self._namespace_offset_for_endpoint(int(ep_tok)))
            return int(idx) if 0 <= idx < int(self.max_num_nodes) else -1
        for off in (
            self.net_idx_offset,
            self.load_stub_idx_offset,
            self.driver_stub_idx_offset,
            self.unknown_stub_idx_offset,
        ):
            if off is None:
                continue
            idx = int(tok) - int(off)
            if 0 <= idx < int(self.max_num_nodes):
                return int(idx)
        return -1

    def graph_node_idx_for_endpoint(self, ep_tok: int, idx_value: int) -> int:
        """Map namespaced endpoint ids into a PyG node-id space disjoint from gate ids."""
        n = int(self.max_num_nodes or 0)
        if int(ep_tok) == int(self.ep_load_stub):
            return int((2 * n) + int(idx_value))
        if int(ep_tok) in {int(self.ep_driver_stub), int(self.ep_optional_driver_stub)}:
            return int((3 * n) + int(idx_value))
        if int(ep_tok) == int(self.ep_unknown_stub):
            return int((4 * n) + int(idx_value))
        if int(ep_tok) in {int(self.ep_input_pool), int(self.ep_private_out), int(self.ep_pi), int(self.ep_const0), int(self.ep_const1)}:
            return int((5 * n) + int(idx_value))
        return int(n + int(idx_value))

    def idx_mask_for_endpoint_token(self, ep_tok: int, device=None):
        mask = torch.zeros((len(self),), dtype=torch.bool, device=device)
        off = int(self._namespace_offset_for_endpoint(int(ep_tok)))
        mask[off : off + int(self.max_num_nodes)] = True
        return mask

    def _endpoint_from_edge(self, data: Data, eidx: int, dst: int, role: int, source_tok: int) -> Tuple[int, int, int]:
        x = data.x.reshape(-1).to(torch.long)
        edge_endpoint_type = getattr(data, "edge_endpoint_type", None)
        ep_marker = None
        if edge_endpoint_type is not None and int(edge_endpoint_type.numel()) == int(data.edge_index.size(1)):
            ep_marker = int(edge_endpoint_type[int(eidx)].item())
        dst_label = int(x[int(dst)].item())
        role = int(role)
        source_tok = int(source_tok)
        is_syn = 0
        # Accept both tokenizer token ids and FULL_V1 endpoint schema ids.
        full_load_stub = 2
        full_driver_stub = 3
        full_optional_driver_stub = 4
        full_unknown_stub = 5
        full_pi = 6
        full_input_pool = 7
        full_private_out = 8
        full_const0 = 9
        full_const1 = 10
        if ep_marker in {int(self.ep_load_stub), full_load_stub} or (dst_label == int(self.boundary_stub_id) and role == 1):
            return int(self.ep_load_stub), int(dst), 0
        if ep_marker in {int(self.ep_optional_driver_stub), full_optional_driver_stub}:
            return int(self.ep_optional_driver_stub), int(dst), 0
        if ep_marker in {int(self.ep_driver_stub), full_driver_stub} or (dst_label == int(self.boundary_stub_id) and role == 2):
            return int(self.ep_driver_stub), int(dst), 0
        if ep_marker in {int(self.ep_unknown_stub), full_unknown_stub} or (dst_label == int(self.boundary_stub_id)):
            return int(self.ep_unknown_stub), int(dst), 0
        if ep_marker in {int(self.ep_pi), full_pi}:
            return int(self.ep_pi), int(dst), 1
        if ep_marker in {int(self.ep_input_pool), full_input_pool}:
            return int(self.ep_input_pool), int(dst), 1
        if ep_marker in {int(self.ep_private_out), full_private_out}:
            return int(self.ep_private_out), int(dst), 1
        if ep_marker in {int(self.ep_const0), full_const0}:
            return int(self.ep_const0), int(dst), 1
        if ep_marker in {int(self.ep_const1), full_const1}:
            return int(self.ep_const1), int(dst), 1
        if dst_label == int(self.net_id):
            return int(self.ep_net), int(dst), 0
        if role == 1:
            return int(self.ep_input_pool), int(dst), 1
        if role == 2:
            return int(self.ep_private_out), int(dst), 1
        return int(self.ep_net), int(dst), 1

    def _build_pin_edges(self, data: Data, gate_idx: int) -> Dict[str, Tuple[int, int, int, int]]:
        edge_index = data.edge_index.to(torch.long)
        edge_role = getattr(data, "edge_role", None)
        if edge_role is None:
            edge_role = getattr(data, "edge_attr", None)
        edge_pin = getattr(data, "edge_pin_id", None)
        pin_id_to_name = getattr(data, "pin_id_to_name", None)
        out: Dict[str, Tuple[int, int, int, int]] = {}
        if edge_role is None or edge_pin is None or pin_id_to_name is None:
            return out
        if int(edge_role.numel()) != int(edge_index.size(1)) or int(edge_pin.numel()) != int(edge_index.size(1)):
            return out
        for eidx, (src, dst) in enumerate(edge_index.t().tolist()):
            if int(src) != int(gate_idx):
                continue
            role = int(edge_role[int(eidx)].item())
            if role not in {1, 2}:
                continue
            pid = int(edge_pin[int(eidx)].item())
            if pid < 0 or pid >= len(pin_id_to_name):
                continue
            pname = str(pin_id_to_name[pid])
            if pname in out:
                continue
            ep_tok, ep_id, is_syn = self._endpoint_from_edge(data, eidx, int(dst), role, self.pin_in if role == 1 else self.pin_out)
            out[pname] = (int(ep_tok), int(ep_id), int(is_syn), int(eidx))
        return out

    def tokenize(self, data: Data):
        if self.max_num_nodes is None or self.node_type_offset is None:
            raise ValueError("tokenizer not initialized")
        x = data.x.reshape(-1).to(torch.long)
        tokens: List[int] = [int(self.sos)]
        for gate_idx in self._gate_nodes(data):
            cell_label = int(x[gate_idx].item())
            cell_name = self.label_to_cell.get(int(cell_label))
            spec = self.pin_specs.get(str(cell_name)) if cell_name is not None else None
            if spec is None:
                spec = PinSlotSpec(inputs=[], outputs=[], min_required_outputs=0)
            pin_edges = self._build_pin_edges(data, int(gate_idx))
            assigned_eidx = set()
            tokens.extend([int(self.gate), self._tok_idx(int(gate_idx)), self._tok_cell_label(cell_label)])
            for pname in spec.inputs:
                ep = pin_edges.get(str(pname))
                tokens.extend([int(self.pin_in), self._tok_pin(str(pname))])
                if ep is None:
                    ep_tok, ep_id = int(self.ep_input_pool), int(gate_idx)
                else:
                    ep_tok, ep_id, _is_syn, eidx = ep
                    assigned_eidx.add(int(eidx))
                    if ep_tok in self.driver_stub_endpoint_tokens:
                        ep_tok = int(self.ep_load_stub)
                tokens.extend([int(ep_tok), self._tok_endpoint_idx(int(ep_tok), int(ep_id))])
            out_pins = list(spec.outputs)
            min_out = int(max(0, spec.min_required_outputs))
            for pname in out_pins[:min_out]:
                ep = pin_edges.get(str(pname))
                tokens.extend([int(self.pin_out), self._tok_pin(str(pname))])
                if ep is None:
                    ep_tok, ep_id = int(self.ep_private_out), int(gate_idx)
                else:
                    ep_tok, ep_id, _is_syn, eidx = ep
                    assigned_eidx.add(int(eidx))
                    if ep_tok == int(self.ep_load_stub):
                        ep_tok = int(self.ep_driver_stub)
                tokens.extend([int(ep_tok), self._tok_endpoint_idx(int(ep_tok), int(ep_id))])
            for pname in out_pins[min_out:]:
                ep = pin_edges.get(str(pname))
                if ep is None:
                    tokens.extend([int(self.pin_skip), self._tok_pin(str(pname))])
                    continue
                ep_tok, ep_id, _is_syn, eidx = ep
                assigned_eidx.add(int(eidx))
                if ep_tok == int(self.ep_driver_stub):
                    ep_tok = int(self.ep_optional_driver_stub)
                tokens.extend([int(self.pin_opt_out), self._tok_pin(str(pname)), int(ep_tok), self._tok_endpoint_idx(int(ep_tok), int(ep_id))])

            for src, dst, role, pname in self._incident_edges(data, int(gate_idx)):
                del src
                # Keep EXCESS/UNKNOWN round-trip visible, but V2.1 generation policy disables them.
                eidx = None
                for j, pair in enumerate(data.edge_index.t().tolist()):
                    if int(pair[0]) == int(gate_idx) and int(pair[1]) == int(dst):
                        eidx = int(j)
                        break
                if eidx is not None and eidx in assigned_eidx:
                    continue
                if int(role) not in {0, 3}:
                    continue
                tok_role = int(self.pin_excess) if int(role) == 3 else int(self.pin_unknown)
                pname = pname if pname else ("__EXCESS__" if int(role) == 3 else "__UNKNOWN__")
                ep_tok, ep_id, _is_syn = self._endpoint_from_edge(data, eidx or 0, int(dst), int(role), tok_role)
                if ep_tok in self.stub_endpoint_tokens:
                    ep_tok = int(self.ep_unknown_stub)
                tokens.extend([tok_role, self._tok_pin(str(pname)), int(ep_tok), self._tok_endpoint_idx(int(ep_tok), int(ep_id))])
            tokens.append(int(self.end_gate))
        if self.append_eos:
            tokens.append(int(self.eos))
        out = torch.tensor(tokens, dtype=torch.long)
        if self.max_length is not None and int(self.max_length) > 0:
            out = out[: int(self.max_length)]
        return out

    def decode(self, token_ids: torch.Tensor) -> Data:
        if self.max_num_nodes is None or self.pin_offset is None or self.node_type_offset is None:
            raise ValueError("tokenizer not initialized")
        t = token_ids.reshape(-1).to(torch.long)
        t = t[(t != int(self.pad)) & (t != int(self.sos))]
        x_labels: Dict[int, int] = {}
        edges: List[Tuple[int, int]] = []
        roles: List[int] = []
        pin_names: List[str] = []
        pin_name_to_local: Dict[str, int] = {}
        edge_pin: List[int] = []
        synthetic_edge: List[int] = []
        edge_endpoint_type: List[int] = []
        edge_stub_role: List[int] = []
        stub_owner: List[int] = []
        stub_pin_role: List[int] = []
        stub_pin_id: List[int] = []
        max_node = -1
        synthetic_nodes = set()
        decode_valid = True
        decode_incomplete_gate = False
        decode_eos_before_gate_completion = False
        decode_invalid_token = False
        decode_endpoint_id_collision = False
        decode_collision_events: List[Dict[str, int | str]] = []
        decode_invalid_endpoint_events: List[Dict[str, int | str]] = []

        def pin_id(name: str) -> int:
            if name not in pin_name_to_local:
                pin_name_to_local[name] = len(pin_names)
                pin_names.append(name)
            return int(pin_name_to_local[name])

        def ensure(idx: int, label: int, *, event_type: str = "node_id_collision", token_pos: int = -1, **ctx):
            nonlocal max_node, decode_valid, decode_endpoint_id_collision
            if idx < 0:
                return
            max_node = max(max_node, int(idx))
            old = x_labels.get(int(idx))
            if old is not None and int(old) != int(label):
                decode_valid = False
                decode_endpoint_id_collision = True
                event = {
                    "event_type": str(event_type),
                    "token_pos": int(token_pos),
                    "node_idx": int(idx),
                    "previous_label": int(old),
                    "current_label": int(label),
                }
                event.update(ctx)
                decode_collision_events.append(event)
            x_labels[int(idx)] = int(label)

        def append_edge(gate_idx: int, ep_idx: int, role: int, pname: str, ep_tok: int, source_tok: int):
            pid = pin_id(str(pname))
            edges.append((int(gate_idx), int(ep_idx)))
            roles.append(int(role))
            edge_pin.append(int(pid))
            is_syn = int(ep_tok in {self.ep_input_pool, self.ep_private_out, self.ep_const0, self.ep_const1})
            synthetic_edge.append(is_syn)
            if is_syn:
                synthetic_nodes.add(int(ep_idx))
            edge_endpoint_type.append(int(ep_tok))
            sr = 0
            if ep_tok == int(self.ep_load_stub):
                sr = 1
            elif ep_tok == int(self.ep_driver_stub):
                sr = 2
            elif ep_tok == int(self.ep_optional_driver_stub):
                sr = 3
            elif ep_tok == int(self.ep_unknown_stub):
                sr = 4
            edge_stub_role.append(sr)
            stub_owner.append(int(gate_idx) if sr else -1)
            stub_pin_role.append(int(role) if sr else 0)
            stub_pin_id.append(int(pid) if sr else -1)

        pos = 0
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
            gate_idx = self.idx_value_from_token(int(t[pos + 1].item()), int(self.ep_net))
            cell_label = self._cell_from_tok(int(t[pos + 2].item()))
            if gate_idx < 0:
                decode_valid = False
                decode_invalid_token = True
                pos += 1
                continue
            ensure(
                gate_idx,
                cell_label,
                event_type="gate_id_collision",
                token_pos=int(pos),
                gate_idx=int(gate_idx),
                cell_label=int(cell_label),
            )
            pos += 3
            complete = False
            while pos < int(t.numel()):
                tok = int(t[pos].item())
                if tok == int(self.eos):
                    decode_valid = False
                    decode_eos_before_gate_completion = True
                    pos = int(t.numel())
                    break
                if tok == int(self.end_gate):
                    pos += 1
                    complete = True
                    break
                if tok in {int(self.pin_in), int(self.pin_out), int(self.pin_opt_out), int(self.pin_excess), int(self.pin_unknown)}:
                    if pos + 3 >= int(t.numel()):
                        pos = int(t.numel())
                        break
                    pname = self._pin_from_tok(int(t[pos + 1].item()))
                    ep_tok = int(t[pos + 2].item())
                    ep_idx_value = self.idx_value_from_token(int(t[pos + 3].item()), ep_tok)
                    if pname == "__INVALID_PIN__" or ep_idx_value < 0:
                        decode_valid = False
                        decode_invalid_token = True
                        decode_invalid_endpoint_events.append({
                            "token_pos": int(pos),
                            "gate_idx": int(gate_idx),
                            "pin_name": str(pname),
                            "pin_role_token": int(tok),
                            "endpoint_type_token": int(ep_tok),
                            "endpoint_idx_token": int(t[pos + 3].item()),
                            "endpoint_idx_value": int(ep_idx_value),
                        })
                        pos += 4
                        continue
                    ep_idx = self.graph_node_idx_for_endpoint(ep_tok, ep_idx_value)
                    if tok == int(self.pin_in):
                        role = 1
                    elif tok in {int(self.pin_out), int(self.pin_opt_out)}:
                        role = 2
                    elif tok == int(self.pin_excess):
                        role = 3
                    else:
                        role = 0
                    label = int(self.boundary_stub_id) if ep_tok in self.stub_endpoint_tokens else int(self.net_id)
                    ensure(
                        ep_idx,
                        label,
                        event_type="endpoint_node_collision",
                        token_pos=int(pos),
                        gate_idx=int(gate_idx),
                        pin_name=str(pname),
                        pin_role_token=int(tok),
                        endpoint_type_token=int(ep_tok),
                        endpoint_idx_value=int(ep_idx_value),
                    )
                    append_edge(gate_idx, ep_idx, role, pname, ep_tok, tok)
                    pos += 4
                    continue
                if tok == int(self.pin_skip):
                    pos += 2
                    continue
                pos += 1
            if not complete:
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
        data.stub_owner_gate_id = torch.tensor(stub_owner, dtype=torch.long) if stub_owner else torch.full((0,), -1, dtype=torch.long)
        data.stub_pin_role = torch.tensor(stub_pin_role, dtype=torch.long) if stub_pin_role else torch.zeros((0,), dtype=torch.long)
        data.stub_pin_id = torch.tensor(stub_pin_id, dtype=torch.long) if stub_pin_id else torch.full((0,), -1, dtype=torch.long)
        if synthetic_nodes:
            data.synthetic_endpoint_base = int(min(synthetic_nodes))
            data.synthetic_endpoint_count = int(max(synthetic_nodes) - int(min(synthetic_nodes)) + 1)
        data.decode_valid = bool(decode_valid)
        data.decode_incomplete_gate = bool(decode_incomplete_gate)
        data.decode_eos_before_gate_completion = bool(decode_eos_before_gate_completion)
        data.decode_invalid_token = bool(decode_invalid_token)
        data.decode_endpoint_id_collision = bool(decode_endpoint_id_collision)
        data.decode_collision_events = decode_collision_events
        data.decode_invalid_endpoint_events = decode_invalid_endpoint_events
        return data


class CircuitPinSlotV21GCTokenizer(CircuitPinSlotV21Tokenizer):
    def __init__(
        self,
        *,
        gate_count_buckets: Optional[List[int]] = None,
        driver_stub_buckets: Optional[List[int]] = None,
        load_stub_buckets: Optional[List[int]] = None,
        driver_load_ratio_buckets: Optional[List[int]] = None,
        enable_contract_prefix: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        buckets = [64, 80, 96, 112, 128] if gate_count_buckets is None else [int(x) for x in gate_count_buckets]
        self.gate_count_buckets = sorted(set(int(x) for x in buckets if int(x) > 0))
        self.gc_tokens = {}
        for b in self.gate_count_buckets:
            self.gc_tokens[int(b)] = int(len(self.special_toks))
            self.special_toks.append(f"GC_{int(b)}")
        self.enable_contract_prefix = bool(enable_contract_prefix)
        self.driver_stub_buckets = sorted(set(int(x) for x in (driver_stub_buckets or [0, 1, 2, 3, 4, 5, 6, 7, 8])))
        self.load_stub_buckets = sorted(set(int(x) for x in (load_stub_buckets or [0, 1, 2, 3, 4, 5, 6, 7, 8])))
        self.driver_load_ratio_buckets = sorted(set(int(x) for x in (driver_load_ratio_buckets or [0, 1, 2, 3, 4])))
        self.driver_stub_bucket_tokens = {}
        self.load_stub_bucket_tokens = {}
        self.driver_load_ratio_bucket_tokens = {}
        if self.enable_contract_prefix:
            for b in self.driver_stub_buckets:
                self.driver_stub_bucket_tokens[int(b)] = int(len(self.special_toks))
                self.special_toks.append(f"DRV_STUB_B{int(b)}")
            for b in self.load_stub_buckets:
                self.load_stub_bucket_tokens[int(b)] = int(len(self.special_toks))
                self.special_toks.append(f"LOAD_STUB_B{int(b)}")
            for b in self.driver_load_ratio_buckets:
                self.driver_load_ratio_bucket_tokens[int(b)] = int(len(self.special_toks))
                self.special_toks.append(f"DRV_LOAD_RATIO_B{int(b)}")
        self.contract_prefix_token_maps = {
            "schema_target_driver_stub_bucket": self.driver_stub_bucket_tokens,
            "schema_target_load_stub_bucket": self.load_stub_bucket_tokens,
            "schema_target_driver_load_ratio_bucket": self.driver_load_ratio_bucket_tokens,
        }
        self.idx_offset = len(self.special_toks)

    def _bucket_for_gate_count(self, gate_count: int) -> int:
        if not self.gate_count_buckets:
            return int(gate_count)
        return min(self.gate_count_buckets, key=lambda b: abs(int(b) - int(gate_count)))

    def tokenize(self, data: Data):
        tokens = super().tokenize(data)
        if int(tokens.numel()) <= 0 or int(tokens[0].item()) != int(self.sos):
            return tokens
        x = data.x.reshape(-1).to(torch.long)
        gate_count = int(((x != int(self.net_id)) & (x != int(self.boundary_stub_id))).sum().item())
        prefix = []
        gate_bucket = getattr(data, "schema_target_gate_count_bucket", self._bucket_for_gate_count(gate_count))
        if isinstance(gate_bucket, torch.Tensor):
            gate_bucket = int(gate_bucket.reshape(-1)[0].item()) if int(gate_bucket.numel()) > 0 else self._bucket_for_gate_count(gate_count)
        tok = self.gc_tokens.get(int(gate_bucket))
        if tok is None:
            tok = self.gc_tokens.get(int(self._bucket_for_gate_count(gate_count)))
        if tok is not None:
            prefix.append(int(tok))
        for attr, tok_map in [
            ("schema_target_driver_stub_bucket", self.driver_stub_bucket_tokens),
            ("schema_target_load_stub_bucket", self.load_stub_bucket_tokens),
            ("schema_target_driver_load_ratio_bucket", self.driver_load_ratio_bucket_tokens),
        ]:
            if not tok_map:
                continue
            val = getattr(data, attr, None)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                if int(val.numel()) == 0:
                    continue
                val = int(val.reshape(-1)[0].item())
            tok = tok_map.get(int(val))
            if tok is not None:
                prefix.append(int(tok))
        if not prefix:
            return tokens
        return torch.cat([tokens[:1], torch.tensor(prefix, dtype=torch.long), tokens[1:]], dim=0)

    def decode(self, t: torch.Tensor):
        t = t.detach().cpu().reshape(-1).to(torch.long)
        target_bucket = None
        target_driver_stub_bucket = None
        target_load_stub_bucket = None
        target_driver_load_ratio_bucket = None
        if int(t.numel()) >= 2 and int(t[0].item()) == int(self.sos):
            inv_gc = {int(v): int(k) for k, v in self.gc_tokens.items()}
            inv_drv = {int(v): int(k) for k, v in self.driver_stub_bucket_tokens.items()}
            inv_load = {int(v): int(k) for k, v in self.load_stub_bucket_tokens.items()}
            inv_ratio = {int(v): int(k) for k, v in self.driver_load_ratio_bucket_tokens.items()}
            keep = [int(t[0].item())]
            pos = 1
            while pos < int(t.numel()):
                tok = int(t[pos].item())
                if tok in inv_gc:
                    target_bucket = int(inv_gc[tok])
                    pos += 1
                    continue
                if tok in inv_drv:
                    target_driver_stub_bucket = int(inv_drv[tok])
                    pos += 1
                    continue
                if tok in inv_load:
                    target_load_stub_bucket = int(inv_load[tok])
                    pos += 1
                    continue
                if tok in inv_ratio:
                    target_driver_load_ratio_bucket = int(inv_ratio[tok])
                    pos += 1
                    continue
                break
            if pos < int(t.numel()):
                keep.extend(int(x) for x in t[pos:].tolist())
            t = torch.tensor(keep, dtype=torch.long)
        g = super().decode(t)
        if target_bucket is not None:
            g.target_gate_count_bucket = int(target_bucket)
        if target_driver_stub_bucket is not None:
            g.target_driver_stub_bucket = int(target_driver_stub_bucket)
        if target_load_stub_bucket is not None:
            g.target_load_stub_bucket = int(target_load_stub_bucket)
        if target_driver_load_ratio_bucket is not None:
            g.target_driver_load_ratio_bucket = int(target_driver_load_ratio_bucket)
        return g
