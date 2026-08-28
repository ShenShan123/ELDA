from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class RoleAwareSerializationSpec:
    net_id: int
    boundary_stub_id: int
    label_to_cell: Dict[int, str]
    cell_to_label: Dict[str, int]


def serialize_role_aware_graph(
    graph: Data,
    *,
    spec: RoleAwareSerializationSpec,
    include_pin: bool = True,
) -> List[str]:
    x = graph.x.reshape(-1).to(torch.long)
    edge_index = graph.edge_index.to(torch.long)
    edge_role = getattr(graph, 'edge_role', None)
    edge_pin = getattr(graph, 'edge_pin_id', None)
    pin_vocab = getattr(graph, 'pin_id_to_name', None)
    if edge_role is None or int(edge_role.numel()) != int(edge_index.size(1)):
        raise ValueError('graph.edge_role is required')

    out: List[str] = []
    out.append('ROLE_AWARE_REALIZATION_V1')
    out.append(f'NET_ID {int(spec.net_id)}')
    out.append(f'STUB_ID {int(spec.boundary_stub_id)}')
    out.append(f'NUM_NODES {int(x.numel())}')
    out.append(f'NUM_EDGES {int(edge_index.size(1))}')

    for idx in range(int(x.numel())):
        label = int(x[idx].item())
        if label == int(spec.net_id):
            out.append(f'NODE {idx} NET')
        elif label == int(spec.boundary_stub_id):
            out.append(f'NODE {idx} STUB')
        else:
            cell = spec.label_to_cell.get(label, f'LABEL_{label}')
            out.append(f'NODE {idx} CELL {cell}')

    for eidx, (src, dst) in enumerate(edge_index.t().tolist()):
        r = int(edge_role[int(eidx)].item())
        role_tok = 'UNKNOWN'
        if r == 1:
            role_tok = 'GATE_INPUT'
        elif r == 2:
            role_tok = 'GATE_OUTPUT'
        elif r == 3:
            role_tok = 'EXCESS'
        if include_pin and isinstance(edge_pin, torch.Tensor) and isinstance(pin_vocab, list):
            if int(edge_pin.numel()) == int(edge_index.size(1)):
                pid = int(edge_pin[int(eidx)].item())
                pname = pin_vocab[pid] if 0 <= pid < len(pin_vocab) else None
                if pname is not None and int(pid) >= 0:
                    out.append(f'EDGE {int(src)} {int(dst)} {role_tok} PIN {str(pname)}')
                    continue
        out.append(f'EDGE {int(src)} {int(dst)} {role_tok}')

    return out


def deserialize_role_aware_graph(
    tokens: Iterable[str],
    *,
    spec: RoleAwareSerializationSpec,
) -> Data:
    tokens = list(tokens)
    if not tokens or tokens[0].strip() != 'ROLE_AWARE_REALIZATION_V1':
        raise ValueError('missing ROLE_AWARE_REALIZATION_V1 header')

    num_nodes = None
    num_edges = None
    node_labels: Dict[int, int] = {}
    edges: List[Tuple[int, int]] = []
    roles: List[int] = []
    pins: List[int] = []
    pin_to_id: Dict[str, int] = {}
    id_to_pin: List[str] = []

    def pin_id(name: str) -> int:
        if name not in pin_to_id:
            pin_to_id[name] = len(id_to_pin)
            id_to_pin.append(name)
        return int(pin_to_id[name])

    for raw in tokens[1:]:
        parts = raw.strip().split()
        if not parts:
            continue
        if parts[0] == 'NUM_NODES':
            num_nodes = int(parts[1])
            continue
        if parts[0] == 'NUM_EDGES':
            num_edges = int(parts[1])
            continue
        if parts[0] == 'NODE':
            idx = int(parts[1])
            if parts[2] == 'NET':
                node_labels[idx] = int(spec.net_id)
            elif parts[2] == 'STUB':
                node_labels[idx] = int(spec.boundary_stub_id)
            else:
                cell = str(parts[3])
                node_labels[idx] = int(spec.cell_to_label.get(cell, 0))
            continue
        if parts[0] == 'EDGE':
            src = int(parts[1])
            dst = int(parts[2])
            role_tok = str(parts[3])
            r = 0
            if role_tok == 'GATE_INPUT':
                r = 1
            elif role_tok == 'GATE_OUTPUT':
                r = 2
            elif role_tok == 'EXCESS':
                r = 3
            pid = -1
            if len(parts) >= 6 and parts[4] == 'PIN':
                pid = pin_id(str(parts[5]))
            edges.append((src, dst))
            roles.append(int(r))
            pins.append(int(pid))
            continue

    if num_nodes is None:
        num_nodes = max(node_labels.keys()) + 1 if node_labels else 0
    if num_edges is None:
        num_edges = len(edges)
    if len(edges) != int(num_edges):
        raise ValueError('edge count mismatch')

    x = torch.zeros((int(num_nodes),), dtype=torch.long)
    for idx, label in node_labels.items():
        x[int(idx)] = int(label)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    edge_role = torch.tensor(roles, dtype=torch.long) if roles else torch.zeros((0,), dtype=torch.long)
    edge_pin_id = torch.tensor(pins, dtype=torch.long) if pins else torch.full((0,), -1, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_role.clone(), num_nodes=int(num_nodes))
    data.edge_role = edge_role
    data.edge_pin_id = edge_pin_id
    data.pin_id_to_name = list(id_to_pin)
    return data
