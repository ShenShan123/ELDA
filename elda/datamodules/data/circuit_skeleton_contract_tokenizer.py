from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class SkeletonContractTokenLayout:
    sos: int = 0
    eos: int = 1
    pad: int = 2
    skel: int = 3
    nodes: int = 4
    end_nodes: int = 5
    edges: int = 6
    end_edges: int = 7
    sn: int = 8
    se: int = 9
    value_offset: int = 1024


class CircuitSkeletonContractTokenizer:
    """Deterministic prototype tokenizer for V2.1 contract-aware skeleton graphs.

    The format is intentionally separate from partition pin-slot tokenizers:
    skeleton nodes represent partition-slot contracts, and skeleton edges represent
    cross-partition connection demand. Values are fixed-order integer attributes,
    offset away from control tokens so round-trip smoke tests do not depend on a
    learned vocabulary.
    """

    node_fields: Tuple[str, ...] = (
        "schema_target_gate_count_bucket",
        "schema_target_driver_stub_bucket",
        "schema_target_load_stub_bucket",
        "schema_target_driver_load_ratio_bucket",
        "schema_node_driver_demand_bucket",
        "schema_node_load_demand_bucket",
        "schema_node_shared_net_out_bucket",
        "schema_node_shared_net_in_bucket",
        "schema_node_contract_risk_bucket",
        "schema_size_bucket",
        "schema_stub_bucket",
        "schema_pin_bucket",
    )
    edge_fields: Tuple[str, ...] = (
        "schema_edge_role_type",
        "schema_edge_direction_prior",
        "schema_edge_group_size_bucket",
        "schema_edge_fanout_bucket",
        "schema_edge_contract_risk_bucket",
        "schema_edge_shared_net_group_count_bucket",
        "schema_typed_edge_teacher_confidence",
    )

    def __init__(self, max_value: int = 1_000_000):
        self.layout = SkeletonContractTokenLayout()
        self.max_value = int(max_value)

    def _enc_value(self, value: int) -> int:
        value = max(0, min(int(value), self.max_value))
        return self.layout.value_offset + value

    def _dec_value(self, token: int) -> int:
        return max(0, int(token) - self.layout.value_offset)

    @staticmethod
    def _tensor_values(graph: Data, name: str, length: int) -> List[int]:
        if not hasattr(graph, name):
            return [0] * length
        value = getattr(graph, name)
        if isinstance(value, torch.Tensor):
            flat = value.detach().cpu().view(-1).tolist()
            return [int(x) for x in flat[:length]] + [0] * max(0, length - len(flat))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            vals = [int(x) for x in list(value)[:length]]
            return vals + [0] * max(0, length - len(vals))
        return [int(value)] * length

    def encode_skeleton_contract_graph(self, graph: Data) -> List[int]:
        num_nodes = int(graph.num_nodes)
        edge_index = graph.edge_index.detach().cpu().long()
        edges = sorted((int(edge_index[0, i]), int(edge_index[1, i]), i) for i in range(edge_index.shape[1]))

        node_values: Dict[str, List[int]] = {
            field: self._tensor_values(graph, field, num_nodes) for field in self.node_fields
        }
        edge_values: Dict[str, List[int]] = {
            field: self._tensor_values(graph, field, edge_index.shape[1]) for field in self.edge_fields
        }

        toks: List[int] = [self.layout.sos, self.layout.skel, self.layout.nodes]
        for node_id in range(num_nodes):
            toks.extend([self.layout.sn, self._enc_value(node_id)])
            toks.extend(self._enc_value(node_values[field][node_id]) for field in self.node_fields)
        toks.extend([self.layout.end_nodes, self.layout.edges])

        for src, dst, edge_pos in edges:
            toks.extend([self.layout.se, self._enc_value(src), self._enc_value(dst)])
            toks.extend(self._enc_value(edge_values[field][edge_pos]) for field in self.edge_fields)
        toks.extend([self.layout.end_edges, self.layout.eos])
        return toks

    def decode_skeleton_contract_sequence(self, tokens: Iterable[int]) -> Data:
        tokens = list(int(t) for t in tokens)
        if len(tokens) < 5 or tokens[0] != self.layout.sos or tokens[-1] != self.layout.eos:
            raise ValueError("invalid skeleton contract sequence boundary")
        if self.layout.nodes not in tokens or self.layout.edges not in tokens:
            raise ValueError("missing NODES/EDGES section")

        i = tokens.index(self.layout.nodes) + 1
        nodes: Dict[int, List[int]] = {}
        while i < len(tokens) and tokens[i] != self.layout.end_nodes:
            if tokens[i] != self.layout.sn:
                raise ValueError(f"expected SN at token {i}, got {tokens[i]}")
            node_id = self._dec_value(tokens[i + 1])
            start = i + 2
            end = start + len(self.node_fields)
            nodes[node_id] = [self._dec_value(t) for t in tokens[start:end]]
            i = end
        if i >= len(tokens) or tokens[i] != self.layout.end_nodes:
            raise ValueError("unterminated NODES section")
        if tokens[i + 1] != self.layout.edges:
            raise ValueError("missing EDGES section")
        i += 2

        edge_pairs: List[Tuple[int, int]] = []
        edge_attrs: Dict[str, List[int]] = {field: [] for field in self.edge_fields}
        while i < len(tokens) and tokens[i] != self.layout.end_edges:
            if tokens[i] != self.layout.se:
                raise ValueError(f"expected SE at token {i}, got {tokens[i]}")
            src = self._dec_value(tokens[i + 1])
            dst = self._dec_value(tokens[i + 2])
            start = i + 3
            end = start + len(self.edge_fields)
            vals = [self._dec_value(t) for t in tokens[start:end]]
            edge_pairs.append((src, dst))
            for field, val in zip(self.edge_fields, vals):
                edge_attrs[field].append(val)
            i = end
        if i >= len(tokens) or tokens[i] != self.layout.end_edges:
            raise ValueError("unterminated EDGES section")

        num_nodes = max(nodes.keys(), default=-1) + 1
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(x=torch.zeros(num_nodes, dtype=torch.long), edge_index=edge_index, num_nodes=num_nodes)
        for idx, field in enumerate(self.node_fields):
            values = [nodes.get(n, [0] * len(self.node_fields))[idx] for n in range(num_nodes)]
            setattr(data, field, torch.tensor(values, dtype=torch.long))
        for field, values in edge_attrs.items():
            setattr(data, field, torch.tensor(values, dtype=torch.long))
        return data


def encode_skeleton_contract_graph(graph: Data) -> List[int]:
    return CircuitSkeletonContractTokenizer().encode_skeleton_contract_graph(graph)


def decode_skeleton_contract_sequence(tokens: Iterable[int]) -> Data:
    return CircuitSkeletonContractTokenizer().decode_skeleton_contract_sequence(tokens)
