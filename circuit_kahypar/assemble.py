from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch_geometric.data import Data

from circuit_kahypar.pin_spec_table import CellPinSpecTable
from circuit_kahypar.role_aware import annotate_gate_net_edge_roles
from torch_geometric.utils import coalesce, remove_isolated_nodes, to_undirected
from scipy.optimize import linear_sum_assignment

from .schema import (
    classify_partition_graph,
    decode_interface_class_meta,
    infer_cell_spec,
    infer_boundary_stub_roles,
    interface_bucket_distance,
    normalize_circuit_graph,
    normalize_generated_partition_graph,
    partition_pin_semantic_metrics,
    partition_graph_metrics,
)
from .liberty import pin_count_spec


COMPLEX_CELL_FAMILY_WEIGHTS = {
    'MUX': 3.0,
    'AOI': 2.5,
    'OAI': 2.5,
    'DFF': 2.5,
    'HA': 2.0,
    'FA': 2.5,
    'NAND3': 1.8,
    'NAND4': 2.2,
    'NOR3': 1.8,
    'NOR4': 2.2,
}


DEFAULT_ASSIGNMENT_COST_WEIGHTS = {
    'stub_shortfall': 1000.0,
    'largest_stub_shortfall': 800.0,
    'interface_distance': 100.0,
    'pin_deficit': 60.0,
    'underconnected': 40.0,
    'synthetic_proxy': 50.0,
    'stub_excess': 10.0,
    'largest_stub_excess': 5.0,
    'non_stub_components': 20.0,
    'non_stub_lcc': -20.0,
    'weighted_type_gain': -80.0,
    'medium_type_gain': -60.0,
    'used_type_gain': -5.0,
    'x1_excess': 10.0,
    'estimated_input_deficit': 4.0,
    'estimated_output_deficit': 8.0,
    'estimated_reuse_risk': 3.0,
    'estimated_synthetic_input_risk': 3.0,
    'estimated_repaired_output_risk': 5.0,
}


QUALITY_BALANCED_ASSIGNMENT_COST_WEIGHTS = {
    **DEFAULT_ASSIGNMENT_COST_WEIGHTS,
    # v9 proved that pin-clean global matching alone can fragment the
    # assembled graph.  This profile keeps the pin/interface terms but gives
    # the matcher a stronger preference for internally connected partitions
    # whose usable stubs attach to the largest component.
    'stub_shortfall': 1300.0,
    'largest_stub_shortfall': 1600.0,
    'interface_distance': 90.0,
    'pin_deficit': 55.0,
    'underconnected': 35.0,
    'synthetic_proxy': 45.0,
    'stub_excess': 8.0,
    'largest_stub_excess': 3.0,
    'non_stub_components': 110.0,
    'non_stub_lcc': -130.0,
    'weighted_type_gain': -120.0,
    'medium_type_gain': -85.0,
    'used_type_gain': -12.0,
    'estimated_input_deficit': 4.5,
    'estimated_output_deficit': 9.0,
    'estimated_reuse_risk': 2.5,
    'estimated_synthetic_input_risk': 4.0,
    'estimated_repaired_output_risk': 5.5,
}


BUDGET_FLOW_ASSIGNMENT_COST_WEIGHTS = {
    **QUALITY_BALANCED_ASSIGNMENT_COST_WEIGHTS,
    # This profile is used by the beam/min-cost-flow-style global assignment.
    # It keeps per-node interface matching, but adds stronger global pressure
    # toward connected, pin-complete, type-diverse selections.
    'stub_shortfall': 1500.0,
    'largest_stub_shortfall': 1800.0,
    'non_stub_components': 150.0,
    'non_stub_lcc': -180.0,
    'weighted_type_gain': -170.0,
    'medium_type_gain': -120.0,
    'used_type_gain': -20.0,
    'estimated_input_deficit': 5.0,
    'estimated_output_deficit': 11.0,
    'estimated_synthetic_input_risk': 5.0,
    'estimated_repaired_output_risk': 7.0,
}


ROLE_BALANCED_ASSIGNMENT_COST_WEIGHTS = {
    **BUDGET_FLOW_ASSIGNMENT_COST_WEIGHTS,
    # Push the matcher away from partitions whose boundary roles imply that
    # projection will need to invent many extra inputs/outputs later.
    'driver_role_shortfall': 280.0,
    'load_role_shortfall': 420.0,
    'role_imbalance': 55.0,
    'estimated_input_deficit': 7.5,
    'estimated_output_deficit': 12.0,
    'estimated_reuse_risk': 4.5,
    'estimated_synthetic_input_risk': 6.5,
    'estimated_repaired_output_risk': 8.0,
}


CONTRACT_BALANCED_ASSIGNMENT_COST_WEIGHTS = {
    **ROLE_BALANCED_ASSIGNMENT_COST_WEIGHTS,
    # Push assignment further toward partitions whose boundary contract is
    # likely to survive assembly without heavy semantic projection.
    'stub_shortfall': 1700.0,
    'largest_stub_shortfall': 1900.0,
    'pin_deficit': 70.0,
    'synthetic_proxy': 65.0,
    'estimated_input_deficit': 9.0,
    'estimated_output_deficit': 14.0,
    'estimated_reuse_risk': 6.0,
    'estimated_synthetic_input_risk': 8.5,
    'estimated_repaired_output_risk': 9.0,
    'driver_role_shortfall': 360.0,
    'load_role_shortfall': 520.0,
    'role_imbalance': 70.0,
    'contract_projection_repair_bucket': 450.0,
    'contract_net_export_cleanliness_bucket': -220.0,
    'contract_driver_supply_bucket': -45.0,
    'contract_load_supply_bucket': -70.0,
    'contract_bidir_supply_bucket': -25.0,
    'contract_role_entropy_bucket': -20.0,
}


ROLE_NAMES = {
    -1: 'none',
    0: 'ambig',
    1: 'driver',
    2: 'load',
    3: 'bidir',
}


FUTURE_CONTRACT_ATTRS = {
    'contract_driver_supply_bucket': ('schema_driver_stub_supply_bucket',),
    'contract_load_supply_bucket': ('schema_load_stub_supply_bucket',),
    'contract_bidir_supply_bucket': ('schema_bidir_stub_supply_bucket',),
    'contract_role_entropy_bucket': ('schema_boundary_role_entropy',),
    'contract_projection_repair_bucket': ('schema_projection_repair_budget_bucket',),
    'contract_net_export_cleanliness_bucket': ('schema_net_export_cleanliness_bucket',),
    # Decoded V2.1-GC partition candidates carry generation targets without the
    # schema_ prefix. Preserve them in the interface profile so assignment
    # manifests can distinguish "no contract field" from "target-conditioned".
    'contract_target_gate_count_bucket': ('schema_target_gate_count_bucket', 'target_gate_count_bucket'),
    'contract_target_driver_stub_bucket': ('schema_target_driver_stub_bucket', 'target_driver_stub_bucket'),
    'contract_target_load_stub_bucket': ('schema_target_load_stub_bucket', 'target_load_stub_bucket'),
    'contract_target_driver_load_ratio_bucket': (
        'schema_target_driver_load_ratio_bucket',
        'target_driver_load_ratio_bucket',
    ),
}


def _scalar_graph_attr(data: Data, attr: str, default=None):
    if not hasattr(data, attr):
        return default
    value = getattr(data, attr)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return int(value[0])
    try:
        return int(value)
    except Exception:
        return default


def _first_scalar_graph_attr(data: Data, attrs, default=None):
    for attr in attrs:
        value = _scalar_graph_attr(data, attr, default=None)
        if value is not None:
            return value
    return default


def _future_contract_profile(data: Data) -> Dict[str, object]:
    values: Dict[str, object] = {}
    present = 0
    for key, attrs in FUTURE_CONTRACT_ATTRS.items():
        value = _first_scalar_graph_attr(data, attrs, default=None)
        if value is not None:
            values[key] = int(value)
            present += 1
        else:
            values[key] = None
    values['contract_field_count'] = int(present)
    values['contract_available'] = int(present > 0)
    return values


def _cell_family(cell_name: str) -> str:
    name = str(cell_name).upper()
    for prefix in ('AOI', 'OAI', 'MUX', 'DFF', 'NAND4', 'NAND3', 'NOR4', 'NOR3', 'NAND2', 'NOR2', 'HA', 'FA', 'XOR', 'XNOR', 'INV', 'BUF'):
        if name.startswith(prefix):
            return prefix
    return name.split('_', 1)[0]


def _complex_cell_risk(cell_name: str) -> float:
    family = _cell_family(cell_name)
    return float(COMPLEX_CELL_FAMILY_WEIGHTS.get(family, 1.0 if family else 0.0))


def _non_stub_component_stats(data: Data, boundary_stub_id: int) -> Dict[str, object]:
    x = data.x.reshape(-1).to(torch.long)
    is_stub = x == int(boundary_stub_id)
    non_stub_nodes = [idx for idx, flag in enumerate(is_stub.tolist()) if not bool(flag)]
    if not non_stub_nodes:
        return {
            'non_stub_count': 0,
            'non_stub_component_count': 0,
            'non_stub_largest_component_ratio': 0.0,
            'largest_component_nodes': set(),
        }

    non_stub_set = set(non_stub_nodes)
    adj: Dict[int, set] = {idx: set() for idx in non_stub_nodes}
    for src, dst in data.edge_index.t().tolist():
        if src in non_stub_set and dst in non_stub_set and src != dst:
            adj[src].add(dst)
            adj[dst].add(src)

    seen = set()
    components: List[set] = []
    for node in non_stub_nodes:
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        comp = set()
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nbr in adj[cur]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        components.append(comp)

    largest = max(components, key=len) if components else set()
    return {
        'non_stub_count': int(len(non_stub_nodes)),
        'non_stub_component_count': int(len(components)),
        'non_stub_largest_component_ratio': float(len(largest) / len(non_stub_nodes)),
        'largest_component_nodes': largest,
    }


def _boundary_stub_connectivity(data: Data, boundary_stub_id: int) -> Dict[str, float]:
    x = data.x.reshape(-1).to(torch.long)
    stub_ids = (x == int(boundary_stub_id)).nonzero(as_tuple=False).reshape(-1).tolist()
    comp_stats = _non_stub_component_stats(data, boundary_stub_id)
    largest_nodes = set(comp_stats.get('largest_component_nodes', set()))
    stub_with_neighbors = 0
    stub_on_largest = 0
    for stub_idx in stub_ids:
        nbrs = []
        for src, dst in data.edge_index.t().tolist():
            if src == stub_idx and int(x[dst].item()) != int(boundary_stub_id):
                nbrs.append(dst)
            elif dst == stub_idx and int(x[src].item()) != int(boundary_stub_id):
                nbrs.append(src)
        if nbrs:
            stub_with_neighbors += 1
        if any(nbr in largest_nodes for nbr in nbrs):
            stub_on_largest += 1
    total = max(1, len(stub_ids))
    return {
        'boundary_stub_with_neighbors': float(stub_with_neighbors),
        'boundary_stub_on_largest_component': float(stub_on_largest),
        'boundary_stub_attachment_ratio': float(stub_with_neighbors / total),
        'boundary_stub_largest_component_ratio': float(stub_on_largest / total),
        'non_stub_component_count': float(comp_stats['non_stub_component_count']),
        'non_stub_largest_component_ratio': float(comp_stats['non_stub_largest_component_ratio']),
    }


def compute_partition_interface_profile(
    data: Data,
    meta: Dict,
    label_to_cell: Dict[int, str] | None = None,
    cell_pin_specs: Dict[str, Dict[str, List[str]]] | None = None,
) -> Dict[str, object]:
    net_id = int(meta['net_id'])
    boundary_stub_id = int(meta['boundary_stub_id'])
    x = data.x.reshape(-1).to(torch.long)
    degrees = torch.bincount(data.edge_index[0], minlength=int(x.numel()))
    gate_nodes = ((x != net_id) & (x != boundary_stub_id)).nonzero(as_tuple=False).reshape(-1).tolist()
    stub_nodes = (x == boundary_stub_id).nonzero(as_tuple=False).reshape(-1).tolist()
    connectivity = _boundary_stub_connectivity(data, boundary_stub_id)
    family_counts: Counter = Counter()
    complex_cell_count = 0
    input_deficit = 0
    output_deficit = 0
    reuse_risk = 0.0
    synthetic_risk = 0.0
    repaired_output_risk = 0.0
    total_required_inputs = 0
    total_required_outputs = 0
    role_summary = infer_boundary_stub_roles(
        data,
        label_to_cell=label_to_cell,
        net_id=net_id,
        boundary_stub_id=boundary_stub_id,
    )
    driver_stub_count = int(role_summary.get('driver_stub_count', 0))
    load_stub_count = int(role_summary.get('load_stub_count', 0))
    bidir_stub_count = int(role_summary.get('bidir_stub_count', 0))
    ambig_stub_count = int(role_summary.get('ambig_stub_count', 0))

    for gate_idx in gate_nodes:
        label = int(x[gate_idx].item())
        cell_name = label_to_cell.get(label, f'LABEL_{label}') if label_to_cell else f'LABEL_{label}'
        spec = pin_count_spec(cell_name, cell_pin_specs) if cell_pin_specs is not None else None
        if spec is None:
            spec = infer_cell_spec(cell_name)
        req_inputs = int(spec['inputs'])
        req_outputs = int(spec['outputs'])
        incident = int(degrees[gate_idx].item())
        available_inputs = max(0, incident - req_outputs)
        in_def = max(0, req_inputs - available_inputs)
        out_def = max(0, req_outputs - min(req_outputs, incident))
        family = _cell_family(cell_name)
        family_counts[family] += 1
        family_risk = _complex_cell_risk(cell_name)
        if family_risk > 1.0:
            complex_cell_count += 1
        input_deficit += in_def
        output_deficit += out_def
        total_required_inputs += req_inputs
        total_required_outputs += req_outputs
        reuse_risk += float(in_def) * max(1.0, family_risk)
        synthetic_risk += float(max(0, req_inputs - incident)) * max(1.0, family_risk)
        repaired_output_risk += float(out_def) * max(1.0, family_risk)

    pin_metrics = partition_graph_metrics(data, net_id=net_id, boundary_stub_id=boundary_stub_id)
    total_role_supply = max(1, driver_stub_count + load_stub_count + bidir_stub_count + ambig_stub_count)
    contract_profile = _future_contract_profile(data)
    return {
        'boundary_stub_count': int(pin_metrics.get('boundary_stub_count', 0)),
        'boundary_stub_on_largest_component': int(connectivity.get('boundary_stub_on_largest_component', 0.0)),
        'driver_stub_count': int(driver_stub_count),
        'load_stub_count': int(load_stub_count),
        'bidir_stub_count': int(bidir_stub_count),
        'ambig_stub_count': int(ambig_stub_count),
        'driver_stub_supply': int(driver_stub_count + bidir_stub_count),
        'load_stub_supply': int(load_stub_count + bidir_stub_count),
        'driver_stub_ratio': float((driver_stub_count + bidir_stub_count) / total_role_supply),
        'load_stub_ratio': float((load_stub_count + bidir_stub_count) / total_role_supply),
        'role_complete': int(role_summary.get('role_complete', 0)),
        'gate_pin_deficit': float(partition_pin_semantic_metrics(data, label_to_cell, net_id=net_id, boundary_stub_id=boundary_stub_id).get('gate_pin_deficit', 0.0)) if label_to_cell else 0.0,
        'underconnected_gate_fraction': float(partition_pin_semantic_metrics(data, label_to_cell, net_id=net_id, boundary_stub_id=boundary_stub_id).get('underconnected_gate_fraction', 0.0)) if label_to_cell else 0.0,
        'synthetic_input_proxy_ratio': float(partition_pin_semantic_metrics(data, label_to_cell, net_id=net_id, boundary_stub_id=boundary_stub_id).get('synthetic_input_proxy_ratio', 0.0)) if label_to_cell else 0.0,
        'cell_family_counts': {str(k): int(v) for k, v in sorted(family_counts.items())},
        'complex_cell_count': int(complex_cell_count),
        'estimated_input_deficit': int(input_deficit),
        'estimated_output_deficit': int(output_deficit),
        'estimated_reuse_risk': float(reuse_risk),
        'estimated_synthetic_input_risk': float(synthetic_risk),
        'estimated_repaired_output_risk': float(repaired_output_risk),
        'required_input_pins': int(total_required_inputs),
        'required_output_pins': int(total_required_outputs),
        **contract_profile,
    }


def _available_boundary_stubs(data: Data, boundary_stub_id: int) -> List[int]:
    x = data.x.reshape(-1).to(torch.long)
    stub_ids = (x == boundary_stub_id).nonzero(as_tuple=False).reshape(-1).tolist()
    degrees = torch.bincount(data.edge_index[0], minlength=int(x.numel()))
    largest_nodes = set(_non_stub_component_stats(data, boundary_stub_id).get('largest_component_nodes', set()))

    def key(idx: int):
        incident = []
        for src, dst in data.edge_index.t().tolist():
            if src == idx and int(x[dst].item()) != int(boundary_stub_id):
                incident.append(dst)
            elif dst == idx and int(x[src].item()) != int(boundary_stub_id):
                incident.append(src)
        attached_to_largest = any(nbr in largest_nodes for nbr in incident)
        return (0 if attached_to_largest else 1, -int(degrees[idx].item()), idx)

    return sorted(stub_ids, key=key)


def _ordered_boundary_stubs_with_profile(
    data: Data,
    boundary_stub_id: int,
    label_to_cell: Dict[int, str] | None = None,
    cell_pin_specs: Dict[str, Dict[str, List[str]]] | None = None,
) -> List[int]:
    x = data.x.reshape(-1).to(torch.long)
    stub_ids = (x == boundary_stub_id).nonzero(as_tuple=False).reshape(-1).tolist()
    degrees = torch.bincount(data.edge_index[0], minlength=int(x.numel()))
    largest_nodes = set(_non_stub_component_stats(data, boundary_stub_id).get('largest_component_nodes', set()))

    def _neighbor_nodes(stub_idx: int) -> List[int]:
        out = []
        for src, dst in data.edge_index.t().tolist():
            if src == stub_idx and int(x[dst].item()) != int(boundary_stub_id):
                out.append(dst)
            elif dst == stub_idx and int(x[src].item()) != int(boundary_stub_id):
                out.append(src)
        return out

    def key(stub_idx: int):
        nbrs = _neighbor_nodes(stub_idx)
        attached_to_largest = any(nbr in largest_nodes for nbr in nbrs)
        neighbor_degree = max([int(degrees[nbr].item()) for nbr in nbrs] or [0])
        pin_deficit = 0
        pin_slack = 0
        complex_risk = 0.0
        for nbr in nbrs:
            label = int(x[nbr].item())
            if label_to_cell is None:
                continue
            cell_name = label_to_cell.get(label, f'LABEL_{label}')
            spec = pin_count_spec(cell_name, cell_pin_specs) if cell_pin_specs is not None else None
            if spec is None:
                spec = infer_cell_spec(cell_name)
            req_total = int(spec['inputs']) + int(spec['outputs'])
            degree = int(degrees[nbr].item())
            pin_deficit += max(0, req_total - degree)
            pin_slack += max(0, degree - req_total)
            complex_risk += _complex_cell_risk(cell_name)
        return (
            0 if attached_to_largest else 1,
            pin_deficit,
            complex_risk,
            -pin_slack,
            -neighbor_degree,
            stub_idx,
        )

    return sorted(stub_ids, key=key)


def _non_stub_copy(
    data: Data,
    boundary_stub_id: int,
    global_x: List[int],
    global_edges: List[Tuple[int, int]],
    global_edge_role: List[int] | None = None,
    global_edge_pin_id: List[int] | None = None,
) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, Dict[int, Tuple[int, int]]]]:
    x = data.x.reshape(-1).to(torch.long)
    is_stub = x == boundary_stub_id
    local_to_global: Dict[int, int] = {}
    stub_neighbors: Dict[int, List[int]] = defaultdict(list)
    stub_neighbor_meta: Dict[int, Dict[int, Tuple[int, int]]] = defaultdict(dict)
    for local_idx, label in enumerate(x.tolist()):
        if label == boundary_stub_id:
            continue
        local_to_global[local_idx] = len(global_x)
        global_x.append(int(label))

    edge_role = getattr(data, 'edge_role', None)
    edge_pin_id = getattr(data, 'edge_pin_id', None)
    role_values = edge_role.reshape(-1).to(torch.long) if isinstance(edge_role, torch.Tensor) and edge_role.numel() == data.edge_index.size(1) else None
    pin_values = edge_pin_id.reshape(-1).to(torch.long) if isinstance(edge_pin_id, torch.Tensor) and edge_pin_id.numel() == data.edge_index.size(1) else None

    for eidx, (src, dst) in enumerate(data.edge_index.t().tolist()):
        src_stub = bool(is_stub[src].item())
        dst_stub = bool(is_stub[dst].item())
        e_role = int(role_values[int(eidx)].item()) if role_values is not None else 0
        e_pin = int(pin_values[int(eidx)].item()) if pin_values is not None else -1
        if not src_stub and not dst_stub:
            global_edges.append((local_to_global[src], local_to_global[dst]))
            if global_edge_role is not None:
                global_edge_role.append(int(e_role))
            if global_edge_pin_id is not None:
                global_edge_pin_id.append(int(e_pin))
        elif src_stub and not dst_stub:
            nbr = int(local_to_global[dst])
            if nbr not in stub_neighbor_meta[int(src)]:
                stub_neighbors[src].append(nbr)
                stub_neighbor_meta[int(src)][nbr] = (int(e_role), int(e_pin))
        elif dst_stub and not src_stub:
            nbr = int(local_to_global[src])
            if nbr not in stub_neighbor_meta[int(dst)]:
                stub_neighbors[dst].append(nbr)
                stub_neighbor_meta[int(dst)][nbr] = (int(e_role), int(e_pin))
    return local_to_global, stub_neighbors, stub_neighbor_meta


def _canonical_skeleton_edges(skeleton: Data) -> Dict[Tuple[int, int], int]:
    unique_edges = {}
    edge_attr = getattr(skeleton, 'edge_attr', None)
    for edge_idx, (src, dst) in enumerate(skeleton.edge_index.t().tolist()):
        if src == dst:
            continue
        key = (src, dst) if src < dst else (dst, src)
        if key in unique_edges:
            continue
        label = 0 if edge_attr is None or edge_attr.numel() == 0 else int(edge_attr[edge_idx].item())
        unique_edges[key] = label
    return unique_edges


def _edge_attr_value(skeleton: Data, attr: str, edge_idx: int, default: int = 0) -> int:
    if not hasattr(skeleton, attr):
        return int(default)
    value = getattr(skeleton, attr)
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return int(default)
    flat = value.reshape(-1).to(torch.long)
    if int(flat.numel()) == 1:
        return int(flat[0].item())
    if 0 <= int(edge_idx) < int(flat.numel()):
        return int(flat[int(edge_idx)].item())
    return int(default)


def _canonical_skeleton_edge_records(skeleton: Data) -> Dict[Tuple[int, int], Dict[str, int]]:
    records: Dict[Tuple[int, int], Dict[str, int]] = {}
    edge_attr = getattr(skeleton, 'edge_attr', None)
    for edge_idx, (src, dst) in enumerate(skeleton.edge_index.t().tolist()):
        if src == dst:
            continue
        src = int(src)
        dst = int(dst)
        reverse = src > dst
        key = (src, dst) if src < dst else (dst, src)
        if key in records:
            continue
        raw_label = 0 if edge_attr is None or edge_attr.numel() == 0 else int(edge_attr[edge_idx].item())
        direction_prior = _edge_attr_value(skeleton, 'schema_edge_direction_prior', edge_idx, 0)
        if reverse:
            if direction_prior == 1:
                direction_prior = 2
            elif direction_prior == 2:
                direction_prior = 1
        records[key] = {
            'src_partition': int(key[0]),
            'dst_partition': int(key[1]),
            'raw_edge_class': int(raw_label),
            'typed_role_type': _edge_attr_value(skeleton, 'schema_edge_role_type', edge_idx, 0),
            'typed_direction_prior': int(direction_prior),
            'typed_fanout': _edge_attr_value(skeleton, 'schema_edge_fanout', edge_idx, 0),
            'typed_group_size': _edge_attr_value(skeleton, 'schema_edge_group_size', edge_idx, 0),
            'typed_shared_net_count': _edge_attr_value(skeleton, 'schema_edge_shared_net_count', edge_idx, 0),
            'typed_shared_net_group_count': _edge_attr_value(skeleton, 'schema_edge_shared_net_group_count', edge_idx, 0),
            'typed_contract_risk': _edge_attr_value(skeleton, 'schema_edge_contract_risk', edge_idx, 0),
        }
    return records


def _decode_skeleton_cut_label(label: int, meta: Dict) -> int:
    """Map a possibly promoted typed-edge label back to cut multiplicity."""
    raw_label = max(0, int(label))
    typed_info = dict(meta.get('typed_edge_enrichment') or {})
    if typed_info.get('promote_edge_attr') == 'role_direction_cut':
        base = max(1, int(typed_info.get('promote_original_edge_classes') or 1))
        raw_label = raw_label % base
    edge_spec = meta['buckets']['cut_multiplicity']
    max_label = max(0, len(edge_spec.get('representative_values', [])) - 1)
    return max(0, min(raw_label, max_label))


def _strict_requested_multiplicity(label: int, edge_spec: Dict) -> int:
    return max(1, int(round(edge_spec['representative_values'][label])))


def _conservative_requested_multiplicity(label: int, edge_spec: Dict) -> int:
    upper_bounds = edge_spec['upper_bounds']
    if int(label) <= 0:
        return 1
    lower_bound = upper_bounds[int(label) - 1] + 1
    return max(1, int(lower_bound))


def skeleton_stub_requirements(skeleton: Data, meta: Dict, mode: str = 'conservative') -> Dict[int, int]:
    edge_spec = meta['buckets']['cut_multiplicity']
    requirements: Dict[int, int] = defaultdict(int)
    for (src, dst), label in _canonical_skeleton_edges(skeleton).items():
        cut_label = _decode_skeleton_cut_label(label, meta)
        if mode == 'strict':
            requested = _strict_requested_multiplicity(cut_label, edge_spec)
        elif mode == 'conservative':
            requested = min(_conservative_requested_multiplicity(cut_label, edge_spec), 1)
        elif mode == 'capacity':
            requested = _conservative_requested_multiplicity(cut_label, edge_spec)
        else:
            raise ValueError(f'unknown assembly mode: {mode}')
        requirements[int(src)] += int(requested)
        requirements[int(dst)] += int(requested)
    return dict(requirements)


def _plan_connection_counts(
    unique_edges: Dict[Tuple[int, int], object],
    stub_inventory: Dict[int, List[Tuple[int, List[int]]]],
    meta: Dict,
    mode: str,
    *,
    enable_cut_group_expansion: bool = False,
    cut_group_source: str = 'typed_shared_net_group_count',
    cut_group_pair_counts: Dict[Tuple[int, int], int] | None = None,
    max_cut_groups_per_pair: int = 1,
    min_cut_groups_per_pair: int = 1,
    cut_group_expansion_mode: str = 'smoke',
) -> Tuple[List[Dict], Dict[int, int], List[Dict]]:
    edge_spec = meta['buckets']['cut_multiplicity']
    remaining = {node: len(stubs) for node, stubs in stub_inventory.items()}
    plan: List[Dict] = []
    dropped: List[Dict] = []

    for (src, dst), edge_payload in sorted(unique_edges.items()):
        if isinstance(edge_payload, dict):
            raw_label = int(edge_payload.get('raw_edge_class', edge_payload.get('edge_class', 0)))
            typed_payload = edge_payload
        else:
            raw_label = int(edge_payload)
            typed_payload = {}
        label = _decode_skeleton_cut_label(raw_label, meta)
        source_count = None
        if enable_cut_group_expansion:
            key = (int(src), int(dst)) if int(src) < int(dst) else (int(dst), int(src))
            if cut_group_source == 'manifest' and cut_group_pair_counts is not None:
                source_count = int(cut_group_pair_counts.get(key, 0) or 0)
            elif cut_group_source in {'typed_shared_net_group_count', 'typed_shared_net_group_count_max'}:
                source_count = int(typed_payload.get('typed_shared_net_group_count', 0) or 0)
            if source_count is None or source_count <= 0:
                source_count = _conservative_requested_multiplicity(label, edge_spec)
            requested_cut_groups = max(int(min_cut_groups_per_pair), int(source_count))
            if int(max_cut_groups_per_pair) > 0:
                requested_cut_groups = min(requested_cut_groups, int(max_cut_groups_per_pair))
            requested_cut_groups = max(1, int(requested_cut_groups))
            driver_available = int(remaining.get(src, 0))
            load_available = int(remaining.get(dst, 0))
            planned = min(requested_cut_groups, driver_available, load_available)
            if planned <= 0:
                reason = 'driver_and_load_stub_shortage'
                if driver_available > 0 and load_available <= 0:
                    reason = 'load_stub_shortage'
                elif driver_available <= 0 and load_available > 0:
                    reason = 'driver_stub_shortage'
                dropped.append(
                    {
                        'src_partition': int(src),
                        'dst_partition': int(dst),
                        'edge_class': int(label),
                        'raw_edge_class': int(raw_label),
                        'requested': int(requested_cut_groups),
                        'requested_cut_group_count': int(requested_cut_groups),
                        'planned_cut_group_count': 0,
                        'dropped_cut_group_count': int(requested_cut_groups),
                        'reason': reason,
                        'cut_group_expansion_enabled': True,
                        'cut_group_expansion_source': str(cut_group_source),
                    }
                )
                continue
            dropped_count = max(0, requested_cut_groups - planned)
            if dropped_count > 0:
                reason = 'stub_shortage'
                if driver_available < requested_cut_groups and load_available < requested_cut_groups:
                    reason = 'driver_and_load_stub_shortage'
                elif driver_available < requested_cut_groups:
                    reason = 'driver_stub_shortage'
                elif load_available < requested_cut_groups:
                    reason = 'load_stub_shortage'
                dropped.append(
                    {
                        'src_partition': int(src),
                        'dst_partition': int(dst),
                        'edge_class': int(label),
                        'raw_edge_class': int(raw_label),
                        'requested': int(requested_cut_groups),
                        'requested_cut_group_count': int(requested_cut_groups),
                        'planned_cut_group_count': int(planned),
                        'dropped_cut_group_count': int(dropped_count),
                        'reason': reason,
                        'cut_group_expansion_enabled': True,
                        'cut_group_expansion_source': str(cut_group_source),
                    }
                )
            requested = int(requested_cut_groups)
        elif mode == 'strict':
            requested = _strict_requested_multiplicity(label, edge_spec)
            planned = requested
            if remaining.get(src, 0) < planned or remaining.get(dst, 0) < planned:
                raise RuntimeError(
                    f'unsatisfied interface demand between skeleton nodes {src} and {dst}: '
                    f'need {planned} stubs per side, have {remaining.get(src, 0)} and {remaining.get(dst, 0)}'
                )
        elif mode == 'conservative':
            requested = _conservative_requested_multiplicity(label, edge_spec)
            planned = min(requested, remaining.get(src, 0), remaining.get(dst, 0), 1)
            if planned <= 0:
                dropped.append(
                    {
                        'src_partition': int(src),
                        'dst_partition': int(dst),
                        'edge_class': int(label),
                        'raw_edge_class': int(raw_label),
                        'requested': int(requested),
                        'reason': 'stub_budget_exhausted',
                    }
                )
                continue
        elif mode == 'capacity':
            requested = _conservative_requested_multiplicity(label, edge_spec)
            planned = min(requested, remaining.get(src, 0), remaining.get(dst, 0))
            if planned <= 0:
                dropped.append(
                    {
                        'src_partition': int(src),
                        'dst_partition': int(dst),
                        'edge_class': int(label),
                        'raw_edge_class': int(raw_label),
                        'requested': int(requested),
                        'reason': 'stub_budget_exhausted',
                    }
                )
                continue
        else:
            raise ValueError(f'unknown assembly mode: {mode}')

        remaining[src] = remaining.get(src, 0) - planned
        remaining[dst] = remaining.get(dst, 0) - planned
        plan.append(
            {
                'src_partition': int(src),
                'dst_partition': int(dst),
                'edge_class': int(label),
                'raw_edge_class': int(raw_label),
                'requested': int(requested),
                'planned': int(planned),
                'requested_cut_group_count': int(requested),
                'planned_cut_group_count': int(planned),
                'cut_group_expansion_enabled': bool(enable_cut_group_expansion),
                'cut_group_expansion_source': str(cut_group_source if enable_cut_group_expansion else 'disabled'),
                'cut_group_expansion_cap': int(max_cut_groups_per_pair),
                'cut_group_expansion_mode': str(cut_group_expansion_mode),
                'force_separate_global_net': bool(enable_cut_group_expansion),
                'typed_role_type': int(typed_payload.get('typed_role_type', 0)),
                'typed_direction_prior': int(typed_payload.get('typed_direction_prior', 0)),
                'typed_fanout': int(typed_payload.get('typed_fanout', 0)),
                'typed_group_size': int(typed_payload.get('typed_group_size', 0)),
                'typed_shared_net_count': int(typed_payload.get('typed_shared_net_count', 0)),
                'typed_shared_net_group_count': int(typed_payload.get('typed_shared_net_group_count', 0)),
                'typed_contract_risk': int(typed_payload.get('typed_contract_risk', 0)),
            }
        )
    return plan, remaining, dropped


def _stub_role_for_node(
    data: Data,
    boundary_stub_id: int,
    stub_idx: int,
    *,
    net_id: int | None = None,
    label_to_cell: Dict[int, str] | None = None,
) -> int:
    roles = getattr(data, 'schema_boundary_stub_roles', None)
    if roles is None:
        if net_id is None:
            return 0
        try:
            roles = infer_boundary_stub_roles(
                data,
                label_to_cell=label_to_cell,
                net_id=int(net_id),
                boundary_stub_id=int(boundary_stub_id),
            ).get('stub_roles')
        except Exception:
            return 0
        if roles is None:
            return 0
    try:
        role_values = roles.reshape(-1).to(torch.long)
        x = data.x.reshape(-1).to(torch.long)
        if int(role_values.numel()) == int(x.numel()):
            if 0 <= int(stub_idx) < int(role_values.numel()):
                return int(role_values[int(stub_idx)].item())
            return 0
        # Backward-compatible path if a future dataset stores compact
        # stub-order roles rather than full node-index roles.
        stub_nodes = (x == int(boundary_stub_id)).nonzero(as_tuple=False).reshape(-1).tolist()
        local_order = {int(node): idx for idx, node in enumerate(stub_nodes)}
        order = local_order.get(int(stub_idx))
        if order is None or order < 0 or order >= int(role_values.numel()):
            return 0
        return int(role_values[int(order)].item())
    except Exception:
        return 0


def _pop_preferred_stub(
    stubs: List[Tuple[int, List[int], int, Dict[int, Tuple[int, int]]]],
    *,
    prefer_driver: bool,
) -> Tuple[int, List[int], int, Dict[int, Tuple[int, int]], bool]:
    if not stubs:
        raise RuntimeError('stub inventory exhausted')
    if prefer_driver:
        priority = {1: 0, 3: 1, 0: 2, -1: 3, 2: 4}
    else:
        priority = {2: 0, 3: 1, 0: 2, -1: 3, 1: 4}
    best_idx = min(range(len(stubs)), key=lambda idx: (priority.get(int(stubs[idx][2]), 5), idx))
    stub_idx, neighbors, role, neighbor_meta = stubs.pop(best_idx)
    matched = (int(role) in ({1, 3} if prefer_driver else {2, 3}))
    return int(stub_idx), neighbors, int(role), neighbor_meta, bool(matched)


def _connection_driver_side(src_assignment: Dict, dst_assignment: Dict, src_role: int = 0, dst_role: int = 0) -> Tuple[int, int, str]:
    src_node = int(src_assignment['skeleton_node'])
    dst_node = int(dst_assignment['skeleton_node'])
    if src_role in {1, 3} and dst_role == 2:
        return src_node, dst_node, 'stub_role'
    if dst_role in {1, 3} and src_role == 2:
        return dst_node, src_node, 'stub_role'

    src_profile = src_assignment.get('interface_profile', {}) or {}
    dst_profile = dst_assignment.get('interface_profile', {}) or {}
    src_score = (
        float(src_profile.get('estimated_output_deficit', 0.0))
        + 0.25 * float(src_profile.get('estimated_repaired_output_risk', 0.0))
        - 0.1 * float(src_profile.get('boundary_stub_on_largest_component', 0.0))
    )
    dst_score = (
        float(dst_profile.get('estimated_output_deficit', 0.0))
        + 0.25 * float(dst_profile.get('estimated_repaired_output_risk', 0.0))
        - 0.1 * float(dst_profile.get('boundary_stub_on_largest_component', 0.0))
    )
    if src_score <= dst_score:
        return src_node, dst_node, 'output_deficit_proxy'
    return dst_node, src_node, 'output_deficit_proxy'


def _edge_orientation_local_cost(
    driver_assignment: Dict,
    load_assignment: Dict,
) -> float:
    driver_profile = driver_assignment.get('interface_profile', {}) or {}
    load_profile = load_assignment.get('interface_profile', {}) or {}
    driver_metrics = driver_assignment.get('metrics', {}) or {}
    load_metrics = load_assignment.get('metrics', {}) or {}
    return float(
        4.0 * float(driver_profile.get('estimated_output_deficit', 0.0))
        + 1.0 * float(driver_profile.get('estimated_repaired_output_risk', 0.0))
        + 5.0 * float(load_profile.get('estimated_input_deficit', 0.0))
        + 1.5 * float(load_profile.get('estimated_reuse_risk', 0.0))
        + 2.0 * float(load_profile.get('estimated_synthetic_input_risk', 0.0))
        + 4.0 * float(load_metrics.get('gate_pin_deficit', 0.0))
        + 2.0 * float(load_metrics.get('synthetic_input_proxy_ratio', 0.0))
        + 0.5 * float(driver_metrics.get('underconnected_gate_fraction', 0.0))
        + 0.5 * float(load_metrics.get('underconnected_gate_fraction', 0.0))
    )


def _orientation_budget_penalty(
    usage: Dict[int, Dict[str, float]],
    assignments_by_node: Dict[int, Dict],
) -> float:
    penalty = 0.0
    for node_idx, counts in usage.items():
        profile = (assignments_by_node.get(int(node_idx), {}) or {}).get('interface_profile', {}) or {}
        driver_supply = max(0.0, float(profile.get('driver_stub_supply', 0.0)))
        load_supply = max(0.0, float(profile.get('load_stub_supply', 0.0)))
        driver_used = float(counts.get('driver_edges', 0.0))
        load_used = float(counts.get('load_edges', 0.0))
        driver_over = max(0.0, driver_used - driver_supply)
        load_over = max(0.0, load_used - load_supply)
        penalty += 220.0 * driver_over
        penalty += 340.0 * load_over
        penalty += 12.0 * abs(driver_used - load_used)
    return float(penalty)


def _expand_connection_instances(connection_plan: List[Dict]) -> List[Dict]:
    expanded: List[Dict] = []
    edge_idx = 0
    for item in connection_plan:
        for local_idx in range(int(item.get('planned', 0))):
            expanded.append(
                {
                    'edge_instance_id': int(edge_idx),
                    'src_partition': int(item['src_partition']),
                    'dst_partition': int(item['dst_partition']),
                    'edge_class': int(item['edge_class']),
                    'requested': int(item['requested']),
                    'planned_multiplicity': int(item.get('planned', 0)),
                    'local_instance_idx': int(local_idx),
                    'force_separate_global_net': bool(item.get('force_separate_global_net', False)),
                    'requested_cut_group_count': int(item.get('requested_cut_group_count', item.get('requested', 1))),
                    'planned_cut_group_count': int(item.get('planned_cut_group_count', item.get('planned', 0))),
                    'cut_group_expansion_enabled': bool(item.get('cut_group_expansion_enabled', False)),
                    'cut_group_expansion_source': str(item.get('cut_group_expansion_source', 'disabled')),
                    'cut_group_expansion_cap': int(item.get('cut_group_expansion_cap', 0)),
                    'cut_group_expansion_mode': str(item.get('cut_group_expansion_mode', 'disabled')),
                    'typed_role_type': int(item.get('typed_role_type', 0)),
                    'typed_direction_prior': int(item.get('typed_direction_prior', 0)),
                    'typed_fanout': int(item.get('typed_fanout', 0)),
                    'typed_group_size': int(item.get('typed_group_size', 0)),
                    'typed_shared_net_count': int(item.get('typed_shared_net_count', 0)),
                    'typed_shared_net_group_count': int(item.get('typed_shared_net_group_count', 0)),
                    'typed_contract_risk': int(item.get('typed_contract_risk', 0)),
                }
            )
            edge_idx += 1
    return expanded


def _plan_oriented_connection_instances(
    connection_plan: List[Dict],
    assignments_by_node: Dict[int, Dict],
    *,
    beam_width: int = 256,
) -> Tuple[List[Dict], Dict[str, object]]:
    instances = _expand_connection_instances(connection_plan)
    if not instances:
        return [], {
            'driver_edge_total': 0.0,
            'load_edge_total': 0.0,
            'driver_overuse_total': 0.0,
            'load_overuse_total': 0.0,
            'orientation_beam_score': 0.0,
            'orientation_fallback_count': 0.0,
            'orientation_reason_counts': {},
        }

    def _instance_priority(item: Dict) -> Tuple[float, int]:
        src = int(item['src_partition'])
        dst = int(item['dst_partition'])
        src_profile = (assignments_by_node.get(src, {}) or {}).get('interface_profile', {}) or {}
        dst_profile = (assignments_by_node.get(dst, {}) or {}).get('interface_profile', {}) or {}
        tightness = min(
            float(src_profile.get('driver_stub_supply', 0.0) + src_profile.get('load_stub_supply', 0.0)),
            float(dst_profile.get('driver_stub_supply', 0.0) + dst_profile.get('load_stub_supply', 0.0)),
        )
        return (tightness, int(item['edge_instance_id']))

    ordered_instances = sorted(instances, key=_instance_priority)
    fallback_count = 0
    usage: Dict[int, Dict[str, float]] = {}
    oriented: List[Dict] = []
    reason_counts: Counter = Counter()
    total_score = 0.0

    # The original version used a beam over all edge orientations.  That is
    # unnecessarily expensive for generated skeletons with hundreds of
    # cross-partition edge instances.  A deterministic greedy planner is
    # sufficient here because orientation is a legalization/budget step: score
    # both directions against current role usage and commit the cheaper one.
    for item in ordered_instances:
        src = int(item['src_partition'])
        dst = int(item['dst_partition'])
        src_assignment = assignments_by_node.get(src, {'skeleton_node': src, 'interface_profile': {}, 'metrics': {}})
        dst_assignment = assignments_by_node.get(dst, {'skeleton_node': dst, 'interface_profile': {}, 'metrics': {}})
        typed_direction = int(item.get('typed_direction_prior', 0))
        if typed_direction == 1:
            heur_driver, heur_load, heur_reason = src, dst, 'typed_edge_direction_prior'
        elif typed_direction == 2:
            heur_driver, heur_load, heur_reason = dst, src, 'typed_edge_direction_prior'
        else:
            heur_driver, heur_load, heur_reason = _connection_driver_side(src_assignment, dst_assignment)
        options = []
        if heur_driver == src and heur_load == dst:
            options = [(src, dst, heur_reason), (dst, src, 'fallback_reverse')]
        else:
            options = [(dst, src, heur_reason), (src, dst, 'fallback_reverse')]

        choices = []
        for driver_partition, load_partition, reason in options:
            driver_assignment = assignments_by_node[int(driver_partition)]
            load_assignment = assignments_by_node[int(load_partition)]
            local_cost = _edge_orientation_local_cost(driver_assignment, load_assignment)
            new_usage = {
                int(node): {
                    'driver_edges': float(stats.get('driver_edges', 0.0)),
                    'load_edges': float(stats.get('load_edges', 0.0)),
                }
                for node, stats in usage.items()
            }
            new_usage.setdefault(int(driver_partition), {'driver_edges': 0.0, 'load_edges': 0.0})
            new_usage.setdefault(int(load_partition), {'driver_edges': 0.0, 'load_edges': 0.0})
            old_budget = _orientation_budget_penalty(usage, assignments_by_node)
            new_usage[int(driver_partition)]['driver_edges'] += 1.0
            new_usage[int(load_partition)]['load_edges'] += 1.0
            budget_delta = _orientation_budget_penalty(new_usage, assignments_by_node) - old_budget
            choices.append(
                (
                    float(local_cost + budget_delta),
                    int(driver_partition),
                    int(load_partition),
                    str(reason),
                    new_usage,
                )
            )
        if not choices:
            fallback_count += 1
            continue
        choices.sort(key=lambda entry: (entry[0], entry[1], entry[2], entry[3]))
        choice_score, driver_partition, load_partition, reason, usage = choices[0]
        total_score += float(choice_score)
        reason_counts[str(reason)] += 1
        oriented.append(
            {
                **item,
                'driver_partition': int(driver_partition),
                'load_partition': int(load_partition),
                'driver_selection_reason': str(reason),
                'typed_direction_used': bool(str(reason) == 'typed_edge_direction_prior'),
            }
        )

    driver_overuse_total = 0.0
    load_overuse_total = 0.0
    for node_idx, stats in usage.items():
        profile = (assignments_by_node.get(int(node_idx), {}) or {}).get('interface_profile', {}) or {}
        driver_supply = max(0.0, float(profile.get('driver_stub_supply', 0.0)))
        load_supply = max(0.0, float(profile.get('load_stub_supply', 0.0)))
        driver_overuse_total += max(0.0, float(stats.get('driver_edges', 0.0)) - driver_supply)
        load_overuse_total += max(0.0, float(stats.get('load_edges', 0.0)) - load_supply)

    return oriented, {
        'driver_edge_total': float(sum(float(stats.get('driver_edges', 0.0)) for stats in usage.values())),
        'load_edge_total': float(sum(float(stats.get('load_edges', 0.0)) for stats in usage.values())),
        'driver_overuse_total': float(driver_overuse_total),
        'load_overuse_total': float(load_overuse_total),
        'orientation_beam_score': float(total_score),
        'orientation_algorithm': 'greedy_role_budget',
        'orientation_fallback_count': float(fallback_count),
        'orientation_reason_counts': {str(k): int(v) for k, v in sorted(reason_counts.items())},
    }


def _build_global_net_plan(
    connection_plan: List[Dict],
    assignments_by_node: Dict[int, Dict],
) -> Tuple[List[Dict], Dict[str, object]]:
    oriented_instances, orientation_summary = _plan_oriented_connection_instances(
        connection_plan,
        assignments_by_node,
    )
    grouped: Dict[Tuple[int, int, int], List[Dict]] = defaultdict(list)
    for item in oriented_instances:
        driver_partition = int(item['driver_partition'])
        edge_class = int(item['edge_class'])
        profile = (assignments_by_node.get(driver_partition, {}) or {}).get('interface_profile', {}) or {}
        driver_supply = max(1, int(profile.get('driver_stub_supply', 1)))
        typed_fanout = max(0, int(item.get('typed_fanout', 0)))
        typed_group_size = max(0, int(item.get('typed_group_size', 0)))
        typed_shared_groups = max(0, int(item.get('typed_shared_net_group_count', 0)))
        if typed_fanout > 1:
            fanout_cap = max(2, min(16, typed_fanout))
        elif typed_group_size > 1:
            fanout_cap = max(2, min(16, typed_group_size))
        else:
            fanout_cap = max(2, min(8, driver_supply * 2))
        # Spread loads across a small number of planned driver nets rather than
        # giving every pairwise edge its own net.
        semantic_bucket = int(min(max(typed_shared_groups, typed_group_size, typed_fanout), 16))
        if bool(item.get('force_separate_global_net', False)):
            group_id = int(item.get('edge_instance_id', 0))
            grouped[(driver_partition, edge_class, group_id, semantic_bucket)].append(item)
            continue
        existing_group_ids = [
            key for key in grouped.keys()
            if int(key[0]) == driver_partition and int(key[1]) == edge_class and int(key[3]) == semantic_bucket
        ]
        target_group_id = None
        for key in sorted(existing_group_ids, key=lambda value: value[2]):
            if len(grouped[key]) < fanout_cap:
                target_group_id = key[2]
                break
        if target_group_id is None:
            target_group_id = max([key[2] for key in existing_group_ids] + [-1]) + 1
        grouped[(driver_partition, edge_class, target_group_id, semantic_bucket)].append(item)

    planned: List[Dict] = []
    net_idx = 0
    group_sizes: List[int] = []
    typed_direction_used = 0
    for (driver_partition, edge_class, group_id, semantic_bucket), items in sorted(grouped.items()):
        load_partitions = [int(item['load_partition']) for item in items]
        endpoints = sorted({int(item['src_partition']) for item in items} | {int(item['dst_partition']) for item in items})
        reason_counts = Counter(str(item.get('driver_selection_reason', 'unknown')) for item in items)
        typed_direction_used += sum(1 for item in items if item.get('typed_direction_used'))
        planned.append(
            {
                'planned_net_id': int(net_idx),
                'driver_partition': int(driver_partition),
                'load_partitions': load_partitions,
                'group_fanout': int(len(load_partitions)),
                'group_id': int(group_id),
                'semantic_bucket': int(semantic_bucket),
                'group_endpoints': endpoints,
                'driver_selection_reason': str(reason_counts.most_common(1)[0][0]) if reason_counts else 'unknown',
                'driver_selection_reason_counts': {str(k): int(v) for k, v in sorted(reason_counts.items())},
                'edge_class': int(edge_class),
                'typed_fanout_max': int(max(int(item.get('typed_fanout', 0)) for item in items) if items else 0),
                'typed_group_size_max': int(max(int(item.get('typed_group_size', 0)) for item in items) if items else 0),
                'typed_shared_net_group_count_max': int(max(int(item.get('typed_shared_net_group_count', 0)) for item in items) if items else 0),
                'requested': int(sum(int(item.get('requested', 1)) for item in items)),
                'requested_cut_group_count': int(max(int(item.get('requested_cut_group_count', 1)) for item in items) if items else 0),
                'planned_cut_group_count': int(len(items)),
                'cut_group_expansion_enabled': bool(any(bool(item.get('cut_group_expansion_enabled', False)) for item in items)),
                'cut_group_expansion_source': str(next((item.get('cut_group_expansion_source') for item in items if item.get('cut_group_expansion_source')), 'disabled')),
                'cut_group_expansion_cap': int(max(int(item.get('cut_group_expansion_cap', 0)) for item in items) if items else 0),
                'cut_group_expansion_mode': str(next((item.get('cut_group_expansion_mode') for item in items if item.get('cut_group_expansion_mode')), 'disabled')),
                'grouped_edge_instance_ids': [int(item['edge_instance_id']) for item in items],
                'grouped_edge_count': int(len(items)),
            }
        )
        group_sizes.append(len(load_partitions))
        net_idx += 1
    orientation_summary = {
        **orientation_summary,
        'planned_net_count': float(len(planned)),
        'avg_group_fanout': float(sum(group_sizes) / max(1, len(group_sizes))),
        'max_group_fanout': float(max(group_sizes) if group_sizes else 0.0),
        'typed_direction_used_count': float(typed_direction_used),
        'typed_net_plan_overlay': True,
    }
    return planned, orientation_summary


def _summarize_role_slot_budget(
    global_net_plan: List[Dict],
    assignments_by_node: Dict[int, Dict],
) -> Dict[str, object]:
    per_node: Dict[int, Dict[str, float]] = {}
    for node_idx, assignment in assignments_by_node.items():
        profile = assignment.get('interface_profile', {}) or {}
        per_node[int(node_idx)] = {
            'driver_demand': 0.0,
            'load_demand': 0.0,
            'driver_supply': float(profile.get('driver_stub_supply', 0.0)),
            'load_supply': float(profile.get('load_stub_supply', 0.0)),
            'driver_shortfall': 0.0,
            'load_shortfall': 0.0,
        }

    for item in global_net_plan:
        driver_node = int(item.get('driver_partition', -1))
        if driver_node in per_node:
            per_node[driver_node]['driver_demand'] += 1.0
        for load_node in item.get('load_partitions', []):
            load_node = int(load_node)
            if load_node in per_node:
                per_node[load_node]['load_demand'] += 1.0

    totals = {
        'driver_demand_total': 0.0,
        'load_demand_total': 0.0,
        'driver_supply_total': 0.0,
        'load_supply_total': 0.0,
        'driver_shortfall_total': 0.0,
        'load_shortfall_total': 0.0,
        'role_slot_deficit_nodes': 0.0,
    }
    serializable_nodes: Dict[str, Dict[str, float]] = {}
    for node_idx, stats in sorted(per_node.items()):
        stats['driver_shortfall'] = max(0.0, stats['driver_demand'] - stats['driver_supply'])
        stats['load_shortfall'] = max(0.0, stats['load_demand'] - stats['load_supply'])
        if stats['driver_shortfall'] > 0 or stats['load_shortfall'] > 0:
            totals['role_slot_deficit_nodes'] += 1.0
        for key in ['driver_demand', 'load_demand', 'driver_supply', 'load_supply', 'driver_shortfall', 'load_shortfall']:
            totals[f'{key}_total'] = totals.get(f'{key}_total', 0.0) + float(stats[key])
        serializable_nodes[str(int(node_idx))] = {key: float(val) for key, val in stats.items()}

    totals['driver_utilization_ratio'] = float(
        totals['driver_demand_total'] / max(1.0, totals['driver_supply_total'])
    )
    totals['load_utilization_ratio'] = float(
        totals['load_demand_total'] / max(1.0, totals['load_supply_total'])
    )
    totals['used_driver_stub_count'] = float(totals['driver_demand_total'])
    totals['used_load_stub_count'] = float(totals['load_demand_total'])
    totals['unused_driver_stub_count'] = float(max(0.0, totals['driver_supply_total'] - totals['driver_demand_total']))
    totals['unused_load_stub_count'] = float(max(0.0, totals['load_supply_total'] - totals['load_demand_total']))
    # Compatibility keys consumed by assembly smoke reports.
    totals['used_driver_stubs'] = totals['used_driver_stub_count']
    totals['used_load_stubs'] = totals['used_load_stub_count']
    totals['unused_driver_stubs'] = totals['unused_driver_stub_count']
    totals['unused_load_stubs'] = totals['unused_load_stub_count']
    return {
        'summary': {str(k): float(v) for k, v in totals.items()},
        'per_node': serializable_nodes,
    }


def _semantic_project_assembled_graph(
    assembled: Data,
    label_to_cell: Dict[int, str],
    net_id: int,
    mode: str = 'repair',
    cell_pin_specs: Dict[str, Dict[str, List[str]]] | None = None,
) -> Tuple[Data, Dict[str, int]]:
    if mode not in {'audit', 'conservative', 'repair'}:
        raise ValueError(f'unknown semantic projection mode: {mode}')
    data = normalize_circuit_graph(assembled, net_id=net_id, remove_isolates=False)
    x_list = data.x.reshape(-1).to(torch.long).tolist()
    neighbors = [set() for _ in range(len(x_list))]
    for src, dst in data.edge_index.t().tolist():
        if src == dst:
            continue
        neighbors[src].add(dst)
        neighbors[dst].add(src)

    cell_to_label = {str(cell): int(label) for label, cell in label_to_cell.items()}
    helper_label = cell_to_label.get('BUF_X1')
    if helper_label is None:
        helper_label = cell_to_label.get('INV_X1')

    gate_nodes = [idx for idx, label in enumerate(x_list) if int(label) != int(net_id)]
    net_nodes = {idx for idx, label in enumerate(x_list) if int(label) == int(net_id)}
    output_owner: Dict[int, int] = {}
    summary = {
        'projection_mode': mode,
        'gates_processed': 0,
        'gates_changed': 0,
        'would_add_output_nets': 0,
        'would_add_input_edges': 0,
        'would_add_input_nets': 0,
        'would_reuse_input_nets': 0,
        'would_remove_excess_edges': 0,
        'would_insert_helper_gates': 0,
        'added_output_nets': 0,
        'added_input_edges': 0,
        'added_input_nets': 0,
        'removed_excess_edges': 0,
        'helper_gates_inserted': 0,
        'helper_nets_inserted': 0,
        'helper_input_repairs': 0,
        'missing_liberty_cell_specs': 0,
    }

    def _is_net(node_idx: int) -> bool:
        return int(x_list[node_idx]) == int(net_id)

    def _owned_outputs(gate_idx: int):
        return [
            net for net in neighbors[gate_idx]
            if _is_net(net) and output_owner.get(net) == gate_idx
        ]

    def _input_nets(gate_idx: int):
        return [
            net for net in neighbors[gate_idx]
            if _is_net(net) and output_owner.get(net) != gate_idx
        ]

    def _choose_existing_outputs(gate_idx: int, req_outputs: int, req_inputs: int) -> List[int]:
        if req_outputs <= 0:
            return []
        incident_nets = sorted(net for net in neighbors[gate_idx] if _is_net(net))
        already_owned = sorted(net for net in incident_nets if output_owner.get(net) == gate_idx)
        if len(already_owned) >= req_outputs:
            return already_owned[:req_outputs]
        needed = req_outputs - len(already_owned)
        free_nets = sorted(net for net in incident_nets if output_owner.get(net) is None)
        choose = min(needed, len(free_nets))
        if choose <= 0:
            return already_owned

        best_subset = None
        best_score = None
        for subset in itertools.combinations(free_nets, choose):
            subset_set = set(subset)
            remaining_inputs = [
                net for net in incident_nets
                if net not in subset_set and net not in set(already_owned)
            ]
            shortage = max(0, req_inputs - len(remaining_inputs))
            score = (
                shortage,
                -sum(len(neighbors[net]) for net in subset),
                tuple(sorted(subset)),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_subset = list(subset)
        chosen = sorted(best_subset) if best_subset is not None else free_nets[:choose]
        return already_owned + chosen

    def _add_private_net(gate_idx: int, reserve_output: bool = False) -> int:
        new_idx = len(x_list)
        x_list.append(int(net_id))
        neighbors.append({gate_idx})
        neighbors[gate_idx].add(new_idx)
        net_nodes.add(new_idx)
        if reserve_output:
            output_owner[new_idx] = gate_idx
        return new_idx

    def _insert_helper_stage(source_net: int, target_gate: int) -> None:
        helper_gate_idx = len(x_list)
        x_list.append(int(helper_label))
        neighbors.append({source_net})
        neighbors[source_net].add(helper_gate_idx)

        helper_out_net = len(x_list)
        x_list.append(int(net_id))
        neighbors.append({helper_gate_idx, target_gate})
        neighbors[helper_gate_idx].add(helper_out_net)
        neighbors[target_gate].add(helper_out_net)
        net_nodes.add(helper_out_net)

        summary['helper_gates_inserted'] += 1
        summary['helper_nets_inserted'] += 1
        summary['helper_input_repairs'] += 1

    def _prefer_helper(cell_name: str, req_inputs: int, req_total: int) -> bool:
        name = str(cell_name).upper()
        if req_inputs >= 3 or req_total >= 4:
            return True
        semantic_families = ('MUX', 'AOI', 'OAI', 'FA', 'HA', 'XOR', 'XNOR')
        return any(tok in name for tok in semantic_families)

    def _gate_priority(gate_idx: int):
        cell_name = label_to_cell.get(int(x_list[gate_idx]), f'LABEL_{int(x_list[gate_idx])}')
        spec = pin_count_spec(cell_name, cell_pin_specs) or infer_cell_spec(cell_name)
        req_inputs = int(spec['inputs'])
        req_outputs = int(spec['outputs'])
        req_total = req_inputs + req_outputs
        slack = max(0, len(neighbors[gate_idx]) - req_total)
        return (slack, len(neighbors[gate_idx]), -req_outputs, -req_inputs, gate_idx)

    for gate_idx in sorted(gate_nodes, key=_gate_priority):
        cell_name = label_to_cell.get(int(x_list[gate_idx]), f'LABEL_{int(x_list[gate_idx])}')
        spec = pin_count_spec(cell_name, cell_pin_specs)
        if spec is None:
            spec = infer_cell_spec(cell_name)
            if cell_pin_specs is not None:
                summary['missing_liberty_cell_specs'] += 1
        req_inputs = int(spec['inputs'])
        req_outputs = int(spec['outputs'])
        req_total = req_inputs + req_outputs
        if req_total <= 0:
            continue
        summary['gates_processed'] += 1
        changed = False

        owned_outputs = _choose_existing_outputs(gate_idx, req_outputs=req_outputs, req_inputs=req_inputs)
        for net in owned_outputs:
            output_owner[net] = gate_idx

        output_deficit = max(0, req_outputs - len(owned_outputs))
        summary['would_add_output_nets'] += int(output_deficit)
        if mode == 'repair':
            for _ in range(output_deficit):
                _add_private_net(gate_idx, reserve_output=True)
                summary['added_output_nets'] += 1
                changed = True

        total_excess = max(0, len(neighbors[gate_idx]) - req_total)
        summary['would_remove_excess_edges'] += int(total_excess)
        if total_excess > 0:
            removable_nets = [
                net for net in neighbors[gate_idx]
                if _is_net(net) and output_owner.get(net) != gate_idx and len(neighbors[net]) > 1
            ]
            removable_nets.sort(key=lambda net: (-len(neighbors[net]), net))
            if mode in {'conservative', 'repair'}:
                for net in removable_nets[:total_excess]:
                    neighbors[gate_idx].discard(net)
                    neighbors[net].discard(gate_idx)
                    summary['removed_excess_edges'] += 1
                    changed = True

        input_deficit = max(0, req_inputs - len(_input_nets(gate_idx)))
        summary['would_insert_helper_gates'] += int(input_deficit if (input_deficit > 0 and helper_label is not None and _prefer_helper(cell_name, req_inputs, req_total)) else 0)
        helper_budget = 0
        if mode == 'repair' and input_deficit > 0 and helper_label is not None and _prefer_helper(cell_name, req_inputs, req_total):
            local_sources = [
                net for net in _input_nets(gate_idx)
                if len(neighbors[net]) >= 2
            ]
            local_sources.sort(key=lambda net: (len(neighbors[net]), net))
            if local_sources:
                helper_budget = input_deficit
                for idx in range(helper_budget):
                    _insert_helper_stage(local_sources[idx % len(local_sources)], gate_idx)
                    changed = True

        remaining_input_deficit = max(0, req_inputs - len(_input_nets(gate_idx)))
        if remaining_input_deficit > 0 and len(_input_nets(gate_idx)) > 0:
            summary['would_reuse_input_nets'] += int(remaining_input_deficit)
        counted_private_input_net_deficit = 0
        if remaining_input_deficit > 0:
            candidate_nets = [
                net for net in net_nodes
                if net not in neighbors[gate_idx]
                and output_owner.get(net) is None
                and len(neighbors[net]) <= 2
            ]
            candidate_nets.sort(key=lambda net: (len(neighbors[net]), net))
            used_existing = 0
            summary['would_add_input_edges'] += int(min(remaining_input_deficit, len(candidate_nets)))
            if mode in {'conservative', 'repair'}:
                for net in candidate_nets[:remaining_input_deficit]:
                    neighbors[gate_idx].add(net)
                    neighbors[net].add(gate_idx)
                    used_existing += 1
                    summary['added_input_edges'] += 1
                    changed = True
            remaining_input_deficit -= used_existing

            if mode == 'repair' and remaining_input_deficit > 0 and helper_label is not None:
                global_sources = [
                    net for net in sorted(net_nodes)
                    if output_owner.get(net) is None and len(neighbors[net]) >= 2
                ]
                if global_sources:
                    helper_repairs = min(remaining_input_deficit, len(global_sources))
                    for idx in range(helper_repairs):
                        _insert_helper_stage(global_sources[idx % len(global_sources)], gate_idx)
                        changed = True
                    remaining_input_deficit -= helper_repairs

            summary['would_add_input_nets'] += int(max(0, remaining_input_deficit))
            counted_private_input_net_deficit += int(max(0, remaining_input_deficit))
            for _ in range(max(0, remaining_input_deficit)):
                if mode == 'repair':
                    _add_private_net(gate_idx, reserve_output=False)
                    summary['added_input_nets'] += 1
                    changed = True

        total_deficit = max(0, req_total - len(neighbors[gate_idx]))
        residual_private_input_deficit = max(0, int(total_deficit) - int(output_deficit) - int(counted_private_input_net_deficit))
        summary['would_add_input_nets'] += int(residual_private_input_deficit)
        for _ in range(residual_private_input_deficit):
            if mode == 'repair':
                _add_private_net(gate_idx, reserve_output=False)
                summary['added_input_nets'] += 1
                changed = True

        if changed:
            summary['gates_changed'] += 1

    edges = []
    for src, nbrs in enumerate(neighbors):
        for dst in sorted(nbrs):
            edges.append((src, dst))
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=len(x_list))
        edge_index = coalesce(edge_index, num_nodes=len(x_list))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)
    x = torch.tensor(x_list, dtype=torch.long)
    edge_index, edge_attr, mask = remove_isolated_nodes(edge_index, edge_attr=edge_attr, num_nodes=int(x.numel()))
    x = x[mask]
    if mode == 'audit':
        return data, summary

    projected = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=int(x.numel()))
    if hasattr(data, 'pin_id_to_name'):
        try:
            projected.pin_id_to_name = list(getattr(data, 'pin_id_to_name'))
        except Exception:
            pass
    try:
        if mode != 'audit' and hasattr(data, 'edge_role'):
            pin_table = CellPinSpecTable(allow_fallback=True)
            pin_table.pre_register_cells(label_to_cell.values())
            projected, _ = annotate_gate_net_edge_roles(
                projected,
                label_to_cell,
                pin_table,
                net_id=int(net_id),
                boundary_stub_id=-1,
                preserve_existing=False,
                reserve_output_when_degree_short=True,
            )
    except Exception:
        pass
    return projected, summary


def _graph_cell_counts(graph: Data, label_to_cell: Dict[int, str] | None, net_id: int) -> Counter:
    counts: Counter = Counter()
    if label_to_cell is None or getattr(graph, 'x', None) is None:
        return counts
    for label in graph.x.reshape(-1).to(torch.long).tolist():
        label = int(label)
        if label == int(net_id):
            continue
        cell = label_to_cell.get(label)
        if cell is not None:
            counts[str(cell)] += 1
    return counts


def _clone_graph_data(graph: Data) -> Data:
    out = Data(
        x=graph.x.clone(),
        edge_index=graph.edge_index.clone(),
        edge_attr=(None if getattr(graph, 'edge_attr', None) is None else graph.edge_attr.clone()),
        num_nodes=int(graph.num_nodes),
    )
    if hasattr(graph, 'edge_role'):
        try:
            out.edge_role = graph.edge_role.clone()
        except Exception:
            pass
    if hasattr(graph, 'edge_pin_id'):
        try:
            out.edge_pin_id = graph.edge_pin_id.clone()
        except Exception:
            pass
    if hasattr(graph, 'pin_id_to_name'):
        try:
            out.pin_id_to_name = list(getattr(graph, 'pin_id_to_name'))
        except Exception:
            pass
    return out


def _assignment_cost_terms(
    *,
    node_idx: int,
    demand_class: int,
    record: Dict,
    meta: Dict,
    required_stub_count: int,
    ref_probs: Dict[str, float],
    medium_types: set,
) -> Dict[str, float]:
    metrics = record['metrics']
    interface_profile = record.get('interface_profile', {})
    cell_counts = record.get('cell_counts', Counter())
    dist = float(interface_bucket_distance(int(demand_class), int(metrics['interface_class']), meta))
    supply = int(metrics.get('boundary_stub_count', 0))
    largest_stub_supply = int(metrics.get('boundary_stub_on_largest_component', 0.0))
    driver_stub_supply = int(interface_profile.get('driver_stub_supply', 0))
    load_stub_supply = int(interface_profile.get('load_stub_supply', 0))
    total_cells = float(sum(cell_counts.values()))
    weighted_gain = sum(ref_probs.get(cell, 0.0) for cell in cell_counts)
    medium_gain = sum(ref_probs.get(cell, 0.0) for cell in cell_counts if cell in medium_types)
    x1_ratio = (sum(count for cell, count in cell_counts.items() if cell.endswith('_X1')) / total_cells) if total_cells > 0 else 1.0
    driver_target = 1 if required_stub_count > 0 else 0
    load_target = max(1, min(required_stub_count, max(1, required_stub_count // 2))) if required_stub_count > 0 else 0
    driver_shortfall = max(0, driver_target - driver_stub_supply)
    load_shortfall = max(0, load_target - load_stub_supply)
    role_imbalance = abs(float(interface_profile.get('driver_stub_ratio', 0.0)) - float(interface_profile.get('load_stub_ratio', 0.0)))
    return {
        'skeleton_node': float(node_idx),
        'interface_distance': float(dist),
        'stub_shortfall': float(max(0, required_stub_count - supply)),
        'largest_stub_shortfall': float(max(0, required_stub_count - largest_stub_supply)),
        'stub_excess': float(max(0, supply - required_stub_count)),
        'largest_stub_excess': float(max(0, largest_stub_supply - required_stub_count)),
        'pin_deficit': float(metrics.get('gate_pin_deficit', 0.0)),
        'underconnected': float(metrics.get('underconnected_gate_fraction', 0.0)),
        'synthetic_proxy': float(metrics.get('synthetic_input_proxy_ratio', 0.0)),
        'non_stub_components': float(metrics.get('non_stub_component_count', 0.0)),
        'non_stub_lcc': float(metrics.get('non_stub_largest_component_ratio', 0.0)),
        'weighted_type_gain': float(weighted_gain),
        'medium_type_gain': float(medium_gain),
        'used_type_gain': float(len(cell_counts)),
        'x1_excess': float(x1_ratio),
        'estimated_input_deficit': float(interface_profile.get('estimated_input_deficit', 0.0)),
        'estimated_output_deficit': float(interface_profile.get('estimated_output_deficit', 0.0)),
        'estimated_reuse_risk': float(interface_profile.get('estimated_reuse_risk', 0.0)),
        'estimated_synthetic_input_risk': float(interface_profile.get('estimated_synthetic_input_risk', 0.0)),
        'estimated_repaired_output_risk': float(interface_profile.get('estimated_repaired_output_risk', 0.0)),
        'driver_role_shortfall': float(driver_shortfall),
        'load_role_shortfall': float(load_shortfall),
        'role_imbalance': float(role_imbalance),
        'contract_driver_supply_bucket': float(interface_profile.get('contract_driver_supply_bucket') or 0.0),
        'contract_load_supply_bucket': float(interface_profile.get('contract_load_supply_bucket') or 0.0),
        'contract_bidir_supply_bucket': float(interface_profile.get('contract_bidir_supply_bucket') or 0.0),
        'contract_role_entropy_bucket': float(interface_profile.get('contract_role_entropy_bucket') or 0.0),
        'contract_projection_repair_bucket': float(interface_profile.get('contract_projection_repair_bucket') or 0.0),
        'contract_net_export_cleanliness_bucket': float(interface_profile.get('contract_net_export_cleanliness_bucket') or 0.0),
        'contract_available': float(interface_profile.get('contract_available', 0.0) or 0.0),
    }


def _hard_infeasible_candidate(
    *,
    record: Dict,
    required_stub_count: int,
) -> bool:
    metrics = record.get('metrics', {}) or {}
    profile = record.get('interface_profile', {}) or {}
    boundary_stub_count = int(metrics.get('boundary_stub_count', 0))
    largest_boundary_stub_count = int(metrics.get('boundary_stub_on_largest_component', 0.0))
    driver_supply = int(profile.get('driver_stub_supply', 0))
    load_supply = int(profile.get('load_stub_supply', 0))
    if boundary_stub_count <= 0:
        return True
    if required_stub_count > 0 and largest_boundary_stub_count <= 0:
        return True
    if required_stub_count >= 2 and (driver_supply <= 0 or load_supply <= 0):
        return True
    if float(profile.get('estimated_input_deficit', 0.0)) > 2000.0:
        return True
    if float(profile.get('estimated_synthetic_input_risk', 0.0)) > 2500.0:
        return True
    if int(profile.get('contract_available', 0) or 0):
        projection_bucket = profile.get('contract_projection_repair_bucket')
        cleanliness_bucket = profile.get('contract_net_export_cleanliness_bucket')
        if projection_bucket is not None and cleanliness_bucket is not None:
            if int(projection_bucket) >= 3 and int(cleanliness_bucket) <= 1:
                return True
    return False


def _weighted_assignment_cost(terms: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    weights = dict(DEFAULT_ASSIGNMENT_COST_WEIGHTS if weights is None else weights)
    return float(sum(float(weights.get(key, 0.0)) * float(value) for key, value in terms.items()))


def _assignment_global_budget_summary(
    pairs: List[Tuple[int, int]],
    normalized_pool: List[Dict],
    terms_matrix: List[List[Dict[str, float]]],
    reference_profile: Dict | None = None,
) -> Dict[str, float]:
    selected_counts: Counter = Counter()
    totals: Counter = Counter()
    for node_idx, pool_idx in pairs:
        record = normalized_pool[int(pool_idx)]
        selected_counts.update(record.get('cell_counts', Counter()))
        for key, value in terms_matrix[int(node_idx)][int(pool_idx)].items():
            if key == 'skeleton_node':
                continue
            totals[key] += float(value)

    ref_probs = {str(k): float(v) for k, v in dict((reference_profile or {}).get('probabilities', {})).items()}
    medium_types = {str(cell) for cell in list((reference_profile or {}).get('medium_types', []))}
    total_cells = float(sum(selected_counts.values()))
    weighted_cov = sum(ref_probs.get(cell, 0.0) for cell in selected_counts if cell in ref_probs)
    medium_cov = (sum(1.0 for cell in medium_types if cell in selected_counts) / float(len(medium_types))) if medium_types else 0.0
    x1_ratio = (sum(count for cell, count in selected_counts.items() if str(cell).endswith('_X1')) / total_cells) if total_cells > 0 else 1.0
    ref_x1 = float((reference_profile or {}).get('x1_ratio', 0.0))
    return {
        'selected_partition_count': float(len(pairs)),
        'selected_gate_count': float(total_cells),
        'used_type_count': float(len(selected_counts)),
        'weighted_type_coverage_gap': float(1.0 - weighted_cov),
        'medium_type_coverage_gap': float(1.0 - medium_cov),
        'x1_excess_ratio': float(max(0.0, x1_ratio - ref_x1)),
        'stub_shortfall_total': float(totals.get('stub_shortfall', 0.0)),
        'largest_stub_shortfall_total': float(totals.get('largest_stub_shortfall', 0.0)),
        'pin_deficit_total': float(totals.get('pin_deficit', 0.0)),
        'underconnected_total': float(totals.get('underconnected', 0.0)),
        'synthetic_proxy_total': float(totals.get('synthetic_proxy', 0.0)),
        'estimated_input_deficit_total': float(totals.get('estimated_input_deficit', 0.0)),
        'estimated_output_deficit_total': float(totals.get('estimated_output_deficit', 0.0)),
        'estimated_reuse_risk_total': float(totals.get('estimated_reuse_risk', 0.0)),
        'estimated_synthetic_input_risk_total': float(totals.get('estimated_synthetic_input_risk', 0.0)),
        'estimated_repaired_output_risk_total': float(totals.get('estimated_repaired_output_risk', 0.0)),
        'driver_role_shortfall_total': float(totals.get('driver_role_shortfall', 0.0)),
        'load_role_shortfall_total': float(totals.get('load_role_shortfall', 0.0)),
        'role_imbalance_total': float(totals.get('role_imbalance', 0.0)),
        'contract_available_total': float(totals.get('contract_available', 0.0)),
        'contract_projection_repair_bucket_total': float(totals.get('contract_projection_repair_bucket', 0.0)),
        'contract_net_export_cleanliness_bucket_total': float(totals.get('contract_net_export_cleanliness_bucket', 0.0)),
        'contract_driver_supply_bucket_total': float(totals.get('contract_driver_supply_bucket', 0.0)),
        'contract_load_supply_bucket_total': float(totals.get('contract_load_supply_bucket', 0.0)),
        'contract_bidir_supply_bucket_total': float(totals.get('contract_bidir_supply_bucket', 0.0)),
        'contract_role_entropy_bucket_total': float(totals.get('contract_role_entropy_bucket', 0.0)),
        'non_stub_components_total': float(totals.get('non_stub_components', 0.0)),
        'non_stub_lcc_total': float(totals.get('non_stub_lcc', 0.0)),
    }


def _global_budget_penalty(summary: Dict[str, float], node_count: int) -> float:
    n = max(1.0, float(node_count))
    return float(
        900.0 * summary.get('stub_shortfall_total', 0.0)
        + 1100.0 * summary.get('largest_stub_shortfall_total', 0.0)
        + 180.0 * summary.get('non_stub_components_total', 0.0)
        - 180.0 * summary.get('non_stub_lcc_total', 0.0)
        + 2200.0 * summary.get('weighted_type_coverage_gap', 1.0)
        + 1200.0 * summary.get('medium_type_coverage_gap', 1.0)
        + 250.0 * summary.get('x1_excess_ratio', 0.0)
        + 2.0 * summary.get('estimated_input_deficit_total', 0.0) / n
        + 4.0 * summary.get('estimated_output_deficit_total', 0.0) / n
        + 2.0 * summary.get('estimated_reuse_risk_total', 0.0) / n
        + 3.0 * summary.get('estimated_synthetic_input_risk_total', 0.0) / n
        + 4.0 * summary.get('estimated_repaired_output_risk_total', 0.0) / n
        + 180.0 * summary.get('driver_role_shortfall_total', 0.0)
        + 260.0 * summary.get('load_role_shortfall_total', 0.0)
        + 35.0 * summary.get('role_imbalance_total', 0.0)
        + 120.0 * summary.get('contract_projection_repair_bucket_total', 0.0) / n
        - 80.0 * summary.get('contract_net_export_cleanliness_bucket_total', 0.0) / n
        - 20.0 * summary.get('contract_load_supply_bucket_total', 0.0) / n
        - 12.0 * summary.get('contract_driver_supply_bucket_total', 0.0) / n
    )


def _assign_global_budget_flow(
    cost_matrix: torch.Tensor,
    terms_matrix: List[List[Dict[str, float]]],
    normalized_pool: List[Dict],
    reference_profile: Dict | None = None,
    *,
    beam_width: int = 128,
    candidates_per_node: int = 24,
) -> Tuple[Dict[int, int], Dict[str, float]]:
    rows, cols = int(cost_matrix.size(0)), int(cost_matrix.size(1))
    if cols < rows:
        raise RuntimeError(f'partition pool too small for global budget assignment: need {rows}, got {cols}')

    def _score_pairs(pairs: List[Tuple[int, int]]) -> Tuple[float, Dict[str, float]]:
        base_cost = sum(float(cost_matrix[int(row), int(col)].item()) for row, col in pairs)
        summary = _assignment_global_budget_summary(
            pairs,
            normalized_pool,
            terms_matrix,
            reference_profile=reference_profile,
        )
        score = base_cost + _global_budget_penalty(summary, node_count=len(pairs))
        return float(score), summary

    # Start from a true global one-to-one solution.  The previous beam-search
    # implementation expanded O(rows * beam_width * rows) candidates and
    # recomputed global summaries for each expansion; for 200+ skeleton nodes
    # it could run for hours.  Linear assignment gives a fast feasible base,
    # then bounded local repair improves the global budget objective.
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    if len(row_ind) != rows:
        raise RuntimeError(f'global budget assignment did not cover all skeleton nodes: covered {len(row_ind)}, need {rows}')
    best_pairs = [(int(row), int(col)) for row, col in zip(row_ind.tolist(), col_ind.tolist())]
    best_pairs.sort(key=lambda item: item[0])
    initial_score, initial_summary = _score_pairs(best_pairs)

    by_row = {int(row): int(col) for row, col in best_pairs}
    used_cols = {int(col) for _, col in best_pairs}
    infeasible_selected = [
        int(row)
        for row, col in best_pairs
        if float(cost_matrix[int(row), int(col)].item()) >= 1.0e11
    ]

    # Use a bounded local-search pass over unused candidates.  This keeps the
    # assignment budget-aware without letting smoke runs explode.  The
    # beam_width argument is retained for backward compatibility and now caps
    # the number of accepted local repairs rather than controlling an
    # exponential beam.
    candidate_k = min(cols, max(8, int(candidates_per_node)))
    candidate_lists: List[List[int]] = []
    for row_idx in range(rows):
        order = torch.argsort(cost_matrix[row_idx])[:candidate_k].tolist()
        candidate_lists.append([int(idx) for idx in order])

    current_score = float(initial_score)
    current_summary = dict(initial_summary)
    accepted_swaps = 0
    evaluated_swaps = 0
    max_accepted_swaps = max(0, int(beam_width))
    max_passes = 2
    for _ in range(max_passes):
        improved = False
        row_order = sorted(range(rows), key=lambda r: float(cost_matrix[r, by_row[r]].item()), reverse=True)
        for row_idx in row_order:
            old_col = by_row[int(row_idx)]
            best_local_score = current_score
            best_local_col = None
            best_local_summary = None
            for new_col in candidate_lists[int(row_idx)]:
                if new_col == old_col or new_col in used_cols:
                    continue
                trial_pairs = [(row, (new_col if row == row_idx else col)) for row, col in best_pairs]
                trial_score, trial_summary = _score_pairs(trial_pairs)
                evaluated_swaps += 1
                if trial_score + 1.0e-9 < best_local_score:
                    best_local_score = float(trial_score)
                    best_local_col = int(new_col)
                    best_local_summary = trial_summary
            if best_local_col is None:
                continue
            used_cols.remove(old_col)
            used_cols.add(best_local_col)
            by_row[int(row_idx)] = best_local_col
            best_pairs = [(row, (best_local_col if row == row_idx else col)) for row, col in best_pairs]
            current_score = float(best_local_score)
            current_summary = dict(best_local_summary or {})
            accepted_swaps += 1
            improved = True
            if max_accepted_swaps > 0 and accepted_swaps >= max_accepted_swaps:
                break
        if not improved or (max_accepted_swaps > 0 and accepted_swaps >= max_accepted_swaps):
            break

    summary = _assignment_global_budget_summary(
        best_pairs,
        normalized_pool,
        terms_matrix,
        reference_profile=reference_profile,
    )
    final_score = float(current_score)
    summary['global_budget_score'] = float(final_score)
    summary['global_budget_initial_score'] = float(initial_score)
    summary['global_budget_score_delta'] = float(final_score - float(initial_score))
    summary['global_budget_algorithm'] = 'linear_sum_assignment_local_budget_repair'
    summary['global_budget_candidate_k'] = float(candidate_k)
    summary['global_budget_evaluated_swaps'] = float(evaluated_swaps)
    summary['global_budget_accepted_swaps'] = float(accepted_swaps)
    summary['global_budget_infeasible_selected_count'] = float(len(infeasible_selected))
    summary['global_budget_infeasible_selected_rows'] = [int(row) for row in infeasible_selected[:64]]
    summary['fallback_full_candidate_rows'] = []
    summary['fallback_full_candidate_count'] = 0.0
    return {int(row): int(col) for row, col in best_pairs}, summary


def _build_assignment_cost_matrix(
    skeleton: Data,
    normalized_pool: List[Dict],
    meta: Dict,
    reference_profile: Dict | None = None,
    mode: str = 'conservative',
    weights: Dict[str, float] | None = None,
) -> Tuple[torch.Tensor, List[List[Dict[str, float]]]]:
    demand_classes = skeleton.x.reshape(-1).to(torch.long).tolist()
    required_stubs = skeleton_stub_requirements(skeleton, meta, mode=mode)
    medium_types = {str(cell) for cell in list((reference_profile or {}).get('medium_types', []))}
    ref_probs = {str(k): float(v) for k, v in dict((reference_profile or {}).get('probabilities', {})).items()}
    rows = len(demand_classes)
    cols = len(normalized_pool)
    cost = torch.empty((rows, cols), dtype=torch.float64)
    terms_matrix: List[List[Dict[str, float]]] = []
    for node_idx, demand_class in enumerate(demand_classes):
        row_terms = []
        required_stub_count = int(required_stubs.get(int(node_idx), 0))
        for record in normalized_pool:
            terms = _assignment_cost_terms(
                node_idx=int(node_idx),
                demand_class=int(demand_class),
                record=record,
                meta=meta,
                required_stub_count=required_stub_count,
                ref_probs=ref_probs,
                medium_types=medium_types,
            )
            row_terms.append(terms)
        terms_matrix.append(row_terms)
        for pool_idx, terms in enumerate(row_terms):
            record = normalized_pool[pool_idx]
            if _hard_infeasible_candidate(record=record, required_stub_count=required_stub_count):
                cost[node_idx, pool_idx] = 1.0e12
            else:
                cost[node_idx, pool_idx] = _weighted_assignment_cost(terms, weights=weights)
    return cost, terms_matrix


def assign_partitions_to_skeleton(
    skeleton: Data,
    partition_pool: Iterable[Data],
    meta: Dict,
    label_to_cell: Dict[int, str] | None = None,
    reference_profile: Dict | None = None,
    mode: str = 'conservative',
    assignment_mode: str = 'greedy',
    assignment_cost_weights: Dict[str, float] | None = None,
    cell_pin_specs: Dict[str, Dict[str, List[str]]] | None = None,
) -> List[Dict]:
    if assignment_mode not in {'greedy', 'global_linear_sum', 'global_budget_flow'}:
        raise ValueError(f'unknown assignment_mode: {assignment_mode}')
    boundary_stub_id = int(meta['boundary_stub_id'])
    normalized_pool = []
    net_id = int(meta['net_id'])
    medium_types = {str(cell) for cell in list((reference_profile or {}).get('medium_types', []))}
    ref_probs = {str(k): float(v) for k, v in dict((reference_profile or {}).get('probabilities', {})).items()}
    for graph in partition_pool:
        g = normalize_generated_partition_graph(
            graph,
            net_id=net_id,
            boundary_stub_id=boundary_stub_id,
        )
        metrics = classify_partition_graph(g, meta, label_to_cell=label_to_cell)
        metrics.update(_boundary_stub_connectivity(g, boundary_stub_id))
        cell_counts = _graph_cell_counts(g, label_to_cell, net_id)
        interface_profile = compute_partition_interface_profile(
            g,
            meta,
            label_to_cell=label_to_cell,
            cell_pin_specs=cell_pin_specs,
        )
        normalized_pool.append({'graph': g, 'metrics': metrics, 'cell_counts': cell_counts, 'interface_profile': interface_profile})

    if assignment_mode in {'global_linear_sum', 'global_budget_flow'}:
        demand_classes = skeleton.x.reshape(-1).to(torch.long).tolist()
        if len(normalized_pool) < len(demand_classes):
            raise RuntimeError(f'partition pool too small for global assignment: need {len(demand_classes)}, got {len(normalized_pool)}')
        cost_matrix, terms_matrix = _build_assignment_cost_matrix(
            skeleton,
            normalized_pool,
            meta,
            reference_profile=reference_profile,
            mode=mode,
            weights=assignment_cost_weights,
        )
        global_budget_summary = {}
        if assignment_mode == 'global_budget_flow':
            by_row, global_budget_summary = _assign_global_budget_flow(
                cost_matrix,
                terms_matrix,
                normalized_pool,
                reference_profile=reference_profile,
            )
        else:
            row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
            by_row = {int(row): int(col) for row, col in zip(row_ind.tolist(), col_ind.tolist())}
        required_stubs = skeleton_stub_requirements(skeleton, meta, mode=mode)
        assignments: List[Dict] = []
        for node_idx, demand_class in enumerate(demand_classes):
            pool_idx = by_row.get(int(node_idx))
            if pool_idx is None:
                raise RuntimeError(f'global assignment did not cover skeleton node {node_idx}')
            record = normalized_pool[pool_idx]
            demand_buckets = decode_interface_class_meta(int(demand_class), meta)
            terms = terms_matrix[node_idx][pool_idx]
            assignments.append(
                {
                    'skeleton_node': int(node_idx),
                    'demand_class': int(demand_class),
                    'supply_class': int(record['metrics']['interface_class']),
                    'exact_match': bool(record['metrics']['interface_class'] == int(demand_class)),
                    'pool_index': int(pool_idx),
                    'required_stub_count': int(required_stubs.get(int(node_idx), 0)),
                    'demand_buckets': demand_buckets,
                    'graph': record['graph'],
                    'metrics': record['metrics'],
                    'interface_profile': record.get('interface_profile', {}),
                    'assignment_cost': float(cost_matrix[node_idx, pool_idx].item()),
                    'assignment_cost_terms': terms,
                    'assignment_global_budget_summary': global_budget_summary,
                    'assignment_mode': str(assignment_mode),
                }
            )
        return assignments

    used = set()
    assignments: List[Dict] = []
    selected_cell_counts: Counter = Counter()
    demand_classes = skeleton.x.reshape(-1).to(torch.long).tolist()
    required_stubs = skeleton_stub_requirements(skeleton, meta, mode=mode)
    for node_idx, demand_class in enumerate(demand_classes):
        best = None
        best_key = None
        demand_buckets = decode_interface_class_meta(int(demand_class), meta)
        required_stub_count = int(required_stubs.get(int(node_idx), 0))
        current_medium_missing = {cell for cell in medium_types if selected_cell_counts.get(cell, 0) <= 0}
        for pool_idx, record in enumerate(normalized_pool):
            if pool_idx in used:
                continue
            dist = interface_bucket_distance(int(demand_class), int(record['metrics']['interface_class']), meta)
            supply = int(record['metrics']['boundary_stub_count'])
            largest_stub_supply = int(record['metrics'].get('boundary_stub_on_largest_component', 0.0))
            stub_shortfall = max(0, required_stub_count - supply)
            largest_stub_shortfall = max(0, required_stub_count - largest_stub_supply)
            stub_excess = max(0, supply - required_stub_count)
            largest_stub_excess = max(0, largest_stub_supply - required_stub_count)
            pin_deficit = float(record['metrics'].get('gate_pin_deficit', 0.0))
            underconnected = float(record['metrics'].get('underconnected_gate_fraction', 0.0))
            synthetic_proxy = float(record['metrics'].get('synthetic_input_proxy_ratio', 0.0))
            non_stub_lcc = float(record['metrics'].get('non_stub_largest_component_ratio', 0.0))
            non_stub_components = float(record['metrics'].get('non_stub_component_count', 0.0))
            cell_counts = record.get('cell_counts', Counter())
            missing_medium_gain = sum(ref_probs.get(cell, 0.0) for cell in cell_counts if cell in current_medium_missing)
            weighted_gain = sum(ref_probs.get(cell, 0.0) for cell in cell_counts if selected_cell_counts.get(cell, 0) <= 0)
            total_cells = float(sum(cell_counts.values()))
            x1_ratio = (sum(count for cell, count in cell_counts.items() if cell.endswith('_X1')) / total_cells) if total_cells > 0 else 1.0
            key = (
                stub_shortfall,
                largest_stub_shortfall,
                stub_excess,
                largest_stub_excess,
                dist,
                pin_deficit,
                underconnected,
                synthetic_proxy,
                non_stub_components,
                -non_stub_lcc,
                -missing_medium_gain,
                -weighted_gain,
                x1_ratio,
                -len(cell_counts),
                abs(supply - required_stub_count),
                abs(largest_stub_supply - required_stub_count),
                pool_idx,
            )
            if best_key is None or key < best_key:
                best_key = key
                best = (pool_idx, record)
        if best is None:
            raise RuntimeError(f'no available partition candidate for skeleton node {node_idx}')
        pool_idx, record = best
        used.add(pool_idx)
        selected_cell_counts.update(record.get('cell_counts', Counter()))
        assignments.append(
            {
                'skeleton_node': int(node_idx),
                'demand_class': int(demand_class),
                'supply_class': int(record['metrics']['interface_class']),
                'exact_match': bool(record['metrics']['interface_class'] == int(demand_class)),
                'pool_index': int(pool_idx),
                'required_stub_count': int(required_stub_count),
                'demand_buckets': demand_buckets,
                'graph': record['graph'],
                'metrics': record['metrics'],
                'interface_profile': record.get('interface_profile', {}),
                'assignment_cost': float(best_key[0]) if best_key is not None else 0.0,
                'assignment_cost_terms': {},
                'assignment_mode': str(assignment_mode),
            }
        )
    return assignments


def assemble_generated_design(
    skeleton: Data,
    assignments: List[Dict],
    meta: Dict,
    mode: str = 'conservative',
    label_to_cell: Dict[int, str] | None = None,
    semantic_projection: bool = False,
    semantic_projection_mode: str = 'repair',
    cell_pin_specs: Dict[str, Dict[str, List[str]]] | None = None,
    return_projection_artifacts: bool = False,
    interface_aware_stub_ordering: bool = False,
    enable_cut_group_expansion: bool = False,
    cut_group_source: str = 'typed_shared_net_group_count',
    cut_group_pair_counts: Dict[Tuple[int, int], int] | None = None,
    max_cut_groups_per_pair: int = 1,
    min_cut_groups_per_pair: int = 1,
    cut_group_expansion_mode: str = 'smoke',
) -> Tuple[Data, Dict] | Tuple[Data, Dict, Dict]:
    net_id = int(meta['net_id'])
    boundary_stub_id = int(meta['boundary_stub_id'])

    global_x: List[int] = []
    global_edges: List[Tuple[int, int]] = []
    global_edge_role: List[int] = []
    global_edge_pin_id: List[int] = []
    stub_inventory: Dict[int, List[Tuple[int, List[int], int, Dict[int, Tuple[int, int]]]]] = defaultdict(list)
    partition_manifests: List[Dict] = []
    assignments_by_node = {int(item['skeleton_node']): item for item in assignments}

    for assignment in assignments:
        graph = assignment['graph']
        _, stub_neighbors, stub_neighbor_meta = _non_stub_copy(
            graph,
            boundary_stub_id,
            global_x,
            global_edges,
            global_edge_role=global_edge_role,
            global_edge_pin_id=global_edge_pin_id,
        )
        if interface_aware_stub_ordering:
            ordered_stubs = _ordered_boundary_stubs_with_profile(
                graph,
                boundary_stub_id,
                label_to_cell=label_to_cell,
                cell_pin_specs=cell_pin_specs,
            )
        else:
            ordered_stubs = _available_boundary_stubs(graph, boundary_stub_id)
        for stub_idx in ordered_stubs:
            role = _stub_role_for_node(
                graph,
                boundary_stub_id,
                int(stub_idx),
                net_id=net_id,
                label_to_cell=label_to_cell,
            )
            stub_inventory[assignment['skeleton_node']].append(
                (
                    int(stub_idx),
                    list(stub_neighbors.get(int(stub_idx), [])),
                    int(role),
                    dict(stub_neighbor_meta.get(int(stub_idx), {})),
                )
            )
        partition_manifests.append(
            {
                'skeleton_node': assignment['skeleton_node'],
                'demand_class': assignment['demand_class'],
                'supply_class': assignment['supply_class'],
                'exact_match': assignment['exact_match'],
                'required_stub_count': int(assignment.get('required_stub_count', 0)),
                'demand_buckets': assignment.get('demand_buckets', {}),
                'metrics': assignment['metrics'],
                'interface_profile': assignment.get('interface_profile', {}),
                'assignment_cost': float(assignment.get('assignment_cost', 0.0)),
                'assignment_cost_terms': assignment.get('assignment_cost_terms', {}),
                'assignment_global_budget_summary': assignment.get('assignment_global_budget_summary', {}),
                'assignment_mode': assignment.get('assignment_mode', 'greedy'),
            }
        )

    unique_edges = _canonical_skeleton_edge_records(skeleton)
    connection_plan, remaining_stub_budget, dropped_edges = _plan_connection_counts(
        unique_edges,
        stub_inventory,
        meta,
        mode=mode,
        enable_cut_group_expansion=bool(enable_cut_group_expansion),
        cut_group_source=str(cut_group_source),
        cut_group_pair_counts=cut_group_pair_counts,
        max_cut_groups_per_pair=int(max_cut_groups_per_pair),
        min_cut_groups_per_pair=int(min_cut_groups_per_pair),
        cut_group_expansion_mode=str(cut_group_expansion_mode),
    )
    global_net_plan, orientation_summary = _build_global_net_plan(connection_plan, assignments_by_node)
    role_slot_budget = _summarize_role_slot_budget(global_net_plan, assignments_by_node)

    role_preserved_boundary_edges = 0
    role_lost_boundary_edges = 0

    connection_records = []
    for net_plan in global_net_plan:
        driver_partition = int(net_plan.get('driver_partition', -1))
        load_partitions = [int(node) for node in net_plan.get('load_partitions', [])]
        if driver_partition < 0 or not load_partitions:
            continue
        driver_stub_idx, driver_neighbors, driver_role, driver_meta, driver_role_matched = _pop_preferred_stub(
            stub_inventory[driver_partition],
            prefer_driver=True,
        )
        load_records = []
        all_neighbors = list(driver_neighbors)
        for load_partition in load_partitions:
            load_stub_idx, load_neighbors, load_role, load_meta, load_role_matched = _pop_preferred_stub(
                stub_inventory[load_partition],
                prefer_driver=False,
            )
            load_records.append(
                {
                    'load_partition': int(load_partition),
                    'stub_local_idx': int(load_stub_idx),
                    'stub_role': ROLE_NAMES.get(int(load_role), str(load_role)),
                    'stub_role_matched_preference': bool(load_role_matched),
                    'neighbor_count': int(len(load_neighbors)),
                }
            )
            all_neighbors.extend(load_neighbors)
            for nbr, meta in dict(load_meta).items():
                if nbr not in driver_meta:
                    driver_meta[nbr] = meta
        global_net_idx = len(global_x)
        global_x.append(net_id)
        connected_neighbors = sorted(set(all_neighbors))
        for neighbor in connected_neighbors:
            global_edges.append((global_net_idx, neighbor))
            global_edges.append((neighbor, global_net_idx))
            if int(neighbor) in driver_meta:
                role_preserved_boundary_edges += 1
            else:
                role_lost_boundary_edges += 1
            r, p = driver_meta.get(int(neighbor), (0, -1))
            global_edge_role.append(int(r))
            global_edge_pin_id.append(int(p))
            global_edge_role.append(int(r))
            global_edge_pin_id.append(int(p))
        neighbor_labels = [int(global_x[nbr]) for nbr in connected_neighbors if 0 <= int(nbr) < len(global_x)]
        neighbor_cells = [
            label_to_cell.get(int(label), f'LABEL_{int(label)}') if label_to_cell is not None and int(label) != net_id else ('NET' if int(label) == net_id else f'LABEL_{int(label)}')
            for label in neighbor_labels
        ]
        connection_records.append(
            {
                'planned_net_id': int(net_plan.get('planned_net_id', -1)),
                'planned_driver_partition': int(driver_partition),
                'planned_load_partitions': list(load_partitions),
                'driver_stub_local_idx': int(driver_stub_idx),
                'driver_stub_role': ROLE_NAMES.get(int(driver_role), str(driver_role)),
                'driver_stub_preferred_role': 'driver',
                'driver_stub_role_matched_preference': bool(driver_role_matched),
                'driver_selection_reason': str(net_plan.get('driver_selection_reason', 'unknown')),
                'driver_selection_reason_counts': dict(net_plan.get('driver_selection_reason_counts', {})),
                'edge_class': int(net_plan.get('edge_class', 0)),
                'semantic_bucket': int(net_plan.get('semantic_bucket', 0)),
                'typed_fanout_max': int(net_plan.get('typed_fanout_max', 0)),
                'typed_group_size_max': int(net_plan.get('typed_group_size_max', 0)),
                'typed_shared_net_group_count_max': int(net_plan.get('typed_shared_net_group_count_max', 0)),
                'requested': int(net_plan.get('requested', 0)),
                'group_id': int(net_plan.get('group_id', 0)),
                'group_endpoints': list(net_plan.get('group_endpoints', [])),
                'group_fanout': int(net_plan.get('group_fanout', len(load_partitions))),
                'grouped_edge_count': int(net_plan.get('grouped_edge_count', len(load_partitions))),
                'grouped_edge_instance_ids': list(net_plan.get('grouped_edge_instance_ids', [])),
                'load_stub_records': load_records,
                'connected_neighbors': int(len(connected_neighbors)),
                'connected_neighbor_labels': neighbor_labels,
                'connected_neighbor_cells': neighbor_cells,
            }
        )

    if global_edges:
        edge_index = torch.tensor(global_edges, dtype=torch.long).t().contiguous()
        edge_feat = torch.stack(
            [
                torch.tensor(global_edge_role, dtype=torch.long),
                torch.tensor([int(v) + 1 for v in global_edge_pin_id], dtype=torch.long),
            ],
            dim=1,
        )
        edge_index, edge_feat = to_undirected(edge_index, edge_attr=edge_feat, num_nodes=len(global_x), reduce='max')
        edge_index, edge_feat = coalesce(edge_index, edge_feat, num_nodes=len(global_x), reduce='max')
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_feat = torch.empty((0, 2), dtype=torch.long)
    edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)
    x = torch.tensor(global_x, dtype=torch.long)
    edge_index, edge_feat, mask = remove_isolated_nodes(edge_index, edge_attr=edge_feat, num_nodes=int(x.numel()))
    x = x[mask]
    assembled = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=int(x.numel()))
    assembled.edge_role = edge_feat[:, 0].reshape(-1).to(torch.long)
    assembled.edge_pin_id = (edge_feat[:, 1].reshape(-1).to(torch.long) - 1)
    for assignment in assignments:
        if hasattr(assignment.get('graph', None), 'pin_id_to_name'):
            assembled.pin_id_to_name = list(getattr(assignment['graph'], 'pin_id_to_name'))
            break
    pre_projection = _clone_graph_data(assembled)

    pre_projection_edge_role_hist = None
    pre_projection_unknown_role_edges = None
    if hasattr(pre_projection, 'edge_role') and isinstance(pre_projection.edge_role, torch.Tensor):
        roles = pre_projection.edge_role.reshape(-1).to(torch.long)
        hist = torch.bincount(roles.clamp(min=0), minlength=4).tolist()
        pre_projection_edge_role_hist = {
            'unknown': int(hist[0]),
            'gate_input': int(hist[1]),
            'gate_output': int(hist[2]),
            'excess': int(hist[3]),
        }
        pre_projection_unknown_role_edges = int(hist[0])
    projection_summary = None
    if semantic_projection and label_to_cell is not None:
        assembled, projection_summary = _semantic_project_assembled_graph(
            assembled,
            label_to_cell=label_to_cell,
            net_id=net_id,
            mode=semantic_projection_mode,
            cell_pin_specs=cell_pin_specs,
        )
    post_projection = _clone_graph_data(assembled)
    requested_edge_count = int(len(unique_edges))
    realized_edge_count = int(len(connection_plan))
    dropped_edge_count = int(len(dropped_edges))
    requested_cut_group_count = int(
        sum(int(item.get('requested_cut_group_count', item.get('requested', 1))) for item in connection_plan)
        + sum(
            int(item.get('requested_cut_group_count', 0))
            for item in dropped_edges
            if int(item.get('planned_cut_group_count', 0)) <= 0
        )
    )
    realized_cut_group_count = int(sum(int(item.get('planned_cut_group_count', item.get('planned', 0))) for item in connection_plan))
    dropped_cut_group_count = int(max(0, requested_cut_group_count - realized_cut_group_count))
    cut_group_drop_reasons = Counter()
    for item in dropped_edges:
        cut_group_drop_reasons[str(item.get('reason', 'unknown'))] += int(item.get('dropped_cut_group_count', 1))
    manifest = {
        'assembly_mode': mode,
        'semantic_projection': bool(semantic_projection and label_to_cell is not None),
        'semantic_projection_mode': str(semantic_projection_mode),
        'interface_aware_stub_ordering': bool(interface_aware_stub_ordering),
        'semantic_projection_summary': projection_summary,
        'pre_projection_metrics': partition_graph_metrics(
            pre_projection,
            net_id=net_id,
            boundary_stub_id=boundary_stub_id,
        ),
        'partition_assignments': partition_manifests,
        'skeleton_num_nodes': int(skeleton.num_nodes),
        'skeleton_num_edges': int(len(unique_edges)),
        'unique_pair_edge_count': int(len(unique_edges)),
        'requested_skeleton_edges': requested_edge_count,
        'realized_skeleton_edges': realized_edge_count,
        'dropped_skeleton_edges': dropped_edge_count,
        'dropped_skeleton_edge_fraction': float(dropped_edge_count / max(1, requested_edge_count)),
        'cut_group_expansion_enabled': bool(enable_cut_group_expansion),
        'cut_group_expansion_source': str(cut_group_source if enable_cut_group_expansion else 'disabled'),
        'cut_group_expansion_mode': str(cut_group_expansion_mode if enable_cut_group_expansion else 'disabled'),
        'cut_group_expansion_cap': int(max_cut_groups_per_pair if enable_cut_group_expansion else 1),
        'min_cut_groups_per_pair': int(min_cut_groups_per_pair if enable_cut_group_expansion else 1),
        'requested_cut_group_count': int(requested_cut_group_count),
        'realized_cut_group_count': int(realized_cut_group_count),
        'dropped_cut_group_count': int(dropped_cut_group_count),
        'cut_group_realization_ratio': float(realized_cut_group_count / max(1, requested_cut_group_count)),
        'cut_group_dropped_due_to_driver_shortage': int(
            cut_group_drop_reasons.get('driver_stub_shortage', 0)
            + cut_group_drop_reasons.get('driver_and_load_stub_shortage', 0)
        ),
        'cut_group_dropped_due_to_load_shortage': int(
            cut_group_drop_reasons.get('load_stub_shortage', 0)
            + cut_group_drop_reasons.get('driver_and_load_stub_shortage', 0)
        ),
        'realized_connections': connection_records,
        'global_net_plan': global_net_plan,
        'orientation_summary': orientation_summary,
        'role_slot_budget': role_slot_budget,
        'role_preserved_boundary_edges': int(role_preserved_boundary_edges),
        'role_lost_boundary_edges': int(role_lost_boundary_edges),
        'role_loss_rate': float(role_lost_boundary_edges / max(1, role_preserved_boundary_edges + role_lost_boundary_edges)),
        'pre_projection_edge_role_hist': pre_projection_edge_role_hist,
        'pre_projection_role_unknown_edge_count': None if pre_projection_unknown_role_edges is None else int(pre_projection_unknown_role_edges),
        'dropped_edges': dropped_edges,
        'remaining_stub_budget': {int(k): int(v) for k, v in remaining_stub_budget.items()},
        'assembled_metrics': partition_graph_metrics(
            assembled,
            net_id=net_id,
            boundary_stub_id=boundary_stub_id,
        ),
    }
    if not return_projection_artifacts:
        return assembled, manifest
    projection_artifacts = {
        'pre_projection_graph': pre_projection,
        'post_projection_graph': post_projection,
        'teacher_available': bool(semantic_projection and label_to_cell is not None),
        'semantic_projection_enabled': bool(semantic_projection),
        'semantic_projection_mode': str(semantic_projection_mode),
    }
    return assembled, manifest, projection_artifacts


def reconstruct_original_from_manifest(manifest: Dict, meta: Dict) -> Data:
    net_id = int(meta['net_id'])
    source_graph = normalize_circuit_graph(
        torch.load(manifest['source_graph_path'], map_location='cpu', weights_only=False),
        net_id=net_id,
    )
    num_nodes = int(source_graph.num_nodes)
    x = torch.full((num_nodes,), fill_value=net_id, dtype=torch.long)
    edges = set()
    base_dir = Path(meta['dataset_root'])
    partition_cache = {}

    for partition in manifest['partition_records']:
        graph_path = base_dir / partition['graph_rel_path']
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)
        partition_cache[int(partition['partition_id'])] = graph
        local_to_original = partition['local_to_original']
        labels = graph.x.reshape(-1).to(torch.long).tolist()
        for local_idx, original_idx in enumerate(local_to_original):
            if int(original_idx) < 0:
                continue
            x[int(original_idx)] = int(labels[local_idx])
        for src, dst in graph.edge_index.t().tolist():
            src_orig = int(local_to_original[src])
            dst_orig = int(local_to_original[dst])
            if src_orig < 0 or dst_orig < 0:
                continue
            edges.add((src_orig, dst_orig))

    for cut_group in manifest['cut_groups']:
        net_idx = int(cut_group['original_net_id'])
        x[net_idx] = net_id
        neighbors = set()
        for stub_ref in cut_group['stub_refs']:
            partition = manifest['partition_records'][stub_ref['partition_id']]
            graph = partition_cache[int(stub_ref['partition_id'])]
            local_to_original = partition['local_to_original']
            stub_idx = int(stub_ref['stub_local_idx'])
            incident = graph.edge_index[1][graph.edge_index[0] == stub_idx].tolist()
            incident += graph.edge_index[0][graph.edge_index[1] == stub_idx].tolist()
            for local_idx in incident:
                original_idx = int(local_to_original[int(local_idx)])
                if original_idx >= 0:
                    neighbors.add(original_idx)
        for neighbor in neighbors:
            edges.add((net_idx, neighbor))
            edges.add((neighbor, net_idx))

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)
    out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    return normalize_circuit_graph(out, net_id=net_id)


def manifest_cut_group_replay_assembly(
    manifest: Dict,
    meta: Dict,
    root: Path | str | None = None,
) -> Data:
    """Replay a partition manifest into a fresh assembled coordinate system.

    Unlike reconstruct_original_from_manifest(), this does not reuse original
    node ids.  It copies all non-stub partition nodes, preserves intra-partition
    edges, then creates one global net node for each manifest cut group and
    connects it to the real neighbors of the referenced boundary stubs.  This
    is an oracle assembly baseline for checking whether partition stubs and
    cut-group metadata are sufficient before evaluating generated-style
    skeleton assembly heuristics.
    """
    net_id = int(meta['net_id'])
    boundary_stub_id = int(meta['boundary_stub_id'])
    base_dir = Path(root) if root is not None else Path(meta['dataset_root'])

    global_x: List[int] = []
    global_edges: List[Tuple[int, int]] = []
    local_to_global: Dict[Tuple[int, int], int] = {}
    partition_cache: Dict[int, Data] = {}

    for partition in sorted(manifest['partition_records'], key=lambda item: int(item['partition_id'])):
        part_id = int(partition['partition_id'])
        graph = torch.load(base_dir / partition['graph_rel_path'], map_location='cpu', weights_only=False)
        partition_cache[part_id] = graph
        labels = graph.x.reshape(-1).to(torch.long).tolist()
        for local_idx, label in enumerate(labels):
            if int(label) == boundary_stub_id:
                continue
            global_idx = len(global_x)
            global_x.append(int(label))
            local_to_global[(part_id, int(local_idx))] = int(global_idx)
        for src, dst in graph.edge_index.t().tolist():
            src_global = local_to_global.get((part_id, int(src)))
            dst_global = local_to_global.get((part_id, int(dst)))
            if src_global is None or dst_global is None:
                continue
            global_edges.append((src_global, dst_global))

    for cut_group in manifest.get('cut_groups', []):
        net_global = len(global_x)
        global_x.append(net_id)
        neighbors = set()
        for stub_ref in cut_group.get('stub_refs', []):
            part_id = int(stub_ref['partition_id'])
            stub_idx = int(stub_ref['stub_local_idx'])
            graph = partition_cache[part_id]
            incident = graph.edge_index[1][graph.edge_index[0] == stub_idx].tolist()
            incident += graph.edge_index[0][graph.edge_index[1] == stub_idx].tolist()
            for local_idx in incident:
                neighbor = local_to_global.get((part_id, int(local_idx)))
                if neighbor is not None:
                    neighbors.add(int(neighbor))
        for neighbor in sorted(neighbors):
            global_edges.append((net_global, neighbor))
            global_edges.append((neighbor, net_global))

    edge_index = torch.tensor(global_edges, dtype=torch.long).t().contiguous() if global_edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)
    out = Data(
        x=torch.tensor(global_x, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(global_x),
    )
    return normalize_circuit_graph(out, net_id=net_id, remove_isolates=False)
