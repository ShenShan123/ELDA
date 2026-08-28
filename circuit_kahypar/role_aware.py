from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from circuit_kahypar.pin_spec_table import CellPinSpecTable
from circuit_kahypar.schema import (
    BOUNDARY_ROLE_AMBIG,
    BOUNDARY_ROLE_BIDIR,
    BOUNDARY_ROLE_DRIVER,
    BOUNDARY_ROLE_LOAD,
    BOUNDARY_ROLE_NONE,
)


EDGE_ROLE_UNKNOWN = 0
EDGE_ROLE_GATE_INPUT = 1
EDGE_ROLE_GATE_OUTPUT = 2
EDGE_ROLE_EXCESS = 3


ROLE_AWARE_REALIZATION_VERSION = 'role_aware_realization_v1'


@dataclass
class RoleAwareAudit:
    decoded_gate_count: int = 0
    decoded_missing_input_edges: int = 0
    decoded_missing_output_edges: int = 0
    decoded_excess_gate_edges: int = 0
    decoded_complete_pin_ratio: float = 0.0
    decoded_unknown_cell_spec_count: int = 0
    role_realizer_multi_driver_avoided: int = 0
    role_realizer_output_conflict_downgraded: int = 0
    role_realizer_private_output_created: int = 0
    role_realizer_optional_output_skipped: int = 0
    multi_driver_net_count_after_role_realization: int = 0
    input_salvage_from_unknown_edges: int = 0
    input_salvage_from_excess_edges: int = 0
    input_salvage_from_optional_outputs: int = 0
    input_salvage_from_private_outputs: int = 0
    input_salvage_unresolved_hard_degree_short: int = 0
    input_salvage_unresolved_realization_degree_short: int = 0
    input_salvage_unresolved_role_assignment_short: int = 0
    role_matching_required_inputs_satisfied: int = 0
    role_matching_required_inputs_missing: int = 0
    role_matching_min_outputs_satisfied: int = 0
    role_matching_optional_outputs_assigned: int = 0
    role_matching_optional_outputs_skipped: int = 0
    role_matching_private_outputs_created: int = 0
    realization_short_inputs_completed_with_private_output: int = 0
    realization_short_real_output_edge_reassigned_to_input: int = 0
    realization_short_private_outputs_created: int = 0
    realization_short_missing_inputs_after_fix: int = 0
    stub_driver_count: int = 0
    stub_load_count: int = 0
    stub_unknown_count: int = 0
    stub_role_inferred_from_edge_role_count: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            'decoded_gate_count': int(self.decoded_gate_count),
            'decoded_missing_input_edges': int(self.decoded_missing_input_edges),
            'decoded_missing_output_edges': int(self.decoded_missing_output_edges),
            'decoded_excess_gate_edges': int(self.decoded_excess_gate_edges),
            'decoded_complete_pin_ratio': float(self.decoded_complete_pin_ratio),
            'decoded_unknown_cell_spec_count': int(self.decoded_unknown_cell_spec_count),
            'role_realizer_multi_driver_avoided': int(self.role_realizer_multi_driver_avoided),
            'role_realizer_output_conflict_downgraded': int(self.role_realizer_output_conflict_downgraded),
            'role_realizer_private_output_created': int(self.role_realizer_private_output_created),
            'role_realizer_optional_output_skipped': int(self.role_realizer_optional_output_skipped),
            'multi_driver_net_count_after_role_realization': int(self.multi_driver_net_count_after_role_realization),
            'input_salvage_from_unknown_edges': int(self.input_salvage_from_unknown_edges),
            'input_salvage_from_excess_edges': int(self.input_salvage_from_excess_edges),
            'input_salvage_from_optional_outputs': int(self.input_salvage_from_optional_outputs),
            'input_salvage_from_private_outputs': int(self.input_salvage_from_private_outputs),
            'input_salvage_unresolved_hard_degree_short': int(self.input_salvage_unresolved_hard_degree_short),
            'input_salvage_unresolved_realization_degree_short': int(self.input_salvage_unresolved_realization_degree_short),
            'input_salvage_unresolved_role_assignment_short': int(self.input_salvage_unresolved_role_assignment_short),
            'role_matching_required_inputs_satisfied': int(self.role_matching_required_inputs_satisfied),
            'role_matching_required_inputs_missing': int(self.role_matching_required_inputs_missing),
            'role_matching_min_outputs_satisfied': int(self.role_matching_min_outputs_satisfied),
            'role_matching_optional_outputs_assigned': int(self.role_matching_optional_outputs_assigned),
            'role_matching_optional_outputs_skipped': int(self.role_matching_optional_outputs_skipped),
            'role_matching_private_outputs_created': int(self.role_matching_private_outputs_created),
            'realization_short_inputs_completed_with_private_output': int(self.realization_short_inputs_completed_with_private_output),
            'realization_short_real_output_edge_reassigned_to_input': int(self.realization_short_real_output_edge_reassigned_to_input),
            'realization_short_private_outputs_created': int(self.realization_short_private_outputs_created),
            'realization_short_missing_inputs_after_fix': int(self.realization_short_missing_inputs_after_fix),
            'stub_driver_count': int(self.stub_driver_count),
            'stub_load_count': int(self.stub_load_count),
            'stub_unknown_count': int(self.stub_unknown_count),
            'stub_role_inferred_from_edge_role_count': int(self.stub_role_inferred_from_edge_role_count),
        }


def _build_edge_lookup(edge_index: torch.Tensor) -> Dict[Tuple[int, int], int]:
    lookup: Dict[Tuple[int, int], int] = {}
    for idx, (src, dst) in enumerate(edge_index.t().tolist()):
        lookup[(int(src), int(dst))] = int(idx)
    return lookup


def annotate_gate_net_edge_roles(
    graph: Data,
    label_to_cell: Dict[int, str],
    pin_table: CellPinSpecTable,
    *,
    net_id: int,
    boundary_stub_id: int,
    preserve_existing: bool = True,
    reserve_output_when_degree_short: bool = True,
    driver_aware: bool = True,
    create_private_output_net_on_conflict: bool = True,
) -> Tuple[Data, RoleAwareAudit]:
    if getattr(graph, 'x', None) is None or getattr(graph, 'edge_index', None) is None:
        raise ValueError('graph.x and graph.edge_index are required')

    x = graph.x.reshape(-1).to(torch.long)
    edge_index = graph.edge_index.to(torch.long)
    num_edges = int(edge_index.size(1))
    num_nodes = int(x.numel())
    base_num_nodes = int(num_nodes)

    existing_role = getattr(graph, 'edge_role', None)
    existing_pin = getattr(graph, 'edge_pin_id', None)
    if preserve_existing and existing_role is not None and int(existing_role.numel()) == num_edges:
        role = existing_role.reshape(-1).to(torch.long).clone()
    else:
        role = torch.full((num_edges,), EDGE_ROLE_UNKNOWN, dtype=torch.long)
    if preserve_existing and existing_pin is not None and int(existing_pin.numel()) == num_edges:
        pin_id = existing_pin.reshape(-1).to(torch.long).clone()
    else:
        pin_id = torch.full((num_edges,), -1, dtype=torch.long)

    neighbors_set: List[set[int]] = [set() for _ in range(num_nodes)]
    for _, (src, dst) in enumerate(edge_index.t().tolist()):
        if src == dst:
            continue
        s = int(src)
        d = int(dst)
        neighbors_set[s].add(d)
        neighbors_set[d].add(s)
    neighbors: List[List[int]] = [sorted(list(s)) for s in neighbors_set]

    degrees = torch.bincount(edge_index[0], minlength=num_nodes)
    edge_lookup = _build_edge_lookup(edge_index)

    audit = RoleAwareAudit()
    gate_nodes = ((x != int(net_id)) & (x != int(boundary_stub_id))).nonzero(as_tuple=False).reshape(-1).tolist()
    gate_nodes = sorted(int(i) for i in gate_nodes)
    complete_slots = 0
    total_slots = 0

    taken_output_stubs: set[int] = set()
    net_driver_count: Dict[int, int] = {}
    has_stub_nodes = bool((x == int(boundary_stub_id)).any().item()) if boundary_stub_id is not None else False
    extra_nodes: List[int] = []
    extra_edges: List[Tuple[int, int]] = []
    extra_roles: List[int] = []
    extra_pins: List[int] = []

    for gate_idx in gate_nodes:
        label = int(x[gate_idx].item())
        cell_name = label_to_cell.get(label, f'LABEL_{label}')
        spec = pin_table.get_cell_pin_spec(cell_name)
        if spec.fallback:
            audit.decoded_unknown_cell_spec_count += 1

        inputs = list(spec.inputs)
        outputs = list(spec.outputs)
        req_inputs = len(inputs)
        req_outputs = len(outputs)
        min_req_outputs = int(spec.min_required_outputs if hasattr(spec, 'min_required_outputs') else (1 if req_outputs > 0 else 0))
        if req_inputs + req_outputs <= 0:
            continue

        incident = [
            int(n)
            for n in set(neighbors[int(gate_idx)])
            if int(x[n].item()) in {int(net_id), int(boundary_stub_id)}
        ]
        if not incident:
            audit.decoded_gate_count += 1
            audit.decoded_missing_input_edges += int(req_inputs)
            audit.decoded_missing_output_edges += int(req_outputs)
            total_slots += int(req_inputs + req_outputs)
            continue

        stub_neighbors = [n for n in incident if int(x[n].item()) == int(boundary_stub_id)]
        net_neighbors = [n for n in incident if int(x[n].item()) == int(net_id)]

        def cand_key(n: int) -> Tuple[int, int, int, int]:
            is_stub = 1 if int(x[n].item()) == int(boundary_stub_id) else 0
            driver_ok = 1
            if int(x[n].item()) == int(net_id):
                driver_ok = 1 if net_driver_count.get(int(n), 0) == 0 else 0
            return (driver_ok, is_stub, -int(degrees[n].item()), int(n))

        candidates = sorted(stub_neighbors + net_neighbors, key=cand_key, reverse=True)
        incident_edges = int(len(candidates))
        hard_degree_short = bool(incident_edges < req_inputs)
        realization_degree_short = bool((not hard_degree_short) and incident_edges < (req_inputs + min_req_outputs))

        used: set[int] = set()
        input_nodes: List[int] = []
        output_nodes: List[int] = []
        excess_nodes: List[int] = []

        if realization_degree_short and min_req_outputs > 0:
            for n in candidates:
                if len(input_nodes) >= req_inputs:
                    break
                used.add(int(n))
                input_nodes.append(int(n))
        else:
            for slot in range(req_inputs):
                if slot >= len(inputs):
                    break
                if not candidates:
                    break
                found = None
                for n in candidates:
                    if int(n) not in used:
                        found = int(n)
                        break
                if found is None:
                    break
                used.add(found)
                input_nodes.append(found)

        output_satisfied = 0
        if min_req_outputs > 0:
            if realization_degree_short and create_private_output_net_on_conflict and outputs:
                new_idx = int(num_nodes + len(extra_nodes))
                extra_nodes.append(int(net_id))
                pid = pin_table.pin_name_to_id(outputs[0])
                extra_edges.append((int(gate_idx), int(new_idx)))
                extra_edges.append((int(new_idx), int(gate_idx)))
                extra_roles.append(EDGE_ROLE_GATE_OUTPUT)
                extra_roles.append(EDGE_ROLE_GATE_OUTPUT)
                extra_pins.append(int(pid))
                extra_pins.append(int(pid))
                audit.role_matching_private_outputs_created += 1
                audit.role_realizer_private_output_created += 1
                audit.input_salvage_from_private_outputs += 1
                audit.realization_short_private_outputs_created += 1
                if len(input_nodes) >= req_inputs:
                    audit.realization_short_inputs_completed_with_private_output += 1
                audit.realization_short_missing_inputs_after_fix += int(max(0, req_inputs - len(input_nodes)))
                output_satisfied = 1
            else:
                picked = None
                for n in candidates:
                    if int(n) in used:
                        continue
                    if int(x[int(n)].item()) == int(net_id) and driver_aware and net_driver_count.get(int(n), 0) > 0:
                        continue
                    picked = int(n)
                    break
                if picked is None and create_private_output_net_on_conflict and outputs:
                    new_idx = int(num_nodes + len(extra_nodes))
                    extra_nodes.append(int(net_id))
                    pid = pin_table.pin_name_to_id(outputs[0])
                    extra_edges.append((int(gate_idx), int(new_idx)))
                    extra_edges.append((int(new_idx), int(gate_idx)))
                    extra_roles.append(EDGE_ROLE_GATE_OUTPUT)
                    extra_roles.append(EDGE_ROLE_GATE_OUTPUT)
                    extra_pins.append(int(pid))
                    extra_pins.append(int(pid))
                    audit.role_matching_private_outputs_created += 1
                    audit.role_realizer_private_output_created += 1
                    output_satisfied = 1
                elif picked is not None:
                    used.add(picked)
                    output_nodes.append(picked)
                    output_satisfied = 1
                    if int(x[int(picked)].item()) == int(net_id) and driver_aware:
                        if net_driver_count.get(int(picked), 0) == 0:
                            audit.role_realizer_multi_driver_avoided += 1
                        net_driver_count[int(picked)] = int(net_driver_count.get(int(picked), 0) + 1)

        for n in candidates:
            if int(n) in used:
                continue
            excess_nodes.append(int(n))

        missing_inputs = int(max(0, req_inputs - len(input_nodes)))
        if hard_degree_short and missing_inputs > 0:
            audit.input_salvage_unresolved_hard_degree_short += int(missing_inputs)
        elif realization_degree_short and missing_inputs > 0:
            audit.input_salvage_unresolved_realization_degree_short += int(missing_inputs)
        elif (not hard_degree_short) and (not realization_degree_short) and missing_inputs > 0:
            audit.input_salvage_unresolved_role_assignment_short += int(missing_inputs)

        audit.role_matching_required_inputs_satisfied += int(req_inputs - missing_inputs)
        audit.role_matching_required_inputs_missing += int(missing_inputs)
        audit.role_matching_min_outputs_satisfied += int(min_req_outputs > 0 and output_satisfied > 0)
        audit.role_matching_optional_outputs_skipped += int(max(0, req_outputs - min_req_outputs))
        audit.role_realizer_optional_output_skipped += int(max(0, req_outputs - min_req_outputs))

        audit.decoded_gate_count += 1
        missing_outputs = int(max(0, min_req_outputs - len(output_nodes)))
        audit.decoded_missing_input_edges += int(missing_inputs)
        audit.decoded_missing_output_edges += int(missing_outputs)
        audit.decoded_excess_gate_edges += int(len(excess_nodes))
        complete_slots += int((missing_inputs == 0) and ((min_req_outputs == 0) or (len(output_nodes) > 0)))
        total_slots += int(req_inputs + min_req_outputs)

        def _set_pair(nei: int, r: int, p: int):
            e1 = edge_lookup.get((int(gate_idx), int(nei)))
            e2 = edge_lookup.get((int(nei), int(gate_idx)))
            if e1 is not None:
                role[e1] = int(r)
                pin_id[e1] = int(p)
            if e2 is not None:
                role[e2] = int(r)
                pin_id[e2] = int(p)

        for slot, nei in enumerate(input_nodes):
            pin = inputs[slot] if slot < len(inputs) else f'IN{slot}'
            _set_pair(nei, EDGE_ROLE_GATE_INPUT, pin_table.pin_name_to_id(pin))

        for slot, nei in enumerate(output_nodes):
            pin = outputs[0] if outputs else f'OUT{slot}'
            _set_pair(nei, EDGE_ROLE_GATE_OUTPUT, pin_table.pin_name_to_id(pin))

        for nei in excess_nodes:
            _set_pair(nei, EDGE_ROLE_EXCESS, -1)

    if extra_nodes:
        new_x = torch.cat([x, torch.tensor(extra_nodes, dtype=torch.long)], dim=0)
        new_edges = torch.cat(
            [edge_index, torch.tensor(extra_edges, dtype=torch.long).t().contiguous()],
            dim=1,
        )
        role = torch.cat([role, torch.tensor(extra_roles, dtype=torch.long)], dim=0)
        pin_id = torch.cat([pin_id, torch.tensor(extra_pins, dtype=torch.long)], dim=0)
        graph.x = new_x
        graph.edge_index = new_edges
        graph.num_nodes = int(new_x.numel())
        graph.base_num_nodes = int(base_num_nodes)
        graph.private_output_net_indices = [int(base_num_nodes + i) for i in range(len(extra_nodes))]
    else:
        graph.base_num_nodes = int(base_num_nodes)
        graph.private_output_net_indices = []

    graph.edge_role = role
    graph.edge_pin_id = pin_id
    graph.pin_id_to_name = pin_table.pin_id_to_name

    if driver_aware:
        x2 = graph.x.reshape(-1).to(torch.long)
        e2 = graph.edge_index.to(torch.long)
        r2 = graph.edge_role.reshape(-1).to(torch.long)
        net_drivers: Dict[int, set[int]] = {}
        for eidx, (src, dst) in enumerate(e2.t().tolist()):
            if src == dst:
                continue
            if int(x2[src].item()) != int(net_id):
                continue
            if int(x2[dst].item()) == int(net_id) or int(x2[dst].item()) == int(boundary_stub_id):
                continue
            if int(r2[int(eidx)].item()) != EDGE_ROLE_GATE_OUTPUT:
                continue
            net_drivers.setdefault(int(src), set()).add(int(dst))
        audit.multi_driver_net_count_after_role_realization = int(sum(1 for s in net_drivers.values() if len(s) > 1))
    audit.decoded_complete_pin_ratio = float(complete_slots / max(1, total_slots))
    return graph, audit


def infer_boundary_stub_roles_from_edge_roles(
    graph: Data,
    *,
    net_id: int,
    boundary_stub_id: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    x = graph.x.reshape(-1).to(torch.long)
    edge_index = graph.edge_index.to(torch.long)
    edge_role = getattr(graph, 'edge_role', None)
    if edge_role is None or int(edge_role.numel()) != int(edge_index.size(1)):
        roles = torch.full((int(x.numel()),), BOUNDARY_ROLE_NONE, dtype=torch.long)
        stub_nodes = (x == int(boundary_stub_id)).nonzero(as_tuple=False).reshape(-1)
        roles[stub_nodes] = BOUNDARY_ROLE_AMBIG
        return roles, {
            'stub_driver_count': 0,
            'stub_load_count': 0,
            'stub_unknown_count': int(stub_nodes.numel()),
            'stub_role_inferred_from_edge_role_count': 0,
        }

    neighbors: List[List[int]] = [[] for _ in range(int(x.numel()))]
    role_by_pair: Dict[Tuple[int, int], int] = {}
    for idx, (src, dst) in enumerate(edge_index.t().tolist()):
        if src == dst:
            continue
        neighbors[int(src)].append(int(dst))
        role_by_pair[(int(src), int(dst))] = int(edge_role[idx].item())

    out = torch.full((int(x.numel()),), BOUNDARY_ROLE_NONE, dtype=torch.long)
    stub_nodes = (x == int(boundary_stub_id)).nonzero(as_tuple=False).reshape(-1).tolist()
    driver = 0
    load = 0
    unknown = 0
    inferred = 0
    for stub in stub_nodes:
        votes = set()
        for nei in neighbors[int(stub)]:
            if int(x[nei].item()) == int(net_id) or int(x[nei].item()) == int(boundary_stub_id):
                continue
            r = role_by_pair.get((int(nei), int(stub)))
            if r == EDGE_ROLE_GATE_OUTPUT:
                votes.add('driver')
            elif r == EDGE_ROLE_GATE_INPUT:
                votes.add('load')
        if votes == {'driver'}:
            out[int(stub)] = BOUNDARY_ROLE_DRIVER
            driver += 1
            inferred += 1
        elif votes == {'load'}:
            out[int(stub)] = BOUNDARY_ROLE_LOAD
            load += 1
            inferred += 1
        elif votes == {'driver', 'load'}:
            out[int(stub)] = BOUNDARY_ROLE_BIDIR
            inferred += 1
        else:
            out[int(stub)] = BOUNDARY_ROLE_AMBIG
            unknown += 1

    return out, {
        'stub_driver_count': int(driver),
        'stub_load_count': int(load),
        'stub_unknown_count': int(unknown),
        'stub_role_inferred_from_edge_role_count': int(inferred),
    }
