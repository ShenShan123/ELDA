#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch_geometric.data import Data

from circuit_kahypar.liberty import default_nangate45_liberty, parse_liberty_pin_specs
from circuit_kahypar.schema import derive_special_node_ids


def load_mapping(path: str | Path) -> Dict[int, str]:
    text = Path(path).read_text(encoding='utf-8')
    _, rhs = text.split('=', 1)
    mapping = ast.literal_eval(rhs.strip())
    return {int(label): str(cell) for cell, label in mapping.items()}


def resolve_net_id(mapping_path: str | Path | None, explicit_net_id: int) -> int:
    if int(explicit_net_id) > 0:
        return int(explicit_net_id)
    net_id, _ = derive_special_node_ids(mapping_path)
    return int(net_id)


def infer_cell_spec(cell_name: str) -> Dict[str, int]:
    name = cell_name.upper()
    if name in {'LOGIC0_X1', 'LOGIC1_X1'}:
        return {'inputs': 0, 'outputs': 1}
    if name.startswith(('INV_', 'BUF_', 'CLKBUF_')):
        return {'inputs': 1, 'outputs': 1}
    if name.startswith(('TBUF_', 'TINV_')):
        return {'inputs': 2, 'outputs': 1}
    if name.startswith(('XOR2_', 'XNOR2_')):
        return {'inputs': 2, 'outputs': 1}
    if name.startswith(('AOI21_', 'OAI21_')):
        return {'inputs': 3, 'outputs': 1}
    if name.startswith(('AOI22_', 'OAI22_')):
        return {'inputs': 4, 'outputs': 1}
    if name.startswith('MUX2_'):
        return {'inputs': 3, 'outputs': 1}
    if name.startswith('HA_'):
        return {'inputs': 2, 'outputs': 2}
    if name.startswith('FA_'):
        return {'inputs': 3, 'outputs': 2}
    if name.startswith('DFFSR_'):
        return {'inputs': 4, 'outputs': 1}
    if name.startswith(('DFFR_', 'DFFS_')):
        return {'inputs': 3, 'outputs': 1}
    if name.startswith('DFF_'):
        return {'inputs': 2, 'outputs': 1}
    if name.startswith(('FILLCELL_', 'ANTENNA_')):
        return {'inputs': 0, 'outputs': 0}
    m = re.match(r'^(NAND|NOR|AND|OR)(\d)_', name)
    if m:
        return {'inputs': int(m.group(2)), 'outputs': 1}
    if name == 'UNKNOWN':
        return {'inputs': 6, 'outputs': 1}
    return {'inputs': 6, 'outputs': 1}


def _fallback_pin_spec(cell_name: str) -> Dict[str, List[str] | bool]:
    counts = infer_cell_spec(cell_name)
    outputs = [f'OUT{i}' for i in range(int(counts['outputs']))]
    min_required_outputs = 1 if len(outputs) > 0 else 0
    return {
        'inputs': [f'IN{i}' for i in range(int(counts['inputs']))],
        'outputs': outputs,
        'inouts': [],
        'min_required_outputs': int(min_required_outputs),
        'fallback': True,
    }


def _cell_pin_spec(cell_name: str, liberty_specs: Dict[str, Dict[str, List[str]]] | None) -> Dict[str, List[str] | bool]:
    if liberty_specs and cell_name in liberty_specs:
        spec = liberty_specs[cell_name]
        outputs = list(spec.get('outputs', []))
        min_required_outputs = 1 if len(outputs) > 0 else 0
        return {
            'inputs': list(spec.get('inputs', [])),
            'outputs': outputs,
            'inouts': list(spec.get('inouts', [])),
            'min_required_outputs': int(min_required_outputs),
            'fallback': False,
        }
    return _fallback_pin_spec(cell_name)


def _choose_real_outputs(
    incident_nets: List[int],
    output_count: int,
    input_count: int,
    driver_count: Dict[int, int],
    net_degree: Dict[int, int],
) -> List[int]:
    if output_count <= 0 or not incident_nets:
        return []
    free_nets = [net for net in incident_nets if driver_count.get(net, 0) == 0]
    choose = min(output_count, len(free_nets))
    if choose <= 0:
        return []
    if choose == len(free_nets):
        return sorted(free_nets)

    best_subset = None
    best_score = None
    for subset in itertools.combinations(free_nets, choose):
        subset_set = set(subset)
        remaining = [net for net in incident_nets if net not in subset_set]
        shortage = max(0, input_count - len(remaining))
        score = (
            shortage,
            -sum(net_degree.get(net, 0) for net in subset),
            tuple(sorted(subset)),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_subset = list(subset)
    return sorted(best_subset) if best_subset is not None else sorted(free_nets[:choose])


def _fill_inputs_with_reuse(real_inputs: List[int], input_count: int) -> Tuple[List[int], List[int]]:
    reused_inputs: List[int] = []
    if len(real_inputs) >= input_count:
        return real_inputs[:input_count], reused_inputs
    if real_inputs:
        idx = 0
        while len(real_inputs) + len(reused_inputs) < input_count:
            reused_inputs.append(real_inputs[idx % len(real_inputs)])
            idx += 1
    return real_inputs, reused_inputs


def choose_gate_nets(
    graph: Data,
    net_id: int,
    label_to_cell: Dict[int, str],
    liberty_specs: Dict[str, Dict[str, List[str]]] | None = None,
    *,
    use_edge_roles: bool = True,
    strict_mode: bool = False,
    split_multi_driver_nets: bool = True,
    input_pool_enabled: bool = True,
    max_shared_input_fanout: int = 64,
    use_const_pool_for_missing_inputs: bool = False,
    enable_helper_input_stages: bool = True,
    allow_input_reuse: bool = True,
    fallback_events: List[Dict[str, Any]] | None = None,
    fallback_event_context: Dict[str, Any] | None = None,
    max_reused_source_fanout: int | None = None,
    forbid_clock_like_net_as_data_fallback: bool = False,
    penalize_role_mixed_reuse: bool = False,
    report_repeated_input_signatures: bool = False,
    critical_optional_output_policy: str = 'report_only',
) -> Tuple[List[Dict], Dict[int, int], Dict[int, int], List[str], Dict[str, Dict[str, int]], Dict[str, int], Dict[str, Dict[str, float]]]:
    x = graph.x.reshape(-1).to(torch.long)
    num_nodes = int(graph.num_nodes)
    neighbors = [set() for _ in range(num_nodes)]
    for u, v in graph.edge_index.t().tolist():
        if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
            neighbors[u].add(v)
            neighbors[v].add(u)

    gate_nodes = [idx for idx in range(num_nodes) if int(x[idx].item()) != int(net_id)]
    net_nodes = [idx for idx in range(num_nodes) if int(x[idx].item()) == int(net_id)]
    net_degree = {net: len(neighbors[net]) for net in net_nodes}
    driver_count = {net: 0 for net in net_nodes}
    load_count = {net: 0 for net in net_nodes}
    warnings: List[str] = []
    gate_records: List[Dict] = []
    used_specs: Dict[str, Dict[str, int]] = {}
    issue_counter: Counter[str] = Counter()
    cell_rollup: Dict[str, Counter] = defaultdict(Counter)
    helper_cell_name = 'BUF_X1' if 'BUF_X1' in set(label_to_cell.values()) else ('INV_X1' if 'INV_X1' in set(label_to_cell.values()) else None)

    pool_fanout: Dict[str, int] = {}
    created_input_pool_nets = 0
    reused_input_pool_connections = 0
    max_pool_fanout_seen = 0
    fallback_events = fallback_events if fallback_events is not None else []
    fallback_event_context = dict(fallback_event_context or {})

    def _net_name(value: int | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, int):
            return f'n_{value}'
        return str(value)

    def _pin_role_name(cell_name: str, pin_name: str | None, is_output: bool = False) -> str:
        pin = str(pin_name or '').upper()
        if is_output:
            return 'output'
        if pin in {'CK', 'CLK', 'CLOCK'}:
            return 'clock'
        if pin in {'RST', 'RESET', 'RN', 'SN'}:
            return 'reset'
        if cell_name.upper().startswith(('FA_', 'HA_')):
            return 'arithmetic_input'
        return 'data_input'

    def _add_fallback_event(
        *,
        fallback_kind: str,
        gate_idx: int,
        cell_name: str,
        pin_name: str | None,
        pin_role: str | None = None,
        source_net_before: int | str | None = None,
        source_net_after: int | str | None = None,
        reused_from_pin: str | None = None,
        reused_from_net: int | str | None = None,
        reason: str = '',
        is_cut_group_related: bool | None = None,
    ) -> None:
        after = _net_name(source_net_after)
        before = _net_name(source_net_before)
        reused = _net_name(reused_from_net)
        event = {
            **fallback_event_context,
            'gate_id': int(gate_idx),
            'cell_type': str(cell_name),
            'pin_name': str(pin_name) if pin_name is not None else None,
            'pin_role': str(pin_role or _pin_role_name(cell_name, pin_name)),
            'fallback_kind': str(fallback_kind),
            'source_net_before': before,
            'source_net_after': after,
            'reused_from_pin': reused_from_pin,
            'reused_from_net': reused,
            'is_top_input_source': None,
            'is_pool_source': bool(after and after.startswith('pool_')),
            'is_cut_group_related': is_cut_group_related,
            'reason': str(reason),
        }
        fallback_events.append(event)

    def _alloc_input_from_pool(pin_name: str | None) -> str:
        nonlocal created_input_pool_nets, reused_input_pool_connections, max_pool_fanout_seen
        group = 'pi'
        name = (pin_name or '').upper()
        if use_const_pool_for_missing_inputs and name in {'A', 'B', 'C', 'D'}:
            group = 'pi'
        if name in {'CK', 'CLK', 'CLOCK'}:
            group = 'clk'
        if name in {'RST', 'RESET', 'RN', 'SN'}:
            group = 'rst'
        idx = 0
        while True:
            net = f'pool_{group}_{idx}'
            fan = int(pool_fanout.get(net, 0))
            if fan < int(max_shared_input_fanout):
                if fan == 0:
                    created_input_pool_nets += 1
                pool_fanout[net] = fan + 1
                reused_input_pool_connections += 1
                max_pool_fanout_seen = max(max_pool_fanout_seen, fan + 1)
                return net
            idx += 1

    edge_role = getattr(graph, 'edge_role', None)
    edge_pin = getattr(graph, 'edge_pin_id', None)
    pin_id_to_name = getattr(graph, 'pin_id_to_name', None)
    has_roles = bool(use_edge_roles) and isinstance(edge_role, torch.Tensor) and int(edge_role.numel()) == int(graph.edge_index.size(1))
    has_pins = (
        isinstance(edge_pin, torch.Tensor)
        and int(edge_pin.numel()) == int(graph.edge_index.size(1))
        and isinstance(pin_id_to_name, list)
        and len(pin_id_to_name) > 0
    )
    edge_lookup = {(int(src), int(dst)): int(idx) for idx, (src, dst) in enumerate(graph.edge_index.t().tolist())}

    def _gate_priority(idx: int):
        label = int(x[idx].item())
        cell_name = label_to_cell.get(label, f'LABEL_{label}')
        spec = _cell_pin_spec(cell_name, liberty_specs)
        req_total = len(spec['inputs']) + len(spec['outputs'])
        slack = max(0, len(neighbors[idx]) - req_total)
        return (
            slack,
            len(neighbors[idx]),
            -len(spec['outputs']),
            -len(spec['inputs']),
            idx,
        )

    gate_nodes_sorted = sorted(gate_nodes, key=_gate_priority)
    for gate_idx in gate_nodes_sorted:
        label = int(x[gate_idx].item())
        cell_name = label_to_cell.get(label, f'LABEL_{label}')
        spec = _cell_pin_spec(cell_name, liberty_specs)
        used_specs[cell_name] = spec
        if bool(spec.get('fallback', False)) and liberty_specs is not None:
            issue_counter['missing_liberty_cell_specs'] += 1
        incident_nets = sorted([n for n in neighbors[gate_idx] if int(x[n].item()) == int(net_id)])

        available_outputs = list(spec.get('outputs', []))
        output_count = len(available_outputs)
        min_required_outputs = int(spec.get('min_required_outputs', 1 if output_count > 0 else 0))
        required_output_count = int(min_required_outputs)
        input_count = len(spec['inputs'])
        role_output_nets: List[int] = []
        role_input_nets: List[int] = []
        pin_to_net: Dict[str, str] = {}
        pin_role: Dict[str, int] = {}
        if has_roles:
            for net in incident_nets:
                eidx = edge_lookup.get((int(gate_idx), int(net)))
                if eidx is None:
                    continue
                r = int(edge_role[int(eidx)].item())
                if r == 2:
                    role_output_nets.append(int(net))
                elif r == 1:
                    role_input_nets.append(int(net))
                if has_pins:
                    pid = int(edge_pin[int(eidx)].item())
                    if 0 <= pid < len(pin_id_to_name):
                        name = str(pin_id_to_name[pid])
                        pin_to_net.setdefault(name, f'n_{int(net)}')
                        pin_role.setdefault(name, int(r))
            role_output_nets = sorted(set(role_output_nets))
            role_input_nets = sorted(set(role_input_nets))

        if role_output_nets:
            preferred_outputs = [n for n in role_output_nets if driver_count.get(n, 0) == 0]
            preferred_outputs = preferred_outputs or role_output_nets
            real_outputs = preferred_outputs[:required_output_count]
            if len(real_outputs) < required_output_count:
                extra = _choose_real_outputs(
                    incident_nets=[n for n in incident_nets if n not in set(real_outputs)],
                    output_count=required_output_count - len(real_outputs),
                    input_count=input_count,
                    driver_count=driver_count,
                    net_degree=net_degree,
                )
                real_outputs.extend(extra)
        else:
            real_outputs = _choose_real_outputs(
                incident_nets=incident_nets,
                output_count=required_output_count,
                input_count=input_count,
                driver_count=driver_count,
                net_degree=net_degree,
            )
        repaired_output_nets: List[str] = []
        connected_outputs = len(real_outputs)
        missing_required_outputs = max(0, min_required_outputs - connected_outputs)
        missing_optional_outputs = max(0, output_count - max(connected_outputs, min_required_outputs))
        if missing_required_outputs and not strict_mode:
            repaired_output_nets = [f'syn_out_g{gate_idx}_{idx}' for idx in range(missing_required_outputs)]
            issue_counter['repaired_output_nets'] += missing_required_outputs
            issue_counter['synthetic_output_nets'] += missing_required_outputs
            for idx, net_name in enumerate(repaired_output_nets):
                pin_name = available_outputs[connected_outputs + idx] if (connected_outputs + idx) < len(available_outputs) else None
                _add_fallback_event(
                    fallback_kind='repaired_output',
                    gate_idx=gate_idx,
                    cell_name=cell_name,
                    pin_name=pin_name,
                    pin_role='output',
                    source_net_after=net_name,
                    reason='required output pin lacked a graph source net',
                )
                _add_fallback_event(
                    fallback_kind='synthetic_output',
                    gate_idx=gate_idx,
                    cell_name=cell_name,
                    pin_name=pin_name,
                    pin_role='output',
                    source_net_after=net_name,
                    reason='required output repaired with synthetic output net',
                )
            warnings.append(
                f'gate {gate_idx} ({cell_name}) repairs {missing_required_outputs} missing outputs with synthetic internal output nets'
            )
        if missing_required_outputs and strict_mode:
            issue_counter['missing_output_pins'] += int(missing_required_outputs)
        if missing_optional_outputs:
            issue_counter['missing_optional_output_pins'] += int(missing_optional_outputs)
            for idx in range(missing_optional_outputs):
                pin_idx = max(connected_outputs, min_required_outputs) + idx
                pin_name = available_outputs[pin_idx] if pin_idx < len(available_outputs) else None
                _add_fallback_event(
                    fallback_kind='missing_optional_output',
                    gate_idx=gate_idx,
                    cell_name=cell_name,
                    pin_name=pin_name,
                    pin_role='output',
                    reason='optional output pin was left floating',
                )
        if connected_outputs <= 0 and output_count > 0:
            issue_counter['missing_all_output_count'] += 1

        for net in real_outputs:
            driver_count[net] = driver_count.get(net, 0) + 1

        remaining_nets = [net for net in incident_nets if net not in set(real_outputs)]
        if role_input_nets:
            remaining_nets = sorted(set(role_input_nets + remaining_nets))
        remaining_nets = sorted(
            remaining_nets,
            key=lambda net: (
                0 if driver_count.get(net, 0) > 0 else 1,
                -driver_count.get(net, 0),
                -net_degree.get(net, 0),
                net,
            ),
        )
        base_inputs = remaining_nets[:input_count]
        helper_inputs: List[Dict[str, str]] = []
        synthetic_inputs: List[str] = []
        if strict_mode:
            real_inputs = base_inputs[:input_count]
            reused_inputs = []
            missing = max(0, input_count - len(real_inputs))
            if missing:
                issue_counter['missing_input_pins'] += int(missing)
                pin_names = list(spec.get('inputs', []))
                for idx in range(missing):
                    pin_idx = len(real_inputs) + idx
                    _add_fallback_event(
                        fallback_kind='missing_required_before_pool',
                        gate_idx=gate_idx,
                        cell_name=cell_name,
                        pin_name=pin_names[pin_idx] if pin_idx < len(pin_names) else None,
                        reason='strict export lacks enough usable input nets',
                    )
            issue_counter['missing_required_input_before_pool'] += int(missing)
            issue_counter['missing_required_input_after_pool'] += int(missing)
        else:
            if allow_input_reuse:
                real_inputs, reused_inputs = _fill_inputs_with_reuse(base_inputs, input_count)
            else:
                real_inputs = base_inputs[:input_count]
                reused_inputs = []
            current_input_count = len(real_inputs) + len(reused_inputs)
            if current_input_count < input_count:
                issue_counter['missing_required_input_before_pool'] += int(input_count - current_input_count)
                pin_names = list(spec.get('inputs', []))
                for pin_idx in range(current_input_count, input_count):
                    _add_fallback_event(
                        fallback_kind='missing_required_before_pool',
                        gate_idx=gate_idx,
                        cell_name=cell_name,
                        pin_name=pin_names[pin_idx] if pin_idx < len(pin_names) else None,
                        reason='practical export lacks enough native usable input nets before pool/helper repair',
                    )

            if enable_helper_input_stages and current_input_count < input_count and helper_cell_name is not None:
                source_candidates = [f'n_{net}' for net in (real_inputs + reused_inputs)]
                missing = input_count - current_input_count
                helper_repairs = min(missing, len(source_candidates))
                for idx in range(helper_repairs):
                    pin_names = list(spec.get('inputs', []))
                    pin_idx = current_input_count + idx
                    src = source_candidates[idx % len(source_candidates)]
                    helper_inputs.append({
                        'cell_name': helper_cell_name,
                        'src': src,
                        'out': f'helper_in_g{gate_idx}_{idx}',
                    })
                    _add_fallback_event(
                        fallback_kind='reused_input',
                        gate_idx=gate_idx,
                        cell_name=cell_name,
                        pin_name=pin_names[pin_idx] if pin_idx < len(pin_names) else None,
                        source_net_before=src,
                        source_net_after=f'helper_in_g{gate_idx}_{idx}',
                        reused_from_net=src,
                        reason='helper stage reused an existing source to repair missing input',
                    )
                if helper_repairs:
                    used_specs[helper_cell_name] = _cell_pin_spec(helper_cell_name, liberty_specs)
                    issue_counter['helper_input_stages'] += helper_repairs
                    warnings.append(
                        f'gate {gate_idx} ({cell_name}) inserts {helper_repairs} helper stages to repair missing inputs'
                    )
                current_input_count += helper_repairs
            if current_input_count < input_count:
                missing = input_count - current_input_count
                if input_pool_enabled:
                    pin_names = list(spec.get('inputs', []))
                    for idx in range(missing):
                        pname = pin_names[current_input_count + idx] if (current_input_count + idx) < len(pin_names) else None
                        pool_net = _alloc_input_from_pool(pname)
                        synthetic_inputs.append(pool_net)
                        _add_fallback_event(
                            fallback_kind='input_pool',
                            gate_idx=gate_idx,
                            cell_name=cell_name,
                            pin_name=pname,
                            source_net_after=pool_net,
                            reason='input pool repaired missing required input',
                        )
                    issue_counter['synthetic_input_connections'] += missing
                    issue_counter['missing_required_input_after_pool'] += 0
                    warnings.append(
                        f'gate {gate_idx} ({cell_name}) uses {missing} pooled primary input connections'
                    )
                else:
                    synthetic_inputs = [f'syn_in_g{gate_idx}_{idx}' for idx in range(missing)]
                    pin_names = list(spec.get('inputs', []))
                    for idx, syn_net in enumerate(synthetic_inputs):
                        pname = pin_names[current_input_count + idx] if (current_input_count + idx) < len(pin_names) else None
                        _add_fallback_event(
                            fallback_kind='synthetic_input',
                            gate_idx=gate_idx,
                            cell_name=cell_name,
                            pin_name=pname,
                            source_net_after=syn_net,
                            reason='synthetic input repaired missing required input',
                        )
                    issue_counter['synthetic_inputs'] += missing
                    issue_counter['synthetic_input_connections'] += missing
                    issue_counter['missing_required_input_after_pool'] += 0
                    warnings.append(
                        f'gate {gate_idx} ({cell_name}) uses {missing} synthetic primary inputs because the graph does not expose enough usable input nets'
                    )
            if reused_inputs:
                issue_counter['reused_inputs'] += len(reused_inputs)
                pin_names = list(spec.get('inputs', []))
                for idx, reused_net in enumerate(reused_inputs):
                    pin_idx = len(real_inputs) + idx
                    _add_fallback_event(
                        fallback_kind='reused_input',
                        gate_idx=gate_idx,
                        cell_name=cell_name,
                        pin_name=pin_names[pin_idx] if pin_idx < len(pin_names) else None,
                        source_net_after=reused_net,
                        reused_from_net=reused_net,
                        reason='existing input net reused to satisfy inferred pin arity',
                    )
                warnings.append(
                    f'gate {gate_idx} ({cell_name}) reuses {len(reused_inputs)} existing input nets to satisfy inferred pin arity'
                )

        input_pins_for_events = list(spec.get('inputs', []))
        all_input_nets_for_events = [f'n_{net}' for net in real_inputs] + [f'n_{net}' for net in reused_inputs] + [item['out'] for item in helper_inputs] + synthetic_inputs
        seen_input_nets: Dict[str, str] = {}
        for idx, net_name in enumerate(all_input_nets_for_events[:len(input_pins_for_events)]):
            pin_name = str(input_pins_for_events[idx])
            if net_name in seen_input_nets:
                _add_fallback_event(
                    fallback_kind='same_net_reuse',
                    gate_idx=gate_idx,
                    cell_name=cell_name,
                    pin_name=pin_name,
                    source_net_after=net_name,
                    reused_from_pin=seen_input_nets[net_name],
                    reused_from_net=net_name,
                    reason='same source net appears on multiple input pins before final Verilog legalization',
                )
            else:
                seen_input_nets[net_name] = pin_name

        dropped_incident = remaining_nets[input_count:]
        if dropped_incident:
            issue_counter['dropped_extra_incident'] += len(dropped_incident)
            warnings.append(
                f'gate {gate_idx} ({cell_name}) ignores {len(dropped_incident)} extra incident nets beyond inferred arity'
            )

        for net in real_inputs + reused_inputs:
            load_count[net] = load_count.get(net, 0) + 1

        gate_records.append({
            'gate_idx': gate_idx,
            'label': label,
            'cell_name': cell_name,
            'real_inputs': real_inputs,
            'reused_inputs': reused_inputs,
            'helper_inputs': helper_inputs,
            'synthetic_inputs': synthetic_inputs,
            'real_outputs': real_outputs,
            'repaired_output_nets': repaired_output_nets,
            'unconnected_outputs': int(missing_required_outputs),
            'dropped_incident': dropped_incident,
            'spec': spec,
            'incident_net_count': len(incident_nets),
            'pin_to_net': pin_to_net,
            'pin_role': pin_role,
            'role_driven': bool(has_roles),
            'incident_nets': incident_nets,
            'role_output_nets': role_output_nets,
            'role_input_nets': role_input_nets,
            'available_outputs': available_outputs,
            'min_required_outputs': int(min_required_outputs),
        })

        roll = cell_rollup[cell_name]
        roll['count'] += 1
        roll['incident_nets'] += len(incident_nets)
        roll['required_pins'] += input_count + output_count
        roll['real_inputs'] += len(real_inputs)
        roll['reused_inputs'] += len(reused_inputs)
        roll['helper_inputs'] += len(helper_inputs)
        roll['synthetic_inputs'] += len(synthetic_inputs)
        roll['real_outputs'] += len(real_outputs) + len(repaired_output_nets)
        roll['unconnected_outputs'] += 0

    for net, count in driver_count.items():
        if count > 1:
            warnings.append(f'net {net} has {count} inferred drivers')

    issue_counter.setdefault('synthetic_output_nets', 0)
    issue_counter.setdefault('missing_input_pins', 0)
    issue_counter.setdefault('missing_output_pins', 0)
    issue_counter.setdefault('missing_optional_output_pins', 0)
    issue_counter.setdefault('missing_all_output_count', 0)
    issue_counter.setdefault('multi_driver_splits', 0)
    issue_counter.setdefault('synthetic_input_connections', 0)
    issue_counter.setdefault('missing_required_input_before_pool', 0)
    issue_counter.setdefault('missing_required_input_after_pool', 0)
    if input_pool_enabled and not strict_mode:
        issue_counter['created_input_pool_nets'] = int(created_input_pool_nets)
        issue_counter['input_pool_connections'] = int(reused_input_pool_connections)
        issue_counter['max_input_pool_fanout'] = int(max_pool_fanout_seen)
        issue_counter['avg_input_pool_fanout'] = float((sum(pool_fanout.values()) / max(1, len(pool_fanout))) if pool_fanout else 0.0)
        issue_counter['synthetic_inputs'] = int(created_input_pool_nets)
    issue_counter['multi_driver_net_count_before'] = sum(1 for c in driver_count.values() if int(c) > 1)

    if split_multi_driver_nets and not strict_mode:
        moved = 0
        per_net_drivers: Dict[int, List[int]] = defaultdict(list)
        for ridx, rec in enumerate(gate_records):
            for net in rec.get('real_outputs', []):
                per_net_drivers[int(net)].append(int(ridx))
        for net, drivers in list(per_net_drivers.items()):
            if len(drivers) <= 1:
                continue
            for ridx in drivers[1:]:
                rec = gate_records[int(ridx)]
                candidates = [int(n) for n in rec.get('role_output_nets', []) if driver_count.get(int(n), 0) == 0]
                if not candidates:
                    candidates = [int(n) for n in rec.get('incident_nets', []) if driver_count.get(int(n), 0) == 0]
                if not candidates:
                    continue
                new_net = max(candidates, key=lambda n: (net_degree.get(int(n), 0), -int(n)))
                if int(net) in set(rec.get('real_outputs', [])):
                    rec['real_outputs'] = [int(new_net) if int(n) == int(net) else int(n) for n in rec.get('real_outputs', [])]
                    pin_to_net = rec.get('pin_to_net') if isinstance(rec.get('pin_to_net'), dict) else {}
                    pin_role = rec.get('pin_role') if isinstance(rec.get('pin_role'), dict) else {}
                    for pin_name, net_name in list(pin_to_net.items()):
                        if int(pin_role.get(pin_name, 0)) == 2 and str(net_name) == f'n_{int(net)}':
                            pin_to_net[str(pin_name)] = f'n_{int(new_net)}'
                    driver_count[int(net)] = max(0, int(driver_count.get(int(net), 1)) - 1)
                    driver_count[int(new_net)] = int(driver_count.get(int(new_net), 0) + 1)
                    moved += 1
        issue_counter['role_realizer_output_conflict_reassigned'] = int(moved)

    if split_multi_driver_nets and not strict_mode:
        net_to_driver_records: Dict[int, List[int]] = defaultdict(list)
        for ridx, record in enumerate(gate_records):
            for net in record.get('real_outputs', []):
                net_to_driver_records[int(net)].append(int(ridx))
        for net, drivers in net_to_driver_records.items():
            if len(drivers) <= 1:
                continue
            for extra_idx, ridx in enumerate(drivers[1:], start=1):
                record = gate_records[ridx]
                if int(net) not in set(record.get('real_outputs', [])):
                    continue
                split_wire = f'syn_split_out_g{record["gate_idx"]}_n{int(net)}_{extra_idx}'
                record['real_outputs'] = [int(n) for n in record.get('real_outputs', []) if int(n) != int(net)]
                record['repaired_output_nets'].append(split_wire)
                pin_to_net = record.get('pin_to_net') if isinstance(record.get('pin_to_net'), dict) else {}
                pin_role = record.get('pin_role') if isinstance(record.get('pin_role'), dict) else {}
                for pin_name, net_name in list(pin_to_net.items()):
                    if int(pin_role.get(pin_name, 0)) == 2 and str(net_name) == f'n_{int(net)}':
                        pin_to_net[str(pin_name)] = str(split_wire)
                issue_counter['multi_driver_splits'] += 1
                driver_count[int(net)] = max(1, int(driver_count.get(int(net), 1)) - 1)
    issue_counter['multi_driver_net_count_after'] = sum(1 for c in driver_count.values() if int(c) > 1)

    input_fanout_by_net: Counter[str] = Counter()
    input_roles_by_net: Dict[str, set[str]] = defaultdict(set)
    repeated_signatures: Counter[str] = Counter()
    critical_optional_total = 0
    critical_optional_connected = 0
    for record in gate_records:
        spec = record.get('spec') or {}
        cell_name = str(record.get('cell_name'))
        input_pins = [str(p) for p in spec.get('inputs', [])]
        output_pins = [str(p) for p in spec.get('outputs', [])]
        all_inputs = [f'n_{net}' for net in record['real_inputs']] + [f'n_{net}' for net in record['reused_inputs']] + [item['out'] for item in record.get('helper_inputs', [])] + record['synthetic_inputs']
        sig = f"{cell_name}|" + ",".join(all_inputs[:len(input_pins)])
        repeated_signatures[sig] += 1
        for idx, net_name in enumerate(all_inputs[:len(input_pins)]):
            pin_name = input_pins[idx] if idx < len(input_pins) else None
            input_fanout_by_net[str(net_name)] += 1
            input_roles_by_net[str(net_name)].add(_pin_role_name(cell_name, pin_name))
        if cell_name.startswith(('FA_', 'HA_')) and 'S' in output_pins:
            critical_optional_total += 1
            output_count = len(record.get('real_outputs', [])) + len(record.get('repaired_output_nets', []))
            s_index = output_pins.index('S')
            if s_index < output_count:
                critical_optional_connected += 1
    top_reused_source_net = None
    if input_fanout_by_net:
        top_reused_source_net, top_reused_source_fanout = input_fanout_by_net.most_common(1)[0]
    else:
        top_reused_source_fanout = 0
    role_mixed_count = sum(1 for net, roles in input_roles_by_net.items() if len(roles) > 1 and int(input_fanout_by_net.get(net, 0)) > 16)
    clock_like_data_reuse = sum(
        int(input_fanout_by_net.get(net, 0))
        for net, roles in input_roles_by_net.items()
        if 'clock' in roles and any(r not in {'clock', 'reset'} for r in roles)
    )
    repeated_input_signature_count = sum(c - 1 for c in repeated_signatures.values() if c > 1)
    if report_repeated_input_signatures:
        issue_counter['repeated_input_signature_count'] = int(repeated_input_signature_count)
    else:
        issue_counter['repeated_input_signature_count'] = int(repeated_input_signature_count)
    issue_counter['max_reused_source_fanout'] = int(top_reused_source_fanout)
    issue_counter['top_reused_source_net'] = str(top_reused_source_net or '')
    issue_counter['role_mixed_high_fanout_count'] = int(role_mixed_count)
    issue_counter['clock_like_net_used_as_data_count'] = int(clock_like_data_reuse)
    issue_counter['critical_optional_output_missing_count'] = int(max(0, critical_optional_total - critical_optional_connected))
    issue_counter['FA_HA_S_connected_rate_x1000'] = int(round(1000.0 * critical_optional_connected / max(1, critical_optional_total)))
    if max_reused_source_fanout is not None and top_reused_source_fanout > int(max_reused_source_fanout):
        issue_counter['max_reused_source_fanout_report_only_violation'] = int(top_reused_source_fanout - int(max_reused_source_fanout))
    if forbid_clock_like_net_as_data_fallback and clock_like_data_reuse:
        issue_counter['clock_like_net_data_reuse_report_only_violation'] = int(clock_like_data_reuse)
    if penalize_role_mixed_reuse and role_mixed_count:
        issue_counter['role_mixed_reuse_report_only_violation'] = int(role_mixed_count)

    if split_multi_driver_nets and not strict_mode:
        for record in gate_records:
            pin_to_net = record.get('pin_to_net') if isinstance(record.get('pin_to_net'), dict) else {}
            pin_role = record.get('pin_role') if isinstance(record.get('pin_role'), dict) else {}
            output_pins = list(record.get('spec', {}).get('outputs', []))
            actual_outputs = [f'n_{int(net)}' for net in record.get('real_outputs', [])] + [
                str(net) for net in record.get('repaired_output_nets', [])
            ]
            for idx, pin_name in enumerate(output_pins):
                if idx >= len(actual_outputs):
                    continue
                pin_to_net[str(pin_name)] = str(actual_outputs[idx])
                pin_role[str(pin_name)] = 2

    cell_summary = {}
    for cell_name, roll in sorted(cell_rollup.items()):
        count = int(roll['count'])
        cell_summary[cell_name] = {
            'count': count,
            'avg_incident_nets': roll['incident_nets'] / count if count else 0.0,
            'avg_required_pins': roll['required_pins'] / count if count else 0.0,
            'avg_real_inputs': roll['real_inputs'] / count if count else 0.0,
            'avg_reused_inputs': roll['reused_inputs'] / count if count else 0.0,
            'avg_helper_inputs': roll['helper_inputs'] / count if count else 0.0,
            'avg_synthetic_inputs': roll['synthetic_inputs'] / count if count else 0.0,
            'avg_real_outputs': roll['real_outputs'] / count if count else 0.0,
            'avg_unconnected_outputs': roll['unconnected_outputs'] / count if count else 0.0,
        }

    return gate_records, driver_count, load_count, warnings, used_specs, dict(issue_counter), cell_summary


def compute_graph_netlist_quality(
    graph: Data,
    gate_records: List[Dict],
    driver_count: Dict[int, int],
    load_count: Dict[int, int],
    issue_counter: Dict[str, int],
    label_to_cell: Dict[int, str],
    net_id: int,
    liberty_specs: Dict[str, Dict[str, List[str]]] | None,
    reference_gate_profile: Dict | None = None,
) -> Dict:
    x = graph.x.reshape(-1).to(torch.long)
    n = int(graph.num_nodes)
    edges = set()
    for src, dst in graph.edge_index.t().tolist():
        if src == dst:
            continue
        key = (int(src), int(dst)) if int(src) < int(dst) else (int(dst), int(src))
        edges.add(key)
    adj = [set() for _ in range(n)]
    for src, dst in edges:
        adj[src].add(dst)
        adj[dst].add(src)
    seen = set()
    comp_sizes = []
    for idx in range(n):
        if idx in seen:
            continue
        stack = [idx]
        seen.add(idx)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nbr in adj[cur]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        comp_sizes.append(size)
    gate_nodes = [idx for idx in range(n) if int(x[idx].item()) != int(net_id)]
    net_nodes = [idx for idx in range(n) if int(x[idx].item()) == int(net_id)]
    gate_type_counts = Counter()
    missing_specs = 0
    required_input_pins = 0
    required_output_pins = 0
    for idx in gate_nodes:
        cell_name = label_to_cell.get(int(x[idx].item()), f'LABEL_{int(x[idx].item())}')
        gate_type_counts[cell_name] += 1
        spec = _cell_pin_spec(cell_name, liberty_specs)
        if bool(spec.get('fallback', False)) and liberty_specs is not None:
            missing_specs += 1
        required_input_pins += len(spec.get('inputs', []))
        required_output_pins += len(spec.get('outputs', []))

    total_gates = float(sum(gate_type_counts.values()))
    gate_type_tv = None
    weighted_type_coverage_gap = None
    if reference_gate_profile and total_gates > 0:
        ref_probs = {str(k): float(v) for k, v in dict(reference_gate_profile.get('probabilities', {})).items()}
        probs = {cell: count / total_gates for cell, count in gate_type_counts.items()}
        all_types = set(ref_probs) | set(probs)
        gate_type_tv = 0.5 * sum(abs(ref_probs.get(cell, 0.0) - probs.get(cell, 0.0)) for cell in all_types)
        weighted_cov = sum(ref_probs.get(cell, 0.0) for cell in gate_type_counts if cell in ref_probs)
        weighted_type_coverage_gap = 1.0 - weighted_cov

    driver_conflict_nets = sum(1 for count in driver_count.values() if int(count) > 1)
    floating_input_nets = sum(1 for net in net_nodes if driver_count.get(net, 0) == 0 and load_count.get(net, 0) > 0)
    dangling_output_nets = sum(1 for net in net_nodes if driver_count.get(net, 0) > 0 and load_count.get(net, 0) == 0)
    connected_nets = sum(1 for net in net_nodes if driver_count.get(net, 0) > 0 and load_count.get(net, 0) > 0)
    pin_legal_connections = 0
    real_pin_connections = 0
    reused_input_ports = 0
    synthetic_input_ports = 0
    helper_input_ports = 0
    repaired_output_ports = 0
    total_required_ports_raw = required_input_pins + required_output_pins
    pin_expected_connections = max(1, total_required_ports_raw)
    for record in gate_records:
        real_inputs = len(record.get('real_inputs', []))
        reused_inputs = len(record.get('reused_inputs', []))
        synthetic_inputs = len(record.get('synthetic_inputs', []))
        helper_inputs = len(record.get('helper_inputs', []))
        real_outputs = len(record.get('real_outputs', []))
        repaired_outputs = len(record.get('repaired_output_nets', []))

        real_pin_connections += real_inputs + real_outputs
        reused_input_ports += reused_inputs
        synthetic_input_ports += synthetic_inputs
        helper_input_ports += helper_inputs
        repaired_output_ports += repaired_outputs

        # Backward-compatible proxy: counts legalized connections as pin coverage.
        pin_legal_connections += (
            real_inputs
            + reused_inputs
            + synthetic_inputs
            + helper_inputs
            + real_outputs
            + repaired_outputs
        )

    role_driven_gate_count = sum(1 for rec in gate_records if rec.get('role_driven'))
    heuristic_fallback_gate_count = max(0, len(gate_records) - role_driven_gate_count)

    synthetic_ports = synthetic_input_ports + repaired_output_ports
    repair_ports = synthetic_input_ports + reused_input_ports + helper_input_ports + repaired_output_ports
    topology_purity = max(0.0, 1.0 - (synthetic_ports / pin_expected_connections))
    topology_repair_free_ratio = max(0.0, 1.0 - (repair_ports / pin_expected_connections))
    real_pin_connection_ratio = min(1.0, real_pin_connections / pin_expected_connections)

    return {
        'graph': {
            'node_count': int(n),
            'edge_count': int(len(edges)),
            'component_count': int(len(comp_sizes)),
            'largest_component_ratio': float((max(comp_sizes) / n) if n else 0.0),
            'avg_degree': float((2.0 * len(edges) / n) if n else 0.0),
            'gate_count': int(len(gate_nodes)),
            'net_count': int(len(net_nodes)),
            'used_gate_type_count': int(len(gate_type_counts)),
            'gate_type_tv': None if gate_type_tv is None else float(gate_type_tv),
            'weighted_type_coverage_gap': None if weighted_type_coverage_gap is None else float(weighted_type_coverage_gap),
            'top_gate_types': {str(k): int(v) for k, v in gate_type_counts.most_common(20)},
        },
        'netlist': {
            'driver_conflict_nets': int(driver_conflict_nets),
            'floating_input_nets': int(floating_input_nets),
            'dangling_output_nets': int(dangling_output_nets),
            'connected_internal_nets': int(connected_nets),
            'synthetic_inputs': int(issue_counter.get('synthetic_inputs', 0)),
            'synthetic_input_connections': int(issue_counter.get('synthetic_input_connections', 0)),
            'missing_required_input_before_pool': int(issue_counter.get('missing_required_input_before_pool', 0)),
            'missing_required_input_after_pool': int(issue_counter.get('missing_required_input_after_pool', 0)),
            'created_input_pool_nets': int(issue_counter.get('created_input_pool_nets', 0)),
            'input_pool_connections': int(issue_counter.get('input_pool_connections', 0)),
            'max_input_pool_fanout': int(issue_counter.get('max_input_pool_fanout', 0)),
            'avg_input_pool_fanout': float(issue_counter.get('avg_input_pool_fanout', 0.0)),
            'reused_inputs': int(issue_counter.get('reused_inputs', 0)),
            'repaired_output_nets': int(issue_counter.get('repaired_output_nets', 0)),
            'synthetic_output_nets': int(issue_counter.get('synthetic_output_nets', 0)),
            'missing_input_pins': int(issue_counter.get('missing_input_pins', 0)),
            'missing_output_pins': int(issue_counter.get('missing_output_pins', 0)),
            'missing_optional_output_pins': int(issue_counter.get('missing_optional_output_pins', 0)),
            'missing_all_output_count': int(issue_counter.get('missing_all_output_count', 0)),
            'multi_driver_net_count_before': int(issue_counter.get('multi_driver_net_count_before', 0)),
            'multi_driver_net_count_after': int(issue_counter.get('multi_driver_net_count_after', 0)),
            'multi_driver_splits': int(issue_counter.get('multi_driver_splits', 0)),
            'role_realizer_output_conflict_reassigned': int(issue_counter.get('role_realizer_output_conflict_reassigned', 0)),
            'helper_input_stages': int(issue_counter.get('helper_input_stages', 0)),
            'dropped_extra_incident': int(issue_counter.get('dropped_extra_incident', 0)),
            'missing_liberty_cell_specs': int(missing_specs),
            'exporter_role_driven_gate_count': int(role_driven_gate_count),
            'exporter_heuristic_fallback_gate_count': int(heuristic_fallback_gate_count),
            'required_input_pins': int(required_input_pins),
            'required_output_pins': int(required_output_pins),
            'total_required_ports': int(total_required_ports_raw),
            'real_pin_connections': int(real_pin_connections),
            'synthetic_ports': int(synthetic_ports),
            'repair_ports': int(repair_ports),
            'topology_purity': float(topology_purity),
            'topology_repair_free_ratio': float(topology_repair_free_ratio),
            'real_pin_connection_ratio': float(real_pin_connection_ratio),
            'pin_legal_connection_ratio_proxy': float(min(1.0, pin_legal_connections / pin_expected_connections)),
            'fanout_histogram': {str(k): int(v) for k, v in sorted(Counter(load_count.values()).items())},
            'driver_count_histogram': {str(k): int(v) for k, v in sorted(Counter(driver_count.values()).items())},
        },
        'quality_flags': {
            'has_driver_conflict': bool(driver_conflict_nets > 0),
            'has_synthetic_inputs': bool(issue_counter.get('synthetic_inputs', 0) > 0),
            'has_reused_inputs': bool(issue_counter.get('reused_inputs', 0) > 0),
            'has_repaired_outputs': bool(issue_counter.get('repaired_output_nets', 0) > 0),
            'low_lcc': bool(((max(comp_sizes) / n) if n else 0.0) < 0.90),
            'low_topology_purity': bool(topology_purity < 0.95),
            'low_topology_repair_free_ratio': bool(topology_repair_free_ratio < 0.90),
        },
    }


def emit_verilog(
    graph: Data,
    gate_records: List[Dict],
    driver_count: Dict[int, int],
    load_count: Dict[int, int],
    used_specs: Dict[str, Dict[str, int]],
    output_path: str | Path,
    module_name: str,
    net_id: int,
    emit_blackboxes: bool = False,
    extra_output_nets: List[int] | None = None,
    extra_output_wires: List[str] | None = None,
    keep_debug: bool = False,
    keep_instances: bool = False,
    export_synthetic_outputs_as_ports: bool = False,
    exporter_name: str = 'practical_legalized_exporter',
    exporter_header_note: str | None = None,
) -> Dict:
    x = graph.x.reshape(-1).to(torch.long)
    net_nodes = sorted([idx for idx in range(int(graph.num_nodes)) if int(x[idx].item()) == int(net_id)])

    primary_inputs = sorted([n for n in net_nodes if driver_count.get(n, 0) == 0 and load_count.get(n, 0) > 0])
    primary_outputs = sorted([n for n in net_nodes if driver_count.get(n, 0) > 0 and load_count.get(n, 0) == 0])
    internal_nets = sorted([n for n in net_nodes if n not in primary_inputs and n not in primary_outputs])

    requested_extra_output_nets = sorted(
        {int(n) for n in (extra_output_nets or []) if int(n) in set(net_nodes)}
    )
    extra_output_nets = sorted(
        set(requested_extra_output_nets) - set(primary_inputs) - set(primary_outputs)
    )

    synthetic_inputs = sorted({name for record in gate_records for name in record['synthetic_inputs']})
    helper_wires = sorted({item['out'] for record in gate_records for item in record.get('helper_inputs', [])})
    synthetic_output_wires = sorted({name for record in gate_records for name in record.get('repaired_output_nets', [])})
    synthetic_outputs = list(synthetic_output_wires)

    extra_output_wires = [str(w) for w in (extra_output_wires or []) if str(w)]
    extra_output_wires = sorted(set(extra_output_wires) - set(synthetic_outputs))
    legalize_stats: Counter[str] = Counter()

    def _record_pin_map(record: Dict) -> Dict[str, str]:
        spec = record['spec']
        input_pins = list(spec.get('inputs', []))
        output_pins = list(spec.get('outputs', []))
        pin_to_net = record.get('pin_to_net') if isinstance(record.get('pin_to_net'), dict) else {}
        all_inputs = [f'n_{net}' for net in record['real_inputs']] + [f'n_{net}' for net in record['reused_inputs']] + [item['out'] for item in record.get('helper_inputs', [])] + record['synthetic_inputs']
        all_outputs = [f'n_{net}' for net in record['real_outputs']] + record.get('repaired_output_nets', [])
        fmap: Dict[str, str] = {}
        for idx, pin in enumerate(input_pins):
            if idx < len(all_inputs):
                fmap[str(pin)] = str(all_inputs[idx])
        for idx, pin in enumerate(output_pins):
            if idx < len(all_outputs):
                fmap[str(pin)] = str(all_outputs[idx])
        for k, v in pin_to_net.items():
            if v:
                fmap[str(k)] = str(v)
        return fmap

    def _ensure_pin_to_net(record: Dict) -> Dict[str, str]:
        if not isinstance(record.get('pin_to_net'), dict):
            record['pin_to_net'] = {}
        return record['pin_to_net']

    pool_idx = 0

    def _new_input_pool() -> str:
        nonlocal pool_idx
        while True:
            name = f'pool_pi_legalize_{pool_idx}'
            pool_idx += 1
            if name not in synthetic_inputs:
                synthetic_inputs.append(name)
                return name

    def _input_sources(exclude: set[str] | None = None) -> List[str]:
        exclude = exclude or set()
        sources = [f'n_{n}' for n in primary_inputs] + list(synthetic_inputs)
        # Already-driven internal nets are also legal sources for input pins.
        driven = set()
        for rec in gate_records:
            fmap = _record_pin_map(rec)
            for pin in rec['spec'].get('outputs', []):
                if pin in fmap:
                    driven.add(str(fmap[pin]))
        sources += sorted(driven)
        out = []
        seen = set()
        for s in sources:
            if str(s) in exclude or str(s) in seen:
                continue
            seen.add(str(s))
            out.append(str(s))
        return out

    # Practical legalization is explicit and counted. It does not apply to the strict exporter.
    for record in gate_records:
        spec = record['spec']
        input_pins = [str(p) for p in spec.get('inputs', [])]
        output_pins = [str(p) for p in spec.get('outputs', [])]
        if not input_pins:
            continue
        pin_to_net = _ensure_pin_to_net(record)
        fmap = _record_pin_map(record)
        output_nets = {str(fmap[p]) for p in output_pins if p in fmap}
        used_inputs = []
        for pin in input_pins:
            if pin not in fmap:
                continue
            cur = str(fmap[pin])
            if cur in output_nets:
                candidates = _input_sources(exclude=output_nets | set(used_inputs))
                new_net = candidates[0] if candidates else _new_input_pool()
                pin_to_net[pin] = new_net
                fmap[pin] = new_net
                legalize_stats['self_loop_fixed_by_input_remap'] += 1
                cur = new_net
            if cur in used_inputs:
                candidates = _input_sources(exclude=output_nets | set(used_inputs))
                new_net = candidates[0] if candidates else _new_input_pool()
                pin_to_net[pin] = new_net
                fmap[pin] = new_net
                legalize_stats['same_net_reuse_reduced'] += 1
                cur = new_net
            used_inputs.append(cur)

    # Promote remaining undriven load nets to PI-like inputs.
    driver_names = set()
    load_names = set()
    for record in gate_records:
        fmap = _record_pin_map(record)
        for pin in record['spec'].get('outputs', []):
            if pin in fmap:
                driver_names.add(str(fmap[pin]))
        for pin in record['spec'].get('inputs', []):
            if pin in fmap:
                load_names.add(str(fmap[pin]))
    input_port_names = {f'n_{n}' for n in primary_inputs} | set(synthetic_inputs)
    for net_name in sorted(load_names - driver_names - input_port_names):
        if net_name.startswith('n_') and net_name[2:].isdigit():
            primary_inputs.append(int(net_name[2:]))
            legalize_stats['undriven_load_promoted_to_pi'] += 1
        elif not net_name.startswith(("1'b", "1’b")):
            synthetic_inputs.append(net_name)
            legalize_stats['undriven_load_promoted_to_pi'] += 1
    primary_inputs = sorted(set(primary_inputs))
    synthetic_inputs = sorted(set(synthetic_inputs))

    # Recompute residual quality counters after legalization.
    remap_driver_names = set()
    remap_load_names = set()
    self_loop_remaining = 0
    same_net_reuse_remaining = 0
    for record in gate_records:
        fmap = _record_pin_map(record)
        outs = {str(fmap[p]) for p in record['spec'].get('outputs', []) if p in fmap}
        ins = [str(fmap[p]) for p in record['spec'].get('inputs', []) if p in fmap]
        remap_driver_names |= outs
        remap_load_names |= set(ins)
        if set(ins) & outs:
            self_loop_remaining += 1
        if len(ins) != len(set(ins)):
            same_net_reuse_remaining += 1
    legalize_stats['self_loop_remaining'] = int(self_loop_remaining)
    legalize_stats['same_net_reuse_remaining'] = int(same_net_reuse_remaining)

    # Port directions must be derived from the final rendered pin map.  The
    # role-aware map can contain valid secondary outputs (for example FA/HA S
    # and DFF QN) that are not represented by the minimum-output driver_count
    # used during practical assignment.  Classifying ports from that earlier
    # count declared those internally driven nets as top-level inputs.
    def _numeric_net_ids(names: set[str]) -> set[int]:
        return {
            int(name[2:])
            for name in names
            if name.startswith('n_') and name[2:].isdigit() and int(name[2:]) in set(net_nodes)
        }

    provisional_primary_inputs = set(primary_inputs)
    final_driver_nets = _numeric_net_ids(remap_driver_names)
    final_load_nets = _numeric_net_ids(remap_load_names)
    primary_inputs = sorted(final_load_nets - final_driver_nets)
    primary_outputs = sorted(final_driver_nets - final_load_nets)
    extra_output_nets = sorted(
        set(requested_extra_output_nets) - set(primary_inputs) - set(primary_outputs)
    )
    internal_nets = sorted(
        n for n in net_nodes
        if n not in set(primary_inputs) and n not in set(primary_outputs)
    )
    legalize_stats['port_directions_recomputed_from_final_pin_map'] = int(
        len(provisional_primary_inputs.symmetric_difference(primary_inputs))
    )

    final_input_names = {f'n_{n}' for n in primary_inputs} | set(synthetic_inputs)
    driven_input_names = final_input_names & remap_driver_names
    if driven_input_names:
        raise RuntimeError(
            'final port classification left internally driven input ports: '
            + ', '.join(sorted(driven_input_names)[:20])
        )

    used_signal_names = set(remap_driver_names) | set(remap_load_names) | {f'n_{n}' for n in primary_inputs} | {f'n_{n}' for n in primary_outputs} | {f'n_{n}' for n in extra_output_nets}
    used_signal_names |= set(synthetic_inputs) | set(extra_output_wires) | set(synthetic_outputs)
    for record in gate_records:
        for helper in record.get('helper_inputs', []):
            used_signal_names.add(str(helper.get('src')))
            used_signal_names.add(str(helper.get('out')))

    lines: List[str] = []
    if exporter_header_note is None and exporter_name == 'practical_legalized_exporter':
        exporter_header_note = (
            'Practical exporter: role-driven pin assignment with legalization fallback; '
            'may use input pool, helper stages, repaired outputs, and deterministic split nets.'
        )
    lines.append('// Auto-generated simplified netlist from assembled circuit graph')
    lines.append(f'// exporter: {exporter_name}')
    lines.append('// Cell instance pin names come from Nangate45 Liberty when available.')
    if exporter_header_note:
        lines.append(f'// {exporter_header_note}')
    if any(rec.get('role_driven') for rec in gate_records):
        lines.append('// Net assignment uses edge_role/edge_pin_id pin-slot metadata when available.')
        lines.append('// Any synthetic split/helper nets below are legalization repairs and should be audited.')
    else:
        lines.append('// Net assignment is heuristic because edge_role/edge_pin_id metadata was not available.')
    lines.append('')

    for cell_name, spec in sorted(used_specs.items()):
        if cell_name in {'LOGIC0_X1', 'LOGIC1_X1'}:
            continue
        if not emit_blackboxes and not bool(spec.get('fallback', False)):
            continue
        input_pins = list(spec.get('inputs', []))
        output_pins = list(spec.get('outputs', []))
        pin_list = input_pins + output_pins
        lines.append('(* blackbox *)')
        lines.append(f'module {cell_name}(')
        lines.append('    ' + ', '.join(pin_list))
        lines.append(');')
        for pin in input_pins:
            lines.append(f'  input {pin};')
        for pin in output_pins:
            lines.append(f'  output {pin};')
        lines.append('endmodule')
        lines.append('')

    port_list = [f'n_{n}' for n in primary_inputs] + synthetic_inputs + [f'n_{n}' for n in primary_outputs] + [f'n_{n}' for n in extra_output_nets]
    if extra_output_wires:
        port_list += extra_output_wires
    if export_synthetic_outputs_as_ports and synthetic_outputs:
        port_list += synthetic_outputs
    lines.append(f'module {module_name}(' + (', '.join(port_list) if port_list else '') + ');')
    for net in primary_inputs:
        lines.append(f"  {'(* keep *) ' if keep_debug else ''}input n_{net};")
    for name in synthetic_inputs:
        lines.append(f"  {'(* keep *) ' if keep_debug else ''}input {name};")
    for net in primary_outputs:
        lines.append(f"  {'(* keep *) ' if keep_debug else ''}output n_{net};")
    for net in extra_output_nets:
        lines.append(f"  {'(* keep *) ' if keep_debug else ''}output n_{net};")
    for name in extra_output_wires:
        lines.append(f"  {'(* keep *) ' if keep_debug else ''}output {name};")
    if export_synthetic_outputs_as_ports:
        for name in synthetic_outputs:
            lines.append(f"  {'(* keep *) ' if keep_debug else ''}output {name};")
    for net in internal_nets:
        if int(net) in set(extra_output_nets):
            continue
        if f'n_{net}' not in used_signal_names:
            continue
        lines.append(f"  {'(* keep *) ' if keep_debug else ''}wire n_{net};")
    for name in helper_wires:
        if str(name) not in used_signal_names:
            continue
        lines.append(f"  {'(* keep *) ' if keep_debug else ''}wire {name};")
    for name in synthetic_output_wires:
        if str(name) in set(extra_output_wires):
            continue
        if export_synthetic_outputs_as_ports and str(name) in set(synthetic_outputs):
            continue
        if str(name) not in used_signal_names:
            continue
        lines.append(f"  {'(* keep *) ' if keep_debug else ''}wire {name};")

    instance_lines: List[str] = []
    for record in gate_records:
        gate_idx = int(record['gate_idx'])
        cell_name = record['cell_name']
        if cell_name.startswith('FILLCELL_') or cell_name.startswith('ANTENNA_'):
            continue
        if cell_name == 'LOGIC0_X1':
            for target in record['real_outputs']:
                instance_lines.append(f"  assign n_{target} = 1'b0;")
            continue
        if cell_name == 'LOGIC1_X1':
            for target in record['real_outputs']:
                instance_lines.append(f"  assign n_{target} = 1'b1;")
            continue

        spec = record['spec']
        conns = []
        for helper_idx, helper in enumerate(record.get('helper_inputs', [])):
            helper_spec = used_specs.get(helper['cell_name'], _fallback_pin_spec(helper['cell_name']))
            helper_input_pins = list(helper_spec.get('inputs', []))
            helper_output_pins = list(helper_spec.get('outputs', []))
            helper_conns = []
            if helper_input_pins:
                helper_conns.append(f".{helper_input_pins[0]}({helper['src']})")
            if helper_output_pins:
                helper_conns.append(f".{helper_output_pins[0]}({helper['out']})")
            prefix = '  (* keep *) ' if keep_instances else '  '
            instance_lines.append(prefix + f"{helper['cell_name']} u_hg_{gate_idx}_{helper_idx}(" + ', '.join(helper_conns) + ');')
        input_pins = list(spec.get('inputs', []))
        output_pins = list(spec.get('outputs', []))
        pin_to_net = record.get('pin_to_net') if isinstance(record.get('pin_to_net'), dict) else {}
        all_inputs = [f'n_{net}' for net in record['real_inputs']] + [f'n_{net}' for net in record['reused_inputs']] + [item['out'] for item in record.get('helper_inputs', [])] + record['synthetic_inputs']
        all_outputs = [f'n_{net}' for net in record['real_outputs']] + record.get('repaired_output_nets', [])
        fallback_map = {}
        for idx, pin in enumerate(input_pins):
            if idx < len(all_inputs):
                fallback_map[pin] = all_inputs[idx]
        for idx, pin in enumerate(output_pins):
            if idx < len(all_outputs):
                fallback_map[pin] = all_outputs[idx]

        if pin_to_net:
            for k, v in pin_to_net.items():
                if v:
                    fallback_map[str(k)] = str(v)

        for pin in input_pins:
            conns.append(f'.{pin}({fallback_map[pin]})' if pin in fallback_map else f'.{pin}()')
        for pin in output_pins:
            conns.append(f'.{pin}({fallback_map[pin]})' if pin in fallback_map else f'.{pin}()')
        prefix = '  (* keep *) ' if keep_instances else '  '
        instance_lines.append(prefix + f'{cell_name} u_{gate_idx}(' + ', '.join(conns) + ');')

    if internal_nets:
        lines.append('')
    lines.extend(instance_lines)
    lines.append('endmodule')
    lines.append('')

    Path(output_path).write_text('\n'.join(lines), encoding='utf-8')
    return {
        'module_name': module_name,
        'primary_input_count': len(primary_inputs),
        'primary_output_count': len(primary_outputs),
        'extra_output_count': len(extra_output_nets),
        'extra_output_wire_count': len(extra_output_wires),
        'synthetic_output_port_count': len(synthetic_outputs) if export_synthetic_outputs_as_ports else 0,
        'synthetic_input_count': len(synthetic_inputs),
        'helper_wire_count': len(helper_wires),
        'synthetic_output_wire_count': len(synthetic_output_wires),
        'internal_net_count': len(internal_nets),
        'primary_inputs': primary_inputs,
        'primary_outputs': primary_outputs,
        'extra_outputs': extra_output_nets,
        'extra_output_wires': extra_output_wires,
        'synthetic_inputs': synthetic_inputs,
        'synthetic_outputs': synthetic_outputs if export_synthetic_outputs_as_ports else [],
        'blackbox_modules_emitted': bool(emit_blackboxes),
        'exporter_name': str(exporter_name),
        'practical_legalization': {str(k): int(v) for k, v in legalize_stats.items()},
    }


def run_yosys_check(
    verilog_path: str | Path,
    module_name: str,
    output_dir: str | Path,
    liberty_path: str | Path | None = None,
    yosys_bin: str = 'yosys',
) -> Dict:
    log_path = Path(output_dir) / 'yosys_check.log'
    script_parts = []
    if liberty_path is not None:
        script_parts.append(f'read_liberty -lib {Path(liberty_path)}')
    script_parts.extend([
        f'read_verilog {Path(verilog_path)}',
        f'hierarchy -check -top {module_name}',
        'proc',
        'check',
    ])
    command = [yosys_bin, '-p', '; '.join(script_parts)]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(proc.stdout, encoding='utf-8', errors='ignore')
    return {
        'command': command,
        'returncode': int(proc.returncode),
        'ok': bool(proc.returncode == 0),
        'log': str(log_path),
    }


def run_yosys_flow(
    verilog_path: str | Path,
    module_name: str,
    output_dir: str | Path,
    liberty_path: str | Path | None = None,
    yosys_bin: str = 'yosys',
) -> Dict:
    log_path = Path(output_dir) / 'yosys_flow.log'
    script_parts = []
    if liberty_path is not None:
        script_parts.append(f'read_liberty -lib {Path(liberty_path)}')
    script_parts.extend([
        f'read_verilog {Path(verilog_path)}',
        f'hierarchy -check -top {module_name}',
        'proc',
        'check',
        'tee -o /dev/null stat',
        f'synth -top {module_name}',
        'check',
        'tee -o /dev/null stat',
    ])
    command = [yosys_bin, '-p', '; '.join(script_parts)]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    out = proc.stdout
    log_path.write_text(out, encoding='utf-8', errors='ignore')

    def parse_stat(text: str) -> Dict[str, int]:
        cells = None
        wires = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('Number of cells:'):
                try:
                    cells = int(line.split(':', 1)[1].strip())
                except Exception:
                    pass
            if line.startswith('Number of wires:'):
                try:
                    wires = int(line.split(':', 1)[1].strip())
                except Exception:
                    pass
        return {
            'cell_count': -1 if cells is None else int(cells),
            'wire_count': -1 if wires is None else int(wires),
        }

    parts = out.split('===')
    stat = parse_stat(out)
    return {
        'command': command,
        'returncode': int(proc.returncode),
        'yosys_read_verilog_pass': bool(proc.returncode == 0),
        'yosys_check_pass': bool(proc.returncode == 0),
        'yosys_synth_pass': bool(proc.returncode == 0),
        'final_cell_count_after_yosys': int(stat['cell_count']),
        'final_wire_count_after_yosys': int(stat['wire_count']),
        'log': str(log_path),
    }


def _check_budget(name: str, value: int, limit: int | None, violations: List[str]) -> None:
    if limit is not None and int(value) > int(limit):
        violations.append(f'{name}={int(value)} exceeds limit {int(limit)}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Export an assembled circuit graph to a Nangate45-pin-aware Verilog netlist.')
    parser.add_argument('--graph', required=True)
    parser.add_argument(
        '--mapping',
        default=os.environ.get(
            'ELDA_CELL_MAPPING', str(Path(__file__).resolve().parents[1] / 'mapping.txt')
        ),
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--module-name', default='generated_circuit')
    parser.add_argument(
        '--net-id',
        type=int,
        default=0,
        help='Net node label. Default 0 derives it from mapping.txt as max gate label + 1.',
    )
    parser.add_argument('--liberty', default=None)
    parser.add_argument('--emit-blackboxes', action='store_true')
    parser.add_argument('--run-yosys', action='store_true')
    parser.add_argument('--yosys-bin', default='yosys')
    parser.add_argument('--reference-gate-profile', default=None)
    parser.add_argument('--max-synthetic-inputs', type=int, default=None)
    parser.add_argument('--max-repaired-output-nets', type=int, default=None)
    parser.add_argument('--max-helper-input-stages', type=int, default=None)
    parser.add_argument('--max-reused-inputs', type=int, default=None)
    parser.add_argument('--max-reused-source-fanout', type=int, default=None)
    parser.add_argument('--forbid-clock-like-net-as-data-fallback', action='store_true')
    parser.add_argument('--penalize-role-mixed-reuse', action='store_true')
    parser.add_argument('--report-repeated-input-signatures', action='store_true')
    parser.add_argument('--critical-optional-output-policy', default='report_only', choices=['report_only'])
    args = parser.parse_args()

    graph = torch.load(args.graph, map_location='cpu', weights_only=False)
    if not isinstance(graph, Data):
        raise TypeError('expected a torch_geometric.data.Data graph')

    label_to_cell = load_mapping(args.mapping)
    net_id = resolve_net_id(args.mapping, int(args.net_id))
    liberty_path = Path(args.liberty) if args.liberty else default_nangate45_liberty()
    liberty_specs = None
    if liberty_path is not None and liberty_path.exists():
        liberty_specs = parse_liberty_pin_specs(liberty_path)
    reference_gate_profile = None
    if args.reference_gate_profile:
        reference_gate_profile = json.loads(Path(args.reference_gate_profile).read_text(encoding='utf-8'))
    fallback_events: List[Dict[str, Any]] = []
    gate_records, driver_count, load_count, warnings, used_specs, issue_counter, cell_summary = choose_gate_nets(
        graph=graph,
        net_id=net_id,
        label_to_cell=label_to_cell,
        liberty_specs=liberty_specs,
        fallback_events=fallback_events,
        fallback_event_context={'assembly_mode': 'standalone_export'},
        max_reused_source_fanout=args.max_reused_source_fanout,
        forbid_clock_like_net_as_data_fallback=args.forbid_clock_like_net_as_data_fallback,
        penalize_role_mixed_reuse=args.penalize_role_mixed_reuse,
        report_repeated_input_signatures=args.report_repeated_input_signatures,
        critical_optional_output_policy=args.critical_optional_output_policy,
    )
    quality_metrics = compute_graph_netlist_quality(
        graph=graph,
        gate_records=gate_records,
        driver_count=driver_count,
        load_count=load_count,
        issue_counter=issue_counter,
        label_to_cell=label_to_cell,
        net_id=net_id,
        liberty_specs=liberty_specs,
        reference_gate_profile=reference_gate_profile,
    )
    budget_violations: List[str] = []
    _check_budget('synthetic_inputs', issue_counter.get('synthetic_inputs', 0), args.max_synthetic_inputs, budget_violations)
    _check_budget('repaired_output_nets', issue_counter.get('repaired_output_nets', 0), args.max_repaired_output_nets, budget_violations)
    _check_budget('helper_input_stages', issue_counter.get('helper_input_stages', 0), args.max_helper_input_stages, budget_violations)
    _check_budget('reused_inputs', issue_counter.get('reused_inputs', 0), args.max_reused_inputs, budget_violations)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    verilog_path = output_dir / f'{args.module_name}.v'
    stats = emit_verilog(
        graph=graph,
        gate_records=gate_records,
        driver_count=driver_count,
        load_count=load_count,
        used_specs=used_specs,
        output_path=verilog_path,
        module_name=args.module_name,
        net_id=net_id,
        emit_blackboxes=bool(args.emit_blackboxes or liberty_specs is None),
    )
    yosys_report = None
    if args.run_yosys:
        yosys_report = run_yosys_check(
            verilog_path=verilog_path,
            module_name=args.module_name,
            output_dir=output_dir,
            liberty_path=None if args.emit_blackboxes else liberty_path,
            yosys_bin=args.yosys_bin,
        )

    report = {
        'exporter_name': 'practical_legalized_exporter',
        'exporter_contract': {
            'allows_input_pool': True,
            'allows_repaired_output': True,
            'allows_helper_fallback': True,
            'purpose': 'final best-effort synthesizable Verilog export with legalization fallback',
        },
        'graph': str(Path(args.graph)),
        'mapping': str(Path(args.mapping)),
        'net_id': int(net_id),
        'liberty': str(liberty_path) if liberty_path is not None else None,
        'liberty_cell_specs_loaded': int(len(liberty_specs or {})),
        'verilog': str(verilog_path),
        'num_nodes': int(graph.num_nodes),
        'num_edges': int(graph.edge_index.size(1)),
        'num_gate_instances': len(gate_records),
        'warnings': warnings,
        'issue_counter': issue_counter,
        'fallback_events_path': str(output_dir / 'fallback_events.jsonl'),
        'fallback_event_count': len(fallback_events),
        'quality_metrics': quality_metrics,
        'budget_violations': budget_violations,
        'stats': stats,
        'yosys': yosys_report,
        'used_cell_types': sorted(used_specs.keys()),
        'cell_summary': cell_summary,
    }
    with (output_dir / 'fallback_events.jsonl').open('w', encoding='utf-8') as fh:
        for event in fallback_events:
            fh.write(json.dumps(event, sort_keys=True) + '\n')
    (output_dir / 'export_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
