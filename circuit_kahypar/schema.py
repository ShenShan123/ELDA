from __future__ import annotations

import ast
import math
from typing import Dict, Iterable, List, Optional, Tuple

from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_isolated_nodes, remove_self_loops, to_undirected


DEFAULT_NET_ID = 96
DEFAULT_BOUNDARY_STUB_ID = 97
BOUNDARY_ROLE_NONE = -1
BOUNDARY_ROLE_AMBIG = 0
BOUNDARY_ROLE_DRIVER = 1
BOUNDARY_ROLE_LOAD = 2
BOUNDARY_ROLE_BIDIR = 3


def load_cell_to_label_mapping(path: str | Path) -> Dict[str, int]:
    text = Path(path).read_text(encoding="utf-8")
    _, rhs = text.split("=", 1)
    mapping = ast.literal_eval(rhs.strip())
    return {str(cell): int(label) for cell, label in mapping.items()}


def derive_special_node_ids(
    cell_mapping_path: str | Path | None = None,
    *,
    fallback_net_id: int = DEFAULT_NET_ID,
    fallback_boundary_stub_id: int = DEFAULT_BOUNDARY_STUB_ID,
) -> Tuple[int, int]:
    if cell_mapping_path is None:
        return int(fallback_net_id), int(fallback_boundary_stub_id)
    path = Path(cell_mapping_path)
    if not path.exists():
        return int(fallback_net_id), int(fallback_boundary_stub_id)
    mapping = load_cell_to_label_mapping(path)
    valid = [int(v) for v in mapping.values() if int(v) >= 0]
    if not valid:
        return int(fallback_net_id), int(fallback_boundary_stub_id)
    max_gate_label = max(valid)
    return int(max_gate_label + 1), int(max_gate_label + 2)


def load_label_to_cell_mapping(path: str) -> Dict[int, str]:
    mapping = load_cell_to_label_mapping(path)
    return {int(label): str(cell) for cell, label in mapping.items()}


def infer_cell_spec(cell_name: str) -> Dict[str, int]:
    name = cell_name.upper()
    if name in {"LOGIC0_X1", "LOGIC1_X1"}:
        return {"inputs": 0, "outputs": 1}
    if name.startswith(("INV_", "BUF_", "CLKBUF_")):
        return {"inputs": 1, "outputs": 1}
    if name.startswith(("TBUF_", "TINV_")):
        return {"inputs": 2, "outputs": 1}
    if name.startswith(("XOR2_", "XNOR2_")):
        return {"inputs": 2, "outputs": 1}
    if name.startswith(("AOI21_", "OAI21_")):
        return {"inputs": 3, "outputs": 1}
    if name.startswith(("AOI22_", "OAI22_")):
        return {"inputs": 4, "outputs": 1}
    if name.startswith("MUX2_"):
        return {"inputs": 3, "outputs": 1}
    if name.startswith("HA_"):
        return {"inputs": 2, "outputs": 2}
    if name.startswith("FA_"):
        return {"inputs": 3, "outputs": 2}
    if name.startswith("DFFSR_"):
        return {"inputs": 4, "outputs": 1}
    if name.startswith(("DFFR_", "DFFS_")):
        return {"inputs": 3, "outputs": 1}
    if name.startswith("DFF_"):
        return {"inputs": 2, "outputs": 1}
    if name.startswith(("FILLCELL_", "ANTENNA_")):
        return {"inputs": 0, "outputs": 0}
    if name == "UNKNOWN":
        return {"inputs": 6, "outputs": 1}
    for prefix in ("NAND", "NOR", "AND", "OR"):
        if name.startswith(prefix):
            remainder = name[len(prefix):]
            digits = []
            for ch in remainder:
                if ch.isdigit():
                    digits.append(ch)
                else:
                    break
            if digits:
                return {"inputs": int("".join(digits)), "outputs": 1}
    return {"inputs": 6, "outputs": 1}


def normalize_circuit_graph(
    data: Data,
    net_id: int = DEFAULT_NET_ID,
    remove_isolates: bool = False,
) -> Data:
    if not isinstance(data, Data):
        raise TypeError("expected torch_geometric.data.Data")
    if getattr(data, "x", None) is None or getattr(data, "edge_index", None) is None:
        raise ValueError("data.x and data.edge_index are required")

    x = data.x.reshape(-1).detach().cpu()
    if x.is_floating_point():
        x = x.round()
    x = x.to(torch.long)
    x[x < 0] = net_id

    edge_index = data.edge_index.detach().cpu().to(torch.long)

    in_role = getattr(data, 'edge_role', None)
    in_pin = getattr(data, 'edge_pin_id', None)
    role_values = (
        in_role.detach().cpu().reshape(-1).to(torch.long)
        if isinstance(in_role, torch.Tensor) and int(in_role.numel()) == int(edge_index.size(1))
        else None
    )
    pin_values = (
        in_pin.detach().cpu().reshape(-1).to(torch.long)
        if isinstance(in_pin, torch.Tensor) and int(in_pin.numel()) == int(edge_index.size(1))
        else None
    )
    if role_values is not None or pin_values is not None:
        if role_values is None:
            role_values = torch.zeros(edge_index.size(1), dtype=torch.long)
        if pin_values is None:
            pin_values = torch.full((edge_index.size(1),), -1, dtype=torch.long)
        edge_feat = torch.stack([role_values, pin_values + 1], dim=1)
    else:
        edge_feat = None

    edge_index = to_undirected(edge_index, num_nodes=int(x.numel()))
    edge_index, edge_feat = remove_self_loops(edge_index, edge_attr=edge_feat)
    if edge_feat is not None:
        edge_index, edge_feat = coalesce(edge_index, edge_feat, num_nodes=int(x.numel()), reduce='max')
    else:
        edge_index = coalesce(edge_index, num_nodes=int(x.numel()))

    edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)
    if remove_isolates:
        if edge_feat is not None:
            edge_index, edge_feat, mask = remove_isolated_nodes(
                edge_index,
                edge_attr=edge_feat,
                num_nodes=int(x.numel()),
            )
        else:
            edge_index, edge_attr, mask = remove_isolated_nodes(
                edge_index,
                edge_attr=edge_attr,
                num_nodes=int(x.numel()),
            )
        x = x[mask]

    out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=int(x.numel()))
    if edge_feat is not None:
        out.edge_role = edge_feat[:, 0].reshape(-1).to(torch.long)
        out.edge_pin_id = (edge_feat[:, 1].reshape(-1).to(torch.long) - 1)
    if hasattr(data, 'pin_id_to_name'):
        out.pin_id_to_name = list(getattr(data, 'pin_id_to_name'))
    for key in list(getattr(data, "keys", lambda: [])()):
        if str(key).startswith("schema_") and key not in {"schema_boundary_stub_roles"}:
            try:
                setattr(out, str(key), getattr(data, str(key)))
            except Exception:
                pass
    return out


def normalize_generated_partition_graph(
    data: Data,
    net_id: int = DEFAULT_NET_ID,
    boundary_stub_id: int = DEFAULT_BOUNDARY_STUB_ID,
) -> Data:
    out = normalize_circuit_graph(data, net_id=net_id, remove_isolates=False)
    x = out.x.reshape(-1)
    net_like = (x == net_id) | (x == boundary_stub_id)
    keep = net_like[out.edge_index[0]] ^ net_like[out.edge_index[1]]
    edge_index = out.edge_index[:, keep]
    edge_feat = None
    if hasattr(out, 'edge_role') or hasattr(out, 'edge_pin_id'):
        role_values = getattr(out, 'edge_role', None)
        pin_values = getattr(out, 'edge_pin_id', None)
        role_values = (
            role_values.reshape(-1).to(torch.long)
            if isinstance(role_values, torch.Tensor) and int(role_values.numel()) == int(out.edge_index.size(1))
            else None
        )
        pin_values = (
            pin_values.reshape(-1).to(torch.long)
            if isinstance(pin_values, torch.Tensor) and int(pin_values.numel()) == int(out.edge_index.size(1))
            else None
        )
        if role_values is not None or pin_values is not None:
            if role_values is None:
                role_values = torch.zeros(out.edge_index.size(1), dtype=torch.long)
            if pin_values is None:
                pin_values = torch.full((out.edge_index.size(1),), -1, dtype=torch.long)
            edge_feat = torch.stack([role_values, pin_values + 1], dim=1)
            edge_feat = edge_feat[keep]

    if edge_feat is not None:
        edge_index, edge_feat = coalesce(edge_index, edge_feat, num_nodes=int(x.numel()), reduce='max')
        edge_index, edge_feat, mask = remove_isolated_nodes(
            edge_index,
            edge_attr=edge_feat,
            num_nodes=int(x.numel()),
        )
    else:
        edge_index = coalesce(edge_index, num_nodes=int(x.numel()))
        edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)
        edge_index, edge_attr, mask = remove_isolated_nodes(
            edge_index,
            edge_attr=edge_attr,
            num_nodes=int(x.numel()),
        )
    x = x[mask]
    normalized = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=int(x.numel()))
    if edge_feat is not None:
        normalized.edge_role = edge_feat[:, 0].reshape(-1).to(torch.long)
        normalized.edge_pin_id = (edge_feat[:, 1].reshape(-1).to(torch.long) - 1)
    if hasattr(out, 'pin_id_to_name'):
        normalized.pin_id_to_name = list(getattr(out, 'pin_id_to_name'))
    for key in list(getattr(data, "keys", lambda: [])()):
        if str(key).startswith("schema_") and key not in {"schema_boundary_stub_roles"}:
            try:
                setattr(normalized, str(key), getattr(data, str(key)))
            except Exception:
                pass
    return normalized


def partition_pin_semantic_metrics(
    data: Data,
    label_to_cell: Dict[int, str],
    net_id: int = DEFAULT_NET_ID,
    boundary_stub_id: int = DEFAULT_BOUNDARY_STUB_ID,
) -> Dict[str, float]:
    x = data.x.reshape(-1).to(torch.long)
    degrees = torch.bincount(data.edge_index[0], minlength=int(x.numel()))
    gate_nodes = ((x != net_id) & (x != boundary_stub_id)).nonzero(as_tuple=False).reshape(-1).tolist()
    if not gate_nodes:
        return {
            "gate_pin_deficit": 0.0,
            "underconnected_gate_fraction": 0.0,
            "synthetic_input_proxy_ratio": 0.0,
            "required_input_pins": 0.0,
            "required_output_pins": 0.0,
            "required_total_pins": 0.0,
        }

    total_required_inputs = 0
    total_required_outputs = 0
    total_required_pins = 0
    total_input_deficit = 0
    total_pin_deficit = 0
    underconnected = 0
    for gate_idx in gate_nodes:
        label = int(x[gate_idx].item())
        cell_name = label_to_cell.get(label, f"LABEL_{label}")
        spec = infer_cell_spec(cell_name)
        req_inputs = int(spec["inputs"])
        req_outputs = int(spec["outputs"])
        req_total = req_inputs + req_outputs
        incident = int(degrees[gate_idx].item())
        total_required_inputs += req_inputs
        total_required_outputs += req_outputs
        total_required_pins += req_total
        pin_deficit = max(0, req_total - incident)
        input_deficit = max(0, req_inputs - max(0, incident - req_outputs))
        total_pin_deficit += pin_deficit
        total_input_deficit += input_deficit
        if pin_deficit > 0:
            underconnected += 1

    return {
        "gate_pin_deficit": float(total_pin_deficit / max(total_required_pins, 1)),
        "underconnected_gate_fraction": float(underconnected / max(len(gate_nodes), 1)),
        "synthetic_input_proxy_ratio": float(total_input_deficit / max(total_required_inputs, 1)),
        "required_input_pins": float(total_required_inputs),
        "required_output_pins": float(total_required_outputs),
        "required_total_pins": float(total_required_pins),
    }


def partition_graph_metrics(
    data: Data,
    net_id: int = DEFAULT_NET_ID,
    boundary_stub_id: int = DEFAULT_BOUNDARY_STUB_ID,
) -> Dict[str, int]:
    x = data.x.reshape(-1).to(torch.long)
    stub_mask = x == boundary_stub_id
    gate_mask = (x != net_id) & (x != boundary_stub_id)
    degrees = torch.bincount(data.edge_index[0], minlength=int(x.numel()))
    boundary_pin_count = int(degrees[stub_mask].sum().item())
    return {
        "num_nodes": int(x.numel()),
        "num_gates": int(gate_mask.sum().item()),
        "num_internal_nets": int((x == net_id).sum().item()),
        "boundary_stub_count": int(stub_mask.sum().item()),
        "boundary_pin_count": boundary_pin_count,
    }


def infer_boundary_stub_roles(
    data: Data,
    label_to_cell: Optional[Dict[int, str]] = None,
    net_id: int = DEFAULT_NET_ID,
    boundary_stub_id: int = DEFAULT_BOUNDARY_STUB_ID,
) -> Dict[str, object]:
    """Infer coarse boundary driver/load roles from local pin-capacity evidence.

    Existing partition graphs are undirected and do not carry original Verilog
    pin names.  This is therefore a compatibility-layer proxy, not a lossless
    netlist semantic recovery.  It assigns each gate up to its required output
    count by preferring incident nets/stubs with lower fanout; boundary stubs
    chosen as outputs are driver-facing, the remaining incident stubs are
    load-facing.  Conflicts become bidirectional/ambiguous instead of being
    silently forced.
    """
    x = data.x.reshape(-1).to(torch.long)
    degrees = torch.bincount(data.edge_index[0], minlength=int(x.numel()))

    existing_roles = getattr(data, "schema_boundary_stub_roles", None)
    if existing_roles is not None:
        try:
            roles = existing_roles.detach().cpu().reshape(-1).to(torch.long)
            if int(roles.numel()) == int(x.numel()):
                stub_nodes = (x == boundary_stub_id).nonzero(as_tuple=False).reshape(-1)
                stub_roles = roles[stub_nodes] if int(stub_nodes.numel()) else torch.empty((0,), dtype=torch.long)
                driver_count = int((stub_roles == BOUNDARY_ROLE_DRIVER).sum().item())
                load_count = int((stub_roles == BOUNDARY_ROLE_LOAD).sum().item())
                bidir_count = int((stub_roles == BOUNDARY_ROLE_BIDIR).sum().item())
                ambig_count = int((stub_roles == BOUNDARY_ROLE_AMBIG).sum().item())
                if int(stub_nodes.numel()) == driver_count + load_count + bidir_count + ambig_count:
                    return {
                        "stub_roles": roles,
                        "driver_stub_count": driver_count,
                        "load_stub_count": load_count,
                        "bidir_stub_count": bidir_count,
                        "ambig_stub_count": ambig_count,
                        "role_complete": int(ambig_count == 0),
                    }
        except Exception:
            pass

    roles = torch.full((int(x.numel()),), BOUNDARY_ROLE_NONE, dtype=torch.long)
    stub_nodes = (x == boundary_stub_id).nonzero(as_tuple=False).reshape(-1).tolist()
    for stub_idx in stub_nodes:
        roles[stub_idx] = BOUNDARY_ROLE_AMBIG

    if not stub_nodes:
        return {
            "stub_roles": roles,
            "driver_stub_count": 0,
            "load_stub_count": 0,
            "bidir_stub_count": 0,
            "ambig_stub_count": 0,
            "role_complete": 1,
        }

    neighbors: List[set[int]] = [set() for _ in range(int(x.numel()))]
    for src, dst in data.edge_index.t().tolist():
        if src == dst:
            continue
        neighbors[int(src)].add(int(dst))
        neighbors[int(dst)].add(int(src))

    role_votes: Dict[int, set[str]] = {int(stub): set() for stub in stub_nodes}
    gate_nodes = ((x != net_id) & (x != boundary_stub_id)).nonzero(as_tuple=False).reshape(-1).tolist()
    for gate_idx in gate_nodes:
        incident = sorted(int(n) for n in neighbors[int(gate_idx)] if int(x[n].item()) in {int(net_id), int(boundary_stub_id)})
        if not incident:
            continue
        label = int(x[gate_idx].item())
        cell_name = label_to_cell.get(label, f"LABEL_{label}") if label_to_cell else f"LABEL_{label}"
        spec = infer_cell_spec(cell_name)
        req_outputs = max(0, int(spec.get("outputs", 1)))
        if req_outputs <= 0:
            continue

        def output_key(node_idx: int) -> tuple:
            is_stub = int(x[node_idx].item()) == int(boundary_stub_id)
            return (
                int(degrees[node_idx].item()),
                0 if is_stub else 1,
                int(node_idx),
            )

        chosen_outputs = set(sorted(incident, key=output_key)[: min(req_outputs, len(incident))])
        for node_idx in incident:
            if int(x[node_idx].item()) != int(boundary_stub_id):
                continue
            role_votes[int(node_idx)].add("driver" if node_idx in chosen_outputs else "load")

    driver_count = 0
    load_count = 0
    bidir_count = 0
    ambig_count = 0
    for stub_idx in stub_nodes:
        votes = role_votes.get(int(stub_idx), set())
        if votes == {"driver"}:
            roles[int(stub_idx)] = BOUNDARY_ROLE_DRIVER
            driver_count += 1
        elif votes == {"load"}:
            roles[int(stub_idx)] = BOUNDARY_ROLE_LOAD
            load_count += 1
        elif votes == {"driver", "load"}:
            roles[int(stub_idx)] = BOUNDARY_ROLE_BIDIR
            bidir_count += 1
        else:
            roles[int(stub_idx)] = BOUNDARY_ROLE_AMBIG
            ambig_count += 1

    return {
        "stub_roles": roles,
        "driver_stub_count": int(driver_count),
        "load_stub_count": int(load_count),
        "bidir_stub_count": int(bidir_count),
        "ambig_stub_count": int(ambig_count),
        "role_complete": int(ambig_count == 0),
    }


def partition_schema_summary(
    data: Data,
    net_id: int = DEFAULT_NET_ID,
    boundary_stub_id: int = DEFAULT_BOUNDARY_STUB_ID,
    label_to_cell: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    x = data.x.reshape(-1).to(torch.long)
    gate_mask = (x != net_id) & (x != boundary_stub_id)
    stub_mask = x == boundary_stub_id
    net_mask = x == net_id

    gate_labels = x[gate_mask]
    if gate_labels.numel() > 0:
        gate_type_labels, gate_type_counts = torch.unique(gate_labels, sorted=True, return_counts=True)
    else:
        gate_type_labels = torch.empty((0,), dtype=torch.long)
        gate_type_counts = torch.empty((0,), dtype=torch.long)

    degrees = torch.bincount(data.edge_index[0], minlength=int(x.numel()))
    stub_degrees = degrees[stub_mask]
    stub_degree_one = int((stub_degrees == 1).sum().item())
    stub_degree_two = int((stub_degrees == 2).sum().item())
    stub_degree_ge_three = int((stub_degrees >= 3).sum().item())

    boundary_stub_count = int(stub_mask.sum().item())
    boundary_pin_count = int(stub_degrees.sum().item()) if stub_degrees.numel() > 0 else 0

    role_summary = infer_boundary_stub_roles(
        data,
        label_to_cell=label_to_cell,
        net_id=net_id,
        boundary_stub_id=boundary_stub_id,
    )
    return {
        "schema_gate_count": int(gate_mask.sum().item()),
        "schema_internal_net_count": int(net_mask.sum().item()),
        "schema_boundary_stub_count": int(boundary_stub_count),
        "schema_boundary_pin_count": int(boundary_pin_count),
        "schema_gate_type_unique_count": int(gate_type_labels.numel()),
        "schema_gate_type_labels": gate_type_labels.to(torch.long),
        "schema_gate_type_counts": gate_type_counts.to(torch.long),
        "schema_stub_degree_one_count": int(stub_degree_one),
        "schema_stub_degree_two_count": int(stub_degree_two),
        "schema_stub_degree_ge_three_count": int(stub_degree_ge_three),
        "schema_boundary_driver_stub_count": int(role_summary["driver_stub_count"]),
        "schema_boundary_load_stub_count": int(role_summary["load_stub_count"]),
        "schema_boundary_bidir_stub_count": int(role_summary["bidir_stub_count"]),
        "schema_boundary_ambig_stub_count": int(role_summary["ambig_stub_count"]),
        "schema_boundary_stub_roles": role_summary["stub_roles"].to(torch.long),
        "schema_role_complete": int(role_summary["role_complete"]),
    }


def build_bucket_spec(values: Iterable[int], num_buckets: int = 4) -> Dict[str, List[int]]:
    vals = sorted(int(v) for v in values if int(v) >= 0)
    if not vals:
        return {"upper_bounds": [0]}
    num_buckets = max(1, int(num_buckets))
    upper_bounds: List[int] = []
    for idx in range(num_buckets):
        q = (idx + 1) / num_buckets
        pos = min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))
        upper_bounds.append(int(vals[pos]))
    deduped: List[int] = []
    for value in upper_bounds:
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    if deduped[-1] != vals[-1]:
        deduped[-1] = vals[-1]
    return {"upper_bounds": deduped}


def bucket_value(value: int, spec: Dict[str, List[int]]) -> int:
    upper_bounds = spec["upper_bounds"]
    value = int(value)
    for idx, bound in enumerate(upper_bounds):
        if value <= bound:
            return idx
    return len(upper_bounds) - 1


def bucket_midpoint(spec: Dict[str, List[int]], bucket_idx: int) -> int:
    upper_bounds = spec["upper_bounds"]
    upper = upper_bounds[bucket_idx]
    lower = 0 if bucket_idx == 0 else upper_bounds[bucket_idx - 1] + 1
    return int(round((lower + upper) / 2))


INTERFACE_BUCKET_TO_SPEC = {
    "size_bucket": "partition_size",
    "stub_bucket": "boundary_stub_count",
    "pin_bucket": "boundary_pin_count",
    "pin_deficit_bucket": "gate_pin_deficit_scaled",
    "underconnected_bucket": "underconnected_gate_fraction_scaled",
    "synthetic_input_bucket": "synthetic_input_proxy_ratio_scaled",
}


def _default_interface_bucket_keys(meta: Optional[Dict] = None) -> List[str]:
    return list((meta or {}).get("interface_bucket_keys", ["size_bucket", "stub_bucket", "pin_bucket"]))


def interface_bucket_counts(meta: Dict) -> List[int]:
    return [len(meta["buckets"][INTERFACE_BUCKET_TO_SPEC[key]]["upper_bounds"]) for key in _default_interface_bucket_keys(meta)]


def encode_bucket_tuple(bucket_values: List[int], bucket_counts: List[int]) -> int:
    class_id = 0
    for value, count in zip(bucket_values, bucket_counts):
        class_id = class_id * int(count) + int(value)
    return int(class_id)


def decode_bucket_tuple(class_id: int, bucket_counts: List[int]) -> List[int]:
    rem = int(class_id)
    decoded = [0] * len(bucket_counts)
    for idx in range(len(bucket_counts) - 1, -1, -1):
        count = int(bucket_counts[idx])
        decoded[idx] = rem % count
        rem //= count
    return decoded


def decode_interface_class_meta(class_id: int, meta: Dict) -> Dict[str, int]:
    keys = _default_interface_bucket_keys(meta)
    counts = interface_bucket_counts(meta)
    values = decode_bucket_tuple(class_id, counts)
    return {key: int(value) for key, value in zip(keys, values)}


def encode_metrics_interface_class(metrics: Dict[str, float], meta: Dict) -> Tuple[int, Dict[str, int]]:
    keys = _default_interface_bucket_keys(meta)
    bucket_counts = interface_bucket_counts(meta)
    bucket_values = []
    named = {}
    for key in keys:
        if key not in metrics:
            raise KeyError(f'missing interface metric bucket key: {key}')
        value = int(metrics[key])
        named[key] = value
        bucket_values.append(value)
    return encode_bucket_tuple(bucket_values, bucket_counts), named


def encode_interface_class(
    size_bucket: int,
    stub_bucket: int,
    pin_bucket: int,
    stub_bucket_count: int,
    pin_bucket_count: int,
) -> int:
    return int(size_bucket * stub_bucket_count * pin_bucket_count + stub_bucket * pin_bucket_count + pin_bucket)


def decode_interface_class(
    class_id: int,
    stub_bucket_count: int,
    pin_bucket_count: int,
) -> Tuple[int, int, int]:
    class_id = int(class_id)
    size_bucket = class_id // (stub_bucket_count * pin_bucket_count)
    rem = class_id % (stub_bucket_count * pin_bucket_count)
    stub_bucket = rem // pin_bucket_count
    pin_bucket = rem % pin_bucket_count
    return size_bucket, stub_bucket, pin_bucket


def classify_partition_graph(
    data: Data,
    meta: Dict,
    label_to_cell: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    metrics = partition_graph_metrics(
        data,
        net_id=int(meta["net_id"]),
        boundary_stub_id=int(meta["boundary_stub_id"]),
    )
    if label_to_cell is not None:
        metrics.update(
            partition_pin_semantic_metrics(
                data,
                label_to_cell=label_to_cell,
                net_id=int(meta["net_id"]),
                boundary_stub_id=int(meta["boundary_stub_id"]),
            )
        )
    else:
        metrics.update(
            {
                "gate_pin_deficit": 0.0,
                "underconnected_gate_fraction": 0.0,
                "synthetic_input_proxy_ratio": 0.0,
            }
        )

    semantic_scale = int(meta.get("semantic_bucket_scale", 1000))
    metrics.update(
        {
            "size_bucket": bucket_value(metrics["num_nodes"], meta["buckets"]["partition_size"]),
            "stub_bucket": bucket_value(metrics["boundary_stub_count"], meta["buckets"]["boundary_stub_count"]),
            "pin_bucket": bucket_value(metrics["boundary_pin_count"], meta["buckets"]["boundary_pin_count"]),
            "pin_deficit_scaled": int(round(float(metrics["gate_pin_deficit"]) * semantic_scale)),
            "underconnected_fraction_scaled": int(round(float(metrics["underconnected_gate_fraction"]) * semantic_scale)),
            "synthetic_input_proxy_scaled": int(round(float(metrics["synthetic_input_proxy_ratio"]) * semantic_scale)),
        }
    )
    metrics.update(
        {
            "pin_deficit_bucket": bucket_value(metrics["pin_deficit_scaled"], meta["buckets"]["gate_pin_deficit_scaled"]),
            "underconnected_bucket": bucket_value(metrics["underconnected_fraction_scaled"], meta["buckets"]["underconnected_gate_fraction_scaled"]),
            "synthetic_input_bucket": bucket_value(metrics["synthetic_input_proxy_scaled"], meta["buckets"]["synthetic_input_proxy_ratio_scaled"]),
        }
    )
    interface_class, interface_buckets = encode_metrics_interface_class(metrics, meta)
    metrics["interface_class"] = int(interface_class)
    metrics.update(interface_buckets)
    return metrics


def interface_bucket_distance(class_a: int, class_b: int, meta: Dict) -> int:
    a = decode_interface_class_meta(class_a, meta)
    b = decode_interface_class_meta(class_b, meta)
    distance = 0
    for key in _default_interface_bucket_keys(meta):
        distance += abs(int(a[key]) - int(b[key]))
    return int(distance)
