from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import torch
from torch_geometric.data import Data


def count_bucket(value: int) -> str:
    if value <= 0:
        return "FANOUT_0"
    if value == 1:
        return "FANOUT_1"
    if value <= 4:
        return "FANOUT_2_4"
    if value <= 8:
        return "FANOUT_5_8"
    if value <= 16:
        return "FANOUT_9_16"
    return "FANOUT_17_PLUS"


def edge_rows(graph: Data) -> list[tuple[int, int, int, str]]:
    edge_index = graph.edge_index.to(torch.long)
    roles = getattr(graph, "edge_role", getattr(graph, "edge_attr", None))
    pins = getattr(graph, "edge_pin_id", None)
    pin_names = list(getattr(graph, "pin_id_to_name", []))
    if roles is None or pins is None:
        raise ValueError("V6.1 requires edge_role and edge_pin_id")
    if int(roles.numel()) != int(edge_index.size(1)) or int(pins.numel()) != int(edge_index.size(1)):
        raise ValueError("V6.1 edge metadata length mismatch")
    rows = []
    for index, (src, dst) in enumerate(edge_index.t().tolist()):
        pin_id = int(pins[index].item())
        pin = pin_names[pin_id] if 0 <= pin_id < len(pin_names) else "__INVALID_PIN__"
        rows.append((int(src), int(dst), int(roles[index].item()), str(pin)))
    return rows


def serialize_v61(
    graph: Data,
    *,
    net_id: int,
    boundary_stub_id: int,
    label_to_cell: dict[int, str],
    pin_specs: dict[str, Any],
    chunk_size: int = 32,
) -> dict:
    x = graph.x.reshape(-1).to(torch.long)
    gates = [
        index for index in range(int(x.numel()))
        if int(x[index].item()) not in {int(net_id), int(boundary_stub_id)}
    ]
    gate_to_dense = {gate: dense for dense, gate in enumerate(gates)}
    endpoints: dict[int, dict[str, list[dict]]] = defaultdict(lambda: {"drivers": [], "loads": []})

    for edge_ordinal, (src, dst, role, pin) in enumerate(edge_rows(graph)):
        if src not in gate_to_dense or not (0 <= dst < int(x.numel())):
            continue
        endpoint_type = int(x[dst].item())
        if endpoint_type not in {int(net_id), int(boundary_stub_id)}:
            continue
        endpoint_is_boundary_stub = endpoint_type == int(boundary_stub_id)
        if endpoint_is_boundary_stub:
            direction = "LOAD" if role == 1 else "DRIVER"
            endpoint_key = f"BOUNDARY_{direction}_{gate_to_dense[src]}_{pin}_{edge_ordinal}"
        else:
            endpoint_key = f"NET_{dst}"
        event = {
            "cell_id": f"CELL_{gate_to_dense[src]}",
            "cell_dense_id": gate_to_dense[src],
            "cell_type": label_to_cell.get(int(x[src].item()), f"LABEL_{int(x[src].item())}"),
            "pin": pin,
            "role": role,
            "original_endpoint_id": dst,
            "endpoint_key": endpoint_key,
            "endpoint_type": "BOUNDARY_STUB" if endpoint_type == int(boundary_stub_id) else "NET",
        }
        spec = pin_specs.get(event["cell_type"])
        output_pins = set(str(pin_name) for pin_name in getattr(spec, "outputs", [])) if spec is not None else set()
        if role == 1:
            endpoints[endpoint_key]["loads"].append(event)
        elif role == 2 or (role == 3 and pin in output_pins):
            endpoints[endpoint_key]["drivers"].append(event)

    cells = [
        {
            "cell_id": f"CELL_{dense}",
            "cell_dense_id": dense,
            "cell_type": label_to_cell.get(int(x[gate].item()), f"LABEL_{int(x[gate].item())}"),
        }
        for dense, gate in enumerate(gates)
    ]
    demands = []
    sources = []
    source_nets = []
    duplicate_driver_endpoint_count = 0

    for endpoint in sorted(endpoints):
        drivers = sorted(
            endpoints[endpoint]["drivers"],
            key=lambda row: (row["role"] != 2, row["cell_dense_id"], row["pin"]),
        )
        loads = sorted(
            endpoints[endpoint]["loads"],
            key=lambda row: (row["cell_dense_id"], row["pin"]),
        )
        duplicate_driver_endpoint_count += max(0, len(drivers) - 1)
        driver = drivers[0] if drivers else None
        if driver is None:
            source_id = f"BOUNDARY_SOURCE_{endpoint}"
            source = {
                "source_id": source_id,
                "source_kind": "BOUNDARY_SOURCE",
                "source_cell_id": None,
                "source_pin": "__BOUNDARY__",
                "boundary_pin_id": f"BOUNDARY_PIN_{endpoint}",
                "source_role": "BOUNDARY_PIN",
                "original_endpoint_id": loads[0]["original_endpoint_id"] if loads else None,
                "original_endpoint_key": endpoint,
                "endpoint_type": loads[0]["endpoint_type"] if loads else "NET",
            }
        else:
            source_id = f"CELL_SOURCE_{driver['cell_dense_id']}_{driver['pin']}_{endpoint}"
            source = {
                "source_id": source_id,
                "source_kind": "CELL_OUTPUT",
                "source_cell_id": driver["cell_id"],
                "source_pin": driver["pin"],
                "boundary_pin_id": None,
                "source_role": "OUTPUT_PIN",
                "original_endpoint_id": driver["original_endpoint_id"],
                "original_endpoint_key": endpoint,
                "endpoint_type": driver["endpoint_type"],
            }

        source_demands = []
        for load in loads:
            demand_id = f"DEMAND_{len(demands)}"
            demand = {
                "demand_id": demand_id,
                "load_cell_id": load["cell_id"],
                "load_pin": load["pin"],
                "source_id": source_id,
            }
            demands.append(demand)
            source_demands.append(demand_id)

        fanout = len(source_demands)
        source.update({
            "fanout_bucket": count_bucket(fanout),
            "max_fanout": fanout,
            "used_fanout": fanout,
            "remaining_fanout": 0,
        })
        sources.append(source)
        chunks = [
            {
                "source_id": source_id,
                "chunk_id": chunk_id,
                "demand_ids": source_demands[start:start + chunk_size],
            }
            for chunk_id, start in enumerate(range(0, len(source_demands), chunk_size))
        ]
        source_nets.append({
            "source_id": source_id,
            "fanout": fanout,
            "chunks": chunks,
        })

    required_slots = []
    for cell in cells:
        spec = pin_specs.get(cell["cell_type"])
        inputs = list(getattr(spec, "inputs", [])) if spec is not None else []
        required_slots.extend(
            (cell["cell_id"], str(pin))
            for pin in inputs
        )
    emitted = [
        demand_id
        for source_net in source_nets
        for chunk in source_net["chunks"]
        for demand_id in chunk["demand_ids"]
    ]
    emitted_counts = Counter(emitted)
    demand_ids = {demand["demand_id"] for demand in demands}
    covered_slots = {(demand["load_cell_id"], demand["load_pin"]) for demand in demands}
    return {
        "schema_version": "source_net_v6_1_full_load_per_source_budget_pin_slot_v1",
        "cells": cells,
        "demands": demands,
        "sources": sources,
        "source_nets": source_nets,
        "invariants": {
            "required_pin_slot_count": len(required_slots),
            "covered_pin_slot_count": len(set(required_slots) & covered_slots),
            "missing_required_pin_slots": sorted(set(required_slots) - covered_slots),
            "extra_or_nonrequired_pin_slots": sorted(covered_slots - set(required_slots)),
            "duplicate_required_slot_count": len(demands) - len(covered_slots),
            "source_net_non_emitted_demand_count": len(demand_ids - set(emitted)),
            "source_net_duplicate_demand_count": sum(max(0, count - 1) for count in emitted_counts.values()),
            "source_net_unknown_demand_count": len(set(emitted) - demand_ids),
            "source_net_full_load_coverage": (
                set(emitted) == demand_ids
                and all(count == 1 for count in emitted_counts.values())
            ),
            "fanout_budget_violation_count": sum(
                source["used_fanout"] > source["max_fanout"] for source in sources
            ),
            "boundary_source_identity_count": sum(
                source["source_kind"] == "BOUNDARY_SOURCE" for source in sources
            ),
            "duplicate_driver_endpoint_count": duplicate_driver_endpoint_count,
        },
        "pin_id_to_name": list(getattr(graph, "pin_id_to_name", [])),
    }


def decode_v61(payload: dict, *, net_id: int, boundary_stub_id: int, cell_to_label: dict[str, int]) -> Data:
    cells = list(payload["cells"])
    demands = {demand["demand_id"]: demand for demand in payload["demands"]}
    sources = {source["source_id"]: source for source in payload["sources"]}
    cell_index = {cell["cell_id"]: index for index, cell in enumerate(cells)}
    labels = [int(cell_to_label[cell["cell_type"]]) for cell in cells]
    pin_names = sorted({
        str(demand["load_pin"]) for demand in demands.values()
    } | {
        str(source["source_pin"]) for source in sources.values()
    } | {"__BOUNDARY__"})
    pin_to_id = {pin: index for index, pin in enumerate(pin_names)}
    edges = []
    roles = []
    pins = []

    source_node = {}
    for source_id, source in sources.items():
        node = len(labels)
        source_node[source_id] = node
        labels.append(int(boundary_stub_id) if source["endpoint_type"] == "BOUNDARY_STUB" else int(net_id))
        if source["source_kind"] == "CELL_OUTPUT":
            edges.append((cell_index[source["source_cell_id"]], node))
            roles.append(2)
            pins.append(pin_to_id[source["source_pin"]])

    for source_net in payload["source_nets"]:
        source_id = source_net["source_id"]
        for chunk in source_net["chunks"]:
            for demand_id in chunk["demand_ids"]:
                demand = demands[demand_id]
                edges.append((cell_index[demand["load_cell_id"]], source_node[source_id]))
                roles.append(1)
                pins.append(pin_to_id[demand["load_pin"]])

    graph = Data(
        x=torch.tensor(labels, dtype=torch.long).reshape(-1, 1),
        edge_index=(
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges else torch.empty((2, 0), dtype=torch.long)
        ),
    )
    graph.edge_role = torch.tensor(roles, dtype=torch.long)
    graph.edge_attr = graph.edge_role.clone()
    graph.edge_pin_id = torch.tensor(pins, dtype=torch.long)
    graph.pin_id_to_name = pin_names
    graph.source_net_v61_schema_version = payload["schema_version"]
    return graph


def tokenize_v61_payload(payload: dict, variant: str = "full") -> list[str]:
    """Canonical production-token draft used for length and grammar audits."""
    if variant not in {"full", "compact_budget", "compact_load"}:
        raise ValueError(f"unknown V6.1 tokenization variant: {variant}")
    tokens = ["SOS", "CELL_SECTION_BEGIN"]
    for cell in payload["cells"]:
        tokens.extend([
            "CELL_BEGIN", "CELL_ID", cell["cell_id"],
            "CELL_TYPE", cell["cell_type"], "CELL_END",
        ])
    tokens.append("CELL_SECTION_END")

    tokens.append("DEMAND_SECTION_BEGIN")
    for demand in payload["demands"]:
        tokens.extend([
            "DEMAND_BEGIN", "DEMAND_ID", demand["demand_id"],
            "LOAD_CELL", demand["load_cell_id"],
            "LOAD_PIN", demand["load_pin"], "DEMAND_END",
        ])
    tokens.append("DEMAND_SECTION_END")

    tokens.append("SOURCE_SECTION_BEGIN")
    for source in payload["sources"]:
        tokens.extend([
            "SOURCE_BEGIN", "SOURCE_ID", source["source_id"],
            "SOURCE_KIND", source["source_kind"],
            "SOURCE_CELL", str(source["source_cell_id"] or "NONE"),
            "SOURCE_PIN", source["source_pin"],
            "BOUNDARY_PIN_ID", str(source["boundary_pin_id"] or "NONE"),
            "SOURCE_ROLE", source["source_role"],
        ])
        if variant in {"compact_budget", "compact_load"}:
            tokens.extend([
                "FANOUT_BUCKET", source["fanout_bucket"],
                "MAX_FANOUT", str(source["max_fanout"]),
                "USED_FANOUT", str(source["used_fanout"]),
                "REMAINING_FANOUT", str(source["remaining_fanout"]),
            ])
        tokens.append("SOURCE_END")
    tokens.append("SOURCE_SECTION_END")

    if variant == "full":
        tokens.append("SOURCE_BUDGET_SECTION_BEGIN")
        for source in payload["sources"]:
            tokens.extend([
                "SOURCE_BUDGET_PER_SOURCE_BEGIN",
                "SOURCE_ID", source["source_id"],
                "SOURCE_KIND", source["source_kind"],
                "FANOUT_BUCKET", source["fanout_bucket"],
                "MAX_FANOUT", str(source["max_fanout"]),
                "USED_FANOUT", str(source["used_fanout"]),
                "REMAINING_FANOUT", str(source["remaining_fanout"]),
                "SOURCE_BUDGET_PER_SOURCE_END",
            ])
        tokens.append("SOURCE_BUDGET_SECTION_END")

    tokens.append("SOURCE_NET_SECTION_BEGIN")
    for source_net in payload["source_nets"]:
        if variant == "compact_load":
            demand_ids = [
                demand_id
                for chunk in source_net["chunks"]
                for demand_id in chunk["demand_ids"]
            ]
            tokens.extend([
                "SOURCE_LOAD_BEGIN",
                source_net["source_id"],
                "COUNT_TOKEN", str(len(demand_ids)),
                *demand_ids,
                "SOURCE_LOAD_END",
            ])
            continue
        tokens.extend(["SOURCE_NET_BEGIN", "SOURCE_ID", source_net["source_id"]])
        for chunk in source_net["chunks"]:
            tokens.extend([
                "SOURCE_NET_LOAD_BEGIN",
                "SOURCE_ID", source_net["source_id"],
                "CHUNK_ID", str(chunk["chunk_id"]),
                "CHUNK_SIZE", str(len(chunk["demand_ids"])),
            ])
            for demand_id in chunk["demand_ids"]:
                tokens.extend(["DEMAND_ID", demand_id])
            tokens.append("SOURCE_NET_LOAD_END")
        tokens.extend(["SOURCE_NET_END", "SOURCE_ID", source_net["source_id"]])
    tokens.extend(["SOURCE_NET_SECTION_END", "EOS"])
    return tokens
