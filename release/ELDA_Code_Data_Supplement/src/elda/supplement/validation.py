"""Structured validity report over a production V6.1 Source--Demand object."""

from __future__ import annotations

import copy
from collections import Counter

import networkx as nx

from elda.datamodules.data.circuit_source_net_v61_schema import count_bucket


SEQUENTIAL_PREFIXES = ("DFF", "SDFF", "LATCH", "DLH", "DLL")


def topology_audit(payload: dict) -> dict:
    """Exact topology audit used by the frozen free-run evaluation."""
    cell_types = {cell["cell_id"]: str(cell["cell_type"]) for cell in payload.get("cells", [])}
    sequential = {
        cell_id for cell_id, cell_type in cell_types.items()
        if cell_type.startswith(SEQUENTIAL_PREFIXES)
    }
    demands = {row["demand_id"]: row for row in payload.get("demands", [])}
    sources = {row["source_id"]: row for row in payload.get("sources", [])}
    edges = []
    unknown_refs = 0
    for source_net in payload.get("source_nets", []):
        source = sources.get(source_net.get("source_id"))
        if source is None:
            unknown_refs += 1
            continue
        source_cell = source.get("source_cell_id")
        if source_cell is None:
            continue
        for chunk in source_net.get("chunks", []):
            for demand_id in chunk.get("demand_ids", []):
                demand = demands.get(demand_id)
                if demand is None:
                    unknown_refs += 1
                    continue
                edges.append((source_cell, demand["load_cell_id"]))
    combinational = set(cell_types) - sequential
    graph = nx.DiGraph()
    graph.add_nodes_from(combinational)
    graph.add_edges_from((s, d) for s, d in edges if s in combinational and d in combinational)
    cyclic_sccs = [
        component for component in nx.strongly_connected_components(graph)
        if len(component) > 1 or graph.has_edge(next(iter(component)), next(iter(component)))
    ]
    return {
        "unknown_topology_reference_count": unknown_refs,
        "self_drive_count": sum(s == d for s, d in edges),
        "combinational_self_drive_count": sum(
            s == d for s, d in edges if s in combinational and d in combinational
        ),
        "combinational_cyclic_scc_count": len(cyclic_sccs),
        "combinational_largest_cyclic_scc_size": max(
            (len(component) for component in cyclic_sccs), default=0
        ),
        "combinational_cycle_free": not cyclic_sccs,
    }


def validate_payload(payload: dict, library: dict) -> dict:
    cells = payload.get("cells", [])
    demands = payload.get("demands", [])
    sources = payload.get("sources", [])
    source_nets = payload.get("source_nets", [])
    cell_by_id = {row.get("cell_id"): row for row in cells}
    demand_by_id = {row.get("demand_id"): row for row in demands}
    source_by_id = {row.get("source_id"): row for row in sources}

    object_errors = []
    for name, rows, key in (
        ("cell", cells, "cell_id"),
        ("demand", demands, "demand_id"),
        ("source", sources, "source_id"),
    ):
        ids = [row.get(key) for row in rows]
        if not rows:
            object_errors.append(f"empty_{name}_table")
        if len(ids) != len(set(ids)):
            object_errors.append(f"duplicate_{name}_id")
    source_load_ids = [row.get("source_id") for row in source_nets]
    if set(source_load_ids) != set(source_by_id) or len(source_load_ids) != len(set(source_load_ids)):
        object_errors.append("incomplete_or_duplicate_source_load_table")

    pin_errors = []
    for cell in cells:
        if cell.get("cell_type") not in library:
            pin_errors.append(f"unknown_cell_type:{cell.get('cell_id')}")
    for demand in demands:
        cell = cell_by_id.get(demand.get("load_cell_id"))
        if cell is None:
            pin_errors.append(f"unknown_load_cell:{demand.get('demand_id')}")
        elif demand.get("load_pin") not in library.get(cell.get("cell_type"), {}).get("inputs", []):
            pin_errors.append(f"invalid_load_pin:{demand.get('demand_id')}")

    assigned = []
    assigned_by_source = Counter()
    reference_errors = []
    for row in source_nets:
        source_id = row.get("source_id")
        if source_id not in source_by_id:
            reference_errors.append(f"unknown_source:{source_id}")
        ids = [d for chunk in row.get("chunks", []) for d in chunk.get("demand_ids", [])]
        if int(row.get("fanout", len(ids))) != len(ids):
            reference_errors.append(f"fanout_count_mismatch:{source_id}")
        assigned.extend(ids)
        assigned_by_source[source_id] += len(ids)
        for demand_id in ids:
            if demand_id not in demand_by_id:
                reference_errors.append(f"unknown_demand:{demand_id}")
    counts = Counter(assigned)
    cover_errors = []
    if set(counts) != set(demand_by_id):
        cover_errors.append("demand_set_not_fully_covered")
    if any(value != 1 for value in counts.values()):
        cover_errors.append("demand_not_assigned_exactly_once")

    capacity_errors = []
    for source_id, source in source_by_id.items():
        maximum = int(source.get("max_fanout", -1))
        used = int(source.get("used_fanout", -1))
        remaining = int(source.get("remaining_fanout", -1))
        actual = int(assigned_by_source[source_id])
        if used + remaining != maximum:
            capacity_errors.append(f"capacity_identity:{source_id}")
        if actual != used or actual > maximum:
            capacity_errors.append(f"realized_use:{source_id}")
        if source.get("fanout_bucket") != count_bucket(maximum):
            capacity_errors.append(f"fanout_bucket:{source_id}")

    source_errors = []
    interface_errors = []
    for source_id, source in source_by_id.items():
        kind = source.get("source_kind")
        if kind == "CELL_OUTPUT":
            cell = cell_by_id.get(source.get("source_cell_id"))
            if cell is None:
                source_errors.append(f"missing_source_cell:{source_id}")
            elif source.get("source_pin") not in library.get(cell.get("cell_type"), {}).get("outputs", []):
                source_errors.append(f"invalid_source_pin:{source_id}")
            if source.get("source_role") != "OUTPUT_PIN" or source.get("boundary_pin_id") is not None:
                source_errors.append(f"cell_output_semantics:{source_id}")
        elif kind == "BOUNDARY_SOURCE":
            if source.get("source_cell_id") is not None or source.get("source_role") != "BOUNDARY_PIN":
                source_errors.append(f"boundary_source_semantics:{source_id}")
            if source.get("source_pin") != "__BOUNDARY__" or not source.get("boundary_pin_id"):
                interface_errors.append(f"invalid_boundary_identity:{source_id}")
        elif kind == "CONSTANT_SOURCE":
            if source.get("source_cell_id") is not None or source.get("source_role") != "CONST_PIN":
                source_errors.append(f"constant_source_semantics:{source_id}")
        else:
            source_errors.append(f"invalid_source_kind:{source_id}")

    topology = topology_audit(payload)
    conditions = {
        "ObjectComplete": not object_errors and not reference_errors,
        "Pin": not pin_errors,
        "Cover": not cover_errors,
        "Cap": not capacity_errors,
        "SrcSem": not source_errors,
        "Iface": not interface_errors,
        "TopologySafe": (
            topology["unknown_topology_reference_count"] == 0
            and topology["self_drive_count"] == 0
            and topology["combinational_cyclic_scc_count"] == 0
        ),
    }
    errors = {
        "ObjectComplete": object_errors + reference_errors,
        "Pin": pin_errors,
        "Cover": cover_errors,
        "Cap": capacity_errors,
        "SrcSem": source_errors,
        "Iface": interface_errors,
        "TopologySafe": ([] if conditions["TopologySafe"] else ["self_drive_or_combinational_cycle"]),
    }
    return {
        "schema_version": payload.get("schema_version"),
        "conditions": conditions,
        "overall_valid": all(conditions.values()),
        "errors": errors,
        "counts": {
            "cells": len(cells),
            "demands": len(demands),
            "sources": len(sources),
            "source_load_records": len(source_nets),
        },
        "topology": topology,
    }


def negative_cases(good: dict) -> dict[str, tuple[dict, str]]:
    source_index = {row["source_id"]: index for index, row in enumerate(good["sources"])}
    net_index = {row["source_id"]: index for index, row in enumerate(good["source_nets"])}
    boundary_ids = [row["source_id"] for row in good["sources"] if row["source_kind"] == "BOUNDARY_SOURCE"]
    output_by_cell = {
        row["source_cell_id"]: row["source_id"]
        for row in good["sources"] if row["source_kind"] == "CELL_OUTPUT"
    }
    cases = {}
    row = copy.deepcopy(good)
    row["source_nets"] = row["source_nets"][:-1]
    cases["incomplete_object"] = (row, "ObjectComplete")
    row = copy.deepcopy(good)
    row["demands"][0]["load_pin"] = "INVALID_PIN"
    cases["invalid_pin"] = (row, "Pin")
    row = copy.deepcopy(good)
    assigned_boundary = boundary_ids[0]
    ni = net_index[assigned_boundary]
    si = source_index[assigned_boundary]
    row["source_nets"][ni]["chunks"][0]["demand_ids"] = []
    row["source_nets"][ni]["fanout"] = 0
    row["sources"][si]["used_fanout"] = 0
    row["sources"][si]["remaining_fanout"] = 1
    cases["missing_coverage"] = (row, "Cover")
    row = copy.deepcopy(good)
    row["sources"][si]["used_fanout"] = 0
    row["sources"][si]["remaining_fanout"] = 1
    cases["capacity_mismatch"] = (row, "Cap")
    row = copy.deepcopy(good)
    first_output = output_by_cell["CELL_0"]
    row["sources"][source_index[first_output]]["source_role"] = "BOUNDARY_PIN"
    cases["invalid_source_semantics"] = (row, "SrcSem")
    row = copy.deepcopy(good)
    row["sources"][si]["boundary_pin_id"] = None
    cases["invalid_interface"] = (row, "Iface")
    row = copy.deepcopy(good)
    for source_id, demand_ids in (
        (output_by_cell["CELL_0"], ["DEMAND_1"]),
        (output_by_cell["CELL_1"], ["DEMAND_0"]),
    ):
        source_pos = source_index[source_id]
        net_pos = net_index[source_id]
        row["sources"][source_pos].update(
            max_fanout=1, used_fanout=1, remaining_fanout=0, fanout_bucket="FANOUT_1"
        )
        row["source_nets"][net_pos]["fanout"] = 1
        row["source_nets"][net_pos]["chunks"][0]["demand_ids"] = demand_ids
    for source_id in boundary_ids[:2]:
        source_pos = source_index[source_id]
        net_pos = net_index[source_id]
        row["sources"][source_pos].update(
            max_fanout=0, used_fanout=0, remaining_fanout=0, fanout_bucket="FANOUT_0"
        )
        row["source_nets"][net_pos]["fanout"] = 0
        row["source_nets"][net_pos]["chunks"][0]["demand_ids"] = []
    cases["combinational_cycle"] = (row, "TopologySafe")
    return cases
