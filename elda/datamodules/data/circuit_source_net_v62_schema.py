from __future__ import annotations

from copy import deepcopy
from collections import Counter
from typing import Any, Mapping

from .circuit_source_net_v61_schema import decode_v61, serialize_v61


SCHEMA_VERSION = "source_net_v6_2_profile_compact_load_v4"
FANOUT_BUCKETS = (
    "FANOUT_0", "FANOUT_1", "FANOUT_2_4", "FANOUT_5_8",
    "FANOUT_9_16", "FANOUT_17_PLUS",
)
PROFILE_BIN_NAMES = ("PROFILE_BIN_0", "PROFILE_BIN_1", "PROFILE_BIN_2", "PROFILE_BIN_3")


def _profile_bin(value: float, boundaries: list[float]) -> str:
    if len(boundaries) != 3:
        raise ValueError(f"V6.2_PROFILE requires exactly 3 frozen train boundaries, got {boundaries}")
    return PROFILE_BIN_NAMES[sum(float(value) > float(boundary) for boundary in boundaries)]


def build_audit_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Derive offline statistics that are never encoded in the V6.2 prefix."""
    cells = list(payload.get("cells", []))
    demands = list(payload.get("demands", []))
    sources = list(payload.get("sources", []))
    fanout_hist = Counter(str(row["fanout_bucket"]) for row in sources)

    # Materialized GC_FULL incidence has one source/net object per source,
    # one driver incidence for cell outputs, and one load incidence per demand.
    node_count = len(cells) + len(sources)
    incidence_count = len(demands) + sum(
        row.get("source_kind") == "CELL_OUTPUT" for row in sources
    )
    undirected_pairs = []
    cell_index = {row["cell_id"]: i for i, row in enumerate(cells)}
    source_index = {row["source_id"]: len(cells) + i for i, row in enumerate(sources)}
    demand_by_id = {row["demand_id"]: row for row in demands}
    for source in sources:
        source_node = source_index[source["source_id"]]
        if source.get("source_kind") == "CELL_OUTPUT":
            undirected_pairs.append((cell_index[source["source_cell_id"]], source_node))
    for source_net in payload.get("source_nets", []):
        source_node = source_index[source_net["source_id"]]
        for chunk in source_net.get("chunks", []):
            for demand_id in chunk.get("demand_ids", []):
                demand = demand_by_id[demand_id]
                undirected_pairs.append((source_node, cell_index[demand["load_cell_id"]]))

    parent = list(range(node_count))
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left
    for left, right in undirected_pairs:
        union(left, right)
    component_count = len({find(index) for index in range(node_count)}) if node_count else 0
    density = incidence_count / max(1, len(cells) * len(sources))
    return {
        "cell_count": len(cells),
        "source_count": len(sources),
        "demand_count": len(demands),
        "fanout_histogram": {key: int(fanout_hist.get(key, 0)) for key in FANOUT_BUCKETS},
        "component_count": component_count,
        "edge_density": float(density),
        "materialized_incidence_count": int(incidence_count),
    }


def build_control_profile(
    payload: Mapping[str, Any],
    *,
    profile_config: Mapping[str, Any],
    pin_specs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the finite, leakage-safe inference condition.

    Exact fanout histogram, component count and density remain in audit_profile
    and are deliberately excluded from this control prefix.
    """
    cells = list(payload.get("cells", []))
    demands = list(payload.get("demands", []))
    sources = list(payload.get("sources", []))
    cell_types = [str(row["cell_type"]) for row in cells]
    sequential_types = set(str(value) for value in profile_config.get("sequential_cell_types", []))
    rare_types = set(str(value) for value in profile_config.get("rare_cell_types", []))
    seq_ratio = sum(cell in sequential_types for cell in cell_types) / max(1, len(cell_types))
    arities = [len(getattr(pin_specs.get(cell), "inputs", [])) for cell in cell_types]
    pin_complexity = max(arities or [0])
    rare_cell_count = sum(cell in rare_types for cell in cell_types)
    rare_ratio = rare_cell_count / max(1, len(cell_types))
    boundaries = dict(profile_config.get("bucket_boundaries", {}))
    required = {"cell_count", "demand_count", "source_count", "seq_ratio", "pin_complexity"}
    if set(boundaries) < required:
        raise ValueError(f"V6.2_PROFILE config missing boundaries: {sorted(required - set(boundaries))}")
    return {
        "profile_mode": "PROFILE_COARSE",
        "cell_count_bin": _profile_bin(len(cells), boundaries["cell_count"]),
        "demand_count_bin": _profile_bin(len(demands), boundaries["demand_count"]),
        "source_count_bin": _profile_bin(len(sources), boundaries["source_count"]),
        "seq_ratio_bin": _profile_bin(seq_ratio, boundaries["seq_ratio"]),
        "pin_complexity_bin": _profile_bin(pin_complexity, boundaries["pin_complexity"]),
        "rare_cell_enriched": bool(
            rare_cell_count >= int(profile_config["rare_partition_min_count"])
            if profile_config.get("rare_partition_rule") == "rare_cell_count_at_least_min_count"
            else rare_ratio >= float(profile_config.get("rare_partition_ratio_threshold", 0.05))
        ),
        "has_mux": any("MUX" in cell for cell in cell_types),
        "has_aoi_oai": any(cell.startswith(("AOI", "OAI")) for cell in cell_types),
        "has_multi_output_cell": any(
            len(getattr(pin_specs.get(cell), "outputs", [])) > 1 for cell in cell_types
        ),
        "profile_config_version": str(profile_config.get("version", "")),
    }


def validate_audit_profile(payload: Mapping[str, Any]) -> None:
    profile = payload.get("audit_profile")
    if profile is None:
        return
    if not isinstance(profile, Mapping):
        raise ValueError("V6.2 audit_profile must be a mapping")
    actual = build_audit_profile(payload)
    for key in ("cell_count", "source_count", "demand_count"):
        if int(profile.get(key, -1)) != int(actual[key]):
            raise ValueError(f"V6.2 audit_profile mismatch for {key}")
    supplied_hist = {key: int(profile.get("fanout_histogram", {}).get(key, 0)) for key in FANOUT_BUCKETS}
    if supplied_hist != actual["fanout_histogram"]:
        raise ValueError("V6.2 audit_profile fanout histogram mismatch")


def serialize_v62(
    graph: Any,
    *,
    net_id: int,
    boundary_stub_id: int,
    label_to_cell: dict[int, str],
    pin_specs: dict[str, Any],
    profile_config: Mapping[str, Any],
    chunk_size: int = 32,
) -> dict:
    """Build V6.2 without changing any V6.1 object or assignment identity."""
    payload = deepcopy(serialize_v61(
        graph,
        net_id=net_id,
        boundary_stub_id=boundary_stub_id,
        label_to_cell=label_to_cell,
        pin_specs=pin_specs,
        chunk_size=chunk_size,
    ))
    payload["schema_version"] = SCHEMA_VERSION
    payload["control_profile"] = build_control_profile(
        payload, profile_config=profile_config, pin_specs=pin_specs
    )
    return payload


def decode_v62(payload: dict, *, net_id: int, boundary_stub_id: int, cell_to_label: dict[str, int]):
    graph = decode_v61(
        payload,
        net_id=net_id,
        boundary_stub_id=boundary_stub_id,
        cell_to_label=cell_to_label,
    )
    graph.source_net_v62_schema_version = payload["schema_version"]
    graph.source_net_v62_control_profile = deepcopy(payload.get("control_profile", {}))
    return graph
