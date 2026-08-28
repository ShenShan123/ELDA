from __future__ import annotations

import json
import csv
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected

from .schema import (
    DEFAULT_BOUNDARY_STUB_ID,
    DEFAULT_NET_ID,
    build_bucket_spec,
    bucket_midpoint,
    bucket_value,
    classify_partition_graph,
    interface_bucket_counts,
    derive_special_node_ids,
    infer_cell_spec,
    load_label_to_cell_mapping,
    normalize_circuit_graph,
    partition_pin_semantic_metrics,
    partition_schema_summary,
)


def _design_id_from_path(path: Path) -> str:
    return path.stem


def _canonical_split_design_id(design_id: str) -> str:
    name = str(design_id)
    name = re.sub(r"__area\d+$", "", name)
    name = re.sub(r"_area\d+$", "", name)
    name = re.sub(r"__delay\d+$", "", name)
    name = re.sub(r"_delay\d+$", "", name)
    return name


def _load_design_entries(input_dir: Path | None, manifest_csv: Path | None) -> List[Tuple[str, Path]]:
    if manifest_csv is not None:
        rows: List[Tuple[str, Path]] = []
        with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pt_path = (row.get("pt_path") or "").strip()
                design_id = (row.get("design") or "").strip()
                status = (row.get("status") or "success").strip()
                if status and status != "success":
                    continue
                if not pt_path or not design_id:
                    continue
                path = Path(pt_path)
                if not path.exists() or not path.is_file():
                    continue
                rows.append((design_id, path))
        return rows

    if input_dir is None:
        raise RuntimeError("either input_dir or manifest_csv must be provided")
    return [(_design_id_from_path(path), path) for path in sorted(input_dir.glob("*.pt"))]


def _load_resume_designs(resume_log: Path | None) -> set[str]:
    if resume_log is None or not resume_log.exists():
        return set()
    finished = set()
    pattern = re.compile(r"\[\d+/\d+\] finished ([^:]+):")
    with resume_log.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                finished.add(match.group(1))
    return finished


def _checkpoint_dir(output_root: Path) -> Path:
    return output_root / "checkpoints" / "design_records"


def _checkpoint_path(output_root: Path, design_id: str) -> Path:
    return _checkpoint_dir(output_root) / f"{design_id}.pt"


def _adjacency_from_cut_groups(cut_groups: List[Dict]) -> Dict[Tuple[int, int], int]:
    adjacency_multiplicity: Dict[Tuple[int, int], int] = defaultdict(int)
    for entry in cut_groups:
        part_ids = sorted(int(part_id) for part_id in entry.get("partition_ids", []))
        for idx, src in enumerate(part_ids):
            for dst in part_ids[idx + 1 :]:
                adjacency_multiplicity[(src, dst)] += 1
    return dict(adjacency_multiplicity)


def _load_existing_design_records(output_root: Path) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}

    manifest_dir = output_root / "manifests"
    if manifest_dir.exists():
        for path in sorted(manifest_dir.glob("*.pt")):
            record = torch.load(path, map_location="cpu", weights_only=False)
            design_id = str(record.get("design_id") or path.stem)
            restored = {
                "design_id": design_id,
                "source_graph_path": record.get("source_graph_path", ""),
                "split": record.get("split", ""),
                "source_num_nodes": int(record.get("source_num_nodes", 0)),
                "skeleton_graph_rel_path": record.get("skeleton_graph_rel_path", ""),
                "partition_records": record.get("partition_records", []),
                "cut_groups": record.get("cut_groups", []),
                "adjacency_multiplicity": _adjacency_from_cut_groups(record.get("cut_groups", [])),
                "_finalized": True,
                "_source_output_root": str(output_root),
            }
            records[design_id] = restored

    checkpoint_dir = _checkpoint_dir(output_root)
    if checkpoint_dir.exists():
        for path in sorted(checkpoint_dir.glob("*.pt")):
            design_id = path.stem
            if design_id in records:
                continue
            record = torch.load(path, map_location="cpu", weights_only=False)
            record["_finalized"] = False
            record["_checkpoint_path"] = str(path)
            record["_source_output_root"] = str(output_root)
            records[design_id] = record

    return records


def _merge_existing_design_records(
    primary_records: Dict[str, Dict],
    inherited_records: Dict[str, Dict],
) -> Dict[str, Dict]:
    merged = dict(primary_records)
    for design_id, record in inherited_records.items():
        if design_id in merged:
            continue
        inherited = dict(record)
        merged[design_id] = inherited
    return merged


def _materialize_inherited_finalized_record(output_root: Path, design_record: Dict) -> None:
    source_output_root = Path(str(design_record.get("_source_output_root", "") or ""))
    if not source_output_root.exists() or source_output_root.resolve() == output_root.resolve():
        return

    manifest_src = source_output_root / "manifests" / f"{design_record['design_id']}.pt"
    manifest_dst = output_root / "manifests" / f"{design_record['design_id']}.pt"
    if manifest_src.exists() and not manifest_dst.exists():
        shutil.copy2(manifest_src, manifest_dst)

    skeleton_rel = str(design_record.get("skeleton_graph_rel_path", "") or "")
    if skeleton_rel:
        skeleton_src = source_output_root / skeleton_rel
        skeleton_dst = output_root / skeleton_rel
        skeleton_dst.parent.mkdir(parents=True, exist_ok=True)
        if skeleton_src.exists() and not skeleton_dst.exists():
            shutil.copy2(skeleton_src, skeleton_dst)

    for partition in design_record.get("partition_records", []):
        partition_rel = str(partition.get("graph_rel_path", "") or "")
        if not partition_rel:
            continue
        partition_src = source_output_root / partition_rel
        partition_dst = output_root / partition_rel
        partition_dst.parent.mkdir(parents=True, exist_ok=True)
        if partition_src.exists() and not partition_dst.exists():
            shutil.copy2(partition_src, partition_dst)


def _accumulate_design_record_statistics(
    design_record: Dict,
    partition_size_values: List[int],
    stub_count_values: List[int],
    pin_count_values: List[int],
    pin_deficit_scaled_values: List[int],
    underconnected_scaled_values: List[int],
    synthetic_input_scaled_values: List[int],
    multiplicity_values: List[int],
    semantic_bucket_scale: int,
) -> None:
    for record in design_record.get("partition_records", []):
        metrics = record.get("metrics", {})
        partition_size_values.append(int(metrics.get("num_nodes", 0)))
        stub_count_values.append(int(metrics.get("boundary_stub_count", 0)))
        pin_count_values.append(int(metrics.get("boundary_pin_count", 0)))
        pin_deficit_scaled_values.append(int(round(float(metrics.get("gate_pin_deficit", 0.0)) * semantic_bucket_scale)))
        underconnected_scaled_values.append(int(round(float(metrics.get("underconnected_gate_fraction", 0.0)) * semantic_bucket_scale)))
        synthetic_input_scaled_values.append(int(round(float(metrics.get("synthetic_input_proxy_ratio", 0.0)) * semantic_bucket_scale)))
    multiplicity_values.extend(int(value) for value in design_record.get("adjacency_multiplicity", {}).values())


def _edge_pairs(data: Data) -> Iterable[Tuple[int, int]]:
    return data.edge_index.t().tolist()


def _build_circuit_views(data: Data, net_id: int) -> Tuple[List[int], Dict[int, List[int]], Dict[int, List[int]]]:
    x = data.x.reshape(-1).to(torch.long)
    gate_nodes = (x != net_id).nonzero(as_tuple=False).reshape(-1).tolist()
    gate_to_nets: Dict[int, List[int]] = defaultdict(list)
    net_to_gates: Dict[int, List[int]] = defaultdict(list)
    for src, dst in _edge_pairs(data):
        if int(x[src].item()) == net_id and int(x[dst].item()) != net_id:
            net_to_gates[src].append(dst)
            gate_to_nets[dst].append(src)
        elif int(x[dst].item()) == net_id and int(x[src].item()) != net_id:
            net_to_gates[dst].append(src)
            gate_to_nets[src].append(dst)
    for key, values in gate_to_nets.items():
        gate_to_nets[key] = sorted(set(values))
    for key, values in net_to_gates.items():
        net_to_gates[key] = sorted(set(values))
    return gate_nodes, gate_to_nets, net_to_gates


def _estimate_partition_size(gate_subset: Sequence[int], net_to_gates: Dict[int, List[int]]) -> int:
    gate_set = set(int(g) for g in gate_subset)
    internal_nets = 0
    boundary_nets = 0
    for incident_gates in net_to_gates.values():
        inside = [gate for gate in incident_gates if gate in gate_set]
        if not inside:
            continue
        if len(inside) == len(incident_gates):
            internal_nets += 1
        else:
            boundary_nets += 1
    return len(gate_set) + internal_nets + boundary_nets


def _cell_family(cell_name: str) -> str:
    name = str(cell_name or "").upper()
    for prefix in (
        "CLKGATETST",
        "CLKGATE",
        "CLKBUF",
        "SDFFRS",
        "SDFFR",
        "SDFFS",
        "SDFF",
        "DFFSR",
        "DFFRS",
        "DFFR",
        "DFFS",
        "DFF",
        "MUX",
        "AOI",
        "OAI",
        "FA",
        "HA",
        "XOR",
        "XNOR",
        "BUF",
        "INV",
        "LOGIC0",
        "LOGIC1",
    ):
        if name.startswith(prefix):
            return prefix
    for prefix in ("NAND", "NOR", "AND", "OR"):
        if name.startswith(prefix):
            return prefix
    return "OTHER"


def _is_clock_or_control_family(family: str) -> bool:
    return family in {"CLKBUF", "CLKGATE", "CLKGATETST"}


def _is_sequential_family(family: str) -> bool:
    return family.startswith("DFF") or family.startswith("SDFF")


def _is_semantic_cluster_family(family: str) -> bool:
    return family in {"MUX", "AOI", "OAI", "FA", "HA", "XOR", "XNOR"}


def _build_semantic_hyperedge_weights(
    data: Data,
    gate_to_nets: Dict[int, List[int]],
    net_to_gates: Dict[int, List[int]],
    *,
    label_to_cell: Dict[int, str] | None,
    net_id: int,
    semantic_cut_bias: str,
) -> Dict[int, int]:
    if semantic_cut_bias == "none" or not label_to_cell:
        return {}

    x = data.x.reshape(-1).to(torch.long)
    weights: Dict[int, int] = {}
    for original_net_id, incident_gates in net_to_gates.items():
        if not incident_gates:
            continue
        families = []
        total_outputs = 0
        total_inputs = 0
        for gate_id in incident_gates:
            label = int(x[int(gate_id)].item())
            cell_name = label_to_cell.get(label, f"LABEL_{label}")
            family = _cell_family(cell_name)
            families.append(family)
            spec = infer_cell_spec(cell_name)
            total_outputs += int(spec.get("outputs", 1))
            total_inputs += int(spec.get("inputs", 0))

        fanout = len(incident_gates)
        family_set = set(families)
        sequential_count = sum(1 for fam in families if _is_sequential_family(fam))
        control_count = sum(1 for fam in families if _is_clock_or_control_family(fam))
        semantic_count = sum(1 for fam in families if _is_semantic_cluster_family(fam))
        buffer_like_count = sum(1 for fam in families if fam in {"BUF", "INV", "CLKBUF", "LOGIC0", "LOGIC1"})

        # Start from a neutral cut cost. Higher weight means KaHyPar is less
        # willing to cut that net, lower weight makes it a preferred cut site.
        weight = 100.0

        if fanout >= 128:
            weight *= 0.05
        elif fanout >= 64:
            weight *= 0.10
        elif fanout >= 32:
            weight *= 0.18
        elif fanout >= 16:
            weight *= 0.35
        elif fanout >= 8:
            weight *= 0.60

        if control_count > 0:
            weight *= 0.20

        # Prefer to cut around sequential boundaries instead of through dense
        # combinational cones. Without pin directions we cannot isolate D/Q/CLK
        # exactly, so we bias nets touching sequential cells to be cheaper cuts.
        if sequential_count > 0:
            weight *= 0.65 if fanout <= 8 else 0.40

        if semantic_count > 0 and sequential_count == 0 and control_count == 0:
            weight *= 1.60
            if "FA" in family_set or "HA" in family_set:
                weight *= 1.40
            if "MUX" in family_set:
                weight *= 1.20

        if buffer_like_count == len(families):
            weight *= 0.35

        # Nets that look like small functional local signals should stay
        # inside partitions when possible.
        if fanout <= 4 and semantic_count > 0 and sequential_count == 0:
            weight *= 1.25

        # If the net touches many cells but likely only carries a single
        # producer, do not over-protect it unless it is clearly functional.
        if total_outputs <= 1 and fanout >= 12 and semantic_count == 0:
            weight *= 0.70

        weight_int = max(1, min(400, int(round(weight))))
        weights[int(original_net_id)] = weight_int
    return weights


def _write_hgr(
    path: Path,
    local_gate_ids: Sequence[int],
    gate_to_nets: Dict[int, List[int]],
    *,
    hyperedge_weights: Dict[int, int] | None = None,
) -> None:
    local_index = {gate_id: idx + 1 for idx, gate_id in enumerate(local_gate_ids)}
    net_to_local_vertices: Dict[int, List[int]] = defaultdict(list)
    for gate_id in local_gate_ids:
        for net_id in gate_to_nets[int(gate_id)]:
            net_to_local_vertices[int(net_id)].append(local_index[int(gate_id)])
    hyperedges: List[List[int]] = []
    edge_weights: List[int] = []
    for net_id, vertices in net_to_local_vertices.items():
        pins = sorted(set(vertices))
        if not pins:
            continue
        hyperedges.append(pins)
        edge_weights.append(max(1, int((hyperedge_weights or {}).get(int(net_id), 1))))
    if not hyperedges:
        hyperedges = [[idx] for idx in range(1, len(local_gate_ids) + 1)]
        edge_weights = [1 for _ in hyperedges]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(hyperedges)} {len(local_gate_ids)} 1\n")
        for idx, pins in enumerate(hyperedges):
            handle.write(f"{int(edge_weights[idx])} " + " ".join(str(pin) for pin in pins) + "\n")


def _run_kahypar_two_way(
    gate_subset: Sequence[int],
    gate_to_nets: Dict[int, List[int]],
    kahypar_bin: Path,
    preset: Path,
    epsilon: float,
    seed: int,
    hyperedge_weights: Dict[int, int] | None = None,
    kahypar_threads: int | None = None,
) -> List[int]:
    with tempfile.TemporaryDirectory(prefix="kahypar_circuit_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        hgr_path = tmp_path / "subgraph.hgr"
        _write_hgr(
            hgr_path,
            gate_subset,
            gate_to_nets,
            hyperedge_weights=hyperedge_weights,
        )
        cmd = [
            str(kahypar_bin),
            "-h",
            str(hgr_path),
            "-k",
            "2",
            "-e",
            str(epsilon),
            "-o",
            "km1",
            "-m",
            "recursive",
            "-p",
            str(preset),
            "-w",
            "true",
            "-q",
            "true",
            "--seed",
            str(seed),
        ]
        env = None
        if kahypar_threads and kahypar_threads > 0:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(int(kahypar_threads))
        subprocess.run(
            cmd,
            check=True,
            cwd=tmp_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        part_path = next(tmp_path.glob("subgraph.hgr.part2.*.KaHyPar"))
        assignments = [int(line.strip()) for line in part_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(assignments) != len(gate_subset):
        raise RuntimeError("KaHyPar output length does not match the input gate subset")
    return assignments


def _fallback_split(gate_subset: Sequence[int], gate_to_nets: Dict[int, List[int]]) -> Tuple[List[int], List[int]]:
    ordered = sorted(int(gate) for gate in gate_subset)
    ordered.sort(key=lambda gate_id: (-len(gate_to_nets[gate_id]), gate_id))
    mid = max(1, len(ordered) // 2)
    return ordered[:mid], ordered[mid:]


def _recursive_partition(
    gate_subset: Sequence[int],
    gate_to_nets: Dict[int, List[int]],
    net_to_gates: Dict[int, List[int]],
    partition_cap: int,
    kahypar_bin: Path,
    preset: Path,
    epsilon: float,
    seed: int,
    hyperedge_weights: Dict[int, int] | None = None,
    kahypar_threads: int | None = None,
    progress: Dict | None = None,
    design_id: str | None = None,
) -> List[List[int]]:
    estimated = _estimate_partition_size(gate_subset, net_to_gates)
    if estimated <= partition_cap or len(gate_subset) <= 1:
        if progress is not None:
            progress["leaf_partitions"] += 1
            progress["gate_covered"] += len(gate_subset)
            now = time.time()
            if progress["leaf_partitions"] - progress["last_logged"] >= progress["log_every"]:
                progress["last_logged"] = progress["leaf_partitions"]
                progress["last_log_time"] = now
                percent = 100.0 * progress["gate_covered"] / max(1, progress["total_gates"])
                print(
                    f"[circuit_kahypar] partition_progress design={design_id} "
                    f"leaf_partitions={progress['leaf_partitions']} "
                    f"gate_covered={progress['gate_covered']} "
                    f"total_gates={progress['total_gates']} "
                    f"approx_done={percent:.1f}%",
                    flush=True,
                )
        return [sorted(int(g) for g in gate_subset)]

    try:
        assignment = _run_kahypar_two_way(
            gate_subset,
            gate_to_nets,
            kahypar_bin,
            preset,
            epsilon,
            seed,
            hyperedge_weights,
            kahypar_threads,
        )
        left = [int(gate_subset[idx]) for idx, part in enumerate(assignment) if part == 0]
        right = [int(gate_subset[idx]) for idx, part in enumerate(assignment) if part == 1]
    except Exception:
        left, right = _fallback_split(gate_subset, gate_to_nets)

    if not left or not right:
        left, right = _fallback_split(gate_subset, gate_to_nets)
    return _recursive_partition(
        left,
        gate_to_nets,
        net_to_gates,
        partition_cap,
        kahypar_bin,
        preset,
        epsilon,
        seed + 1,
        hyperedge_weights,
        kahypar_threads,
        progress,
        design_id,
    ) + _recursive_partition(
        right,
        gate_to_nets,
        net_to_gates,
        partition_cap,
        kahypar_bin,
        preset,
        epsilon,
        seed + 2,
        hyperedge_weights,
        kahypar_threads,
        progress,
        design_id,
    )


def _build_partition_graphs(
    data: Data,
    design_id: str,
    partitions: List[List[int]],
    net_id: int,
    boundary_stub_id: int,
    gate_to_nets: Dict[int, List[int]],
    net_to_gates: Dict[int, List[int]],
) -> Tuple[List[Dict], List[Dict], Dict[Tuple[int, int], int]]:
    gate_to_partition = {}
    for part_idx, gate_ids in enumerate(partitions):
        for gate_id in gate_ids:
            gate_to_partition[int(gate_id)] = int(part_idx)

    adjacency_multiplicity: Dict[Tuple[int, int], int] = defaultdict(int)
    partition_neighbors: Dict[int, set] = defaultdict(set)
    cut_groups: List[Dict] = []
    for original_net_id, incident_gates in sorted(net_to_gates.items()):
        part_ids = sorted({gate_to_partition[gate] for gate in incident_gates if gate in gate_to_partition})
        if len(part_ids) <= 1:
            continue
        stub_refs = []
        for part_id in part_ids:
            stub_refs.append({"partition_id": int(part_id), "stub_local_idx": None})
        cut_groups.append(
            {
                "cut_group_id": len(cut_groups),
                "original_net_id": int(original_net_id),
                "partition_ids": part_ids,
                "stub_refs": stub_refs,
            }
        )
        for idx, src in enumerate(part_ids):
            for dst in part_ids[idx + 1 :]:
                adjacency_multiplicity[(src, dst)] += 1
                partition_neighbors[src].add(dst)
                partition_neighbors[dst].add(src)

    cut_group_by_net = {entry["original_net_id"]: entry for entry in cut_groups}
    partition_records: List[Dict] = []
    x = data.x.reshape(-1).to(torch.long)
    total_parts = len(partitions)
    build_start = time.time()
    last_logged = 0
    log_every = 100

    for part_idx, gate_ids in enumerate(partitions):
        gate_set = set(gate_ids)
        internal_nets = []
        boundary_nets = []
        for original_net_id, incident_gates in sorted(net_to_gates.items()):
            inside = [gate for gate in incident_gates if gate in gate_set]
            if not inside:
                continue
            if len(inside) == len(incident_gates):
                internal_nets.append(original_net_id)
            else:
                boundary_nets.append(original_net_id)

        local_to_original: List[int] = []
        local_labels: List[int] = []
        local_index: Dict[Tuple[str, int], int] = {}
        for gate_id in sorted(gate_ids):
            local_index[("gate", gate_id)] = len(local_to_original)
            local_to_original.append(int(gate_id))
            local_labels.append(int(x[gate_id].item()))
        for original_net_id in internal_nets:
            local_index[("net", original_net_id)] = len(local_to_original)
            local_to_original.append(int(original_net_id))
            local_labels.append(int(net_id))
        boundary_records = []
        for original_net_id in boundary_nets:
            local_idx = len(local_to_original)
            local_index[("stub", original_net_id)] = local_idx
            local_to_original.append(-1)
            local_labels.append(int(boundary_stub_id))
            boundary_records.append(
                {
                    "original_net_id": int(original_net_id),
                    "stub_local_idx": int(local_idx),
                    "gate_ids": [int(g) for g in net_to_gates[original_net_id] if g in gate_set],
                }
            )

        edges: List[Tuple[int, int]] = []
        for gate_id in gate_ids:
            gate_local = local_index[("gate", gate_id)]
            for original_net_id in gate_to_nets[gate_id]:
                if ("net", original_net_id) in local_index:
                    net_local = local_index[("net", original_net_id)]
                elif ("stub", original_net_id) in local_index:
                    net_local = local_index[("stub", original_net_id)]
                else:
                    continue
                edges.append((gate_local, net_local))
                edges.append((net_local, gate_local))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=len(local_labels))
        edge_index = coalesce(edge_index, num_nodes=len(local_labels))
        graph = Data(
            x=torch.tensor(local_labels, dtype=torch.long),
            edge_index=edge_index,
            edge_attr=torch.zeros(edge_index.size(1), dtype=torch.long),
            num_nodes=len(local_labels),
        )

        for boundary_record in boundary_records:
            cut_group = cut_group_by_net[boundary_record["original_net_id"]]
            for stub_ref in cut_group["stub_refs"]:
                if int(stub_ref["partition_id"]) == int(part_idx):
                    stub_ref["stub_local_idx"] = int(boundary_record["stub_local_idx"])
                    break

        partition_records.append(
            {
                "partition_id": int(part_idx),
                "design_id": design_id,
                "graph": graph,
                "local_to_original": local_to_original,
                "original_gate_node_ids": [int(g) for g in sorted(gate_ids)],
                "internal_net_node_ids": [int(n) for n in internal_nets],
                "boundary_records": boundary_records,
                "cut_neighbor_count": int(len(partition_neighbors.get(part_idx, set()))),
            }
        )

        processed = part_idx + 1
        if processed - last_logged >= log_every or processed == total_parts:
            last_logged = processed
            percent = 100.0 * processed / max(1, total_parts)
            print(
                f"[circuit_kahypar] build_partition_graphs_progress design={design_id} "
                f"partitions_done={processed}/{total_parts} "
                f"approx_done={percent:.1f}% "
                f"elapsed={time.time() - build_start:.1f}s",
                flush=True,
            )

    return partition_records, cut_groups, adjacency_multiplicity


def _assign_design_splits(design_ids: Sequence[str], split_ratios: Tuple[float, float, float], seed: int) -> Dict[str, str]:
    family_to_designs: Dict[str, List[str]] = defaultdict(list)
    for design_id in design_ids:
        family_to_designs[_canonical_split_design_id(str(design_id))].append(str(design_id))
    ordered = sorted(family_to_designs)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_total = len(ordered)
    n_train = max(1, int(n_total * split_ratios[0]))
    n_val = max(1, int(n_total * split_ratios[1]))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    splits = {}
    for idx, canonical_id in enumerate(ordered):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"
        for design_id in family_to_designs[canonical_id]:
            splits[design_id] = split
    return splits


def preprocess_dataset(
    input_dir: Path | None,
    output_root: Path,
    kahypar_bin: Path,
    preset: Path,
    partition_cap: int = 100,
    net_id: int | None = None,
    boundary_stub_id: int | None = None,
    epsilon: float = 0.03,
    split_seed: int = 42,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    num_buckets: int = 4,
    limit_designs: int = 0,
    cell_mapping: Path | None = None,
    manifest_csv: Path | None = None,
    kahypar_threads: int | None = None,
    resume_log: Path | None = None,
    resume_from_output_root: Path | None = None,
    semantic_cut_bias: str = "semantic_v1",
) -> Dict:
    all_design_entries = _load_design_entries(input_dir, manifest_csv)
    if limit_designs > 0:
        all_design_entries = all_design_entries[:limit_designs]
    if not all_design_entries:
        source = str(manifest_csv) if manifest_csv is not None else str(input_dir)
        raise RuntimeError(f"no .pt files found from {source}")

    if net_id is None or boundary_stub_id is None:
        derived_net_id, derived_boundary_stub_id = derive_special_node_ids(cell_mapping)
        if net_id is None:
            net_id = int(derived_net_id)
        if boundary_stub_id is None:
            boundary_stub_id = int(derived_boundary_stub_id)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "partitions" / "graphs").mkdir(parents=True, exist_ok=True)
    (output_root / "skeletons" / "graphs").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    _checkpoint_dir(output_root).mkdir(parents=True, exist_ok=True)

    resume_done = _load_resume_designs(resume_log)
    existing_design_records = _load_existing_design_records(output_root)
    inherited_design_records: Dict[str, Dict] = {}
    if resume_from_output_root is not None:
        resume_from_output_root = resume_from_output_root.resolve()
        if output_root.resolve() != resume_from_output_root:
            inherited_design_records = _load_existing_design_records(resume_from_output_root)
            existing_design_records = _merge_existing_design_records(
                existing_design_records,
                inherited_design_records,
            )
    existing_artifact_done = set(existing_design_records)
    resume_artifact_done = {design_id for design_id in resume_done if design_id in existing_artifact_done}
    skip_designs = set(existing_artifact_done)
    design_entries = [
        (design_id, design_path)
        for design_id, design_path in all_design_entries
        if design_id not in skip_designs
    ]

    splits = _assign_design_splits([design_id for design_id, _ in all_design_entries], split_ratios, split_seed)
    partition_size_values = []
    stub_count_values = []
    pin_count_values = []
    pin_deficit_scaled_values = []
    underconnected_scaled_values = []
    synthetic_input_scaled_values = []
    multiplicity_values = []
    design_records = []
    semantic_bucket_scale = 1000
    label_to_cell = load_label_to_cell_mapping(cell_mapping) if cell_mapping is not None and Path(cell_mapping).exists() else None

    for design_id, _ in all_design_entries:
        existing_record = existing_design_records.get(design_id)
        if existing_record is None:
            continue
        existing_record["split"] = splits[design_id]
        design_records.append(existing_record)
        _accumulate_design_record_statistics(
            existing_record,
            partition_size_values,
            stub_count_values,
            pin_count_values,
            pin_deficit_scaled_values,
            underconnected_scaled_values,
            synthetic_input_scaled_values,
            multiplicity_values,
            semantic_bucket_scale,
        )

    source_desc = str(manifest_csv) if manifest_csv is not None else str(input_dir)
    print(
        f"[circuit_kahypar] start preprocess: designs={len(design_entries)} "
        f"partition_cap={partition_cap} net_id={int(net_id)} "
        f"boundary_stub_id={int(boundary_stub_id)} "
        f"semantic_cut_bias={semantic_cut_bias} "
        f"kahypar_threads={kahypar_threads if kahypar_threads else 'default'} "
        f"source={source_desc}",
        flush=True,
    )
    if existing_artifact_done:
        finalized_count = sum(1 for record in existing_design_records.values() if bool(record.get("_finalized", False)))
        checkpoint_count = len(existing_design_records) - finalized_count
        print(
            f"[circuit_kahypar] existing_artifacts loaded_designs={len(existing_design_records)} "
            f"finalized={finalized_count} checkpoints={checkpoint_count}",
            flush=True,
        )
    if inherited_design_records:
        inherited_finalized_count = sum(
            1 for record in inherited_design_records.values() if bool(record.get("_finalized", False))
        )
        inherited_checkpoint_count = len(inherited_design_records) - inherited_finalized_count
        print(
            f"[circuit_kahypar] inherited_artifacts from={resume_from_output_root} "
            f"loaded_designs={len(inherited_design_records)} "
            f"finalized={inherited_finalized_count} checkpoints={inherited_checkpoint_count}",
            flush=True,
        )
    if resume_done:
        print(
            f"[circuit_kahypar] resume_log={resume_log} skipped_designs={len(resume_artifact_done)}",
            flush=True,
        )

    for design_idx, (design_id, design_path) in enumerate(design_entries):
        design_start = time.time()
        print(
            f"[circuit_kahypar] [{design_idx + 1}/{len(design_entries)}] "
            f"processing {design_id} from {design_path}",
            flush=True,
        )
        graph = normalize_circuit_graph(
            torch.load(design_path, map_location="cpu", weights_only=False),
            net_id=net_id,
        )
        print(
            f"[circuit_kahypar] [{design_idx + 1}/{len(design_entries)}] "
            f"stage=load_graph elapsed={time.time() - design_start:.1f}s",
            flush=True,
        )
        gate_nodes, gate_to_nets, net_to_gates = _build_circuit_views(graph, net_id)
        semantic_hyperedge_weights = _build_semantic_hyperedge_weights(
            graph,
            gate_to_nets,
            net_to_gates,
            label_to_cell=label_to_cell,
            net_id=net_id,
            semantic_cut_bias=semantic_cut_bias,
        )
        print(
            f"[circuit_kahypar] [{design_idx + 1}/{len(design_entries)}] "
            f"stage=build_views elapsed={time.time() - design_start:.1f}s",
            flush=True,
        )
        progress_state = {
            "leaf_partitions": 0,
            "gate_covered": 0,
            "total_gates": len(gate_nodes),
            "last_logged": 0,
            "last_log_time": time.time(),
            "log_every": 50,
        }
        partitions = _recursive_partition(
            gate_nodes,
            gate_to_nets,
            net_to_gates,
            partition_cap,
            kahypar_bin,
            preset,
            epsilon,
            split_seed + design_idx * 17,
            semantic_hyperedge_weights,
            kahypar_threads,
            progress_state,
            design_id,
        )
        print(
            f"[circuit_kahypar] [{design_idx + 1}/{len(design_entries)}] "
            f"stage=partition elapsed={time.time() - design_start:.1f}s",
            flush=True,
        )
        partition_records, cut_groups, adjacency_multiplicity = _build_partition_graphs(
            graph,
            design_id,
            partitions,
            net_id,
            boundary_stub_id,
            gate_to_nets,
            net_to_gates,
        )
        print(
            f"[circuit_kahypar] [{design_idx + 1}/{len(design_entries)}] "
            f"stage=build_partition_graphs elapsed={time.time() - design_start:.1f}s",
            flush=True,
        )
        for record in partition_records:
            metrics = {
                "num_nodes": int(record["graph"].num_nodes),
                "boundary_stub_count": int(len(record["boundary_records"])),
                "boundary_pin_count": int(sum(len(entry["gate_ids"]) for entry in record["boundary_records"])),
            }
            if label_to_cell is not None:
                metrics.update(
                    partition_pin_semantic_metrics(
                        record["graph"],
                        label_to_cell=label_to_cell,
                        net_id=net_id,
                        boundary_stub_id=boundary_stub_id,
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
            record["metrics"] = metrics
            partition_size_values.append(metrics["num_nodes"])
            stub_count_values.append(metrics["boundary_stub_count"])
            pin_count_values.append(metrics["boundary_pin_count"])
            pin_deficit_scaled_values.append(int(round(float(metrics["gate_pin_deficit"]) * semantic_bucket_scale)))
            underconnected_scaled_values.append(int(round(float(metrics["underconnected_gate_fraction"]) * semantic_bucket_scale)))
            synthetic_input_scaled_values.append(int(round(float(metrics["synthetic_input_proxy_ratio"]) * semantic_bucket_scale)))
        multiplicity_values.extend(int(value) for value in adjacency_multiplicity.values())
        design_record = {
            "design_id": design_id,
            "source_graph_path": str(design_path),
            "split": splits[design_id],
            "source_num_nodes": int(graph.num_nodes),
            "partition_records": partition_records,
            "cut_groups": cut_groups,
            "adjacency_multiplicity": dict(adjacency_multiplicity),
            "_finalized": False,
            "_checkpoint_path": str(_checkpoint_path(output_root, design_id)),
        }
        checkpoint_path = _checkpoint_path(output_root, design_id)
        torch.save(dict(design_record), checkpoint_path)
        design_records.append(design_record)
        print(
            f"[circuit_kahypar] [{design_idx + 1}/{len(design_entries)}] "
            f"finished {design_id}: partitions={len(partition_records)} "
            f"cut_groups={len(cut_groups)} source_nodes={int(graph.num_nodes)}",
            flush=True,
        )

    bucket_specs = {
        "partition_size": build_bucket_spec(partition_size_values, num_buckets=num_buckets),
        "boundary_stub_count": build_bucket_spec(stub_count_values, num_buckets=num_buckets),
        "boundary_pin_count": build_bucket_spec(pin_count_values, num_buckets=num_buckets),
        "gate_pin_deficit_scaled": build_bucket_spec(pin_deficit_scaled_values or [0], num_buckets=num_buckets),
        "underconnected_gate_fraction_scaled": build_bucket_spec(underconnected_scaled_values or [0], num_buckets=num_buckets),
        "synthetic_input_proxy_ratio_scaled": build_bucket_spec(synthetic_input_scaled_values or [0], num_buckets=num_buckets),
        "cut_multiplicity": build_bucket_spec(multiplicity_values or [1], num_buckets=num_buckets),
    }
    bucket_specs["cut_multiplicity"]["representative_values"] = [
        bucket_midpoint(bucket_specs["cut_multiplicity"], idx)
        for idx in range(len(bucket_specs["cut_multiplicity"]["upper_bounds"]))
    ]

    partition_file_counts = {"train": 0, "val": 0, "test": 0}
    skeleton_file_counts = {"train": 0, "val": 0, "test": 0}

    interface_bucket_keys = [
        "size_bucket",
        "stub_bucket",
        "pin_bucket",
        "pin_deficit_bucket",
        "underconnected_bucket",
        "synthetic_input_bucket",
    ]

    meta_stub = {
        "net_id": int(net_id),
        "boundary_stub_id": int(boundary_stub_id),
        "buckets": bucket_specs,
        "semantic_bucket_scale": int(semantic_bucket_scale),
        "interface_bucket_keys": interface_bucket_keys,
    }
    interface_class_count = 1
    for bucket_count in interface_bucket_counts(meta_stub):
        interface_class_count *= int(bucket_count)

    for design_record in design_records:
        split = design_record["split"]
        if bool(design_record.get("_finalized", False)):
            _materialize_inherited_finalized_record(output_root, design_record)
            partition_file_counts[split] += len(design_record["partition_records"])
            skeleton_file_counts[split] += 1
            continue
        interface_classes = []
        for partition in design_record["partition_records"]:
            graph = partition["graph"]
            classified = classify_partition_graph(graph, meta_stub, label_to_cell=label_to_cell)
            partition["metrics"] = classified
            partition["interface_class"] = int(classified["interface_class"])
            partition["interface_buckets"] = {
                key: int(classified[key])
                for key in interface_bucket_keys
            }
            graph.interface_class = int(classified["interface_class"])
            graph.design_id = design_record["design_id"]
            graph.partition_id = int(partition["partition_id"])
            graph.split = split
            graph.boundary_stub_count = int(classified["boundary_stub_count"])
            graph.boundary_pin_count = int(classified["boundary_pin_count"])
            graph.cut_neighbor_count = int(partition["cut_neighbor_count"])
            graph.gate_pin_deficit = float(classified["gate_pin_deficit"])
            graph.underconnected_gate_fraction = float(classified["underconnected_gate_fraction"])
            graph.synthetic_input_proxy_ratio = float(classified["synthetic_input_proxy_ratio"])
            graph.schema_size_bucket = int(classified["size_bucket"])
            graph.schema_stub_bucket = int(classified["stub_bucket"])
            graph.schema_pin_bucket = int(classified["pin_bucket"])
            graph.schema_pin_deficit_bucket = int(classified["pin_deficit_bucket"])
            graph.schema_underconnected_bucket = int(classified["underconnected_bucket"])
            graph.schema_synthetic_input_bucket = int(classified["synthetic_input_bucket"])
            schema_summary = partition_schema_summary(
                graph,
                net_id=net_id,
                boundary_stub_id=boundary_stub_id,
                label_to_cell=label_to_cell,
            )
            for key, value in schema_summary.items():
                setattr(graph, key, value)
            partition_path = output_root / "partitions" / "graphs" / f"{design_record['design_id']}__p{partition['partition_id']}.pt"
            torch.save(graph, partition_path)
            partition["graph_rel_path"] = str(partition_path.relative_to(output_root))
            del partition["graph"]
            interface_classes.append(int(classified["interface_class"]))
            partition_file_counts[split] += 1

        skeleton_edges = []
        skeleton_edge_attr = []
        for (src, dst), multiplicity in sorted(design_record["adjacency_multiplicity"].items()):
            edge_class = bucket_value(int(multiplicity), bucket_specs["cut_multiplicity"])
            skeleton_edges.extend([(int(src), int(dst)), (int(dst), int(src))])
            skeleton_edge_attr.extend([int(edge_class), int(edge_class)])
        if skeleton_edges:
            edge_index = torch.tensor(skeleton_edges, dtype=torch.long).t().contiguous()
            edge_index = coalesce(edge_index, num_nodes=len(interface_classes))
            edge_attr = torch.tensor(skeleton_edge_attr, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
        skeleton = Data(
            x=torch.tensor(interface_classes, dtype=torch.long),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(interface_classes),
        )
        skeleton.design_id = design_record["design_id"]
        skeleton.split = split
        skeleton_path = output_root / "skeletons" / "graphs" / f"{design_record['design_id']}.pt"
        torch.save(skeleton, skeleton_path)
        skeleton_file_counts[split] += 1

        manifest = {
            "design_id": design_record["design_id"],
            "source_graph_path": design_record["source_graph_path"],
            "split": split,
            "source_num_nodes": int(design_record["source_num_nodes"]),
            "partition_cap": int(partition_cap),
            "skeleton_graph_rel_path": str(skeleton_path.relative_to(output_root)),
            "partition_records": design_record["partition_records"],
            "cut_groups": design_record["cut_groups"],
        }
        torch.save(manifest, output_root / "manifests" / f"{design_record['design_id']}.pt")
        checkpoint_path_str = design_record.get("_checkpoint_path")
        if checkpoint_path_str:
            checkpoint_path = Path(checkpoint_path_str)
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    meta = {
        "version": 3,
        "dataset_root": str(output_root),
        "source_root": str(input_dir) if input_dir is not None else None,
        "manifest_csv": str(manifest_csv) if manifest_csv is not None else None,
        "net_id": int(net_id),
        "boundary_stub_id": int(boundary_stub_id),
        "partition_cap": int(partition_cap),
        "split_seed": int(split_seed),
        "cell_mapping": str(cell_mapping) if cell_mapping is not None else None,
        "semantic_bucket_scale": int(semantic_bucket_scale),
        "semantic_cut_bias": str(semantic_cut_bias),
        "interface_bucket_keys": interface_bucket_keys,
        "interface_class_count": int(interface_class_count),
        "schema_features": [
            "schema_size_bucket",
            "schema_stub_bucket",
            "schema_pin_bucket",
            "schema_pin_deficit_bucket",
            "schema_underconnected_bucket",
            "schema_synthetic_input_bucket",
            "schema_gate_count",
            "schema_internal_net_count",
            "schema_boundary_stub_count",
            "schema_boundary_pin_count",
            "schema_gate_type_unique_count",
            "schema_gate_type_labels",
            "schema_gate_type_counts",
            "schema_stub_degree_one_count",
            "schema_stub_degree_two_count",
            "schema_stub_degree_ge_three_count",
            "schema_boundary_driver_stub_count",
            "schema_boundary_load_stub_count",
            "schema_boundary_bidir_stub_count",
            "schema_boundary_ambig_stub_count",
            "schema_boundary_stub_roles",
            "schema_role_complete",
        ],
        "schema_token_specs": [
            {"name": "size_bucket", "attr": "schema_size_bucket", "count": len(bucket_specs["partition_size"]["upper_bounds"])},
            {"name": "stub_bucket", "attr": "schema_stub_bucket", "count": len(bucket_specs["boundary_stub_count"]["upper_bounds"])},
            {"name": "pin_bucket", "attr": "schema_pin_bucket", "count": len(bucket_specs["boundary_pin_count"]["upper_bounds"])},
            {"name": "pin_deficit_bucket", "attr": "schema_pin_deficit_bucket", "count": len(bucket_specs["gate_pin_deficit_scaled"]["upper_bounds"])},
            {"name": "underconnected_bucket", "attr": "schema_underconnected_bucket", "count": len(bucket_specs["underconnected_gate_fraction_scaled"]["upper_bounds"])},
            {"name": "synthetic_input_bucket", "attr": "schema_synthetic_input_bucket", "count": len(bucket_specs["synthetic_input_proxy_ratio_scaled"]["upper_bounds"])},
        ],
        "split_ratios": tuple(float(v) for v in split_ratios),
        "kahypar_bin": str(kahypar_bin),
        "kahypar_preset": str(preset),
        "kahypar_threads": int(kahypar_threads) if kahypar_threads else None,
        "buckets": bucket_specs,
        "splits": {
            "design_to_split": splits,
            "partition_file_counts": partition_file_counts,
            "skeleton_file_counts": skeleton_file_counts,
        },
    }
    torch.save(meta, output_root / "meta.pt")
    json_meta = {
        **meta,
        "split_ratios": list(meta["split_ratios"]),
    }
    (output_root / "meta.json").write_text(json.dumps(json_meta, indent=2), encoding="utf-8")
    print(
        f"[circuit_kahypar] completed preprocess: output_root={output_root} "
        f"partition_files={sum(partition_file_counts.values())} "
        f"skeleton_files={sum(skeleton_file_counts.values())}",
        flush=True,
    )
    return meta
