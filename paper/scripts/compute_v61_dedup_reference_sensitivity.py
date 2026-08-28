#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from elda_paths import DATA_ROOT, ELDA_ROOT, PAPER_ROOT, V5_DATA_ROOT

ROOT = PAPER_ROOT
ELDA = ELDA_ROOT
DATASET = DATA_ROOT
V5_ROOT = V5_DATA_ROOT
ATTEMPTS = ROOT / "results/source_net_v61/topology_safe_mask_n1024_best7epoch/attempts"
AUDIT = (
    ROOT
    / "results/source_net_v61/phase4g_source_clean_manifest/strict_no_cross_duplicate_splits"
    / "v61_object_sequence_cross_split_audit.json"
)
OUT = ROOT / "reports/final_baseline/phase10_6_main_graph_quality_table"

for import_path in (ELDA, ELDA / "tools"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from elda.datamodules.data.circuit_source_net_v61_tokenizer import (  # noqa: E402
    CircuitSourceNetV61Tokenizer,
)
from v5_source_net_common import build_v5_tokenizer  # noqa: E402


FANOUT_BINS = [
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 4, "3-4"),
    (5, 8, "5-8"),
    (9, 16, "9-16"),
    (17, None, "17+"),
]

_TOKENIZER = None


def worker_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        v5, _ = build_v5_tokenizer(V5_ROOT, max_length=24576)
        tokenizer = CircuitSourceNetV61Tokenizer(
            max_length=24576,
            net_id=int(v5.net_id),
            boundary_stub_id=int(v5.boundary_stub_id),
            label_to_cell=dict(v5.label_to_cell),
            pin_specs=v5.pin_specs,
        )
        meta = torch.load(DATASET / "meta.pt", map_location="cpu", weights_only=False)
        tokenizer.set_num_nodes(int(meta["max_num_nodes"]))
        tokenizer.set_pin_name_vocab(list(meta["pin_name_vocab"]))
        tokenizer.set_num_node_and_edge_types(len(meta["cell_type_vocab"]), 0)
        _TOKENIZER = tokenizer
    return _TOKENIZER


def scalar_labels(graph: Any) -> list[int] | None:
    x = getattr(graph, "x", None)
    if x is None:
        return None
    x = x.detach().cpu()
    if x.dim() == 1:
        return [int(v) for v in x.tolist()]
    if x.dim() == 2 and x.shape[1] == 1:
        return [int(v) for v in x[:, 0].tolist()]
    return None


def graph_feature_payload(graph: Any) -> dict[str, Any]:
    n = int(getattr(graph, "num_nodes", 0) or 0)
    edge_index = getattr(graph, "edge_index", None)
    edges: list[tuple[int, int]] = []
    if edge_index is not None:
        for u, v in edge_index.detach().cpu().t().tolist():
            u, v = int(u), int(v)
            if 0 <= u < n and 0 <= v < n and u != v:
                edges.append((min(u, v), max(u, v)))
    edges = sorted(set(edges))
    labels = scalar_labels(graph)
    degrees = [0] * n
    adjacency = [set() for _ in range(n)]
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
        adjacency[u].add(v)
        adjacency[v].add(u)
    comp_count, lcc_ratio = components(adjacency)
    return {
        "node_count": n,
        "labels": labels,
        "edges": edges,
        "edge_count": len(edges),
        "edge_density": (2.0 * len(edges) / (n * (n - 1))) if n > 1 else 0.0,
        "degrees": degrees,
        "component_count": comp_count,
        "lcc_ratio": lcc_ratio,
        "h_a2": h_a2(labels, adjacency),
    }


def components(adjacency: list[set[int]]) -> tuple[int, float]:
    n = len(adjacency)
    if n == 0:
        return 0, 0.0
    seen = [False] * n
    sizes = []
    for start in range(n):
        if seen[start]:
            continue
        q: deque[int] = deque([start])
        seen[start] = True
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for nxt in adjacency[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        sizes.append(size)
    return len(sizes), max(sizes) / n


def h_a2(labels: list[int] | None, adjacency: list[set[int]]) -> float | None:
    if labels is None or len(labels) != len(adjacency):
        return None
    same = 0
    total = 0
    for center, nbrs in enumerate(adjacency):
        nbr_list = list(nbrs)
        for src in nbr_list:
            src_label = labels[src]
            for dst in nbr_list:
                total += 1
                if src_label == labels[dst]:
                    same += 1
    if total == 0:
        return None
    return same / total


def fanout_bin(value: int) -> str:
    for lo, hi, label in FANOUT_BINS:
        if value >= lo and (hi is None or value <= hi):
            return label
    raise ValueError(value)


def normalize_v61_source_type(source: dict[str, Any]) -> str:
    kind = str(source.get("source_kind") or source.get("endpoint_type") or "UNKNOWN").upper()
    role = str(source.get("source_role") or "").upper()
    if "BOUNDARY" in kind or "BOUNDARY" in role:
        return "boundary_source"
    if "CONST" in kind or "CONST" in role:
        return "constant_source"
    if "CELL_OUTPUT" in kind or "OUTPUT" in role:
        return "cell_output_source"
    return kind.lower()


def v61_source_load_hist(payload: dict[str, Any]) -> Counter[tuple[str, str]]:
    sources = payload.get("sources", [])
    source_nets = payload.get("source_nets", [])
    source_type_by_id = {
        str(source.get("source_id")): normalize_v61_source_type(source)
        for source in sources
    }
    fanout_by_id = {source_id: 0 for source_id in source_type_by_id}
    for record in source_nets:
        source_id = str(record.get("source_id"))
        count = 0
        for chunk in record.get("chunks", []) or []:
            demand_ids = chunk.get("demand_ids", [])
            if isinstance(demand_ids, list):
                count += len(demand_ids)
        if not count and isinstance(record.get("demand_ids"), list):
            count = len(record["demand_ids"])
        elif not count and record.get("fanout") is not None:
            count = int(record["fanout"])
        fanout_by_id[source_id] = count
    hist: Counter[tuple[str, str]] = Counter()
    for source_id, source_type in source_type_by_id.items():
        hist[(source_type, fanout_bin(int(fanout_by_id.get(source_id, 0))))] += 1
    return hist


def inspect_reference(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    tokenizer = worker_tokenizer()
    graph = torch.load(path, map_location="cpu", weights_only=False)
    sequence = tokenizer(graph)
    payload, _report = tokenizer.parse_tokens(sequence)
    decoded = tokenizer.decode(sequence)
    return {
        "path": path_text,
        "feature": graph_feature_payload(decoded),
        "fanout_hist": dict(v61_source_load_hist(payload)),
        "error": "",
    }


def inspect_generated(pair: tuple[str, str]) -> dict[str, Any]:
    graph_path, payload_path = pair
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    return {
        "path": graph_path,
        "feature": graph_feature_payload(graph),
        "fanout_hist": dict(v61_source_load_hist(payload)),
        "error": "",
    }


def wasserstein_1d(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    xs = sorted(set(a) | set(b))
    ia = ib = 0
    sa = sorted(a)
    sb = sorted(b)
    cdfa = cdfb = 0.0
    total = 0.0
    prev = xs[0]
    for x in xs:
        total += abs(cdfa - cdfb) * (x - prev)
        while ia < len(sa) and sa[ia] <= x:
            ia += 1
        while ib < len(sb) and sb[ib] <= x:
            ib += 1
        cdfa = ia / len(sa)
        cdfb = ib / len(sb)
        prev = x
    return total


def degree_hist_vector(feature: dict[str, Any], max_degree: int = 256) -> np.ndarray:
    hist = np.zeros(max_degree + 1, dtype=np.float64)
    for degree in feature["degrees"]:
        hist[min(int(degree), max_degree)] += 1
    total = hist.sum()
    return hist / total if total else hist


def gaussian_mmd2(x: list[np.ndarray], y: list[np.ndarray]) -> float | None:
    if not x or not y:
        return None
    xs = np.stack(x).astype(np.float32, copy=False)
    ys = np.stack(y).astype(np.float32, copy=False)
    pooled = np.concatenate([xs[: min(len(xs), 256)], ys[: min(len(ys), 256)]], axis=0)
    dists = []
    for i in range(min(len(pooled), 256)):
        delta = pooled[i + 1 :] - pooled[i]
        if len(delta):
            dists.extend(np.sum(delta * delta, axis=1).tolist())
    positive = [d for d in dists if d > 0]
    sigma2 = float(np.median(positive)) if positive else 1.0
    if sigma2 <= 0:
        sigma2 = 1.0

    def kernel(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        b_norm = np.sum(b * b, axis=1)[None, :]
        for i in range(0, len(a), 256):
            aa = a[i : i + 256]
            distances = (
                np.sum(aa * aa, axis=1)[:, None]
                + b_norm
                - 2.0 * (aa @ b.T)
            )
            np.maximum(distances, 0.0, out=distances)
            total += float(np.exp(-distances / (2.0 * sigma2)).sum())
        return total

    return kernel(xs, xs) / (len(xs) ** 2) + kernel(ys, ys) / (len(ys) ** 2) - 2.0 * kernel(xs, ys) / (len(xs) * len(ys))


def tv(real: Counter[tuple[str, str]], gen: Counter[tuple[str, str]]) -> float | None:
    real_total = sum(real.values())
    gen_total = sum(gen.values())
    if real_total == 0 or gen_total == 0:
        return None
    keys = set(real) | set(gen)
    return 0.5 * sum(abs(real.get(k, 0) / real_total - gen.get(k, 0) / gen_total) for k in keys)


def merged_hist(rows: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    out: Counter[tuple[str, str]] = Counter()
    for row in rows:
        out.update({tuple(k): v for k, v in row["fanout_hist"].items()})
    return out


def summarize(reference_rows: list[dict[str, Any]], generated_rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    ref = [row["feature"] for row in reference_rows]
    gen = [row["feature"] for row in generated_rows]
    ref_h = [float(item["h_a2"]) for item in ref if item.get("h_a2") is not None]
    gen_h = [float(item["h_a2"]) for item in gen if item.get("h_a2") is not None]
    ref_components = [f["component_count"] for f in ref]
    gen_components = [f["component_count"] for f in gen]
    components_w1 = wasserstein_1d(ref_components, gen_components)
    ref_components_mean = mean(ref_components) if ref_components else None
    return {
        "reference_test_set": name,
        "test_size": len(reference_rows),
        "generated_size": len(generated_rows),
        "density_W1": wasserstein_1d([f["edge_density"] for f in ref], [f["edge_density"] for f in gen]),
        "degree_MMD": gaussian_mmd2(
            [degree_hist_vector(f) for f in ref],
            [degree_hist_vector(f) for f in gen],
        ),
        "components_W1": components_w1,
        "components_mean_real": ref_components_mean,
        "components_mean_generated": mean(gen_components) if gen_components else None,
        "norm_components_W1": (
            components_w1 / max(float(ref_components_mean), 1.0)
            if components_w1 is not None and ref_components_mean is not None
            else None
        ),
        "lcc_W1": wasserstein_1d(
            [f["lcc_ratio"] for f in ref],
            [f["lcc_ratio"] for f in gen],
        ),
        "type_mixing_hA2_ratio": (mean(gen_h) / mean(ref_h)) if ref_h and gen_h and mean(ref_h) else None,
        "type_mixing_hA2_ref_mean": mean(ref_h) if ref_h else None,
        "type_mixing_hA2_gen_mean": mean(gen_h) if gen_h else None,
        "fanout_TV": tv(merged_hist(reference_rows), merged_hist(generated_rows)),
    }


def matched_test_paths_to_remove() -> set[str]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    remove: set[str] = set()
    for group in audit["cross_split_groups"]:
        paths_by_split = group.get("paths_by_split", {})
        if (paths_by_split.get("train") or paths_by_split.get("val")) and paths_by_split.get("test"):
            remove.update(paths_by_split["test"])
    return remove


def main() -> None:
    test_paths = [
        line.strip()
        for line in (DATASET / "test_source_clean.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    remove = matched_test_paths_to_remove()
    dedup_paths = [path for path in test_paths if path not in remove]
    if len(test_paths) != 11574:
        raise RuntimeError(f"unexpected full test size: {len(test_paths)}")
    if len(dedup_paths) != 11390:
        raise RuntimeError(f"unexpected deduplicated test size: {len(dedup_paths)}")

    gen_pairs = [
        (str(path / "decoded_graph.pt"), str(path / "payload.json"))
        for path in sorted(ATTEMPTS.glob("attempt_*"))
        if (path / "decoded_graph.pt").is_file() and (path / "payload.json").is_file()
    ]
    if len(gen_pairs) != 1024:
        raise RuntimeError(f"unexpected generated count: {len(gen_pairs)}")

    with ProcessPoolExecutor(max_workers=16) as pool:
        full_rows = list(pool.map(inspect_reference, test_paths, chunksize=32))
    full_by_path = {row["path"]: row for row in full_rows}
    dedup_rows = [full_by_path[path] for path in dedup_paths]
    with ProcessPoolExecutor(max_workers=8) as pool:
        generated_rows = list(pool.map(inspect_generated, gen_pairs, chunksize=32))

    rows = [
        summarize(full_rows, generated_rows, "Full test set"),
        summarize(dedup_rows, generated_rows, "Development-deduplicated test reference"),
    ]
    result = {
        "schema_version": "elda_reference_development_dedup_sensitivity_v2",
        "method": "Source-Net V6.1 best7epoch topology-safe",
        "reference_view": "deterministic ELDA serialization/decode for source-clean test subcircuits",
        "dedup_policy": "Remove test subcircuits whose exact ELDA object sequence has a training- or validation-split exact match.",
        "removed_test_partitions": len(remove),
        "generated_candidates": len(generated_rows),
        "metrics": rows,
        "degree_mmd_reference_cap": None,
        "degree_mmd_reference_population": len(dedup_rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v61_reference_dedup_sensitivity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["Reference test set", "Test size", "Density W1", "Degree MMD", "Rel. Components W1", "LCC W1", "Type-mixing", "Fanout TV"]
    csv_rows = []
    for row in rows:
        csv_rows.append({
            "Reference test set": row["reference_test_set"],
            "Test size": row["test_size"],
            "Density W1": f"{row['density_W1']:.6f}",
            "Degree MMD": f"{row['degree_MMD']:.6f}",
            "Rel. Components W1": f"{row['norm_components_W1']:.6f}",
            "LCC W1": f"{row['lcc_W1']:.6f}",
            "Type-mixing": f"{row['type_mixing_hA2_ratio']:.6f}",
            "Fanout TV": f"{row['fanout_TV']:.6f}",
        })
    with (OUT / "v61_reference_dedup_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "| Reference test set | Test size | Density W1 | Degree MMD | Rel. Components W1 | LCC W1 | Type-mixing | Fanout TV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in csv_rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    lines.extend([
        "",
        "Notes:",
        "- Generated set is fixed: Source-Net V6.1 best7epoch topology-safe, 1024 candidates.",
        "- Reference view uses deterministic ELDA serialization/decode, matching the generated decoded graph view.",
        "- The development-deduplicated test reference removes 184 test subcircuits: 182 with training exact endpoint-sequence matches and 2 additional validation-only matches.",
        "- All metrics, including blockwise Gaussian-kernel Degree MMD, use the full listed reference set without random reference subsampling.",
        "- Rel. Components W1 = W1(C_gen, C_ref) / max(mean(C_ref), 1), where C is the raw connected-component count per graph. Raw component-count W1 and means are retained in the JSON artifact.",
        "- Type-mixing is the generated/reference mean ratio of two-hop label homophily `h(A^2,Y)`; values closer to 1 indicate better agreement.",
        "- Fanout TV uses exact ELDA source-load demand incidence over `P(source_type, fanout_bin)`.",
    ])
    (OUT / "v61_reference_dedup_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
