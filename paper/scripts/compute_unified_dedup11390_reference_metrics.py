#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
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
AUDIT = (
    ROOT
    / "results/source_net_v61/phase4g_source_clean_manifest/strict_no_cross_duplicate_splits"
    / "v61_object_sequence_cross_split_audit.json"
)
OUT = ROOT / "reports/final_baseline/phase10_6_main_graph_quality_table"
CACHE = OUT / "unified_dedup11390_cache_v1"
V5_PROJECTION_CACHE = OUT / "unified_dedup11390_elda_v5_projection"

NET_NODE_TYPE = 139
BOUNDARY_NODE_TYPE = 140
FANOUT_BINS = [
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 4, "3-4"),
    (5, 8, "5-8"),
    (9, 16, "9-16"),
    (17, None, "17+"),
]

for import_path in (ELDA, ELDA / "tools"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from elda.datamodules.data.circuit_source_net_v61_tokenizer import (  # noqa: E402
    CircuitSourceNetV61Tokenizer,
)
from run_source_net_v5_output_repair_autopsy import _complete_output_nets  # noqa: E402
from v5_source_net_common import build_v5_tokenizer  # noqa: E402


_TOKENIZER = None
_V5_TOKENIZER = None


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


def worker_v5_tokenizer():
    global _V5_TOKENIZER
    if _V5_TOKENIZER is None:
        _V5_TOKENIZER, _ = build_v5_tokenizer(V5_ROOT, max_length=24576)
    return _V5_TOKENIZER


def load_json_graph(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    node_count = int(data.get("node_count", len(nodes)))
    labels = data.get("node_types")
    if labels is None and nodes and isinstance(nodes[0], dict):
        labels = [node.get("type", node.get("node_type")) for node in nodes]
    edges = []
    for edge in data.get("edges", []):
        if isinstance(edge, dict):
            u = edge.get("source", edge.get("src", edge.get("u")))
            v = edge.get("target", edge.get("dst", edge.get("v")))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            u, v = edge[0], edge[1]
        else:
            continue
        try:
            edges.append((int(u), int(v)))
        except Exception:
            continue
    return {"node_count": node_count, "edges": edges, "labels": normalize_labels(labels, node_count)}


def load_pt_graph(path: Path) -> dict[str, Any]:
    graph = torch.load(path, map_location="cpu", weights_only=False)
    node_count = int(getattr(graph, "num_nodes", 0) or 0)
    edges = []
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None:
        for u, v in edge_index.detach().cpu().t().tolist():
            edges.append((int(u), int(v)))
    x = getattr(graph, "x", None)
    labels = None
    if x is not None:
        x = x.detach().cpu()
        if x.dim() == 1:
            labels = [int(v) for v in x.tolist()]
        elif x.dim() == 2 and x.shape[1] == 1:
            labels = [int(v) for v in x[:, 0].tolist()]
        elif x.dim() == 2:
            labels = [int(v) for v in x.argmax(dim=1).tolist()]
    return {"node_count": node_count, "edges": edges, "labels": normalize_labels(labels, node_count)}


def normalize_labels(labels: Any, n: int) -> list[int] | None:
    if not isinstance(labels, list) or len(labels) != n:
        return None
    try:
        return [int(value) for value in labels]
    except Exception:
        return None


def load_graph(path: Path) -> dict[str, Any]:
    return load_json_graph(path) if path.suffix == ".json" else load_pt_graph(path)


def fanout_bin(value: int) -> str:
    for lo, hi, label in FANOUT_BINS:
        if value >= lo and (hi is None or value <= hi):
            return label
    raise ValueError(value)


def graph_features(graph: dict[str, Any]) -> dict[str, Any]:
    n = int(graph["node_count"])
    labels = graph.get("labels")
    edges = sorted(
        {
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in graph["edges"]
            if 0 <= int(u) < n and 0 <= int(v) < n and int(u) != int(v)
        }
    )
    adjacency = [set() for _ in range(n)]
    degrees = [0] * n
    out_degrees = [0] * n
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
        degrees[u] += 1
        degrees[v] += 1
    for u, v in graph["edges"]:
        if 0 <= int(u) < n and 0 <= int(v) < n:
            out_degrees[int(u)] += 1
    comp_count, lcc_ratio = components(adjacency)
    return {
        "node_count": n,
        "edge_count": len(edges),
        "edge_density": (2.0 * len(edges) / (n * (n - 1))) if n > 1 else 0.0,
        "out_degree": (len(graph["edges"]) / n) if n else 0.0,
        "degrees": degrees,
        "component_count": comp_count,
        "lcc_ratio": lcc_ratio,
        "h_a2": h_a2(labels, adjacency),
        "labels": labels,
        "structural_hash": structural_hash(n, labels, edges),
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
        q = deque([start])
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
    for nbrs in adjacency:
        nbrs = list(nbrs)
        for src in nbrs:
            label = labels[src]
            for dst in nbrs:
                total += 1
                if label == labels[dst]:
                    same += 1
    return None if total == 0 else same / total


def structural_hash(n: int, labels: list[int] | None, edges: list[tuple[int, int]]) -> str:
    payload = {"n": n, "labels": labels, "edges": edges}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def graph_source_load_hist(graph: dict[str, Any]) -> tuple[Counter[tuple[str, str]], bool]:
    n = int(graph["node_count"])
    labels = graph.get("labels")
    if labels is None or len(labels) != n:
        return Counter(), False
    adjacency = [set() for _ in range(n)]
    for u, v in graph["edges"]:
        u, v = int(u), int(v)
        if 0 <= u < n and 0 <= v < n and u != v:
            adjacency[u].add(v)
            adjacency[v].add(u)
    hist: Counter[tuple[str, str]] = Counter()
    for idx, label in enumerate(labels):
        if label not in {NET_NODE_TYPE, BOUNDARY_NODE_TYPE}:
            continue
        load_incidence = sum(1 for nbr in adjacency[idx] if labels[nbr] not in {NET_NODE_TYPE, BOUNDARY_NODE_TYPE})
        if label == NET_NODE_TYPE:
            hist[("cell_output_source", fanout_bin(max(load_incidence - 1, 0)))] += 1
        else:
            hist[("boundary_source", fanout_bin(load_incidence))] += 1
    return hist, True


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
    source_type_by_id = {str(s.get("source_id")): normalize_v61_source_type(s) for s in sources}
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
    graph = load_pt_graph(Path(path_text))
    tokenizer = worker_tokenizer()
    raw = torch.load(path_text, map_location="cpu", weights_only=False)
    sequence = tokenizer(raw)
    payload, _ = tokenizer.parse_tokens(sequence)
    return {
        "path": path_text,
        "feature": graph_features(graph),
        "fanout_hist": counter_to_jsonable(v61_source_load_hist(payload)),
        "error": "",
    }


def inspect_graph_path(path_text: str) -> dict[str, Any]:
    graph = load_graph(Path(path_text))
    hist, ok = graph_source_load_hist(graph)
    return {
        "path": path_text,
        "feature": graph_features(graph),
        "fanout_hist": counter_to_jsonable(hist),
        "fanout_ok": ok,
        "error": "",
    }


def inspect_v61_attempt(attempt_dir: str) -> dict[str, Any]:
    path = Path(attempt_dir)
    graph = load_pt_graph(path / "decoded_graph.pt")
    payload = json.loads((path / "payload.json").read_text(encoding="utf-8"))
    return {
        "path": str(path / "decoded_graph.pt"),
        "feature": graph_features(graph),
        "fanout_hist": counter_to_jsonable(v61_source_load_hist(payload)),
        "fanout_ok": True,
        "error": "",
    }


def v5_project_one(args: tuple[str, str]) -> dict[str, Any]:
    source_text, target_text = args
    source = Path(source_text)
    target = Path(target_text)
    if target.exists():
        return {"source": source_text, "target": target_text, "cached": True, "error": ""}
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = worker_v5_tokenizer()
    graph = torch.load(source, map_location="cpu", weights_only=False)
    decoded = tokenizer.decode(tokenizer.tokenize(graph))
    completed, stats = _complete_output_nets(decoded, tokenizer)
    torch.save(completed, target)
    return {
        "source": source_text,
        "target": target_text,
        "cached": False,
        "created_net_count": int(stats.get("created_net_count", 0)) if isinstance(stats, dict) else None,
        "error": "",
    }


def counter_to_jsonable(counter: Counter[tuple[str, str]]) -> dict[str, int]:
    return {"|".join(key): int(value) for key, value in counter.items()}


def counter_from_jsonable(data: dict[str, int]) -> Counter[tuple[str, str]]:
    out: Counter[tuple[str, str]] = Counter()
    for key, value in data.items():
        left, right = key.split("|", 1)
        out[(left, right)] += int(value)
    return out


def dedup_reference_paths() -> list[str]:
    paths = [line.strip() for line in (DATASET / "test_source_clean.txt").read_text().splitlines() if line.strip()]
    audit = json.loads(AUDIT.read_text())
    remove = set()
    for group in audit["cross_split_groups"]:
        by_split = group.get("paths_by_split", {})
        if (by_split.get("train") or by_split.get("val")) and by_split.get("test"):
            remove.update(by_split["test"])
    dedup = [path for path in paths if path not in remove]
    if len(dedup) != 11390:
        raise RuntimeError(f"expected 11390 dedup references, got {len(dedup)}")
    return dedup


def project_v5_paths(paths: list[str], out_dir: Path, prefix: str, workers: int = 8) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [str(out_dir / f"{prefix}_{index:05d}.pt") for index in range(len(paths))]
    manifest_path = out_dir / f"{prefix}_projection_manifest.jsonl"
    existing = 0
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(v5_project_one, zip(paths, targets), chunksize=16), start=1):
            rows.append(row)
            if row.get("cached"):
                existing += 1
            if index % 1000 == 0 or index == len(paths):
                print(
                    f"[v5-project] {prefix}: {index}/{len(paths)} cached={existing}",
                    flush=True,
                )
    manifest_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    return targets


def method_paths() -> dict[str, tuple[str, list[str]]]:
    return {
        "AutoGraph-labeled-llama-s-e20-best": (
            "graph",
            [str(p) for p in sorted((ROOT / "results/autograph_native/phase4_full_labeled_llama_s_max2048_bf16_nw8_cyfix_v4/raw_outputs").glob("*_graph.pt"))],
        ),
        "G2PT-labeled-e25-best": (
            "graph",
            [str(p) for p in sorted((ROOT / "results/g2pt/phase6_full_labeled_e25_v2_best_resample1024/decoded_graphs").glob("generated_sequence_*.json"))],
        ),
        "DiGress-full": (
            "graph",
            [str(p) for p in sorted((ROOT / "results/digress/phase9_full_optional_e20/decoded_graphs").glob("digress_*.json"))],
        ),
    }


def elda_generated_decoded_graph_paths() -> list[str]:
    attempt_root = ROOT / "results/source_net_v61/topology_safe_mask_n1024_best7epoch/attempts"
    return [
        str(path / "decoded_graph.pt")
        for path in sorted(attempt_root.glob("attempt_*"))
        if (path / "decoded_graph.pt").exists()
    ]


def load_or_build_rows(cache_name: str, paths: list[str], kind: str, workers: int = 16) -> list[dict[str, Any]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{cache_name}.jsonl"
    if cache.exists():
        return [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
    if kind == "reference":
        func = inspect_reference
    elif kind == "v61_attempt":
        func = inspect_v61_attempt
    else:
        func = inspect_graph_path
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(func, paths, chunksize=32), start=1):
            rows.append(row)
            if index % 10000 == 0 or index == len(paths):
                print(f"[load] {cache_name}: {index}/{len(paths)}", flush=True)
    cache.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    return rows


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row["feature"][key]) for row in rows]


def wasserstein_1d(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    xs = sorted(set(a) | set(b))
    ia = ib = 0
    sa, sb = sorted(a), sorted(b)
    cdfa = cdfb = total = 0.0
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


def density_vector(feature: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(feature["edge_density"])], dtype=np.float64)


def gaussian_mmd2(x: list[np.ndarray], y: list[np.ndarray]) -> float | None:
    if not x or not y:
        return None
    xs = np.stack(x)
    ys = np.stack(y)
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
        b_norm = np.sum(b * b, axis=1)
        for i in range(0, len(a), 128):
            aa = a[i : i + 128]
            dist = np.sum(aa * aa, axis=1)[:, None] + b_norm[None, :] - 2.0 * aa.dot(b.T)
            np.maximum(dist, 0.0, out=dist)
            total += float(np.exp(-dist / (2.0 * sigma2)).sum())
        return total

    return kernel(xs, xs) / (len(xs) ** 2) + kernel(ys, ys) / (len(ys) ** 2) - 2.0 * kernel(xs, ys) / (len(xs) * len(ys))


def merged_hist(rows: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    out: Counter[tuple[str, str]] = Counter()
    for row in rows:
        out.update(counter_from_jsonable(row["fanout_hist"]))
    return out


def tv(real: Counter[tuple[str, str]], gen: Counter[tuple[str, str]]) -> float | None:
    real_total = sum(real.values())
    gen_total = sum(gen.values())
    if real_total == 0 or gen_total == 0:
        return None
    keys = set(real) | set(gen)
    return 0.5 * sum(abs(real.get(k, 0) / real_total - gen.get(k, 0) / gen_total) for k in keys)


def summarize(
    name: str,
    ref: list[dict[str, Any]],
    gen: list[dict[str, Any]],
    reference_label: str = "development-deduplicated ELDA source-clean test set",
) -> dict[str, Any]:
    ref_components = values(ref, "component_count")
    gen_components = values(gen, "component_count")
    comp_w1 = wasserstein_1d(ref_components, gen_components)
    ref_comp_mean = mean(ref_components) if ref_components else None
    ref_h = [float(row["feature"]["h_a2"]) for row in ref if row["feature"].get("h_a2") is not None]
    gen_h = [float(row["feature"]["h_a2"]) for row in gen if row["feature"].get("h_a2") is not None]
    return {
        "method": name,
        "generated_graph_count": len(gen),
        "reference_test_set": reference_label,
        "reference_sample_count": len(ref),
        "density_W1": wasserstein_1d(values(ref, "edge_density"), values(gen, "edge_density")),
        "density_MMD": gaussian_mmd2([density_vector(row["feature"]) for row in ref], [density_vector(row["feature"]) for row in gen]),
        "degree_MMD": gaussian_mmd2([degree_hist_vector(row["feature"]) for row in ref], [degree_hist_vector(row["feature"]) for row in gen]),
        "relative_components_W1": comp_w1 / max(float(ref_comp_mean), 1.0) if comp_w1 is not None and ref_comp_mean is not None else None,
        "components_W1": comp_w1,
        "components_mean_real": ref_comp_mean,
        "components_mean_generated": mean(gen_components) if gen_components else None,
        "type_mixing_hA2_ratio": (mean(gen_h) / mean(ref_h)) if ref_h and gen_h and mean(ref_h) else None,
        "type_mixing_hA2_ref_mean": mean(ref_h) if ref_h else None,
        "type_mixing_hA2_generated_mean": mean(gen_h) if gen_h else None,
        "lcc_W1": wasserstein_1d(values(ref, "lcc_ratio"), values(gen, "lcc_ratio")),
        "fanout_TV": tv(merged_hist(ref), merged_hist(gen)),
    }


def profile_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scalars = []
    hists = []
    paths = []
    for row in rows:
        f = row["feature"]
        scalars.append([
            f["node_count"],
            f["edge_count"],
            f["edge_density"] * 1000.0,
            f["component_count"],
            f["lcc_ratio"] * 100.0,
            -1.0 if f.get("h_a2") is None else float(f["h_a2"]) * 100.0,
        ])
        hists.append(degree_hist_vector(f))
        paths.append(row["path"])
    return np.asarray(scalars, dtype=np.float32), np.stack(hists).astype(np.float32), paths


def rerun_retrieval(ref_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_paths = [line.strip() for line in (DATASET / "train_source_clean.txt").read_text().splitlines() if line.strip()]
    train_rows = load_or_build_rows("train_source_clean_graph_profiles", train_paths, "graph", workers=16)
    rng = random.Random(20260623)
    target_rows = [ref_rows[index] for index in sorted(rng.sample(range(len(ref_rows)), 1024))]
    train_scalars, train_hists, train_paths_ordered = profile_arrays(train_rows)
    target_scalars, target_hists, _ = profile_arrays(target_rows)
    q25, q75 = np.quantile(train_scalars, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale[scale < 1.0] = 1.0
    selected = []
    selected_signatures = set()
    signature_by_idx = {}
    for idx in range(len(target_rows)):
        scalar_distance = np.abs(train_scalars - target_scalars[idx]) / scale
        scalar_score = scalar_distance[:, 0] + scalar_distance[:, 1] + scalar_distance[:, 2] + 0.5 * scalar_distance[:, 3:].sum(axis=1)
        shortlist_n = min(512, len(train_rows))
        shortlist = np.argpartition(scalar_score, shortlist_n - 1)[:shortlist_n]
        hist_score = 0.5 * np.abs(train_hists[shortlist] - target_hists[idx]).sum(axis=1)
        order = shortlist[np.argsort(scalar_score[shortlist] + hist_score, kind="stable")]
        chosen = None
        for candidate in order:
            candidate = int(candidate)
            sig = signature_by_idx.get(candidate)
            if sig is None:
                sig = train_rows[candidate]["feature"]["structural_hash"]
                signature_by_idx[candidate] = sig
            if sig in selected_signatures:
                continue
            chosen = candidate
            selected_signatures.add(sig)
            break
        if chosen is None:
            chosen = int(order[0])
        selected.append(train_rows[chosen])
    records = [
        {
            "sample_id": i,
            "target_path": target_rows[i]["path"],
            "retrieved_train_path": selected[i]["path"],
            "retrieved_structural_hash": selected[i]["feature"]["structural_hash"],
        }
        for i in range(len(selected))
    ]
    (OUT / "unified_dedup11390_nearest_profile_retrieval_records.json").write_text(json.dumps(records, indent=2) + "\n")
    return selected


def render(rows: list[dict[str, Any]]) -> None:
    fields = [
        "Method",
        "Generated",
        "Reference",
        "Density W1",
        "Density MMD",
        "Degree MMD",
        "Rel. Components W1",
        "Type-mixing",
        "LCC W1",
        "Fanout TV",
    ]
    table_rows = []
    for row in rows:
        table_rows.append({
            "Method": row["method"],
            "Generated": row["generated_graph_count"],
            "Reference": row["reference_sample_count"],
            "Density W1": f"{row['density_W1']:.6f}",
            "Density MMD": f"{row['density_MMD']:.6f}",
            "Degree MMD": f"{row['degree_MMD']:.6f}",
            "Rel. Components W1": f"{row['relative_components_W1']:.6f}",
            "Type-mixing": f"{row['type_mixing_hA2_ratio']:.6f}",
            "LCC W1": f"{row['lcc_W1']:.6f}",
            "Fanout TV": f"{row['fanout_TV']:.6f}" if row["fanout_TV"] is not None else "N/A",
        })
    with (OUT / "unified_dedup11392_reference_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)
    lines = [
        "| " + " | ".join(fields) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    lines.extend([
        "",
        "Notes:",
        "- All rows use the same 11,390 development-deduplicated test subcircuits as the reference source.",
        "- AutoGraph, G2PT, DiGress, and retrieval rows use the source-clean graph view. The ELDA row uses the V5-projected view of the same 11,390 reference subcircuits and V5-projected ELDA outputs, matching the ELDA evaluation protocol.",
        "- Existing generated samples/checkpoints are fixed; no model is retrained or resampled.",
        "- Nearest-Profile-Retrieval is rerun against a deterministic 1024-target subset of the 11,390 development-deduplicated reference set using train-only candidates and structural-unique selection.",
        "- Rel. Components W1 = W1(C_gen, C_ref) / max(mean(C_ref), 1), where C is raw connected-component count per graph.",
        "- Type-mixing is generated/reference mean ratio of two-hop label homophily h(A^2,Y).",
        "- Fanout TV uses the canonical incidence proxy for graph artifacts in this table. The exact ELDA source-load Fanout TV should be reported from the native ELDA audit when needed.",
        "- Caveat: this is a strict deduplicated-reference sensitivity table. Native-view main table values can differ because ELDA and generic graph baselines do not share a lossless common semantic graph view.",
    ])
    (OUT / "unified_dedup11392_reference_metrics.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    V5_PROJECTION_CACHE.mkdir(parents=True, exist_ok=True)
    ref_paths = dedup_reference_paths()
    ref_rows = load_or_build_rows("reference_dedup11390", ref_paths, "reference", workers=16)
    rows = []
    for name, (kind, paths) in method_paths().items():
        gen_rows = load_or_build_rows("generated_" + name.lower().replace(" ", "_").replace("-", "_").replace(".", "").replace("/", "_"), paths, kind, workers=8)
        rows.append(summarize(name, ref_rows, gen_rows))
        print(f"[metrics] {name}", flush=True)
    elda_ref_paths = project_v5_paths(ref_paths, V5_PROJECTION_CACHE / "reference_v5", "reference", workers=8)
    elda_gen_paths = project_v5_paths(
        elda_generated_decoded_graph_paths(),
        V5_PROJECTION_CACHE / "elda_generated_v5",
        "elda",
        workers=8,
    )
    elda_ref_rows = load_or_build_rows("elda_v5_reference_dedup11390", elda_ref_paths, "graph", workers=16)
    elda_gen_rows = load_or_build_rows("elda_v5_generated_best7_topology_safe", elda_gen_paths, "graph", workers=8)
    rows.append(
        summarize(
            "Source-Net V6.1 best7epoch topology-safe (V5 projected)",
            elda_ref_rows,
            elda_gen_rows,
            reference_label="development-deduplicated ELDA test set projected through Source-Net V5 decode + canonical output completion",
        )
    )
    print("[metrics] Source-Net V6.1 best7epoch topology-safe (V5 projected)", flush=True)
    retrieval_rows = rerun_retrieval(ref_rows)
    rows.append(summarize("Nearest-Profile-Retrieval structural unique (rerun on dedup targets)", ref_rows, retrieval_rows))
    result = {
        "schema_version": "unified_dedup11390_reference_metrics_v1_elda_v5_projected",
        "reference_policy": "11,390 source-clean test subcircuits after removing exact ELDA object-sequence training or validation matches; ELDA row uses V5-projected view of the same reference subcircuits",
        "fixed_generated_samples": True,
        "retrained_models": False,
        "resampled_models": False,
        "elda_projection_policy": "V5 tokenize/decode plus canonical output completion for both reference and generated ELDA artifacts",
        "metrics": rows,
    }
    (OUT / "unified_dedup11392_reference_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    render(rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
