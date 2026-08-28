#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from elda_paths import COMMON_DATA_ROOT, PAPER_ROOT

ROOT = PAPER_ROOT
DATASET = COMMON_DATA_ROOT
OUT = ROOT / "reports/final_baseline/phase10_2_final_common_tables"
TABLE_CSV = OUT / "tables/table1_graph_candidate_quality.csv"
TABLE_MD = OUT / "markdown/table1_graph_candidate_quality.md"
REPORT_DIR = OUT / "reports"


def scalar_labels(graph: Any) -> list[int]:
    x = getattr(graph, "x", None)
    if x is None:
        return []
    x = x.detach().cpu()
    if x.ndim == 1:
        return [int(v) for v in x.tolist()]
    if x.ndim == 2 and x.shape[1] == 1:
        return [int(v[0]) for v in x.tolist()]
    if x.ndim == 2 and x.shape[0] > 0:
        return [int(v) for v in x.argmax(dim=1).tolist()]
    return []


def canonical_signature_from_pt(path: Path) -> str:
    graph = torch.load(path, map_location="cpu", weights_only=False)
    num_nodes = int(getattr(graph, "num_nodes", 0) or 0)
    edge_index = getattr(graph, "edge_index", None)
    edges: list[tuple[int, int]] = []
    if edge_index is not None:
        edge_index = edge_index.detach().cpu()
        for u, v in edge_index.t().tolist():
            ui, vi = int(u), int(v)
            if ui != vi:
                edges.append((min(ui, vi), max(ui, vi)))
    labels = scalar_labels(graph)
    if len(labels) != num_nodes:
        labels = []
    payload = {
        "node_count": num_nodes,
        "node_types": labels,
        "edges": sorted(set(edges)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json_node_types(obj: dict[str, Any], node_count: int) -> list[Any]:
    labels = obj.get("node_types")
    if isinstance(labels, list) and len(labels) == node_count:
        return labels
    nodes = obj.get("nodes")
    if isinstance(nodes, list) and nodes and all(isinstance(n, dict) for n in nodes):
        out = []
        for n in nodes:
            out.append(n.get("type", n.get("node_type", n.get("label"))))
        if len(out) == node_count and all(v is not None for v in out):
            return out
    return []


def canonical_signature_from_json(path: Path) -> str:
    obj = json.loads(path.read_text())
    nodes = obj.get("nodes", [])
    node_count = len(nodes) if isinstance(nodes, list) else int(obj.get("node_count", 0) or 0)
    raw_edges = obj.get("edges", [])
    edges: list[tuple[int, int]] = []
    for e in raw_edges:
        if isinstance(e, dict):
            u = e.get("source", e.get("src", e.get("u")))
            v = e.get("target", e.get("dst", e.get("v")))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            u, v = e[0], e[1]
        else:
            continue
        try:
            ui, vi = int(u), int(v)
        except Exception:
            continue
        if ui != vi:
            edges.append((min(ui, vi), max(ui, vi)))
    labels = _json_node_types(obj, node_count)
    payload = {
        "node_count": node_count,
        "node_types": labels,
        "edges": sorted(set(edges)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_signature(path: Path) -> str:
    if path.suffix == ".pt":
        return canonical_signature_from_pt(path)
    if path.suffix == ".json":
        return canonical_signature_from_json(path)
    raise ValueError(f"unsupported graph artifact: {path}")


def train_paths() -> list[Path]:
    manifest = DATASET / "full_manifest.csv"
    paths: list[Path] = []
    with manifest.open() as f:
        for row in csv.DictReader(f):
            if row.get("split") == "train":
                p = Path(row["path"])
                if p.exists():
                    paths.append(p)
    return paths


def build_or_load_train_hashes(cache_path: Path) -> set[str]:
    if cache_path.exists():
        return set(x.strip() for x in cache_path.read_text().splitlines() if x.strip())
    paths = train_paths()
    hashes: set[str] = set()
    failures = 0
    for i, path in enumerate(paths, 1):
        try:
            hashes.add(canonical_signature(path))
        except Exception:
            failures += 1
        if i % 20000 == 0:
            print(f"[train-fingerprint] {i}/{len(paths)} paths, unique={len(hashes)}, failures={failures}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(sorted(hashes)) + "\n")
    meta = {
        "dataset": str(DATASET),
        "split": "train",
        "input_paths": len(paths),
        "unique_fingerprints": len(hashes),
        "load_failures": failures,
        "definition": "sha256 of canonical undirected edge set plus node_count and scalar/argmax node type labels when available",
    }
    (cache_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2) + "\n")
    return hashes


def artifact_paths_for_method(method: str) -> list[Path]:
    if method == "AutoGraph":
        return sorted(
            (ROOT / "results/autograph_native/phase4_full_labeled_llama_s_max2048_bf16_nw8_cyfix_v4/raw_outputs").glob("*_graph.pt")
        )
    if method == "G2PT-full-e20-best":
        return sorted(
            (ROOT / "results/g2pt/phase6_full_labeled_e25_v2_best_resample1024/decoded_graphs").glob("generated_sequence_*.json")
        )
    if method == "DiGress-full":
        return sorted((ROOT / "results/digress/phase9_full_optional_e20/decoded_graphs").glob("digress_*.json"))
    if method == "Nearest-Profile-Retrieval":
        records = json.loads(
            (ROOT / "results/retrieval/nearest_profile_gc_full_n1024_structural_unique/retrieval_records.json").read_text()
        )
        paths = []
        for row in records:
            p = ROOT / row["artifact_path"]
            if p.exists():
                paths.append(p)
        return paths
    return []


def compute_method_novelty(method: str, train_hashes: set[str]) -> dict[str, Any]:
    paths = artifact_paths_for_method(method)
    hashes = []
    failures = []
    for path in paths:
        try:
            hashes.append(canonical_signature(path))
        except Exception as exc:
            failures.append({"path": str(path), "error": repr(exc)})
    valid = len(hashes)
    unique_hashes = set(hashes)
    novel = sum(1 for h in hashes if h not in train_hashes)
    novel_unique = len(unique_hashes - train_hashes)
    return {
        "method_name": method,
        "artifact_count": len(paths),
        "valid_fingerprint_count": valid,
        "unique_valid_fingerprint_count": len(unique_hashes),
        "duplicate_valid_count": valid - len(unique_hashes),
        "novel_count": novel,
        "train_exact_copy_count": valid - novel,
        "novel_unique_count": novel_unique,
        "unique_rate_over_valid": (
            len(unique_hashes) / valid if valid else None
        ),
        "canonical_novelty": (novel / valid) if valid else None,
        "VUN_rate_over_1024_attempts": novel_unique / 1024,
        "fingerprint_failures": len(failures),
        "failure_examples": failures[:5],
    }


def fmt_value(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_table_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_table_md(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    lines.append("")
    lines.append(
        "Novel↑ is canonical exact-copy novelty: among valid generated graph fingerprints, "
        "the fraction whose canonical graph fingerprint is not present in the GC_FULL train split. "
        "It is an anti-memorization check, not a claim of functional or semantic novelty."
    )
    path.write_text("\n".join(lines) + "\n")


def update_table(report: dict[str, Any]) -> None:
    rows = read_table(TABLE_CSV)
    fields = list(rows[0].keys())
    if "Novel↑" not in fields:
        insert_at = fields.index("unique_ratio") + 1 if "unique_ratio" in fields else len(fields)
        fields.insert(insert_at, "Novel↑")
    novelty = {r["method_name"]: r["canonical_novelty"] for r in report["methods"]}
    for row in rows:
        method = row["method_name"]
        if method in novelty:
            row["Novel↑"] = fmt_value(novelty[method])
        elif method == "Source-Net":
            row["Novel↑"] = "N/A"
        else:
            row["Novel↑"] = row.get("Novel↑", "N/A") or "N/A"
    write_table_csv(TABLE_CSV, rows, fields)
    write_table_md(TABLE_MD, rows, fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default=str(REPORT_DIR / "gc_full_train_canonical_fingerprints.txt"))
    parser.add_argument("--update-table", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_hashes = build_or_load_train_hashes(Path(args.cache_path))
    methods = [
        "AutoGraph",
        "G2PT-full-e20-best",
        "DiGress-full",
        "Nearest-Profile-Retrieval",
    ]
    method_reports = [compute_method_novelty(m, train_hashes) for m in methods]
    report = {
        "metric": "Canonical Novelty",
        "definition": "Novel = |{valid generated canonical graph fingerprints not in GC_FULL train fingerprints}| / |valid generated canonical graph fingerprints|",
        "scope_note": "Exact-copy anti-memorization check only; not functional or semantic novelty.",
        "train_fingerprint_count": len(train_hashes),
        "methods": method_reports,
    }
    report_path = REPORT_DIR / "table1_canonical_novelty_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.update_table:
        update_table(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
