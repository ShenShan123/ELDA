#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from elda_paths import DATA_ROOT, PAPER_ROOT

ROOT = PAPER_ROOT
DATASET = DATA_ROOT
CACHE = (
    ROOT
    / "reports/final_baseline/phase10_6_main_graph_quality_table/"
    "unified_dedup11390_cache_v1"
)
OUT = (
    ROOT
    / "reports/final_baseline/"
    "appendix_size_stratified_cell_inventory_20260730"
)

NET_NODE_TYPE = 139
BOUNDARY_NODE_TYPE = 140
CONNECTOR_TYPES = frozenset({NET_NODE_TYPE, BOUNDARY_NODE_TYPE, -1})
REFERENCE_EXPECTED = 11_390
CORPUS_EXPECTED = 191_910
CELL_EXPECTED = 16_116_889
INVENTORY_CHUNK = 512
WORKERS = 16

INPUTS = {
    "Reference": CACHE / "reference_dedup11392.jsonl",
    "AutoGraph": CACHE / "generated_autograph_labeled_llama_s_e20_best.jsonl",
    "G2PT": CACHE / "generated_g2pt_labeled_e25_best.jsonl",
    "DiGress": CACHE / "generated_digress_full.jsonl",
    "ELDA": CACHE / "panda_v5_generated_best7_topology_safe.jsonl",
}
ELDA_REFERENCE = CACHE / "panda_v5_reference_dedup11392.jsonl"
ELDA_NATIVE_GENERATED = (
    ROOT
    / "reports/final_baseline/phase10_6_main_graph_quality_table/"
    "unified_dedup11392_cache/"
    "generated_source_net_v61_best7epoch_topology_safe.jsonl"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def standard_cell_count(row: dict[str, Any]) -> int | None:
    feature = row["feature"]
    labels = feature.get("labels")
    node_count = feature.get("node_count")
    if (
        not isinstance(labels, list)
        or not isinstance(node_count, int)
        or len(labels) != node_count
        or any(
            not isinstance(label, (int, float)) or int(label) < 0
            for label in labels
        )
    ):
        return None
    return sum(int(label) not in CONNECTOR_TYPES for label in labels)


def nearest_rank(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    rank = max(1, int(np.ceil(quantile * len(ordered))))
    return int(ordered[rank - 1])


def size_bin(size: int, q1: int, q2: int) -> str:
    if size <= q1:
        return "Small"
    if size <= q2:
        return "Medium"
    return "Large"


def values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row["feature"][field]) for row in rows]


def wasserstein_1d(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    xs = sorted(set(left) | set(right))
    left = sorted(left)
    right = sorted(right)
    li = ri = 0
    lcdf = rcdf = total = 0.0
    previous = xs[0]
    for value in xs:
        total += abs(lcdf - rcdf) * (value - previous)
        while li < len(left) and left[li] <= value:
            li += 1
        while ri < len(right) and right[ri] <= value:
            ri += 1
        lcdf = li / len(left)
        rcdf = ri / len(right)
        previous = value
    return float(total)


def degree_hist_vector(row: dict[str, Any], max_degree: int = 256) -> np.ndarray:
    vector = np.zeros(max_degree + 1, dtype=np.float64)
    for degree in row["feature"]["degrees"]:
        vector[min(int(degree), max_degree)] += 1
    total = float(vector.sum())
    return vector / total if total else vector


def gaussian_degree_mmd2(
    reference: list[dict[str, Any]],
    generated: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    if not reference or not generated:
        return None, None
    left = np.stack([degree_hist_vector(row) for row in reference])
    right = np.stack([degree_hist_vector(row) for row in generated])

    # This intentionally reproduces the frozen main-metric bandwidth rule:
    # pool the first min(N, 256) rows of each population in artifact order,
    # use the median positive squared pairwise distance, and fall back to 1.
    pooled = np.concatenate(
        [
            left[: min(len(left), 256)],
            right[: min(len(right), 256)],
        ],
        axis=0,
    )
    distances: list[float] = []
    for index in range(min(len(pooled), 256)):
        delta = pooled[index + 1 :] - pooled[index]
        if len(delta):
            distances.extend(np.sum(delta * delta, axis=1).tolist())
    positive = [value for value in distances if value > 0]
    sigma2 = float(np.median(positive)) if positive else 1.0
    if sigma2 <= 0:
        sigma2 = 1.0

    def kernel_sum(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        b_norm = np.sum(b * b, axis=1)
        for start in range(0, len(a), 128):
            aa = a[start : start + 128]
            distance = (
                np.sum(aa * aa, axis=1)[:, None]
                + b_norm[None, :]
                - 2.0 * aa.dot(b.T)
            )
            np.maximum(distance, 0.0, out=distance)
            total += float(np.exp(-distance / (2.0 * sigma2)).sum())
        return total

    mmd2 = (
        kernel_sum(left, left) / (len(left) ** 2)
        + kernel_sum(right, right) / (len(right) ** 2)
        - 2.0 * kernel_sum(left, right) / (len(left) * len(right))
    )
    return float(mmd2), sigma2


def parse_hist(row: dict[str, Any]) -> Counter[str]:
    return Counter(
        {
            str(key): int(value)
            for key, value in (row.get("fanout_hist") or {}).items()
        }
    )


def merged_hist(rows: list[dict[str, Any]]) -> Counter[str]:
    result: Counter[str] = Counter()
    for row in rows:
        result.update(parse_hist(row))
    return result


def total_variation(left: Counter[str], right: Counter[str]) -> float | None:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        return None
    # Sort categorical keys so floating-point accumulation and serialized
    # artifacts are invariant to Python's per-process hash seed.
    keys = sorted(set(left) | set(right))
    return float(
        0.5
        * sum(
            abs(
                left.get(key, 0) / left_total
                - right.get(key, 0) / right_total
            )
            for key in keys
        )
    )


def fmt(value: float | None, digits: int = 6) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def compute_size_stratified() -> dict[str, Any]:
    rows = {name: load_jsonl(path) for name, path in INPUTS.items()}
    elda_reference = load_jsonl(ELDA_REFERENCE)
    elda_native_generated = load_jsonl(ELDA_NATIVE_GENERATED)
    if len(rows["Reference"]) != REFERENCE_EXPECTED:
        raise RuntimeError(
            f"expected {REFERENCE_EXPECTED} reference rows, "
            f"got {len(rows['Reference'])}"
        )
    if len(elda_reference) != REFERENCE_EXPECTED:
        raise RuntimeError(
            f"expected {REFERENCE_EXPECTED} ELDA-reference rows, "
            f"got {len(elda_reference)}"
        )
    if len(elda_native_generated) != len(rows["ELDA"]):
        raise RuntimeError(
            "native and common-view ELDA generated populations differ"
        )

    reference_sizes = [standard_cell_count(row) for row in rows["Reference"]]
    elda_reference_sizes = [standard_cell_count(row) for row in elda_reference]
    if any(value is None for value in reference_sizes):
        raise RuntimeError("reference contains size-unresolved rows")
    if any(value is None for value in elda_reference_sizes):
        raise RuntimeError("ELDA-projected reference contains size-unresolved rows")
    reference_sizes = [int(value) for value in reference_sizes]
    elda_reference_sizes = [int(value) for value in elda_reference_sizes]
    if reference_sizes != elda_reference_sizes:
        raise RuntimeError(
            "generic and ELDA-projected reference cell-count vectors differ"
        )
    q1 = nearest_rank(reference_sizes, 1.0 / 3.0)
    q2 = nearest_rank(reference_sizes, 2.0 / 3.0)
    bins = {
        "Small": {"minimum": None, "maximum": q1, "label": f"cells <= {q1}"},
        "Medium": {
            "minimum": q1 + 1,
            "maximum": q2,
            "label": f"{q1 + 1} <= cells <= {q2}",
        },
        "Large": {
            "minimum": q2 + 1,
            "maximum": None,
            "label": f"cells >= {q2 + 1}",
        },
    }

    partitioned: dict[str, dict[str, list[dict[str, Any]]]] = {}
    unresolved: dict[str, list[str]] = {}
    for method, method_rows in rows.items():
        partitioned[method] = {name: [] for name in bins}
        unresolved[method] = []
        for row in method_rows:
            count = standard_cell_count(row)
            if count is None:
                unresolved[method].append(str(row.get("path") or ""))
                continue
            name = size_bin(count, q1, q2)
            partitioned[method][name].append(row)
    elda_reference_by_bin = {name: [] for name in bins}
    for row in elda_reference:
        count = standard_cell_count(row)
        if count is None:
            raise RuntimeError("ELDA-projected reference size became unresolved")
        name = size_bin(count, q1, q2)
        elda_reference_by_bin[name].append(row)
    elda_native_generated_by_bin = {name: [] for name in bins}
    native_sizes = []
    for row in elda_native_generated:
        count = standard_cell_count(row)
        if count is None:
            raise RuntimeError("native ELDA generated size became unresolved")
        native_sizes.append(count)
        name = size_bin(count, q1, q2)
        elda_native_generated_by_bin[name].append(row)
    projected_sizes = [
        standard_cell_count(row)
        for row in rows["ELDA"]
    ]
    if native_sizes != projected_sizes:
        raise RuntimeError(
            "native and common-view ELDA cell-count vectors differ"
        )

    counts = {
        method: {
            name: len(partitioned[method][name])
            for name in bins
        }
        for method in rows
    }
    metrics: list[dict[str, Any]] = []
    for method in ("AutoGraph", "G2PT", "DiGress", "ELDA"):
        for bin_name in bins:
            reference = (
                elda_reference_by_bin[bin_name]
                if method == "ELDA"
                else partitioned["Reference"][bin_name]
            )
            generated = partitioned[method][bin_name]
            fanout_reference = partitioned["Reference"][bin_name]
            fanout_generated = (
                elda_native_generated_by_bin[bin_name]
                if method == "ELDA"
                else generated
            )
            degree_mmd2, sigma2 = gaussian_degree_mmd2(
                reference, generated
            )
            metrics.append(
                {
                    "method": method,
                    "size_bin": bin_name,
                    "size_range": bins[bin_name]["label"],
                    "reference_n": len(reference),
                    "generated_n": len(generated),
                    "density_W1": wasserstein_1d(
                        values(reference, "edge_density"),
                        values(generated, "edge_density"),
                    ),
                    "degree_MMD2": degree_mmd2,
                    "degree_kernel_sigma2": sigma2,
                    "LCC_W1": wasserstein_1d(
                        values(reference, "lcc_ratio"),
                        values(generated, "lcc_ratio"),
                    ),
                    "fanout_TV": total_variation(
                        merged_hist(fanout_reference),
                        merged_hist(fanout_generated),
                    ),
                    "fanout_observation": (
                        "exact native source-load assignment"
                        if method == "ELDA"
                        else "common-view incidence proxy"
                    ),
                }
            )

    input_hashes = {
        path.name: sha256(path)
        for path in [
            *INPUTS.values(),
            ELDA_REFERENCE,
            ELDA_NATIVE_GENERATED,
        ]
    }
    payload = {
        "schema_version": "elda_appendix_size_stratified_v1",
        "reference_population": REFERENCE_EXPECTED,
        "size_definition": (
            "number of standard-cell nodes; node types 139 (NET), "
            "140 (BOUNDARY_STUB), and -1 are excluded"
        ),
        "bin_rule": (
            "nearest-rank tertiles of the fixed 11,390-object reference; "
            "boundaries are computed once and reused for every method"
        ),
        "bins": bins,
        "common_projection": (
            "frozen undirected simple cell/net/boundary graph view; "
            "ELDA and its reference are projected through the same V5 "
            "decode plus canonical output completion"
        ),
        "empty_bin_policy": (
            "report N=0 and all distribution metrics as N/A; "
            "do not merge bins or substitute another reference"
        ),
        "size_unresolved_policy": (
            "a generated graph without one node-type value per declared node "
            "cannot be assigned a standard-cell count and is excluded from "
            "the conditional size-stratified population; paths and counts "
            "are reported explicitly"
        ),
        "reference_policy": (
            "each generated size bin is compared only with the corresponding "
            "size bin of the fixed 11,390-object reference"
        ),
        "degree_mmd2": {
            "histogram_support": "degrees 0..255 plus overflow >=256",
            "normalization": "per-graph probability vector",
            "estimator": "biased squared Gaussian-kernel MMD",
            "bandwidth": (
                "median positive squared pairwise distance over the frozen "
                "pooled prefixes (first min(N,256) reference and generated "
                "rows in artifact order), fallback sigma^2=1"
            ),
        },
        "fanout_TV": {
            "distribution": "pooled P(source class, fanout bin)",
            "reference": (
                "exact native source-load assignment from the corresponding "
                "fixed 11,390-object reference size bin"
            ),
            "generic_generated": (
                "common-view incidence proxy: internal NET cell incidence "
                "minus one assumed driver; BOUNDARY_STUB cell incidence"
            ),
            "ELDA_generated": "exact native source-load assignment",
            "bins": ["0", "1", "2", "3-4", "5-8", "9-16", "17+"],
        },
        "counts": counts,
        "size_unresolved": {
            method: {
                "count": len(paths),
                "paths": paths,
            }
            for method, paths in unresolved.items()
        },
        "metrics": metrics,
        "input_sha256": input_hashes,
    }
    return payload


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def count_type_chunk(paths: list[str]) -> dict[int, int]:
    counter: Counter[int] = Counter()
    for path_text in paths:
        graph = torch.load(
            path_text,
            map_location="cpu",
            weights_only=False,
        )
        x = graph.x.detach().cpu()
        if x.dim() == 1:
            labels = x.tolist()
        elif x.dim() == 2 and x.shape[1] == 1:
            labels = x[:, 0].tolist()
        else:
            labels = x.argmax(dim=1).tolist()
        counter.update(
            int(label)
            for label in labels
            if int(label) not in CONNECTOR_TYPES
        )
    return dict(counter)


def count_split(paths: list[str]) -> Counter[int]:
    result: Counter[int] = Counter()
    jobs = list(chunks(paths, INVENTORY_CHUNK))
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for index, partial in enumerate(
            pool.map(count_type_chunk, jobs, chunksize=1),
            start=1,
        ):
            result.update(partial)
            if index % 50 == 0 or index == len(jobs):
                print(
                    f"[cell-inventory] chunks {index}/{len(jobs)}",
                    flush=True,
                )
    return result


def load_split_paths(name: str) -> list[str]:
    path = DATASET / f"{name}_source_clean.txt"
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compute_cell_inventory() -> dict[str, Any]:
    meta = torch.load(
        DATASET / "meta.pt",
        map_location="cpu",
        weights_only=False,
    )
    vocab = {
        int(key): str(value)
        for key, value in meta["cell_type_vocab"].items()
    }
    split_paths = {
        "train": load_split_paths("train"),
        "val": load_split_paths("val"),
        "test": load_split_paths("test"),
    }
    if sum(map(len, split_paths.values())) != CORPUS_EXPECTED:
        raise RuntimeError("final corpus path count does not equal 191,910")

    split_counts: dict[str, Counter[int]] = {}
    for split, paths in split_paths.items():
        print(
            f"[cell-inventory] {split}: {len(paths)} subcircuits",
            flush=True,
        )
        split_counts[split] = count_split(paths)
    total: Counter[int] = Counter()
    for counter in split_counts.values():
        total.update(counter)
    if sum(total.values()) != CELL_EXPECTED:
        raise RuntimeError(
            f"expected {CELL_EXPECTED} cells, got {sum(total.values())}"
        )

    rows = []
    for type_id, count in sorted(
        total.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        rows.append(
            {
                "rank": len(rows) + 1,
                "type_id": type_id,
                "cell_type": vocab.get(type_id, f"TYPE_{type_id}"),
                "train_count": split_counts["train"].get(type_id, 0),
                "validation_count": split_counts["val"].get(type_id, 0),
                "test_count": split_counts["test"].get(type_id, 0),
                "total_count": count,
                "total_fraction": count / CELL_EXPECTED,
            }
        )

    observed = {
        split: sorted(
            type_id
            for type_id, count in counter.items()
            if count > 0
        )
        for split, counter in split_counts.items()
    }
    val_only = sorted(set(observed["val"]) - set(observed["train"]))
    test_only = sorted(set(observed["test"]) - set(observed["train"]))
    payload = {
        "schema_version": "elda_standard_cell_inventory_v1",
        "population": {
            "subcircuits": {
                split: len(paths)
                for split, paths in split_paths.items()
            },
            "cells": {
                split: sum(counter.values())
                for split, counter in split_counts.items()
            },
            "total_subcircuits": sum(map(len, split_paths.values())),
            "total_cells": sum(total.values()),
        },
        "library_vocabulary_size": len(vocab),
        "observed_type_count": len(total),
        "split_observed_type_count": {
            split: len(ids) for split, ids in observed.items()
        },
        "validation_types_unseen_in_train": [
            {"type_id": value, "cell_type": vocab.get(value)}
            for value in val_only
        ],
        "test_types_unseen_in_train": [
            {"type_id": value, "cell_type": vocab.get(value)}
            for value in test_only
        ],
        "long_tail": {
            "types_with_fewer_than_100_instances": sum(
                count < 100 for count in total.values()
            ),
            "types_with_fewer_than_1000_instances": sum(
                count < 1_000 for count in total.values()
            ),
            "types_with_fewer_than_10000_instances": sum(
                count < 10_000 for count in total.values()
            ),
            "top10_instance_share": (
                sum(row["total_count"] for row in rows[:10])
                / CELL_EXPECTED
            ),
            "top15_instance_share": (
                sum(row["total_count"] for row in rows[:15])
                / CELL_EXPECTED
            ),
        },
        "rows": rows,
        "input_sha256": {
            "meta.pt": sha256(DATASET / "meta.pt"),
            "mapping_v61.txt": sha256(DATASET / "mapping_v61.txt"),
            **{
                f"{split}_source_clean.txt": sha256(
                    DATASET / f"{split}_source_clean.txt"
                )
                for split in split_paths
            },
        },
    }
    return payload


def write_size_outputs(payload: dict[str, Any]) -> None:
    (OUT / "size_stratified_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    count_fields = ["Size bin", "Range", "Reference", "AutoGraph", "G2PT", "DiGress", "ELDA"]
    with (OUT / "size_stratified_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=count_fields)
        writer.writeheader()
        for bin_name, spec in payload["bins"].items():
            writer.writerow(
                {
                    "Size bin": bin_name,
                    "Range": spec["label"],
                    **{
                        method: payload["counts"][method][bin_name]
                        for method in (
                            "Reference", "AutoGraph", "G2PT", "DiGress", "ELDA"
                        )
                    },
                }
            )
        writer.writerow(
            {
                "Size bin": "Size unresolved",
                "Range": "missing/inconsistent node-type vector",
                **{
                    method: payload["size_unresolved"][method]["count"]
                    for method in (
                        "Reference", "AutoGraph", "G2PT", "DiGress", "ELDA"
                    )
                },
            }
        )
    metric_fields = [
        "Method",
        "Size bin",
        "Range",
        "Reference N",
        "Generated N",
        "Density W1",
        "Degree MMD2",
        "Degree sigma2",
        "LCC W1",
        "Fanout TV",
    ]
    with (OUT / "size_stratified_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for row in payload["metrics"]:
            writer.writerow(
                {
                    "Method": row["method"],
                    "Size bin": row["size_bin"],
                    "Range": row["size_range"],
                    "Reference N": row["reference_n"],
                    "Generated N": row["generated_n"],
                    "Density W1": row["density_W1"],
                    "Degree MMD2": row["degree_MMD2"],
                    "Degree sigma2": row["degree_kernel_sigma2"],
                    "LCC W1": row["LCC_W1"],
                    "Fanout TV": row["fanout_TV"],
                }
            )


def write_inventory_outputs(payload: dict[str, Any]) -> None:
    (OUT / "standard_cell_inventory.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = [
        "rank",
        "type_id",
        "cell_type",
        "train_count",
        "validation_count",
        "test_count",
        "total_count",
        "total_fraction",
    ]
    with (OUT / "standard_cell_inventory_96.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["rows"])


def write_markdown(
    size_payload: dict[str, Any],
    inventory_payload: dict[str, Any],
) -> None:
    lines = [
        "# Size-stratified distribution and standard-cell inventory",
        "",
        "## Size-stratified evaluation",
        "",
        "Size is the number of standard-cell nodes in the frozen common "
        "cell/net/boundary projection. The nearest-rank tertiles of the "
        "fixed 11,390-object reference are computed once and reused for "
        "every method. Each generated bin is compared only with the "
        "corresponding reference bin.",
        "",
        "| Size bin | Cell-count range | Reference N | AutoGraph N | G2PT N | DiGress N | ELDA N |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bin_name, spec in size_payload["bins"].items():
        lines.append(
            f"| {bin_name} | {spec['label']} | "
            + " | ".join(
                str(size_payload["counts"][method][bin_name])
                for method in (
                    "Reference",
                    "AutoGraph",
                    "G2PT",
                    "DiGress",
                    "ELDA",
                )
            )
            + " |"
        )
    lines.append(
        "| Size unresolved | missing/inconsistent node-type vector | "
        + " | ".join(
            str(size_payload["size_unresolved"][method]["count"])
            for method in (
                "Reference",
                "AutoGraph",
                "G2PT",
                "DiGress",
                "ELDA",
            )
        )
        + " |"
    )
    lines.extend(
        [
            "",
            "| Method | Size bin | Ref. N | Gen. N | Density W1 ↓ | Degree MMD² ↓ | σ² | LCC W1 ↓ | Fanout TV ↓ |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in size_payload["metrics"]:
        lines.append(
            f"| {row['method']} | {row['size_bin']} | "
            f"{row['reference_n']} | {row['generated_n']} | "
            f"{fmt(row['density_W1'])} | {fmt(row['degree_MMD2'])} | "
            f"{fmt(row['degree_kernel_sigma2'])} | "
            f"{fmt(row['LCC_W1'])} | {fmt(row['fanout_TV'])} |"
        )
    lines.extend(
        [
            "",
            "Degree MMD² uses the same 257-bin normalized degree histogram, "
            "biased Gaussian-kernel estimator, and deterministic pooled-prefix "
            "median bandwidth rule as the frozen aggregate metric. An empty "
            "generated or reference bin is reported as N=0 and N/A without "
            "bin merging. Fanout TV uses the exact native source-load "
            "reference in every bin. ELDA generated fanout is exact; generic "
            "baseline fanout uses the frozen common-view incidence proxy "
            "because native source identity is unavailable. Graphs without one valid "
            "node-type value per declared node are listed as size unresolved "
            "and are not silently assigned using total node count.",
            "",
            "## Standard-cell inventory and long-tail composition",
            "",
        ]
    )
    population = inventory_payload["population"]
    long_tail = inventory_payload["long_tail"]
    lines.extend(
        [
            f"The final 191,910-subcircuit corpus contains "
            f"{population['total_cells']:,} standard-cell instances and "
            f"{inventory_payload['observed_type_count']} observed types from "
            f"the {inventory_payload['library_vocabulary_size']}-entry mapping. "
            f"Train/validation/test cover "
            f"{inventory_payload['split_observed_type_count']['train']}/"
            f"{inventory_payload['split_observed_type_count']['val']}/"
            f"{inventory_payload['split_observed_type_count']['test']} types. "
            f"Validation-only and test-only types relative to training: "
            f"{len(inventory_payload['validation_types_unseen_in_train'])} and "
            f"{len(inventory_payload['test_types_unseen_in_train'])}.",
            "",
            f"Long-tail counts are {long_tail['types_with_fewer_than_100_instances']} "
            f"types with fewer than 100 instances, "
            f"{long_tail['types_with_fewer_than_1000_instances']} with fewer "
            f"than 1,000, and "
            f"{long_tail['types_with_fewer_than_10000_instances']} with fewer "
            f"than 10,000. The top 10 and top 15 types account for "
            f"{long_tail['top10_instance_share']:.2%} and "
            f"{long_tail['top15_instance_share']:.2%} of all cells.",
            "",
            "| Rank | Nangate45 cell type | Train | Validation | Test | Total | Share |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in inventory_payload["rows"][:15]:
        lines.append(
            f"| {row['rank']} | {row['cell_type']} | "
            f"{row['train_count']:,} | {row['validation_count']:,} | "
            f"{row['test_count']:,} | {row['total_count']:,} | "
            f"{row['total_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The complete 96-type splitwise inventory is provided in "
            "`standard_cell_inventory_96.csv`. Counts are cell-instance "
            "counts; `NET` and `BOUNDARY_STUB` connector nodes are excluded.",
        ]
    )
    (OUT / "appendix_tables.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    size_payload = compute_size_stratified()
    write_size_outputs(size_payload)
    inventory_payload = compute_cell_inventory()
    write_inventory_outputs(inventory_payload)
    write_markdown(size_payload, inventory_payload)
    manifest = {
        "schema_version": "appendix_size_and_cell_inventory_manifest_v1",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "outputs": {
            path.name: sha256(path)
            for path in sorted(OUT.iterdir())
            if path.is_file() and path.name != "reproducibility_manifest.json"
        },
    }
    (OUT / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
