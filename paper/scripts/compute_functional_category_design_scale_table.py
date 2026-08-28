#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from elda_paths import PAPER_ROOT, WORK_DATA_ROOT

ROOT = PAPER_ROOT
DATA = WORK_DATA_ROOT
OUT = (
    ROOT
    / "reports/final_baseline/appendix_dataset_split_inventory_20260726"
)
FAMILY_AUDIT = OUT / "table_A3_base_family_function_audit_520.csv"
DESIGN_INVENTORY = OUT / "table_A4_design_inventory_full.csv"
SCALE_REPORT = (
    DATA / "expanded_raw_graph_scale_report_synth_variants_2026-04-13.csv"
)

EXPECTED_FAMILIES = 520
EXPECTED_VARIANTS = 1_003
EXPECTED_PARTITIONS = 191_910

CATEGORY_ORDER = [
    "Processor and SoC",
    "Hardware accelerator",
    "Cryptography",
    "DSP and multimedia",
    "Memory subsystem",
    "Interconnect and bus",
    "Communication and I/O",
    "Arithmetic datapath",
    "Control and miscellaneous",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def nearest_rank(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(quantile * len(ordered))) - 1]


def raw_graph_cell_net_counts(path: Path) -> tuple[int, int]:
    graph = torch.load(path, map_location="cpu", weights_only=False)
    x = graph.x.detach().cpu()
    if x.ndim == 1:
        labels = x
    elif x.ndim == 2 and x.shape[1] == 1:
        labels = x[:, 0]
    else:
        raise RuntimeError(f"unexpected raw-graph x shape at {path}: {tuple(x.shape)}")
    labels = labels.to(torch.int64)
    # The frozen complete mapped-design graph schema represents standard-cell
    # nodes by IDs 0..138 and connector/net nodes by -1.
    cell_count = int((labels != -1).sum().item())
    net_count = int((labels == -1).sum().item())
    if cell_count + net_count != int(graph.num_nodes):
        raise RuntimeError(f"node accounting failed at {path}")
    return cell_count, net_count


def fmt_median(value: float) -> str:
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,.1f}"


def main() -> None:
    family_rows = read_csv(FAMILY_AUDIT)
    inventory_rows = read_csv(DESIGN_INVENTORY)
    scale_rows = read_csv(SCALE_REPORT)
    scale_by_design = {row["design"]: row for row in scale_rows}

    if len(family_rows) != EXPECTED_FAMILIES:
        raise RuntimeError(f"expected {EXPECTED_FAMILIES} families")
    if len(inventory_rows) != EXPECTED_VARIANTS:
        raise RuntimeError(f"expected {EXPECTED_VARIANTS} design variants")
    if len(scale_by_design) != len(scale_rows):
        raise RuntimeError("scale report contains duplicate design IDs")

    family_category = {
        row["Base family"]: row["Primary function"]
        for row in family_rows
    }
    if len(family_category) != EXPECTED_FAMILIES:
        raise RuntimeError("family audit contains duplicate base families")

    enriched: list[dict[str, Any]] = []
    derived_from_graph = 0
    for row in inventory_rows:
        design_id = row["Design ID"]
        base_family = row["Base family"]
        scale = scale_by_design.get(design_id)
        if scale is None:
            raise RuntimeError(f"design missing from scale report: {design_id}")
        if family_category.get(base_family) != row["Primary function"]:
            raise RuntimeError(f"family/category mismatch: {design_id}")

        if scale.get("cells") and scale.get("nets"):
            mapped_cells = int(scale["cells"])
            mapped_nets = int(scale["nets"])
            size_source = "frozen_scale_report"
        else:
            mapped_cells, mapped_nets = raw_graph_cell_net_counts(
                Path(scale["pt_path"])
            )
            size_source = "complete_mapped_graph_node_labels"
            derived_from_graph += 1

        graph_nodes = int(scale["graph_nodes"])
        if mapped_cells + mapped_nets != graph_nodes:
            raise RuntimeError(
                f"mapped cell/net count does not equal graph nodes: {design_id}"
            )
        enriched.append(
            {
                **row,
                "Synthesis variant": scale["synth_variant"],
                "Mapped standard cells": mapped_cells,
                "Mapped nets": mapped_nets,
                "Mapped graph nodes": graph_nodes,
                "Mapped graph edges": int(scale["graph_edges"]),
                "Mapped-size source": size_source,
            }
        )

    if len({row["Design ID"] for row in enriched}) != EXPECTED_VARIANTS:
        raise RuntimeError("enriched inventory does not contain 1,003 unique variants")
    if sum(int(row["Gate-level subcircuits"]) for row in enriched) != EXPECTED_PARTITIONS:
        raise RuntimeError("partition total does not equal 191,910")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[row["Primary function"]].append(row)
    if set(grouped) != set(CATEGORY_ORDER):
        raise RuntimeError("functional category set differs from frozen taxonomy")

    summaries: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        rows = grouped[category]
        cells = [int(row["Mapped standard cells"]) for row in rows]
        nets = [int(row["Mapped nets"]) for row in rows]
        partitions = [int(row["Gate-level subcircuits"]) for row in rows]
        total_partitions = sum(partitions)
        summaries.append(
            {
                "Functional category": category,
                "Base families": len({row["Base family"] for row in rows}),
                "Design variants": len(rows),
                "Partitions": total_partitions,
                "Partition share": total_partitions / EXPECTED_PARTITIONS,
                "Mapped cells min": min(cells),
                "Mapped cells median": statistics.median(cells),
                "Mapped cells p95": nearest_rank(cells, 0.95),
                "Mapped cells max": max(cells),
                "Mapped nets min": min(nets),
                "Mapped nets median": statistics.median(nets),
                "Mapped nets p95": nearest_rank(nets, 0.95),
                "Mapped nets max": max(nets),
                "Partitions per variant min": min(partitions),
                "Partitions per variant median": statistics.median(partitions),
                "Partitions per variant p95": nearest_rank(partitions, 0.95),
                "Partitions per variant max": max(partitions),
            }
        )

    if sum(row["Base families"] for row in summaries) != EXPECTED_FAMILIES:
        raise RuntimeError("functional family total does not equal 520")
    if sum(row["Design variants"] for row in summaries) != EXPECTED_VARIANTS:
        raise RuntimeError("functional variant total does not equal 1,003")
    if sum(row["Partitions"] for row in summaries) != EXPECTED_PARTITIONS:
        raise RuntimeError("functional partition total does not equal 191,910")

    summary_csv = OUT / "table_A3_functional_category_design_scale.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    enriched_csv = OUT / "table_A4_design_inventory_full_with_scale.csv"
    with enriched_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = list(enriched[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(enriched)

    md_lines = [
        "# Functional-category design-scale summary",
        "",
        "| Functional category | Base families | Design variants | Partitions | Partition share | Mapped cells/variant: min / median / max | Partitions/variant: min / median / max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        md_lines.append(
            f"| {row['Functional category']} | {row['Base families']} | "
            f"{row['Design variants']} | {row['Partitions']:,} | "
            f"{row['Partition share']:.2%} | "
            f"{row['Mapped cells min']:,} / "
            f"{fmt_median(row['Mapped cells median'])} / "
            f"{row['Mapped cells max']:,} | "
            f"{row['Partitions per variant min']:,} / "
            f"{fmt_median(row['Partitions per variant median'])} / "
            f"{row['Partitions per variant max']:,} |"
        )
    md_lines.extend(
        [
            f"| **Total** | **{EXPECTED_FAMILIES}** | "
            f"**{EXPECTED_VARIANTS:,}** | **{EXPECTED_PARTITIONS:,}** | "
            "**100.00%** | **1 / 1,448 / 794,241** | "
            "**1 / 16 / 8,285** |",
            "",
            "Functional labels are assigned at the base-family level. Mapped "
            "cell counts are measured on each complete mapped design variant, "
            "excluding connector/net nodes. Partition counts are source-clean "
            "objects contributed by that variant to the final corpus. Medians "
            "use the conventional sample median. Category-level P95 mapped-cell, "
            "mapped-net, and partition-count statistics are retained in the CSV.",
        ]
    )
    summary_md = OUT / "table_A3_functional_category_design_scale.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "elda_functional_category_design_scale_v1",
        "statistical_units": {
            "functional_label": "base family",
            "mapped_design_size": "complete mapped design variant",
            "partition_contribution": "source-clean gate-level subcircuit",
        },
        "median_definition": "conventional sample median",
        "p95_definition": "nearest rank",
        "mapped_cell_definition": (
            "number of standard-cell nodes in the complete mapped-design graph; "
            "connector/net nodes with raw type -1 are excluded"
        ),
        "fallback_size_derivation": {
            "variant_count": derived_from_graph,
            "reason": "legacy 30pt rows have blank cells/nets in scale report",
            "rule": "cells=count(x != -1); nets=count(x == -1)",
        },
        "asserted_totals": {
            "base_families": EXPECTED_FAMILIES,
            "mapped_design_variants": EXPECTED_VARIANTS,
            "source_clean_partitions": EXPECTED_PARTITIONS,
        },
        "input_sha256": {
            FAMILY_AUDIT.name: sha256(FAMILY_AUDIT),
            DESIGN_INVENTORY.name: sha256(DESIGN_INVENTORY),
            SCALE_REPORT.name: sha256(SCALE_REPORT),
        },
        "output_sha256": {
            path.name: sha256(path)
            for path in (summary_csv, summary_md, enriched_csv)
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = OUT / "table_A3_design_scale_reproducibility_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
