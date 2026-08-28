#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


EXPECTED = {"train": 171192, "val": 9639, "test": 11574}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-root", required=True)
    parser.add_argument("--clean-split-dir", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    source = Path(args.v5_root)
    clean = Path(args.clean_split_dir)
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partitions").mkdir(exist_ok=True)

    split_lists = {}
    counts = {}
    for split, expected in EXPECTED.items():
        source_list = clean / f"{split}_source_clean.txt"
        paths = [line.strip() for line in source_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(paths) != expected:
            raise RuntimeError(f"{split} clean count {len(paths)} != expected {expected}")
        if len(set(paths)) != len(paths):
            raise RuntimeError(f"{split} clean list contains duplicates")
        target = out / f"{split}_source_clean.txt"
        shutil.copyfile(source_list, target)
        split_lists[split] = str(target)
        counts[split] = len(paths)

    meta = torch.load(source / "meta.pt", map_location="cpu", weights_only=False)
    meta.update({
        "dataset_name": "CIRCUIT_SOURCE_NET_PARTITION_V6_1_CLEAN_COMPACT_LOAD",
        "source_dataset_root": str(meta.get("source_dataset_root")),
        "cell_mapping": str(out / "mapping_v61.txt"),
        "file_backed_partitions": True,
        "split_file_lists": split_lists,
        "clean_split_counts": counts,
        "source_invalid_excluded": True,
        "source_invalid_retained_for_all_reference_reporting": True,
        "tokenizer_type": "source_net_v61",
        "serializer_version": "source_net_v6_1_clean_compact_load_v1",
        "tokenizer_version": "source_net_v6_1_clean_compact_load_v1",
        "max_sequence_length": 24576,
        "truncation_disabled": True,
    })
    torch.save(meta, out / "meta.pt")
    shutil.copyfile(source / "mapping_v5.txt", out / "mapping_v61.txt")
    manifest = {
        "dataset_name": meta["dataset_name"],
        "counts": counts,
        "source_root": meta["source_dataset_root"],
        "split_file_lists": split_lists,
        "tokenizer": "V6.1 compact-load",
        "max_sequence_length": 24576,
        "truncation": "disabled",
        "source_invalid_policy": "excluded from training/validation/test clean splits; retained externally for all-reference reporting",
    }
    (out / "v61_dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "serializer_version.json").write_text(json.dumps({"serializer_version": meta["serializer_version"]}, indent=2) + "\n")
    (out / "tokenizer_version.json").write_text(json.dumps({"tokenizer_version": meta["tokenizer_version"]}, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
