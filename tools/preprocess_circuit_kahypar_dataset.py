#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from circuit_kahypar.preprocess import preprocess_dataset


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a metadata-rich KaHyPar circuit dataset for ELDA.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--manifest-csv", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--kahypar-bin", default=os.environ.get("ELDA_KAHYPAR_BIN"), required=not os.environ.get("ELDA_KAHYPAR_BIN"))
    parser.add_argument("--kahypar-preset", default=os.environ.get("ELDA_KAHYPAR_PRESET"), required=not os.environ.get("ELDA_KAHYPAR_PRESET"))
    parser.add_argument("--kahypar-threads", type=int, default=0, help="Set OMP_NUM_THREADS for KaHyPar. 0 means default.")
    parser.add_argument("--partition-cap", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-buckets", type=int, default=4)
    parser.add_argument("--limit-designs", type=int, default=0)
    parser.add_argument("--cell-mapping", default=str(root / "mapping.txt"))
    parser.add_argument("--net-id", type=int, default=0, help="Override net node label. Default 0 means derive dynamically from mapping.txt.")
    parser.add_argument("--boundary-stub-id", type=int, default=0, help="Override boundary stub label. Default 0 means derive dynamically from mapping.txt.")
    parser.add_argument("--resume-log", default="", help="Optional log file path to skip already finished designs.")
    parser.add_argument(
        "--semantic-cut-bias",
        default="semantic_v1",
        choices=["none", "semantic_v1"],
        help="Bias KaHyPar edge weights toward more semantic-friendly cuts.",
    )
    parser.add_argument(
        "--resume-from-output-root",
        default="",
        help="Optional existing output root to reuse finalized manifests/checkpoints from.",
    )
    args = parser.parse_args()

    preprocess_dataset(
        input_dir=Path(args.input_dir) if args.input_dir else None,
        output_root=Path(args.output_root),
        kahypar_bin=Path(args.kahypar_bin),
        preset=Path(args.kahypar_preset),
        partition_cap=args.partition_cap,
        net_id=(int(args.net_id) if int(args.net_id) > 0 else None),
        boundary_stub_id=(int(args.boundary_stub_id) if int(args.boundary_stub_id) > 0 else None),
        epsilon=args.epsilon,
        split_seed=args.split_seed,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        num_buckets=args.num_buckets,
        limit_designs=args.limit_designs,
        cell_mapping=Path(args.cell_mapping) if args.cell_mapping else None,
        manifest_csv=Path(args.manifest_csv) if args.manifest_csv else None,
        kahypar_threads=(int(args.kahypar_threads) if int(args.kahypar_threads) > 0 else None),
        resume_log=Path(args.resume_log) if args.resume_log else None,
        resume_from_output_root=Path(args.resume_from_output_root) if args.resume_from_output_root else None,
        semantic_cut_bias=str(args.semantic_cut_bias),
    )


if __name__ == "__main__":
    main()
