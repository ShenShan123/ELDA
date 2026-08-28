"""Portable paths shared by ELDA publication scripts."""

from __future__ import annotations

import os
from pathlib import Path


ELDA_ROOT = Path(
    os.environ.get("ELDA_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
PAPER_ROOT = Path(
    os.environ.get("ELDA_PAPER_ROOT", ELDA_ROOT / "paper")
).expanduser().resolve()
DATA_ROOT = Path(
    os.environ.get(
        "ELDA_DATA_ROOT",
        ELDA_ROOT / "datasets/CIRCUIT_SOURCE_NET_PARTITION_V6_1_CLEAN_COMPACT_LOAD",
    )
).expanduser().resolve()
V5_DATA_ROOT = Path(
    os.environ.get(
        "ELDA_V5_DATA_ROOT",
        ELDA_ROOT / "datasets/CIRCUIT_SOURCE_NET_PARTITION_V5_0_ROLE3FIX_DUALVIEW",
    )
).expanduser().resolve()
COMMON_DATA_ROOT = Path(
    os.environ.get(
        "ELDA_COMMON_DATA_ROOT",
        ELDA_ROOT / "datasets/CIRCUIT_PIN_SLOT_PARTITION_V2_1_GC_FULL",
    )
).expanduser().resolve()
WORK_DATA_ROOT = Path(
    os.environ.get("ELDA_WORK_DATA_ROOT", DATA_ROOT.parent)
).expanduser().resolve()
