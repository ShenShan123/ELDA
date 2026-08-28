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
        ELDA_ROOT / "datasets/elda",
    )
).expanduser().resolve()
PROJECTION_DATA_ROOT = Path(
    os.environ.get(
        "ELDA_PROJECTION_DATA_ROOT",
        ELDA_ROOT / "datasets/common_cell_projection",
    )
).expanduser().resolve()
COMMON_DATA_ROOT = Path(
    os.environ.get(
        "ELDA_COMMON_DATA_ROOT",
        ELDA_ROOT / "datasets/common_graph",
    )
).expanduser().resolve()
WORK_DATA_ROOT = Path(
    os.environ.get("ELDA_WORK_DATA_ROOT", DATA_ROOT.parent)
).expanduser().resolve()
