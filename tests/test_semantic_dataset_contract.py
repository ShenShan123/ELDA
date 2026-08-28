from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KAHYPAR_BIN = os.environ.get("ELDA_KAHYPAR_BIN")
KAHYPAR_PRESET = os.environ.get("ELDA_KAHYPAR_PRESET")
pytestmark = pytest.mark.skipif(
    not KAHYPAR_BIN or not KAHYPAR_PRESET,
    reason="set ELDA_KAHYPAR_BIN and ELDA_KAHYPAR_PRESET",
)

from elda.datamodules import GraphDataset
from elda.utils.circuit_semantic_contract import validate_semantic_dataset_root
from circuit_kahypar.preprocess import preprocess_dataset


def _toy_graph() -> Data:
    x = torch.tensor([0, 1, 15, 24, 33, 42, -1, -1, -1, -1], dtype=torch.float32).reshape(-1, 1)
    undirected_edges = [
        (0, 6), (1, 6),
        (1, 7), (2, 7),
        (2, 8), (3, 8), (4, 8),
        (4, 9), (5, 9),
    ]
    edge_pairs = undirected_edges + [(b, a) for a, b in undirected_edges]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, num_nodes=int(x.size(0)))


def _build_dataset_root(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir(parents=True)
    torch.save(_toy_graph(), input_dir / "toy_design.pt")
    preprocess_dataset(
        input_dir=input_dir,
        output_root=output_dir,
        kahypar_bin=Path(KAHYPAR_BIN),
        preset=Path(KAHYPAR_PRESET),
        partition_cap=5,
        split_seed=7,
        split_ratios=(1.0, 0.0, 0.0),
        num_buckets=2,
    )
    gate_profile = {
        "manifest_count": 1,
        "total_gates": 6,
        "used_type_count": 6,
        "medium_types": [],
    }
    (output_dir / "gate_type_profile.json").write_text(json.dumps(gate_profile), encoding="utf-8")
    return output_dir


def test_validate_semantic_dataset_root_ok(tmp_path):
    root = _build_dataset_root(tmp_path)
    report = validate_semantic_dataset_root(
        root=root,
        dataset_names=["CIRCUIT_PARTITION", "CIRCUIT_SKELETON"],
        schema_fields=[
            "size_bucket",
            "stub_bucket",
            "pin_bucket",
            "pin_deficit_bucket",
            "underconnected_bucket",
            "synthetic_input_bucket",
        ],
        require_gate_profile=True,
    )
    assert report["ok"], report
    assert not report["errors"]


def test_graphdataset_enforces_semantic_contract(tmp_path):
    root = _build_dataset_root(tmp_path)
    dm = GraphDataset(
        root=str(root),
        dataset_names="CIRCUIT_PARTITION",
        labeled_graph=True,
        init_tokenizer=True,
        weighted_sampling=True,
        gate_type_profile_path=str(root / "gate_type_profile.json"),
        schema_fields=[
            "size_bucket",
            "stub_bucket",
            "pin_bucket",
            "pin_deficit_bucket",
            "underconnected_bucket",
            "synthetic_input_bucket",
        ],
    )
    assert dm._semantic_contract_report is not None
    assert dm._semantic_contract_report["ok"]
