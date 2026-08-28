from __future__ import annotations

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

from elda.datamodules.circuit_meta_dataset import CircuitPartitionDataset, CircuitSkeletonDataset
from circuit_kahypar.assemble import reconstruct_original_from_manifest
from circuit_kahypar.preprocess import preprocess_dataset
from circuit_kahypar.schema import normalize_circuit_graph


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


def test_preprocess_and_roundtrip(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'dataset'
    input_dir.mkdir(parents=True)
    source_path = input_dir / 'toy_design.pt'
    torch.save(_toy_graph(), source_path)

    meta = preprocess_dataset(
        input_dir=input_dir,
        output_root=output_dir,
        kahypar_bin=Path(KAHYPAR_BIN),
        preset=Path(KAHYPAR_PRESET),
        partition_cap=5,
        split_seed=7,
        split_ratios=(1.0, 0.0, 0.0),
        num_buckets=2,
    )
    manifest = torch.load(output_dir / 'manifests' / 'toy_design.pt', map_location='cpu', weights_only=False)
    reconstructed = reconstruct_original_from_manifest(manifest, meta)
    expected = normalize_circuit_graph(torch.load(source_path, map_location='cpu', weights_only=False))
    partition_graph = torch.load(output_dir / manifest['partition_records'][0]['graph_rel_path'], map_location='cpu', weights_only=False)

    assert meta['version'] == 3
    assert 'interface_bucket_keys' in meta
    assert 'interface_class_count' in meta
    assert 'synthetic_input_bucket' in meta['interface_bucket_keys']
    assert all(part['metrics']['num_nodes'] <= 5 for part in manifest['partition_records'])
    assert hasattr(partition_graph, 'gate_pin_deficit')
    assert hasattr(partition_graph, 'underconnected_gate_fraction')
    assert hasattr(partition_graph, 'synthetic_input_proxy_ratio')
    assert torch.equal(reconstructed.x, expected.x)
    assert torch.equal(reconstructed.edge_index, expected.edge_index)


def test_elda_meta_datasets_load(tmp_path):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'dataset'
    input_dir.mkdir(parents=True)
    torch.save(_toy_graph(), input_dir / 'toy_design.pt')

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

    partition_ds = CircuitPartitionDataset(root=str(output_dir), split='train')
    skeleton_ds = CircuitSkeletonDataset(root=str(output_dir), split='train')
    meta = torch.load(output_dir / 'meta.pt', map_location='cpu', weights_only=False)

    assert len(partition_ds) > 0
    assert len(skeleton_ds) == 1
    assert partition_ds.num_node_types >= 2
    assert skeleton_ds.num_node_types >= 1
    assert meta['interface_class_count'] >= skeleton_ds.num_node_types
    assert 'synthetic_input_bucket' in meta['interface_bucket_keys']
