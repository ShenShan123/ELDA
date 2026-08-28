import os
from pathlib import Path

import pytest
import torch

from elda.datamodules.data.circuit_source_net_v61_ablation_tokenizers import (
    CircuitSourceNetV61R1NoBoundaryIdentityTokenizer,
    CircuitSourceNetV61R2NoPerSourceBudgetTokenizer,
    CircuitSourceNetV61R3NoFullLoadAssignmentTokenizer,
    CircuitSourceNetV61R4CellLevelDemandTokenizer,
)
from elda.datamodules.data.circuit_source_net_v61_tokenizer import (
    CircuitSourceNetV61Tokenizer,
)


DATA_ROOT_VALUE = os.environ.get("ELDA_DATA_ROOT")
if not DATA_ROOT_VALUE:
    pytest.skip(
        "set ELDA_DATA_ROOT to run dataset-backed V6.1 tokenizer tests",
        allow_module_level=True,
    )
DATA_ROOT = Path(DATA_ROOT_VALUE).expanduser().resolve()


def _tokenizers():
    # Reuse the fully initialized production tokenizer state; the variant
    # classes change only their versioned token schema methods.
    from omegaconf import OmegaConf
    from elda.datamodules.graph_dataset import GraphDataset

    cfg = OmegaConf.create({
        "root": str(DATA_ROOT),
        "dataset_names": "CIRCUIT_SOURCE_NET_PARTITION_V6_1_CLEAN_COMPACT_LOAD",
        "tokenizer_type": "source_net_v61",
        "max_length": 24576,
        "truncation_length": None,
        "no_silent_truncate": True,
        "cell_mapping_path": str(DATA_ROOT / "mapping_v61.txt"),
        "batch_size": 1,
        "num_workers": 0,
    })
    dm = GraphDataset(**cfg)
    dm.prepare_data()
    dm.setup("fit")
    base = dm.tokenizer
    variants = []
    for cls in (
        CircuitSourceNetV61R1NoBoundaryIdentityTokenizer,
        CircuitSourceNetV61R2NoPerSourceBudgetTokenizer,
        CircuitSourceNetV61R3NoFullLoadAssignmentTokenizer,
        CircuitSourceNetV61R4CellLevelDemandTokenizer,
    ):
        tok = cls(
            max_length=24576,
            net_id=base.net_id,
            boundary_stub_id=base.boundary_stub_id,
            label_to_cell=base.label_to_cell,
            pin_specs=base.pin_specs,
        )
        tok.set_num_nodes(base.max_num_nodes)
        tok.set_pin_name_vocab(base.pin_names)
        tok.set_num_node_and_edge_types(base.num_node_types, 0)
        variants.append(tok)
    return dm, base, variants


def test_r0_regression_and_r1_r4_schema_smoke():
    dm, r0, variants = _tokenizers()
    first_reference = next(
        line.strip()
        for line in (DATA_ROOT / "train_source_clean.txt").read_text().splitlines()
        if line.strip()
    )
    graph_path = Path(first_reference)
    if not graph_path.is_absolute():
        graph_path = DATA_ROOT / graph_path
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    r0_sequence = r0(graph)
    r0_payload, r0_report = r0.parse_tokens(r0_sequence)
    assert r0_report["token_decode_valid"]
    assert r0_report["source_load_full_coverage"]
    assert r0.decode(r0_sequence).x.numel() > 0

    sequences = [tokenizer(graph) for tokenizer in variants]
    for tokenizer, sequence in zip(variants, sequences):
        payload, report = tokenizer.parse_tokens(sequence)
        assert report["token_decode_valid"]
        assert payload["schema_version"] == tokenizer.serializer_version
        assert payload["representation_variant"] == tokenizer.representation_variant

    assert all(
        source["boundary_pin_id"] is None
        for source in variants[0].parse_tokens(sequences[0])[0]["sources"]
    )
    assert variants[1].parse_tokens(sequences[1])[1]["source_budget_encoded"] is False
    assert variants[2].parse_tokens(sequences[2])[1]["source_load_full_coverage"] is False
    assert variants[3].parse_tokens(sequences[3])[1]["pin_slot_identity_encoded"] is False

    assert variants[0].decode(sequences[0]).x.numel() > 0
    assert variants[1].decode(sequences[1]).x.numel() > 0
    with pytest.raises(ValueError, match="intentionally lacks information"):
        variants[2].decode(sequences[2])
    with pytest.raises(ValueError, match="intentionally lacks information"):
        variants[3].decode(sequences[3])

    assert len({tokenizer.tokenizer_version for tokenizer in variants}) == 4
    assert all(
        tokenizer.tokenizer_version != CircuitSourceNetV61Tokenizer.tokenizer_version
        for tokenizer in variants
    )
