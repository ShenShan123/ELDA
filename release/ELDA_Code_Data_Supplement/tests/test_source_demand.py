from __future__ import annotations

from pathlib import Path

from elda.datamodules.data.circuit_source_net_v61_tokenizer import CircuitSourceNetV61Tokenizer
from elda.supplement.sample_io import (
    BOUNDARY_STUB_ID,
    LABEL_TO_CELL,
    NET_ID,
    PIN_SPECS,
    load_raw_graph,
)


ROOT = Path(__file__).resolve().parents[1]


def tokenizer():
    value = CircuitSourceNetV61Tokenizer(
        max_length=24576,
        net_id=NET_ID,
        boundary_stub_id=BOUNDARY_STUB_ID,
        label_to_cell=LABEL_TO_CELL,
        pin_specs=PIN_SPECS,
    )
    value.set_num_nodes(64)
    value.set_pin_name_vocab(["A", "Z", "__BOUNDARY__"])
    value.set_num_node_and_edge_types(max(LABEL_TO_CELL) + 1, 0)
    return value


def test_production_tokenizer_round_trip():
    graph = load_raw_graph(ROOT / "data_sample" / "raw_subcircuit.json")
    tok = tokenizer()
    sequence = tok.tokenize(graph)
    payload, report = tok.parse_tokens(sequence)
    assert report["token_decode_valid"]
    assert report["source_load_full_coverage"]
    decoded = tok.decode(sequence)
    second_payload, second_report = tok.parse_tokens(tok.tokenize(decoded))
    assert second_report["source_load_full_coverage"]
    assert payload == second_payload


def test_sos_is_context_not_a_prediction_target():
    graph = load_raw_graph(ROOT / "data_sample" / "raw_subcircuit.json")
    tok = tokenizer()
    sequence = tok.tokenize(graph)
    assert int(sequence[0]) == tok.sos
    assert tok.sos not in [int(value) for value in sequence[1:]]
