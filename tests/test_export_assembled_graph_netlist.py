from pathlib import Path

import torch
from torch_geometric.data import Data

from tools.export_assembled_graph_netlist import emit_verilog


def test_emit_verilog_classifies_ports_from_final_multi_output_pin_map(tmp_path: Path) -> None:
    graph = Data(
        x=torch.tensor([1, 2, -1, -1, -1, -1], dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    spec = {
        "inputs": ["A"],
        "outputs": ["CO", "S"],
        "inouts": [],
        "min_required_outputs": 1,
        "fallback": False,
    }
    fa_record = {
        "gate_idx": 0,
        "cell_name": "FA_X1",
        "spec": spec,
        "real_inputs": [5],
        "reused_inputs": [],
        "helper_inputs": [],
        "synthetic_inputs": [],
        "real_outputs": [3],
        "repaired_output_nets": [],
        "pin_to_net": {"A": "n_5", "CO": "n_3", "S": "n_2"},
    }
    buf_spec = {
        "inputs": ["A"],
        "outputs": ["Z"],
        "inouts": [],
        "min_required_outputs": 1,
        "fallback": False,
    }
    buf_record = {
        "gate_idx": 1,
        "cell_name": "BUF_X1",
        "spec": buf_spec,
        "real_inputs": [2],
        "reused_inputs": [],
        "helper_inputs": [],
        "synthetic_inputs": [],
        "real_outputs": [4],
        "repaired_output_nets": [],
        "pin_to_net": {"A": "n_2", "Z": "n_4"},
    }
    output = tmp_path / "multi_output.v"

    emit_verilog(
        graph=graph,
        gate_records=[fa_record, buf_record],
        driver_count={2: 0, 3: 1, 4: 1, 5: 0},
        load_count={2: 1, 3: 0, 4: 0, 5: 1},
        used_specs={"FA_X1": spec, "BUF_X1": buf_spec},
        output_path=output,
        module_name="multi_output",
        net_id=-1,
    )

    text = output.read_text(encoding="utf-8")
    assert "input n_2;" not in text
    assert "wire n_2;" in text
    assert "output n_3;" in text
    assert ".S(n_2)" in text
