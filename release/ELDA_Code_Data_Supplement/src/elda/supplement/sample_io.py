"""Portable JSON I/O for the included raw cell--endpoint graph."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data


def load_raw_graph(path: str | Path) -> Data:
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = Data(
        x=torch.tensor(row["x"], dtype=torch.long).reshape(-1, 1),
        edge_index=torch.tensor(row["edge_index"], dtype=torch.long),
        edge_role=torch.tensor(row["edge_role"], dtype=torch.long),
        edge_attr=torch.tensor(row["edge_role"], dtype=torch.long),
        edge_pin_id=torch.tensor(row["edge_pin_id"], dtype=torch.long),
    )
    graph.pin_id_to_name = list(row["pin_id_to_name"])
    graph.source_net_schema_version = row["schema_version"]
    return graph


def graph_signature(graph: Data) -> dict:
    """Return a JSON-stable structural signature used by round-trip tests."""
    edges = []
    for index, (src, dst) in enumerate(graph.edge_index.t().tolist()):
        pin_id = int(graph.edge_pin_id[index].item())
        edges.append(
            [
                int(src),
                int(dst),
                int(graph.edge_role[index].item()),
                str(graph.pin_id_to_name[pin_id]),
            ]
        )
    return {
        "x": [int(value) for value in graph.x.reshape(-1).tolist()],
        "edges": sorted(edges),
        "schema_version": str(graph.source_net_schema_version),
    }


class PinSpec:
    def __init__(self, inputs, outputs):
        self.inputs = list(inputs)
        self.outputs = list(outputs)


LABEL_TO_CELL = {6: "BUF_X1"}
CELL_TO_LABEL = {value: key for key, value in LABEL_TO_CELL.items()}
PIN_SPECS = {"BUF_X1": PinSpec(inputs=["A"], outputs=["Z"])}
NET_ID = 139
BOUNDARY_STUB_ID = 140
