from __future__ import annotations

import ast
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from elda.datamodules.data.circuit_pin_slot_tokenizer import PinSlotSpec
from elda.datamodules.data.circuit_source_net_v5_tokenizer import CircuitSourceNetV5Tokenizer


SERIALIZER_VERSION = "source_net_v5_dual_view_demand_constrained_v1"
TOKENIZER_VERSION = "source_net_v5_dual_view_demand_constrained_v1"
SOURCE_ROOT = Path(
    os.environ.get("ELDA_V4_DATA_ROOT", REPO_ROOT / "datasets/source_net_v4")
).expanduser().resolve()
V5_ROOT = Path(
    os.environ.get("ELDA_V5_DATA_ROOT", REPO_ROOT / "datasets/source_net_v5")
).expanduser().resolve()
PREP_DIR = Path(
    os.environ.get("ELDA_V5_PREP_ROOT", REPO_ROOT / "outputs/source_net_v5_prepare")
).expanduser().resolve()


def load_label_to_cell(mapping_path: Path) -> Dict[int, str]:
    text = mapping_path.read_text(encoding="utf-8")
    _, rhs = text.split("=", 1)
    mapping = ast.literal_eval(rhs.strip())
    return {int(label): str(cell) for cell, label in mapping.items()}


def load_pin_specs(label_to_cell: Dict[int, str]):
    try:
        from circuit_kahypar.pin_spec_table import CellPinSpecTable

        table = CellPinSpecTable(allow_fallback=True)
        table.pre_register_cells(label_to_cell.values())
        pin_specs = {}
        pin_names = set()
        for cell in label_to_cell.values():
            spec = table.get_cell_pin_spec(str(cell))
            inputs = [str(p) for p in spec.inputs]
            outputs = [str(p) for p in spec.outputs]
            pin_specs[str(cell)] = PinSlotSpec(inputs=inputs, outputs=outputs, min_required_outputs=int(spec.min_required_outputs))
            pin_names.update(inputs); pin_names.update(outputs)
        return pin_specs, sorted(pin_names | {"__UNKNOWN__", "__BOUNDARY__", "__CONST0__", "__CONST1__"})
    except Exception:
        pins = {"A", "A1", "A2", "A3", "A4", "B", "B1", "B2", "CI", "CK", "CO", "D", "Q", "QN", "S", "Z", "ZN"}
        return {}, sorted(pins | {"__UNKNOWN__", "__BOUNDARY__", "__CONST0__", "__CONST1__"})


def build_v5_tokenizer(dataset_root: Path = V5_ROOT, max_length: int = 12288) -> Tuple[CircuitSourceNetV5Tokenizer, dict]:
    meta = torch.load(dataset_root / "meta.pt", map_location="cpu", weights_only=False)
    mapping_path = Path(str(meta.get("cell_mapping", dataset_root / "mapping_v5.txt")))
    label_to_cell = load_label_to_cell(mapping_path)
    pin_specs, pin_names = load_pin_specs(label_to_cell)
    tok = CircuitSourceNetV5Tokenizer(
        dataset_names=[],
        max_length=int(max_length),
        truncation_length=None,
        append_eos=True,
        net_id=int(meta.get("net_id", 139)),
        boundary_stub_id=int(meta.get("boundary_stub_id", 140)),
        label_to_cell=label_to_cell,
        pin_specs=pin_specs,
        source_net_v5_compact=bool(meta.get("source_net_v5_compact", True)),
        compact_demand_record=bool(meta.get("compact_demand_record", True)),
        compact_sink_demand_record=bool(meta.get("compact_sink_demand_record", True)),
        compact_source_net_hist=bool(meta.get("compact_source_net_hist", True)),
    )
    tok.set_num_nodes(int(meta.get("max_num_nodes", 8192)))
    tok.set_pin_name_vocab(pin_names)
    tok.set_num_node_and_edge_types(num_node_types=int(meta.get("boundary_stub_id", 140)) + 1, num_edge_types=4)
    return tok, meta


def iter_partition_paths(dataset_root: Path, limit: int = 0, split: str | None = None) -> Iterable[Path]:
    meta = torch.load(dataset_root / "meta.pt", map_location="cpu", weights_only=False)
    source = Path(str(meta.get("source_dataset_root") or meta.get("source_dataset_path") or dataset_root))
    graph_dir = source / "partitions" / "graphs"
    count = 0
    for entry in os.scandir(graph_dir):
        if not entry.name.endswith(".pt"):
            continue
        if split:
            design = entry.name.rsplit("__p", 1)[0]
            prefix, sep, rest = design.partition("__")
            if sep and prefix.isdigit() and rest:
                design = rest
            design_to_split = meta.get("splits", {}).get("design_to_split", {})
            if design_to_split and design_to_split.get(design) != split:
                continue
        yield Path(entry.path)
        count += 1
        if limit and count >= limit:
            return


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return float(vals[idx])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
