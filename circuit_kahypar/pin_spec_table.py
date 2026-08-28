from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from circuit_kahypar.liberty import default_nangate45_liberty, parse_liberty_pin_specs


@dataclass(frozen=True)
class CellPinSpec:
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    inouts: Tuple[str, ...] = ()
    min_required_outputs: int = 0
    fallback: bool = False


def _fallback_pin_spec(cell_name: str) -> CellPinSpec:
    name = str(cell_name).upper()
    if name in {"LOGIC0_X1", "LOGIC1_X1"}:
        return CellPinSpec(inputs=(), outputs=("Z",), min_required_outputs=1, fallback=True)
    if name.startswith(("INV_", "BUF_", "CLKBUF_")):
        return CellPinSpec(inputs=("A",), outputs=("ZN",), min_required_outputs=1, fallback=True)
    if name.startswith(("TBUF_", "TINV_")):
        return CellPinSpec(inputs=("A", "EN"), outputs=("ZN",), min_required_outputs=1, fallback=True)
    if name.startswith(("XOR2_", "XNOR2_")):
        return CellPinSpec(inputs=("A", "B"), outputs=("Z",), min_required_outputs=1, fallback=True)
    if name.startswith(("AOI21_", "OAI21_")):
        return CellPinSpec(inputs=("A", "B1", "B2"), outputs=("ZN",), min_required_outputs=1, fallback=True)
    if name.startswith(("AOI22_", "OAI22_")):
        return CellPinSpec(inputs=("A1", "A2", "B1", "B2"), outputs=("ZN",), min_required_outputs=1, fallback=True)
    if name.startswith("MUX2_"):
        return CellPinSpec(inputs=("A", "B", "S"), outputs=("Z",), min_required_outputs=1, fallback=True)
    if name.startswith("HA_"):
        return CellPinSpec(inputs=("A", "B"), outputs=("S", "CO"), min_required_outputs=1, fallback=True)
    if name.startswith("FA_"):
        return CellPinSpec(inputs=("A", "B", "CI"), outputs=("S", "CO"), min_required_outputs=1, fallback=True)
    if name.startswith("DFFSR_"):
        return CellPinSpec(inputs=("D", "CK", "RN", "SN"), outputs=("Q", "QN"), min_required_outputs=1, fallback=True)
    if name.startswith(("DFFR_", "DFFS_")):
        return CellPinSpec(inputs=("D", "CK", "RN"), outputs=("Q", "QN"), min_required_outputs=1, fallback=True)
    if name.startswith("DFF_"):
        return CellPinSpec(inputs=("D", "CK"), outputs=("Q", "QN"), min_required_outputs=1, fallback=True)
    if name.startswith(("FILLCELL_", "ANTENNA_")):
        return CellPinSpec(inputs=(), outputs=(), min_required_outputs=0, fallback=True)
    for prefix in ("NAND", "NOR", "AND", "OR"):
        if name.startswith(prefix):
            remainder = name[len(prefix):]
            digits = []
            for ch in remainder:
                if ch.isdigit():
                    digits.append(ch)
                else:
                    break
            if digits:
                n = int("".join(digits))
                return CellPinSpec(inputs=tuple(f"A{i+1}" for i in range(n)), outputs=("ZN",), min_required_outputs=1, fallback=True)
    return CellPinSpec(inputs=("A1", "A2", "A3", "A4", "A5", "A6"), outputs=("ZN",), min_required_outputs=1, fallback=True)


class CellPinSpecTable:
    def __init__(
        self,
        liberty_path: str | Path | None = None,
        *,
        allow_fallback: bool = True,
    ):
        if liberty_path is None:
            liberty_path = default_nangate45_liberty()
        self.liberty_path = Path(liberty_path) if liberty_path else None
        self.allow_fallback = bool(allow_fallback)
        self._specs: Dict[str, Dict[str, List[str]]] | None = None
        self._warnings: List[str] = []
        self._pin_to_id: Dict[str, int] = {}
        self._id_to_pin: List[str] = []

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    @property
    def pin_id_to_name(self) -> List[str]:
        return list(self._id_to_pin)

    def pin_name_to_id(self, name: str) -> int:
        name = str(name)
        if name not in self._pin_to_id:
            self._pin_to_id[name] = len(self._id_to_pin)
            self._id_to_pin.append(name)
        return int(self._pin_to_id[name])

    def _ensure_loaded(self) -> None:
        if self._specs is not None:
            return
        if self.liberty_path is None or not self.liberty_path.exists():
            self._specs = {}
            if self.liberty_path is None:
                self._warnings.append("liberty_missing: None")
            else:
                self._warnings.append(f"liberty_missing: {self.liberty_path}")
            return
        self._specs = parse_liberty_pin_specs(self.liberty_path)

    def get_cell_pin_spec(self, cell_type: str) -> CellPinSpec:
        name = str(cell_type)
        self._ensure_loaded()
        if self._specs and name in self._specs:
            spec = self._specs[name]
            outputs = tuple(spec.get('outputs', []))
            min_required_outputs = 0
            if len(outputs) > 0:
                min_required_outputs = 1
            return CellPinSpec(
                inputs=tuple(spec.get("inputs", [])),
                outputs=outputs,
                inouts=tuple(spec.get("inouts", [])),
                min_required_outputs=int(min_required_outputs),
                fallback=False,
            )
        if self.allow_fallback:
            self._warnings.append(f"cell_spec_fallback: {name}")
            return _fallback_pin_spec(name)
        self._warnings.append(f"cell_spec_missing: {name}")
        return CellPinSpec(inputs=(), outputs=(), inouts=(), min_required_outputs=0, fallback=True)

    def pre_register_cells(self, cell_types: Iterable[str]) -> None:
        for cell in sorted(set(str(c) for c in cell_types)):
            spec = self.get_cell_pin_spec(cell)
            for pin in list(spec.inputs) + list(spec.outputs) + list(spec.inouts):
                self.pin_name_to_id(pin)
