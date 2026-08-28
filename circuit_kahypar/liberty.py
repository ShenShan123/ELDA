from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional


def default_nangate45_liberty() -> Optional[Path]:
    candidates = []
    explicit = os.environ.get('ELDA_NANGATE45_LIBERTY')
    if explicit:
        candidates.append(Path(explicit).expanduser())
    flow_root_value = os.environ.get('OPENROAD_FLOW_ROOT') or os.environ.get('ORFS_ROOT')
    flow_root = Path(flow_root_value).expanduser() if flow_root_value else None
    if flow_root is not None:
        candidates.append(
            flow_root / 'platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib'
        )
    for path in candidates:
        if path.exists():
            return path
    root = flow_root / 'objects/nangate45' if flow_root is not None else None
    if root is not None and root.exists():
        matches = sorted(root.glob('*/base/lib/NangateOpenCellLibrary_typical.lib'))
        if matches:
            return matches[0]
    return None


def _extract_block(text: str, open_brace_idx: int) -> str:
    depth = 0
    start = open_brace_idx + 1
    for idx in range(open_brace_idx, len(text)):
        ch = text[idx]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:idx]
    raise ValueError('unterminated Liberty block')


def _iter_named_blocks(text: str, keyword: str):
    pattern = re.compile(rf'\b{re.escape(keyword)}\s*\(\s*([A-Za-z0-9_]+)\s*\)\s*\{{')
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if match is None:
            break
        name = match.group(1)
        open_brace_idx = text.find('{', match.end() - 1)
        if open_brace_idx < 0:
            break
        block = _extract_block(text, open_brace_idx)
        yield name, block
        pos = open_brace_idx + len(block) + 2


def parse_liberty_pin_specs(path: str | Path) -> Dict[str, Dict[str, List[str]]]:
    """Parse enough Liberty to recover signal pin names and directions per cell."""
    text = Path(path).read_text(encoding='utf-8', errors='ignore')
    specs: Dict[str, Dict[str, List[str]]] = {}
    for cell_name, cell_block in _iter_named_blocks(text, 'cell'):
        inputs: List[str] = []
        outputs: List[str] = []
        inouts: List[str] = []
        for pin_name, pin_block in _iter_named_blocks(cell_block, 'pin'):
            direction_match = re.search(r'\bdirection\s*:\s*([A-Za-z_]+)\s*;', pin_block)
            if direction_match is None:
                continue
            direction = direction_match.group(1).lower()
            if direction == 'input':
                inputs.append(pin_name)
            elif direction == 'output':
                outputs.append(pin_name)
            elif direction in {'inout', 'internal'}:
                inouts.append(pin_name)
        specs[cell_name] = {
            'inputs': inputs,
            'outputs': outputs,
            'inouts': inouts,
        }
    return specs


def pin_count_spec(cell_name: str, pin_specs: Dict[str, Dict[str, List[str]]] | None) -> Optional[Dict[str, int]]:
    if not pin_specs:
        return None
    spec = pin_specs.get(str(cell_name))
    if spec is None:
        return None
    return {
        'inputs': int(len(spec.get('inputs', []))),
        'outputs': int(len(spec.get('outputs', []))),
    }
