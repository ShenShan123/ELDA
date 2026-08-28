"""Repair-free structural Verilog materialization for endpoint-complete objects."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _index(identifier: str) -> int:
    return int(str(identifier).rsplit("_", 1)[-1])


def emit_structural_verilog(payload: dict, library: dict, path: str | Path) -> dict:
    """Deterministically emit the already-complete assignment without repair."""
    path = Path(path)
    cells = {row["cell_id"]: row for row in payload["cells"]}
    demands = {row["demand_id"]: row for row in payload["demands"]}
    sources = {row["source_id"]: row for row in payload["sources"]}
    source_net_name = {}
    input_ports = []
    assignments = {}
    for source_id, source in sorted(sources.items(), key=lambda item: _index(item[0])):
        suffix = _index(source_id)
        if source["source_kind"] == "BOUNDARY_SOURCE":
            net = f"boundary_in_{suffix}"
            input_ports.append(net)
        elif source["source_kind"] == "CONSTANT_SOURCE":
            net = "1'b1" if "1" in str(source.get("source_pin", "")) else "1'b0"
        else:
            net = f"source_net_{suffix}"
        source_net_name[source_id] = net
    for row in payload["source_nets"]:
        net = source_net_name[row["source_id"]]
        for chunk in row.get("chunks", []):
            for demand_id in chunk.get("demand_ids", []):
                demand = demands[demand_id]
                assignments[(demand["load_cell_id"], demand["load_pin"])] = net

    output_ports = []
    driven_outputs = {}
    for source_id, source in sources.items():
        if source["source_kind"] == "CELL_OUTPUT":
            net = source_net_name[source_id]
            driven_outputs[(source["source_cell_id"], source["source_pin"])] = net
            if int(source.get("used_fanout", 0)) == 0:
                output_ports.append(net)
    ports = sorted(input_ports) + sorted(output_ports)
    lines = [
        "// Deterministic repair-free materialization of an endpoint-complete ELDA object.",
        "module elda_sample(" + ", ".join(ports) + ");",
    ]
    for name in sorted(input_ports):
        lines.append(f"  input {name};")
    for name in sorted(output_ports):
        lines.append(f"  output {name};")
    internal = sorted(set(source_net_name.values()) - set(input_ports) - set(output_ports) - {"1'b0", "1'b1"})
    for name in internal:
        lines.append(f"  wire {name};")
    for cell_id, cell in sorted(cells.items(), key=lambda item: _index(item[0])):
        spec = library[cell["cell_type"]]
        connections = []
        for pin in spec["inputs"]:
            connections.append(f".{pin}({assignments[(cell_id, pin)]})")
        for pin in spec["outputs"]:
            connections.append(f".{pin}({driven_outputs[(cell_id, pin)]})")
        lines.append(
            f"  {cell['cell_type']} u_{_index(cell_id)}(" + ", ".join(connections) + ");"
        )
    lines.extend(["endmodule", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "module": "elda_sample",
        "cell_instances": len(cells),
        "input_ports": len(input_ports),
        "output_ports": len(output_ports),
        "repair_operations": 0,
    }


def run_yosys(verilog: str | Path, cells: str | Path, log_path: str | Path, yosys_bin: str) -> dict:
    verilog = Path(verilog).resolve()
    cells = Path(cells).resolve()
    command = (
        f"read_verilog {cells}; read_verilog {verilog}; "
        "hierarchy -check -top elda_sample; proc; check; stat"
    )
    completed = subprocess.run(
        [yosys_bin, "-p", command], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    raw = completed.stdout
    Path(log_path).write_text(raw, encoding="utf-8")
    warning_count = len(re.findall(r"\bWarning:", raw))
    error_count = len(re.findall(r"\bERROR:", raw, flags=re.I))
    return {
        "returncode": int(completed.returncode),
        "read_check_pass": completed.returncode == 0 and error_count == 0,
        "warning_count": warning_count,
        "error_count": error_count,
        "yosys_version": subprocess.run(
            [yosys_bin, "-V"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        ).stdout.strip(),
    }
