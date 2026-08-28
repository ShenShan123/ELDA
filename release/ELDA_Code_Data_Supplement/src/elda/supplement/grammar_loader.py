"""Load the exact production SourceNetV61Grammar without importing training code.

The frozen production module contains the complete model stack and therefore
imports training-only packages.  This loader extracts the single grammar class
from the included, unmodified source snapshot with Python's AST.  The class body
executed here is byte-for-byte the class used by the paper experiments.
"""

from __future__ import annotations

import ast
import math
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import LogitsProcessor


def load_source_net_v61_grammar():
    snapshot = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "_production_snapshot"
        / "seq_models.py"
    )
    source = snapshot.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(snapshot))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SourceNetV61Grammar"
    ]
    if len(nodes) != 1:
        raise RuntimeError("The production snapshot must contain exactly one SourceNetV61Grammar")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "__name__": "elda.models.source_net_v61_grammar_snapshot",
        "math": math,
        "OrderedDict": OrderedDict,
        "torch": torch,
        "LogitsProcessor": LogitsProcessor,
    }
    exec(compile(module, str(snapshot), "exec"), namespace)
    return namespace["SourceNetV61Grammar"]


SourceNetV61Grammar = load_source_net_v61_grammar()
