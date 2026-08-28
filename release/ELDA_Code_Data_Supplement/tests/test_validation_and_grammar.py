from __future__ import annotations

import json
from pathlib import Path

from elda.supplement.grammar_loader import ELDAGrammar
from elda.supplement.validation import negative_cases, validate_payload
from scripts.run_pipeline import configure_tokenizer


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = {"BUF_X1": {"inputs": ["A"], "outputs": ["Z"]}}


def payload():
    return json.loads((ROOT / "expected_outputs" / "source_demand_object.json").read_text())


def test_positive_and_each_negative_rule():
    good = payload()
    assert validate_payload(good, LIBRARY)["overall_valid"]
    for _name, (bad, expected_rule) in negative_cases(good).items():
        report = validate_payload(bad, LIBRARY)
        assert not report["overall_valid"]
        assert not report["conditions"][expected_rule]


def test_exact_production_grammar_accepts_frozen_sequence():
    tok = configure_tokenizer()
    sequence = json.loads((ROOT / "expected_outputs" / "token_sequence.json").read_text())["ids"]
    grammar = ELDAGrammar(tok, batch_size=1, mask_mode="reference")
    state = grammar.states[0]
    for chosen in sequence[1:]:
        assert chosen in grammar._allowed(state)
        grammar._advance(state, chosen)
    assert state["expect"] == "done"
