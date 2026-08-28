import os
import unittest
from pathlib import Path

import torch

from elda.datamodules.data.circuit_source_net_ablation_tokenizers import (
    CircuitSourceNetR1NoBoundaryIdentityTokenizer,
    CircuitSourceNetR2NoPerSourceBudgetTokenizer,
    CircuitSourceNetR3NoFullLoadAssignmentTokenizer,
    CircuitSourceNetR4CellLevelDemandTokenizer,
)
from elda.datamodules.graph_dataset import GraphDataset
from elda.models.seq_models import ELDAGrammar


DATA_ROOT_VALUE = os.environ.get("ELDA_DATA_ROOT")


class SourceNetELDAVariantGrammarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA_ROOT_VALUE:
            raise unittest.SkipTest(
                "set ELDA_DATA_ROOT to run dataset-backed ELDA grammar tests"
            )
        data_root = Path(DATA_ROOT_VALUE).expanduser().resolve()
        dm = GraphDataset(
            root=str(data_root),
            dataset_names="ELDA_REFERENCE",
            tokenizer_type="elda",
            max_length=24576,
            truncation_length=None,
            no_silent_truncate=True,
            cell_mapping_path=str(data_root / "mapping.txt"),
            batch_size=1,
            num_workers=0,
        )
        dm.prepare_data()
        dm.setup("fit")
        cls.base = dm.tokenizer
        first_reference = next(
            line.strip()
            for line in (data_root / "train_source_clean.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
        graph_path = Path(first_reference)
        if not graph_path.is_absolute():
            graph_path = data_root / graph_path
        cls.graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    def _tokenizer(self, tokenizer_class):
        tokenizer = tokenizer_class(
            max_length=24576,
            net_id=self.base.net_id,
            boundary_stub_id=self.base.boundary_stub_id,
            label_to_cell=self.base.label_to_cell,
            pin_specs=self.base.pin_specs,
        )
        tokenizer.set_num_nodes(self.base.max_num_nodes)
        tokenizer.set_pin_name_vocab(self.base.pin_names)
        tokenizer.set_num_node_and_edge_types(self.base.num_node_types, 0)
        return tokenizer

    def test_oracle_sequences_are_accepted_by_matching_variant_grammar(self):
        for tokenizer_class in (
            CircuitSourceNetR1NoBoundaryIdentityTokenizer,
            CircuitSourceNetR2NoPerSourceBudgetTokenizer,
            CircuitSourceNetR3NoFullLoadAssignmentTokenizer,
            CircuitSourceNetR4CellLevelDemandTokenizer,
        ):
            with self.subTest(tokenizer=tokenizer_class.__name__):
                tokenizer = self._tokenizer(tokenizer_class)
                sequence = tokenizer(self.graph).tolist()
                grammar = ELDAGrammar(
                    tokenizer, batch_size=1, device="cpu",
                    mask_mode="d6_no_topology_safety",
                )
                state = grammar.states[0]
                for position, token in enumerate(sequence[1:], start=1):
                    allowed = grammar._allowed(state)
                    self.assertIn(
                        token,
                        allowed,
                        (
                            f"position={position} token={token} "
                            f"expect={state['expect']} "
                            f"variant={tokenizer.representation_variant}"
                        ),
                    )
                    grammar._advance(state, token)
                    self.assertNotEqual(state["expect"], "invalid")
                self.assertEqual(state["expect"], "done")

    def test_r3_has_empty_assignment_section(self):
        tokenizer = self._tokenizer(
            CircuitSourceNetR3NoFullLoadAssignmentTokenizer
        )
        sequence = tokenizer(self.graph).tolist()
        begin = sequence.index(tokenizer.source_net_section_begin)
        self.assertEqual(sequence[begin + 1], tokenizer.source_net_section_end)
        self.assertEqual(sequence[begin + 2], tokenizer.eos)

    def test_r4_demand_records_are_cell_level_without_pin_tokens(self):
        tokenizer = self._tokenizer(
            CircuitSourceNetR4CellLevelDemandTokenizer
        )
        sequence = tokenizer(self.graph).tolist()
        begin = sequence.index(tokenizer.demand_section_begin)
        end = sequence.index(tokenizer.demand_section_end, begin + 1)
        self.assertNotIn(tokenizer.load_pin, sequence[begin + 1:end])
        self.assertGreater(sequence[begin + 1:end].count(tokenizer.demand_begin), 0)

    def test_r2_basic_disables_future_feasibility_only(self):
        tokenizer = self._tokenizer(
            CircuitSourceNetR2NoPerSourceBudgetTokenizer
        )
        grammar = ELDAGrammar(
            tokenizer, batch_size=1, device="cpu",
            mask_mode="r2_basic",
        )
        self.assertFalse(grammar.r2_future_feasibility)
        self.assertFalse(grammar.features["gale_ryser"])
        self.assertFalse(grammar.features["source_budget"])
        for feature in (
            "same_cell_exclusion", "demand_exactly_once",
            "completion_eos", "pointer_mask",
        ):
            self.assertTrue(grammar.features[feature])

    def test_topology_tail_lookahead_rejects_joint_future_cycle(self):
        tokenizer = self.base
        grammar = ELDAGrammar(
            tokenizer, batch_size=1, device="cpu",
            mask_mode="reference",
        )
        state = grammar._new_state()
        state["cells"] = [
            {"cell_type": "NAND2_X1"},
            {"cell_type": "NOR2_X1"},
            {"cell_type": "AND2_X1"},
        ]
        state["demands"] = [
            {"load_cell": 1},
            {"load_cell": 2},
            {"load_cell": 0},
        ]
        state["sources"] = [
            {
                "source_kind": tokenizer.cell_output,
                "source_cell": tokenizer.gate_offset + 0,
                "max_fanout": 1,
            },
            {
                "source_kind": tokenizer.cell_output,
                "source_cell": tokenizer.gate_offset + 1,
                "max_fanout": 1,
            },
            {
                "source_kind": tokenizer.cell_output,
                "source_cell": tokenizer.gate_offset + 2,
                "max_fanout": 1,
            },
        ]
        state["load_source"] = 0
        self.assertFalse(
            grammar._topology_residual_feasible(
                state,
                chosen_index=0,
                chosen_cell=1,
                current_left=0,
            )
        )

    def test_decoder_ablation_flags_change_state_transitions(self):
        tokenizer = self.base

        d0 = ELDAGrammar(
            tokenizer, 1, mask_mode="reference"
        )
        d2 = ELDAGrammar(
            tokenizer, 1, mask_mode="d2_no_same_cell_exclusion"
        )
        repeated_cell_capacity_state = {
            "expect": "max_fanout_value",
            "cells": [{}],
            "demands": [{"load_cell": 0}, {"load_cell": 0}],
            "sources": [],
            "current": {"fanout_bucket": tokenizer.fanout_2_4},
        }
        # D0's simple-bipartite Gale-Ryser check rejects degree two into one
        # cell, while D2 must allow it because two distinct input pins on that
        # cell are now legal targets.
        self.assertNotIn(
            tokenizer.count_offset + 2,
            d0._allowed({**d0._new_state(), **repeated_cell_capacity_state}),
        )
        self.assertIn(
            tokenizer.count_offset + 2,
            d2._allowed({**d2._new_state(), **repeated_cell_capacity_state}),
        )

        state = d2._new_state()
        state.update({
            "expect": "load_demand_value",
            "demands": [{"load_cell": 0}, {"load_cell": 0}],
            "sources": [{
                "source_kind": tokenizer.boundary_source,
                "source_cell": tokenizer.none,
                "max_fanout": 2,
            }],
            "load_source": 0,
            "load_left": 1,
            "assigned": {0},
            "load_cells": {0},
        })
        d2._advance(state, tokenizer.demand_offset + 1)
        self.assertNotEqual(state["expect"], "invalid")

        d3 = ELDAGrammar(
            tokenizer, 1, mask_mode="d3_no_source_budget_mask"
        )
        state = d3._new_state()
        state.update({
            "expect": "load_count_value",
            "demands": [{"load_cell": 0}, {"load_cell": 1}],
            "sources": [{"max_fanout": 1}],
            "load_source": 0,
        })
        d3._advance(state, tokenizer.count_offset + 2)
        self.assertNotEqual(state["expect"], "invalid")

        d4 = ELDAGrammar(
            tokenizer, 1, mask_mode="d4_no_demand_exactly_once_mask"
        )
        state = d4._new_state()
        state.update({
            "expect": "load_demand_value",
            "demands": [{"load_cell": 0}],
            "sources": [{
                "source_kind": tokenizer.boundary_source,
                "source_cell": tokenizer.none,
                "max_fanout": 1,
            }],
            "load_source": 0,
            "load_left": 1,
            "assigned": {0},
            "load_cells": set(),
        })
        d4._advance(state, tokenizer.demand_offset)
        self.assertNotEqual(state["expect"], "invalid")

        d5 = ELDAGrammar(
            tokenizer, 1, mask_mode="d5_no_completion_eos_gate"
        )
        state = d5._new_state()
        state.update({
            "expect": "load_choice",
            "demands": [{"load_cell": 0}],
            "sources": [{"max_fanout": 1}],
        })
        d5._advance(state, tokenizer.source_net_section_end)
        self.assertEqual(state["expect"], "eos")


if __name__ == "__main__":
    unittest.main()
