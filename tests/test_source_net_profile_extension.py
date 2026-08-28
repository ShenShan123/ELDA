import unittest
import hashlib

import torch
from torch_geometric.data import Data

from elda.datamodules.data.circuit_source_net_profile_schema import build_audit_profile, serialize_profiled_source_net
from elda.datamodules.data.circuit_source_net_profile_tokenizer import (
    CircuitSourceNetProfileTokenizer,
)
from elda.datamodules.data.circuit_source_net_tokenizer import CircuitSourceNetTokenizer
from elda.models.seq_models import ELDAGrammar
from elda.datamodules.graph_dataset import ProfileMixtureSampler


class _Pins:
    inputs = ["A"]
    outputs = ["Z"]


class _MuxPins:
    inputs = ["A", "B", "S"]
    outputs = ["Z"]


class SourceNetProfileExtensionTest(unittest.TestCase):
    PROFILE_CONFIG = {
        "version": "test_train_frozen_v1",
        "bucket_boundaries": {
            "cell_count": [1, 2, 3], "demand_count": [1, 2, 3],
            "source_count": [1, 2, 3], "seq_ratio": [0.0, 0.25, 0.75],
            "pin_complexity": [1, 2, 3],
        },
        "sequential_cell_types": [],
        "rare_cell_types": ["BUF"],
        "rare_partition_ratio_threshold": 0.05,
        "rare_partition_min_count": 1,
        "rare_partition_rule": "rare_cell_count_at_least_min_count",
        "active_profile_bins": {
            "cell_count": ["PROFILE_BIN_0", "PROFILE_BIN_1", "PROFILE_BIN_2", "PROFILE_BIN_3"],
            "demand_count": ["PROFILE_BIN_0", "PROFILE_BIN_1", "PROFILE_BIN_2", "PROFILE_BIN_3"],
            "source_count": ["PROFILE_BIN_0", "PROFILE_BIN_1", "PROFILE_BIN_2", "PROFILE_BIN_3"],
            "seq_ratio": ["PROFILE_BIN_0", "PROFILE_BIN_1", "PROFILE_BIN_2", "PROFILE_BIN_3"],
            "pin_complexity": ["PROFILE_BIN_0", "PROFILE_BIN_1", "PROFILE_BIN_3"],
        },
    }
    @staticmethod
    def _graph():
        graph = Data(
            x=torch.tensor([0, 99]),
            edge_index=torch.tensor([[0, 0], [1, 1]]),
        )
        graph.edge_role = torch.tensor([2, 1])
        graph.edge_pin_id = torch.tensor([1, 0])
        graph.pin_id_to_name = ["A", "Z"]
        graph.source_exposure_by_endpoint_id = {1: "MODULE_OUTPUT"}
        return graph

    @staticmethod
    def _tokenizer(**kwargs):
        tokenizer = CircuitSourceNetProfileTokenizer(
            net_id=99,
            boundary_stub_id=100,
            label_to_cell={0: "BUF", 1: "MUX2_X1"},
            pin_specs={"BUF": _Pins(), "MUX2_X1": _MuxPins()},
            profile_config=SourceNetProfileExtensionTest.PROFILE_CONFIG,
            max_length=256,
            **kwargs,
        )
        tokenizer.set_num_nodes(1024)
        tokenizer.set_pin_name_vocab(["A", "B", "S", "Z"])
        tokenizer.set_num_node_and_edge_types(2, 0)
        return tokenizer

    def test_serialization_adds_profile_without_changing_elda_object_semantics(self):
        graph = self._graph()
        payload = serialize_profiled_source_net(
            graph,
            net_id=99,
            boundary_stub_id=100,
            label_to_cell={0: "BUF"},
            pin_specs={"BUF": _Pins()},
            profile_config=self.PROFILE_CONFIG,
        )
        self.assertNotIn("source_exposure", payload["sources"][0])
        self.assertEqual(payload["sources"][0]["used_fanout"], 1)
        self.assertEqual(len(payload["demands"]), 1)
        self.assertEqual(payload["control_profile"]["cell_count_bin"], "PROFILE_BIN_0")
        self.assertTrue(payload["control_profile"]["rare_cell_enriched"])
        self.assertNotIn("audit_profile", payload)
        self.assertIn("fanout_histogram", build_audit_profile(payload))

    def test_target_profile_token_roundtrip_and_grammar_replay(self):
        tokenizer = self._tokenizer()
        sequence = tokenizer.tokenize(self._graph())
        payload, report = tokenizer.parse_tokens(sequence)
        self.assertTrue(report["target_profile_consistent"])
        self.assertEqual(payload["control_profile"]["demand_count_bin"], "PROFILE_BIN_0")

        grammar = ELDAGrammar(tokenizer, 1, mask_mode="d6_no_topology_safety")
        state = grammar.states[0]
        for token in sequence.tolist()[1:]:
            self.assertIn(token, grammar._allowed(state), msg=f"state={state['expect']}")
            grammar._advance(state, token)
        self.assertEqual(state["expect"], "done")

    def test_exact_graph_metrics_are_not_in_inference_prefix(self):
        tokenizer = self._tokenizer()
        sequence = tokenizer.tokenize(self._graph())
        self.assertEqual(
            hashlib.sha256(sequence.numpy().tobytes()).hexdigest(),
            "c667036b21881b9e1114b63cfdef2aa809bdcdcc2823130215a513ebe20d6e57",
        )
        prefix_names = [
            tokenizer.special_toks[token]
            for token in sequence.tolist()
            if token < len(tokenizer.special_toks)
        ][:30]
        for forbidden in (
            "TARGET_FANOUT_HIST", "TARGET_COMPONENT_HINT", "TARGET_EDGE_DENSITY",
        ):
            self.assertNotIn(forbidden, prefix_names)

    def test_condition_dropout_is_train_only_and_supports_profile_free(self):
        graph = self._graph()
        graph.elda_profile_split = "train"
        tokenizer = self._tokenizer(condition_dropout_prob=1.0)
        payload, report = tokenizer.parse_tokens(tokenizer.tokenize(graph))
        self.assertEqual(payload["control_profile"], {"profile_mode": "PROFILE_FREE"})
        self.assertTrue(report["target_profile_consistent"])

        graph.elda_profile_split = "val"
        payload, _ = tokenizer.parse_tokens(tokenizer.tokenize(graph))
        self.assertEqual(payload["control_profile"]["profile_mode"], "PROFILE_COARSE")

    def test_condition_dropout_is_seed_reproducible_at_configured_rate(self):
        graph = self._graph()
        graph.elda_profile_split = "train"
        tokenizer = self._tokenizer(condition_dropout_prob=0.15)

        def modes(seed):
            torch.manual_seed(seed)
            return [
                tokenizer.parse_tokens(tokenizer.tokenize(graph))[0]["control_profile"]["profile_mode"]
                for _ in range(200)
            ]

        first = modes(20260711)
        self.assertEqual(first, modes(20260711))
        free_count = first.count("PROFILE_FREE")
        self.assertGreaterEqual(free_count, 20)
        self.assertLessEqual(free_count, 40)

    def test_positive_mux_flag_masks_non_mux_last_cell(self):
        tokenizer = self._tokenizer()
        sequence = tokenizer.tokenize(self._graph()).tolist()
        mux_tag = sequence.index(tokenizer.has_mux)
        sequence[mux_tag + 1] = tokenizer.bool_true
        grammar = ELDAGrammar(tokenizer, 1, mask_mode="d6_no_topology_safety")
        state = grammar.states[0]
        failed_state = None
        for token in sequence[1:]:
            if token not in grammar._allowed(state):
                failed_state = state["expect"]
                break
            grammar._advance(state, token)
        self.assertEqual(failed_state, "cell_type_value")

    def test_negative_complex_flag_excludes_mux_cell_token(self):
        tokenizer = self._tokenizer()
        sequence = tokenizer.tokenize(self._graph()).tolist()
        grammar = ELDAGrammar(tokenizer, 1, mask_mode="profile_reference")
        state = grammar.states[0]
        for token in sequence[1:]:
            if state["expect"] == "cell_type_value":
                break
            self.assertIn(token, grammar._allowed(state))
            grammar._advance(state, token)
        self.assertEqual(state["expect"], "cell_type_value")
        self.assertNotIn(tokenizer._cell_type("MUX2_X1"), grammar._allowed(state))

    def test_unreachable_sequential_ratio_produces_mask_failure(self):
        tokenizer = self._tokenizer()
        sequence = tokenizer.tokenize(self._graph()).tolist()
        tag = sequence.index(tokenizer.seq_ratio_bin)
        sequence[tag + 1] = tokenizer.profile_bin_3
        grammar = ELDAGrammar(tokenizer, 1, mask_mode="profile_reference")
        state = grammar.states[0]
        for token in sequence[1:]:
            if state["expect"] == "cell_type_value":
                break
            self.assertIn(token, grammar._allowed(state))
            grammar._advance(state, token)
        self.assertEqual(state["expect"], "cell_type_value")
        self.assertEqual(grammar._allowed(state), [])

    def test_unreachable_pin_complexity_produces_mask_failure(self):
        tokenizer = self._tokenizer()
        sequence = tokenizer.tokenize(self._graph()).tolist()
        tag = sequence.index(tokenizer.pin_complexity_bin)
        sequence[tag + 1] = tokenizer.profile_bin_3
        grammar = ELDAGrammar(tokenizer, 1, mask_mode="profile_reference")
        state = grammar.states[0]
        for token in sequence[1:]:
            if state["expect"] == "cell_type_value":
                break
            self.assertIn(token, grammar._allowed(state))
            grammar._advance(state, token)
        self.assertEqual(state["expect"], "cell_type_value")
        self.assertEqual(grammar._allowed(state), [])

    def test_explicit_profile_free_prefix_enters_cell_section_without_reference(self):
        tokenizer = self._tokenizer()
        prefix = tokenizer.encode_profile_prefix({"profile_mode": "PROFILE_FREE"})
        grammar = ELDAGrammar(tokenizer, 1, mask_mode="profile_reference")
        state = grammar.states[0]
        for token in prefix.tolist()[1:]:
            self.assertIn(token, grammar._allowed(state))
            grammar._advance(state, token)
        self.assertEqual(state["expect"], "cell_section_begin")

    def test_inactive_train_bucket_is_rejected_before_generation(self):
        tokenizer = self._tokenizer()
        payload = tokenizer._serialize_payload(self._graph())
        profile = dict(payload["control_profile"])
        profile["pin_complexity_bin"] = "PROFILE_BIN_2"
        with self.assertRaisesRegex(ValueError, "inactive target profile bucket"):
            tokenizer.encode_profile_prefix(profile)

    def test_profile_ablation_modes_separate_profile_and_materialization_masks(self):
        tokenizer = self._tokenizer()
        feasible = ELDAGrammar(
            tokenizer, 1, mask_mode="profile_feasible_control"
        )
        self.assertTrue(feasible.features["profile_feasibility"])
        for feature in (
            "gale_ryser", "same_cell_exclusion", "source_budget",
            "demand_exactly_once", "completion_eos", "output_source_coverage",
            "self_drive_exclusion", "combinational_cycle_exclusion",
        ):
            self.assertFalse(feasible.features[feature])
        full = ELDAGrammar(tokenizer, 1, mask_mode="profile_reference")
        self.assertTrue(all(full.features.values()))

    def test_profile_mixture_sampler_draws_whole_partition_indices(self):
        torch.manual_seed(7)
        sampler = ProfileMixtureSampler(
            100, range(0, 10), range(10, 20), range(20, 30),
            {"normal": 0.70, "rare_complex": 0.20, "hard": 0.10},
        )
        values = list(iter(sampler))
        self.assertEqual(len(values), 100)
        self.assertEqual(sum(value < 10 for value in values), 70)
        self.assertEqual(sum(10 <= value < 20 for value in values), 20)
        self.assertEqual(sum(20 <= value < 30 for value in values), 10)

    def test_profile_mixture_sampler_balances_rare_semantic_subpools(self):
        torch.manual_seed(7)
        sampler = ProfileMixtureSampler(
            100, range(0, 10), range(10, 40), range(40, 50),
            {"normal": 0.70, "rare_complex": 0.20, "hard": 0.10},
            rare_subpools={
                "rare_cell": range(10, 20),
                "mux": range(20, 30),
                "aoi_oai": range(30, 40),
            },
        )
        list(iter(sampler))
        self.assertEqual(
            sampler.last_epoch_report["rare_complex_draw_counts"],
            {"aoi_oai": 7, "mux": 7, "rare_cell": 6},
        )

    def test_elda_token_sequence_and_vocabulary_remain_frozen(self):
        tokenizer = CircuitSourceNetTokenizer(
            net_id=99, boundary_stub_id=100, label_to_cell={0: "BUF"},
            pin_specs={"BUF": _Pins()}, max_length=256,
        )
        tokenizer.set_num_nodes(1024)
        tokenizer.set_pin_name_vocab(["A", "Z"])
        tokenizer.set_num_node_and_edge_types(1, 0)
        sequence = tokenizer.tokenize(self._graph())
        digest = hashlib.sha256(sequence.numpy().tobytes()).hexdigest()
        self.assertEqual(tokenizer.tokenizer_version, "elda_source_demand_v1")
        self.assertNotIn("TARGET_PROFILE_BEGIN", tokenizer.special_toks)
        self.assertEqual(digest, "8c702699072f1cb814c10b7c56be4d6aee3131b1906a5db68cdfd28860408c22")

if __name__ == "__main__":
    unittest.main()
