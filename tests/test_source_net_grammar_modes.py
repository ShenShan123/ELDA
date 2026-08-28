import unittest

from elda.models.seq_models import ELDAGrammar


class ELDAGrammarModesTest(unittest.TestCase):
    def test_topology_safe_d0_d6_feature_deltas(self):
        tokenizer = object()
        expected = {
            "reference": (),
            "d1_no_gale_ryser": ("gale_ryser",),
            "d2_no_same_cell_exclusion": ("same_cell_exclusion",),
            "d3_no_source_budget_mask": ("source_budget",),
            "d4_no_demand_exactly_once_mask": ("demand_exactly_once",),
            "d5_no_completion_eos_gate": ("completion_eos",),
            "d6_no_topology_safety": (
                "self_drive_exclusion", "combinational_cycle_exclusion",
            ),
        }
        for mode, disabled in expected.items():
            grammar = ELDAGrammar(tokenizer, 1, mask_mode=mode)
            actual = tuple(
                name for name, enabled in grammar.features.items() if not enabled
            )
            self.assertEqual(actual, disabled, mode)

    def test_unknown_mode_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown ELDA grammar mode"):
            ELDAGrammar(object(), 1, mask_mode="elda_unknown")


if __name__ == "__main__":
    unittest.main()
