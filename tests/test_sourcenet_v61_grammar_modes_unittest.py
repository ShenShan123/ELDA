import unittest

from elda.models.seq_models import SourceNetV61Grammar


class SourceNetV61GrammarModesTest(unittest.TestCase):
    def test_topology_safe_d0_d6_feature_deltas(self):
        tokenizer = object()
        expected = {
            "v61_d0_topology_safe": (),
            "v61_d1_no_gale_ryser": ("gale_ryser",),
            "v61_d2_no_same_cell_exclusion": ("same_cell_exclusion",),
            "v61_d3_no_source_budget_mask": ("source_budget",),
            "v61_d4_no_demand_exactly_once_mask": ("demand_exactly_once",),
            "v61_d5_no_completion_eos_gate": ("completion_eos",),
            "v61_d6_no_topology_safety": (
                "self_drive_exclusion", "combinational_cycle_exclusion",
            ),
        }
        for mode, disabled in expected.items():
            grammar = SourceNetV61Grammar(tokenizer, 1, mask_mode=mode)
            actual = tuple(
                name for name, enabled in grammar.features.items() if not enabled
            )
            self.assertEqual(actual, disabled, mode)

    def test_legacy_full_is_equivalent_to_no_topology_safety(self):
        legacy = SourceNetV61Grammar(object(), 1, mask_mode="v61_d0_full")
        d6 = SourceNetV61Grammar(
            object(), 1, mask_mode="v61_d6_no_topology_safety"
        )
        self.assertEqual(legacy.features, d6.features)

    def test_unknown_mode_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown V6.1 grammar mode"):
            SourceNetV61Grammar(object(), 1, mask_mode="v61_unknown")


if __name__ == "__main__":
    unittest.main()
