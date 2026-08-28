#!/usr/bin/env python3
from __future__ import annotations

from typing import Any


POST_STITCH_ACCOUNTING_PROTOCOL = "post_stitch_effective_mutation_v2"


def post_stitch_operation_breakdown(
    issues: dict[str, Any],
    verilog_metadata: dict[str, Any] | None = None,
    *,
    explicit_pin_overlay_complete: bool = False,
) -> dict[str, int]:
    """Return effective post-stitch mutations and separately retained diagnostics."""
    verilog_metadata = verilog_metadata or {}
    legalization = verilog_metadata.get("practical_legalization", {})
    if not isinstance(legalization, dict):
        legalization = {}

    final_synthetic_inputs = int(
        verilog_metadata.get("synthetic_input_count", 0) or 0
    )
    input_pool_nets = int(
        issues.get(
            "created_input_pool_nets",
            issues.get("synthetic_inputs", 0),
        )
        or 0
    )
    heuristic_input_reuses = int(issues.get("reused_inputs", 0) or 0)
    heuristic_dropped_incidence = int(
        issues.get("dropped_extra_incident", 0) or 0
    )

    breakdown = {
        "cell_spec_fallbacks": int(
            issues.get("missing_liberty_cell_specs", 0) or 0
        ),
        "input_pool_net_creations": input_pool_nets,
        "input_connection_completions": int(
            issues.get("synthetic_input_connections", 0) or 0
        ),
        "helper_input_stages": int(
            issues.get("helper_input_stages", 0) or 0
        ),
        "input_reuses": (
            0 if explicit_pin_overlay_complete else heuristic_input_reuses
        ),
        "output_net_completions": int(
            issues.get("repaired_output_nets", 0) or 0
        ),
        "output_conflict_reassignments": int(
            issues.get("role_realizer_output_conflict_reassigned", 0) or 0
        ),
        "multi_driver_splits": int(
            issues.get("multi_driver_splits", 0) or 0
        ),
        "dropped_extra_incidence": (
            0
            if explicit_pin_overlay_complete
            else heuristic_dropped_incidence
        ),
        "legalizer_self_loop_input_remaps": int(
            legalization.get("self_loop_fixed_by_input_remap", 0) or 0
        ),
        "legalizer_same_net_reuse_remaps": int(
            legalization.get("same_net_reuse_reduced", 0) or 0
        ),
        "legalizer_undriven_load_pi_promotions": int(
            legalization.get("undriven_load_promoted_to_pi", 0) or 0
        ),
        "legalizer_input_pool_net_creations": max(
            0, final_synthetic_inputs - input_pool_nets
        ),
        "diagnostic_heuristic_input_reuse_proposals": (
            heuristic_input_reuses if explicit_pin_overlay_complete else 0
        ),
        "diagnostic_heuristic_extra_incidence_proposals": (
            heuristic_dropped_incidence if explicit_pin_overlay_complete else 0
        ),
    }
    effective_keys = [
        key
        for key in breakdown
        if not key.startswith("diagnostic_")
    ]
    breakdown["completion_ops"] = sum(
        breakdown[key]
        for key in effective_keys
        if key != "dropped_extra_incidence"
    )
    breakdown["total_ops"] = (
        breakdown["completion_ops"]
        + breakdown["dropped_extra_incidence"]
    )
    return breakdown
