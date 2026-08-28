from __future__ import annotations

import torch

from .circuit_source_net_tokenizer import CircuitSourceNetTokenizer
from .circuit_source_net_profile_schema import (
    PROFILE_BIN_NAMES,
    build_control_profile,
    decode_profiled_source_net,
    serialize_profiled_source_net,
)


class CircuitSourceNetProfileTokenizer(CircuitSourceNetTokenizer):
    serializer_version = "elda_profile_source_demand_v1"
    tokenizer_version = "elda_profile_source_demand_v1"
    include_target_profile = True

    def __init__(self, **kwargs):
        self.include_target_profile = bool(kwargs.pop("profile_prefix_enabled", True))
        self.condition_dropout_prob = float(kwargs.pop("condition_dropout_prob", 0.0))
        self.profile_mode_override = str(kwargs.pop("profile_mode_override", "") or "")
        self.profile_config = dict(kwargs.pop("profile_config", {}) or {})
        if not self.profile_config:
            raise ValueError("ELDA profile extension tokenizer requires frozen train-split profile_config")
        if not 0.0 <= self.condition_dropout_prob <= 1.0:
            raise ValueError("condition_dropout_prob must be in [0, 1]")
        super().__init__(**kwargs)
        profile_tokens = [
            "TARGET_PROFILE_BEGIN", "TARGET_PROFILE_END",
            "PROFILE_MODE", "PROFILE_FREE", "PROFILE_COARSE",
            "CELL_COUNT_BIN", "SOURCE_COUNT_BIN", "DEMAND_COUNT_BIN",
            "SEQ_RATIO_BIN", "PIN_COMPLEXITY_BIN", "COMPLEX_CELL_FLAGS",
            "RARE_CELL_ENRICHED", "HAS_MUX", "HAS_AOI_OAI",
            "HAS_MULTI_OUTPUT_CELL", "BOOL_FALSE", "BOOL_TRUE",
            *PROFILE_BIN_NAMES,
        ]
        # The optional target-profile extension uses a separate vocabulary. Appending here leaves every ELDA token ID
        # unchanged while placing target-profile tags before pointer ranges.
        for name in profile_tokens:
            if name not in self.special_toks:
                setattr(self, name.lower(), len(self.special_toks))
                self.special_toks.append(name)
        self.idx_offset = len(self.special_toks)
        self._refresh_offsets()
        # ELDA profile extension retains the ELDA grammar with one additional typed source field.
        self.is_source_net_tokenizer = True
        self.is_elda_profile_tokenizer = True

    def _serialize_payload(self, data):
        payload = serialize_profiled_source_net(
            data,
            net_id=self.net_id,
            boundary_stub_id=self.boundary_stub_id,
            label_to_cell=self.label_to_cell,
            pin_specs=self.pin_specs,
            profile_config=self.profile_config,
            chunk_size=max(1, self.max_num_nodes),
        )
        split = str(getattr(data, "elda_profile_split", ""))
        use_free = self.profile_mode_override == "PROFILE_FREE"
        if self.profile_mode_override == "PROFILE_COARSE":
            use_free = False
        elif split == "train" and self.condition_dropout_prob > 0.0:
            use_free = bool(torch.rand(()).item() < self.condition_dropout_prob)
        if use_free:
            payload["control_profile"] = {"profile_mode": "PROFILE_FREE"}
        return payload

    def control_profile_matches(self, payload, control_profile):
        if control_profile.get("profile_mode") == "PROFILE_FREE":
            return True
        expected = build_control_profile(
            payload, profile_config=self.profile_config, pin_specs=self.pin_specs
        )
        return all(expected.get(key) == value for key, value in control_profile.items())

    def realized_control_profile(self, payload):
        return build_control_profile(
            payload, profile_config=self.profile_config, pin_specs=self.pin_specs
        )

    def encode_profile_prefix(self, profile):
        """Encode only SOS + TARGET_PROFILE_SECTION for one-shot generation."""
        mode = str(profile.get("profile_mode", ""))
        tokens = [self.sos, self.target_profile_begin, self.profile_mode, self._static(mode)]
        if mode == "PROFILE_COARSE":
            for key, tag in (
                ("cell_count_bin", self.cell_count_bin),
                ("demand_count_bin", self.demand_count_bin),
                ("source_count_bin", self.source_count_bin),
                ("seq_ratio_bin", self.seq_ratio_bin),
                ("pin_complexity_bin", self.pin_complexity_bin),
            ):
                config_key = key.removesuffix("_bin")
                active = set(
                    self.profile_config.get("active_profile_bins", {}).get(config_key, [])
                )
                if active and str(profile[key]) not in active:
                    raise ValueError(
                        f"inactive target profile bucket for {key}: {profile[key]}; "
                        f"train-active={sorted(active)}"
                    )
                tokens.extend([tag, self._static(profile[key])])
            tokens.append(self.complex_cell_flags)
            for key, tag in (
                ("rare_cell_enriched", self.rare_cell_enriched),
                ("has_mux", self.has_mux),
                ("has_aoi_oai", self.has_aoi_oai),
                ("has_multi_output_cell", self.has_multi_output_cell),
            ):
                tokens.extend([tag, self._static("BOOL_TRUE" if profile[key] else "BOOL_FALSE")])
        elif mode != "PROFILE_FREE":
            raise ValueError(f"unknown target profile mode: {mode}")
        tokens.append(self.target_profile_end)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, sequence):
        payload, report = self.parse_tokens(sequence)
        if not report["source_load_full_coverage"]:
            raise ValueError("ELDA profile extension decode requires full source-load coverage")
        if report["source_budget_violation"]:
            raise ValueError("ELDA profile extension decode rejected source budget violation")
        if report["same_cell_same_net_reuse"]:
            raise ValueError("ELDA profile extension decode rejected same-cell same-net input reuse")
        return decode_profiled_source_net(
            payload,
            net_id=self.net_id,
            boundary_stub_id=self.boundary_stub_id,
            cell_to_label=self.cell_to_label,
        )
