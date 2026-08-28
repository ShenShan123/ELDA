import torch
import ast
import csv
import json
import os
from collections import Counter
from pathlib import Path
from functools import partial
import pytorch_lightning as pl
from omegaconf import ListConfig
from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    Dataset,
    Sampler,
    WeightedRandomSampler,
)
from torch_geometric.loader import DataLoader as DataLoaderPyG
from .spectre_dataset import SpectreGraphDataset
from .synthetic_dataset import SyntheticDataset
from .networkx_dataset import NetworkXDataset
from .protein_dataset import ProteinDataset
from .point_cloud_dataset import PointCloudDataset
from .mol_dataset import QM9Dataset, MOSESDataset, GuacamolDataset
from .circuit_dataset import CircuitDataset
from .circuit_meta_dataset import CircuitPartitionDataset, CircuitSkeletonDataset, CircuitSourceNetV4NetsectionOnlyDataset
from .data.tokenizer import Graph2TrailTokenizer
from .data.circuit_pin_slot_tokenizer import CircuitPinSlotTokenizer, PinSlotSpec
from .data.circuit_pin_slot_gc_tokenizer import CircuitPinSlotGCTokenizer
from .data.circuit_pin_slot_v2_1_tokenizer import CircuitPinSlotV21Tokenizer, CircuitPinSlotV21GCTokenizer
from .data.circuit_source_net_v4_tokenizer import CircuitSourceNetV4Tokenizer
from .data.circuit_source_net_v41_tokenizer import CircuitSourceNetV41SinkFirstTokenizer
from .data.circuit_source_net_v5_tokenizer import CircuitSourceNetV5Tokenizer
from .data.circuit_source_net_v61_tokenizer import CircuitSourceNetV61Tokenizer
from .data.circuit_source_net_v62_tokenizer import CircuitSourceNetV62Tokenizer
from .data.circuit_source_net_v61_ablation_tokenizers import (
    CircuitSourceNetV61R1NoBoundaryIdentityTokenizer,
    CircuitSourceNetV61R2NoPerSourceBudgetTokenizer,
    CircuitSourceNetV61R3NoFullLoadAssignmentTokenizer,
    CircuitSourceNetV61R4CellLevelDemandTokenizer,
)


class ProfileMixtureSampler(Sampler):
    """Whole-partition 70/20/10 profile mixture with replacement."""

    def __init__(
        self, total_size, normal_indices, rare_indices, hard_indices, mixture,
        profile_labels=None, rare_subpools=None,
    ):
        self.total_size = int(total_size)
        self.pools = {
            "normal": torch.tensor(list(normal_indices), dtype=torch.long),
            "rare_complex": torch.tensor(list(rare_indices), dtype=torch.long),
            "hard": torch.tensor(list(hard_indices), dtype=torch.long),
        }
        self.mixture = {key: float(mixture[key]) for key in self.pools}
        if abs(sum(self.mixture.values()) - 1.0) > 1e-6:
            raise ValueError(f"profile sampler mixture must sum to 1: {self.mixture}")
        for key, pool in self.pools.items():
            if pool.numel() == 0 and self.mixture[key] > 0:
                raise ValueError(f"profile sampler pool {key} is empty")
        self.epoch = 0
        self.profile_labels = list(profile_labels or [""] * self.total_size)
        self.rare_subpools = {
            str(key): torch.tensor(list(values), dtype=torch.long)
            for key, values in dict(rare_subpools or {}).items()
            if len(values) > 0
        }
        self.last_epoch_report = None

    def __len__(self):
        return self.total_size

    def __iter__(self):
        counts = {
            "normal": int(round(self.total_size * self.mixture["normal"])),
            "rare_complex": int(round(self.total_size * self.mixture["rare_complex"])),
        }
        counts["hard"] = self.total_size - counts["normal"] - counts["rare_complex"]
        chunks = []
        for key in ("normal", "hard"):
            pool = self.pools[key]
            selected = pool[torch.randint(0, int(pool.numel()), (counts[key],))]
            chunks.append(selected)
        if self.rare_subpools:
            names = sorted(self.rare_subpools)
            base = counts["rare_complex"] // len(names)
            rare_counts = {name: base for name in names}
            for name in names[: counts["rare_complex"] - base * len(names)]:
                rare_counts[name] += 1
            rare_chunks = []
            for name in names:
                pool = self.rare_subpools[name]
                rare_chunks.append(pool[torch.randint(0, int(pool.numel()), (rare_counts[name],))])
            chunks.append(torch.cat(rare_chunks))
        else:
            pool = self.pools["rare_complex"]
            chunks.append(pool[torch.randint(0, int(pool.numel()), (counts["rare_complex"],))])
        result = torch.cat(chunks)
        result = result[torch.randperm(int(result.numel()))]
        self.epoch += 1
        selected = result.tolist()
        rare_set = set(self.pools["rare_complex"].tolist())
        hard_set = set(self.pools["hard"].tolist())
        combinations = Counter(
            self.profile_labels[index] for index in selected
            if 0 <= index < len(self.profile_labels) and self.profile_labels[index]
        )
        self.last_epoch_report = {
            "v62_profile_sampler_epoch": self.epoch,
            "draw_counts": counts,
            "pool_sizes": {key: int(value.numel()) for key, value in self.pools.items()},
            "rare_complex_draw_counts": (
                rare_counts if self.rare_subpools else {"union": counts["rare_complex"]}
            ),
            "rare_complex_subpool_sizes": {
                key: int(value.numel()) for key, value in self.rare_subpools.items()
            },
            "realized_rare_complex_draws": sum(index in rare_set for index in selected),
            "realized_hard_draws": sum(index in hard_set for index in selected),
            "realized_profile_combination_count": len(combinations),
            "realized_profile_combinations_top20": combinations.most_common(20),
        }
        print(json.dumps(self.last_epoch_report, sort_keys=True), flush=True)
        return iter(selected)
from elda.utils.circuit_semantic_contract import assert_semantic_dataset_root


def add_dataset_name(data, dataset_name):
    data.dataset_name = dataset_name
    return data


class _IndexSubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class _LazyOverlengthFilteredDataset(Dataset):
    def __init__(self, dataset, threshold: int, report_prefix: str = ""):
        self.dataset = dataset
        self.threshold = int(threshold)
        self.report_prefix = str(report_prefix)
        self.skipped = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        n = len(self.dataset)
        start = int(idx) % max(1, n)
        for offset in range(n):
            cur = (start + offset) % n
            item = self.dataset[cur]
            seq = item[0] if isinstance(item, (tuple, list)) else item
            length = int(seq.numel()) if hasattr(seq, "numel") else int(len(seq))
            if length <= self.threshold:
                if offset:
                    self.skipped += offset
                return item
        raise RuntimeError(f"{self.report_prefix} all samples exceed overlength threshold {self.threshold}")


def _resolve_schema_token_specs(root, schema_fields):
    if not schema_fields:
        return None
    meta_path = Path(root) / "meta.pt"
    if not meta_path.exists():
        return None
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    explicit_specs = list(meta.get("schema_token_specs", []))

    if explicit_specs:
        by_name = {str(spec.get("name")): spec for spec in explicit_specs if spec.get("name") is not None}
        requested_fields = [str(field) for field in schema_fields]

        if any(field in {"all", "*", "__all__"} for field in requested_fields):
            resolved = []
            for spec in explicit_specs:
                count = int(spec.get("count", 0))
                if count <= 0 or spec.get("name") is None:
                    continue
                resolved.append(
                    {
                        "name": str(spec["name"]),
                        "attr": str(spec.get("attr", f"schema_{spec['name']}")),
                        "count": int(count),
                    }
                )
            if resolved:
                return resolved

        specs = []
        # V3: Always attempt to include contract supply if available
        v3_mandatory = {"boundary_driver_stub_count", "boundary_load_stub_count"}
        for field in (list(requested_fields) + list(v3_mandatory)):
            if field in {s["name"] for s in specs}:
                continue
            spec = by_name.get(str(field))
            if spec is None:
                continue
            count = int(spec.get("count", 0))
            if count <= 0:
                continue
            specs.append(
                {
                    "name": str(spec["name"]),
                    "attr": str(spec.get("attr", f"schema_{field}")),
                    "count": int(count),
                }
            )
        if specs:
            return specs

    # Fallback to bucket specs if explicit specs are not found
    bucket_count_map = {}
    for key in meta.get("interface_bucket_keys", []):
        spec_name = {
            "size_bucket": "partition_size",
            "stub_bucket": "boundary_stub_count",
            "pin_bucket": "boundary_pin_count",
            "pin_deficit_bucket": "gate_pin_deficit_scaled",
            "underconnected_bucket": "underconnected_gate_fraction_scaled",
            "synthetic_input_bucket": "synthetic_input_proxy_ratio_scaled",
        }.get(key)
        if spec_name is None:
            continue
        bucket_count_map[key] = len(meta.get("buckets", {}).get(spec_name, {}).get("upper_bounds", []))

    specs = []
    for field in schema_fields:
        count = int(bucket_count_map.get(field, 0))
        if count <= 0:
            continue
        specs.append({"name": str(field), "attr": f"schema_{field}", "count": count})
    return specs or None


class GraphDataset(pl.LightningDataModule):
    datasets_map = {
        'planar': partial(SpectreGraphDataset, dataset_name='planar'),
        'sbm': partial(SpectreGraphDataset, dataset_name='sbm'),
        'comm20': partial(SpectreGraphDataset, dataset_name='comm20'),
        'ER': partial(SyntheticDataset, dataset_name='ER'),
        'BA': partial(SyntheticDataset, dataset_name='BA'),
        'waxman': partial(SyntheticDataset, dataset_name='waxman'),
        'random_geo': partial(SyntheticDataset, dataset_name='random_geo'),
        'classic': partial(NetworkXDataset, dataset_name='classic'),
        'lattice': partial(NetworkXDataset, dataset_name='lattice'),
        'small': partial(NetworkXDataset, dataset_name='small'),
        'random': partial(NetworkXDataset, dataset_name='random'),
        'geometric': partial(NetworkXDataset, dataset_name='geometric'),
        'tree': partial(NetworkXDataset, dataset_name='tree'),
        'community': partial(NetworkXDataset, dataset_name='community'),
        'social': partial(NetworkXDataset, dataset_name='social'),
        'classic_big': partial(NetworkXDataset, dataset_name='classic', big=True),
        'lattice_big': partial(NetworkXDataset, dataset_name='lattice', big=True),
        'small_big': partial(NetworkXDataset, dataset_name='small', big=True),
        'random_big': partial(NetworkXDataset, dataset_name='random', big=True),
        'geometric_big': partial(NetworkXDataset, dataset_name='geometric', big=True),
        'tree_big': partial(NetworkXDataset, dataset_name='tree', big=True),
        'community_big': partial(NetworkXDataset, dataset_name='community', big=True),
        'social_big': partial(NetworkXDataset, dataset_name='social', big=True),
        'DD': partial(ProteinDataset, dataset_name='DD'),
        'FIRSTMM_DB': partial(PointCloudDataset, dataset_name='FIRSTMM_DB'),
        'QM9': QM9Dataset,
        'MOSES': MOSESDataset,
        'Guacamol': GuacamolDataset,
        'CIRCUIT': CircuitDataset,
        'CIRCUIT_PARTITION': CircuitPartitionDataset,
        'CIRCUIT_ROLE_AWARE_PARTITION_V1': CircuitPartitionDataset,
        'CIRCUIT_PIN_SLOT_PARTITION_V2': CircuitPartitionDataset,
        'CIRCUIT_PIN_SLOT_PARTITION_V2_1_GC_MEDIUM': CircuitPartitionDataset,
        'CIRCUIT_PIN_SLOT_PARTITION_V2_1_GC_MEDIUM1024': CircuitPartitionDataset,
        'CIRCUIT_PIN_SLOT_PARTITION_V2_1_GC_MEDIUM_LONG4096': CircuitPartitionDataset,
        'CIRCUIT_PIN_SLOT_PARTITION_V2_1_GC_FULL': CircuitPartitionDataset,
        'CIRCUIT_SOURCE_NET_PARTITION_V4_1': CircuitPartitionDataset,
        'CIRCUIT_SOURCE_NET_PARTITION_V4_1_ROLE3FIX': CircuitPartitionDataset,
        'CIRCUIT_SOURCE_NET_PARTITION_V4_1_ROLE3FIX_SINKFIRST': CircuitPartitionDataset,
        'CIRCUIT_SOURCE_NET_PARTITION_V5_0_ROLE3FIX_DUALVIEW': CircuitPartitionDataset,
        'CIRCUIT_SOURCE_NET_PARTITION_V6_1_CLEAN_COMPACT_LOAD': CircuitPartitionDataset,
        'CIRCUIT_SOURCE_NET_PARTITION_V6_2_PROFILE': CircuitPartitionDataset,
        'CIRCUIT_SOURCE_NET_PARTITION_V6_1_R1_NO_BOUNDARY_IDENTITY': CircuitPartitionDataset,
        'CIRCUIT_SOURCE_NET_PARTITION_V6_1_R2_NO_PER_SOURCE_BUDGET': CircuitPartitionDataset,
        'CIRCUIT_SOURCE_NET_PARTITION_V6_1_R3_NO_FULL_LOAD_ASSIGNMENT': CircuitPartitionDataset,
        'CIRCUIT_SOURCE_NET_PARTITION_V6_1_R4_CELL_LEVEL_DEMAND': CircuitPartitionDataset,
        'CIRCUIT_SOURCE_NET_PARTITION_V4_1_ROLE3FIX_NETSECTION_ONLY': CircuitSourceNetV4NetsectionOnlyDataset,
        'CIRCUIT_SKELETON': CircuitSkeletonDataset,
    }

    def __init__(
        self,
        root,
        dataset_names='all',
        tokenizer=None,
        init_tokenizer=True,
        max_length=-1,
        truncation_length=None,
        labeled_graph=False,
        undirected=True,
        tokenizer_type: str = 'sent',
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.dataset_names = dataset_names
        self.set_val_metric(dataset_names)
        if dataset_names == 'all':
            self.dataset_names = list(self.datasets_map.keys())
        elif dataset_names == 'spectre':
            self.dataset_names = ['planar', 'sbm', 'comm20']
        elif dataset_names == 'synthetic':
            self.dataset_names = ['ER', 'BA', 'waxman', 'random_geo']
        elif dataset_names == 'networkx':
            self.dataset_names = ['classic', 'lattice', 'small', 'random', 'geometric']
        elif dataset_names == 'networkx_big':
            self.dataset_names = ['classic_big', 'lattice_big', 'small_big', 'random_big', 'geometric_big', 'tree_big', 'community_big']
        elif dataset_names == 'protein':
            self.dataset_names = ['DD']
        elif dataset_names == 'point_cloud':
            self.dataset_names = ['FIRSTMM_DB']
        else:
            if isinstance(dataset_names, (list, tuple, ListConfig)):
                self.dataset_names = [str(name) for name in dataset_names]
            else:
                self.dataset_names = [dataset_names]
            for dataset_name in self.dataset_names:
                assert dataset_name in self.datasets_map.keys(), 'Not included in the database!'

        self.labeled_graph = labeled_graph
        self.tokenizer_type = str(tokenizer_type)
        self.serializer_version = kwargs.pop("serializer_version", None)
        self.source_net_v5_compact = bool(kwargs.pop("source_net_v5_compact", False))
        self.compact_demand_record = bool(kwargs.pop("compact_demand_record", self.source_net_v5_compact))
        self.compact_sink_demand_record = bool(kwargs.pop("compact_sink_demand_record", self.source_net_v5_compact))
        self.compact_source_net_hist = bool(kwargs.pop("compact_source_net_hist", self.source_net_v5_compact))
        self.source_clean_split_required = bool(kwargs.pop("source_clean_split_required", False))
        self.strict_no_cross_duplicate_split_required = bool(
            kwargs.pop("strict_no_cross_duplicate_split_required", False)
        )
        self.expected_split_counts = {
            str(split): int(count)
            for split, count in dict(kwargs.pop("expected_split_counts", {}) or {}).items()
        }
        if self.source_clean_split_required or self.strict_no_cross_duplicate_split_required:
            self._validate_partition_split_contract()
        self.no_silent_truncate = bool(kwargs.pop("no_silent_truncate", False))
        self.overlength_policy = str(kwargs.pop("overlength_policy", "allow"))
        self.overlength_threshold = int(kwargs.pop("overlength_threshold", max_length if max_length is not None else -1))
        self.overlength_guard_eager_scan = bool(kwargs.pop("overlength_guard_eager_scan", False))
        if self.tokenizer_type in {
            "source_net_v5", "source_net_v61", "source_net_v62", "source_net_v61_r1",
            "source_net_v61_r2", "source_net_v61_r3", "source_net_v61_r4",
        } and self.no_silent_truncate:
            truncation_length = None
        self.tokenizer = tokenizer
        self.weighted_sampling = bool(kwargs.pop("weighted_sampling", False))
        self.medium_type_sample_scale = float(kwargs.pop("medium_type_sample_scale", 0.0))
        self.diverse_sample_scale = float(kwargs.pop("diverse_sample_scale", 0.0))
        self.clean_sample_scale = float(kwargs.pop("clean_sample_scale", 0.0))
        self.sample_weight_max = float(kwargs.pop("sample_weight_max", 4.0))
        self.sample_weight_power = float(kwargs.pop("sample_weight_power", 1.0))
        self.gate_type_profile_path = kwargs.pop("gate_type_profile_path", None)
        self.rare_cell_partition_sampling_profile_path = kwargs.pop("rare_cell_partition_sampling_profile_path", None)
        self.rare_cell_sampling_strategy = str(kwargs.pop("rare_cell_sampling_strategy", ""))
        self.rare_mid_partition_sample_scale = float(kwargs.pop("rare_mid_partition_sample_scale", 0.0))
        self.high_pin_complexity_partition_sample_scale = float(kwargs.pop("high_pin_complexity_partition_sample_scale", 0.0))
        self.source_pressure_partition_sample_scale = float(kwargs.pop("source_pressure_partition_sample_scale", 0.0))
        self.balanced_v3_1_sampling_profile_path = kwargs.pop("balanced_v3_1_sampling_profile_path", None)
        self.balanced_v3_2_sampling_profile_path = kwargs.pop("balanced_v3_2_sampling_profile_path", None)
        self.balanced_v3_3_sampling_profile_path = kwargs.pop("balanced_v3_3_sampling_profile_path", None)
        self.balanced_v3_4a_sampling_profile_path = kwargs.pop("balanced_v3_4a_sampling_profile_path", None)
        self.balanced_v3_5_sampling_profile_path = kwargs.pop("balanced_v3_5_sampling_profile_path", None)
        self.distribution_guard_enabled = bool(kwargs.pop("distribution_guard_enabled", False))
        self.clean_but_buf_only_guard_enabled = bool(kwargs.pop("clean_but_buf_only_guard_enabled", False))
        self.source_clean_load_coverage_reporting = bool(kwargs.pop("source_clean_load_coverage_reporting", False))
        self.cut_group_ready_score_reporting = bool(kwargs.pop("cut_group_ready_score_reporting", False))
        self.cut_group_request_ready_reporting = bool(kwargs.pop("cut_group_request_ready_reporting", False))
        self.source_clean_cut_rich_reporting = bool(kwargs.pop("source_clean_cut_rich_reporting", False))
        self.source_clean_cut_request_rich_reporting = bool(kwargs.pop("source_clean_cut_request_rich_reporting", False))
        self.tail_source_clean_cut_request_rich_reporting = bool(kwargs.pop("tail_source_clean_cut_request_rich_reporting", False))
        self.multi_output_cut_request_rich_reporting = bool(kwargs.pop("multi_output_cut_request_rich_reporting", False))
        self.high_gc_cut_rich_reporting = bool(kwargs.pop("high_gc_cut_rich_reporting", False))
        self.source_diversity_reporting = bool(kwargs.pop("source_diversity_reporting", False))
        self.critical_output_reporting = bool(kwargs.pop("critical_output_reporting", False))
        self.role_mixed_high_fanout_reporting = bool(kwargs.pop("role_mixed_high_fanout_reporting", False))
        self.repeated_input_signature_reporting = bool(kwargs.pop("repeated_input_signature_reporting", False))
        self.high_gc_cut_request_rich_reporting = bool(kwargs.pop("high_gc_cut_request_rich_reporting", False))
        self.clean_but_low_cut_request_guard_enabled = bool(kwargs.pop("clean_but_low_cut_request_guard_enabled", False))
        self.cell_mapping_path = kwargs.pop("cell_mapping_path", None)
        self.aux_fields = kwargs.pop("aux_fields", None)
        self.condition_dropout_probability = float(
            kwargs.pop("condition_dropout_probability", 0.0)
        )
        self.profile_config_path = kwargs.pop("profile_config_path", None)
        self.profile_prefix_enabled = bool(kwargs.pop("profile_prefix_enabled", True))
        self.profile_balanced_sampling = bool(
            kwargs.pop("profile_balanced_sampling", False)
        )
        self.profile_sampling_rows_path = kwargs.pop("profile_sampling_rows_path", None)
        raw_mixture = kwargs.pop("profile_sampling_mixture", None)
        self.profile_sampling_mixture = dict(raw_mixture or {
            "normal": 0.70, "rare_complex": 0.20, "hard": 0.10,
        })
        self.schema_fields = kwargs.pop("schema_fields", None)
        self.gate_count_buckets = kwargs.pop("gate_count_buckets", None)
        self.driver_stub_buckets = kwargs.pop("driver_stub_buckets", None)
        self.load_stub_buckets = kwargs.pop("load_stub_buckets", None)
        self.driver_load_ratio_buckets = kwargs.pop("driver_load_ratio_buckets", None)
        self.enable_contract_prefix = bool(kwargs.pop("enable_contract_prefix", True))
        self._semantic_contract_report = None
        if any(name in {"CIRCUIT_PARTITION", "CIRCUIT_SKELETON"} for name in self.dataset_names):
            self._semantic_contract_report = assert_semantic_dataset_root(
                root=self.root,
                dataset_names=self.dataset_names,
                schema_fields=self.schema_fields,
                gate_type_profile_path=self.gate_type_profile_path,
                require_gate_profile=bool(self.weighted_sampling or self.gate_type_profile_path),
            )
        if tokenizer is None and init_tokenizer:
            schema_token_specs = _resolve_schema_token_specs(self.root, self.schema_fields)
            if self.tokenizer_type in {'pin_slot', 'pin_slot_gc', 'pin_slot_v2_1', 'pin_slot_v2_1_gc', 'source_net_v4', 'source_net_v41_sinkfirst', 'source_net_v5', 'source_net_v61', 'source_net_v62', 'source_net_v61_r1', 'source_net_v61_r2', 'source_net_v61_r3', 'source_net_v61_r4'}:
                meta_path = Path(self.root) / 'meta.pt'
                meta = torch.load(meta_path, map_location='cpu', weights_only=False) if meta_path.exists() else {}
                net_id = int(meta.get('net_id', 0))
                boundary_stub_id = int(meta.get('boundary_stub_id', -1))
                mapping_path = Path(self.cell_mapping_path) if self.cell_mapping_path else Path(str(meta.get('cell_mapping', '')))
                label_to_cell = self._load_label_to_cell_mapping(mapping_path) if mapping_path.exists() else {}
                pin_specs = {}
                try:
                    from circuit_kahypar.pin_spec_table import CellPinSpecTable
                    table = CellPinSpecTable(allow_fallback=True)
                    if (
                        self.tokenizer_type.startswith("source_net_v61")
                        or self.tokenizer_type.startswith("source_net_v62")
                    ) and table.liberty_path is None:
                        raise FileNotFoundError(
                            "Nangate45 Liberty file not found; set "
                            "ELDA_NANGATE45_LIBERTY or OPENROAD_FLOW_ROOT"
                        )
                    table.pre_register_cells(label_to_cell.values())
                    pin_names = set()
                    for cell in label_to_cell.values():
                        spec = table.get_cell_pin_spec(str(cell))
                        inputs = [str(p) for p in spec.inputs]
                        outputs = [str(p) for p in spec.outputs]
                        min_out = int(spec.min_required_outputs)
                        pin_specs[str(cell)] = PinSlotSpec(inputs=inputs, outputs=outputs, min_required_outputs=min_out)
                        pin_names.update(inputs)
                        pin_names.update(outputs)
                    pin_names.update({'__EXCESS__', '__UNKNOWN__'})
                    pin_names = sorted(pin_names)
                except Exception:
                    # Endpoint-complete V6.1/V6.2 serialization is defined by
                    # the target Liberty pin vocabulary. Silently falling back
                    # to an empty vocabulary would create non-reproducible
                    # sequences and delayed grammar failures.
                    if (
                        self.tokenizer_type.startswith("source_net_v61")
                        or self.tokenizer_type.startswith("source_net_v62")
                    ):
                        raise
                    pin_names = []
                tok_cls = {
                    'pin_slot': CircuitPinSlotTokenizer,
                    'pin_slot_gc': CircuitPinSlotGCTokenizer,
                    'pin_slot_v2_1': CircuitPinSlotV21Tokenizer,
                    'pin_slot_v2_1_gc': CircuitPinSlotV21GCTokenizer,
                    'source_net_v4': CircuitSourceNetV4Tokenizer,
                    'source_net_v41_sinkfirst': CircuitSourceNetV41SinkFirstTokenizer,
                    'source_net_v5': CircuitSourceNetV5Tokenizer,
                    'source_net_v61': CircuitSourceNetV61Tokenizer,
                    'source_net_v62': CircuitSourceNetV62Tokenizer,
                    'source_net_v61_r1': CircuitSourceNetV61R1NoBoundaryIdentityTokenizer,
                    'source_net_v61_r2': CircuitSourceNetV61R2NoPerSourceBudgetTokenizer,
                    'source_net_v61_r3': CircuitSourceNetV61R3NoFullLoadAssignmentTokenizer,
                    'source_net_v61_r4': CircuitSourceNetV61R4CellLevelDemandTokenizer,
                }[self.tokenizer_type]
                tok_kwargs = {}
                if tok_cls in {CircuitPinSlotGCTokenizer, CircuitPinSlotV21GCTokenizer} and self.gate_count_buckets is not None:
                    tok_kwargs['gate_count_buckets'] = self.gate_count_buckets
                if tok_cls is CircuitPinSlotV21GCTokenizer:
                    tok_kwargs['enable_contract_prefix'] = self.enable_contract_prefix
                    if self.driver_stub_buckets is not None:
                        tok_kwargs['driver_stub_buckets'] = self.driver_stub_buckets
                    if self.load_stub_buckets is not None:
                        tok_kwargs['load_stub_buckets'] = self.load_stub_buckets
                    if self.driver_load_ratio_buckets is not None:
                        tok_kwargs['driver_load_ratio_buckets'] = self.driver_load_ratio_buckets
                if tok_cls is CircuitSourceNetV5Tokenizer:
                    tok_kwargs['source_net_v5_compact'] = self.source_net_v5_compact
                    tok_kwargs['compact_demand_record'] = self.compact_demand_record
                    tok_kwargs['compact_sink_demand_record'] = self.compact_sink_demand_record
                    tok_kwargs['compact_source_net_hist'] = self.compact_source_net_hist
                if tok_cls is CircuitSourceNetV62Tokenizer:
                    profile_path = Path(
                        self.profile_config_path
                        or meta.get('profile_config_path', '')
                        or (Path(self.root) / 'v62_profile_config.json')
                    )
                    if not profile_path.exists():
                        raise FileNotFoundError(
                            f"V6.2_PROFILE requires frozen train-only profile config: {profile_path}"
                        )
                    tok_kwargs['profile_config'] = json.loads(
                        profile_path.read_text(encoding='utf-8')
                    )
                    tok_kwargs['condition_dropout_prob'] = self.condition_dropout_probability
                    tok_kwargs['profile_prefix_enabled'] = self.profile_prefix_enabled
                self.tokenizer = tok_cls(
                    dataset_names=[],
                    max_length=max_length,
                    truncation_length=truncation_length,
                    append_eos=True,
                    net_id=net_id,
                    boundary_stub_id=boundary_stub_id,
                    label_to_cell=label_to_cell,
                    pin_specs=pin_specs,
                    **tok_kwargs,
                )
                self.tokenizer.set_pin_name_vocab(pin_names)
            else:
                self.tokenizer = Graph2TrailTokenizer(
                    dataset_names=[],
                    max_length=max_length,
                    truncation_length=truncation_length,
                    labeled_graph=labeled_graph,
                    undirected=undirected,
                    aux_fields=self.aux_fields,
                    schema_token_specs=schema_token_specs,
                )
        elif self.tokenizer is not None and self.aux_fields is not None:
            self.tokenizer.aux_fields = list(self.aux_fields)
        if self.tokenizer is not None and self.schema_fields is not None:
            schema_token_specs = _resolve_schema_token_specs(self.root, self.schema_fields)
            self.tokenizer.schema_token_specs = list(schema_token_specs) if schema_token_specs is not None else []
            self.tokenizer.schema_token_ranges = {}
            schema_offset = len(self.tokenizer.special_toks) + len(self.tokenizer.dataset_names)
            for spec in self.tokenizer.schema_token_specs:
                name = str(spec["name"])
                count = int(spec["count"])
                self.tokenizer.schema_token_ranges[name] = (schema_offset, count)
                schema_offset += count
            self.tokenizer.schema_token_count = schema_offset - (len(self.tokenizer.special_toks) + len(self.tokenizer.dataset_names))
            self.tokenizer.idx_offset = schema_offset

        self.kwargs = kwargs
        self.collate_fn = self.tokenizer.batch_converter() if self.tokenizer is not None else None
        self._train_sampler = None
        self._rare_cell_sampling_rows = None
        self._balanced_v3_sampler_report = None
        self._overlength_guard_reports = {}

    @staticmethod
    def _load_label_to_cell_mapping(path):
        text = Path(path).read_text(encoding="utf-8")
        _, rhs = text.split("=", 1)
        mapping = ast.literal_eval(rhs.strip())
        return {int(label): str(cell) for cell, label in mapping.items()}

    def _has_partition_dataset(self):
        return any(self.datasets_map.get(name) is CircuitPartitionDataset for name in self.dataset_names)

    def _validate_partition_split_contract(self):
        meta_path = Path(self.root) / "meta.pt"
        if not meta_path.exists():
            raise RuntimeError(f"Required partition split metadata is missing: {meta_path}")
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        split_file_lists = dict(meta.get("split_file_lists") or {})
        for split in ("train", "val", "test"):
            configured = split_file_lists.get(split)
            if not configured:
                raise RuntimeError(f"meta.pt missing required split_file_lists.{split}: {meta_path}")
            path = Path(str(configured))
            if not path.is_absolute():
                path = Path(self.root) / path
            if not path.is_file():
                raise RuntimeError(f"Required {split} split list does not exist: {path}")
            if split in self.expected_split_counts:
                with path.open("r", encoding="utf-8") as handle:
                    actual = sum(1 for line in handle if line.strip())
                expected = self.expected_split_counts[split]
                if actual != expected:
                    raise RuntimeError(
                        f"Partition split count mismatch for {split}: actual={actual}, "
                        f"expected={expected}, path={path}"
                    )
        if self.strict_no_cross_duplicate_split_required:
            strict_filter = dict(meta.get("strict_no_cross_duplicate_filter") or {})
            if not strict_filter.get("enabled", False):
                raise RuntimeError(
                    f"Strict no-cross-duplicate split is required but not published in {meta_path}"
                )
            if int(strict_filter.get("cross_split_exact_duplicate_groups", -1)) != 0:
                raise RuntimeError(f"Cross-split exact duplicate groups remain in {meta_path}")
            if int(strict_filter.get("cross_split_near_duplicate_pairs", -1)) != 0:
                raise RuntimeError(f"Cross-split near-duplicate pairs remain in {meta_path}")

    def _resolve_partition_weighting_assets(self):
        if not self.weighted_sampling or not self._has_partition_dataset():
            return None, None, None
        root = Path(self.root)
        meta_path = root / "meta.pt"
        if not meta_path.exists():
            return None, None, None
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        profile_path = Path(self.gate_type_profile_path) if self.gate_type_profile_path else root / "gate_type_profile.json"
        mapping_path = Path(self.cell_mapping_path) if self.cell_mapping_path else Path(str(meta.get("cell_mapping", "")))
        if not profile_path.exists() or not mapping_path.exists():
            return None, None, None
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        medium_types = set(profile.get("medium_types", []))
        label_to_cell = self._load_label_to_cell_mapping(mapping_path)
        medium_labels = {label for label, cell in label_to_cell.items() if cell in medium_types}
        return meta, label_to_cell, medium_labels

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value, default=0):
        try:
            if value is None or value == "":
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _load_rare_cell_sampling_rows(self):
        if self._rare_cell_sampling_rows is not None:
            return self._rare_cell_sampling_rows
        rows = {}
        path_value = self.rare_cell_partition_sampling_profile_path
        if not path_value:
            self._rare_cell_sampling_rows = rows
            return rows
        path = Path(path_value)
        if not path.exists():
            self._rare_cell_sampling_rows = rows
            return rows

        def add_row(row):
            graph_path = str(row.get("path") or row.get("graph_path") or "")
            graph_name = str(row.get("graph_name") or row.get("name") or Path(graph_path).name)
            if graph_path:
                rows[graph_path] = row
                rows[str(Path(graph_path).resolve())] = row
                rows[Path(graph_path).name] = row
            if graph_name:
                rows[graph_name] = row

        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    add_row(row)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for row in payload.get("rows", []):
                    if isinstance(row, dict):
                        add_row(row)
            elif isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        add_row(row)
        self._rare_cell_sampling_rows = rows
        return rows

    def _partition_sampling_profile_for_path(self, path):
        rows = self._load_rare_cell_sampling_rows()
        if not rows or path is None:
            return None
        path = Path(path)
        return rows.get(str(path)) or rows.get(str(path.resolve())) or rows.get(path.name)

    def _partition_profile_weight_terms(self, row):
        if not row:
            return None
        if (
            "balanced_v3_2" in str(self.rare_cell_sampling_strategy)
            and row.get("balanced_v3_2_sample_weight") not in (None, "")
        ):
            return {
                "direct_weight": max(
                    1e-6,
                    min(
                        float(self.sample_weight_max),
                        self._safe_float(row.get("balanced_v3_2_sample_weight"), 1.0),
                    ),
                )
            }
        if (
            "balanced_v3_3" in str(self.rare_cell_sampling_strategy)
            and row.get("balanced_v3_3_sample_weight") not in (None, "")
        ):
            return {
                "direct_weight": max(
                    1e-6,
                    min(
                        float(self.sample_weight_max),
                        self._safe_float(row.get("balanced_v3_3_sample_weight"), 1.0),
                    ),
                )
            }
        if (
            "balanced_v3_4a" in str(self.rare_cell_sampling_strategy)
            and row.get("balanced_v3_4a_sample_weight") not in (None, "")
        ):
            return {
                "direct_weight": max(
                    1e-6,
                    min(
                        float(self.sample_weight_max),
                        self._safe_float(row.get("balanced_v3_4a_sample_weight"), 1.0),
                    ),
                )
            }
        if (
            "balanced_v3_5" in str(self.rare_cell_sampling_strategy)
            and row.get("balanced_v3_5_sample_weight") not in (None, "")
        ):
            return {
                "direct_weight": max(
                    1e-6,
                    min(
                        float(self.sample_weight_max),
                        self._safe_float(row.get("balanced_v3_5_sample_weight"), 1.0),
                    ),
                )
            }
        gate_count = max(1.0, self._safe_float(row.get("gate_count"), 1.0))
        rare_count = self._safe_float(row.get("rare_cell_count"), 0.0)
        mid_count = self._safe_float(row.get("mid_cell_count"), 0.0)
        used_type_count = self._safe_float(row.get("used_type_count"), 0.0)
        source_pressure = self._safe_float(row.get("source_legalizability_pressure"), 0.0)
        input_pool_pressure = self._safe_float(row.get("input_pool_pressure_proxy"), 0.0)
        reused_pressure = self._safe_float(row.get("reused_input_pressure_proxy"), 0.0)
        high_complexity = bool(self._safe_int(row.get("high_complexity_partition"), 0))
        medium_ratio = min(1.0, (rare_count + mid_count) / gate_count)
        diversity_score = min(1.0, used_type_count / 12.0)
        pressure_ratio = min(1.0, (source_pressure + input_pool_pressure + reused_pressure) / gate_count)
        clean_score = 1.0 - pressure_ratio
        rare_mid_score = 1.0 if (rare_count > 0 or mid_count > 0) else 0.0
        hard_score = 1.0 if high_complexity else 0.0
        source_score = pressure_ratio
        return {
            "medium_ratio": medium_ratio,
            "diversity_score": diversity_score,
            "clean_score": clean_score,
            "rare_mid_score": rare_mid_score,
            "hard_score": hard_score,
            "source_score": source_score,
        }

    def _build_partition_train_sampler(self):
        if not isinstance(self.train_dataset, ConcatDataset):
            return None
        if self.profile_balanced_sampling:
            return self._build_v62_profile_sampler()
        meta, label_to_cell, medium_labels = self._resolve_partition_weighting_assets()
        if meta is None:
            return None
        net_id = int(meta.get("net_id", 96))
        boundary_stub_id = int(meta.get("boundary_stub_id", 97))
        weights = []
        profile_hits = 0
        graph_fallbacks = 0
        rare_mid_boosted = 0
        hard_boosted = 0
        source_boosted = 0
        direct_weighted = 0
        for dataset in self.train_dataset.datasets:
            if not isinstance(dataset, CircuitPartitionDataset):
                weights.extend([1.0] * len(dataset))
                continue
            for idx in range(len(dataset)):
                profile_terms = None
                if hasattr(dataset, "_paths"):
                    profile_terms = self._partition_profile_weight_terms(
                        self._partition_sampling_profile_for_path(dataset._paths[int(idx)])
                    )
                if profile_terms is not None:
                    profile_hits += 1
                    direct_weight = profile_terms.get("direct_weight")
                    if direct_weight is not None:
                        direct_weighted += 1
                        weights.append(float(direct_weight))
                        continue
                    medium_ratio = float(profile_terms["medium_ratio"])
                    diversity_score = float(profile_terms["diversity_score"])
                    clean_score = float(profile_terms["clean_score"])
                    rare_mid_score = float(profile_terms["rare_mid_score"])
                    hard_score = float(profile_terms["hard_score"])
                    source_score = float(profile_terms["source_score"])
                else:
                    graph_fallbacks += 1
                    data = dataset.get(idx)
                    x = data.x.reshape(-1).to(torch.long)
                    gate_mask = (x != net_id) & (x != boundary_stub_id)
                    gate_labels = x[gate_mask]
                    gate_count = int(gate_labels.numel())
                    if gate_count <= 0:
                        weights.append(1.0)
                        continue
                    medium_ratio = float(sum(int(label in medium_labels) for label in gate_labels.tolist())) / float(gate_count)
                    diversity_score = min(1.0, float(torch.unique(gate_labels).numel()) / 12.0)
                    clean_score = 1.0 - (
                        min(1.0, float(getattr(data, "gate_pin_deficit", 0.0)))
                        + min(1.0, float(getattr(data, "underconnected_gate_fraction", 0.0)))
                        + min(1.0, float(getattr(data, "synthetic_input_proxy_ratio", 0.0)))
                    ) / 3.0
                    rare_mid_score = 0.0
                    hard_score = 0.0
                    source_score = 0.0
                weight = (
                    1.0
                    + self.medium_type_sample_scale * medium_ratio
                    + self.diverse_sample_scale * diversity_score
                    + self.clean_sample_scale * clean_score
                    + self.rare_mid_partition_sample_scale * rare_mid_score
                    + self.high_pin_complexity_partition_sample_scale * hard_score
                    + self.source_pressure_partition_sample_scale * source_score
                )
                if rare_mid_score > 0:
                    rare_mid_boosted += 1
                if hard_score > 0:
                    hard_boosted += 1
                if source_score > 0:
                    source_boosted += 1
                if self.sample_weight_power != 1.0:
                    weight = weight ** self.sample_weight_power
                weights.append(min(self.sample_weight_max, max(1.0, float(weight))))
        if not weights or max(weights) - min(weights) < 1e-6:
            return None
        weight_tensor = torch.tensor(weights, dtype=torch.double)
        self._balanced_v3_sampler_report = {
            "weighted_sampling": True,
            "strategy": self.rare_cell_sampling_strategy,
            "profile_path": str(self.rare_cell_partition_sampling_profile_path or ""),
            "profile_hits": int(profile_hits),
            "graph_fallbacks": int(graph_fallbacks),
            "num_weights": int(len(weights)),
            "weight_min": float(weight_tensor.min().item()),
            "weight_mean": float(weight_tensor.mean().item()),
            "weight_max": float(weight_tensor.max().item()),
            "rare_mid_boosted": int(rare_mid_boosted),
            "hard_boosted": int(hard_boosted),
            "source_boosted": int(source_boosted),
            "direct_weighted": int(direct_weighted),
            "sample_weight_max": float(self.sample_weight_max),
        }
        print(f"[balanced_v3_sampler] {json.dumps(self._balanced_v3_sampler_report, sort_keys=True)}")
        return WeightedRandomSampler(weight_tensor, num_samples=len(weights), replacement=True)

    def _build_v62_profile_sampler(self):
        path = Path(
            self.profile_sampling_rows_path
            or (Path(self.root) / "v62_profile_rows_train.csv")
        )
        if not path.exists():
            raise FileNotFoundError(f"missing V6.2 profile sampling rows: {path}")
        rows = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                graph_path = str(row.get("path", ""))
                rows[graph_path] = row
                rows[Path(graph_path).name] = row
        all_indices = []
        rare_indices = []
        rare_cell_indices = []
        mux_indices = []
        aoi_oai_indices = []
        hard_indices = []
        profile_labels = []
        offset = 0
        missing = []
        for dataset in self.train_dataset.datasets:
            for local_index in range(len(dataset)):
                global_index = offset + local_index
                all_indices.append(global_index)
                graph_path = (
                    dataset._paths[local_index]
                    if hasattr(dataset, "_paths") else None
                )
                row = rows.get(str(graph_path)) or (
                    rows.get(Path(graph_path).name) if graph_path is not None else None
                )
                if row is None:
                    missing.append(str(graph_path))
                    profile_labels.append("")
                    continue
                profile_labels.append(str(row.get("profile_combination", "")))
                is_rare = any(
                    int(float(row.get(key, 0) or 0))
                    for key in ("rare_cell_enriched", "has_mux", "has_aoi_oai")
                )
                is_hard = str(row.get("pin_complexity_bin")) == "PROFILE_BIN_3"
                if is_rare:
                    rare_indices.append(global_index)
                if int(float(row.get("rare_cell_enriched", 0) or 0)):
                    rare_cell_indices.append(global_index)
                if int(float(row.get("has_mux", 0) or 0)):
                    mux_indices.append(global_index)
                if int(float(row.get("has_aoi_oai", 0) or 0)):
                    aoi_oai_indices.append(global_index)
                if is_hard:
                    hard_indices.append(global_index)
            offset += len(dataset)
        if missing:
            raise RuntimeError(
                f"V6.2 profile sampler missing {len(missing)} train rows; examples={missing[:5]}"
            )
        report = {
            "total": len(all_indices), "normal_pool": len(all_indices),
            "rare_complex_pool": len(rare_indices), "hard_pool": len(hard_indices),
            "rare_complex_subpools": {
                "rare_cell": len(rare_cell_indices),
                "mux": len(mux_indices),
                "aoi_oai": len(aoi_oai_indices),
            },
            "mixture": self.profile_sampling_mixture,
            "whole_partition_sampling": True,
        }
        self._balanced_v3_sampler_report = report
        print(f"[v62_profile_sampler] {json.dumps(report, sort_keys=True)}")
        return ProfileMixtureSampler(
            len(all_indices), all_indices, rare_indices, hard_indices,
            self.profile_sampling_mixture, profile_labels=profile_labels,
            rare_subpools={
                "rare_cell": rare_cell_indices,
                "mux": mux_indices,
                "aoi_oai": aoi_oai_indices,
            },
        )

    def set_val_metric(self, dataset_names):
        if dataset_names in ['planar', 'sbm']:
            self.val_metric = ('vun', 'max')
        else:
            self.val_metric = ('loss', 'min')

    @property
    def atom_decoder(self):
        if self.labeled_graph:
            return self.train_dataset.datasets[0].atom_decoder
        return None

    @property
    def train_smiles(self):
        if self.labeled_graph:
            return self.train_dataset.datasets[0].smiles
        return None

    def prepare_data(self):
        max_num_nodes = {}
        meta_max_num_nodes = None
        meta_path = Path(self.root) / "meta.pt"
        if meta_path.exists():
            try:
                meta = torch.load(meta_path, map_location="cpu", weights_only=False)
                if meta.get("max_num_nodes", None) is not None:
                    meta_max_num_nodes = int(meta["max_num_nodes"])
            except Exception:
                meta_max_num_nodes = None
        for dataset_name in self.dataset_names:
            train_dataset = self.datasets_map[dataset_name](
                root=self.root, split='train', pre_transform=partial(add_dataset_name, dataset_name=dataset_name)
            )
            self.datasets_map[dataset_name](
                root=self.root, split='val', pre_transform=partial(add_dataset_name, dataset_name=dataset_name)
            )
            try:
                self.datasets_map[dataset_name](
                    root=self.root, split='test', pre_transform=partial(add_dataset_name, dataset_name=dataset_name)
                )
            except Exception:
                pass
            if self.tokenizer is not None:
                if meta_max_num_nodes is not None and meta_max_num_nodes > 0:
                    max_num_nodes[dataset_name] = meta_max_num_nodes
                else:
                    max_num_nodes[dataset_name] = max([g.num_nodes for g in train_dataset])
        if self.tokenizer is not None:
            self.max_num_nodes = max_num_nodes
            self.tokenizer.set_num_nodes(max(max_num_nodes.values()))
            print(self.max_num_nodes)
            if self.labeled_graph:
                # TODO: support multiple molecule datasets
                self.tokenizer.set_num_node_and_edge_types(
                    num_node_types=train_dataset.num_node_types,
                    num_edge_types=train_dataset.num_edge_types,
                )
            else:
                if hasattr(train_dataset, 'num_node_types'):
                    self.tokenizer.set_num_node_and_edge_types(
                        num_node_types=getattr(train_dataset, 'num_node_types', 0),
                        num_edge_types=getattr(train_dataset, 'num_edge_types', 0),
                    )

    def setup(self, stage='fit'):
        if stage == 'fit':
            train_dataset = [self._apply_source_net_v5_overlength_guard(self.datasets_map[dataset_name](
                root=self.root,
                split='train',
                transform=self.tokenizer,
                pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
            ), split="train", dataset_name=dataset_name) for dataset_name in self.dataset_names]
            self.train_dataset = ConcatDataset(train_dataset)
            self._train_sampler = self._build_partition_train_sampler()
            val_dataset = [self._apply_source_net_v5_overlength_guard(self.datasets_map[dataset_name](
                root=self.root,
                split='val',
                transform=self.tokenizer,
                pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
            ), split="val", dataset_name=dataset_name) for dataset_name in self.dataset_names]
            self.val_dataset = ConcatDataset(val_dataset)

        if stage == 'test':
            try:
                test_dataset = [self._apply_source_net_v5_overlength_guard(self.datasets_map[dataset_name](
                    root=self.root,
                    split='test',
                    transform=self.tokenizer,
                    pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
                ), split="test", dataset_name=dataset_name) for dataset_name in self.dataset_names]
                self.test_dataset = ConcatDataset(test_dataset)
            except Exception:
                pass

    def _apply_source_net_v5_overlength_guard(self, dataset, split: str, dataset_name: str):
        if self.tokenizer_type != "source_net_v5" or not self.no_silent_truncate:
            return dataset
        if self.overlength_policy not in {"exclude", "exclude_or_split"}:
            return dataset
        threshold = int(self.overlength_threshold)
        if threshold <= 0:
            raise ValueError("source_net_v5 no_silent_truncate requires overlength_threshold > 0")
        if not self.overlength_guard_eager_scan:
            report = {
                "dataset_name": str(dataset_name),
                "split": str(split),
                "policy": self.overlength_policy,
                "threshold": threshold,
                "mode": "lazy_runtime_exclude",
                "total": int(len(dataset)),
                "no_silent_truncate": True,
            }
            self._overlength_guard_reports[f"{split}:{dataset_name}"] = report
            try:
                out = Path(os.environ.get("ELDA_LOG_ROOT", "logs")) / "source_net_v5_prepare"
                out.mkdir(parents=True, exist_ok=True)
                report_path = out / f"source_net_v5_overlength_guard_{split}_{dataset_name}.json"
                report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            except Exception:
                pass
            print(f"[source_net_v5_overlength_guard] {json.dumps(report, sort_keys=True)}")
            return _LazyOverlengthFilteredDataset(dataset, threshold, report_prefix=f"{split}:{dataset_name}")
        keep = []
        excluded = []
        kept_lengths = []
        excluded_lengths = []
        excluded_required_inputs = []
        excluded_cell_counts = []
        excluded_fanout_17_plus = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            seq = item[0] if isinstance(item, (tuple, list)) else item
            length = int(seq.numel()) if hasattr(seq, "numel") else int(len(seq))
            if length <= threshold:
                keep.append(idx)
                kept_lengths.append(length)
                continue
            excluded.append(idx)
            excluded_lengths.append(length)
            payload = getattr(self.tokenizer, "last_pointer_payload", {}) or {}
            obj = payload.get("object_table", {}) if isinstance(payload, dict) else {}
            excluded_required_inputs.append(int(obj.get("demand_count", 0) or 0))
            excluded_cell_counts.append(int(obj.get("cell_count", 0) or 0))
            fanout_hist = payload.get("fanout_hist", {}) if isinstance(payload, dict) else {}
            excluded_fanout_17_plus.append(int(fanout_hist.get("17_PLUS", 0) or 0))
        report = {
            "dataset_name": str(dataset_name),
            "split": str(split),
            "policy": self.overlength_policy,
            "threshold": threshold,
            "total": int(len(dataset)),
            "kept": int(len(keep)),
            "excluded": int(len(excluded)),
            "excluded_ratio": float(len(excluded) / max(1, len(dataset))),
            "max_kept_length": int(max(kept_lengths) if kept_lengths else 0),
            "max_excluded_length": int(max(excluded_lengths) if excluded_lengths else 0),
            "excluded_required_input_count": {
                "min": int(min(excluded_required_inputs) if excluded_required_inputs else 0),
                "max": int(max(excluded_required_inputs) if excluded_required_inputs else 0),
                "mean": float(sum(excluded_required_inputs) / max(1, len(excluded_required_inputs))),
            },
            "excluded_cell_count": {
                "min": int(min(excluded_cell_counts) if excluded_cell_counts else 0),
                "max": int(max(excluded_cell_counts) if excluded_cell_counts else 0),
                "mean": float(sum(excluded_cell_counts) / max(1, len(excluded_cell_counts))),
            },
            "excluded_fanout_17_plus_count": {
                "min": int(min(excluded_fanout_17_plus) if excluded_fanout_17_plus else 0),
                "max": int(max(excluded_fanout_17_plus) if excluded_fanout_17_plus else 0),
                "mean": float(sum(excluded_fanout_17_plus) / max(1, len(excluded_fanout_17_plus))),
            },
        }
        self._overlength_guard_reports[f"{split}:{dataset_name}"] = report
        try:
            out = Path(os.environ.get("ELDA_LOG_ROOT", "logs")) / "source_net_v5_prepare"
            out.mkdir(parents=True, exist_ok=True)
            report_path = out / f"source_net_v5_overlength_guard_{split}_{dataset_name}.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass
        print(f"[source_net_v5_overlength_guard] {json.dumps(report, sort_keys=True)}")
        if not keep:
            raise RuntimeError(f"source_net_v5 overlength guard excluded all samples for {split}:{dataset_name}")
        return _IndexSubsetDataset(dataset, keep)

    def dataloader(self, dataset, **kwargs):
        if self.tokenizer is None:
            return DataLoaderPyG(dataset, **kwargs)
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        if self._train_sampler is not None:
            return self.dataloader(
                self.train_dataset,
                shuffle=False,
                sampler=self._train_sampler,
                collate_fn=self.collate_fn,
                **self.kwargs
            )
        return self.dataloader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            **self.kwargs
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return self.dataloader(
            self.val_dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.kwargs
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return self.dataloader(
            self.test_dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.kwargs
        )

    def predict_dataloader(self) -> DataLoader:
        assert self.pred_dataset is not None
        return self.dataloader(
            self.pred_dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.kwargs
        )
