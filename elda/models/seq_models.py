import os
import os.path as osp
import ast
import json
import math
from collections import OrderedDict
from pathlib import Path
import torch
import hydra
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
from transformers import LogitsProcessor, LogitsProcessorList
from ..evaluation.metrics import get_dataset_metric
from ..evaluation.visualization import plot_gridspec_graphs, plot_smiles
from timeit import default_timer as timer

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")
from torch_geometric.data import Data


INTERFACE_BUCKET_TO_SPEC = {
    "size_bucket": "partition_size",
    "stub_bucket": "boundary_stub_count",
    "pin_bucket": "boundary_pin_count",
    "pin_deficit_bucket": "gate_pin_deficit_scaled",
    "underconnected_bucket": "underconnected_gate_fraction_scaled",
    "synthetic_input_bucket": "synthetic_input_proxy_ratio_scaled",
}


PARTITION_CONTRACT_AUX_FIELDS = {
    "partition_driver_stub_supply_norm",
    "partition_load_stub_supply_norm",
    "partition_bidir_stub_supply_norm",
    "partition_ambig_stub_ratio",
    "partition_boundary_role_entropy_norm",
    "partition_projection_repair_budget_bucket_norm",
    "partition_net_export_cleanliness_bucket_norm",
}

SKELETON_CONTRACT_AUX_FIELDS = {
    "skeleton_driver_stub_supply_norm",
    "skeleton_load_stub_supply_norm",
    "skeleton_bidir_stub_supply_norm",
    "skeleton_ambig_stub_ratio",
}

SKELETON_TYPED_EDGE_AUX_FIELDS = {
    "skeleton_node_driver_demand_mean_norm",
    "skeleton_node_load_demand_mean_norm",
    "skeleton_node_shared_net_out_mean_norm",
    "skeleton_node_shared_net_in_mean_norm",
    "skeleton_node_contract_risk_mean_norm",
    "skeleton_edge_fanout_mean_norm",
    "skeleton_edge_contract_risk_mean_norm",
    "skeleton_edge_shared_net_group_count_mean_norm",
    "skeleton_typed_edge_enriched",
}


def _load_label_to_cell_mapping(path):
    text = Path(path).read_text(encoding="utf-8")
    _, rhs = text.split('=', 1)
    mapping = ast.literal_eval(rhs.strip())
    return {int(label): str(cell) for cell, label in mapping.items()}


def _resolve_cell_mapping_path(cfg, explicit_path=None):
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate
    datamodule_root = getattr(getattr(cfg, "datamodule", None), "root", None)
    if datamodule_root:
        meta_path = Path(datamodule_root) / "meta.pt"
        if meta_path.exists():
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            cell_mapping = meta.get("cell_mapping")
            if cell_mapping:
                candidate = Path(cell_mapping)
                if candidate.exists():
                    return candidate
    return None


def _cfg_uses_dataset(cfg, dataset_name):
    dataset_names = getattr(getattr(cfg, "datamodule", None), "dataset_names", None)
    if isinstance(dataset_names, str):
        return dataset_names == dataset_name
    if isinstance(dataset_names, (list, tuple, set)):
        return dataset_name in dataset_names
    return False


def _resolve_dataset_meta_path(cfg, explicit_path):
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate
    datamodule_root = getattr(getattr(cfg, "datamodule", None), "root", None)
    if datamodule_root:
        candidate = Path(datamodule_root) / "meta.pt"
        if candidate.exists():
            return candidate
    return None


def _resolve_circuit_special_id(cfg, key: str, env_name: str, fallback: int) -> int:
    env_value = os.environ.get(env_name)
    if env_value not in (None, ""):
        try:
            return int(env_value)
        except ValueError:
            pass

    train_cfg = getattr(cfg, "train", None)
    explicit_meta = None
    if train_cfg is not None:
        explicit_meta = getattr(train_cfg, "skeleton_meta_path", None) or getattr(train_cfg, "partition_meta_path", None)
    meta_path = _resolve_dataset_meta_path(cfg, explicit_meta)
    if meta_path is not None:
        try:
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            if key in meta:
                return int(meta[key])
        except Exception:
            pass
    return int(fallback)


def _resolve_circuit_net_id(cfg) -> int:
    return _resolve_circuit_special_id(cfg, "net_id", "CIRCUIT_NET_ID", 96)


def _resolve_circuit_boundary_stub_id(cfg) -> int:
    return _resolve_circuit_special_id(cfg, "boundary_stub_id", "CIRCUIT_BOUNDARY_STUB_ID", 97)


def _interface_bucket_counts(meta):
    keys = list(meta.get("interface_bucket_keys", ["size_bucket", "stub_bucket", "pin_bucket"]))
    counts = []
    for key in keys:
        spec_name = INTERFACE_BUCKET_TO_SPEC.get(key)
        if spec_name is None:
            return []
        bounds = meta.get("buckets", {}).get(spec_name, {}).get("upper_bounds", [])
        if not bounds:
            return []
        counts.append(len(bounds))
    return counts


def _decode_bucket_tuple(class_id, bucket_counts):
    rem = int(class_id)
    decoded = [0] * len(bucket_counts)
    for idx in range(len(bucket_counts) - 1, -1, -1):
        count = int(bucket_counts[idx])
        decoded[idx] = rem % count
        rem //= count
    return decoded


def _normalized_bucket_value(bucket_idx, bucket_count):
    if bucket_count <= 1:
        return 0.0
    return float(bucket_idx) / float(bucket_count - 1)


def _build_skeleton_token_weight_overrides(tokenizer, cfg):
    train_cfg = getattr(cfg, "train", None)
    if train_cfg is None or not bool(getattr(train_cfg, "skeleton_size_reweighting", False)):
        return {}
    if not getattr(tokenizer, "labeled_graph", False) or not hasattr(tokenizer, "node_idx_offset"):
        return {}
    if not _cfg_uses_dataset(cfg, "CIRCUIT_SKELETON"):
        return {}

    meta_path = _resolve_dataset_meta_path(cfg, getattr(train_cfg, "skeleton_meta_path", None))
    if meta_path is None:
        return {}
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    bucket_counts = _interface_bucket_counts(meta)
    if not bucket_counts:
        return {}

    graphs_dir = meta_path.parent / "skeletons" / "graphs"
    if not graphs_dir.exists():
        return {}

    counts = {}
    for graph_path in sorted(graphs_dir.glob("*.pt")):
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
        for label in data.x.reshape(-1).tolist():
            label = int(label)
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return {}

    positive_counts = sorted(v for v in counts.values() if v > 0)
    reference = float(positive_counts[len(positive_counts) // 2])
    alpha = float(getattr(train_cfg, "skeleton_size_reweight_alpha", 0.5))
    scale = float(getattr(train_cfg, "skeleton_size_reweight_scale", 0.75))
    max_weight = float(getattr(train_cfg, "skeleton_size_reweight_max", 2.5))
    large_bucket_min = int(getattr(train_cfg, "skeleton_large_bucket_min", 2))
    small_bucket_min_weight = float(getattr(train_cfg, "skeleton_small_bucket_min_weight", 0.85))
    downweight_small = bool(getattr(train_cfg, "skeleton_downweight_small", True))

    interface_bucket_keys = list(meta.get("interface_bucket_keys", ["size_bucket", "stub_bucket", "pin_bucket"]))
    bucket_count_map = {key: int(count) for key, count in zip(interface_bucket_keys, bucket_counts)}

    size_scale_cfg = float(getattr(train_cfg, "skeleton_size_reweight_scale", 0.75))
    stub_scale_cfg = float(getattr(train_cfg, "skeleton_stub_reweight_scale", 0.0))
    pin_scale_cfg = float(getattr(train_cfg, "skeleton_pin_reweight_scale", 0.0))
    quality_scale_cfg = float(getattr(train_cfg, "skeleton_quality_reweight_scale", 0.0))
    vocab_size = len(tokenizer)
    overrides = {}
    for label, count in counts.items():
        token_id = tokenizer.node_idx_offset + int(label)
        if token_id < 0 or token_id >= vocab_size:
            continue
        decoded = _decode_bucket_tuple(label, bucket_counts)
        decoded_map = {key: int(val) for key, val in zip(interface_bucket_keys, decoded)}

        size_bucket = decoded_map.get("size_bucket", 0)
        size_scale = _normalized_bucket_value(size_bucket, bucket_count_map.get("size_bucket", 1))
        stub_scale = _normalized_bucket_value(
            decoded_map.get("stub_bucket", 0), bucket_count_map.get("stub_bucket", 1)
        )
        pin_scale = _normalized_bucket_value(
            decoded_map.get("pin_bucket", 0), bucket_count_map.get("pin_bucket", 1)
        )
        pin_deficit_scale = _normalized_bucket_value(
            decoded_map.get("pin_deficit_bucket", 0), bucket_count_map.get("pin_deficit_bucket", 1)
        )
        underconnected_scale = _normalized_bucket_value(
            decoded_map.get("underconnected_bucket", 0), bucket_count_map.get("underconnected_bucket", 1)
        )
        synthetic_scale = _normalized_bucket_value(
            decoded_map.get("synthetic_input_bucket", 0), bucket_count_map.get("synthetic_input_bucket", 1)
        )
        quality_scale = 1.0 - ((pin_deficit_scale + underconnected_scale + synthetic_scale) / 3.0)
        rarity_boost = 1.0
        if count < reference:
            rarity_boost = max(1.0, (reference / float(count)) ** alpha)

        if size_bucket >= large_bucket_min:
            weight = (
                1.0
                + size_scale_cfg * size_scale
                + stub_scale_cfg * stub_scale
                + pin_scale_cfg * pin_scale
                + quality_scale_cfg * quality_scale
            ) * rarity_boost
        elif downweight_small and size_bucket == 0:
            weight = min(small_bucket_min_weight, rarity_boost)
        else:
            weight = max(
                1.0,
                1.0
                + 0.15 * size_scale
                + 0.10 * stub_scale_cfg * stub_scale
                + 0.10 * pin_scale_cfg * pin_scale
                + 0.10 * quality_scale_cfg * quality_scale,
            ) * min(rarity_boost, 1.25)

        if downweight_small and size_bucket == 0:
            weight = max(0.5, min(small_bucket_min_weight, weight))
        else:
            weight = max(1.0, min(max_weight, weight))
        if abs(weight - 1.0) > 1e-6:
            overrides[token_id] = float(weight)
    return overrides


def _build_gate_type_token_weight_overrides(tokenizer, cfg):
    train_cfg = getattr(cfg, "train", None)
    if train_cfg is None or not bool(getattr(train_cfg, "gate_type_reweighting", False)):
        return {}
    token_offset = None
    if getattr(tokenizer, "labeled_graph", False) and hasattr(tokenizer, "node_idx_offset"):
        token_offset = int(tokenizer.node_idx_offset)
    elif getattr(tokenizer, "is_pin_slot_tokenizer", False) and hasattr(tokenizer, "node_type_offset"):
        token_offset = int(tokenizer.node_type_offset)
    if token_offset is None:
        return {}

    profile_path = getattr(train_cfg, "gate_type_profile_path", None)
    if not profile_path:
        datamodule_root = getattr(getattr(cfg, "datamodule", None), "root", None)
        if datamodule_root:
            candidate = Path(datamodule_root) / 'gate_type_profile.json'
            if candidate.exists():
                profile_path = str(candidate)
    mapping_path = _resolve_cell_mapping_path(cfg, getattr(train_cfg, "cell_mapping_path", None))
    if not profile_path or mapping_path is None:
        return {}

    profile_file = Path(profile_path)
    mapping_file = Path(mapping_path)
    if not profile_file.exists() or not mapping_file.exists():
        return {}

    profile = json.loads(profile_file.read_text(encoding='utf-8'))
    counts = {str(k): int(v) for k, v in profile.get('counts', {}).items()}
    medium_types = set(profile.get('medium_types', []))
    if not counts or not medium_types:
        return {}

    medium_counts = sorted(counts[cell] for cell in medium_types if counts.get(cell, 0) > 0)
    if not medium_counts:
        return {}
    reference = float(medium_counts[len(medium_counts) // 2])
    alpha = float(getattr(train_cfg, 'gate_type_reweight_alpha', 0.5))
    max_weight = float(getattr(train_cfg, 'gate_type_reweight_max', 3.0))
    head_min_weight = float(getattr(train_cfg, 'gate_type_head_min_weight', 0.75))
    downweight_x1 = bool(getattr(train_cfg, 'gate_type_downweight_x1', True))
    include_all_types = bool(getattr(train_cfg, 'gate_type_include_all_types', False))
    include_tail_types = bool(getattr(train_cfg, 'gate_type_include_tail_types', True))
    quality_scale = float(getattr(train_cfg, 'gate_type_quality_scale', 0.0))
    diversity_scale = float(getattr(train_cfg, 'gate_type_diversity_scale', 0.0))

    label_to_cell = _load_label_to_cell_mapping(mapping_file)
    datamodule_root = getattr(getattr(cfg, "datamodule", None), "root", None)
    cell_quality_stats = {}
    if datamodule_root and _cfg_uses_dataset(cfg, "CIRCUIT_PARTITION"):
        graphs_dir = Path(datamodule_root) / "partitions" / "graphs"
        meta_path = Path(datamodule_root) / "meta.pt"
        if graphs_dir.exists() and meta_path.exists():
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            net_id = int(meta.get("net_id", 96))
            boundary_stub_id = int(meta.get("boundary_stub_id", 97))
            accum = {}
            for graph_path in sorted(graphs_dir.glob("*.pt")):
                data = torch.load(graph_path, map_location="cpu", weights_only=False)
                x = data.x.reshape(-1).to(torch.long)
                gate_mask = (x != net_id) & (x != boundary_stub_id)
                gate_labels = x[gate_mask]
                if gate_labels.numel() == 0:
                    continue
                clean_score = 1.0 - (
                    min(1.0, float(getattr(data, "gate_pin_deficit", 0.0)))
                    + min(1.0, float(getattr(data, "underconnected_gate_fraction", 0.0)))
                    + min(1.0, float(getattr(data, "synthetic_input_proxy_ratio", 0.0)))
                ) / 3.0
                diversity_score = min(1.0, float(torch.unique(gate_labels).numel()) / 12.0)
                for label in gate_labels.tolist():
                    label = int(label)
                    stats = accum.setdefault(label, {"count": 0, "clean_sum": 0.0, "diversity_sum": 0.0})
                    stats["count"] += 1
                    stats["clean_sum"] += clean_score
                    stats["diversity_sum"] += diversity_score
            for label, stats in accum.items():
                if stats["count"] <= 0:
                    continue
                cell_quality_stats[int(label)] = {
                    "avg_clean": float(stats["clean_sum"]) / float(stats["count"]),
                    "avg_diversity": float(stats["diversity_sum"]) / float(stats["count"]),
                }
    vocab_size = len(tokenizer)
    overrides = {}
    for label, cell in label_to_cell.items():
        token_id = token_offset + int(label)
        if token_id < 0 or token_id >= vocab_size:
            continue
        count = counts.get(cell)
        if not count:
            continue
        is_medium = cell in medium_types
        is_tail = count < reference
        if not include_all_types and not is_medium and not (include_tail_types and is_tail):
            continue
        ratio = (reference / float(count)) ** alpha
        quality = cell_quality_stats.get(int(label), {})
        quality_boost = 1.0 + quality_scale * max(0.0, float(quality.get("avg_clean", 0.5)) - 0.5) * 2.0
        diversity_boost = 1.0 + diversity_scale * float(quality.get("avg_diversity", 0.0))
        if count < reference:
            weight = min(max_weight, max(1.0, ratio * quality_boost * diversity_boost))
        elif downweight_x1 and cell.endswith('_X1'):
            weight = max(head_min_weight, min(1.0, ratio))
        else:
            weight = min(max_weight, max(1.0, quality_boost * diversity_boost))
        if abs(weight - 1.0) > 1e-6:
            overrides[token_id] = float(weight)
    return overrides


class SequenceModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.tokenizer = None
        self.instantiate_datamodule()
        self.instantiate_model()
        self.instantiate_aux_heads()
        self.instantiate_loss()
        self.instantiate_metrics()
        self.save_hyperparameters()

    def update_cfg(self, cfg):
        self.cfg = cfg

        self.instantiate_datamodule()
        self.instantiate_aux_heads()
        self.instantiate_loss()
        self.instantiate_metrics()
        self.save_hyperparameters()

    def save_pretrained(self, save_directory, **kwargs):
        self.model.model.save_pretrained(save_directory, **kwargs)

    def instantiate_datamodule(self):
        self._datamodule = hydra.utils.instantiate(self.cfg.datamodule, tokenizer=self.tokenizer)
        self._datamodule.prepare_data()
        self._datamodule.setup('fit')
        if self.tokenizer is None:
            self.tokenizer = self._datamodule.tokenizer

        # V3: Load pin specs for Grammar Masking
        if self.tokenizer is not None and getattr(self.tokenizer, "labeled_graph", False):
            mapping_path = _resolve_cell_mapping_path(self.cfg)
            if mapping_path:
                try:
                    from circuit_kahypar.schema import load_label_to_cell_mapping, infer_cell_spec
                    label_to_cell = load_label_to_cell_mapping(mapping_path)
                    vocab_size = len(self.tokenizer)
                    pin_specs = torch.zeros((vocab_size, 2), dtype=torch.long)
                    # We need to know the node_idx_offset to map labels to tokens
                    if hasattr(self.tokenizer, "node_idx_offset"):
                        for label, cell in label_to_cell.items():
                            spec = infer_cell_spec(cell)
                            token_id = self.tokenizer.node_idx_offset + int(label)
                            if 0 <= token_id < vocab_size:
                                pin_specs[token_id, 0] = spec.get("inputs", 0)
                                pin_specs[token_id, 1] = spec.get("outputs", 0)
                        self.tokenizer.pin_specs = pin_specs
                        print(f"V3: Loaded pin specs for {len(label_to_cell)} cells.")
                except Exception as e:
                    print(f"Warning: Failed to load pin specs for V3 grammar masking: {e}")

    def instantiate_model(self):
        self.model = hydra.utils.instantiate(self.cfg.model, tokenizer=self.tokenizer)

    def _configured_aux_fields(self):
        fields = getattr(getattr(self.cfg, "datamodule", None), "aux_fields", None)
        if fields is None:
            return []
        return [str(field) for field in fields]

    def _aux_field_weight(self, key):
        train_cfg = getattr(self.cfg, "train", None)
        if train_cfg is None:
            return 0.0
        if key == "partition_projection_proxy":
            return float(getattr(train_cfg, "partition_projection_proxy_aux_weight", 0.0))
        if key in {
            "partition_gate_pin_deficit",
            "partition_underconnected_gate_fraction",
            "partition_synthetic_input_proxy_ratio",
        }:
            return float(getattr(train_cfg, "partition_projection_components_aux_weight", 0.0))
        if key in PARTITION_CONTRACT_AUX_FIELDS:
            return float(getattr(train_cfg, "partition_contract_aux_weight", 0.0))
        if key == "skeleton_component_count_norm":
            return float(getattr(train_cfg, "skeleton_component_aux_weight", 0.0))
        if key in {"skeleton_largest_component_ratio", "skeleton_avg_degree_norm"}:
            return float(getattr(train_cfg, "skeleton_connectivity_aux_weight", 0.0))
        if key in SKELETON_CONTRACT_AUX_FIELDS:
            return float(getattr(train_cfg, "skeleton_contract_aux_weight", 0.0))
        if key in SKELETON_TYPED_EDGE_AUX_FIELDS:
            return float(getattr(train_cfg, "skeleton_typed_edge_aux_weight", 0.0))
        return 0.0

    def instantiate_aux_heads(self):
        self.aux_heads = nn.ModuleDict()
        hidden_size = getattr(getattr(self.model, "model", None), "config", None)
        if hidden_size is None:
            return
        hidden_size = getattr(self.model.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.model.config, "n_embd", None)
        if hidden_size is None:
            return
        train_cfg = getattr(self.cfg, "train", None)
        if train_cfg is None:
            return

        if _cfg_uses_dataset(self.cfg, "CIRCUIT_PARTITION"):
            if float(getattr(train_cfg, "partition_projection_proxy_aux_weight", 0.0)) > 0.0:
                self.aux_heads["partition_projection_proxy"] = nn.Linear(hidden_size, 1)
            if float(getattr(train_cfg, "partition_projection_components_aux_weight", 0.0)) > 0.0:
                self.aux_heads["partition_gate_pin_deficit"] = nn.Linear(hidden_size, 1)
                self.aux_heads["partition_underconnected_gate_fraction"] = nn.Linear(hidden_size, 1)
                self.aux_heads["partition_synthetic_input_proxy_ratio"] = nn.Linear(hidden_size, 1)

        if _cfg_uses_dataset(self.cfg, "CIRCUIT_SKELETON"):
            if float(getattr(train_cfg, "skeleton_component_aux_weight", 0.0)) > 0.0:
                self.aux_heads["skeleton_component_count_norm"] = nn.Linear(hidden_size, 1)
            if float(getattr(train_cfg, "skeleton_connectivity_aux_weight", 0.0)) > 0.0:
                self.aux_heads["skeleton_largest_component_ratio"] = nn.Linear(hidden_size, 1)
                self.aux_heads["skeleton_avg_degree_norm"] = nn.Linear(hidden_size, 1)

        for field in self._configured_aux_fields():
            if field in self.aux_heads:
                continue
            if self._aux_field_weight(field) > 0.0:
                self.aux_heads[field] = nn.Linear(hidden_size, 1)

    def instantiate_loss(self):
        weight = None
        self.partition_gate_token_ids = None
        self.partition_gate_subset_index = None
        train_cfg = getattr(self.cfg, "train", None)
        if train_cfg is not None and getattr(self.tokenizer, "labeled_graph", False):
            vocab_size = len(self.tokenizer)
            weight = torch.ones(vocab_size)
            has_custom_weight = False

            net_weight = float(getattr(train_cfg, "net_token_weight", 1.0))
            if hasattr(self.tokenizer, "node_idx_offset") and net_weight != 1.0:
                net_id = _resolve_circuit_net_id(self.cfg)
                net_token_id = self.tokenizer.node_idx_offset + net_id
                if 0 <= net_token_id < vocab_size:
                    weight[net_token_id] = net_weight
                    has_custom_weight = True

            gate_type_overrides = _build_gate_type_token_weight_overrides(self.tokenizer, self.cfg)
            for token_id, token_weight in gate_type_overrides.items():
                weight[token_id] = float(token_weight)
                has_custom_weight = True
            if gate_type_overrides:
                values = list(gate_type_overrides.values())
                print(
                    f"gate-type token reweighting active: {len(values)} tokens, "
                    f"min={min(values):.3f}, max={max(values):.3f}"
                )

            skeleton_overrides = _build_skeleton_token_weight_overrides(self.tokenizer, self.cfg)
            for token_id, token_weight in skeleton_overrides.items():
                weight[token_id] = float(token_weight)
                has_custom_weight = True
            if skeleton_overrides:
                values = list(skeleton_overrides.values())
                print(
                    f"skeleton-size token reweighting active: {len(values)} tokens, "
                    f"min={min(values):.3f}, max={max(values):.3f}"
                )

            if not has_custom_weight:
                weight = None
            hist_aux_weight = float(getattr(train_cfg, "partition_hist_aux_weight", 0.0))
            if hist_aux_weight > 0.0 and _cfg_uses_dataset(self.cfg, "CIRCUIT_PARTITION"):
                mapping_path = _resolve_cell_mapping_path(self.cfg, getattr(train_cfg, "cell_mapping_path", None))
                if mapping_path is not None and hasattr(self.tokenizer, "node_idx_offset"):
                    label_to_cell = _load_label_to_cell_mapping(mapping_path)
                    gate_token_ids = sorted(
                        self.tokenizer.node_idx_offset + int(label)
                        for label in label_to_cell.keys()
                        if 0 <= self.tokenizer.node_idx_offset + int(label) < vocab_size
                    )
                    if gate_token_ids:
                        subset_index = torch.full((vocab_size,), -1, dtype=torch.long)
                        subset_index[torch.tensor(gate_token_ids, dtype=torch.long)] = torch.arange(
                            len(gate_token_ids), dtype=torch.long
                        )
                        self.partition_gate_token_ids = torch.tensor(gate_token_ids, dtype=torch.long)
                        self.partition_gate_subset_index = subset_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad, weight=weight)

    def instantiate_metrics(self):
        if bool(getattr(self.cfg.sampling, "disable_metric_init", False)):
            self.sampling_metric = None
            return
        dataset = hydra.utils.instantiate(
            self.cfg.datamodule, init_tokenizer=False
        )
        dataset.setup('fit')
        dataset.setup('test')
        self.sampling_metric = get_dataset_metric(
            self.cfg.datamodule.dataset_names,
            dataset,
            num_ref_graphs=self.cfg.sampling.num_samples
        )

    def _aux_head_forward(self, key, pooled, batch_aux):
        head = self.aux_heads[key]
        head_dtype = next(head.parameters()).dtype
        pred = head(pooled.to(dtype=head_dtype)).squeeze(-1)
        target = batch_aux[key].to(device=pred.device, dtype=pred.dtype)
        return pred, target

    def _loss_reweight_cfg(self):
        for root_name in ("training", "train"):
            root = getattr(self.cfg, root_name, None)
            if root is None:
                continue
            lrw = getattr(root, "loss_reweight", None)
            if lrw is not None and bool(getattr(lrw, "enabled", False)):
                return lrw
        return None

    def _source_net_v4_netsection_only_loss(self, logits_seq, target_seq, phase: str):
        train_cfg = getattr(self.cfg, "train", None)
        if train_cfg is None or not bool(getattr(train_cfg, "source_net_v4_netsection_only", False)):
            return None
        net_start = getattr(self.tokenizer, "net_section_start", None)
        if net_start is None:
            return None
        ce = F.cross_entropy(
            logits_seq.reshape(-1, logits_seq.shape[-1]),
            target_seq.reshape(-1),
            ignore_index=int(self.tokenizer.pad),
            weight=self.loss_fn.weight,
            reduction="none",
        ).reshape_as(target_seq)
        valid = target_seq != int(self.tokenizer.pad)
        seen_net_start = torch.cumsum((target_seq == int(net_start)).to(dtype=torch.long), dim=1) > 0
        target_mask = seen_net_start & (target_seq != int(net_start)) & valid
        weights = torch.ones_like(ce)
        load_tok = getattr(self.tokenizer, "load", None)
        driver_tok = getattr(self.tokenizer, "driver", None)
        if target_seq.shape[1] > 1:
            prev1 = torch.full_like(target_seq, -1)
            prev1[:, 1:] = target_seq[:, :-1]
            weights[(prev1 == int(load_tok)) & target_mask] *= float(getattr(train_cfg, "load_cell_id_weight", 1.0)) if load_tok is not None else 1.0
            weights[(prev1 == int(driver_tok)) & target_mask] *= float(getattr(train_cfg, "driver_cell_id_weight", 1.0)) if driver_tok is not None else 1.0
        if target_seq.shape[1] > 2:
            prev2 = torch.full_like(target_seq, -1)
            prev2[:, 2:] = target_seq[:, :-2]
            weights[(prev2 == int(load_tok)) & target_mask] *= float(getattr(train_cfg, "load_pin_weight", 1.0)) if load_tok is not None else 1.0
            weights[(prev2 == int(driver_tok)) & target_mask] *= float(getattr(train_cfg, "driver_pin_weight", 1.0)) if driver_tok is not None else 1.0
        if load_tok is not None:
            weights[(target_seq == int(load_tok)) & target_mask] *= float(getattr(train_cfg, "load_token_weight", 1.0))
        if driver_tok is not None:
            weights[(target_seq == int(driver_tok)) & target_mask] *= float(getattr(train_cfg, "driver_token_weight", 1.0))
        fanout_tokens = [
            getattr(self.tokenizer, name, None)
            for name in ("fanout_bucket_0", "fanout_bucket_1", "fanout_bucket_2_4", "fanout_bucket_5_8", "fanout_bucket_9_16", "fanout_bucket_17_plus")
        ]
        fanout_ids = [int(tok) for tok in fanout_tokens if tok is not None]
        if fanout_ids:
            fanout_mask = torch.zeros_like(target_mask)
            for tid in fanout_ids:
                fanout_mask |= target_seq == int(tid)
            weights[fanout_mask & target_mask] *= float(getattr(train_cfg, "fanout_bucket_weight", 1.0))
        weighted = ce * weights * target_mask.to(dtype=ce.dtype)
        denom = (weights * target_mask.to(dtype=weights.dtype)).sum().clamp_min(1.0)
        loss = weighted.sum() / denom
        if phase == "train":
            with torch.no_grad():
                self.log(f"{phase}/netsection_target_token_ratio", target_mask.float().mean(), on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{phase}/netsection_loss_weight_mean", (weights * target_mask.to(dtype=weights.dtype)).sum() / target_mask.to(dtype=weights.dtype).sum().clamp_min(1.0), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def _sequence_ce_loss(self, logits, target, phase: str):
        lrw = self._loss_reweight_cfg()
        if lrw is None:
            return self.loss_fn(logits, target)

        ce = F.cross_entropy(
            logits,
            target,
            ignore_index=int(self.tokenizer.pad),
            weight=self.loss_fn.weight,
            reduction="none",
        )
        weights = torch.ones_like(ce)
        valid = target != int(self.tokenizer.pad)

        def apply_token(tok, value):
            if tok is None:
                return
            try:
                tid = int(tok)
                if 0 <= tid < logits.shape[-1]:
                    weights[target == tid] *= float(value)
            except Exception:
                return

        apply_token(getattr(self.tokenizer, "ep_driver_stub", None), getattr(lrw, "driver_stub_token_weight", 1.0))
        apply_token(getattr(self.tokenizer, "ep_optional_driver_stub", None), getattr(lrw, "optional_driver_stub_token_weight", 1.0))
        apply_token(getattr(self.tokenizer, "ep_load_stub", None), getattr(lrw, "load_stub_token_weight", 1.0))
        apply_token(getattr(self.tokenizer, "ep_net", None), getattr(lrw, "net_token_weight", 1.0))
        apply_token(getattr(self.tokenizer, "pin_in", None), getattr(lrw, "pin_in_token_weight", 1.0))
        apply_token(getattr(self.tokenizer, "pin_opt_out", None), getattr(lrw, "pin_opt_out_token_weight", 1.0))
        apply_token(getattr(self.tokenizer, "pin_skip", None), getattr(lrw, "pin_skip_token_weight", 1.0))
        apply_token(getattr(self.tokenizer, "ep_input_pool", None), getattr(lrw, "input_pool_token_weight", 1.0))
        apply_token(getattr(self.tokenizer, "ep_private_out", None), getattr(lrw, "private_output_token_weight", 1.0))

        token_weights = getattr(lrw, "token_weights", None)
        if token_weights is not None:
            for name, value in dict(token_weights).items():
                attr = str(name).lower()
                if attr.startswith("ep_") or attr.startswith("pin_"):
                    apply_token(getattr(self.tokenizer, attr, None), value)

        pin_offset = getattr(self.tokenizer, "pin_offset", None)
        pin_names = getattr(self.tokenizer, "pin_names", None)
        if pin_offset is not None and isinstance(pin_names, (list, tuple)):
            name_to_id = {str(name): int(idx) for idx, name in enumerate(pin_names)}

            def apply_pin_names(names, value):
                if names is None:
                    return
                for pin_name in list(names):
                    pin_id = name_to_id.get(str(pin_name))
                    if pin_id is not None:
                        apply_token(int(pin_offset) + int(pin_id), value)

            apply_pin_names(getattr(lrw, "required_input_pin_names", None), getattr(lrw, "required_input_pin_token_weight", 1.0))
            apply_pin_names(getattr(lrw, "optional_output_pin_names", None), getattr(lrw, "optional_output_pin_token_weight", 1.0))
            pin_name_weights = getattr(lrw, "pin_name_token_weights", None)
            if pin_name_weights is not None:
                for pin_name, value in dict(pin_name_weights).items():
                    apply_pin_names([pin_name], value)

        idx_weight = float(getattr(lrw, "driver_stub_idx_weight", 1.0))
        off = getattr(self.tokenizer, "driver_stub_idx_offset", None)
        max_nodes = getattr(self.tokenizer, "max_num_nodes", None)
        if off is not None and max_nodes is not None and idx_weight != 1.0:
            lo = int(off)
            hi = int(off) + int(max_nodes)
            weights[(target >= lo) & (target < hi)] *= idx_weight

        weighted = ce * weights * valid.to(dtype=ce.dtype)
        denom = valid.to(dtype=ce.dtype).sum().clamp_min(1.0)
        loss = weighted.sum() / denom
        if phase == "train":
            with torch.no_grad():
                self.log(
                    f"{phase}/loss_reweight_mean",
                    (weights * valid.to(dtype=weights.dtype)).sum() / denom,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        return loss

    def _set_adaptive_gradient_checkpointing(self, batch, phase):
        """Toggle HF activation checkpointing only for configured long batches."""
        train_cfg = getattr(self.cfg, "train", None)
        enabled = bool(
            train_cfg is not None
            and getattr(train_cfg, "adaptive_gradient_checkpointing", False)
        )
        core_model = getattr(self.model, "model", None)
        if core_model is None:
            return

        use_checkpointing = False
        sequence_length = 0
        if torch.is_tensor(batch) and batch.ndim >= 2:
            token_mask = batch.ne(int(self.tokenizer.pad))
            sequence_length = int(token_mask.sum(dim=1).max().detach().cpu().item())
        if enabled and phase == "train":
            threshold = int(
                getattr(train_cfg, "adaptive_gradient_checkpointing_threshold", 16384)
            )
            if threshold <= 0:
                raise ValueError(
                    "adaptive_gradient_checkpointing_threshold must be positive"
                )
            use_checkpointing = sequence_length >= threshold

        current = bool(getattr(core_model, "is_gradient_checkpointing", False))
        if use_checkpointing != current:
            if use_checkpointing:
                core_model.gradient_checkpointing_enable()
            else:
                core_model.gradient_checkpointing_disable()
        if enabled and hasattr(core_model, "config"):
            core_model.config.use_cache = False

        if phase == "train" and enabled:
            self.log(
                "train/sequence_length",
                float(sequence_length),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/adaptive_gc_enabled",
                float(use_checkpointing),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def shared_step(self, batch, batch_idx, phase='train'):
        batch_aux = None
        if isinstance(batch, (tuple, list)):
            batch, batch_aux = batch
        self._set_adaptive_gradient_checkpointing(batch, phase)
        x, y = batch[:, :-1], batch[:, 1:]
        need_hidden = bool(batch_aux) and len(self.aux_heads) > 0
        model_out = self.model(x, return_hidden_states=need_hidden)
        if need_hidden:
            y_pred, hidden_states = model_out
        else:
            y_pred = model_out
            hidden_states = None
        y_pred_seq = y_pred
        if (
            phase == "train"
            and getattr(self.cfg.train, "grammar_mask", False)
            and getattr(self.tokenizer, "labeled_graph", False)
        ):
            # Keep an unmasked copy so we can safely recover rows where the mask
            # would otherwise remove the gold token and force inf loss.
            raw_y_pred = y_pred
            net_id = _resolve_circuit_net_id(self.cfg)
            boundary_stub_id = _resolve_circuit_boundary_stub_id(self.cfg)
            y_pred = apply_labeled_grammar_mask(y_pred, x, self.tokenizer, net_id=net_id, boundary_stub_id=boundary_stub_id)
            flat_masked = y_pred.view(-1, y_pred.shape[-1])
            flat_raw = raw_y_pred.view(-1, raw_y_pred.shape[-1])
            flat_y = y.reshape(-1)
            valid = flat_y != self.tokenizer.pad
            row_idx = torch.arange(flat_masked.shape[0], device=flat_masked.device)
            gold_masked = (flat_masked[row_idx, flat_y] == float("-inf")) & valid
            if gold_masked.any():
                flat_masked[gold_masked] = flat_raw[gold_masked]
                self.log(
                    f"{phase}/mask_invalid_ratio",
                    gold_masked.float().mean(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        y_pred = y_pred.view(-1, y_pred.shape[-1])
        y_flat = y.reshape(-1)
        netsection_loss = self._source_net_v4_netsection_only_loss(y_pred_seq, y, phase)
        loss = netsection_loss if netsection_loss is not None else self._sequence_ce_loss(y_pred, y_flat, phase)
        if (
            phase == "train"
            and self.partition_gate_token_ids is not None
            and self.partition_gate_subset_index is not None
            and float(getattr(self.cfg.train, "partition_hist_aux_weight", 0.0)) > 0.0
        ):
            gate_token_ids = self.partition_gate_token_ids.to(y.device)
            subset_index = self.partition_gate_subset_index.to(y.device)
            target_subset = subset_index[y]
            gate_mask = (y != self.tokenizer.pad) & (target_subset >= 0)
            if gate_mask.any():
                gate_logits = y_pred_seq.index_select(-1, gate_token_ids)
                pred_gate_probs = torch.exp(gate_logits - torch.logsumexp(y_pred_seq, dim=-1, keepdim=True))
                pred_hist = (pred_gate_probs * gate_mask.unsqueeze(-1).float()).sum(dim=1)
                target_hist = torch.zeros_like(pred_hist)
                for sample_idx in range(y.shape[0]):
                    sample_mask = gate_mask[sample_idx]
                    if not sample_mask.any():
                        continue
                    sample_targets = target_subset[sample_idx][sample_mask]
                    target_hist[sample_idx].scatter_add_(
                        0,
                        sample_targets,
                        torch.ones_like(sample_targets, dtype=pred_hist.dtype),
                    )
                pred_hist = pred_hist / pred_hist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                target_hist = target_hist / target_hist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                hist_loss = F.smooth_l1_loss(pred_hist, target_hist)
                hist_weight = float(getattr(self.cfg.train, "partition_hist_aux_weight", 0.0))
                loss = loss + hist_weight * hist_loss
                self.log(f"{phase}/hist_aux_loss", hist_loss, on_step=False, on_epoch=True, sync_dist=True)
        if batch_aux and hidden_states is not None and len(self.aux_heads) > 0:
            token_mask = (x != self.tokenizer.pad).to(dtype=hidden_states.dtype)
            pooled = (hidden_states * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

            train_cfg = getattr(self.cfg, "train", None)
            handled_aux_fields = set()
            if train_cfg is not None:
                if "partition_projection_proxy" in self.aux_heads and "partition_projection_proxy" in batch_aux:
                    pred, target = self._aux_head_forward("partition_projection_proxy", pooled, batch_aux)
                    aux_loss = F.smooth_l1_loss(pred, target)
                    weight = float(getattr(train_cfg, "partition_projection_proxy_aux_weight", 0.0))
                    loss = loss + weight * aux_loss
                    self.log(f"{phase}/projection_proxy_aux_loss", aux_loss, on_step=False, on_epoch=True, sync_dist=True)
                    handled_aux_fields.add("partition_projection_proxy")

                component_weight = float(getattr(train_cfg, "partition_projection_components_aux_weight", 0.0))
                if component_weight > 0.0:
                    part_components = [
                        ("partition_gate_pin_deficit", "projection_gate_pin_deficit_aux_loss"),
                        ("partition_underconnected_gate_fraction", "projection_underconnected_aux_loss"),
                        ("partition_synthetic_input_proxy_ratio", "projection_synth_proxy_aux_loss"),
                    ]
                    for key, log_name in part_components:
                        if key not in self.aux_heads or key not in batch_aux:
                            continue
                        pred, target = self._aux_head_forward(key, pooled, batch_aux)
                        aux_loss = F.smooth_l1_loss(pred, target)
                        loss = loss + component_weight * aux_loss
                        self.log(f"{phase}/{log_name}", aux_loss, on_step=False, on_epoch=True, sync_dist=True)
                        handled_aux_fields.add(key)

                if "skeleton_component_count_norm" in self.aux_heads and "skeleton_component_count_norm" in batch_aux:
                    pred, target = self._aux_head_forward("skeleton_component_count_norm", pooled, batch_aux)
                    aux_loss = F.smooth_l1_loss(pred, target)
                    weight = float(getattr(train_cfg, "skeleton_component_aux_weight", 0.0))
                    loss = loss + weight * aux_loss
                    self.log(f"{phase}/skeleton_component_aux_loss", aux_loss, on_step=False, on_epoch=True, sync_dist=True)
                    handled_aux_fields.add("skeleton_component_count_norm")

                connectivity_weight = float(getattr(train_cfg, "skeleton_connectivity_aux_weight", 0.0))
                if connectivity_weight > 0.0:
                    skeleton_components = [
                        ("skeleton_largest_component_ratio", "skeleton_largest_component_aux_loss"),
                        ("skeleton_avg_degree_norm", "skeleton_avg_degree_aux_loss"),
                    ]
                    for key, log_name in skeleton_components:
                        if key not in self.aux_heads or key not in batch_aux:
                            continue
                        pred, target = self._aux_head_forward(key, pooled, batch_aux)
                        aux_loss = F.smooth_l1_loss(pred, target)
                        loss = loss + connectivity_weight * aux_loss
                        self.log(f"{phase}/{log_name}", aux_loss, on_step=False, on_epoch=True, sync_dist=True)
                        handled_aux_fields.add(key)

                for key in sorted(self.aux_heads.keys()):
                    if key in handled_aux_fields or key not in batch_aux:
                        continue
                    weight = self._aux_field_weight(key)
                    if weight <= 0.0:
                        continue
                    pred, target = self._aux_head_forward(key, pooled, batch_aux)
                    aux_loss = F.smooth_l1_loss(pred, target)
                    loss = loss + float(weight) * aux_loss
                    self.log(f"{phase}/{key}_aux_loss", aux_loss, on_step=False, on_epoch=True, sync_dist=True)
        if phase == "train":
            self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        if (
            phase == "train"
            and getattr(self.tokenizer, "labeled_graph", False)
            and hasattr(self.tokenizer, "node_idx_offset")
        ):
            net_id = _resolve_circuit_net_id(self.cfg)
            net_token_id = self.tokenizer.node_idx_offset + net_id
            if 0 <= net_token_id < len(self.tokenizer):
                net_ratio = (y == net_token_id).float().mean()
                self.log(
                    f"{phase}/net_token_ratio",
                    net_ratio,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase='train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase='val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase='test')
        return loss

    def on_validation_epoch_end(self):
        if self.sampling_metric is None:
            return
        self.evaluate_sampling(phase='val')

    def on_test_epoch_end(self):
        if self.sampling_metric is None:
            return
        self.evaluate_sampling(phase='test')

    def evaluate_sampling(self, phase='val'):
        if self.sampling_metric is None:
            raise RuntimeError("Please run self.init_metric first!")
        num_samples = self.cfg.sampling.num_samples
        if num_samples <= 0:
            num_samples = (
                self.sampling_metric.num_graphs_test if phase == 'test' else self.sampling_metric.num_graphs_val
            )
        graphs, total_time = self.generate(num_samples=num_samples)
        self.log(f"{phase}/time(s)", total_time)
        metric_log = self.sampling_metric(graphs, split=phase)
        for key in metric_log:
            if key in self.sampling_metric.to_log_metrics:
                self.log(f"{phase}/{key}", metric_log[key])
        if self.cfg.datamodule.dataset_names in ['QM9', 'MOSES', 'Guacamol']:
            if phase == "test":
                with open(osp.join(self.cfg.logs.path, "generated.smiles"), 'w') as f:
                    for smiles in metric_log['smiles']:
                        f.write("%s\n" % smiles)
                    print('All smiles saved')
                smiles_path = f"{self.cfg.logs.path}/smiles"
                os.makedirs(smiles_path, exist_ok=True)
                plot_smiles(smiles_path, metric_log['smiles'][:12])
        else:
            if self.cfg.wandb or phase == "test":
                fig_graphs = plot_gridspec_graphs(graphs[:9])
                plt.savefig(osp.join(self.cfg.logs.path, "generated_graphs.pdf"))
                if self.cfg.wandb:
                    import wandb
                    self.logger.experiment.log({"sampling/graphs": [wandb.Image(fig_graphs)]})
                plt.close(fig_graphs)

    def generate(self, num_samples=None, input_ids=None, return_sent=False):
        num_samples = self.cfg.sampling.num_samples if num_samples is None else num_samples
        if input_ids is not None and isinstance(input_ids, torch.Tensor):
            num_samples = input_ids.shape[0]
        graphs = []
        total_time = 0
        min_length = getattr(self.cfg.sampling, "min_length", 0)
        min_edges = getattr(self.cfg.sampling, "min_edges", 0)
        min_net_tokens = getattr(self.cfg.sampling, "min_net_tokens", 0)
        net_token_target = getattr(self.cfg.sampling, "net_token_target", 0)
        net_token_bias = getattr(self.cfg.sampling, "net_token_bias", 0.0)
        pin_budget_mask = bool(getattr(self.cfg.sampling, "pin_budget_mask", False))
        bipartite_node_label_mask = bool(getattr(self.cfg.sampling, "bipartite_node_label_mask", False))
        pin_slot_mask = bool(getattr(self.cfg.sampling, "pin_slot_mask", False))
        pin_slot_mask_mode = getattr(self.cfg.sampling, "pin_slot_mask_mode", None)
        if getattr(self.cfg.sampling, "max_gates", None) is not None:
            try:
                setattr(self.tokenizer, 'max_gates', int(getattr(self.cfg.sampling, "max_gates")))
            except Exception:
                pass
        if getattr(self.cfg.sampling, "target_min_gates", None) is not None:
            try:
                setattr(self.tokenizer, 'target_min_gates', int(getattr(self.cfg.sampling, "target_min_gates")))
            except Exception:
                pass
        elif (
            bool(getattr(self.cfg.sampling, "use_bucket_min_gate_for_eos", False))
            and getattr(self.cfg.sampling, "target_gate_count_bucket", None) is not None
        ):
            try:
                bucket = int(getattr(self.cfg.sampling, "target_gate_count_bucket"))
                buckets = sorted(int(x) for x in getattr(self.tokenizer, "gate_count_buckets", []) if int(x) > 0)
                lower = 1
                if bucket in buckets:
                    idx = buckets.index(bucket)
                    lower = 1 if idx == 0 else int(buckets[idx - 1] + 1)
                setattr(self.tokenizer, 'target_min_gates', int(lower))
            except Exception:
                pass
        else:
            try:
                setattr(self.tokenizer, 'target_min_gates', 0)
            except Exception:
                pass
        if getattr(self.cfg.sampling, "target_max_gates", None) is not None:
            try:
                setattr(self.tokenizer, 'target_max_gates', int(getattr(self.cfg.sampling, "target_max_gates")))
            except Exception:
                pass
        if getattr(self.cfg.sampling, "max_extra_pins_per_gate", None) is not None:
            try:
                setattr(self.tokenizer, 'max_extra_pins_per_gate', int(getattr(self.cfg.sampling, "max_extra_pins_per_gate")))
            except Exception:
                pass
        net_token_id = None
        if min_net_tokens > 0 and getattr(self.tokenizer, "labeled_graph", False):
            net_id = _resolve_circuit_net_id(self.cfg)
            if hasattr(self.tokenizer, "node_idx_offset"):
                net_token_id = self.tokenizer.node_idx_offset + net_id
        if net_token_id is None and net_token_target > 0 and getattr(self.tokenizer, "labeled_graph", False):
            net_id = _resolve_circuit_net_id(self.cfg)
            if hasattr(self.tokenizer, "node_idx_offset"):
                net_token_id = self.tokenizer.node_idx_offset + net_id
        for i in range(0, num_samples, self.cfg.sampling.batch_size):
            batch_size = min(self.cfg.sampling.batch_size, num_samples - i)
            if input_ids is not None and isinstance(input_ids, torch.Tensor):
                init_walk_idx = input_ids[i:i + batch_size]
                init_walk_idx = init_walk_idx.to(self.device)
            else:
                init = [int(self.tokenizer.sos)]
                target_bucket = getattr(self.cfg.sampling, "target_gate_count_bucket", None)
                if target_bucket is not None and getattr(self.tokenizer, 'gc_tokens', None) is not None:
                    tid = getattr(self.tokenizer, 'gc_tokens', {}).get(int(target_bucket))
                    if tid is not None:
                        init.append(int(tid))
                emitted_prefix_tokens = []
                if len(init) > 1:
                    emitted_prefix_tokens.append(int(init[-1]))
                for cfg_name, tok_attr in [
                    ("target_driver_stub_bucket", "driver_stub_bucket_tokens"),
                    ("target_load_stub_bucket", "load_stub_bucket_tokens"),
                    ("target_driver_load_ratio_bucket", "driver_load_ratio_bucket_tokens"),
                ]:
                    target = getattr(self.cfg.sampling, cfg_name, None)
                    tok_map = getattr(self.tokenizer, tok_attr, None)
                    if target is not None and tok_map is not None:
                        tid = tok_map.get(int(target))
                        if tid is not None:
                            init.append(int(tid))
                            emitted_prefix_tokens.append(int(tid))
                try:
                    self.tokenizer.last_sampling_prefix_tokens = list(emitted_prefix_tokens)
                except Exception:
                    pass
                init_walk_idx = torch.tensor(init, dtype=torch.long, device=self.device).reshape(1, -1).repeat(batch_size, 1)
            tic = timer()
            net_id = _resolve_circuit_net_id(self.cfg)
            boundary_stub_id = _resolve_circuit_boundary_stub_id(self.cfg)
            graph = self.model.generate(
                init_walk_idx,
                top_k=self.cfg.sampling.top_k,
                top_p=float(getattr(self.cfg.sampling, "top_p", 1.0)),
                temperature=self.cfg.sampling.temperature,
                max_length=self.cfg.sampling.max_length,
                min_length=min_length,
                min_edges=min_edges,
                min_net_tokens=min_net_tokens,
                net_token_id=net_token_id,
                net_token_target=net_token_target,
                net_token_bias=net_token_bias,
                return_sent=return_sent,
                net_id=net_id,
                boundary_stub_id=boundary_stub_id,
                pin_budget_mask=pin_budget_mask,
                bipartite_node_label_mask=bipartite_node_label_mask,
                pin_slot_mask=pin_slot_mask,
                pin_slot_mask_mode=pin_slot_mask_mode,
            )
            if getattr(self.cfg.sampling, "project_sequence", False):
                if return_sent:
                    graph = [project_labeled_sequence(self.tokenizer, s) for s in graph]
                else:
                    graph = [self.tokenizer.decode(project_labeled_sequence(self.tokenizer, g)) for g in graph]
            if getattr(self.cfg.sampling, "project_bipartite", False) and not return_sent:
                net_id = _resolve_circuit_net_id(self.cfg)
                graph = [project_bipartite_graph(g, net_id=net_id) for g in graph]
            toc = timer()
            graphs.extend(graph)
            total_time += toc - tic
        total_time /= num_samples
        assert len(graphs) == num_samples, "mismatched length"
        return graphs, total_time

    def configure_optimizers(self):
        params = [
            p for p in list(self.model.parameters()) + list(self.aux_heads.parameters())
            if p.requires_grad
        ]
        optimizer = hydra.utils.instantiate(
            self.cfg.train.optimizer, params
        )
        lr_scheduler = hydra.utils.call(self.cfg.train.lr_scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }


class HFSequenceModel(nn.Module):
    model_sizes = {
        'xs': {
            'hidden_size': 384,
            'num_hidden_layers': 6,
            'num_attention_heads': 12,
            'intermediate_size': 4 * 384,
            'n_embd': 384,
            'n_head': 12,
            'n_layer': 6,
        },
        's': {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 4 * 768,
            'n_embd': 768,
            'n_head': 12,
            'n_layer': 12,
        },
        'm': {
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'intermediate_size': 4 * 1024,
            'n_embd': 1024,
            'n_head': 16,
            'n_layer': 24,
        },
    }

    model_params = {
        'mamba': {
            'initializer_range': 0.02,
            'rescale_prenorm_residual': True,
        },
        'llama': {
            "rms_norm_eps": 1e-05,
            "max_position_embeddings": 8192,
        },
        'llama2': {
            "rms_norm_eps": 1e-05,
            "max_position_embeddings": 4096,
        },
        'llama3': {
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            "rope_theta": 500000.0,
        }
    }

    model_classes = {
        'gpt2': (transformers.GPT2Config, transformers.GPT2LMHeadModel),
        'llama': (transformers.LlamaConfig, transformers.LlamaForCausalLM),
        'llama2': (transformers.LlamaConfig, transformers.LlamaForCausalLM),
        'llama3': (transformers.LlamaConfig, transformers.LlamaForCausalLM),
        'gpt_neox': (transformers.GPTNeoXConfig, transformers.GPTNeoXForCausalLM),
        'mamba': (transformers.MambaConfig, transformers.MambaForCausalLM),
    }

    def __init__(self, tokenizer, model_name, **kwargs):
        super().__init__()

        self.tokenizer = tokenizer
        gradient_checkpointing = bool(kwargs.pop("gradient_checkpointing", False))
        self.max_length = tokenizer.truncation_length
        if self.max_length is not None:
            self.max_length = max(2048, self.max_length)
        self.model_name = model_name

        vocab_params = {
            'vocab_size': len(tokenizer),
            'bos_token_id': tokenizer.sos,
            'eos_token_id': tokenizer.eos,
            'pad_token_id': tokenizer.pad,
        }

        model_name_splits = model_name.split('-')
        model_name = model_name_splits[0]
        model_size = model_name_splits[1]

        model_size_params = self.model_sizes[model_size]
        if model_name == "mamba":
            num_layers = model_size_params['num_hidden_layers'] * 2
            model_size_params = {}
            model_size_params['num_hidden_layers'] = num_layers
        model_params = self.model_params.get(model_name, {})

        model_config, model_cls = self.model_classes[model_name]

        merged_config = {}
        merged_config.update(vocab_params)
        merged_config.update(model_size_params)
        merged_config.update(model_params)
        merged_config.update(kwargs)
        self.model = model_cls(model_config(**merged_config))
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

    def forward(self, input_ids, return_hidden_states=False):
        outputs = self.model(input_ids=input_ids, output_hidden_states=return_hidden_states)
        if return_hidden_states:
            return outputs.logits, outputs.hidden_states[-1]
        return outputs.logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        top_k=10,
        top_p=1.0,
        temperature=1.0,
        max_length=None,
        min_length=0,
        min_edges=0,
        min_net_tokens=0,
        net_token_id=None,
        net_token_target=0,
        net_token_bias=0.0,
        return_sent=False,
        net_id=None,
        boundary_stub_id=None,
        pin_budget_mask=False,
        bipartite_node_label_mask=False,
        pin_slot_mask=False,
        pin_slot_mask_mode=None,
    ):
        batch_size = input_ids.shape[0]
        max_length = self.max_length if max_length is None else max_length

        cfg_max_pos = getattr(getattr(self.model, 'config', None), 'max_position_embeddings', None)
        if cfg_max_pos is not None and int(max_length) > int(cfg_max_pos):
            raise ValueError(f"max_length={int(max_length)} exceeds model.max_position_embeddings={int(cfg_max_pos)}")

        logits_processor = None
        processors = []
        if self.tokenizer.labeled_graph:
            processors.append(StartWithNodeIndex(self.tokenizer, input_ids.device))
            processors.append(
                LabeledGraph(
                    self.tokenizer,
                    batch_size,
                    input_ids.device,
                    net_id=net_id,
                    boundary_stub_id=boundary_stub_id,
                    pin_budget_mask=pin_budget_mask,
                    bipartite_node_label_mask=bipartite_node_label_mask,
                )
            )
        try:
            self.tokenizer.last_grammar_processor_created = False
            self.tokenizer.last_grammar_processor_name = None
            self.tokenizer.last_grammar_processor_mode = str(pin_slot_mask_mode or "")
        except Exception:
            pass
        if (not getattr(self.tokenizer, "labeled_graph", False)) and bool(pin_slot_mask):
            if bool(getattr(self.tokenizer, "is_source_net_v61_tokenizer", False)):
                processors.append(SourceNetV61Grammar(
                    self.tokenizer,
                    batch_size,
                    input_ids.device,
                    mask_mode=pin_slot_mask_mode,
                ))
                try:
                    self.tokenizer.last_grammar_processor_created = True
                    self.tokenizer.last_grammar_processor_name = "SourceNetV61Grammar"
                except Exception:
                    pass
            elif bool(getattr(self.tokenizer, "is_source_net_v5_tokenizer", False)):
                processors.append(SourceNetV5Grammar(self.tokenizer, batch_size, input_ids.device, mask_mode=pin_slot_mask_mode))
                try:
                    self.tokenizer.last_grammar_processor_created = True
                    self.tokenizer.last_grammar_processor_name = "SourceNetV5Grammar"
                except Exception:
                    pass
            elif bool(getattr(self.tokenizer, "is_source_net_v41_tokenizer", False)):
                processors.append(SourceNetV41SinkFirstGrammar(self.tokenizer, batch_size, input_ids.device, mask_mode=pin_slot_mask_mode))
                try:
                    self.tokenizer.last_grammar_processor_created = True
                    self.tokenizer.last_grammar_processor_name = "SourceNetV41SinkFirstGrammar"
                except Exception:
                    pass
            elif bool(getattr(self.tokenizer, "is_source_net_v4_tokenizer", False)):
                processors.append(SourceNetV4Grammar(self.tokenizer, batch_size, input_ids.device, mask_mode=pin_slot_mask_mode))
                try:
                    self.tokenizer.last_grammar_processor_created = True
                    self.tokenizer.last_grammar_processor_name = "SourceNetV4Grammar"
                except Exception:
                    pass
            elif bool(getattr(self.tokenizer, "is_pin_slot_tokenizer", False) or hasattr(self.tokenizer, "pin_in")):
                processors.append(PinSlotGrammar(self.tokenizer, batch_size, input_ids.device, mask_mode=pin_slot_mask_mode))
                try:
                    self.tokenizer.last_grammar_processor_created = True
                    self.tokenizer.last_grammar_processor_name = "PinSlotGrammar"
                except Exception:
                    pass
        if min_length > 0 or min_edges > 0 or min_net_tokens > 0:
            if not self.tokenizer.labeled_graph and min_edges > 0:
                min_edges = 0
            processors.append(
                MinLengthAndEdge(
                    self.tokenizer,
                    batch_size,
                    min_length,
                    min_edges,
                    min_net_tokens,
                    net_token_id,
                    input_ids.device,
                )
            )
        if net_token_target > 0 and net_token_bias > 0 and net_token_id is not None:
            processors.append(
                NetTokenBias(
                    net_token_id,
                    net_token_target,
                    net_token_bias,
                    batch_size,
                    input_ids.device,
                )
            )
        if processors:
            processors.append(SafeGenerationLogits(self.tokenizer))
        if processors:
            logits_processor = LogitsProcessorList(processors)
        walk_idx = self.model.generate(
            input_ids,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            max_length=max_length,
            logits_processor=logits_processor,
        )
        edge_index_list = []
        for i in range(batch_size):
            if return_sent:
                edge_index = walk_idx[i]
            else:
                edge_index = self.tokenizer.decode(walk_idx[i])
            edge_index_list.append(edge_index)
        return edge_index_list


class LabeledGraph(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        batch_size,
        device='cpu',
        net_id=None,
        boundary_stub_id=None,
        pin_budget_mask=False,
        bipartite_node_label_mask=False,
    ):
        self.start_bracket = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=device
        )
        self.next_node_idx = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device
        )
        self.tokenizer = tokenizer
        idx = torch.arange(len(tokenizer), device=device)
        self.vocab_idx = idx
        self.special_idx = idx < tokenizer.idx_offset
        self.schema_idx = torch.zeros_like(idx, dtype=torch.bool)
        self.schema_position_masks = []
        if getattr(tokenizer, "has_schema_tokens", False):
            for pos in range(len(getattr(tokenizer, "schema_token_specs", []))):
                mask = torch.zeros_like(idx, dtype=torch.bool)
                for tok_id in tokenizer.get_schema_token_ids_for_position(pos):
                    mask[int(tok_id)] = True
                    self.schema_idx[int(tok_id)] = True
                self.schema_position_masks.append(mask)
        self.idx = (idx >= tokenizer.idx_offset) & (idx < tokenizer.node_idx_offset)
        self.node_idx = (idx >= tokenizer.node_idx_offset) & (idx < tokenizer.edge_idx_offset)
        self.edge_idx = idx >= tokenizer.edge_idx_offset
        self.reset_idx = idx == self.tokenizer.reset
        self.ladj_idx = idx == self.tokenizer.ladj
        self.radj_idx = idx == self.tokenizer.radj
        self.eos_idx = idx == self.tokenizer.eos
        self.pad_idx = idx == self.tokenizer.pad
        self.schema_pos = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        # V3: Pin Budget Tracking
        self.net_id = net_id
        self.boundary_stub_id = boundary_stub_id
        self.pin_specs = getattr(tokenizer, "pin_specs", None)
        self.pin_budget_mask = bool(pin_budget_mask)
        self.bipartite_node_label_mask = bool(bipartite_node_label_mask)
        self.gate_node_idx = self.node_idx.clone()
        self.nongate_node_idx = torch.zeros_like(idx, dtype=torch.bool)
        if net_id is not None and hasattr(tokenizer, "node_idx_offset"):
            net_tok = int(tokenizer.node_idx_offset) + int(net_id)
            if 0 <= net_tok < len(tokenizer):
                self.gate_node_idx[net_tok] = False
                self.nongate_node_idx[net_tok] = True
        if boundary_stub_id is not None and hasattr(tokenizer, "node_idx_offset"):
            stub_tok = int(tokenizer.node_idx_offset) + int(boundary_stub_id)
            if 0 <= stub_tok < len(tokenizer):
                self.gate_node_idx[stub_tok] = False
                self.nongate_node_idx[stub_tok] = True
        if self.pin_specs is not None and self.pin_budget_mask:
            self.pin_specs = self.pin_specs.to(device)
            max_nodes = int(tokenizer.max_num_nodes or 256)
            # node_budgets: [B, max_nodes] (total pins to satisfy)
            self.node_budgets = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
            self.node_is_gate = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
            self.current_u = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
            self.awaiting_new_node_idx = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
            self.pending_label_expect_gate = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
            self.pending_preconsumed_edges = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        else:
            self.node_budgets = None

    def modify_scores(self, scores, sampled_idx):
        scores[:, self.tokenizer.sos] = float("-inf")
        prev_next_node_idx = self.next_node_idx.clone()

        # Track bracket state
        prev_ladj = sampled_idx == self.tokenizer.ladj
        prev_radj = sampled_idx == self.tokenizer.radj
        self.start_bracket = (self.start_bracket | prev_ladj) & (~prev_radj)

        prev_schema = self.schema_idx[sampled_idx]
        self.schema_pos = self.schema_pos + prev_schema.long()

        # Identify token types
        prev_idx = self.idx[sampled_idx]
        prev_node = self.node_idx[sampled_idx]
        prev_edge = self.edge_idx[sampled_idx]
        prev_reset = sampled_idx == self.tokenizer.reset
        prev_sos = sampled_idx == self.tokenizer.sos
        prev_eos = sampled_idx == self.tokenizer.eos
        prev_pad = sampled_idx == self.tokenizer.pad

        # V3: Update budgets and current node
        if self.node_budgets is not None:
            old_current_u = self.current_u.clone()
            old_awaiting_new_node_idx = self.awaiting_new_node_idx.clone()
            old_start_bracket = self.start_bracket.clone()
            # When an idx is sampled outside bracket, it's the new current_u
            is_idx_out = prev_idx & (~self.start_bracket)
            sampled_node_idx = (sampled_idx - self.tokenizer.idx_offset).long()

            if is_idx_out.any():
                for b in range(sampled_idx.shape[0]):
                    if is_idx_out[b] and old_awaiting_new_node_idx[b]:
                        src = int(old_current_u[b].item())
                        if 0 <= src < self.node_is_gate.shape[1]:
                            src_is_gate = bool(self.node_is_gate[b, src].item())
                            self.pending_label_expect_gate[b] = 0 if src_is_gate else 1
                            self.pending_preconsumed_edges[b] = 1
                        else:
                            self.pending_label_expect_gate[b] = -1
                            self.pending_preconsumed_edges[b] = 0
            self.current_u = torch.where(is_idx_out, sampled_node_idx, self.current_u)
            self.awaiting_new_node_idx = torch.where(
                is_idx_out,
                torch.zeros_like(self.awaiting_new_node_idx),
                self.awaiting_new_node_idx,
            )

            # When a node_label is sampled, set its budget
            if prev_node.any():
                for b in range(sampled_idx.shape[0]):
                    if prev_node[b]:
                        u = int(self.current_u[b].item())
                        label_tok = int(sampled_idx[b].item())
                        if 0 <= u < self.node_budgets.shape[1]:
                            # Sum of inputs and outputs for simplicity in budget tracking
                            # In a bipartite graph, any edge connected to a gate consumes one pin slot.
                            pins = self.pin_specs[label_tok].sum().item()
                            pins = max(0, int(pins) - int(self.pending_preconsumed_edges[b].item()))
                            self.node_budgets[b, u] = int(pins)
                            # Check if it's a gate (not net/stub)
                            label_val = label_tok - self.tokenizer.node_idx_offset
                            is_gate = (label_val != self.net_id) and (label_val != self.boundary_stub_id)
                            self.node_is_gate[b, u] = is_gate
                            self.pending_label_expect_gate[b] = -1
                            self.pending_preconsumed_edges[b] = 0

            # When an edge is formed inside [ladj ... radj]
            # (u, v) edge where u is current_u and v is the sampled_idx (which is an idx)
            is_idx_in = prev_idx & self.start_bracket
            if is_idx_in.any():
                for b in range(sampled_idx.shape[0]):
                    if is_idx_in[b]:
                        u = int(self.current_u[b].item())
                        v = int(sampled_node_idx[b].item())
                        if 0 <= u < self.node_budgets.shape[1] and self.node_is_gate[b, u]:
                            self.node_budgets[b, u] = max(0, self.node_budgets[b, u] - 1)
                        if 0 <= v < self.node_budgets.shape[1] and self.node_is_gate[b, v]:
                            self.node_budgets[b, v] = max(0, self.node_budgets[b, v] - 1)

            edge_out = prev_edge & (~old_start_bracket)
            self.awaiting_new_node_idx = torch.where(
                edge_out,
                torch.ones_like(self.awaiting_new_node_idx),
                self.awaiting_new_node_idx,
            )

        allow_idx = self.idx
        allow_node = self.node_idx
        allow_edge = self.edge_idx
        allow_ladj = self.ladj_idx
        allow_radj = self.radj_idx
        allow_reset = self.reset_idx
        allow_eos = self.eos_idx

        has_schema = len(self.schema_position_masks) > 0

        # Labeled SENT node-index tokens are local visit-order ids, not
        # arbitrary graph node ids.  For a new trail/step, force the next
        # unseen node index; inside brackets, only allow references to already
        # emitted node indices.  Without this, generation can emit a high idx
        # token (e.g. 1381) after only a few node labels, and decoding creates
        # thousands of fake isolated nodes.
        self.next_node_idx = torch.where(
            prev_node,
            torch.clamp(self.next_node_idx + 1, max=max(0, int(self.tokenizer.max_num_nodes or 0))),
            self.next_node_idx,
        )
        node_token_ids = self.vocab_idx - self.tokenizer.idx_offset
        force_next_idx_mask = (
            self.idx.unsqueeze(0)
            & (node_token_ids.unsqueeze(0) == prev_next_node_idx)
            & (prev_next_node_idx < int(self.tokenizer.max_num_nodes or 0))
        )
        existing_idx_mask = (
            self.idx.unsqueeze(0)
            & (node_token_ids.unsqueeze(0) >= 0)
            & (node_token_ids.unsqueeze(0) < self.next_node_idx)
        )
        # Start from all masked, then open grammar-valid token sets.
        masked = torch.full_like(scores, float("-inf"))

        # idx -> node_label (outside bracket), or edge_label/radj (inside bracket)
        allowed_idx_out = torch.where(allow_node, scores, float("-inf"))
        if self.bipartite_node_label_mask and self.node_budgets is not None:
            expect_gate = self.pending_label_expect_gate
            allowed_gate = torch.where(self.gate_node_idx, scores, float("-inf"))
            allowed_nongate = torch.where(self.nongate_node_idx, scores, float("-inf"))
            allowed_idx_out = torch.where(expect_gate == 1, allowed_gate, allowed_idx_out)
            allowed_idx_out = torch.where(expect_gate == 0, allowed_nongate, allowed_idx_out)
        allowed_idx_in = torch.where(allow_edge | allow_radj, scores, float("-inf"))
        masked = torch.where(prev_idx & (~self.start_bracket), allowed_idx_out, masked)
        masked = torch.where(prev_idx & self.start_bracket, allowed_idx_in, masked)

        # node_label -> edge_label | ladj | reset | eos
        # V3: Mask reset and eos if budgets are not met
        can_exit = torch.ones((sampled_idx.shape[0],), dtype=torch.bool, device=scores.device)
        if self.node_budgets is not None:
            # Any gate with remaining budget blocks exit
            has_deficit = (self.node_budgets > 0).any(dim=1)
            can_exit = ~has_deficit

        allow_exit = allow_reset | allow_eos
        allowed_after_node_no_exit = torch.where(allow_edge | allow_ladj, scores, float("-inf"))
        allowed_after_node_with_exit = torch.where(allow_edge | allow_ladj | allow_exit, scores, float("-inf"))
        masked = torch.where(prev_node,
                             torch.where(can_exit.unsqueeze(1), allowed_after_node_with_exit, allowed_after_node_no_exit),
                             masked)

        # edge_label -> idx
        allowed_after_edge_out = torch.where(force_next_idx_mask, scores, float("-inf"))
        allowed_after_edge_in = torch.where(existing_idx_mask, scores, float("-inf"))
        allowed_after_edge = torch.where(self.start_bracket, allowed_after_edge_in, allowed_after_edge_out)
        masked = torch.where(prev_edge, allowed_after_edge, masked)

        # ladj -> edge_label
        allowed_after_ladj = torch.where(allow_edge, scores, float("-inf"))
        masked = torch.where(prev_ladj, allowed_after_ladj, masked)

        # radj -> edge_label | reset | eos
        # V3: Mask reset and eos if budgets are not met
        allowed_after_radj_no_exit = torch.where(allow_edge, scores, float("-inf"))
        allowed_after_radj_with_exit = torch.where(allow_edge | allow_exit, scores, float("-inf"))
        masked = torch.where(prev_radj,
                             torch.where(can_exit.unsqueeze(1), allowed_after_radj_with_exit, allowed_after_radj_no_exit),
                             masked)

        # sos -> first schema token when schema prefix is enabled, else idx
        if has_schema:
            first_schema_mask = self.schema_position_masks[0]
            allowed_after_sos = torch.where(first_schema_mask, scores, float("-inf"))
        else:
            allowed_after_sos = torch.where(force_next_idx_mask, scores, float("-inf"))
        masked = torch.where(prev_sos, allowed_after_sos, masked)

        # schema token -> next schema token, otherwise idx
        if has_schema:
            for pos, schema_mask in enumerate(self.schema_position_masks[1:], start=1):
                next_allowed = torch.where(schema_mask, scores, float("-inf"))
                masked = torch.where(prev_schema & (self.schema_pos == pos), next_allowed, masked)
            allowed_after_last_schema = torch.where(force_next_idx_mask, scores, float("-inf"))
            masked = torch.where(prev_schema & (self.schema_pos >= len(self.schema_position_masks)), allowed_after_last_schema, masked)

        # reset -> idx
        allowed_after_reset = torch.where(force_next_idx_mask, scores, float("-inf"))
        masked = torch.where(prev_reset, allowed_after_reset, masked)
        masked = torch.where(prev_sos, allowed_after_sos, masked)

        # eos/pad rows can still be present in batched generation before all
        # sequences finish; keep a valid distribution to avoid all -inf rows.
        allowed_finished = torch.where(self.pad_idx, scores, float("-inf"))
        masked = torch.where(prev_eos | prev_pad, allowed_finished, masked)

        # If the generated graph reaches tokenizer.max_num_nodes, forcing the
        # next unseen local node index would produce an all--inf row.  End the
        # sequence instead of deadlocking generation.  This keeps the hard
        # training-set node cap explicit while making extrapolation failure
        # graceful and diagnosable.
        max_nodes = int(self.tokenizer.max_num_nodes or 0)
        if max_nodes > 0:
            at_node_capacity = self.next_node_idx >= max_nodes
            capacity_exit = torch.where(self.eos_idx | self.reset_idx, scores, float("-inf"))
            masked = torch.where(at_node_capacity & prev_edge & (~self.start_bracket), capacity_exit, masked)
            masked = torch.where(at_node_capacity & prev_reset, torch.where(self.eos_idx, scores, float("-inf")), masked)

        # Last-resort guard for per-sample state drift.  It should be rare, but
        # batched generation must never return an all--inf row to transformers.
        finite_rows = torch.isfinite(masked).any(dim=-1, keepdim=True)
        eos_fallback = torch.where(self.eos_idx | self.pad_idx, scores, float("-inf"))
        masked = torch.where(finite_rows, masked, eos_fallback)

        return masked

    def __call__(self, input_ids, scores):
        return self.modify_scores(scores, input_ids[:, -1:])


class StartWithNodeIndex(LogitsProcessor):
    def __init__(self, tokenizer, device="cpu"):
        self.tokenizer = tokenizer
        idx = torch.arange(len(tokenizer), device=device)
        self.idx_mask = (idx >= tokenizer.idx_offset) & (idx < tokenizer.node_idx_offset)
        self.first_schema_mask = None
        if getattr(tokenizer, "has_schema_tokens", False) and getattr(tokenizer, "schema_token_specs", None):
            self.first_schema_mask = torch.zeros_like(idx, dtype=torch.bool)
            for tok_id in tokenizer.get_schema_token_ids_for_position(0):
                self.first_schema_mask[int(tok_id)] = True

    def __call__(self, input_ids, scores):
        # Enforce the first token after SOS to be the schema prefix start when
        # enabled, otherwise fall back to the first node index token.
        if input_ids.shape[1] == 1:
            if self.first_schema_mask is not None:
                scores = torch.where(self.first_schema_mask, scores, float("-inf"))
            else:
                scores = torch.where(self.idx_mask, scores, float("-inf"))
        return scores


class MinLengthAndEdge(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        batch_size,
        min_length,
        min_edges,
        min_net_tokens,
        net_token_id,
        device='cpu'
    ):
        self.eos_id = tokenizer.eos
        self.min_length = max(0, int(min_length))
        self.min_edges = max(0, int(min_edges))
        self.min_net_tokens = max(0, int(min_net_tokens))
        self.net_token_id = net_token_id
        self.edge_count = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.net_count = torch.zeros((batch_size,), dtype=torch.long, device=device)
        if hasattr(tokenizer, "edge_idx_offset"):
            idx = torch.arange(len(tokenizer), device=device)
            self.is_edge = idx >= tokenizer.edge_idx_offset
        else:
            self.is_edge = torch.zeros((len(tokenizer),), dtype=torch.bool, device=device)

    def __call__(self, input_ids, scores):
        last = input_ids[:, -1]
        self.edge_count += self.is_edge[last].long()
        if self.net_token_id is not None:
            self.net_count += (last == self.net_token_id).long()
        if self.min_length > 0 and input_ids.shape[1] < self.min_length:
            scores[:, self.eos_id] = float('-inf')
        if self.min_edges > 0:
            mask = self.edge_count < self.min_edges
            if mask.any():
                scores[mask, self.eos_id] = float('-inf')
        if self.min_net_tokens > 0 and self.net_token_id is not None:
            mask = self.net_count < self.min_net_tokens
            if mask.any():
                scores[mask, self.eos_id] = float('-inf')
        return scores


class NetTokenBias(LogitsProcessor):
    def __init__(self, net_token_id, target_count, bias, batch_size, device="cpu"):
        self.net_token_id = int(net_token_id)
        self.target_count = max(0, int(target_count))
        self.bias = float(bias)
        self.net_count = torch.zeros((batch_size,), dtype=torch.long, device=device)

    def __call__(self, input_ids, scores):
        last = input_ids[:, -1]
        self.net_count += (last == self.net_token_id).long()
        if self.target_count > 0 and self.bias > 0:
            mask = self.net_count < self.target_count
            if mask.any():
                scores[mask, self.net_token_id] += self.bias
        return scores


class SourceNetV61Grammar(LogitsProcessor):
    """Stateful grammar for V6.1 compact-load object-table generation."""

    FEATURE_NAMES = (
        "gale_ryser",
        "same_cell_exclusion",
        "source_budget",
        "demand_exactly_once",
        "completion_eos",
        "output_source_coverage",
        "pointer_mask",
        "self_drive_exclusion",
        "combinational_cycle_exclusion",
        "profile_feasibility",
    )
    ABLATION_MODES = {
        "v61_full_load": (
            "self_drive_exclusion", "combinational_cycle_exclusion",
        ),
        "v61_d0_full": (
            "self_drive_exclusion", "combinational_cycle_exclusion",
        ),
        "v61_d0_topology_safe": (),
        "v61_d6_no_topology_safety": (
            "self_drive_exclusion", "combinational_cycle_exclusion",
        ),
        "v61_d1_no_gale_ryser": ("gale_ryser",),
        "v61_d2_no_same_cell_exclusion": ("same_cell_exclusion",),
        "v61_d3_no_source_budget_mask": ("source_budget",),
        "v61_d4_no_demand_exactly_once_mask": ("demand_exactly_once",),
        "v61_d5_no_completion_eos_gate": ("completion_eos",),
        "v61_d6_no_output_source_coverage": (
            "output_source_coverage", "self_drive_exclusion",
            "combinational_cycle_exclusion",
        ),
        "v61_d7_no_pointer_mask": (
            "pointer_mask", "self_drive_exclusion",
            "combinational_cycle_exclusion",
        ),
        "v61_syntax_only": (
            "gale_ryser",
            "same_cell_exclusion",
            "source_budget",
            "demand_exactly_once",
            "completion_eos",
            "output_source_coverage",
            "self_drive_exclusion",
            "combinational_cycle_exclusion",
        ),
        # Representation R2 without its variant-specific future-source
        # feasibility solver. Keep only local pointer/pin/demand/section
        # constraints; no encoded or inferred per-source capacity is used.
        "v61_r2_basic": (
            "gale_ryser", "source_budget", "self_drive_exclusion",
            "combinational_cycle_exclusion",
        ),
        # Short-term full-design scaffold replacement mode.  This keeps the
        # V6.1 legality constraints active while adding optional target
        # profile bounds supplied by the sampling script.  It is intentionally
        # a new mode so previous D0/D1-D7 and representation ablation results
        # remain reproducible.
        "v61_scaffold_conditioned": (),
        "v61_scaffold_reuse_tables": (),
        "v62_profile_full": (),
        "v62_profile_feasible_lm": (
            "gale_ryser", "same_cell_exclusion", "source_budget",
            "demand_exactly_once", "completion_eos", "output_source_coverage",
            "self_drive_exclusion", "combinational_cycle_exclusion",
        ),
        "v62_profile_only_lm": (
            "gale_ryser", "same_cell_exclusion", "source_budget",
            "demand_exactly_once", "completion_eos", "output_source_coverage",
            "self_drive_exclusion", "combinational_cycle_exclusion",
            "profile_feasibility",
        ),
    }

    def __init__(self, tokenizer, batch_size, device="cpu", mask_mode=None, features=None):
        self.t = tokenizer
        self.device = device
        self.representation_variant = str(
            getattr(tokenizer, "representation_variant", "") or ""
        )
        mode = str(mask_mode or "v61_d0_full")
        if mode not in self.ABLATION_MODES:
            raise ValueError(
                f"unknown V6.1 grammar mode {mode!r}; expected one of "
                f"{sorted(self.ABLATION_MODES)}"
            )
        self.mask_mode = mode
        self.r2_future_feasibility = not (
            self._is_variant("r2_no_per_source_budget")
            and mode == "v61_r2_basic"
        )
        self.target_profile = (
            getattr(tokenizer, "source_net_v61_target_profile", None)
            if mode == "v61_scaffold_conditioned" else None
        )
        self.features = {name: True for name in self.FEATURE_NAMES}
        for name in self.ABLATION_MODES[mode]:
            self.features[name] = False
        if features is not None:
            unknown = set(features) - set(self.FEATURE_NAMES)
            if unknown:
                raise ValueError(f"unknown V6.1 grammar feature flags: {sorted(unknown)}")
            self.features.update({name: bool(value) for name, value in features.items()})
        self.processed = [1] * int(batch_size)
        self.states = [self._new_state() for _ in range(int(batch_size))]
        self._gale_cache = OrderedDict()
        self._mask_index_cache = OrderedDict()
        self._cache_limit = 4096
        self.cache_stats = {
            "gale_hits": 0,
            "gale_misses": 0,
            "mask_hits": 0,
            "mask_misses": 0,
        }

    def _new_state(self):
        return {
            "expect": (
                "target_profile_begin"
                if getattr(self.t, "include_target_profile", False)
                else "cell_section_begin"
            ),
            "target_profile": {},
            "cells": [],
            "sequential_cell_count": 0,
            "rare_cell_count": 0,
            "max_input_arity": 0,
            "has_mux": False,
            "has_aoi_oai": False,
            "has_multi_output_cell": False,
            "demands": [],
            "sources": [],
            "source_outputs": set(),
            "slots": set(),
            "cell_demand_counts": {},
            "current": {},
            "load_source": None,
            "load_left": 0,
            "assigned": set(),
            "load_cells": set(),
            "emitted_sources": set(),
            "combinational_adjacency": {},
        }

    def _is_variant(self, name):
        return self.representation_variant == name

    def _target_int(self, key, default=None, state=None):
        profile = (
            state.get("target_profile")
            if state is not None and state.get("target_profile")
            else self.target_profile
        )
        if not profile:
            return default
        try:
            value = profile.get(key, default)
        except AttributeError:
            return default
        if value is None:
            return default
        return int(value)

    def _target_min_cells(self, state=None):
        if not self.features.get("profile_feasibility", True):
            return None
        if state is not None and state.get("target_profile", {}).get("cell_count_range"):
            return state["target_profile"]["cell_count_range"][0]
        target = self._target_int("target_cell_count", state=state)
        tolerance = self._target_int("cell_count_tolerance", 0, state=state)
        if target is None:
            return None
        return max(1, int(target) - max(0, int(tolerance)))

    def _target_max_cells(self, state=None):
        if not self.features.get("profile_feasibility", True):
            return None
        if state is not None and state.get("target_profile", {}).get("cell_count_range"):
            return state["target_profile"]["cell_count_range"][1]
        target = self._target_int("target_cell_count", state=state)
        tolerance = self._target_int("cell_count_tolerance", 0, state=state)
        if target is None:
            return None
        return max(1, int(target) + max(0, int(tolerance)))

    def _target_boundary_source_count(self, state=None):
        return self._target_int("target_boundary_count", state=state)

    def _boundary_source_count(self, state):
        return sum(
            int(source.get("source_kind") == self.t.boundary_source)
            for source in state["sources"]
        )

    def _scaffold_source_table_complete(self, state):
        if self.mask_mode != "v61_scaffold_conditioned":
            return True
        target_boundary = self._target_boundary_source_count()
        if (
            target_boundary is not None
            and self._boundary_source_count(state) != int(target_boundary)
        ):
            return False
        return True

    def _required_demand_count(self, state):
        return len(self._required_slots(state))

    def _cell_profile_features(self, cell_type):
        if cell_type is None:
            return {
                "sequential": False, "rare": False, "input_arity": 0,
                "mux": False, "aoi_oai": False, "multi_output": False,
            }
        config = getattr(self.t, "profile_config", {})
        spec = self.t.pin_specs.get(cell_type)
        return {
            "sequential": cell_type in set(config.get("sequential_cell_types", [])),
            "rare": cell_type in set(config.get("rare_cell_types", [])),
            "input_arity": len(getattr(spec, "inputs", [])) if spec is not None else 0,
            "mux": "MUX" in cell_type,
            "aoi_oai": cell_type.startswith(("AOI", "OAI")),
            "multi_output": len(getattr(spec, "outputs", [])) > 1 if spec is not None else False,
        }

    def _cell_profile_satisfied(self, state):
        if not self.features.get("profile_feasibility", True):
            return True
        profile = state.get("target_profile", {})
        if profile.get("profile_mode") != "PROFILE_COARSE":
            return True
        count = len(state["cells"])
        cell_range = profile["cell_count_range"]
        demand_range = profile["demand_count_range"]
        seq_range = profile["seq_ratio_range"]
        pin_range = profile["pin_complexity_range"]
        rare_ratio = state["rare_cell_count"] / max(1, count)
        rare_by_count = (
            self.t.profile_config.get("rare_partition_rule")
            == "rare_cell_count_at_least_min_count"
        )
        rare_satisfied = (
            state["rare_cell_count"]
            >= int(self.t.profile_config.get("rare_partition_min_count", 1))
            if rare_by_count else
            rare_ratio >= float(self.t.profile_config["rare_partition_ratio_threshold"])
        )
        return (
            cell_range[0] <= count <= cell_range[1]
            and demand_range[0] <= self._required_demand_count(state) <= demand_range[1]
            and seq_range[0] <= state["sequential_cell_count"] / max(1, count) <= seq_range[1]
            and pin_range[0] <= state["max_input_arity"] <= pin_range[1]
            and rare_satisfied == bool(profile["rare_cell_enriched"])
            and state["has_mux"] == bool(profile["has_mux"])
            and state["has_aoi_oai"] == bool(profile["has_aoi_oai"])
            and state["has_multi_output_cell"] == bool(profile["has_multi_output_cell"])
        )

    def _candidate_profile_feasible(self, state, cell_type):
        # Pointer/static vocabulary ranges can contain tokens that do not map
        # to a legal cell label.  They are not profile-feasible cell choices.
        if cell_type is None:
            return False
        if not self.features.get("profile_feasibility", True):
            return True
        profile = state.get("target_profile", {})
        if profile.get("profile_mode") != "PROFILE_COARSE":
            return True
        feature = self._cell_profile_features(cell_type)
        # Negative semantic flags are true exclusions; positive flags leave the
        # concrete type/count free and only require eventual reachability.
        for key, feature_key in (
            ("has_mux", "mux"), ("has_aoi_oai", "aoi_oai"),
            ("has_multi_output_cell", "multi_output"),
        ):
            if not profile[key] and feature[feature_key]:
                return False
        pin_low, pin_high = profile["pin_complexity_range"]
        next_max_arity = max(state["max_input_arity"], feature["input_arity"])
        if next_max_arity > pin_high:
            return False

        cell_low, cell_high = profile["cell_count_range"]
        source_range = profile["source_count_range"]
        demand_low, demand_high = profile["demand_count_range"]
        seq_low, seq_high = profile["seq_ratio_range"]
        next_count = len(state["cells"]) + 1
        next_demand = self._required_demand_count(state) + feature["input_arity"]
        next_seq = state["sequential_cell_count"] + int(feature["sequential"])
        next_rare = state["rare_cell_count"] + int(feature["rare"])
        next_flags = {
            "has_mux": state["has_mux"] or feature["mux"],
            "has_aoi_oai": state["has_aoi_oai"] or feature["aoi_oai"],
            "has_multi_output_cell": state["has_multi_output_cell"] or feature["multi_output"],
        }
        all_arities = [
            len(getattr(spec, "inputs", [])) for spec in self.t.pin_specs.values()
        ] or [0]
        min_arity, max_arity = min(all_arities), max(all_arities)
        rare_by_count = (
            self.t.profile_config.get("rare_partition_rule")
            == "rare_cell_count_at_least_min_count"
        )
        rare_min_count = int(self.t.profile_config.get("rare_partition_min_count", 1))
        rare_threshold = float(self.t.profile_config["rare_partition_ratio_threshold"])
        for final_count in range(max(next_count, int(cell_low)), int(cell_high) + 1):
            if final_count > source_range[1]:
                continue
            remaining = final_count - next_count
            if next_demand + remaining * min_arity > demand_high:
                continue
            if next_demand + remaining * max_arity < demand_low:
                continue
            min_seq = next_seq
            max_seq = next_seq + remaining
            if max_seq / final_count < seq_low or min_seq / final_count > seq_high:
                continue
            final_pin_possible = next_max_arity >= pin_low or (
                remaining > 0 and max_arity >= pin_low
            )
            if not final_pin_possible:
                continue
            if rare_by_count:
                if profile["rare_cell_enriched"] and next_rare + remaining < rare_min_count:
                    continue
                if not profile["rare_cell_enriched"] and next_rare >= rare_min_count:
                    continue
            else:
                if profile["rare_cell_enriched"]:
                    if (next_rare + remaining) / final_count < rare_threshold:
                        continue
                elif next_rare / final_count >= rare_threshold:
                    continue
            if any(profile[key] and not next_flags[key] and remaining == 0 for key in next_flags):
                continue
            return True
        return False

    def _profile_range(self, state, key):
        value = state.get("target_profile", {}).get(key)
        return tuple(value) if value is not None else None

    def _profile_bin_range(self, token, field):
        name = self.t.special_toks[int(token)]
        if not name.startswith("PROFILE_BIN_"):
            raise ValueError(f"invalid profile bin token: {name}")
        index = int(name.rsplit("_", 1)[-1])
        boundaries = list(self.t.profile_config["bucket_boundaries"][field])
        if len(boundaries) != 3:
            raise ValueError(f"invalid frozen boundaries for {field}: {boundaries}")
        if field == "seq_ratio":
            lows = [0.0, *(math.nextafter(float(value), math.inf) for value in boundaries)]
            highs = [*boundaries, 1.0]
            return float(lows[index]), float(highs[index])
        maximum = (
            max(len(getattr(spec, "inputs", [])) for spec in self.t.pin_specs.values())
            if field == "pin_complexity" and self.t.pin_specs else int(self.t.max_num_nodes)
        )
        rounded = [int(math.floor(value)) for value in boundaries]
        lows = [0 if field == "pin_complexity" else 1, *(value + 1 for value in rounded)]
        highs = [*rounded, maximum]
        return int(lows[index]), int(highs[index])

    def _demands_complete(self, state):
        target_range = (
            self._profile_range(state, "demand_count_range")
            if self.features.get("profile_feasibility", True) else None
        )
        if target_range is not None and not target_range[0] <= len(state["demands"]) <= target_range[1]:
            return False
        if self._is_variant("r4_cell_level_demand"):
            return len(state["demands"]) == self._required_demand_count(state)
        return state["slots"] == self._required_slots(state)

    def _source_table_complete(self, state):
        if not state["sources"]:
            return False
        outputs_complete = (
            not self.features["output_source_coverage"]
            or {cell for cell, _pin in state["source_outputs"]}
            == set(range(len(state["cells"])))
        )
        if self._is_variant("r2_no_per_source_budget"):
            return outputs_complete
        budget_complete = (
            (self.mask_mode.startswith("v62_") and not self.features["source_budget"])
            or sum(row["max_fanout"] for row in state["sources"])
            == len(state["demands"])
        )
        complete = budget_complete and outputs_complete
        source_range = (
            self._profile_range(state, "source_count_range")
            if self.features.get("profile_feasibility", True) else None
        )
        if source_range is not None:
            complete = complete and source_range[0] <= len(state["sources"]) <= source_range[1]
        return complete

    def _missing_output_identity_count(self, state):
        covered = {cell for cell, _pin in state["source_outputs"]}
        return sum(cell not in covered for cell in range(len(state["cells"])))

    def _r1_source_kinds_available(self, state):
        t = self.t
        result = []
        if self._cells_with_available_outputs(state):
            result.append(t.cell_output)
        boundary_already_emitted = any(
            source.get("source_kind") == t.boundary_source
            for source in state["sources"]
        )
        if not boundary_already_emitted:
            result.append(t.boundary_source)
        return result

    def _r1_current_is_last_source_identity(self, state):
        if not self._is_variant("r1_no_boundary_identity"):
            return False
        t = self.t
        current_kind = state["current"].get("source_kind")
        boundary_remains = (
            current_kind != t.boundary_source
            and not any(
                source.get("source_kind") == t.boundary_source
                for source in state["sources"]
            )
        )
        output_identities = set()
        for cell in self._cells_with_available_outputs(state):
            spec = self.t.pin_specs.get(state["cells"][cell]["cell_type"])
            for pin in list(getattr(spec, "outputs", [])) if spec is not None else []:
                if (cell, pin) not in state["source_outputs"]:
                    output_identities.add((cell, pin))
        if current_kind == t.cell_output:
            current_cell = self._ptr(
                state["current"]["source_cell"], t.gate_offset
            )
            current_pin = self._pin_name(state["current"]["source_pin"])
            output_identities.discard((current_cell, current_pin))
        return not boundary_remains and not output_identities

    def _current_is_last_profile_source_slot(self, state):
        """Whether the in-progress source consumes the frozen-bin upper slot.

        At this point the current record has not yet been appended to
        ``state['sources']``.  If it is the last permitted profile slot, its
        fanout must close the residual demand budget; otherwise the grammar can
        reach an all-masked ``source_choice`` dead end and SafeGeneration emits
        PAD before EOS.
        """
        source_range = (
            self._profile_range(state, "source_count_range")
            if self.features.get("profile_feasibility", True) else None
        )
        return bool(
            source_range is not None
            and len(state["sources"]) + 1 >= int(source_range[1])
        )

    def _finish_source_record(self, state):
        t = self.t
        if state["current"].get("source_kind") == t.cell_output:
            cell = self._ptr(state["current"]["source_cell"], t.gate_offset)
            pin = self._pin_name(state["current"]["source_pin"])
            state["source_outputs"].add((cell, pin))
        state["sources"].append(dict(state["current"]))
        state["expect"] = "source_end"

    def _ptr(self, token, offset):
        return int(token) - int(offset)

    def _pin_name(self, token):
        index = self._ptr(token, self.t.pin_offset)
        return self.t.pin_names[index] if 0 <= index < len(self.t.pin_names) else None

    def _cell_name(self, token):
        label = self._ptr(token, self.t.node_type_offset)
        return self.t.label_to_cell.get(label)

    def _is_sequential_cell(self, state, cell_index):
        if not (0 <= int(cell_index) < len(state["cells"])):
            return False
        name = str(state["cells"][int(cell_index)].get("cell_type") or "")
        return name.startswith(("DFF", "SDFF", "LATCH", "DLH", "DLL"))

    def _current_source_cell(self, state):
        return self._source_cell(state, state["load_source"])

    def _source_cell(self, state, source_index):
        source = state["sources"][source_index]
        if source.get("source_kind") != self.t.cell_output:
            return None
        token = source.get("source_cell")
        if token is None or token == self.t.none:
            return None
        cell = self._ptr(token, self.t.gate_offset)
        return cell if 0 <= cell < len(state["cells"]) else None

    def _would_create_combinational_cycle(
        self, state, source_cell, load_cell, adjacency=None
    ):
        if source_cell == load_cell:
            return True
        if (
            self._is_sequential_cell(state, source_cell)
            or self._is_sequential_cell(state, load_cell)
        ):
            return False
        adjacency = (
            state["combinational_adjacency"]
            if adjacency is None else adjacency
        )
        pending = [load_cell]
        seen = set()
        while pending:
            cell = pending.pop()
            if cell == source_cell:
                return True
            if cell in seen:
                continue
            seen.add(cell)
            pending.extend(adjacency.get(cell, ()))
        return False

    @staticmethod
    def _capacitated_bipartite_feasible(
        source_rows, cell_demands, allow_parallel_edges=False
    ):
        """Exact residual max-flow test.

        D0 uses unit source-to-cell capacities (a simple bipartite graph).
        D2 permits distinct pin demands on the same load cell to share a
        source, so the corresponding source-to-cell edge may carry multiple
        units while all source, demand, and topology capacities remain active.
        """
        total = sum(int(value) for value in cell_demands.values())
        if total == 0:
            return True
        if sum(int(capacity) for capacity, _cells in source_rows) < total:
            return False

        cell_ids = sorted(cell_demands)
        source_count = len(source_rows)
        source_node = 0
        first_source = 1
        first_cell = first_source + source_count
        sink_node = first_cell + len(cell_ids)
        graph = [[] for _ in range(sink_node + 1)]

        def add_edge(left, right, capacity):
            graph[left].append([right, int(capacity), len(graph[right])])
            graph[right].append([left, 0, len(graph[left]) - 1])

        for index, (capacity, cells) in enumerate(source_rows):
            node = first_source + index
            add_edge(source_node, node, capacity)
            allowed = set(cells)
            for cell_offset, cell in enumerate(cell_ids):
                if cell in allowed:
                    edge_capacity = (
                        min(int(capacity), int(cell_demands[cell]))
                        if allow_parallel_edges else 1
                    )
                    add_edge(node, first_cell + cell_offset, edge_capacity)
        for cell_offset, cell in enumerate(cell_ids):
            add_edge(first_cell + cell_offset, sink_node, cell_demands[cell])

        flow = 0
        while flow < total:
            level = [-1] * len(graph)
            level[source_node] = 0
            queue = [source_node]
            for node in queue:
                for target, capacity, _reverse in graph[node]:
                    if capacity > 0 and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink_node] < 0:
                return False
            cursor = [0] * len(graph)

            def send(node, amount):
                if node == sink_node:
                    return amount
                while cursor[node] < len(graph[node]):
                    edge = graph[node][cursor[node]]
                    target, capacity, reverse = edge
                    if capacity > 0 and level[target] == level[node] + 1:
                        pushed = send(target, min(amount, capacity))
                        if pushed:
                            edge[1] -= pushed
                            graph[target][reverse][1] += pushed
                            return pushed
                    cursor[node] += 1
                return 0

            while flow < total:
                pushed = send(source_node, total - flow)
                if not pushed:
                    break
                flow += pushed
        return flow == total

    def _topology_residual_feasible(
        self, state, chosen_index, chosen_cell, current_left
    ):
        """Check tail assignment feasibility under topology-safety forbidden edges."""
        remaining_demands = [
            (index, demand["load_cell"])
            for index, demand in enumerate(state["demands"])
            if index not in state["assigned"] and index != chosen_index
        ]
        # Exact flow is only needed in the tail, where topology forbidden edges
        # can invalidate a degree-only Gale-Ryser result.
        if len(remaining_demands) > 32:
            return True

        cell_demands = {}
        for _index, cell in remaining_demands:
            cell_demands[cell] = cell_demands.get(cell, 0) + 1

        adjacency = {
            cell: set(loads)
            for cell, loads in state["combinational_adjacency"].items()
        }
        current_source_cell = self._current_source_cell(state)
        if (
            current_source_cell is not None
            and not self._is_sequential_cell(state, current_source_cell)
            and not self._is_sequential_cell(state, chosen_cell)
        ):
            adjacency.setdefault(current_source_cell, set()).add(chosen_cell)

        source_rows = []
        source_entries = []
        current_source = state["load_source"]
        for source_index, source in enumerate(state["sources"]):
            if source_index in state["emitted_sources"]:
                continue
            if source_index == current_source:
                capacity = int(current_left)
                blocked_cells = (
                    set(state["load_cells"]) | {chosen_cell}
                    if self.features["same_cell_exclusion"] else set()
                )
            else:
                capacity = int(source.get("max_fanout", 0))
                blocked_cells = set()
            if capacity <= 0:
                continue
            source_cell = self._source_cell(state, source_index)
            allowed_cells = []
            for cell in cell_demands:
                if cell in blocked_cells:
                    continue
                if (
                    source_cell is not None
                    and self.features["self_drive_exclusion"]
                    and cell == source_cell
                ):
                    continue
                if (
                    source_cell is not None
                    and self.features["combinational_cycle_exclusion"]
                    and self._would_create_combinational_cycle(
                        state, source_cell, cell, adjacency=adjacency
                    )
                ):
                    continue
                allowed_cells.append(cell)
            source_rows.append((capacity, allowed_cells))
            source_entries.append(
                (source_cell, capacity, frozenset(blocked_cells))
            )
        if not self._capacitated_bipartite_feasible(
            source_rows,
            cell_demands,
            allow_parallel_edges=not self.features["same_cell_exclusion"],
        ):
            return False
        if len(remaining_demands) > 8:
            return True

        # Static flow does not see cycles formed jointly by two future edges.
        # An exact bounded tail search closes that gap without placing a large
        # backtracking cost on the full sequence.
        cell_ids = tuple(sorted(cell_demands))
        initial_counts = tuple(cell_demands[cell] for cell in cell_ids)
        initial_edges = frozenset(
            (source, load)
            for source, loads in adjacency.items()
            for load in loads
        )
        memo = {}

        def would_cycle(edges, source_cell, load_cell):
            if source_cell == load_cell:
                return True
            if (
                self._is_sequential_cell(state, source_cell)
                or self._is_sequential_cell(state, load_cell)
            ):
                return False
            local = {}
            for source, load in edges:
                local.setdefault(source, set()).add(load)
            pending = [load_cell]
            seen = set()
            while pending:
                cell = pending.pop()
                if cell == source_cell:
                    return True
                if cell in seen:
                    continue
                seen.add(cell)
                pending.extend(local.get(cell, ()))
            return False

        def search(source_pos, capacity_left, counts, blocked, edges):
            while source_pos < len(source_entries) and capacity_left == 0:
                source_pos += 1
                if source_pos < len(source_entries):
                    capacity_left = source_entries[source_pos][1]
                    blocked = source_entries[source_pos][2]
            if source_pos == len(source_entries):
                return not any(counts)
            key = (
                source_pos, capacity_left, counts,
                tuple(sorted(blocked)), edges,
            )
            if key in memo:
                return memo[key]
            source_cell = source_entries[source_pos][0]
            for cell_offset, cell in enumerate(cell_ids):
                if counts[cell_offset] <= 0 or cell in blocked:
                    continue
                if (
                    source_cell is not None
                    and self.features["self_drive_exclusion"]
                    and cell == source_cell
                ):
                    continue
                if (
                    source_cell is not None
                    and self.features["combinational_cycle_exclusion"]
                    and would_cycle(edges, source_cell, cell)
                ):
                    continue
                next_counts = list(counts)
                next_counts[cell_offset] -= 1
                next_edges = edges
                if (
                    source_cell is not None
                    and not self._is_sequential_cell(state, source_cell)
                    and not self._is_sequential_cell(state, cell)
                ):
                    next_edges = edges | {(source_cell, cell)}
                next_blocked = (
                    blocked | {cell}
                    if self.features["same_cell_exclusion"] else blocked
                )
                if search(
                    source_pos,
                    capacity_left - 1,
                    tuple(next_counts),
                    next_blocked,
                    next_edges,
                ):
                    memo[key] = True
                    return True
            memo[key] = False
            return False

        if not source_entries:
            return not remaining_demands
        return search(
            0,
            source_entries[0][1],
            initial_counts,
            source_entries[0][2],
            initial_edges,
        )

    def _record_combinational_edge(self, state, source_cell, load_cell):
        if (
            source_cell is None
            or self._is_sequential_cell(state, source_cell)
            or self._is_sequential_cell(state, load_cell)
        ):
            return
        state["combinational_adjacency"].setdefault(source_cell, set()).add(
            load_cell
        )

    def _required_slots(self, state):
        slots = set()
        for index, cell in enumerate(state["cells"]):
            spec = self.t.pin_specs.get(cell["cell_type"])
            for pin in list(getattr(spec, "inputs", [])) if spec is not None else []:
                slots.add((index, str(pin)))
        return slots

    def _cells_with_available_outputs(self, state):
        result = []
        for index, cell in enumerate(state["cells"]):
            spec = self.t.pin_specs.get(cell["cell_type"])
            outputs = list(getattr(spec, "outputs", [])) if spec is not None else []
            if any((index, pin) not in state["source_outputs"] for pin in outputs):
                result.append(index)
        return result

    @staticmethod
    def _bounded_cache_put(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > limit:
            cache.popitem(last=False)

    def _bipartite_graphical(self, source_degrees, cell_degrees):
        rows = tuple(sorted(
            (int(value) for value in source_degrees if int(value) > 0),
            reverse=True,
        ))
        cols = tuple(sorted(
            (int(value) for value in cell_degrees if int(value) > 0),
            reverse=True,
        ))
        key = (rows, cols)
        cached = self._gale_cache.get(key)
        if cached is not None:
            self.cache_stats["gale_hits"] += 1
            self._gale_cache.move_to_end(key)
            return cached
        self.cache_stats["gale_misses"] += 1

        if sum(rows) != sum(cols):
            result = False
        elif rows and rows[0] > len(cols):
            result = False
        elif cols and cols[0] > len(rows):
            result = False
        else:
            # Prefix sums make each Gale-Ryser inequality O(log |cols|)
            # instead of rescanning every column degree.
            ascending = tuple(reversed(cols))
            prefix = [0]
            for degree in ascending:
                prefix.append(prefix[-1] + degree)
            running = 0
            result = True
            for k, value in enumerate(rows, 1):
                running += value
                lo, hi = 0, len(ascending)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if ascending[mid] < k:
                        lo = mid + 1
                    else:
                        hi = mid
                rhs = prefix[lo] + (len(ascending) - lo) * k
                if running > rhs:
                    result = False
                    break

        self._bounded_cache_put(
            self._gale_cache, key, result, self._cache_limit
        )
        return result

    def _mask_indices(self, allowed, device):
        key = (
            device.type,
            device.index,
            tuple(int(token) for token in allowed),
        )
        cached = self._mask_index_cache.get(key)
        if cached is not None:
            self.cache_stats["mask_hits"] += 1
            self._mask_index_cache.move_to_end(key)
            return cached
        self.cache_stats["mask_misses"] += 1
        indices = torch.tensor(key[2], dtype=torch.long, device=device)
        self._bounded_cache_put(
            self._mask_index_cache, key, indices, self._cache_limit
        )
        return indices

    def _advance(self, state, token):
        t = self.t
        token = int(token)
        expect = state["expect"]
        singleton = {
            **({
                "target_profile_begin": ("profile_mode_tag", t.target_profile_begin),
                "profile_mode_tag": ("profile_mode_value", t.profile_mode),
                "cell_count_bin_tag": ("cell_count_bin_value", t.cell_count_bin),
                "demand_count_bin_tag": ("demand_count_bin_value", t.demand_count_bin),
                "source_count_bin_tag": ("source_count_bin_value", t.source_count_bin),
                "seq_ratio_bin_tag": ("seq_ratio_bin_value", t.seq_ratio_bin),
                "pin_complexity_bin_tag": ("pin_complexity_bin_value", t.pin_complexity_bin),
                "complex_cell_flags_tag": ("rare_cell_flag_tag", t.complex_cell_flags),
                "rare_cell_flag_tag": ("rare_cell_flag_value", t.rare_cell_enriched),
                "has_mux_flag_tag": ("has_mux_flag_value", t.has_mux),
                "has_aoi_oai_flag_tag": ("has_aoi_oai_flag_value", t.has_aoi_oai),
                "has_multi_output_flag_tag": ("has_multi_output_flag_value", t.has_multi_output_cell),
                "target_profile_end": ("cell_section_begin", t.target_profile_end),
            } if getattr(t, "include_target_profile", False) else {}),
            "cell_section_begin": ("cell_choice", t.cell_section_begin),
            "cell_id_tag": ("cell_ptr", t.cell_id),
            "cell_type_tag": ("cell_type_value", t.cell_type),
            "cell_end": ("cell_choice", t.cell_end),
            "demand_section_begin": ("demand_choice", t.demand_section_begin),
            "demand_id_tag": ("demand_ptr", t.demand_id),
            "load_cell_tag": ("load_cell_value", t.load_cell),
            "load_pin_tag": ("load_pin_value", t.load_pin),
            "demand_end": ("demand_choice", t.demand_end),
            "source_section_begin": ("source_choice", t.source_section_begin),
            "source_id_tag": ("source_ptr", t.source_id),
            "source_kind_tag": ("source_kind_value", t.source_kind),
            "source_cell_tag": ("source_cell_value", t.source_cell),
            "source_pin_tag": ("source_pin_value", t.source_pin),
            "boundary_pin_tag": ("boundary_pin_value", t.boundary_pin_id),
            "source_role_tag": ("source_role_value", t.source_role),
            "fanout_bucket_tag": ("fanout_bucket_value", t.fanout_bucket),
            "max_fanout_tag": ("max_fanout_value", t.max_fanout),
            "used_fanout_tag": ("used_fanout_value", t.used_fanout),
            "remaining_fanout_tag": ("remaining_fanout_value", t.remaining_fanout),
            "source_end": ("source_choice", t.source_end),
            "source_net_section_begin": ("load_choice", t.source_net_section_begin),
            "count_tag": ("load_count_value", t.count_token),
            "load_end": ("load_choice", t.source_load_end),
            "eos": ("done", t.eos),
        }
        if expect in singleton:
            next_expect, required = singleton[expect]
            if token != int(required):
                state["expect"] = "invalid"
                return
            state["expect"] = next_expect
            return

        if expect == "profile_mode_value":
            if token == t.profile_free:
                state["target_profile"] = {"profile_mode": "PROFILE_FREE"}
                state["expect"] = "target_profile_end"
            elif token == t.profile_coarse:
                state["target_profile"] = {"profile_mode": "PROFILE_COARSE"}
                state["expect"] = "cell_count_bin_tag"
            else:
                state["expect"] = "invalid"
        elif expect == "cell_count_bin_value":
            state["target_profile"]["cell_count_range"] = self._profile_bin_range(token, "cell_count")
            state["expect"] = "demand_count_bin_tag"
        elif expect == "demand_count_bin_value":
            state["target_profile"]["demand_count_range"] = self._profile_bin_range(token, "demand_count")
            state["expect"] = "source_count_bin_tag"
        elif expect == "source_count_bin_value":
            state["target_profile"]["source_count_range"] = self._profile_bin_range(token, "source_count")
            state["expect"] = "seq_ratio_bin_tag"
        elif expect == "seq_ratio_bin_value":
            state["target_profile"]["seq_ratio_range"] = self._profile_bin_range(token, "seq_ratio")
            state["expect"] = "pin_complexity_bin_tag"
        elif expect == "pin_complexity_bin_value":
            state["target_profile"]["pin_complexity_range"] = self._profile_bin_range(token, "pin_complexity")
            state["expect"] = "complex_cell_flags_tag"
        elif expect in {
            "rare_cell_flag_value", "has_mux_flag_value",
            "has_aoi_oai_flag_value", "has_multi_output_flag_value",
        }:
            if token not in {t.bool_false, t.bool_true}:
                state["expect"] = "invalid"
            else:
                mapping = {
                    "rare_cell_flag_value": ("rare_cell_enriched", "has_mux_flag_tag"),
                    "has_mux_flag_value": ("has_mux", "has_aoi_oai_flag_tag"),
                    "has_aoi_oai_flag_value": ("has_aoi_oai", "has_multi_output_flag_tag"),
                    "has_multi_output_flag_value": ("has_multi_output_cell", "target_profile_end"),
                }
                key, next_state = mapping[expect]
                state["target_profile"][key] = token == t.bool_true
                state["expect"] = next_state
        elif expect == "cell_choice":
            if token == t.cell_begin:
                state["current"] = {}
                state["expect"] = "cell_id_tag"
            elif (
                token == t.cell_section_end
                and state["cells"]
                and self._cell_profile_satisfied(state)
            ):
                state["expect"] = "demand_section_begin"
            else:
                state["expect"] = "invalid"
        elif expect == "cell_ptr":
            value = self._ptr(token, t.gate_offset)
            if value != len(state["cells"]):
                state["expect"] = "invalid"
            else:
                state["current"]["cell_id"] = value
                state["expect"] = "cell_type_tag"
        elif expect == "cell_type_value":
            cell_type = self._cell_name(token)
            if cell_type is None:
                state["expect"] = "invalid"
            else:
                state["current"]["cell_type"] = cell_type
                feature = self._cell_profile_features(cell_type)
                state["sequential_cell_count"] += int(feature["sequential"])
                state["rare_cell_count"] += int(feature["rare"])
                state["max_input_arity"] = max(
                    state["max_input_arity"], feature["input_arity"]
                )
                state["has_mux"] = state["has_mux"] or feature["mux"]
                state["has_aoi_oai"] = state["has_aoi_oai"] or feature["aoi_oai"]
                state["has_multi_output_cell"] = (
                    state["has_multi_output_cell"] or feature["multi_output"]
                )
                state["cells"].append(dict(state["current"]))
                state["expect"] = "cell_end"
        elif expect == "demand_choice":
            if token == t.demand_begin:
                state["current"] = {}
                state["expect"] = "demand_id_tag"
            elif (
                token == t.demand_section_end
                and state["demands"]
                and self._demands_complete(state)
            ):
                state["expect"] = "source_section_begin"
            else:
                state["expect"] = "invalid"
        elif expect == "demand_ptr":
            value = self._ptr(token, t.demand_offset)
            if value != len(state["demands"]):
                state["expect"] = "invalid"
            else:
                state["current"]["demand_id"] = value
                state["expect"] = "load_cell_tag"
        elif expect == "load_cell_value":
            value = self._ptr(token, t.gate_offset)
            if not (0 <= value < len(state["cells"])):
                state["expect"] = "invalid"
            else:
                state["current"]["load_cell"] = value
                if self._is_variant("r4_cell_level_demand"):
                    state["cell_demand_counts"][value] = (
                        state["cell_demand_counts"].get(value, 0) + 1
                    )
                    state["demands"].append(dict(state["current"]))
                    state["expect"] = "demand_end"
                else:
                    state["expect"] = "load_pin_tag"
        elif expect == "load_pin_value":
            pin = self._pin_name(token)
            slot = (state["current"].get("load_cell"), pin)
            if pin is None or slot in state["slots"]:
                state["expect"] = "invalid"
            else:
                state["current"]["load_pin"] = pin
                state["slots"].add(slot)
                state["demands"].append(dict(state["current"]))
                state["expect"] = "demand_end"
        elif expect == "source_choice":
            if token == t.source_begin:
                state["current"] = {}
                state["expect"] = "source_id_tag"
            elif (
                token == t.source_section_end
                and self._source_table_complete(state)
                and self._scaffold_source_table_complete(state)
            ):
                state["expect"] = "source_net_section_begin"
            else:
                state["expect"] = "invalid"
        elif expect == "source_ptr":
            value = self._ptr(token, t.source_offset)
            if value != len(state["sources"]):
                state["expect"] = "invalid"
            else:
                state["current"]["source_id"] = value
                state["expect"] = "source_kind_tag"
        elif expect == "source_kind_value":
            state["current"]["source_kind"] = token
            state["expect"] = "source_cell_tag"
        elif expect == "source_cell_value":
            state["current"]["source_cell"] = token
            state["expect"] = "source_pin_tag"
        elif expect == "source_pin_value":
            state["current"]["source_pin"] = token
            state["expect"] = "boundary_pin_tag"
        elif expect == "boundary_pin_value":
            state["current"]["boundary_pin"] = token
            state["expect"] = "source_role_tag"
        elif expect == "source_role_value":
            state["current"]["source_role"] = token
            if self._is_variant("r2_no_per_source_budget"):
                self._finish_source_record(state)
            else:
                state["expect"] = "fanout_bucket_tag"
        elif expect == "fanout_bucket_value":
            state["current"]["fanout_bucket"] = token
            state["expect"] = "max_fanout_tag"
        elif expect == "max_fanout_value":
            value = self._ptr(token, t.count_offset)
            state["current"]["max_fanout"] = value
            state["expect"] = "used_fanout_tag"
        elif expect == "used_fanout_value":
            value = self._ptr(token, t.count_offset)
            if value != state["current"].get("max_fanout"):
                state["expect"] = "invalid"
            else:
                state["current"]["used_fanout"] = value
                state["expect"] = "remaining_fanout_tag"
        elif expect == "remaining_fanout_value":
            value = self._ptr(token, t.count_offset)
            if value != 0:
                state["expect"] = "invalid"
            else:
                self._finish_source_record(state)
        elif expect == "load_choice":
            if (
                self._is_variant("r3_no_full_load_assignment")
                and token == t.source_net_section_end
            ):
                state["expect"] = "eos"
            elif token == t.source_load_begin and len(state["emitted_sources"]) < len(state["sources"]):
                state["expect"] = "load_source_value"
            elif (
                token == t.source_net_section_end
                and (
                    not self.features["completion_eos"]
                    or (
                        len(state["emitted_sources"]) == len(state["sources"])
                        and len(state["assigned"]) == len(state["demands"])
                    )
                )
            ):
                state["expect"] = "eos"
            else:
                state["expect"] = "invalid"
        elif expect == "load_source_value":
            value = self._ptr(token, t.source_offset)
            expected = len(state["emitted_sources"])
            if value != expected or value >= len(state["sources"]):
                state["expect"] = "invalid"
            else:
                state["load_source"] = value
                state["load_cells"] = set()
                state["expect"] = "count_tag"
        elif expect == "load_count_value":
            value = self._ptr(token, t.count_offset)
            source = state["sources"][state["load_source"]]
            expected = source.get("max_fanout")
            remaining = len(state["demands"]) - len(state["assigned"])
            if (
                value < 0
                or value > remaining
                or (
                    self.features["source_budget"]
                    and expected is not None
                    and value != expected
                )
                or (
                    self._is_variant("r2_no_per_source_budget")
                    and state["load_source"] == len(state["sources"]) - 1
                    and value != remaining
                )
            ):
                state["expect"] = "invalid"
            else:
                if expected is None:
                    source["max_fanout"] = value
                state["load_left"] = value
                if value:
                    state["expect"] = "load_demand_value"
                else:
                    state["emitted_sources"].add(state["load_source"])
                    state["expect"] = "load_end"
        elif expect == "load_demand_value":
            value = self._ptr(token, t.demand_offset)
            if (
                not (0 <= value < len(state["demands"]))
                or (
                    self.features["demand_exactly_once"]
                    and value in state["assigned"]
                )
            ):
                state["expect"] = "invalid"
                return
            load_cell = state["demands"][value]["load_cell"]
            source_cell = self._current_source_cell(state)
            if (
                source_cell is not None
                and (
                    (
                        self.features["self_drive_exclusion"]
                        and source_cell == load_cell
                    )
                    or (
                        self.features["combinational_cycle_exclusion"]
                        and self._would_create_combinational_cycle(
                            state, source_cell, load_cell
                        )
                    )
                )
            ):
                state["expect"] = "invalid"
                return
            if (
                self.features["same_cell_exclusion"]
                and
                load_cell in state["load_cells"]
                and not self._is_variant("r1_no_boundary_identity")
            ):
                state["expect"] = "invalid"
                return
            state["assigned"].add(value)
            state["load_cells"].add(load_cell)
            self._record_combinational_edge(state, source_cell, load_cell)
            state["load_left"] -= 1
            if state["load_left"] == 0:
                state["emitted_sources"].add(state["load_source"])
                state["expect"] = "load_end"
        elif expect == "done":
            pass
        else:
            state["expect"] = "invalid"

    def _allowed(self, state):
        t = self.t
        expect = state["expect"]
        singleton = {
            **({
                "target_profile_begin": [t.target_profile_begin],
                "profile_mode_tag": [t.profile_mode],
                "cell_count_bin_tag": [t.cell_count_bin],
                "demand_count_bin_tag": [t.demand_count_bin],
                "source_count_bin_tag": [t.source_count_bin],
                "seq_ratio_bin_tag": [t.seq_ratio_bin],
                "pin_complexity_bin_tag": [t.pin_complexity_bin],
                "complex_cell_flags_tag": [t.complex_cell_flags],
                "rare_cell_flag_tag": [t.rare_cell_enriched],
                "has_mux_flag_tag": [t.has_mux],
                "has_aoi_oai_flag_tag": [t.has_aoi_oai],
                "has_multi_output_flag_tag": [t.has_multi_output_cell],
                "target_profile_end": [t.target_profile_end],
            } if getattr(t, "include_target_profile", False) else {}),
            "cell_section_begin": [t.cell_section_begin],
            "cell_id_tag": [t.cell_id],
            "cell_type_tag": [t.cell_type],
            "cell_end": [t.cell_end],
            "demand_section_begin": [t.demand_section_begin],
            "demand_id_tag": [t.demand_id],
            "load_cell_tag": [t.load_cell],
            "load_pin_tag": [t.load_pin],
            "demand_end": [t.demand_end],
            "source_section_begin": [t.source_section_begin],
            "source_id_tag": [t.source_id],
            "source_kind_tag": [t.source_kind],
            "source_cell_tag": [t.source_cell],
            "source_pin_tag": [t.source_pin],
            "boundary_pin_tag": [t.boundary_pin_id],
            "source_role_tag": [t.source_role],
            "fanout_bucket_tag": [t.fanout_bucket],
            "max_fanout_tag": [t.max_fanout],
            "used_fanout_tag": [t.used_fanout],
            "remaining_fanout_tag": [t.remaining_fanout],
            "source_end": [t.source_end],
            "source_net_section_begin": [t.source_net_section_begin],
            "count_tag": [t.count_token],
            "load_end": [t.source_load_end],
            "eos": [t.eos],
            "done": [t.pad],
            "invalid": [t.eos],
        }
        if expect in singleton:
            return singleton[expect]
        if expect == "profile_mode_value":
            return [t.profile_free, t.profile_coarse]
        if expect in {
            "cell_count_bin_value", "demand_count_bin_value",
            "source_count_bin_value", "seq_ratio_bin_value",
            "pin_complexity_bin_value",
        }:
            field = expect.removesuffix("_bin_value")
            active = set(t.profile_config.get("active_profile_bins", {}).get(
                field, ["PROFILE_BIN_0", "PROFILE_BIN_1", "PROFILE_BIN_2", "PROFILE_BIN_3"]
            ))
            return [
                token for token in (
                    t.profile_bin_0, t.profile_bin_1, t.profile_bin_2, t.profile_bin_3
                ) if t.special_toks[token] in active
            ]
        if expect in {
            "rare_cell_flag_value", "has_mux_flag_value",
            "has_aoi_oai_flag_value", "has_multi_output_flag_value",
        }:
            return [t.bool_false, t.bool_true]
        if expect == "cell_choice":
            target_max = self._target_max_cells(state)
            can_add_cell = len(state["cells"]) < t.max_num_nodes
            if target_max is not None:
                can_add_cell = can_add_cell and len(state["cells"]) < int(target_max)
            result = [t.cell_begin] if can_add_cell else []
            target_min = self._target_min_cells(state)
            can_end_cells = bool(state["cells"])
            if target_min is not None:
                can_end_cells = can_end_cells and len(state["cells"]) >= int(target_min)
            can_end_cells = can_end_cells and self._cell_profile_satisfied(state)
            if can_end_cells:
                result.append(t.cell_section_end)
            return result
        if expect == "cell_ptr":
            if not self.features["pointer_mask"]:
                return list(range(t.gate_offset, t.gate_offset + t.max_num_nodes))
            return [t.gate_offset + len(state["cells"])]
        if expect == "cell_type_value":
            candidates = list(range(t.node_type_offset, len(t)))
            return [
                token for token in candidates
                if self._candidate_profile_feasible(state, self._cell_name(token))
            ]
        if expect == "demand_choice":
            complete = self._demands_complete(state)
            result = (
                [t.demand_begin]
                if len(state["demands"]) < t.max_num_nodes and not complete
                else []
            )
            if state["demands"] and complete:
                result.append(t.demand_section_end)
            return result
        if expect == "demand_ptr":
            if not self.features["pointer_mask"]:
                return list(range(t.demand_offset, t.demand_offset + t.max_num_nodes))
            return [t.demand_offset + len(state["demands"])]
        if expect == "load_cell_value":
            required = self._required_slots(state)
            if self._is_variant("r4_cell_level_demand"):
                required_by_cell = {}
                for cell, _pin in required:
                    required_by_cell[cell] = required_by_cell.get(cell, 0) + 1
                open_cells = sorted(
                    cell for cell, count in required_by_cell.items()
                    if state["cell_demand_counts"].get(cell, 0) < count
                )
            else:
                open_cells = sorted({cell for cell, pin in required - state["slots"]})
            return [t.gate_offset + index for index in open_cells]
        if expect == "load_pin_value":
            cell = state["current"].get("load_cell")
            cell_type = state["cells"][cell]["cell_type"]
            spec = t.pin_specs.get(cell_type)
            pins = list(getattr(spec, "inputs", [])) if spec is not None else t.pin_names
            return [
                t.pin_offset + t.pin_name_to_id[pin]
                for pin in pins
                if pin in t.pin_name_to_id and (cell, pin) not in state["slots"]
            ]
        if expect == "source_choice":
            complete = self._source_table_complete(state)
            if complete and not self._scaffold_source_table_complete(state):
                complete = False
            identity_available = (
                bool(self._r1_source_kinds_available(state))
                if self._is_variant("r1_no_boundary_identity")
                else True
            )
            source_range = (
                self._profile_range(state, "source_count_range")
                if self.features.get("profile_feasibility", True) else None
            )
            below_target_sources = (
                source_range is None or len(state["sources"]) < source_range[1]
            )
            result = (
                [t.source_begin]
                if (
                    len(state["sources"]) < t.max_num_nodes
                    and below_target_sources
                    and (
                        not complete
                        or self._is_variant("r2_no_per_source_budget")
                    )
                    and identity_available
                )
                else []
            )
            if complete:
                result.append(t.source_section_end)
            return result
        if expect == "source_ptr":
            if not self.features["pointer_mask"]:
                return list(range(t.source_offset, t.source_offset + t.max_num_nodes))
            return [t.source_offset + len(state["sources"])]
        if expect == "source_kind_value":
            if self._is_variant("r1_no_boundary_identity"):
                return self._r1_source_kinds_available(state)
            result = []
            source_range = self._profile_range(state, "source_count_range")
            remaining_source_slots = (
                source_range[1] - len(state["sources"])
                if source_range is not None else None
            )
            missing_outputs = self._missing_output_identity_count(state)
            # Every generated cell must retain at least one output-source
            # identity.  Once all remaining profile slots are reserved by
            # missing outputs, a boundary source would make completion
            # impossible and must not be offered.
            must_emit_output = (
                remaining_source_slots is not None
                and missing_outputs >= remaining_source_slots
            )
            target_boundary = self._target_boundary_source_count(state)
            if (
                not must_emit_output
                and (
                    target_boundary is None
                    or self._boundary_source_count(state) < int(target_boundary)
                )
            ):
                result.append(t.boundary_source)
            if self._cells_with_available_outputs(state):
                result.insert(0, t.cell_output)
            return result
        if expect == "source_cell_value":
            kind = state["current"].get("source_kind")
            if kind == t.cell_output:
                cells = self._cells_with_available_outputs(state)
                source_range = self._profile_range(state, "source_count_range")
                remaining_source_slots = (
                    source_range[1] - len(state["sources"])
                    if source_range is not None else None
                )
                missing_outputs = self._missing_output_identity_count(state)
                if (
                    remaining_source_slots is not None
                    and missing_outputs >= remaining_source_slots
                ):
                    covered = {cell for cell, _pin in state["source_outputs"]}
                    cells = [cell for cell in cells if cell not in covered]
                return [
                    t.gate_offset + index
                    for index in cells
                ]
            return [t.none]
        if expect == "source_pin_value":
            kind = state["current"].get("source_kind")
            if kind == t.cell_output:
                cell_token = state["current"].get("source_cell")
                cell = self._ptr(cell_token, t.gate_offset)
                spec = t.pin_specs.get(state["cells"][cell]["cell_type"])
                pins = list(getattr(spec, "outputs", [])) if spec is not None else t.pin_names
                return [
                    t.pin_offset + t.pin_name_to_id[p]
                    for p in pins
                    if p in t.pin_name_to_id and (cell, p) not in state["source_outputs"]
                ]
            special = "__BOUNDARY__" if kind == t.boundary_source else "__UNKNOWN__"
            return [t.pin_offset + t.pin_name_to_id[special]]
        if expect == "boundary_pin_value":
            if self._is_variant("r1_no_boundary_identity"):
                return [t.none]
            if state["current"].get("source_kind") == t.boundary_source:
                return [t.source_offset + len(state["sources"])]
            return [t.none]
        if expect == "source_role_value":
            kind = state["current"].get("source_kind")
            return [{
                t.cell_output: t.output_pin,
                t.boundary_source: t.boundary_pin,
                t.constant_source: t.const_pin,
            }[kind]]
        if expect == "fanout_bucket_value":
            remaining = len(state["demands"]) - sum(
                row["max_fanout"] for row in state["sources"]
            )
            if (
                self._r1_current_is_last_source_identity(state)
                or self._current_is_last_profile_source_slot(state)
            ):
                if remaining == 0:
                    return [t.fanout_0]
                if remaining == 1:
                    return [t.fanout_1]
                if remaining <= 4:
                    return [t.fanout_2_4]
                if remaining <= 8:
                    return [t.fanout_5_8]
                if remaining <= 16:
                    return [t.fanout_9_16]
                return [t.fanout_17_plus]
            result = [t.fanout_0]
            if remaining >= 1:
                result.append(t.fanout_1)
            if remaining >= 2:
                result.append(t.fanout_2_4)
            if remaining >= 5:
                result.append(t.fanout_5_8)
            if remaining >= 9:
                result.append(t.fanout_9_16)
            if remaining >= 17:
                result.append(t.fanout_17_plus)
            return result
        if expect == "max_fanout_value":
            remaining = len(state["demands"]) - sum(
                row["max_fanout"] for row in state["sources"]
            )
            if (
                self._r1_current_is_last_source_identity(state)
                or self._current_is_last_profile_source_slot(state)
            ):
                return [t.count_offset + remaining]
            if not self.features["source_budget"]:
                return list(range(t.count_offset, t.count_offset + min(t.max_num_nodes, len(state["demands"])) + 1))
            bucket = state["current"].get("fanout_bucket")
            ranges = {
                t.fanout_0: (0, 0), t.fanout_1: (1, 1), t.fanout_2_4: (2, 4),
                t.fanout_5_8: (5, 8), t.fanout_9_16: (9, 16),
                t.fanout_17_plus: (17, remaining),
            }
            low, high = ranges[bucket]
            cell_capacity = (
                remaining
                if (
                    self._is_variant("r1_no_boundary_identity")
                    or not self.features["same_cell_exclusion"]
                )
                else len(state["cells"])
            )
            high = min(high, remaining, cell_capacity)
            demand_by_cell = {}
            for demand in state["demands"]:
                cell = demand["load_cell"]
                demand_by_cell[cell] = demand_by_cell.get(cell, 0) + 1
            existing = [row["max_fanout"] for row in state["sources"]]
            values = []
            for value in range(low, high + 1):
                if (
                    self.features["gale_ryser"]
                    # Gale-Ryser is a simple-bipartite graphicality test.  Once
                    # D2 permits one source to drive multiple pins of the same
                    # cell, its per-cell degree bound is no longer applicable
                    # and must not indirectly re-enable same-cell exclusion.
                    and self.features["same_cell_exclusion"]
                    and not self._is_variant("r1_no_boundary_identity")
                    and value == remaining
                    and not self._bipartite_graphical(
                    existing + [value], demand_by_cell.values()
                    )
                ):
                    continue
                values.append(t.count_offset + value)
            return values
        if expect == "used_fanout_value":
            return [t.count_offset + state["current"]["max_fanout"]]
        if expect == "remaining_fanout_value":
            return [t.count_offset]
        if expect == "load_choice":
            if self._is_variant("r3_no_full_load_assignment"):
                return [t.source_net_section_end]
            result = []
            if len(state["emitted_sources"]) < len(state["sources"]):
                result.append(t.source_load_begin)
            if not self.features["completion_eos"] or (
                len(state["emitted_sources"]) == len(state["sources"])
                and len(state["assigned"]) == len(state["demands"])
            ):
                result.append(t.source_net_section_end)
            return result
        if expect == "load_source_value":
            if not self.features["pointer_mask"]:
                return list(range(t.source_offset, t.source_offset + t.max_num_nodes))
            return [t.source_offset + len(state["emitted_sources"])]
        if expect == "load_count_value":
            source = state["sources"][state["load_source"]]
            if self._is_variant("r2_no_per_source_budget"):
                remaining = len(state["demands"]) - len(state["assigned"])
                remaining_by_cell = {}
                for index, demand in enumerate(state["demands"]):
                    if index not in state["assigned"]:
                        cell = demand["load_cell"]
                        remaining_by_cell[cell] = (
                            remaining_by_cell.get(cell, 0) + 1
                        )
                remaining_cell_count = len(remaining_by_cell)
                if not self.r2_future_feasibility:
                    if state["load_source"] == len(state["sources"]) - 1:
                        values = (
                            [remaining]
                            if remaining <= remaining_cell_count
                            else []
                        )
                    else:
                        values = range(0, min(remaining, remaining_cell_count) + 1)
                    return [t.count_offset + value for value in values]
                future_source_count = (
                    len(state["sources"])
                    - len(state["emitted_sources"])
                    - 1
                )
                if any(
                    count > future_source_count + 1
                    for count in remaining_by_cell.values()
                ):
                    return []
                minimum_count = sum(
                    count > future_source_count
                    for count in remaining_by_cell.values()
                )
                if state["load_source"] == len(state["sources"]) - 1:
                    values = (
                        [remaining]
                        if remaining <= remaining_cell_count
                        else []
                    )
                else:
                    values = range(
                        minimum_count,
                        min(remaining, remaining_cell_count) + 1,
                    )
                return [t.count_offset + value for value in values]
            if not self.features["source_budget"]:
                remaining = len(state["demands"]) - len(state["assigned"])
                return list(range(t.count_offset, t.count_offset + remaining + 1))
            return [t.count_offset + source["max_fanout"]]
        if expect == "load_demand_value":
            eligible = []
            remaining_by_cell = {}
            source_cell = self._current_source_cell(state)
            for index, demand in enumerate(state["demands"]):
                cell = demand["load_cell"]
                if index not in state["assigned"]:
                    remaining_by_cell[cell] = remaining_by_cell.get(cell, 0) + 1
                if (
                    (self.features["demand_exactly_once"] and index in state["assigned"])
                    or (
                        self.features["same_cell_exclusion"]
                        and not self._is_variant("r1_no_boundary_identity")
                        and cell in state["load_cells"]
                    )
                    or (
                        source_cell is not None
                        and self.features["self_drive_exclusion"]
                        and cell == source_cell
                    )
                    or (
                        source_cell is not None
                        and self.features["combinational_cycle_exclusion"]
                        and self._would_create_combinational_cycle(
                            state, source_cell, cell
                        )
                    )
                ):
                    continue
                eligible.append((index, cell))

            allowed = []
            if (
                not self.features["gale_ryser"]
                or not self.features["same_cell_exclusion"]
                or self._is_variant("r1_no_boundary_identity")
                or self._is_variant("r2_no_per_source_budget")
            ):
                if (
                    self._is_variant("r2_no_per_source_budget")
                    and self.r2_future_feasibility
                ):
                    current_left = state["load_left"] - 1
                    future_source_count = (
                        len(state["sources"])
                        - len(state["emitted_sources"])
                        - 1
                    )
                    for index, chosen_cell in eligible:
                        feasible = True
                        next_load_cells = set(state["load_cells"])
                        next_load_cells.add(chosen_cell)
                        after_by_cell = {}
                        for cell, count in remaining_by_cell.items():
                            after = count - int(cell == chosen_cell)
                            after_by_cell[cell] = after
                            current_capacity = int(
                                current_left > 0 and cell not in next_load_cells
                            )
                            if after > future_source_count + current_capacity:
                                feasible = False
                                break
                        required_current_cells = sum(
                            count > future_source_count
                            for count in after_by_cell.values()
                        )
                        if required_current_cells > current_left:
                            feasible = False
                        if feasible:
                            allowed.append(t.demand_offset + index)
                else:
                    current_left = state["load_left"] - 1
                    for index, cell in eligible:
                        feasible = True
                        if (
                            self.features["self_drive_exclusion"]
                            or self.features["combinational_cycle_exclusion"]
                        ):
                            feasible = self._topology_residual_feasible(
                                state,
                                chosen_index=index,
                                chosen_cell=cell,
                                current_left=current_left,
                            )
                        if feasible:
                            allowed.append(t.demand_offset + index)
            else:
                current_left = state["load_left"] - 1
                future_degrees = [
                    source["max_fanout"]
                    for source_index, source in enumerate(state["sources"])
                    if source_index not in state["emitted_sources"]
                    and source_index != state["load_source"]
                ]
                residual_degrees = (
                    ([current_left] if current_left else []) + future_degrees
                )
                feasibility_by_degree = {}
                base_cell_degrees = list(remaining_by_cell.values())
                for index, cell in eligible:
                    degree = remaining_by_cell.get(cell, 0)
                    decrement = index not in state["assigned"]
                    signature = (degree, decrement)
                    feasible = feasibility_by_degree.get(signature)
                    if feasible is None:
                        after = list(base_cell_degrees)
                        if decrement:
                            after[after.index(degree)] -= 1
                        feasible = self._bipartite_graphical(
                            residual_degrees, after
                        )
                        feasibility_by_degree[signature] = feasible
                    if (
                        feasible
                        and (
                            self.features["self_drive_exclusion"]
                            or self.features["combinational_cycle_exclusion"]
                        )
                    ):
                        feasible = self._topology_residual_feasible(
                            state,
                            chosen_index=index,
                            chosen_cell=cell,
                            current_left=current_left,
                        )
                    if feasible:
                        allowed.append(t.demand_offset + index)
            if not self.features["pointer_mask"]:
                allowed.extend(range(t.demand_offset, t.demand_offset + t.max_num_nodes))
                allowed = sorted(set(allowed))
            return allowed
        return [t.eos]

    def __call__(self, input_ids, scores):
        for row in range(input_ids.shape[0]):
            state = self.states[row]
            start = self.processed[row]
            for token in input_ids[row, start:].tolist():
                self._advance(state, token)
            self.processed[row] = int(input_ids.shape[1])
            allowed = self._allowed(state)
            if not allowed:
                scores[row].fill_(-float("inf"))
                continue
            indices = self._mask_indices(allowed, scores.device)
            values = scores[row].index_select(0, indices).clone()
            scores[row].fill_(-float("inf"))
            scores[row].index_copy_(0, indices, values)
        return scores


class SafeGenerationLogits(LogitsProcessor):
    """Sanitize grammar-masked logits before sampled generation.

    Practical generation should terminate a dead-end partial sample, not pass
    NaN/Inf/all--inf logits to torch multinomial where CUDA raises a device-side
    assert. This processor must be last in the generation processor chain.
    """

    def __init__(self, tokenizer):
        self.eos = int(getattr(tokenizer, 'eos', 0))
        self.pad = int(getattr(tokenizer, 'pad', self.eos))

    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=float('-inf'), posinf=1.0e4, neginf=float('-inf'))
        finite_rows = torch.isfinite(scores).any(dim=-1)
        if not bool(finite_rows.all().item()):
            fallback = torch.full_like(scores, float('-inf'))
            if 0 <= self.eos < int(scores.shape[-1]):
                fallback[:, self.eos] = 0.0
            if 0 <= self.pad < int(scores.shape[-1]):
                fallback[:, self.pad] = 0.0
            scores = torch.where(finite_rows.unsqueeze(1), scores, fallback)
        return scores


class SourceNetV5Grammar(LogitsProcessor):
    """Light section-order grammar for Source-Net V5.

    This deliberately enforces only coarse V5.0 constraints.  Strong demand
    consumption, duplicate-sink, fanout-budget and consistency checks are left
    disabled for V5.1-style stateful decoding.
    """

    def __init__(self, tokenizer, batch_size: int, device, mask_mode: str = "source_net_v5_light"):
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.device = device
        self.mask_mode = str(mask_mode or "source_net_v5_light")
        self.vocab_size = len(tokenizer)
        self.section_order = [
            int(tokenizer.cond_section_begin),
            int(tokenizer.demand_section_begin),
            int(tokenizer.source_budget_section_begin),
            int(tokenizer.cell_section_begin),
            int(tokenizer.sink_demand_section_begin),
            int(tokenizer.source_net_section_begin),
        ]
        self.section_end = {
            int(tokenizer.cond_section_begin): int(tokenizer.cond_section_end),
            int(tokenizer.demand_section_begin): int(tokenizer.demand_section_end),
            int(tokenizer.source_budget_section_begin): int(tokenizer.source_budget_section_end),
            int(tokenizer.cell_section_begin): int(tokenizer.cell_section_end),
            int(tokenizer.sink_demand_section_begin): int(tokenizer.sink_demand_section_end),
            int(tokenizer.source_net_section_begin): int(tokenizer.source_net_section_end),
        }
        self.begin_to_idx = {tok: idx for idx, tok in enumerate(self.section_order)}

    def _opened_sections(self, toks):
        opened = []
        closed = set()
        for tok in toks:
            if tok in self.section_order:
                opened.append(tok)
            for begin, end in self.section_end.items():
                if tok == end:
                    closed.add(begin)
        return opened, closed

    def _record_state(self, toks, section_begin, section_end):
        in_section = False
        begin_count = 0
        end_count = 0
        for tok in toks:
            tok = int(tok)
            if tok == int(section_begin):
                in_section = True
                continue
            if tok == int(section_end):
                in_section = False
                continue
            if not in_section:
                continue
            if tok == int(self.tokenizer.demand_begin):
                begin_count += 1
            elif tok == int(self.tokenizer.demand_end):
                end_count += 1
        return {
            "begin_count": int(begin_count),
            "end_count": int(end_count),
            "open_record": int(begin_count) > int(end_count),
        }

    def _expected_count_for_section(self, current):
        if current == int(self.tokenizer.demand_section_begin):
            return int(getattr(self.tokenizer, "source_net_v5_expected_demand_records", -1) or -1)
        if current == int(self.tokenizer.sink_demand_section_begin):
            return int(getattr(self.tokenizer, "source_net_v5_expected_sink_demand_records", -1) or -1)
        if current == int(self.tokenizer.cell_section_begin):
            return int(getattr(self.tokenizer, "source_net_v5_expected_cell_records", -1) or -1)
        if current == int(self.tokenizer.source_net_section_begin):
            return int(getattr(self.tokenizer, "source_net_v5_expected_source_net_records", -1) or -1)
        return -1

    def _marker_state(self, toks, section_begin, section_end, record_begin, record_end=None):
        in_section = False
        begin_count = 0
        end_count = 0
        for tok in toks:
            tok = int(tok)
            if tok == int(section_begin):
                in_section = True
                continue
            if tok == int(section_end):
                in_section = False
                continue
            if not in_section:
                continue
            if tok == int(record_begin):
                begin_count += 1
            elif record_end is not None and tok == int(record_end):
                end_count += 1
        return {
            "begin_count": int(begin_count),
            "end_count": int(end_count),
            "open_record": bool(record_end is not None and int(begin_count) > int(end_count)),
        }

    def _allow(self, scores, b: int, allowed):
        mask = torch.full_like(scores[b], float("-inf"))
        ids = [int(x) for x in allowed if 0 <= int(x) < scores.shape[-1]]
        if not ids:
            return
        mask[torch.tensor(ids, device=scores.device, dtype=torch.long)] = scores[b, torch.tensor(ids, device=scores.device, dtype=torch.long)]
        scores[b] = mask

    def _allow_mask(self, scores, b: int, allowed_mask):
        allowed_mask = allowed_mask.to(device=scores.device, dtype=torch.bool)
        if not bool(allowed_mask.any().item()):
            return
        scores[b] = torch.where(allowed_mask, scores[b], torch.full_like(scores[b], float("-inf")))

    def _range_mask(self, scores, start: int, end: int):
        mask = torch.zeros((scores.shape[-1],), dtype=torch.bool, device=scores.device)
        start = max(0, int(start)); end = min(int(end), int(scores.shape[-1]))
        if end > start:
            mask[start:end] = True
        return mask

    def _tokens_since_last_record_begin(self, toks, section_begin, section_end):
        in_section = False
        current = []
        for tok in toks:
            tok = int(tok)
            if tok == int(section_begin):
                in_section = True
                current = []
                continue
            if tok == int(section_end):
                in_section = False
                current = []
                continue
            if not in_section:
                continue
            if tok == int(self.tokenizer.demand_begin):
                current = [tok]
            elif current:
                current.append(tok)
                if tok == int(self.tokenizer.demand_end):
                    current = []
        return current

    def _tokens_since_last_marker(self, toks, section_begin, section_end, record_begin, record_end):
        in_section = False
        current = []
        for tok in toks:
            tok = int(tok)
            if tok == int(section_begin):
                in_section = True
                current = []
                continue
            if tok == int(section_end):
                in_section = False
                current = []
                continue
            if not in_section:
                continue
            if tok == int(record_begin):
                current = [tok]
            elif current:
                current.append(tok)
                if tok == int(record_end):
                    current = []
        return current

    def _source_net_record_plan(self, rec):
        tok = self.tokenizer
        load_begin_pos = None
        load_end_pos = None
        fanout_bucket = None
        load_count = 0
        try:
            for idx, tid in enumerate(rec):
                tid = int(tid)
                if tid in {int(tok.fanout_1), int(tok.fanout_2_4), int(tok.fanout_5_8), int(tok.fanout_9_16), int(tok.fanout_17_plus)}:
                    fanout_bucket = tid
                if tid in {int(tok.load_group_begin), int(tok.load_preview_begin)}:
                    load_begin_pos = idx
                if tid in {int(tok.load_group_end), int(tok.load_preview_end)}:
                    load_end_pos = idx
                if tid == int(tok.load_cell):
                    load_count += 1
        except Exception:
            pass
        if fanout_bucket == int(tok.fanout_1):
            min_loads, max_loads = 1, 1
            begin_token, end_token = int(tok.load_group_begin), int(tok.load_group_end)
        elif fanout_bucket == int(tok.fanout_2_4):
            min_loads, max_loads = 2, 4
            begin_token, end_token = int(tok.load_group_begin), int(tok.load_group_end)
        elif fanout_bucket == int(tok.fanout_5_8):
            min_loads, max_loads = 5, 8
            begin_token, end_token = int(tok.load_group_begin), int(tok.load_group_end)
        elif fanout_bucket == int(tok.fanout_9_16):
            min_loads, max_loads = 8, 8
            begin_token, end_token = int(tok.load_preview_begin), int(tok.load_preview_end)
        elif fanout_bucket == int(tok.fanout_17_plus):
            min_loads, max_loads = 8, 8
            begin_token, end_token = int(tok.load_preview_begin), int(tok.load_preview_end)
        else:
            min_loads, max_loads = 1, 8
            begin_token, end_token = int(tok.load_group_begin), int(tok.load_group_end)
        return {
            "fanout_bucket": fanout_bucket,
            "load_begin_pos": load_begin_pos,
            "load_end_pos": load_end_pos,
            "load_count": int(load_count),
            "min_loads": int(min_loads),
            "max_loads": int(max_loads),
            "begin_token": int(begin_token),
            "end_token": int(end_token),
        }

    def _apply_source_net_record_template_mask(self, scores, b: int, toks, expected: int, record_state, end_tok: int):
        tok = self.tokenizer
        begin_count = int(record_state["begin_count"])
        open_record = bool(record_state["open_record"])
        hard = self._hard_record("SOURCE_NET", begin_count - 1 if open_record else begin_count)
        if hard is not None:
            if begin_count >= int(expected) and not open_record:
                self._allow(scores, b, [end_tok])
                return True
            if not open_record:
                self._allow(scores, b, [hard[0]])
                return True
            rec = self._tokens_since_last_marker(
                toks,
                int(tok.source_net_section_begin),
                int(tok.source_net_section_end),
                int(tok.source_net_begin),
                int(tok.source_net_end),
            )
            pos = int(len(rec))
            if pos < len(hard):
                self._allow(scores, b, [hard[pos]])
            else:
                self._allow(scores, b, [tok.source_net_end])
            self._note_hard_mask("SOURCE_NET")
            return True
        if begin_count >= int(expected) and not open_record:
            self._allow(scores, b, [end_tok])
            return True
        if not open_record:
            self._allow(scores, b, [tok.source_net_begin])
            return True
        rec = self._tokens_since_last_marker(
            toks,
            int(tok.source_net_section_begin),
            int(tok.source_net_section_end),
            int(tok.source_net_begin),
            int(tok.source_net_end),
        )
        pos = max(0, len(rec))
        fixed = {
            1: tok.source_kind,
            3: tok.source_cell,
            5: tok.source_cell_type,
            7: tok.source_pin,
            9: tok.fanout_bucket,
        }
        if pos in fixed:
            self._allow(scores, b, [fixed[pos]])
        elif pos == 2:
            self._allow(scores, b, [tok.cell_output, tok.boundary_source, tok.stub_source, tok.pi_const_source, tok.input_pool_source, tok.private_out_source])
        elif pos == 4:
            mask = self._range_mask(scores, tok.idx_offset + tok.max_num_nodes, tok.idx_offset + 2 * tok.max_num_nodes)
            for special in [tok.boundary_source, tok.stub_source, tok.pi_const_source, tok.input_pool_source, tok.private_out_source, tok.const0, tok.const1]:
                if 0 <= int(special) < int(mask.numel()):
                    mask[int(special)] = True
            self._allow_mask(scores, b, mask)
        elif pos == 6:
            mask = self._range_mask(scores, tok.node_type_offset, int(scores.shape[-1]))
            for special in [tok.boundary_source, tok.stub_source, tok.pi_const_source, tok.input_pool_source, tok.private_out_source, tok.unknown]:
                if 0 <= int(special) < int(mask.numel()):
                    mask[int(special)] = True
            self._allow_mask(scores, b, mask)
        elif pos == 8:
            self._allow_mask(scores, b, self._range_mask(scores, tok.pin_offset, tok.node_type_offset))
        elif pos == 10:
            self._allow(scores, b, [tok.fanout_1, tok.fanout_2_4, tok.fanout_5_8, tok.fanout_9_16, tok.fanout_17_plus])
        elif pos == 11:
            plan = self._source_net_record_plan(rec)
            self._allow(scores, b, [plan["begin_token"]])
        else:
            plan = self._source_net_record_plan(rec)
            if plan["load_begin_pos"] is None:
                self._allow(scores, b, [plan["begin_token"]])
                return True
            if plan["load_end_pos"] is not None:
                self._allow(scores, b, [tok.source_net_end])
                return True
            rel = pos - int(plan["load_begin_pos"]) - 1
            phase = rel % 6
            load_count = int(plan["load_count"])
            if load_count >= int(plan["max_loads"]) and phase == 0:
                self._allow(scores, b, [plan["end_token"]])
            elif load_count >= int(plan["min_loads"]) and phase == 0:
                self._allow(scores, b, [tok.load_cell, plan["end_token"]])
            elif phase == 0:
                self._allow(scores, b, [tok.load_cell])
            elif phase == 1:
                self._allow_mask(scores, b, self._range_mask(scores, tok.idx_offset, tok.idx_offset + tok.max_num_nodes))
            elif phase == 2:
                self._allow(scores, b, [tok.load_pin])
            elif phase == 3:
                self._allow_mask(scores, b, self._range_mask(scores, tok.pin_offset, tok.node_type_offset))
            elif phase == 4:
                self._allow(scores, b, [tok.demand_id])
            elif phase == 5:
                self._allow_mask(scores, b, self._range_mask(scores, tok.idx_offset + 2 * tok.max_num_nodes, tok.pin_offset))
        return True

    def _tokens_since_last_cell_record(self, toks):
        tok = self.tokenizer
        in_section = False
        current = []
        for tid in toks:
            tid = int(tid)
            if tid == int(tok.cell_section_begin):
                in_section = True
                current = []
                continue
            if tid == int(tok.cell_section_end):
                in_section = False
                current = []
                continue
            if not in_section:
                continue
            if tid == int(tok.cell):
                current = [tid]
            elif current:
                current.append(tid)
        return current

    def _tokens_in_section(self, toks, section_begin, section_end):
        in_section = False
        out = []
        for tid in toks:
            tid = int(tid)
            if tid == int(section_begin):
                in_section = True
                out = []
                continue
            if tid == int(section_end):
                in_section = False
                out = []
                continue
            if in_section:
                out.append(tid)
        return out

    def _bucket_value_tokens(self):
        tok = self.tokenizer
        return [tok.bucket_0, tok.bucket_1, tok.bucket_2_4, tok.bucket_5_8, tok.bucket_9_16, tok.bucket_17_plus]

    def _hard_record(self, section: str, idx: int):
        if not section:
            return None
        if not bool(getattr(self.tokenizer, "source_net_v5_enable_hard_reference_mask", False)):
            return None
        enabled = getattr(self.tokenizer, "source_net_v5_hard_reference_sections", None)
        if enabled is not None and str(section) not in set(str(x) for x in enabled):
            return None
        payload = getattr(self.tokenizer, "source_net_v5_hard_reference_records", None)
        if not isinstance(payload, dict):
            return None
        records = payload.get(str(section))
        if not isinstance(records, (list, tuple)):
            return None
        idx = int(idx)
        if idx < 0 or idx >= len(records):
            return None
        try:
            return [int(x) for x in records[idx]]
        except Exception:
            return None

    def _note_hard_mask(self, section: str):
        try:
            stats = getattr(self.tokenizer, "last_source_net_v5_hard_mask_stats", None)
            if not isinstance(stats, dict):
                stats = {}
            key = f"{str(section).lower()}_forced_token_count"
            stats[key] = int(stats.get(key, 0)) + 1
            self.tokenizer.last_source_net_v5_hard_mask_stats = stats
        except Exception:
            pass

    def _apply_cond_template_mask(self, scores, b: int, toks, end_tok: int):
        tok = self.tokenizer
        body = self._tokens_in_section(toks, int(tok.cond_section_begin), int(tok.cond_section_end))
        pos = int(len(body))
        if pos in {0, 1}:
            self._allow(scores, b, self._bucket_value_tokens())
        else:
            self._allow(scores, b, [end_tok])
        return True

    def _apply_source_budget_template_mask(self, scores, b: int, toks, end_tok: int):
        tok = self.tokenizer
        body = self._tokens_in_section(toks, int(tok.source_budget_section_begin), int(tok.source_budget_section_end))
        pos = int(len(body))
        if pos <= 5:
            self._allow(scores, b, self._bucket_value_tokens())
        elif 6 <= pos <= 17:
            if (pos - 6) % 2 == 0:
                self._allow(scores, b, [tok.cell_output, tok.boundary_source, tok.stub_source, tok.pi_const_source, tok.input_pool_source, tok.private_out_source])
            else:
                self._allow(scores, b, self._bucket_value_tokens())
        else:
            self._allow(scores, b, [end_tok])
        return True

    def _apply_cell_record_template_mask(self, scores, b: int, toks, expected: int, record_state, end_tok: int):
        tok = self.tokenizer
        begin_count = int(record_state["begin_count"])
        rec = self._tokens_since_last_cell_record(toks)
        rec_len = int(len(rec))
        record_open = bool(rec_len > 0 and rec_len < 9)
        hard = self._hard_record("CELL", begin_count - 1 if record_open else begin_count)
        if hard is not None:
            if begin_count >= int(expected) and not record_open:
                self._allow(scores, b, [end_tok])
                return True
            if not record_open:
                self._allow(scores, b, [hard[0]])
                return True
            pos = int(len(rec))
            if pos < len(hard):
                self._allow(scores, b, [hard[pos]])
            else:
                if begin_count < int(expected):
                    self._allow(scores, b, [tok.cell])
                else:
                    self._allow(scores, b, [end_tok])
            self._note_hard_mask("CELL")
            return True
        if begin_count >= int(expected) and not record_open:
            self._allow(scores, b, [end_tok])
            return True
        if not record_open:
            self._allow(scores, b, [tok.cell])
            return True
        pos = rec_len
        if pos == 1:
            self._allow_mask(scores, b, self._range_mask(scores, tok.idx_offset, tok.idx_offset + tok.max_num_nodes))
        elif pos == 2:
            self._allow_mask(scores, b, self._range_mask(scores, tok.node_type_offset, int(scores.shape[-1])))
        elif pos == 3:
            self._allow(scores, b, [tok.topo_unknown, tok.topo_0, tok.topo_1, tok.topo_2, tok.topo_3_plus])
        elif pos == 4:
            self._allow(scores, b, [tok.bucket_0, tok.bucket_1, tok.bucket_2_4, tok.bucket_5_8, tok.bucket_9_16, tok.bucket_17_plus])
        elif pos == 5:
            self._allow(scores, b, [tok.fanout_0, tok.fanout_1, tok.fanout_2_4, tok.fanout_5_8, tok.fanout_9_16, tok.fanout_17_plus])
        elif pos == 6:
            self._allow(scores, b, [tok.has_seq, tok.no_seq])
        elif pos == 7:
            self._allow(scores, b, [tok.has_mux, tok.no_mux])
        elif pos == 8:
            self._allow(scores, b, [tok.has_aoi_oai, tok.no_aoi_oai])
        else:
            if begin_count < int(expected):
                self._allow(scores, b, [tok.cell])
            else:
                self._allow(scores, b, [end_tok])
        return True

    def _apply_v5_record_template_mask(self, scores, b: int, current, toks, expected: int, record_state, end_tok: int):
        tok = self.tokenizer
        begin_count = int(record_state["begin_count"])
        open_record = bool(record_state["open_record"])
        section_key = None
        if current == int(tok.demand_section_begin):
            section_key = "DEMAND"
        elif current == int(tok.sink_demand_section_begin):
            section_key = "SINK_DEMAND"
        hard = self._hard_record(section_key, begin_count - 1 if open_record else begin_count) if section_key else None
        if hard is not None:
            if begin_count >= int(expected) and not open_record:
                self._allow(scores, b, [end_tok])
                return True
            if not open_record:
                self._allow(scores, b, [hard[0]])
                return True
            rec = self._tokens_since_last_record_begin(toks, current, end_tok)
            pos = int(len(rec))
            if pos < len(hard):
                self._allow(scores, b, [hard[pos]])
            else:
                self._allow(scores, b, [tok.demand_end])
            self._note_hard_mask(section_key)
            return True
        if begin_count >= int(expected) and not open_record:
            self._allow(scores, b, [end_tok])
            return True
        if not open_record:
            self._allow(scores, b, [tok.demand_begin])
            return True
        rec = self._tokens_since_last_record_begin(toks, current, end_tok)
        pos = max(0, len(rec))
        # Position includes DEMAND_BEGIN at pos 1.
        if current == int(tok.demand_section_begin):
            fixed = {
                1: tok.demand_id,
                3: tok.load_cell,
                5: tok.load_pin,
                7: tok.demand_kind,
                8: tok.internal_required,
                9: tok.demand_end,
            }
            if pos in fixed:
                self._allow(scores, b, [fixed[pos]])
            elif pos == 2:
                self._allow_mask(scores, b, self._range_mask(scores, tok.idx_offset + 2 * tok.max_num_nodes, tok.pin_offset))
            elif pos == 4:
                self._allow_mask(scores, b, self._range_mask(scores, tok.idx_offset, tok.idx_offset + tok.max_num_nodes))
            elif pos == 6:
                self._allow_mask(scores, b, self._range_mask(scores, tok.pin_offset, tok.node_type_offset))
            else:
                self._allow(scores, b, [tok.demand_end])
            return True
        if current == int(tok.sink_demand_section_begin):
            fixed = {
                1: tok.demand_id,
                3: tok.load_cell,
                5: tok.load_pin,
                7: tok.source_kind,
                9: tok.source_cell,
                11: tok.source_pin,
                13: tok.source_final_fanout_bucket,
                15: tok.demand_end,
            }
            if pos in fixed:
                self._allow(scores, b, [fixed[pos]])
            elif pos == 2:
                self._allow_mask(scores, b, self._range_mask(scores, tok.idx_offset + 2 * tok.max_num_nodes, tok.pin_offset))
            elif pos == 4:
                self._allow_mask(scores, b, self._range_mask(scores, tok.idx_offset, tok.idx_offset + tok.max_num_nodes))
            elif pos == 6 or pos == 12:
                self._allow_mask(scores, b, self._range_mask(scores, tok.pin_offset, tok.node_type_offset))
            elif pos == 8:
                self._allow(scores, b, [tok.cell_output, tok.boundary_source, tok.stub_source, tok.pi_const_source, tok.input_pool_source, tok.private_out_source])
            elif pos == 10:
                mask = self._range_mask(scores, tok.idx_offset + tok.max_num_nodes, tok.idx_offset + 2 * tok.max_num_nodes)
                mask[int(tok.boundary_source)] = True
                mask[int(tok.const0)] = True
                mask[int(tok.const1)] = True
                self._allow_mask(scores, b, mask)
            elif pos == 14:
                self._allow(scores, b, [tok.fanout_1, tok.fanout_2_4, tok.fanout_5_8, tok.fanout_9_16, tok.fanout_17_plus])
            else:
                self._allow(scores, b, [tok.demand_end])
            return True
        return False

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            toks = [int(x) for x in input_ids[b].detach().cpu().tolist() if int(x) != int(getattr(self.tokenizer, "pad", -1))]
            if not toks or toks[-1] == int(self.tokenizer.sos):
                self._allow(scores, b, [self.tokenizer.cond_section_begin])
                continue
            if toks[-1] == int(self.tokenizer.eos):
                self._allow(scores, b, [self.tokenizer.eos])
                continue
            opened, closed = self._opened_sections(toks)
            current = None
            for begin in reversed(opened):
                if begin not in closed:
                    current = begin
                    break
            if current is None:
                done_count = sum(1 for sec in self.section_order if sec in closed)
                if done_count >= len(self.section_order):
                    self._allow(scores, b, [self.tokenizer.eos])
                else:
                    self._allow(scores, b, [self.section_order[done_count]])
                continue
            # Light grammar: keep generation within the current section and
            # permit its proper close token, but do not enforce object-level
            # demand/source consistency.
            end_tok = self.section_end[current]
            disallow = {self.tokenizer.eos}
            for begin in self.section_order:
                if begin != current:
                    disallow.add(begin)
            for begin, end in self.section_end.items():
                if begin != current:
                    disallow.add(end)
            expected = self._expected_count_for_section(current)
            record_state = None
            if current == int(self.tokenizer.cond_section_begin):
                if self._apply_cond_template_mask(scores, b, toks, end_tok):
                    continue
            if current == int(self.tokenizer.source_budget_section_begin):
                if self._apply_source_budget_template_mask(scores, b, toks, end_tok):
                    continue
            target_sections = {
                int(self.tokenizer.demand_section_begin),
                int(self.tokenizer.sink_demand_section_begin),
                int(self.tokenizer.cell_section_begin),
                int(self.tokenizer.source_net_section_begin),
            }
            if expected >= 0 and current in target_sections:
                if current in {int(self.tokenizer.demand_section_begin), int(self.tokenizer.sink_demand_section_begin)}:
                    record_state = self._record_state(toks, current, end_tok)
                    new_record_tok = int(self.tokenizer.demand_begin)
                    section_name = "DEMAND" if current == int(self.tokenizer.demand_section_begin) else "SINK_DEMAND"
                    if self._apply_v5_record_template_mask(scores, b, current, toks, expected, record_state, end_tok):
                        continue
                elif current == int(self.tokenizer.cell_section_begin):
                    record_state = self._marker_state(toks, current, end_tok, int(self.tokenizer.cell), None)
                    new_record_tok = int(self.tokenizer.cell)
                    section_name = "CELL"
                    if self._apply_cell_record_template_mask(scores, b, toks, expected, record_state, end_tok):
                        continue
                else:
                    record_state = self._marker_state(toks, current, end_tok, int(self.tokenizer.source_net_begin), int(self.tokenizer.source_net_end))
                    new_record_tok = int(self.tokenizer.source_net_begin)
                    section_name = "SOURCE_NET"
                    if self._apply_source_net_record_template_mask(scores, b, toks, expected, record_state, end_tok):
                        continue
                begin_count = int(record_state["begin_count"])
                open_record = bool(record_state["open_record"])
                tolerance = int(getattr(self.tokenizer, "source_net_v5_record_budget_tolerance", 0) or 0)
                try:
                    self.tokenizer.last_source_net_v5_budget_state = {
                        "section": section_name,
                        "expected": int(expected),
                        "begin_count": int(begin_count),
                        "end_count": int(record_state["end_count"]),
                        "open_record": bool(open_record),
                        "budget_exceeded": bool(begin_count > expected + tolerance),
                    }
                except Exception:
                    pass
                if begin_count < expected or open_record:
                    disallow.add(end_tok)
                if begin_count >= expected:
                    disallow.add(new_record_tok)
                if begin_count >= expected and not open_record:
                    original_end_score = scores[b, int(end_tok)].clone()
                    scores[b] = float("-inf")
                    scores[b, int(end_tok)] = original_end_score
                    continue
            scores[b, torch.tensor([x for x in disallow if 0 <= int(x) < scores.shape[-1]], device=scores.device, dtype=torch.long)] = float("-inf")
            # The close token remains available under the model probability.
            scores[b, int(end_tok)] = scores[b, int(end_tok)]
        return scores


class SourceNetV41SinkFirstGrammar(LogitsProcessor):
    """Evaluation-time sink-first grammar for V4.1 generated smoke."""

    def __init__(self, tokenizer, batch_size, device='cpu', mask_mode=None):
        self.tok = tokenizer
        self.batch_size = int(batch_size)
        self.device = device
        self.mask_mode = str(mask_mode or "source_net_v41_sinkfirst_primary")
        vocab = int(len(tokenizer))
        self.idx = torch.arange(vocab, device=device)
        self.gate_start = int(tokenizer.idx_offset)
        self.gate_end = int(tokenizer.idx_offset + int(tokenizer.max_num_nodes or 0))
        self.src_start = self.gate_end
        self.src_end = int(tokenizer.pin_offset)
        self.pin_start = int(tokenizer.pin_offset)
        self.pin_end = int(tokenizer.node_type_offset)
        self.cell_start = int(tokenizer.node_type_offset)
        self.is_gate = (self.idx >= self.gate_start) & (self.idx < self.gate_end)
        self.is_src = (self.idx >= self.src_start) & (self.idx < self.src_end)
        self.is_cell = self.idx >= self.cell_start
        self.gc_tokens = {int(getattr(tokenizer, n)) for n in ["gc_bucket_0", "gc_bucket_64", "gc_bucket_96", "gc_bucket_128", "gc_bucket_160", "gc_bucket_224"] if hasattr(tokenizer, n)}
        self.cut_tokens = {int(getattr(tokenizer, n)) for n in ["cut_edge_bucket_0", "cut_edge_bucket_1_4", "cut_edge_bucket_5_16", "cut_edge_bucket_17_32", "cut_edge_bucket_33_plus"] if hasattr(tokenizer, n)}
        self.diversity_tokens = {int(getattr(tokenizer, n)) for n in ["source_diversity_bucket_low", "source_diversity_bucket_mid", "source_diversity_bucket_high"] if hasattr(tokenizer, n)}
        self.fanout_profile_tokens = {int(getattr(tokenizer, n)) for n in ["fanout_profile_bucket_low", "fanout_profile_bucket_mid", "fanout_profile_bucket_high"] if hasattr(tokenizer, n)}
        self.critical_tokens = {int(getattr(tokenizer, n)) for n in ["critical_output_bucket_none", "critical_output_bucket_low", "critical_output_bucket_high"] if hasattr(tokenizer, n)}
        self.pool_tokens = {int(getattr(tokenizer, n)) for n in ["pool_risk_bucket_low", "pool_risk_bucket_mid", "pool_risk_bucket_high"] if hasattr(tokenizer, n)}
        self.min_cell_count = int(getattr(tokenizer, "source_net_v41_min_cell_count", 16) or 16)
        self.min_sink_count = int(getattr(tokenizer, "source_net_v41_min_sink_count", self.min_cell_count) or self.min_cell_count)

    def _mask_only(self, scores, allowed: torch.Tensor):
        masked = torch.where(allowed, torch.nan_to_num(scores, nan=0.0, posinf=1.0e4, neginf=float('-inf')), float('-inf'))
        if bool(allowed.any().item()) and not bool(torch.isfinite(masked).any().item()):
            masked = torch.where(allowed, torch.zeros_like(masked), masked)
        return masked

    def _allow_tokens(self, scores, toks):
        allowed = torch.zeros((scores.shape[-1],), dtype=torch.bool, device=scores.device)
        for tok in toks:
            if tok is not None and 0 <= int(tok) < int(allowed.numel()):
                allowed[int(tok)] = True
        return self._mask_only(scores, allowed)

    def _pin_mask(self, pins, scores):
        allowed = torch.zeros((scores.shape[-1],), dtype=torch.bool, device=scores.device)
        for pin in pins:
            pid = self.tok.pin_name_to_id.get(str(pin))
            if pid is not None:
                tid = int(self.tok.pin_offset + int(pid))
                if 0 <= tid < int(allowed.numel()):
                    allowed[tid] = True
        return self._mask_only(scores, allowed)

    def _gate_tok(self, idx): return int(self.tok.idx_offset + int(idx))
    def _src_tok(self, idx): return int(self.tok.idx_offset + int(self.tok.max_num_nodes or 0) + int(idx))
    def _gate_from_tok(self, tok): return int(tok - int(self.tok.idx_offset))
    def _src_from_tok(self, tok): return int(tok - int(self.tok.idx_offset) - int(self.tok.max_num_nodes or 0))
    def _cell_from_tok(self, tok): return int(tok - int(self.tok.node_type_offset))

    def _parse(self, toks):
        toks = [int(x) for x in toks if int(x) != int(self.tok.pad)]
        if toks and toks[0] == int(self.tok.sos):
            toks = toks[1:]
        cells, used_sinks = {}, set()
        i = 0
        prefix = [
            (int(self.tok.contract_start), "one"),
            (self.gc_tokens, "set"),
            (self.cut_tokens, "set"),
            (self.diversity_tokens, "set"),
            (self.fanout_profile_tokens, "set"),
            (self.critical_tokens, "set"),
            (self.pool_tokens, "set"),
            (int(self.tok.contract_end), "one"),
            (int(self.tok.cell_section_start), "one"),
        ]
        for expected, kind in prefix:
            if i >= len(toks):
                return {"state": f"prefix_{i}", "cells": cells, "used_sinks": used_sinks}
            ok = toks[i] == expected if kind == "one" else toks[i] in expected
            if not ok:
                return {"state": "invalid", "cells": cells, "used_sinks": used_sinks}
            i += 1
        while i < len(toks):
            tok = toks[i]
            if tok == int(self.tok.cell_section_end):
                if len(cells) < self.min_cell_count:
                    return {"state": "invalid", "cells": cells, "used_sinks": used_sinks}
                i += 1
                break
            if tok != int(self.tok.cell):
                return {"state": "invalid", "cells": cells, "used_sinks": used_sinks}
            if i + 1 >= len(toks):
                return {"state": "cell_gate", "cells": cells, "used_sinks": used_sinks, "next_cell": len(cells)}
            if toks[i + 1] != self._gate_tok(len(cells)):
                return {"state": "invalid", "cells": cells, "used_sinks": used_sinks}
            if i + 2 >= len(toks):
                return {"state": "cell_type", "cells": cells, "used_sinks": used_sinks, "gate": len(cells)}
            cell_label = self._cell_from_tok(toks[i + 2])
            cells[len(cells)] = cell_label
            i += 3
        else:
            return {"state": "cell_or_end", "cells": cells, "used_sinks": used_sinks}
        if i >= len(toks):
            return {"state": "need_sink_section_start", "cells": cells, "used_sinks": used_sinks}
        if toks[i] != int(self.tok.sink_connection_section_start):
            return {"state": "invalid", "cells": cells, "used_sinks": used_sinks}
        i += 1
        sink_count = 0
        while i < len(toks):
            tok = toks[i]
            if tok == int(self.tok.sink_connection_section_end):
                if sink_count < self.min_sink_count:
                    return {"state": "invalid", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
                i += 1
                if i >= len(toks):
                    return {"state": "need_eos", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
                if toks[i] == int(self.tok.eos):
                    return {"state": "done", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
                return {"state": "invalid", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
            if tok != int(self.tok.sink):
                return {"state": "invalid", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
            if i + 1 >= len(toks):
                return {"state": "sink_gate", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
            gate = self._gate_from_tok(toks[i + 1])
            if gate not in cells:
                return {"state": "invalid", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
            if i + 2 >= len(toks):
                return {"state": "sink_pin", "cells": cells, "used_sinks": used_sinks, "sink_gate": gate, "sink_count": sink_count}
            sink_pin = self.tok._pin_from_tok(toks[i + 2])
            cell = self.tok._cell_name(cells[gate])
            inputs, _outputs = self.tok._spec_pins(cell)
            if sink_pin not in inputs or (gate, sink_pin) in used_sinks:
                return {"state": "invalid", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
            if i + 3 >= len(toks):
                return {"state": "need_src_marker", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
            if toks[i + 3] != int(self.tok.src):
                return {"state": "invalid", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
            if i + 4 >= len(toks):
                return {"state": "src_kind", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
            src_kind = toks[i + 4]
            if src_kind == int(self.tok.boundary_in):
                if i + 5 >= len(toks):
                    return {"state": "boundary_pin", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
            elif self.src_start <= src_kind < self.src_end:
                src_gate = self._src_from_tok(src_kind)
                if src_gate not in cells:
                    return {"state": "invalid", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
                if i + 5 >= len(toks):
                    return {"state": "src_pin", "cells": cells, "used_sinks": used_sinks, "src_gate": src_gate, "sink_count": sink_count}
                src_pin = self.tok._pin_from_tok(toks[i + 5])
                _ins, outs = self.tok._spec_pins(self.tok._cell_name(cells[src_gate]))
                if src_pin not in outs:
                    return {"state": "invalid", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
            else:
                return {"state": "invalid", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}
            used_sinks.add((gate, sink_pin))
            sink_count += 1
            i += 6
        return {"state": "sink_or_end", "cells": cells, "used_sinks": used_sinks, "sink_count": sink_count}

    def __call__(self, input_ids, scores):
        out = scores.clone()
        for b in range(input_ids.shape[0]):
            st = self._parse(input_ids[b].detach().tolist())
            state = st["state"]
            cells = st.get("cells", {})
            if state.startswith("prefix_"):
                idx = int(state.split("_", 1)[1])
                seq = [int(self.tok.contract_start), self.gc_tokens, self.cut_tokens, self.diversity_tokens, self.fanout_profile_tokens, self.critical_tokens, self.pool_tokens, int(self.tok.contract_end), int(self.tok.cell_section_start)]
                item = seq[idx]
                out[b] = self._allow_tokens(scores[b], item if isinstance(item, set) else [item])
            elif state == "cell_or_end":
                toks = [int(self.tok.cell)]
                if len(cells) >= self.min_cell_count:
                    toks.append(int(self.tok.cell_section_end))
                out[b] = self._allow_tokens(scores[b], toks)
            elif state == "cell_gate":
                out[b] = self._allow_tokens(scores[b], [self._gate_tok(int(st.get("next_cell", len(cells))))])
            elif state == "cell_type":
                allowed = torch.zeros((scores.shape[-1],), dtype=torch.bool, device=scores.device)
                allowed[self.cell_start:] = True
                out[b] = self._mask_only(scores[b], allowed)
            elif state == "need_sink_section_start":
                out[b] = self._allow_tokens(scores[b], [int(self.tok.sink_connection_section_start)])
            elif state == "sink_or_end":
                toks = [int(self.tok.sink)]
                if int(st.get("sink_count", 0)) >= self.min_sink_count:
                    toks.append(int(self.tok.sink_connection_section_end))
                out[b] = self._allow_tokens(scores[b], toks)
            elif state == "sink_gate":
                unused = []
                used = st.get("used_sinks", set())
                for g, label in cells.items():
                    ins, _outs = self.tok._spec_pins(self.tok._cell_name(label))
                    if any((g, p) not in used for p in ins):
                        unused.append(self._gate_tok(g))
                out[b] = self._allow_tokens(scores[b], unused or [self._gate_tok(g) for g in cells])
            elif state == "sink_pin":
                g = int(st.get("sink_gate", 0))
                ins, _outs = self.tok._spec_pins(self.tok._cell_name(cells.get(g, self.tok.net_id)))
                used = {p for gg, p in st.get("used_sinks", set()) if gg == g}
                out[b] = self._pin_mask([p for p in ins if p not in used] or list(ins), scores[b])
            elif state == "need_src_marker":
                out[b] = self._allow_tokens(scores[b], [int(self.tok.src)])
            elif state == "src_kind":
                toks = [int(self.tok.boundary_in)] + [self._src_tok(g) for g in cells]
                out[b] = self._allow_tokens(scores[b], toks)
            elif state == "boundary_pin":
                out[b] = self._pin_mask(["__BOUNDARY__"], scores[b])
            elif state == "src_pin":
                g = int(st.get("src_gate", 0))
                _ins, outs = self.tok._spec_pins(self.tok._cell_name(cells.get(g, self.tok.net_id)))
                out[b] = self._pin_mask(outs, scores[b])
            elif state == "need_eos":
                out[b] = self._allow_tokens(scores[b], [int(self.tok.eos)])
            elif state == "done":
                out[b] = self._allow_tokens(scores[b], [int(self.tok.eos)])
            else:
                out[b] = self._allow_tokens(scores[b], [int(self.tok.eos)])
        return out


class SourceNetV4Grammar(LogitsProcessor):
    """Minimal source-net V4 logits mask for evaluation-time smoke tests.

    This constrains the section order, canonical cell/net ids, existing cell
    references, Liberty-compatible driver/load pins, one driver per net record,
    and duplicate cell.pin loads. It is intentionally conservative; it does not
    try to optimize required-input coverage or assembly quality.
    """

    def __init__(self, tokenizer, batch_size, device='cpu', mask_mode=None):
        self.tok = tokenizer
        self.batch_size = int(batch_size)
        self.device = device
        self.mask_mode = str(mask_mode or "source_net_grammar_primary")
        vocab = int(len(tokenizer))
        self.idx = torch.arange(vocab, device=device)
        self.gate_idx_start = int(tokenizer.idx_offset)
        self.gate_idx_end = int(tokenizer.idx_offset + int(tokenizer.max_num_nodes or 0))
        self.net_idx_start = int(self.gate_idx_end)
        self.net_idx_end = int(tokenizer.pin_offset)
        self.pin_start = int(tokenizer.pin_offset)
        self.pin_end = int(tokenizer.node_type_offset)
        self.cell_start = int(tokenizer.node_type_offset)
        self.is_gate_idx = (self.idx >= self.gate_idx_start) & (self.idx < self.gate_idx_end)
        self.is_net_idx = (self.idx >= self.net_idx_start) & (self.idx < self.net_idx_end)
        self.is_pin = (self.idx >= self.pin_start) & (self.idx < self.pin_end)
        self.is_cell = self.idx >= self.cell_start
        self.gc_tokens = {
            int(getattr(tokenizer, name))
            for name in ["gc_bucket_0", "gc_bucket_64", "gc_bucket_96", "gc_bucket_128", "gc_bucket_160", "gc_bucket_224"]
            if hasattr(tokenizer, name)
        }
        self.cut_tokens = {
            int(getattr(tokenizer, name))
            for name in ["cut_edge_bucket_0", "cut_edge_bucket_1_4", "cut_edge_bucket_5_16", "cut_edge_bucket_17_32", "cut_edge_bucket_33_plus"]
            if hasattr(tokenizer, name)
        }
        self.diversity_tokens = {
            int(getattr(tokenizer, name))
            for name in ["source_diversity_bucket_low", "source_diversity_bucket_mid", "source_diversity_bucket_high"]
            if hasattr(tokenizer, name)
        }
        self.critical_bucket_tokens = {
            int(getattr(tokenizer, name))
            for name in ["critical_output_bucket_none", "critical_output_bucket_low", "critical_output_bucket_high"]
            if hasattr(tokenizer, name)
        }
        self.pool_tokens = {
            int(getattr(tokenizer, name))
            for name in ["pool_risk_bucket_low", "pool_risk_bucket_mid", "pool_risk_bucket_high"]
            if hasattr(tokenizer, name)
        }
        self.fanout_tokens = {
            int(getattr(tokenizer, name))
            for name in ["fanout_bucket_0", "fanout_bucket_1", "fanout_bucket_2_4", "fanout_bucket_5_8", "fanout_bucket_9_16", "fanout_bucket_17_plus"]
            if hasattr(tokenizer, name)
        }
        self.role_tokens = {
            int(getattr(tokenizer, name))
            for name in ["source_role_arith", "source_role_clock", "source_role_nand", "source_role_data", "source_role_mixed", "source_role_boundary"]
            if hasattr(tokenizer, name)
        }
        self.net_marker_tokens = {
            int(tokenizer.driver),
            int(tokenizer.pi_driver),
            int(tokenizer.boundary_driver),
        }
        self.min_cell_count = int(getattr(tokenizer, "source_net_v4_min_cell_count", 16) or 16)
        self.max_cell_count = int(getattr(tokenizer, "source_net_v4_max_cell_count", 0) or 0)
        self.min_net_count = int(getattr(tokenizer, "source_net_v4_min_net_count", max(1, self.min_cell_count // 2)) or 1)
        self.min_load_count = int(getattr(tokenizer, "source_net_v4_min_load_count", max(1, self.min_cell_count)) or 1)
        self.coverage_driven_loads = bool(getattr(tokenizer, "source_net_v4_coverage_driven_loads", False))
        self.fanout_aware_net_generation = bool(getattr(tokenizer, "source_net_v4_fanout_aware_net_generation", False))
        self.disable_boundary_driver = bool(getattr(tokenizer, "source_net_v4_disable_boundary_driver", False))

    def _fanout_lower_bound(self, tok: int) -> int:
        tok = int(tok)
        if tok == int(getattr(self.tok, "fanout_bucket_0", -1)):
            return 0
        if tok == int(getattr(self.tok, "fanout_bucket_1", -1)):
            return 1
        if tok == int(getattr(self.tok, "fanout_bucket_2_4", -1)):
            return 2
        if tok == int(getattr(self.tok, "fanout_bucket_5_8", -1)):
            return 5
        if tok == int(getattr(self.tok, "fanout_bucket_9_16", -1)):
            return 9
        if tok == int(getattr(self.tok, "fanout_bucket_17_plus", -1)):
            return 17
        return 1

    def _mask_only(self, scores, allowed: torch.Tensor):
        masked = torch.where(allowed, torch.nan_to_num(scores, nan=0.0, posinf=1.0e4, neginf=float('-inf')), float('-inf'))
        if bool(allowed.any().item()) and not bool(torch.isfinite(masked).any().item()):
            masked = torch.where(allowed, torch.zeros_like(masked), masked)
        return masked

    def _allow_tokens(self, scores, toks):
        allowed = torch.zeros((scores.shape[-1],), dtype=torch.bool, device=scores.device)
        for tok in toks:
            if tok is not None and 0 <= int(tok) < int(allowed.numel()):
                allowed[int(tok)] = True
        return self._mask_only(scores, allowed)

    def _allow_mask(self, scores, mask):
        return self._mask_only(scores, mask.to(scores.device))

    def _gate_tok(self, idx):
        return int(self.tok.idx_offset + int(idx))

    def _net_tok(self, idx):
        return int(self.tok.idx_offset + int(self.tok.max_num_nodes or 0) + int(idx))

    def _pin_tok(self, name):
        return int(self.tok.pin_offset + int(self.tok.pin_name_to_id[str(name)]))

    def _pin_mask(self, pins):
        allowed = torch.zeros((len(self.tok),), dtype=torch.bool, device=self.device)
        for pin in pins:
            pid = self.tok.pin_name_to_id.get(str(pin))
            if pid is not None:
                tid = int(self.tok.pin_offset + int(pid))
                if 0 <= tid < int(allowed.numel()):
                    allowed[tid] = True
        return allowed

    def _cell_from_tok(self, tok):
        return int(tok - int(self.tok.node_type_offset))

    def _gate_from_tok(self, tok):
        return int(tok - int(self.tok.idx_offset))

    def _parse(self, toks):
        toks = [int(x) for x in toks if int(x) not in {int(self.tok.pad)}]
        if toks and toks[0] == int(self.tok.sos):
            toks = toks[1:]
        cells = {}
        used_loads = set()
        next_cell = 0
        next_net = 0
        total_load_count = 0
        cell_section_closed = False
        i = 0

        prefix = [
            (int(self.tok.contract_start), "one"),
            (self.gc_tokens, "set"),
            (self.cut_tokens, "set"),
            (self.diversity_tokens, "set"),
            (self.critical_bucket_tokens, "set"),
            (self.pool_tokens, "set"),
            (int(self.tok.contract_end), "one"),
            (int(self.tok.cell_section_start), "one"),
        ]
        def state_payload(state, **extra):
            out = {
                "state": state,
                "cells": cells,
                "used_loads": used_loads,
                "next_cell": next_cell,
                "next_net": next_net,
                "total_load_count": total_load_count,
            }
            out.update(extra)
            return out

        for expected, kind in prefix:
            if i >= len(toks):
                return state_payload(f"prefix_{i}")
            ok = toks[i] == expected if kind == "one" else toks[i] in expected
            if not ok:
                return state_payload("invalid")
            i += 1

        while i < len(toks):
            tok = toks[i]
            if tok == int(self.tok.cell_section_end):
                if next_cell < self.min_cell_count:
                    return state_payload("invalid")
                cell_section_closed = True
                i += 1
                break
            if tok != int(self.tok.cell):
                return state_payload("invalid")
            if i + 1 >= len(toks):
                return state_payload("cell_idx")
            gid_tok = toks[i + 1]
            if gid_tok != self._gate_tok(next_cell):
                return state_payload("invalid")
            if i + 2 >= len(toks):
                return state_payload("cell_label")
            label_tok = toks[i + 2]
            if label_tok < self.cell_start:
                return state_payload("invalid")
            cells[next_cell] = self._cell_from_tok(label_tok)
            next_cell += 1
            i += 3

        if i >= len(toks):
            return state_payload("need_net_section_start" if cell_section_closed else "cell_or_cell_section_end")
        if toks[i] != int(self.tok.net_section_start):
            return state_payload("invalid")
        i += 1

        while i < len(toks):
            tok = toks[i]
            if tok == int(self.tok.net_section_end):
                if next_net < self.min_net_count or total_load_count < self.min_load_count:
                    return state_payload("invalid")
                i += 1
                if i >= len(toks):
                    return state_payload("eos")
                if toks[i] == int(self.tok.eos):
                    return state_payload("done")
                return state_payload("invalid")
            if tok != int(self.tok.net):
                return state_payload("invalid")
            if i + 1 >= len(toks):
                return state_payload("net_idx")
            if toks[i + 1] != self._net_tok(next_net):
                return state_payload("invalid")
            if i + 2 >= len(toks):
                return state_payload("fanout")
            if toks[i + 2] not in self.fanout_tokens:
                return state_payload("invalid")
            current_fanout_min = self._fanout_lower_bound(toks[i + 2]) if self.fanout_aware_net_generation else 1
            if i + 3 >= len(toks):
                return state_payload("role")
            if toks[i + 3] not in self.role_tokens:
                return state_payload("invalid")
            i += 4
            while i < len(toks) and toks[i] in {
                int(self.tok.critical_output),
                int(self.tok.optional_output_critical),
                int(self.tok.optional_output_normal),
            }:
                i += 1
            if i >= len(toks):
                return state_payload("driver_kind")
            marker = toks[i]
            if marker == int(self.tok.driver):
                if i + 1 >= len(toks):
                    return state_payload("driver_gate")
                gidx = self._gate_from_tok(toks[i + 1])
                if gidx not in cells:
                    return state_payload("invalid")
                if i + 2 >= len(toks):
                    return state_payload("driver_pin", driver_gate=gidx)
                cell = self.tok._cell_name(cells[gidx])
                _inputs, outputs = self.tok._spec_pins(cell)
                if self.tok._pin_from_tok(toks[i + 2]) not in outputs:
                    return state_payload("invalid")
                i += 3
            elif marker in {int(self.tok.boundary_driver), int(self.tok.pi_driver)}:
                if i + 1 >= len(toks):
                    return state_payload("boundary_driver_idx")
                if toks[i + 1] != self._net_tok(next_net):
                    return state_payload("invalid")
                i += 2
            else:
                return state_payload("invalid")
            current_net_loads = 0
            if i >= len(toks):
                return state_payload("load_or_end", current_net_loads=current_net_loads, current_fanout_min=current_fanout_min)

            while i < len(toks):
                if toks[i] == int(self.tok.end_net):
                    if current_net_loads < max(1, int(current_fanout_min)):
                        return state_payload("invalid")
                    i += 1
                    next_net += 1
                    break
                if toks[i] != int(self.tok.load):
                    return state_payload("invalid")
                if i + 1 >= len(toks):
                    return state_payload("load_gate")
                gidx = self._gate_from_tok(toks[i + 1])
                if gidx not in cells:
                    return state_payload("invalid")
                if i + 2 >= len(toks):
                    return state_payload("load_pin", load_gate=gidx)
                cell = self.tok._cell_name(cells[gidx])
                inputs, _outputs = self.tok._spec_pins(cell)
                pin = self.tok._pin_from_tok(toks[i + 2])
                if pin not in inputs or (gidx, str(pin)) in used_loads:
                    return state_payload("invalid")
                used_loads.add((gidx, str(pin)))
                current_net_loads += 1
                total_load_count += 1
                i += 3
                if i >= len(toks):
                    return state_payload("load_or_end", current_net_loads=current_net_loads, current_fanout_min=current_fanout_min)

        return state_payload("net_or_end")

    def __call__(self, input_ids, scores):
        for b in range(int(input_ids.shape[0])):
            st = self._parse(input_ids[b].detach().tolist())
            state = st["state"]
            cells = st["cells"]
            next_cell = int(st["next_cell"])
            next_net = int(st["next_net"])
            if state.startswith("prefix_"):
                idx = int(state.split("_", 1)[1])
                allowed_seq = [
                    [int(self.tok.contract_start)],
                    list(self.gc_tokens),
                    list(self.cut_tokens),
                    list(self.diversity_tokens),
                    list(self.critical_bucket_tokens),
                    list(self.pool_tokens),
                    [int(self.tok.contract_end)],
                    [int(self.tok.cell_section_start)],
                ]
                scores[b] = self._allow_tokens(scores[b], allowed_seq[min(idx, len(allowed_seq) - 1)])
            elif state == "cell_idx":
                scores[b] = self._allow_tokens(scores[b], [self._gate_tok(next_cell)])
            elif state == "cell_label":
                scores[b] = self._allow_mask(scores[b], self.is_cell)
            elif state == "cell_or_cell_section_end":
                toks = []
                max_cell = self.max_cell_count if self.max_cell_count > 0 else int(self.tok.max_num_nodes or 0)
                if next_cell < max_cell and next_cell < int(self.tok.max_num_nodes or 0):
                    toks.append(int(self.tok.cell))
                if next_cell >= self.min_cell_count:
                    toks.append(int(self.tok.cell_section_end))
                scores[b] = self._allow_tokens(scores[b], toks)
            elif state == "need_net_section_start":
                scores[b] = self._allow_tokens(scores[b], [int(self.tok.net_section_start)])
            elif state == "net_idx":
                scores[b] = self._allow_tokens(scores[b], [self._net_tok(next_net)])
            elif state == "fanout":
                scores[b] = self._allow_tokens(scores[b], self.fanout_tokens)
            elif state == "role":
                scores[b] = self._allow_tokens(scores[b], self.role_tokens)
            elif state == "driver_kind":
                toks = []
                if not self.disable_boundary_driver:
                    toks.extend([int(self.tok.boundary_driver), int(self.tok.pi_driver)])
                if cells:
                    toks.append(int(self.tok.driver))
                scores[b] = self._allow_tokens(scores[b], toks)
            elif state == "driver_gate":
                scores[b] = self._allow_tokens(scores[b], [self._gate_tok(g) for g in sorted(cells)])
            elif state == "driver_pin":
                gidx = int(st["driver_gate"])
                cell = self.tok._cell_name(cells[gidx])
                _inputs, outputs = self.tok._spec_pins(cell)
                scores[b] = self._allow_mask(scores[b], self._pin_mask(outputs))
            elif state == "boundary_driver_idx":
                scores[b] = self._allow_tokens(scores[b], [self._net_tok(next_net)])
            elif state == "load_gate":
                eligible = []
                for gidx, label in cells.items():
                    cell = self.tok._cell_name(label)
                    inputs, _outputs = self.tok._spec_pins(cell)
                    if any((int(gidx), str(pin)) not in st["used_loads"] for pin in inputs):
                        eligible.append(self._gate_tok(gidx))
                if self.coverage_driven_loads and eligible:
                    scores[b] = self._allow_tokens(scores[b], eligible)
                else:
                    scores[b] = self._allow_tokens(scores[b], eligible or [int(self.tok.end_net)])
            elif state == "load_pin":
                gidx = int(st["load_gate"])
                cell = self.tok._cell_name(cells[gidx])
                inputs, _outputs = self.tok._spec_pins(cell)
                pins = [p for p in inputs if (gidx, str(p)) not in st["used_loads"]]
                scores[b] = self._allow_mask(scores[b], self._pin_mask(pins))
            elif state == "load_or_end":
                current_net_loads = int(st.get("current_net_loads", 0))
                current_fanout_min = max(1, int(st.get("current_fanout_min", 1)))
                toks = [int(self.tok.end_net)] if current_net_loads >= current_fanout_min else []
                for gidx, label in cells.items():
                    cell = self.tok._cell_name(label)
                    inputs, _outputs = self.tok._spec_pins(cell)
                    if any((int(gidx), str(pin)) not in st["used_loads"] for pin in inputs):
                        toks.append(int(self.tok.load))
                        break
                scores[b] = self._allow_tokens(scores[b], toks)
            elif state == "net_or_end":
                toks = []
                total_load_count = int(st.get("total_load_count", 0))
                if next_net >= self.min_net_count and total_load_count >= self.min_load_count:
                    toks.append(int(self.tok.net_section_end))
                elif next_net < int(self.tok.max_num_nodes or 0):
                    toks.append(int(self.tok.net))
                scores[b] = self._allow_tokens(scores[b], toks)
            elif state == "eos":
                scores[b] = self._allow_tokens(scores[b], [int(self.tok.eos)])
            elif state == "done":
                scores[b] = self._allow_tokens(scores[b], [int(self.tok.eos), int(self.tok.pad)])
            elif state == "invalid":
                scores[b] = self._allow_tokens(scores[b], [int(self.tok.eos)])
            else:
                scores[b] = self._allow_tokens(scores[b], [int(self.tok.cell), int(self.tok.cell_section_end)])
            if not torch.isfinite(scores[b]).any().item():
                scores[b] = self._allow_tokens(scores[b], [int(self.tok.eos)])
        return scores


class PinSlotGrammar(LogitsProcessor):
    def __init__(self, tokenizer, batch_size, device='cpu', mask_mode=None):
        self.tok = tokenizer
        self.batch_size = int(batch_size)
        self.device = device
        self.mask_mode = str(mask_mode or getattr(tokenizer, 'pin_slot_mask_mode', 'practical_generation'))
        self.mode = [0 for _ in range(int(batch_size))]
        self.gate_count = [0 for _ in range(int(batch_size))]
        self.max_gates = int(getattr(tokenizer, 'max_gates', 1024))
        self.target_min_gates = int(getattr(tokenizer, 'target_min_gates', 0) or 0)
        self.target_max_gates = int(getattr(tokenizer, 'target_max_gates', 0) or 0)
        if self.target_max_gates > 0:
            self.max_gates = min(int(self.max_gates), int(self.target_max_gates))
        self.plan = [None for _ in range(int(batch_size))]
        self.plan_pos = [0 for _ in range(int(batch_size))]
        self.slot_state = [0 for _ in range(int(batch_size))]
        self.last_role = [None for _ in range(int(batch_size))]
        self.current_ep = [None for _ in range(int(batch_size))]
        self.extra_count = [0 for _ in range(int(batch_size))]
        self.max_extra_per_gate = int(getattr(tokenizer, 'max_extra_pins_per_gate', 4))
        self.driven_nets = [set() for _ in range(int(batch_size))]
        self.driven_stubs = [set() for _ in range(int(batch_size))]
        self.used_private_out = [set() for _ in range(int(batch_size))]
        self.stub_indices = [set() for _ in range(int(batch_size))]
        self.net_indices = [set() for _ in range(int(batch_size))]
        self.seen_gc_prefix = [False for _ in range(int(batch_size))]
        self.prefix_tokens = set()
        for attr in ("gc_tokens", "driver_stub_bucket_tokens", "load_stub_bucket_tokens", "driver_load_ratio_bucket_tokens"):
            tok_map = getattr(tokenizer, attr, None)
            if tok_map:
                self.prefix_tokens.update(int(v) for v in tok_map.values())
        self.eos_blocked_count = [0 for _ in range(int(batch_size))]
        self.eos_allowed_after_gate_count = [-1 for _ in range(int(batch_size))]
        idx = torch.arange(len(tokenizer), device=device)
        self.is_idx = (idx >= int(tokenizer.idx_offset)) & (idx < int(tokenizer.pin_offset))
        self.is_pin = (idx >= int(tokenizer.pin_offset)) & (idx < int(tokenizer.node_type_offset))
        self.is_cell = idx >= int(tokenizer.node_type_offset)
        self.stub_endpoint_tokens = set(int(x) for x in getattr(tokenizer, 'stub_endpoint_tokens', {getattr(tokenizer, 'ep_stub', -1)}))
        self.driver_stub_endpoint_tokens = set(int(x) for x in getattr(tokenizer, 'driver_stub_endpoint_tokens', {getattr(tokenizer, 'ep_stub', -1)}))
        self.load_stub_endpoint_tokens = set(int(x) for x in getattr(tokenizer, 'load_stub_endpoint_tokens', {getattr(tokenizer, 'ep_stub', -1)}))

    def _idx_value_from_token(self, tok: int, ep_tok=None) -> int:
        if hasattr(self.tok, 'idx_value_from_token'):
            return int(self.tok.idx_value_from_token(int(tok), ep_tok))
        return int(tok - int(self.tok.idx_offset))

    def _idx_token_for_value(self, idx_value: int, ep_tok=None) -> int:
        if hasattr(self.tok, '_tok_endpoint_idx') and ep_tok is not None:
            return int(self.tok._tok_endpoint_idx(int(ep_tok), int(idx_value)))
        return int(self.tok.idx_offset + int(idx_value))

    def _idx_mask_for_endpoint(self, ep_tok: int) -> torch.Tensor:
        if hasattr(self.tok, 'idx_mask_for_endpoint_token'):
            return self.tok.idx_mask_for_endpoint_token(int(ep_tok), device=self.device)
        return self.is_idx.clone().to(self.device)

    def _is_stub_endpoint(self, ep_tok: int) -> bool:
        return int(ep_tok) in self.stub_endpoint_tokens

    def _is_driver_stub_endpoint(self, ep_tok: int) -> bool:
        return int(ep_tok) in self.driver_stub_endpoint_tokens

    def _build_plan(self, label: int):
        cell_name = self.tok.label_to_cell.get(int(label))
        spec = None
        if cell_name is not None:
            spec = self.tok.pin_specs.get(str(cell_name))
        if spec is None:
            return []
        req_in = list(spec.inputs)
        out_pins = list(spec.outputs)
        min_out = int(spec.min_required_outputs)
        if min_out <= 0 and len(out_pins) > 0:
            min_out = 1
        req_out = out_pins[:min_out]
        opt_out = out_pins[min_out:]
        plan = []
        for p in req_in:
            plan.append(('IN', str(p)))
        for p in req_out:
            plan.append(('OUT_REQ', str(p)))
        for p in opt_out:
            plan.append(('OUT_OPT', str(p)))
        plan.append(('EXTRA', None))
        return plan

    def _mask_only(self, scores, mask: torch.Tensor):
        return torch.where(mask, scores, float('-inf'))

    def _fallback_finish(self, scores):
        allowed = torch.zeros((scores.shape[-1],), dtype=torch.bool, device=scores.device)
        if 0 <= int(self.tok.eos) < int(allowed.numel()):
            allowed[int(self.tok.eos)] = True
        if 0 <= int(self.tok.pad) < int(allowed.numel()):
            allowed[int(self.tok.pad)] = True
        return self._mask_only(scores, allowed)

    def _raise_invalid(self, b: int, reason: str):
        raise RuntimeError(
            f"PinSlotGrammar invalid (mode={self.mask_mode}) b={b} state={self.mode[b]} slot={self.slot_state[b]} plan_pos={self.plan_pos[b]} reason={reason}"
        )

    def _available_net_idx_mask(self, b: int, *, allow_driven: bool) -> torch.Tensor:
        mask = self._idx_mask_for_endpoint(getattr(self.tok, 'ep_net', None))
        if not allow_driven:
            for nid in self.driven_nets[b]:
                tid = self._idx_token_for_value(int(nid), getattr(self.tok, 'ep_net', None))
                if 0 <= tid < int(mask.numel()):
                    mask[int(tid)] = False
        return mask

    def _update_state(self, b: int, last: int):
        if last == int(self.tok.sos):
            self.mode[b] = 0
            self.gate_count[b] = 0
            self.extra_count[b] = 0
            self.stub_indices[b].clear()
            self.net_indices[b].clear()
            self.driven_nets[b].clear()
            self.driven_stubs[b].clear()
            self.used_private_out[b].clear()
            self.seen_gc_prefix[b] = False
            return

        if int(last) in self.prefix_tokens:
            self.seen_gc_prefix[b] = True
            return
        if self.mode[b] == 0:
            if last == int(self.tok.gate):
                self.mode[b] = 1
                self.gate_count[b] += 1
                self.extra_count[b] = 0
            return
        if self.mode[b] == 1:
            if last >= int(self.tok.idx_offset) and last < int(self.tok.pin_offset):
                self.mode[b] = 2
            return
        if self.mode[b] == 2:
            if last >= int(self.tok.node_type_offset):
                label = int(last - int(self.tok.node_type_offset))
                self.plan[b] = self._build_plan(label)
                self.plan_pos[b] = 0
                self.slot_state[b] = 0
                self.last_role[b] = None
                self.current_ep[b] = None
                self.mode[b] = 3
            return
        if self.mode[b] != 3:
            return

        if last == int(self.tok.end_gate):
            self.mode[b] = 0
            self.plan[b] = None
            self.plan_pos[b] = 0
            self.slot_state[b] = 0
            self.last_role[b] = None
            self.current_ep[b] = None
            self.extra_count[b] = 0
            return

        plan = self.plan[b] or []
        if not plan:
            return
        if self.slot_state[b] == 0:
            if last in {int(self.tok.pin_in), int(self.tok.pin_out), int(self.tok.pin_opt_out), int(self.tok.pin_skip), int(self.tok.pin_excess), int(self.tok.pin_unknown)}:
                self.last_role[b] = int(last)
                self.slot_state[b] = 1
            return
        if self.slot_state[b] == 1:
            if last >= int(self.tok.pin_offset) and last < int(self.tok.node_type_offset):
                if self.last_role[b] == int(self.tok.pin_skip):
                    self.plan_pos[b] = min(int(self.plan_pos[b] + 1), len(plan) - 1)
                    self.slot_state[b] = 0
                    self.last_role[b] = None
                else:
                    self.slot_state[b] = 2
            return
        if self.slot_state[b] == 2:
            self.current_ep[b] = int(last)
            self.slot_state[b] = 3
            return
        if self.slot_state[b] == 3:
            if last >= int(self.tok.idx_offset) and last < int(self.tok.pin_offset):
                idx_val = self._idx_value_from_token(int(last), self.current_ep[b])
                if self._is_stub_endpoint(int(self.current_ep[b])):
                    self.stub_indices[b].add(int(idx_val))
                elif self.current_ep[b] == int(self.tok.ep_net):
                    self.net_indices[b].add(int(idx_val))
            if self.current_ep[b] == int(self.tok.ep_net) and self.last_role[b] in {int(self.tok.pin_out), int(self.tok.pin_opt_out)}:
                if last >= int(self.tok.idx_offset) and last < int(self.tok.pin_offset):
                    self.driven_nets[b].add(self._idx_value_from_token(int(last), self.current_ep[b]))
            if self._is_driver_stub_endpoint(int(self.current_ep[b])) and self.last_role[b] in {int(self.tok.pin_out), int(self.tok.pin_opt_out)}:
                if last >= int(self.tok.idx_offset) and last < int(self.tok.pin_offset):
                    self.driven_stubs[b].add(self._idx_value_from_token(int(last), self.current_ep[b]))
            if self.current_ep[b] == int(self.tok.ep_private_out) and self.last_role[b] == int(self.tok.pin_out):
                if last >= int(self.tok.idx_offset) and last < int(self.tok.pin_offset):
                    self.used_private_out[b].add(self._idx_value_from_token(int(last), self.current_ep[b]))
            if (self.plan[b] or [])[min(self.plan_pos[b], len(plan) - 1)][0] != 'EXTRA':
                self.plan_pos[b] = min(int(self.plan_pos[b] + 1), len(plan) - 1)
            else:
                if self.last_role[b] in {int(self.tok.pin_excess), int(self.tok.pin_unknown)}:
                    self.extra_count[b] += 1
            self.slot_state[b] = 0
            self.last_role[b] = None
            self.current_ep[b] = None
            return

    def __call__(self, input_ids, scores):
        last = input_ids[:, -1]
        for b in range(int(last.shape[0])):
            self._update_state(b, int(last[b].item()))
            if self.mode[b] == 0:
                allowed = torch.zeros((scores.shape[-1],), dtype=torch.bool, device=scores.device)
                if int(self.gate_count[b]) < int(self.max_gates):
                    allowed[int(self.tok.gate)] = True
                if int(self.gate_count[b]) > 0 and int(self.gate_count[b]) >= int(self.target_min_gates):
                    allowed[int(self.tok.eos)] = True
                    if self.eos_allowed_after_gate_count[b] < 0:
                        self.eos_allowed_after_gate_count[b] = int(self.gate_count[b])
                elif int(self.gate_count[b]) > 0:
                    self.eos_blocked_count[b] += 1
                if int(self.gate_count[b]) == 0 and not bool(self.seen_gc_prefix[b]) and getattr(self.tok, 'gc_tokens', None) is not None:
                    for tid in self.prefix_tokens:
                        allowed[int(tid)] = True
                try:
                    self.tok.last_eos_diagnostics = {
                        "eos_blocked_count": [int(x) for x in self.eos_blocked_count],
                        "eos_allowed_after_gate_count": [int(x) for x in self.eos_allowed_after_gate_count],
                        "target_min_gates": int(self.target_min_gates),
                        "target_max_gates": int(self.target_max_gates),
                    }
                except Exception:
                    pass
                scores[b] = self._mask_only(scores[b], allowed)
                continue
            if self.mode[b] == 1:
                # Gate ids are not endpoint ids: they must be unique and canonical.
                # Allowing any NET_IDX here lets generation reuse an old gate id with a
                # different cell label, which later appears as a decode id collision.
                allowed = torch.zeros((scores.shape[-1],), dtype=torch.bool, device=scores.device)
                next_gate_idx = int(self.gate_count[b]) - 1
                if 0 <= next_gate_idx < int(getattr(self.tok, 'max_num_nodes', 0) or 0):
                    tid = self._idx_token_for_value(next_gate_idx, getattr(self.tok, 'ep_net', None))
                    if 0 <= tid < int(allowed.numel()):
                        allowed[int(tid)] = True
                scores[b] = self._mask_only(scores[b], allowed)
                continue
            if self.mode[b] == 2:
                scores[b] = self._mask_only(scores[b], self.is_cell)
                continue

            plan = self.plan[b] or []
            if not plan:
                continue
            kind, pname = plan[min(self.plan_pos[b], len(plan) - 1)]

            if self.slot_state[b] == 0:
                allowed = torch.zeros((scores.shape[-1],), dtype=torch.bool, device=scores.device)
                if kind == 'IN':
                    allowed[int(self.tok.pin_in)] = True
                elif kind == 'OUT_REQ':
                    allowed[int(self.tok.pin_out)] = True
                elif kind == 'OUT_OPT':
                    allowed[int(self.tok.pin_opt_out)] = True
                    allowed[int(self.tok.pin_skip)] = True
                else:
                    allowed[int(self.tok.end_gate)] = True
                    if int(self.extra_count[b]) < int(self.max_extra_per_gate):
                        allowed[int(self.tok.pin_excess)] = True
                        allowed[int(self.tok.pin_unknown)] = True
                scores[b] = self._mask_only(scores[b], allowed)
                continue

            if self.slot_state[b] == 1:
                allowed = torch.zeros((scores.shape[-1],), dtype=torch.bool, device=scores.device)
                if kind == 'EXTRA':
                    scores[b] = self._mask_only(scores[b], self.is_pin)
                    continue
                pid = self.tok.pin_name_to_id.get(str(pname))
                if pid is None:
                    scores[b] = self._mask_only(scores[b], self.is_pin)
                    continue
                allowed[int(self.tok.pin_offset + int(pid))] = True
                scores[b] = self._mask_only(scores[b], allowed)
                continue

            if self.slot_state[b] == 2:
                allowed = torch.zeros((scores.shape[-1],), dtype=torch.bool, device=scores.device)
                ep_load_stub = int(getattr(self.tok, 'ep_load_stub', getattr(self.tok, 'ep_stub', -1)))
                ep_driver_stub = int(getattr(self.tok, 'ep_driver_stub', getattr(self.tok, 'ep_stub', -1)))
                ep_opt_driver_stub = int(getattr(self.tok, 'ep_optional_driver_stub', ep_driver_stub))
                ep_unknown_stub = int(getattr(self.tok, 'ep_unknown_stub', getattr(self.tok, 'ep_stub', -1)))
                if kind == 'IN':
                    for t_id in [self.tok.ep_net, ep_load_stub]:
                        allowed[int(t_id)] = True
                    if self.mask_mode == 'practical_generation':
                        for t_id in [self.tok.ep_input_pool, self.tok.ep_pi]:
                            allowed[int(t_id)] = True
                elif kind == 'OUT_REQ':
                    net_idx_ok = bool(self._available_net_idx_mask(b, allow_driven=False).any().item())
                    if net_idx_ok:
                        for t_id in [self.tok.ep_net, ep_driver_stub]:
                            allowed[int(t_id)] = True
                    else:
                        if self.mask_mode == 'practical_generation':
                            allowed[int(self.tok.ep_private_out)] = True
                        else:
                            self._raise_invalid(b, 'no_legal_output_net')
                elif kind == 'OUT_OPT':
                    net_idx_ok = bool(self._available_net_idx_mask(b, allow_driven=False).any().item())
                    if net_idx_ok:
                        for t_id in [self.tok.ep_net, ep_opt_driver_stub]:
                            allowed[int(t_id)] = True
                    else:
                        for t_id in [ep_opt_driver_stub]:
                            allowed[int(t_id)] = True
                else:
                    for t_id in [self.tok.ep_net, ep_unknown_stub]:
                        allowed[int(t_id)] = True
                scores[b] = self._mask_only(scores[b], allowed)
                if not torch.isfinite(scores[b]).any().item():
                    if self.mask_mode == 'strict_native':
                        self._raise_invalid(b, f'endpoint_type_all_inf:{kind}')
                    scores[b] = self._fallback_finish(scores[b])
                continue

            if self.slot_state[b] == 3:
                if self.current_ep[b] == int(self.tok.ep_net):
                    allowed = self._idx_mask_for_endpoint(int(self.current_ep[b]))
                    if kind in {'OUT_REQ', 'OUT_OPT'}:
                        for nid in self.driven_nets[b]:
                            tid = self._idx_token_for_value(int(nid), int(self.current_ep[b]))
                            if 0 <= tid < allowed.numel():
                                allowed[tid] = False
                    for nid in self.stub_indices[b]:
                        tid = self._idx_token_for_value(int(nid), int(self.current_ep[b]))
                        if 0 <= tid < allowed.numel():
                            allowed[tid] = False
                    scores[b] = self._mask_only(scores[b], allowed)
                elif self.current_ep[b] == int(self.tok.ep_private_out) and kind == 'OUT_REQ':
                    allowed = self._idx_mask_for_endpoint(int(self.current_ep[b]))
                    for nid in self.used_private_out[b]:
                        tid = self._idx_token_for_value(int(nid), int(self.current_ep[b]))
                        if 0 <= tid < allowed.numel():
                            allowed[tid] = False
                    scores[b] = self._mask_only(scores[b], allowed)
                elif self._is_stub_endpoint(int(self.current_ep[b])):
                    allowed = self._idx_mask_for_endpoint(int(self.current_ep[b]))
                    for nid in self.net_indices[b]:
                        tid = self._idx_token_for_value(int(nid), int(self.current_ep[b]))
                        if 0 <= tid < allowed.numel():
                            allowed[tid] = False
                    if kind in {'OUT_REQ', 'OUT_OPT'}:
                        for nid in self.driven_stubs[b]:
                            tid = self._idx_token_for_value(int(nid), int(self.current_ep[b]))
                            if 0 <= tid < allowed.numel():
                                allowed[tid] = False
                    scores[b] = self._mask_only(scores[b], allowed)
                else:
                    scores[b] = self._mask_only(scores[b], self._idx_mask_for_endpoint(int(self.current_ep[b])))
                if not torch.isfinite(scores[b]).any().item():
                    if self.mask_mode == 'strict_native':
                        self._raise_invalid(b, f'endpoint_idx_all_inf:{kind}')
                    scores[b] = self._fallback_finish(scores[b])
                continue

        if not torch.isfinite(scores).any(dim=-1).all().item():
            if self.mask_mode == 'strict_native':
                self._raise_invalid(-1, 'row_all_inf')
            finite_rows = torch.isfinite(scores).any(dim=-1)
            for b in range(int(scores.shape[0])):
                if not bool(finite_rows[b].item()):
                    scores[b] = self._fallback_finish(scores[b])
        return scores


def apply_labeled_grammar_mask(logits, input_ids, tokenizer, net_id=None, boundary_stub_id=None):
    # logits: [B, T, V], input_ids: [B, T]
    bsz, seq_len, _ = logits.shape
    logits = logits.clone()
    processor = LabeledGraph(tokenizer, bsz, logits.device, net_id=net_id, boundary_stub_id=boundary_stub_id)
    for t in range(seq_len):
        scores = logits[:, t, :]
        prev = input_ids[:, t:t + 1]
        logits[:, t, :] = processor(prev, scores)
    return logits


def project_labeled_sequence(tokenizer, seq, *, strict: bool = False):
    if not getattr(tokenizer, "labeled_graph", False):
        return seq
    # Handle Data objects by projecting their serialized token sequence
    if isinstance(seq, Data):
        seq = tokenizer(seq)
    if isinstance(seq, (tuple, list)) and len(seq) == 2 and isinstance(seq[1], dict):
        # Tokenizers with aux_fields return (token_ids, aux_targets).
        seq = seq[0]
    if isinstance(seq, torch.Tensor):
        if seq.dim() == 0:
            seq = seq.reshape(1)
        elif seq.dim() > 1:
            seq = seq.reshape(-1)
        seq_list = seq.tolist()
        device = seq.device
        dtype = seq.dtype
    else:
        seq_list = list(seq)
        device = None
        dtype = None

    # Deterministic projection to the labeled SENT grammar:
    # idx -> node_label -> (edge_label -> idx -> node_label)* -> [ladj edge idx radj]* -> (reset|eos)
    EXPECT_IDX = 0
    EXPECT_NODE = 1
    EXPECT_AFTER_NODE = 2
    EXPECT_IDX_AFTER_EDGE = 3
    EXPECT_BR_EDGE = 4
    EXPECT_BR_IDX = 5
    EXPECT_BR_RBR = 6

    def is_idx(tok):
        return tokenizer.idx_offset <= tok < tokenizer.node_idx_offset

    def is_node(tok):
        return tokenizer.node_idx_offset <= tok < tokenizer.edge_idx_offset

    def is_edge(tok):
        return tok >= tokenizer.edge_idx_offset

    buf = []
    prefix = []
    complete_node_pairs = 0
    complete_edge_steps = 0
    seq_work = list(seq_list)
    iter_tokens = seq_work
    if seq_work and seq_work[0] == tokenizer.sos:
        prefix.append(seq_work[0])
        cursor = 1
        for pos in range(len(getattr(tokenizer, "schema_token_specs", []))):
            allowed = set(tokenizer.get_schema_token_ids_for_position(pos))
            if cursor < len(seq_work) and seq_work[cursor] in allowed:
                prefix.append(seq_work[cursor])
                cursor += 1
            else:
                break
        iter_tokens = seq_work[cursor:]

    state = EXPECT_IDX
    if prefix:
        buf.extend(prefix)
    for tok in iter_tokens:
        if tok == tokenizer.eos:
            if state in (EXPECT_AFTER_NODE, EXPECT_IDX):
                buf.append(tok)
                break
            continue
        if tok == tokenizer.reset:
            if state in (EXPECT_AFTER_NODE, EXPECT_IDX):
                buf.append(tok)
                state = EXPECT_IDX
            continue

        if state == EXPECT_IDX:
            if is_idx(tok):
                buf.append(tok)
                state = EXPECT_NODE
            continue
        if state == EXPECT_NODE:
            if is_node(tok):
                buf.append(tok)
                complete_node_pairs += 1
                state = EXPECT_AFTER_NODE
            continue
        if state == EXPECT_AFTER_NODE:
            if tok == tokenizer.ladj:
                buf.append(tok)
                state = EXPECT_BR_EDGE
            elif is_edge(tok):
                buf.append(tok)
                state = EXPECT_IDX_AFTER_EDGE
            continue
        if state == EXPECT_IDX_AFTER_EDGE:
            if is_idx(tok):
                buf.append(tok)
                complete_edge_steps += 1
                state = EXPECT_NODE
            continue
        if state == EXPECT_BR_EDGE:
            if is_edge(tok):
                buf.append(tok)
                state = EXPECT_BR_IDX
            continue
        if state == EXPECT_BR_IDX:
            if is_idx(tok):
                buf.append(tok)
                state = EXPECT_BR_RBR
            continue
        if state == EXPECT_BR_RBR:
            if tok == tokenizer.radj:
                buf.append(tok)
                complete_edge_steps += 1
                state = EXPECT_AFTER_NODE
            continue

    if complete_node_pairs <= 0:
        if strict:
            raise ValueError("labeled sequence projection produced no complete idx->node pair")
        fallback = list(prefix)
        fallback.extend([tokenizer.idx_offset, tokenizer.node_idx_offset, tokenizer.eos])
        buf = fallback
    elif complete_edge_steps <= 0:
        if strict:
            raise ValueError("labeled sequence projection produced no graph edges")
        if not buf or buf[-1] not in {tokenizer.eos, tokenizer.reset}:
            buf.append(tokenizer.eos)

    if device is not None:
        return torch.tensor(buf, dtype=dtype, device=device)
    return buf


def project_bipartite_graph(data, net_id=96, *, remove_isolates: bool = True, audit: bool = True):
    if not isinstance(data, Data):
        return data
    if data.x is None or data.edge_index is None or data.edge_index.numel() == 0:
        return data
    x = data.x.reshape(-1)
    edge_index = data.edge_index
    if edge_index.numel() == 0:
        return data
    u = edge_index[0]
    v = edge_index[1]
    n = int(x.numel())
    valid = (u >= 0) & (v >= 0) & (u < n) & (v < n)
    if valid.sum().item() != edge_index.size(1):
        edge_index = edge_index[:, valid]
        u = edge_index[0]
        v = edge_index[1]
        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
            data.edge_attr = data.edge_attr[valid]
    is_net_u = x[u] == net_id
    is_net_v = x[v] == net_id
    keep = is_net_u ^ is_net_v
    if keep.sum().item() == edge_index.size(1):
        return data
    dropped = int(edge_index.size(1) - keep.sum().item())
    edge_index = edge_index[:, keep]
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
        data.edge_attr = data.edge_attr[keep]
    if audit:
        data.bipartite_projection_dropped_edges = int(
            getattr(data, "bipartite_projection_dropped_edges", 0)
        ) + int(dropped)
    data.edge_index = edge_index
    if remove_isolates:
        from torch_geometric.utils import remove_isolated_nodes
        edge_attr = getattr(data, "edge_attr", None)
        edge_index, edge_attr, mask = remove_isolated_nodes(
            data.edge_index,
            edge_attr=edge_attr,
            num_nodes=int(x.numel()),
        )
        data.x = x[mask]
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.num_nodes = int(data.x.numel())
    return data
