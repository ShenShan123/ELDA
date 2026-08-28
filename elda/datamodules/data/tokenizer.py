import torch
from torch_geometric.data import Data
from .sent_utils_wrapper import (
    sample_sent_from_graph,
    sample_labeled_sent_from_graph,
    get_graph_from_sent,
    get_graph_from_labeled_sent,
)
from .batch_converter import BatchConverter


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


class Graph2TrailTokenizer(object):
    sos: int = 0
    reset: int = 1
    ladj: int = 2
    radj: int = 3
    eos: int = 4
    pad: int = 5
    special_toks = ['sos', 'reset', 'ladj', 'radj', 'eos', 'pad']

    def __init__(
        self,
        dataset_names=[],
        max_length=-1,
        truncation_length=None,
        labeled_graph=False,
        undirected=True,
        append_eos=True,
        rng=None,
        aux_fields=None,
        schema_token_specs=None,
        **kwargs
    ):
        self.dataset_names = dataset_names
        self.max_length = max_length
        self.undirected = undirected
        self.append_eos = append_eos
        self.truncation_length = truncation_length
        self.rng = rng
        self.aux_fields = list(aux_fields) if aux_fields is not None else []
        self.schema_token_specs = list(schema_token_specs) if schema_token_specs is not None else []
        if len(self.dataset_names) > 0:
            self.dataset_to_idx = {
                dataset_name: i + len(self.special_toks) for i, dataset_name in enumerate(self.dataset_names)
            }
        self.schema_token_ranges = {}
        schema_offset = len(self.special_toks) + len(self.dataset_names)
        for spec in self.schema_token_specs:
            name = str(spec["name"])
            count = int(spec["count"])
            self.schema_token_ranges[name] = (schema_offset, count)
            schema_offset += count
        self.schema_token_count = schema_offset - (len(self.special_toks) + len(self.dataset_names))
        self.idx_offset = schema_offset
        self.max_num_nodes = None
        self.labeled_graph = labeled_graph
        self.num_node_types = self.num_edge_types = 0

    def set_num_nodes(self, max_num_nodes):
        if (self.max_num_nodes is None) or (self.max_num_nodes < max_num_nodes):
            self.max_num_nodes = max_num_nodes

    def set_num_node_and_edge_types(self, num_node_types=0, num_edge_types=0):
        if self.labeled_graph:
            self.num_node_types = num_node_types
            self.num_edge_types = num_edge_types
            self.node_idx_offset = self.idx_offset + self.max_num_nodes
            self.edge_idx_offset = self.node_idx_offset + self.num_node_types

    def __len__(self):
        assert self.max_num_nodes is not None, "run self.set_num_nodes() first"
        if self.labeled_graph:
            return self.idx_offset + self.max_num_nodes + self.num_node_types + self.num_edge_types
        return self.idx_offset + self.max_num_nodes

    @property
    def has_schema_tokens(self):
        return self.schema_token_count > 0

    def get_schema_token_ids_for_position(self, position):
        if position < 0 or position >= len(self.schema_token_specs):
            return []
        name = str(self.schema_token_specs[position]["name"])
        start, count = self.schema_token_ranges[name]
        return list(range(start, start + count))

    def _schema_token_from_value(self, name, value):
        token_range = self.schema_token_ranges.get(name)
        if token_range is None:
            return None
        start, count = token_range
        if value is None:
            return None
        value = int(value)
        if value < 0:
            value = 0
        if value >= count:
            value = count - 1
        return start + value

    def _build_schema_prefix_tokens(self, data):
        if not self.has_schema_tokens:
            return []
        tokens = []
        for spec in self.schema_token_specs:
            name = str(spec["name"])
            attr_name = str(spec.get("attr", name))
            token = self._schema_token_from_value(name, getattr(data, attr_name, None))
            if token is None:
                return []
            tokens.append(int(token))
        return tokens

    def __call__(self, data):
        return self.tokenize(data)

    @staticmethod
    def _compute_graph_structure_stats(data):
        num_nodes = int(getattr(data, "num_nodes", 0) or 0)
        if num_nodes <= 0:
            return {
                "component_count": 0.0,
                "largest_component_ratio": 0.0,
                "avg_degree": 0.0,
            }
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            return {
                "component_count": float(num_nodes),
                "largest_component_ratio": 1.0 / float(num_nodes),
                "avg_degree": 0.0,
            }

        unique_edges = set()
        adjacency = [set() for _ in range(num_nodes)]
        for src, dst in edge_index.t().tolist():
            src = int(src)
            dst = int(dst)
            if src == dst or src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
                continue
            a, b = (src, dst) if src < dst else (dst, src)
            if (a, b) in unique_edges:
                continue
            unique_edges.add((a, b))
            adjacency[a].add(b)
            adjacency[b].add(a)

        visited = [False] * num_nodes
        comp_sizes = []
        for node in range(num_nodes):
            if visited[node]:
                continue
            stack = [node]
            visited[node] = True
            size = 0
            while stack:
                cur = stack.pop()
                size += 1
                for nxt in adjacency[cur]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        stack.append(nxt)
            comp_sizes.append(size)
        if not comp_sizes:
            comp_sizes = [num_nodes]
        largest = max(comp_sizes)
        avg_degree = (2.0 * float(len(unique_edges)) / float(num_nodes)) if num_nodes > 0 else 0.0
        return {
            "component_count": float(len(comp_sizes)),
            "largest_component_ratio": float(largest) / float(num_nodes),
            "avg_degree": avg_degree,
        }

    @staticmethod
    def _scalar_attr(data, attr, default=0.0):
        if not hasattr(data, attr):
            return float(default)
        value = getattr(data, attr)
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float(default)
            return float(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            if not value:
                return float(default)
            return float(value[0])
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _mean_attr(data, attr, default=0.0):
        if not hasattr(data, attr):
            return float(default)
        value = getattr(data, attr)
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float(default)
            return float(value.reshape(-1).to(torch.float32).mean().item())
        if isinstance(value, (list, tuple)):
            if not value:
                return float(default)
            return float(sum(float(item) for item in value) / float(len(value)))
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _clamp_unit(value):
        return max(0.0, min(1.0, float(value)))

    def _build_aux_targets(self, data):
        if not self.aux_fields:
            return None
        aux = {}
        structure = None
        for field in self.aux_fields:
            if field == "partition_projection_proxy":
                gate_pin = float(getattr(data, "gate_pin_deficit", 0.0))
                under = float(getattr(data, "underconnected_gate_fraction", 0.0))
                synth = float(getattr(data, "synthetic_input_proxy_ratio", 0.0))
                aux[field] = (gate_pin + under + synth) / 3.0
            elif field == "partition_gate_pin_deficit":
                aux[field] = float(getattr(data, "gate_pin_deficit", 0.0))
            elif field == "partition_underconnected_gate_fraction":
                aux[field] = float(getattr(data, "underconnected_gate_fraction", 0.0))
            elif field == "partition_synthetic_input_proxy_ratio":
                aux[field] = float(getattr(data, "synthetic_input_proxy_ratio", 0.0))
            elif field in {
                "skeleton_component_count_norm",
                "skeleton_largest_component_ratio",
                "skeleton_avg_degree_norm",
            }:
                if structure is None:
                    structure = self._compute_graph_structure_stats(data)
                if field == "skeleton_component_count_norm":
                    aux[field] = min(1.0, torch.log1p(torch.tensor(max(0.0, structure["component_count"] - 1.0))).item() / torch.log1p(torch.tensor(7.0)).item())
                elif field == "skeleton_largest_component_ratio":
                    aux[field] = float(structure["largest_component_ratio"])
                elif field == "skeleton_avg_degree_norm":
                    aux[field] = min(1.0, float(structure["avg_degree"]) / 6.0)
            elif field in PARTITION_CONTRACT_AUX_FIELDS:
                if field == "partition_driver_stub_supply_norm":
                    aux[field] = self._clamp_unit(self._scalar_attr(data, "schema_boundary_driver_stub_count") / 16.0)
                elif field == "partition_load_stub_supply_norm":
                    aux[field] = self._clamp_unit(self._scalar_attr(data, "schema_boundary_load_stub_count") / 16.0)
                elif field == "partition_bidir_stub_supply_norm":
                    aux[field] = self._clamp_unit(self._scalar_attr(data, "schema_boundary_bidir_stub_count") / 16.0)
                elif field == "partition_ambig_stub_ratio":
                    ambig = self._scalar_attr(data, "schema_boundary_ambig_stub_count")
                    total = (
                        self._scalar_attr(data, "schema_boundary_driver_stub_count")
                        + self._scalar_attr(data, "schema_boundary_load_stub_count")
                        + self._scalar_attr(data, "schema_boundary_bidir_stub_count")
                        + ambig
                    )
                    aux[field] = self._clamp_unit(ambig / max(1.0, total))
                elif field == "partition_boundary_role_entropy_norm":
                    aux[field] = self._clamp_unit(self._scalar_attr(data, "schema_boundary_role_entropy") / 5.0)
                elif field == "partition_projection_repair_budget_bucket_norm":
                    aux[field] = self._clamp_unit(self._scalar_attr(data, "schema_projection_repair_budget_bucket") / 5.0)
                elif field == "partition_net_export_cleanliness_bucket_norm":
                    aux[field] = self._clamp_unit(self._scalar_attr(data, "schema_net_export_cleanliness_bucket") / 5.0)
            elif field in SKELETON_CONTRACT_AUX_FIELDS:
                if field == "skeleton_driver_stub_supply_norm":
                    aux[field] = self._clamp_unit(self._mean_attr(data, "schema_node_driver_stub_count") / 16.0)
                elif field == "skeleton_load_stub_supply_norm":
                    aux[field] = self._clamp_unit(self._mean_attr(data, "schema_node_load_stub_count") / 16.0)
                elif field == "skeleton_bidir_stub_supply_norm":
                    aux[field] = self._clamp_unit(self._mean_attr(data, "schema_node_bidir_stub_count") / 16.0)
                elif field == "skeleton_ambig_stub_ratio":
                    ambig = self._mean_attr(data, "schema_node_ambig_stub_count")
                    total = (
                        self._mean_attr(data, "schema_node_driver_stub_count")
                        + self._mean_attr(data, "schema_node_load_stub_count")
                        + self._mean_attr(data, "schema_node_bidir_stub_count")
                        + ambig
                    )
                    aux[field] = self._clamp_unit(ambig / max(1.0, total))
            elif field in SKELETON_TYPED_EDGE_AUX_FIELDS:
                if field == "skeleton_node_driver_demand_mean_norm":
                    aux[field] = self._clamp_unit(self._mean_attr(data, "schema_node_driver_demand_count") / 16.0)
                elif field == "skeleton_node_load_demand_mean_norm":
                    aux[field] = self._clamp_unit(self._mean_attr(data, "schema_node_load_demand_count") / 16.0)
                elif field == "skeleton_node_shared_net_out_mean_norm":
                    aux[field] = self._clamp_unit(self._mean_attr(data, "schema_node_shared_net_out_count") / 16.0)
                elif field == "skeleton_node_shared_net_in_mean_norm":
                    aux[field] = self._clamp_unit(self._mean_attr(data, "schema_node_shared_net_in_count") / 16.0)
                elif field == "skeleton_node_contract_risk_mean_norm":
                    aux[field] = self._clamp_unit(self._mean_attr(data, "schema_node_contract_risk_score") / 100.0)
                elif field == "skeleton_edge_fanout_mean_norm":
                    aux[field] = self._clamp_unit(self._mean_attr(data, "schema_edge_fanout") / 16.0)
                elif field == "skeleton_edge_contract_risk_mean_norm":
                    aux[field] = self._clamp_unit(self._mean_attr(data, "schema_edge_contract_risk") / 100.0)
                elif field == "skeleton_edge_shared_net_group_count_mean_norm":
                    aux[field] = self._clamp_unit(self._mean_attr(data, "schema_edge_shared_net_group_count") / 16.0)
                elif field == "skeleton_typed_edge_enriched":
                    aux[field] = self._clamp_unit(self._scalar_attr(data, "schema_typed_edge_enriched"))
        return aux if aux else None

    def get_dataset_idx(self, dataset_name):
        return self.dataset_to_idx.get(dataset_name, None)

    def tokenize(self, data):
        if not data.is_coalesced():
            data = data.coalesce()
        if self.labeled_graph:
            walk_index, _ = sample_labeled_sent_from_graph(
                edge_index=data.edge_index,
                node_labels=data.x.flatten(),
                edge_labels=data.edge_attr,
                node_idx_offset=self.node_idx_offset,
                edge_idx_offset=self.edge_idx_offset,
                num_nodes=data.num_nodes,
                max_length=self.max_length,
                idx_offset=self.idx_offset,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                undirected=self.undirected,
                rng=self.rng
            )
        else:
            walk_index, _ = sample_sent_from_graph(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                max_length=self.max_length,
                idx_offset=self.idx_offset,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                undirected=self.undirected,
                rng=self.rng
            )

        schema_tokens = self._build_schema_prefix_tokens(data)

        start_offset = 1 # sos
        end_offset = 0
        if self.append_eos:
            end_offset += 1 # eos
        dataset_name_idx = None
        if hasattr(data, "dataset_name") and len(self.dataset_names) > 0:
            dataset_name_idx = self.get_dataset_idx(data.dataset_name)
            if dataset_name_idx is not None:
                start_offset += 1
        start_offset += len(schema_tokens)

        walk_index = torch.from_numpy(walk_index)

        walk_index_new = torch.zeros((walk_index.shape[0] + start_offset + end_offset,), dtype=walk_index.dtype)
        walk_index_new[0] = self.sos
        prefix_pos = 1
        if dataset_name_idx is not None:
            walk_index_new[prefix_pos] = dataset_name_idx
            prefix_pos += 1
        if schema_tokens:
            walk_index_new[prefix_pos:prefix_pos + len(schema_tokens)] = torch.tensor(schema_tokens, dtype=walk_index.dtype)
            prefix_pos += len(schema_tokens)
        if self.append_eos:
            walk_index_new[-1] = self.eos
            walk_index_new[start_offset:-1] = walk_index
        else:
            walk_index_new[start_offset:] = walk_index

        aux_targets = self._build_aux_targets(data)
        if aux_targets is not None:
            return walk_index_new, aux_targets
        return walk_index_new

    def decode(self, walk_index):
        walk_index = walk_index[(walk_index != self.pad) & (walk_index != self.sos) & (walk_index != self.eos)]
        dataset_name = None
        if (
            len(self.dataset_names) > 0
            and walk_index.numel() > 0
            and (walk_index[0] - len(self.special_toks) < len(self.dataset_names))
        ):
            dataset_name = self.dataset_names[walk_index[0] - len(self.special_toks)]
            walk_index = walk_index[1:]
        schema_attrs = {}
        if self.has_schema_tokens and walk_index.numel() > 0:
            schema_prefix_len = 0
            for spec in self.schema_token_specs:
                name = str(spec["name"])
                attr = str(spec.get("attr", f"schema_{name}"))
                start, count = self.schema_token_ranges[name]
                if schema_prefix_len >= walk_index.numel():
                    break
                tok = int(walk_index[schema_prefix_len].item())
                if start <= tok < start + count:
                    schema_attrs[attr] = int(tok - start)
                    schema_prefix_len += 1
                else:
                    break
            if schema_prefix_len > 0:
                walk_index = walk_index[schema_prefix_len:]
        if self.labeled_graph:
            edge_index, node_labels, edge_labels = get_graph_from_labeled_sent(
                walk_index=walk_index,
                idx_offset=self.idx_offset,
                node_idx_offset=self.node_idx_offset,
                edge_idx_offset=self.edge_idx_offset,
                num_node_types=self.num_node_types,
                num_edge_types=self.num_edge_types,
                reset=self.reset,
                ladj=self.ladj,
                radj=self.radj,
                undirected=self.undirected
            )
            if edge_index.numel() == 0:
                num_nodes = int(len(node_labels))
            else:
                num_nodes = int(edge_index.flatten().max().item() + 1)
            data = Data(
                x=node_labels,
                edge_index=edge_index,
                edge_attr=edge_labels,
                num_nodes=max(num_nodes, len(node_labels)),
                dataset_name=dataset_name
            )
            data.edge_role = data.edge_attr
            for attr, value in schema_attrs.items():
                setattr(data, attr, int(value))
            return data
        edge_index = get_graph_from_sent(
            walk_index=walk_index,
            idx_offset=self.idx_offset,
            reset=self.reset,
            ladj=self.ladj,
            radj=self.radj,
            undirected=self.undirected
        )
        if edge_index.numel() == 0:
            num_nodes = 0
        else:
            num_nodes = int(edge_index.flatten().max().item() + 1)
        data = Data(
            edge_index=edge_index,
            dataset_name=dataset_name,
            num_nodes=num_nodes
        )
        for attr, value in schema_attrs.items():
            setattr(data, attr, int(value))
        return data

    def batch_converter(self):
        return BatchConverter(self, self.truncation_length)
