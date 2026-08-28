import os
import os.path as osp
from typing import List
import random

import torch
from torch_geometric.data import Data, InMemoryDataset


def _list_pt_files(raw_dir: str) -> List[str]:
    return sorted(
        [osp.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.pt')]
    )


class CircuitDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        split='train',
        transform=None,
        pre_transform=None,
        pre_filter=None,
        split_ratios=(0.9, 0.05, 0.05),
        seed=42,
    ):
        self.split = split
        self.split_ratios = split_ratios
        self.seed = seed
        root = f'{root}/circuit'
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        self.file_idx = file_idx[split]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self._num_node_types = None
        self._num_edge_types = None
        self._init_type_counts()

    @property
    def raw_file_names(self):
        # No required raw filenames; processing scans raw_dir for .pt files
        return []

    @property
    def processed_file_names(self):
        return [self.split + '.pt']

    @property
    def num_node_types(self):
        return self._num_node_types

    @property
    def num_edge_types(self):
        return self._num_edge_types

    def _init_type_counts(self):
        data = getattr(self, "_data", None)
        if data is None:
            data = self.data
        if getattr(data, 'x', None) is not None and data.x.numel() > 0:
            self._num_node_types = int(data.x.max().item()) + 1
        else:
            self._num_node_types = 0
        if getattr(data, 'edge_attr', None) is not None and data.edge_attr.numel() > 0:
            self._num_edge_types = int(data.edge_attr.max().item()) + 1
        else:
            self._num_edge_types = 0

    @staticmethod
    def _is_valid_graph(data: Data) -> bool:
        if not isinstance(data, Data):
            return False
        if data.x is None or data.edge_index is None:
            return False
        if data.edge_index.numel() == 0:
            return False
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            return False
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes <= 0:
            return False
        if data.x.numel() == 0:
            return False
        if data.edge_index.min().item() < 0:
            return False
        if data.edge_index.max().item() >= num_nodes:
            return False
        return True

    def download(self):
        # No download; expect raw_dir to be populated by user/script
        return

    def process(self):
        files = _list_pt_files(self.raw_dir)
        if not files:
            raise RuntimeError(f'No .pt files found in {self.raw_dir}')

        rng = random.Random(self.seed)
        rng.shuffle(files)

        n = len(files)
        n_train = int(n * self.split_ratios[0])
        n_val = int(n * self.split_ratios[1])
        splits = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val],
            'test': files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            data_list = []
            for path in split_files:
                data = torch.load(path, map_location='cpu', weights_only=False)
                if not self._is_valid_graph(data):
                    continue
                # Normalize graph to avoid Cython sampler crashes.
                data.edge_index = data.edge_index.to(torch.long)
                data.x = data.x.reshape(-1).to(torch.long)
                data.num_nodes = int(data.x.numel())
                data.edge_attr = torch.zeros(data.edge_index.shape[1], dtype=torch.long)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            out_path = osp.join(self.processed_dir, f'{split_name}.pt')
            torch.save(self.collate(data_list), out_path)
