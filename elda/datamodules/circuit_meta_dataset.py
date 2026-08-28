import os
import os.path as osp
import json
from typing import List

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Dataset


class _BaseMetaCircuitDataset(InMemoryDataset):
    family = None

    def __init__(
        self,
        root,
        split='train',
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        if self.family is None:
            raise ValueError('family must be set on subclasses')
        self.dataset_root = root
        self.split = split
        self._require_dataset_ready(root)
        root = osp.join(root, self.family)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self._num_node_types = None
        self._num_edge_types = None
        self._init_type_counts()

    @classmethod
    def _require_dataset_ready(cls, dataset_root: str) -> None:
        meta_path = osp.join(dataset_root, 'meta.pt')
        if osp.exists(meta_path):
            return
        existing = []
        for rel in ['partitions', 'skeletons', 'manifests']:
            path = osp.join(dataset_root, rel)
            if osp.exists(path):
                existing.append(rel)
        if existing:
            raise RuntimeError(
                f'Meta dataset root is incomplete: expected {meta_path} but it does not exist yet. '
                f'Found partial outputs: {existing}. This usually means preprocess_circuit_kahypar_dataset.py '
                f'is still running or exited before finalizing meta.pt. Finish preprocessing first, then rerun training.'
            )
        raise RuntimeError(
            f'Meta dataset root not found or not initialized: expected {meta_path}. '
            f'Run preprocess_circuit_kahypar_dataset.py from the dataset-tools directory first.'
        )

    @property
    def meta_path(self):
        return osp.join(osp.dirname(self.root), 'meta.pt')

    @property
    def raw_dir(self):
        return osp.join(self.root, 'graphs')

    @property
    def raw_file_names(self):
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
        data = getattr(self, '_data', None)
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
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            return False
        num_nodes = data.num_nodes
        if num_nodes is None or num_nodes <= 0:
            return False
        if data.x.numel() == 0:
            return False
        if data.edge_index.numel() == 0:
            return False
        if data.edge_index.min().item() < 0:
            return False
        if data.edge_index.max().item() >= num_nodes:
            return False
        return True

    def download(self):
        return

    def _select_files(self) -> List[str]:
        meta = torch.load(self.meta_path, map_location='cpu', weights_only=False)
        split_file_lists = dict(meta.get('split_file_lists') or {})
        if self.split in split_file_lists:
            list_path = split_file_lists[self.split]
            if not osp.isabs(str(list_path)):
                list_path = osp.join(self.dataset_root, str(list_path))
            with open(list_path, 'r', encoding='utf-8') as handle:
                files = [line.strip() for line in handle if line.strip()]
            missing = [path for path in files if not osp.exists(path)]
            if missing:
                raise RuntimeError(
                    f'Clean split list for {self.split} contains {len(missing)} missing files; '
                    f'first={missing[0]}'
                )
            return files
        design_to_split = meta['splits']['design_to_split']
        files = sorted([osp.join(self.raw_dir, f) for f in os.listdir(self.raw_dir) if f.endswith('.pt')])
        selected = []
        for path in files:
            name = osp.basename(path)
            design_id = self._design_id_from_filename(name)
            if design_to_split.get(design_id) == self.split:
                selected.append(path)
        return selected

    def _design_id_from_filename(self, filename: str) -> str:
        raise NotImplementedError

    def process(self):
        files = self._select_files()
        if not files:
            raise RuntimeError(f'No .pt files found for split={self.split} in {self.raw_dir}')

        data_list = []
        shared_keys = None
        for path in files:
            data = torch.load(path, map_location='cpu', weights_only=False)
            if not self._is_valid_graph(data):
                continue
            data.edge_index = data.edge_index.to(torch.long)
            data.x = data.x.reshape(-1).to(torch.long)
            data.num_nodes = int(data.x.numel())
            if getattr(data, 'edge_attr', None) is None or data.edge_attr.numel() != data.edge_index.shape[1]:
                data.edge_attr = torch.zeros(data.edge_index.shape[1], dtype=torch.long)
            else:
                data.edge_attr = data.edge_attr.reshape(-1).to(torch.long)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            # PyG InMemoryDataset cannot reliably collate arbitrary Python list
            # sidecars.  V2/V2.1 pin-slot tokenizers still need pin_id_to_name;
            # other list-valued audit fields are not part of the training input.
            for key in list(data.keys()):
                if str(key) != "pin_id_to_name" and isinstance(data[str(key)], list):
                    del data[str(key)]
            keys = set(str(key) for key in data.keys())
            shared_keys = keys if shared_keys is None else (shared_keys & keys)
            data_list.append(data)

        if not data_list:
            raise RuntimeError(f'No valid graphs found for split={self.split} in {self.raw_dir}')
        if shared_keys is not None:
            # Some semantic datasets were enriched incrementally, so optional
            # schema attrs may only exist on a subset of graphs. Drop
            # non-shared attrs before PyG collate to avoid KeyError while
            # preserving the fields that are actually consistent dataset-wide.
            for data in data_list:
                for key in list(data.keys()):
                    if str(key) not in shared_keys:
                        del data[str(key)]
        torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{self.split}.pt'))


class CircuitPartitionDataset(_BaseMetaCircuitDataset):
    family = 'partitions'

    def __init__(
        self,
        root,
        split='train',
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        meta_path = osp.join(root, 'meta.pt')
        meta = torch.load(meta_path, map_location='cpu', weights_only=False) if osp.exists(meta_path) else {}
        dataset_name = str(meta.get('dataset_name', ''))
        self._elda_profile_dataset = dataset_name == 'ELDA_PROFILE'
        use_file_backed = (
            os.environ.get('ELDA_FILE_BACKED_PARTITIONS', '').strip() == '1'
            or dataset_name == 'CIRCUIT_PIN_SLOT_PARTITION_V2_1_GC_FULL'
            or bool(meta.get('file_backed_partitions', False))
        )
        if not use_file_backed:
            super().__init__(root, split, transform, pre_transform, pre_filter)
            return
        self.dataset_root = root
        self.split = split
        self._require_dataset_ready(root)
        self.root = osp.join(root, self.family)
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self._paths = self._select_files()
        if not self._paths:
            raise RuntimeError(f'No .pt files found for split={self.split} in {self.raw_dir}')
        self._num_node_types = int(meta.get('boundary_stub_id', 140)) + 1
        self._num_edge_types = 4

    def _design_id_from_filename(self, filename: str) -> str:
        design_id = filename.rsplit('__p', 1)[0]
        # Derived sampled subsets may prefix graph files with a stable numeric
        # export id, e.g. 000123__design_name__p7.pt.  The canonical split map
        # is keyed by design_name, so strip only that numeric prefix.
        prefix, sep, rest = design_id.partition('__')
        if sep and prefix.isdigit() and rest:
            return rest
        return design_id

    def __len__(self):
        if hasattr(self, '_paths'):
            return len(self._paths)
        return super().__len__()

    def get(self, idx: int) -> Data:
        if not hasattr(self, '_paths'):
            return super().get(idx)
        path = self._paths[int(idx)]
        data = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(data, Data):
            raise TypeError(f'expected torch_geometric.data.Data in {path}')
        data.edge_index = data.edge_index.to(torch.long)
        data.x = data.x.reshape(-1).to(torch.long)
        data.num_nodes = int(data.x.numel())
        if getattr(self, '_elda_profile_dataset', False):
            data.elda_profile_split = str(self.split)
        if getattr(data, 'edge_attr', None) is None or data.edge_attr.numel() != data.edge_index.shape[1]:
            data.edge_attr = torch.zeros(data.edge_index.shape[1], dtype=torch.long)
        else:
            data.edge_attr = data.edge_attr.reshape(-1).to(torch.long)
        if self.pre_filter is not None and not self.pre_filter(data):
            return self.get((int(idx) + 1) % len(self._paths))
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __getitem__(self, idx: int) -> Data:
        if hasattr(self, '_paths'):
            return self.get(idx)
        return super().__getitem__(idx)


class CircuitSourceNetV4NetsectionOnlyDataset(Dataset):
    """Lazy teacher-cell NET_SECTION view backed by the original graph files."""

    def __init__(
        self,
        root,
        split='train',
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_root = root
        self.split = split
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        meta_path = osp.join(root, 'meta.pt')
        if not osp.exists(meta_path):
            raise RuntimeError(f'NETSECTION-only view missing meta.pt: {meta_path}')
        self.meta = torch.load(meta_path, map_location='cpu', weights_only=False)
        index_path = self.meta.get('netsection_only_index') or osp.join(root, 'netsection_only_index.jsonl')
        self.index_path = str(index_path)
        if not osp.exists(self.index_path):
            raise RuntimeError(f'NETSECTION-only view missing index: {self.index_path}')
        source_root = self.meta.get('source_dataset_root') or self.meta.get('dataset_root')
        source_meta_path = osp.join(str(source_root), 'meta.pt') if source_root else ''
        source_meta = torch.load(source_meta_path, map_location='cpu', weights_only=False) if source_meta_path and osp.exists(source_meta_path) else {}
        self.design_to_split = source_meta.get('splits', {}).get('design_to_split', {})
        self._paths = []
        with open(self.index_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                path = str(row.get('source_graph_path') or row.get('path') or '')
                if not path:
                    continue
                if self.design_to_split:
                    design_id = self._design_id_from_filename(osp.basename(path))
                    if self.design_to_split.get(design_id) != self.split:
                        continue
                self._paths.append(path)
        if not self._paths:
            raise RuntimeError(f'No NETSECTION-only graph paths found for split={self.split}')
        self._num_node_types = int(self.meta.get('boundary_stub_id', 140)) + 1
        self._num_edge_types = 4

    @property
    def num_node_types(self):
        return self._num_node_types

    @property
    def num_edge_types(self):
        return self._num_edge_types

    @staticmethod
    def _design_id_from_filename(filename: str) -> str:
        design_id = filename.rsplit('__p', 1)[0]
        prefix, sep, rest = design_id.partition('__')
        if sep and prefix.isdigit() and rest:
            return rest
        return design_id

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx: int) -> Data:
        path = self._paths[int(idx)]
        data = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(data, Data):
            raise TypeError(f'expected torch_geometric.data.Data in {path}')
        data.edge_index = data.edge_index.to(torch.long)
        data.x = data.x.reshape(-1).to(torch.long)
        data.num_nodes = int(data.x.numel())
        if getattr(data, 'edge_attr', None) is None or data.edge_attr.numel() != data.edge_index.shape[1]:
            data.edge_attr = torch.zeros(data.edge_index.shape[1], dtype=torch.long)
        else:
            data.edge_attr = data.edge_attr.reshape(-1).to(torch.long)
        if self.pre_filter is not None and not self.pre_filter(data):
            return self.__getitem__((int(idx) + 1) % len(self._paths))
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        if self.transform is not None:
            data = self.transform(data)
        return data


class CircuitSkeletonDataset(_BaseMetaCircuitDataset):
    """File-backed skeleton dataset.

    Skeleton graphs can be extremely large (e.g. hundreds of millions of
    edges). Collating and saving them into a single InMemoryDataset processed
    file can exceed filesystem/PyTorch zip writer limits. This dataset therefore
    keeps the existing semantic dataset layout intact and loads graphs lazily.
    """

    family = 'skeletons'

    def __init__(
        self,
        root,
        split='train',
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_root = root
        self.split = split
        self._require_dataset_ready(root)
        self.root = osp.join(root, self.family)
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self._paths = self._select_files()
        if not self._paths:
            raise RuntimeError(f'No .pt files found for split={self.split} in {self.raw_dir}')
        self._num_node_types = None
        self._num_edge_types = None
        self._init_type_counts_streaming()

    @property
    def meta_path(self):
        return osp.join(osp.dirname(self.root), 'meta.pt')

    @property
    def raw_dir(self):
        return osp.join(self.root, 'graphs')

    @property
    def num_node_types(self):
        return self._num_node_types

    @property
    def num_edge_types(self):
        return self._num_edge_types

    def _design_id_from_filename(self, filename: str) -> str:
        return osp.splitext(filename)[0]

    def _init_type_counts_streaming(self):
        max_node = -1
        max_edge = -1
        for path in self._paths:
            data = torch.load(path, map_location='cpu', weights_only=False)
            if getattr(data, 'x', None) is not None and data.x.numel() > 0:
                x = data.x.reshape(-1)
                max_node = max(max_node, int(x.max().item()))
            if getattr(data, 'edge_attr', None) is not None and data.edge_attr.numel() > 0:
                ea = data.edge_attr.reshape(-1)
                max_edge = max(max_edge, int(ea.max().item()))
        self._num_node_types = int(max_node + 1) if max_node >= 0 else 0
        self._num_edge_types = int(max_edge + 1) if max_edge >= 0 else 0

    def __len__(self):
        return len(self._paths)

    def get(self, idx: int) -> Data:
        path = self._paths[int(idx)]
        data = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(data, Data):
            raise TypeError(f'expected torch_geometric.data.Data in {path}')
        data.edge_index = data.edge_index.to(torch.long)
        data.x = data.x.reshape(-1).to(torch.long)
        data.num_nodes = int(data.x.numel())
        if getattr(data, 'edge_attr', None) is None or data.edge_attr.numel() != data.edge_index.shape[1]:
            data.edge_attr = torch.zeros(data.edge_index.shape[1], dtype=torch.long)
        else:
            data.edge_attr = data.edge_attr.reshape(-1).to(torch.long)
        if self.pre_filter is not None and not self.pre_filter(data):
            return self.get((int(idx) + 1) % len(self._paths))
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __getitem__(self, idx: int) -> Data:
        return self.get(idx)
