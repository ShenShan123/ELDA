import os
import glob
import torch
import numpy as np
from torch_geometric import utils
import pyximport

# Optional pure-Python fallback to avoid C-extension crashes during debugging.
_PURE_PY = os.environ.get("SENT_UTILS_PURE_PY", "0") == "1"
_PURE_PY_DECODE = os.environ.get("SENT_UTILS_DECODE_PURE_PY", "0") == "1"
_ALLOW_UNLABELED_PURE_PY = os.environ.get("SENT_UTILS_ALLOW_UNLABELED_PURE_PY", "0") == "1"


def _raise_pure_py_sent_error(context):
    raise RuntimeError(
        f"{context}: SENT_UTILS_PURE_PY=1 would use a non-topological pure-Python "
        "fallback. That fallback is not semantically equivalent to SENT and can "
        "silently train on node-order/counting artifacts instead of graph structure. "
        "Unset SENT_UTILS_PURE_PY, rebuild sent_utils.pyx, or set "
        "SENT_UTILS_ALLOW_UNLABELED_PURE_PY=1 only for explicit debugging."
    )


def _ensure_fresh_sent_utils_binary():
    """Force rebuild when sent_utils.pyx is newer than compiled shared object."""
    this_dir = os.path.dirname(__file__)
    pyx_path = os.path.join(this_dir, "sent_utils.pyx")
    if not os.path.exists(pyx_path):
        return
    pyx_mtime = os.path.getmtime(pyx_path)
    so_glob = os.path.join(this_dir, "sent_utils*.so")
    for so_path in glob.glob(so_glob):
        try:
            so_mtime = os.path.getmtime(so_path)
        except OSError:
            continue
        if pyx_mtime > so_mtime:
            try:
                os.remove(so_path)
            except OSError:
                pass

if not _PURE_PY:
    _ensure_fresh_sent_utils_binary()
    pyximport.install(setup_args={"include_dirs": np.get_include()}, inplace=True)
    from .sent_utils import (
        sample_sent,
        sample_labeled_sent,
        reconstruct_graph_from_sent,
        reconstruct_graph_from_labeled_sent
    )

def _py_sample_sent(num_nodes, max_length, idx_offset):
    if not _ALLOW_UNLABELED_PURE_PY:
        _raise_pure_py_sent_error("sample_sent_from_graph")
    if max_length is None or max_length < 0:
        max_length = num_nodes
    n = min(num_nodes, max_length)
    walk = np.arange(n, dtype=np.int64) + idx_offset
    return walk, None

def _py_sample_labeled_sent(node_labels, max_length, node_idx_offset):
    raise RuntimeError(
        "SENT_UTILS_PURE_PY is not semantically equivalent for labeled graphs. "
        "Unset SENT_UTILS_PURE_PY for circuit partition/skeleton training and sampling."
    )


def sample_sent_from_graph(
    edge_index,
    num_nodes=None,
    max_length=-1,
    idx_offset=0,
    reset=-1,
    ladj=-2,
    radj=-3,
    undirected=True,
    rng=None,
):
    if _PURE_PY:
        return _py_sample_sent(num_nodes if num_nodes is not None else 0, max_length, idx_offset)
    if rng is None:
        rng = np.random.mtrand._rand
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    csr_matrix = utils.to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes
    ).astype(np.int32).tocsr()
    return sample_sent(
        csr_matrix, max_length, idx_offset, reset, ladj, radj, undirected, rng
    )


def get_graph_from_sent(walk_index, idx_offset, reset, ladj, radj, undirected=True):
    if _PURE_PY:
        if not _ALLOW_UNLABELED_PURE_PY:
            _raise_pure_py_sent_error("get_graph_from_sent")
        # Minimal reconstruction: return empty edge_index for debugging only.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return edge_index
    device = walk_index.device
    walk_index = walk_index.cpu().numpy()
    edge_index = reconstruct_graph_from_sent(walk_index, reset, ladj, radj)
    edge_index = torch.from_numpy(edge_index)
    if undirected:
        edge_index_sym = torch.cat([edge_index[[1]], edge_index[[0]]])
        edge_index = torch.cat([edge_index, edge_index_sym], dim=1)
    edge_index,_ = utils.remove_self_loops(edge_index)
    edge_index, _, _ = utils.remove_isolated_nodes(edge_index)
    edge_index = utils.coalesce(edge_index)
    return edge_index


def sample_labeled_sent_from_graph(
    edge_index,
    node_labels,
    edge_labels,
    node_idx_offset=0,
    edge_idx_offset=0,
    num_nodes=None,
    max_length=-1,
    idx_offset=0,
    reset=-1,
    ladj=-2,
    radj=-3,
    undirected=True,
    rng=None,
):
    if _PURE_PY:
        return _py_sample_labeled_sent(node_labels, max_length, node_idx_offset)
    if rng is None:
        rng = np.random.mtrand._rand
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    csr_matrix = utils.to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes
    ).astype(np.int32).tocsr()
    if isinstance(node_labels, torch.Tensor):
        node_labels = node_labels.numpy()
    if isinstance(edge_labels, torch.Tensor):
        edge_labels = edge_labels.numpy()
    return sample_labeled_sent(
        csr_matrix, node_labels, edge_labels, node_idx_offset, edge_idx_offset,
        max_length, idx_offset, reset, ladj, radj, undirected, rng
    )

def get_graph_from_labeled_sent(
    walk_index, idx_offset, node_idx_offset, edge_idx_offset,
    num_node_types, num_edge_types,
    reset, ladj, radj, undirected=True
):
    reconstruct_fn = globals().get("reconstruct_graph_from_labeled_sent", None)
    if _PURE_PY:
        raise RuntimeError(
            "SENT_UTILS_PURE_PY cannot decode labeled graphs. "
            "Unset SENT_UTILS_PURE_PY for circuit partition/skeleton training and sampling."
        )
    if reconstruct_fn is None:
        raise RuntimeError("labeled SENT decoder is unavailable")
    device = walk_index.device
    walk_index = walk_index.cpu().numpy()
    try:
        edge_index, node_labels, edge_labels = reconstruct_fn(
            walk_index, reset, ladj, radj, idx_offset,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to decode labeled SENT sequence: {exc}") from exc
    if edge_index.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        node_labels = torch.empty((0,), dtype=torch.long)
        edge_labels = torch.empty((0,), dtype=torch.long)
        return edge_index, node_labels, edge_labels
    # Keep only edges whose endpoints are valid node-index tokens.
    valid_tok = (
        (edge_index[0] >= idx_offset) & (edge_index[0] < node_idx_offset)
        & (edge_index[1] >= idx_offset) & (edge_index[1] < node_idx_offset)
    )
    if valid_tok.sum() != edge_index.shape[1]:
        edge_index = edge_index[:, valid_tok]
        edge_labels = edge_labels[valid_tok]
    if edge_index.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        node_labels = torch.empty((0,), dtype=torch.long)
        edge_labels = torch.empty((0,), dtype=torch.long)
        return edge_index, node_labels, edge_labels

    idx_tokens = walk_index[(walk_index >= idx_offset) & (walk_index < node_idx_offset)]
    if idx_tokens.size > 0:
        max_node_idx = int(idx_tokens.max()) + 1
    else:
        max_node_idx = idx_offset
    max_node_idx = min(max_node_idx, node_labels.shape[0])
    node_labels = node_labels[idx_offset:max_node_idx]
    edge_index = torch.from_numpy(edge_index)
    node_labels = torch.from_numpy(node_labels)
    edge_labels = torch.from_numpy(edge_labels)
    edge_index -= idx_offset
    node_labels -= node_idx_offset
    edge_labels -= edge_idx_offset
    valid_node_labels = (node_labels >= 0) & (node_labels < num_node_types)
    if edge_index.numel() > 0:
        valid = (
            (edge_index[0] >= 0) & (edge_index[1] >= 0)
            & (edge_index[0] < node_labels.numel())
            & (edge_index[1] < node_labels.numel())
        )
        if valid.any():
            valid = valid & valid_node_labels[edge_index[0].clamp(0, max(0, node_labels.numel() - 1))]
            valid = valid & valid_node_labels[edge_index[1].clamp(0, max(0, node_labels.numel() - 1))]
        if valid.sum().item() != edge_index.size(1):
            edge_index = edge_index[:, valid]
            edge_labels = edge_labels[valid]
    if valid_node_labels.any():
        old_to_new = torch.full((node_labels.numel(),), -1, dtype=torch.long)
        keep_nodes = torch.nonzero(valid_node_labels, as_tuple=False).flatten()
        old_to_new[keep_nodes] = torch.arange(keep_nodes.numel(), dtype=torch.long)
        node_labels = node_labels[keep_nodes]
        if edge_index.numel() > 0:
            edge_index = old_to_new[edge_index]
            valid = (edge_index[0] >= 0) & (edge_index[1] >= 0)
            if valid.sum().item() != edge_index.size(1):
                edge_index = edge_index[:, valid]
                edge_labels = edge_labels[valid]
    else:
        node_labels = torch.empty((0,), dtype=torch.long)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_labels = torch.empty((0,), dtype=torch.long)
    edge_labels[(edge_labels < 0) | (edge_labels >= num_edge_types)] = 0
    if undirected:
        edge_index_sym = torch.cat([edge_index[[1]], edge_index[[0]]])
        edge_index = torch.cat([edge_index, edge_index_sym], dim=1)
        edge_labels = torch.cat([edge_labels, edge_labels])
    edge_index, edge_labels = utils.remove_self_loops(edge_index, edge_labels)
    edge_index, edge_labels = utils.coalesce(edge_index, edge_labels, reduce='min')
    return edge_index, node_labels, edge_labels
