"""Evaluation for circuit graph generation.

Metrics:
- W1 distance between real vs generated distributions:
  * degree (gate/net in bipartite, and gate-projection)
  * clustering coefficient (gate-projection)
  * 4-node orbit counts (gate-projection, ORCA)
- Scalar stats ratio E[M(G_hat)] / E[M(G)]:
  * triangle count (gate-projection)
  * h(A,X): type-homophily on gate-projection edges
  * h(A^2,X): type-homophily on 2-hop gate-gate relations via net nodes

Usage:
  python -m elda.evaluation.circuit_eval \
    --real /path/to/real_dir_or_pt \
    --gen /path/to/gen_dir_or_pt \
    --net-id 139 \
    --max-graphs 5000 \
    --out-json /tmp/circuit_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset

try:
    from scipy.stats import wasserstein_distance as _w1
except Exception:
    _w1 = None

try:
    from .spectre_utils import orca
    ORCA_AVAILABLE = True
except Exception:
    ORCA_AVAILABLE = False


class _MemDataset(InMemoryDataset):
    def __init__(self, data, slices):
        self.data = data
        self.slices = slices

    def len(self):
        return len(next(iter(self.slices.values()))) - 1


def _load_graphs(path: str, max_graphs: Optional[int] = None) -> List[Data]:
    p = Path(path)
    graphs: List[Data] = []
    if p.is_dir():
        files = sorted([f for f in p.iterdir() if f.suffix == '.pt'])
        if max_graphs:
            files = files[:max_graphs]
        for f in files:
            obj = torch.load(f, map_location='cpu', weights_only=False)
            if isinstance(obj, Data):
                graphs.append(obj)
            elif isinstance(obj, (list, tuple)) and len(obj) == 2:
                data, slices = obj
                ds = _MemDataset(data, slices)
                for i in range(len(ds)):
                    graphs.append(ds.get(i))
            else:
                continue
        return graphs

    # file
    obj = torch.load(p, map_location='cpu', weights_only=False)
    if isinstance(obj, Data):
        graphs = [obj]
    elif isinstance(obj, (list, tuple)) and len(obj) == 2:
        data, slices = obj
        ds = _MemDataset(data, slices)
        graphs = [ds.get(i) for i in range(len(ds))]
    return graphs[:max_graphs] if max_graphs else graphs


def _w1_distance(a: List[float], b: List[float]) -> float:
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    if _w1 is not None:
        return float(_w1(a, b))
    # fallback: match quantiles
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    qs = np.linspace(0, 1, max(len(a), len(b)), endpoint=True)
    a_q = np.interp(qs, np.linspace(0, 1, len(a), endpoint=True), a)
    b_q = np.interp(qs, np.linspace(0, 1, len(b), endpoint=True), b)
    return float(np.mean(np.abs(a_q - b_q)))


def _to_gate_projection(data: Data, net_id: int) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    x = data.x.reshape(-1).to(torch.long)
    edge_index = data.edge_index
    # Guard against inconsistent x/edge_index sizes
    if data.num_nodes is not None:
        num_nodes = int(data.num_nodes)
    else:
        num_nodes = int(x.numel())
    if x.numel() < num_nodes:
        pad = torch.full((num_nodes - x.numel(),), net_id, dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    elif x.numel() > num_nodes:
        x = x[:num_nodes]
    if edge_index.numel() > 0:
        edge_index = edge_index[:, (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)]
    is_gate = x != net_id
    gate_nodes = torch.where(is_gate)[0]
    net_nodes = torch.where(~is_gate)[0]

    # Build net -> gate neighbor list
    net_neighbors: Dict[int, List[int]] = {}
    for u, v in edge_index.t().tolist():
        if not is_gate[u] and is_gate[v]:
            net_neighbors.setdefault(u, []).append(v)
        elif not is_gate[v] and is_gate[u]:
            net_neighbors.setdefault(v, []).append(u)

    # gate index mapping
    gate_idx = {int(n.item()): i for i, n in enumerate(gate_nodes)}
    edges = set()
    max_clique_fanout = int(os.environ.get("CIRCUIT_GATE_PROJECTION_MAX_CLIQUE_FANOUT", "128"))
    for net, neigh in net_neighbors.items():
        # unique gate neighbors for this net
        uniq = list(dict.fromkeys(neigh))
        if len(uniq) < 2:
            continue
        if len(uniq) <= max_clique_fanout:
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    u = gate_idx[uniq[i]]
                    v = gate_idx[uniq[j]]
                    if u != v:
                        edges.add((min(u, v), max(u, v)))
        else:
            # Avoid O(k^2) clique blow-ups for clock/reset-like high-fanout
            # nets.  The star surrogate keeps the projected gate graph
            # connected through that signal while bounding edge count.
            hub = gate_idx[uniq[0]]
            for gate in uniq[1:]:
                v = gate_idx[gate]
                if hub != v:
                    edges.add((min(hub, v), max(hub, v)))

    G = nx.Graph()
    G.add_nodes_from(range(len(gate_nodes)))
    G.add_edges_from(edges)

    x_gate = x[gate_nodes].cpu().numpy()
    return G, x_gate, gate_nodes.cpu().numpy()


def _degree_samples_bipartite(data: Data, net_id: int) -> Tuple[List[int], List[int]]:
    x = data.x.reshape(-1).to(torch.long)
    edge_index = data.edge_index
    if data.num_nodes is not None:
        num_nodes = int(data.num_nodes)
    else:
        num_nodes = int(x.numel())
    deg = torch.bincount(edge_index.reshape(-1), minlength=num_nodes)
    if x.numel() < deg.numel():
        # pad x with net_id for missing labels
        pad = torch.full((deg.numel() - x.numel(),), net_id, dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    elif x.numel() > deg.numel():
        x = x[:deg.numel()]
    gate_deg = deg[x != net_id].tolist()
    net_deg = deg[x == net_id].tolist()
    return gate_deg, net_deg


def _degree_samples_graph(G: nx.Graph) -> List[int]:
    return [d for _, d in G.degree()]


def _clustering_samples(G: nx.Graph) -> List[float]:
    return list(nx.clustering(G).values())


def _triangle_count(G: nx.Graph) -> int:
    return int(sum(nx.triangles(G).values()) // 3)


def _h_ax(G: nx.Graph, x_gate: np.ndarray) -> float:
    # Homophily on gate-projection edges: fraction of edges connecting same type
    if G.number_of_edges() == 0:
        return 0.0
    same = 0
    for u, v in G.edges():
        if x_gate[u] == x_gate[v]:
            same += 1
    return same / G.number_of_edges()


def _h_a2x(data: Data, net_id: int) -> float:
    # Two-hop correlation via net nodes (weighted by shared nets)
    x = data.x.reshape(-1).to(torch.long)
    if data.num_nodes is not None:
        num_nodes = int(data.num_nodes)
    else:
        num_nodes = int(x.numel())
    if x.numel() < num_nodes:
        pad = torch.full((num_nodes - x.numel(),), net_id, dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    elif x.numel() > num_nodes:
        x = x[:num_nodes]
    is_gate = x != net_id
    gate_nodes = torch.where(is_gate)[0]
    gate_types = x[gate_nodes]

    # Build net -> gate neighbors
    net_neighbors: Dict[int, List[int]] = {}
    edge_index = data.edge_index
    if edge_index.numel() > 0:
        edge_index = edge_index[:, (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)]
    for u, v in edge_index.t().tolist():
        if not is_gate[u] and is_gate[v]:
            net_neighbors.setdefault(u, []).append(v)
        elif not is_gate[v] and is_gate[u]:
            net_neighbors.setdefault(v, []).append(u)

    total = 0
    same = 0
    # Count weighted pairs from each net
    gate_type_map = {int(n.item()): int(t.item()) for n, t in zip(gate_nodes, gate_types)}
    for neigh in net_neighbors.values():
        uniq = list(dict.fromkeys(neigh))
        if len(uniq) < 2:
            continue
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                total += 1
                if gate_type_map[uniq[i]] == gate_type_map[uniq[j]]:
                    same += 1
    return (same / total) if total > 0 else 0.0


def _orbit_4_counts(G: nx.Graph) -> Optional[np.ndarray]:
    if not ORCA_AVAILABLE:
        return None
    if G.number_of_nodes() == 0:
        return np.zeros(12, dtype=float)
    oc = orca(G)  # shape [n_nodes, 15]
    # 4-node orbits are indices 3..14 (12 orbits)
    counts = oc[:, 3:15].sum(axis=0)
    # normalize by nodes to compare across sizes
    return counts / max(G.number_of_nodes(), 1)


def _collect_metrics(graphs: List[Data], net_id: int) -> Dict[str, List[float]]:
    gate_deg_all: List[int] = []
    net_deg_all: List[int] = []
    proj_deg_all: List[int] = []
    proj_clust_all: List[float] = []
    tri_counts: List[int] = []
    h_ax_vals: List[float] = []
    h_a2x_vals: List[float] = []
    orbit_4_list: List[np.ndarray] = []

    for data in graphs:
        gdeg, ndeg = _degree_samples_bipartite(data, net_id)
        gate_deg_all.extend(gdeg)
        net_deg_all.extend(ndeg)

        Gp, x_gate, _ = _to_gate_projection(data, net_id)
        proj_deg_all.extend(_degree_samples_graph(Gp))
        proj_clust_all.extend(_clustering_samples(Gp))
        tri_counts.append(_triangle_count(Gp))
        h_ax_vals.append(_h_ax(Gp, x_gate))
        h_a2x_vals.append(_h_a2x(data, net_id))

        oc = _orbit_4_counts(Gp)
        if oc is not None:
            orbit_4_list.append(oc)

    metrics = {
        "gate_deg": gate_deg_all,
        "net_deg": net_deg_all,
        "proj_deg": proj_deg_all,
        "proj_clust": proj_clust_all,
        "tri_count": tri_counts,
        "h_ax": h_ax_vals,
        "h_a2x": h_a2x_vals,
    }
    if orbit_4_list:
        metrics["orbit4"] = orbit_4_list
    return metrics


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else float('nan')


def _load_mapping_max_label(mapping_path: Optional[str]) -> Optional[int]:
    if not mapping_path:
        return None
    path = Path(mapping_path)
    if not path.exists():
        return None
    namespace: Dict[str, object] = {}
    try:
        exec(path.read_text(), {}, namespace)
    except Exception:
        return None
    mapping = namespace.get("COMPLETE_CELL_TYPE_MAPPING")
    if not isinstance(mapping, dict) or not mapping:
        return None
    return max(int(v) for v in mapping.values())


def _resolve_net_id(cli_net_id: Optional[int], meta_path: Optional[str], mapping_path: Optional[str]) -> int:
    if cli_net_id is not None:
        return int(cli_net_id)
    env_value = os.environ.get("CIRCUIT_NET_ID")
    if env_value not in (None, ""):
        return int(env_value)
    if meta_path:
        try:
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            if isinstance(meta, dict) and "net_id" in meta:
                return int(meta["net_id"])
        except Exception:
            pass
    max_cell_label = _load_mapping_max_label(mapping_path)
    if max_cell_label is not None:
        return int(max_cell_label) + 1
    return 96


def evaluate(real_graphs: List[Data], gen_graphs: List[Data], net_id: int) -> Dict[str, float]:
    real = _collect_metrics(real_graphs, net_id)
    gen = _collect_metrics(gen_graphs, net_id)

    out: Dict[str, float] = {}

    # W1 distances for distributions
    out["w1_gate_degree"] = _w1_distance(real["gate_deg"], gen["gate_deg"])
    out["w1_net_degree"] = _w1_distance(real["net_deg"], gen["net_deg"])
    out["w1_proj_degree"] = _w1_distance(real["proj_deg"], gen["proj_deg"])
    out["w1_proj_clustering"] = _w1_distance(real["proj_clust"], gen["proj_clust"])

    # 4-node orbit distribution (average W1 across orbit dimensions)
    if "orbit4" in real and "orbit4" in gen and real["orbit4"] and gen["orbit4"]:
        real_orb = np.stack(real["orbit4"], axis=0)
        gen_orb = np.stack(gen["orbit4"], axis=0)
        w1_orb = []
        for i in range(real_orb.shape[1]):
            w1_orb.append(_w1_distance(real_orb[:, i].tolist(), gen_orb[:, i].tolist()))
        out["w1_orbit4_mean"] = float(np.mean(w1_orb))
        # also store max as a sanity check
        out["w1_orbit4_max"] = float(np.max(w1_orb))
    else:
        out["w1_orbit4_mean"] = float('nan')
        out["w1_orbit4_max"] = float('nan')

    # Scalar stats ratios
    out["ratio_tri_count"] = _mean(gen["tri_count"]) / max(_mean(real["tri_count"]), 1e-9)
    out["ratio_h_ax"] = _mean(gen["h_ax"]) / max(_mean(real["h_ax"]), 1e-9)
    out["ratio_h_a2x"] = _mean(gen["h_a2x"]) / max(_mean(real["h_a2x"]), 1e-9)

    # Also provide raw means for reference
    out["mean_tri_real"] = _mean(real["tri_count"])
    out["mean_tri_gen"] = _mean(gen["tri_count"])
    out["mean_h_ax_real"] = _mean(real["h_ax"])
    out["mean_h_ax_gen"] = _mean(gen["h_ax"])
    out["mean_h_a2x_real"] = _mean(real["h_a2x"])
    out["mean_h_a2x_gen"] = _mean(gen["h_a2x"])

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', required=True, help='real graphs dir or .pt file')
    parser.add_argument('--gen', required=True, help='generated graphs dir or .pt file')
    parser.add_argument('--net-id', type=int, default=None)
    parser.add_argument('--meta-path', default=None, help='dataset meta.pt containing net_id')
    parser.add_argument('--mapping-path', default='mapping.txt', help='mapping.txt fallback; net_id=max(cell_label)+1')
    parser.add_argument('--max-graphs', type=int, default=None)
    parser.add_argument('--out-json', default=None)
    args = parser.parse_args()

    real_graphs = _load_graphs(args.real, max_graphs=args.max_graphs)
    gen_graphs = _load_graphs(args.gen, max_graphs=args.max_graphs)

    if not real_graphs:
        raise SystemExit('No real graphs loaded')
    if not gen_graphs:
        raise SystemExit('No generated graphs loaded')

    net_id = _resolve_net_id(args.net_id, args.meta_path, args.mapping_path)
    res = evaluate(real_graphs, gen_graphs, net_id=net_id)

    print('Evaluation Results:')
    print(f'net_id: {net_id}')
    for k in sorted(res.keys()):
        print(f'{k}: {res[k]}')

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
