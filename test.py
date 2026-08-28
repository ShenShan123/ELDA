import os
import logging
from pathlib import Path
import hydra
from pyprojroot import here
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from elda.models.seq_models import SequenceModel
from elda.datamodules.circuit_dataset import CircuitDataset
from elda.evaluation.circuit_eval import _to_gate_projection, _triangle_count

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OmegaConf.register_new_resolver('eval', eval)

log = logging.getLogger(__name__)


def _load_mapping_max_label(mapping_path):
    path = Path(mapping_path)
    if not path.exists():
        return None
    namespace = {}
    try:
        exec(path.read_text(), {}, namespace)
    except Exception:
        return None
    mapping = namespace.get("COMPLETE_CELL_TYPE_MAPPING")
    if not isinstance(mapping, dict) or not mapping:
        return None
    return max(int(v) for v in mapping.values())


def _candidate_meta_paths(cfg):
    paths = []
    datamodule = getattr(cfg, "datamodule", None)
    model = getattr(cfg, "model", None)
    for obj in (datamodule, model):
        if obj is None:
            continue
        for key in ("skeleton_meta_path", "meta_path"):
            value = getattr(obj, key, None)
            if value:
                paths.append(Path(str(value)))
    root = getattr(datamodule, "root", None) if datamodule is not None else None
    if root:
        paths.append(Path(str(root)) / "meta.pt")
    return paths


def resolve_circuit_net_id(cfg):
    env_value = os.environ.get("CIRCUIT_NET_ID")
    if env_value not in (None, ""):
        return int(env_value)

    for meta_path in _candidate_meta_paths(cfg):
        if not meta_path.exists():
            continue
        try:
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if isinstance(meta, dict) and "net_id" in meta:
            net_id = int(meta["net_id"])
            os.environ["CIRCUIT_NET_ID"] = str(net_id)
            log.info("Resolved CIRCUIT_NET_ID=%s from %s", net_id, meta_path)
            return net_id

    mapping_path = getattr(getattr(cfg, "datamodule", None), "mapping_path", None)
    mapping_path = mapping_path or str(here() / "mapping.txt")
    max_cell_label = _load_mapping_max_label(mapping_path)
    if max_cell_label is not None:
        net_id = int(max_cell_label) + 1
        os.environ["CIRCUIT_NET_ID"] = str(net_id)
        log.warning(
            "Resolved CIRCUIT_NET_ID=%s as max(mapping)+1 from %s; prefer dataset meta.pt when available",
            net_id,
            mapping_path,
        )
        return net_id

    log.warning("Falling back to legacy CIRCUIT_NET_ID=96; diagnostics may be wrong for expanded mappings")
    return 96

def _summarize(values):
    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def _graph_stats(data, net_id):
    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    if x is None or x.numel() == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0.0,
            "gate_ratio": float("nan"),
            "gate_proj_edges": 0,
            "triangles": 0,
            "empty_x": True,
            "empty_edges": True,
        }
    x = x.reshape(-1)
    n = int(x.numel())
    gate_ratio = float((x != net_id).float().mean().item())

    if edge_index is None or edge_index.numel() == 0 or n < 2:
        return {
            "num_nodes": n,
            "num_edges": 0,
            "density": 0.0,
            "gate_ratio": gate_ratio,
            "gate_proj_edges": 0,
            "triangles": 0,
            "empty_x": False,
            "empty_edges": True,
        }

    u = edge_index[0].tolist()
    v = edge_index[1].tolist()
    edges = set()
    for a, b in zip(u, v):
        if a == b:
            continue
        edges.add((a, b) if a < b else (b, a))
    e = len(edges)
    density = 2.0 * e / (n * (n - 1))

    try:
        G, _, _ = _to_gate_projection(data, net_id)
        gate_proj_edges = int(G.number_of_edges())
        triangles = int(_triangle_count(G))
    except Exception:
        gate_proj_edges = 0
        triangles = 0

    return {
        "num_nodes": n,
        "num_edges": e,
        "density": density,
        "gate_ratio": gate_ratio,
        "gate_proj_edges": gate_proj_edges,
        "triangles": triangles,
        "empty_x": False,
        "empty_edges": False,
    }


def diagnose_circuit_generation(model, cfg):
    dataset_name = getattr(cfg.datamodule, "dataset_names", None)
    if dataset_name not in ["CIRCUIT", "circuit"]:
        log.info("diagnose_circuit_generation skipped (dataset_names=%s)", dataset_name)
        return

    net_id = resolve_circuit_net_id(cfg)
    num_samples = int(getattr(cfg, "diag_num_samples", 50))

    net_token_id = None
    if hasattr(model.tokenizer, "node_idx_offset"):
        net_token_id = model.tokenizer.node_idx_offset + net_id
    log.info("Diagnose CIRCUIT generation: net_id=%s num_samples=%s", net_id, num_samples)
    log.info(
        "Tokenizer stats: vocab=%s idx_offset=%s node_idx_offset=%s edge_idx_offset=%s num_node_types=%s net_token_id=%s",
        len(model.tokenizer),
        getattr(model.tokenizer, "idx_offset", None),
        getattr(model.tokenizer, "node_idx_offset", None),
        getattr(model.tokenizer, "edge_idx_offset", None),
        getattr(model.tokenizer, "num_node_types", None),
        net_token_id,
    )

    model.cfg.sampling.num_samples = num_samples
    model.cfg.sampling.batch_size = min(getattr(cfg.sampling, "batch_size", num_samples), num_samples)
    graphs, _ = model.generate(num_samples=num_samples, return_sent=True)

    gen_graphs = []
    decode_fail = 0
    empty_graphs = 0
    dump_enabled = bool(getattr(cfg, "diagnose_dump", False))
    dump_n = int(getattr(cfg, "diagnose_dump_n", 3))
    dump_samples = []
    token_stats = {
        "sos": 0,
        "reset": 0,
        "ladj": 0,
        "radj": 0,
        "eos": 0,
        "pad": 0,
        "node": 0,
        "edge": 0,
        "net_token": 0,
        "gate": 0,
        "net": 0,
    }

    for seq in graphs:
        seq = seq.cpu()
        # count token types
        for t in seq.tolist():
            if t == model.tokenizer.sos:
                token_stats["sos"] += 1
            elif t == model.tokenizer.reset:
                token_stats["reset"] += 1
            elif t == model.tokenizer.ladj:
                token_stats["ladj"] += 1
            elif t == model.tokenizer.radj:
                token_stats["radj"] += 1
            elif t == model.tokenizer.eos:
                token_stats["eos"] += 1
            elif t == model.tokenizer.pad:
                token_stats["pad"] += 1
            elif model.tokenizer.idx_offset <= t < model.tokenizer.node_idx_offset:
                token_stats["node"] += 1
            elif t >= model.tokenizer.edge_idx_offset:
                token_stats["edge"] += 1
            elif model.tokenizer.node_idx_offset <= t < model.tokenizer.edge_idx_offset:
                token_stats["node_label"] = token_stats.get("node_label", 0) + 1
            if net_token_id is not None and t == net_token_id:
                token_stats["net_token"] += 1

        # decode to graph for structural stats
        try:
            g = model.tokenizer.decode(seq)
            gen_graphs.append(g)
            if g.x is None or g.x.numel() == 0 or g.edge_index is None or g.edge_index.numel() == 0:
                empty_graphs += 1
                if dump_enabled and len(dump_samples) < dump_n:
                    dump_samples.append(seq.tolist())
        except Exception:
            decode_fail += 1
            if dump_enabled and len(dump_samples) < dump_n:
                dump_samples.append(seq.tolist())
            continue

        # estimate gate/net ratios from node labels in decoded graph
        if g.x is not None and g.x.numel() > 0:
            x = g.x.reshape(-1)
            token_stats["gate"] += int((x != net_id).sum().item())
            token_stats["net"] += int((x == net_id).sum().item())

    gen_stats = [_graph_stats(g, net_id) for g in gen_graphs]

    ref_ds = CircuitDataset(root=cfg.datamodule.root, split="test")
    ref_graphs = [ref_ds[i] for i in range(len(ref_ds))]
    ref_stats = [_graph_stats(g, net_id) for g in ref_graphs]

    def _collect(stats, key):
        return [s[key] for s in stats]

    def _empty_ratio(stats, key):
        if not stats:
            return 0.0
        return float(sum(1 for s in stats if s[key]) / len(stats))

    log.info(
        "Generated graphs: count=%d empty_x=%.3f empty_edges=%.3f zero_gate_proj=%.3f zero_tri=%.3f",
        len(gen_stats),
        _empty_ratio(gen_stats, "empty_x"),
        _empty_ratio(gen_stats, "empty_edges"),
        float(sum(1 for s in gen_stats if s["gate_proj_edges"] == 0) / max(len(gen_stats), 1)),
        float(sum(1 for s in gen_stats if s["triangles"] == 0) / max(len(gen_stats), 1)),
    )
    log.info(
        "Decode stats: decoded=%d decode_fail=%d empty_graphs=%d",
        len(gen_graphs),
        decode_fail,
        empty_graphs,
    )
    log.info(
        "Real graphs: count=%d empty_x=%.3f empty_edges=%.3f zero_gate_proj=%.3f zero_tri=%.3f",
        len(ref_stats),
        _empty_ratio(ref_stats, "empty_x"),
        _empty_ratio(ref_stats, "empty_edges"),
        float(sum(1 for s in ref_stats if s["gate_proj_edges"] == 0) / max(len(ref_stats), 1)),
        float(sum(1 for s in ref_stats if s["triangles"] == 0) / max(len(ref_stats), 1)),
    )

    for name, stats in [("gen", gen_stats), ("real", ref_stats)]:
        log.info("[%s] nodes %s", name, _summarize(_collect(stats, "num_nodes")))
        log.info("[%s] edges %s", name, _summarize(_collect(stats, "num_edges")))
        log.info("[%s] density %s", name, _summarize(_collect(stats, "density")))
        log.info("[%s] gate_ratio %s", name, _summarize(_collect(stats, "gate_ratio")))
        log.info("[%s] gate_proj_edges %s", name, _summarize(_collect(stats, "gate_proj_edges")))
        log.info("[%s] triangles %s", name, _summarize(_collect(stats, "triangles")))

    total_seq_tokens = sum(
        token_stats[k] for k in ["sos", "reset", "ladj", "radj", "eos", "pad", "node", "edge"]
    )
    if total_seq_tokens > 0:
        log.info(
            "Token stats (seq-level): total=%d sos=%d reset=%d ladj=%d radj=%d eos=%d pad=%d node=%d edge=%d net_token=%d",
            total_seq_tokens,
            token_stats["sos"],
            token_stats["reset"],
            token_stats["ladj"],
            token_stats["radj"],
            token_stats["eos"],
            token_stats["pad"],
            token_stats["node"],
            token_stats["edge"],
            token_stats["net_token"],
        )
        if "node_label" in token_stats:
            log.info("Token stats (seq-level): node_label=%d", token_stats["node_label"])
    total_nodes = token_stats["gate"] + token_stats["net"]
    if total_nodes > 0:
        log.info(
            "Token stats (node labels): total_nodes=%d gate=%d net=%d gate_ratio=%.3f",
            total_nodes,
            token_stats["gate"],
            token_stats["net"],
            float(token_stats["gate"] / total_nodes),
        )

    if dump_enabled and dump_samples:
        log.info("Decode failed samples (token ids, first %d): %s", dump_n, dump_samples)


def diagnose_teacher_forcing_net_prob(model, cfg):
    dataset_name = getattr(cfg.datamodule, "dataset_names", None)
    if dataset_name not in ["CIRCUIT", "circuit"]:
        log.info("diagnose_teacher_forcing_net_prob skipped (dataset_names=%s)", dataset_name)
        return

    if not getattr(model.tokenizer, "labeled_graph", False):
        log.info("diagnose_teacher_forcing_net_prob skipped (tokenizer not labeled_graph)")
        return

    net_id = resolve_circuit_net_id(cfg)
    if not hasattr(model.tokenizer, "node_idx_offset"):
        log.info("diagnose_teacher_forcing_net_prob skipped (tokenizer missing node_idx_offset)")
        return
    net_token_id = model.tokenizer.node_idx_offset + net_id
    if not (0 <= net_token_id < len(model.tokenizer)):
        log.info("diagnose_teacher_forcing_net_prob skipped (invalid net_token_id=%s)", net_token_id)
        return

    num_batches = int(getattr(cfg, "diag_tf_batches", 4))
    if num_batches <= 0:
        return
    max_tokens = int(getattr(cfg, "diag_tf_max_tokens", 0))

    split = str(getattr(cfg, "diag_tf_split", "test")).lower()
    datamodule = model._datamodule
    if split == "test":
        try:
            datamodule.setup("test")
            dataloader = datamodule.test_dataloader()
        except Exception:
            log.warning("diag_tf_split=test unavailable; falling back to train_dataloader")
            datamodule.setup("fit")
            dataloader = datamodule.train_dataloader()
    elif split == "val":
        datamodule.setup("fit")
        dataloader = datamodule.val_dataloader()
    else:
        datamodule.setup("fit")
        dataloader = datamodule.train_dataloader()
    totals = {
        "count_valid": 0,
        "count_node_label": 0,
        "count_net_target": 0,
        "sum_net_prob_valid": 0.0,
        "sum_net_prob_node_label": 0.0,
        "sum_net_prob_net_target": 0.0,
        "sum_pred_net_valid": 0.0,
        "sum_pred_net_node_label": 0.0,
        "sum_pred_net_net_target": 0.0,
        "sum_target_net_node_label": 0.0,
    }

    model.eval()
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            if not torch.is_tensor(batch) or batch.numel() == 0:
                continue
            if max_tokens > 0 and batch.shape[1] > max_tokens:
                batch = batch[:, :max_tokens]
            batch = batch.to(model.device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            logits = model.model(x)
            log_probs = torch.log_softmax(logits, dim=-1)
            net_prob = log_probs[..., net_token_id].exp()
            pred_net = (logits.argmax(dim=-1) == net_token_id).float()

            pad = model.tokenizer.pad
            valid = y != pad
            node_label = (y >= model.tokenizer.node_idx_offset) & (y < model.tokenizer.edge_idx_offset)
            net_target = y == net_token_id

            totals["count_valid"] += int(valid.sum().item())
            totals["count_node_label"] += int(node_label.sum().item())
            totals["count_net_target"] += int(net_target.sum().item())
            totals["sum_target_net_node_label"] += float(net_target[node_label].sum().item())

            totals["sum_net_prob_valid"] += float(net_prob[valid].sum().item())
            totals["sum_net_prob_node_label"] += float(net_prob[node_label].sum().item())
            totals["sum_net_prob_net_target"] += float(net_prob[net_target].sum().item())

            totals["sum_pred_net_valid"] += float(pred_net[valid].sum().item())
            totals["sum_pred_net_node_label"] += float(pred_net[node_label].sum().item())
            totals["sum_pred_net_net_target"] += float(pred_net[net_target].sum().item())

    def _safe_div(num, den):
        return float(num) / float(den) if den > 0 else float("nan")

    log.info(
        "Teacher-forcing net prob: avg_net_prob_valid=%.6f avg_net_prob_node_label=%.6f avg_net_prob_on_net_target=%.6f",
        _safe_div(totals["sum_net_prob_valid"], totals["count_valid"]),
        _safe_div(totals["sum_net_prob_node_label"], totals["count_node_label"]),
        _safe_div(totals["sum_net_prob_net_target"], totals["count_net_target"]),
    )
    log.info(
        "Teacher-forcing net top1 rate: pred_net_valid=%.6f pred_net_node_label=%.6f pred_net_on_net_target=%.6f",
        _safe_div(totals["sum_pred_net_valid"], totals["count_valid"]),
        _safe_div(totals["sum_pred_net_node_label"], totals["count_node_label"]),
        _safe_div(totals["sum_pred_net_net_target"], totals["count_net_target"]),
    )
    log.info(
        "Teacher-forcing target ratios: net_target/valid=%.6f net_target/node_label=%.6f",
        _safe_div(totals["count_net_target"], totals["count_valid"]),
        _safe_div(totals["sum_target_net_node_label"], totals["count_node_label"]),
    )


@hydra.main(
    version_base="1.3", config_path=str(here() / "configs"), config_name="test"
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    if not cfg.model.pretrained_path:
        raise ValueError("cfg.model.pretrained_path is required for testing")

    log.info(f"Loading model from {cfg.model.pretrained_path}...")
    model = SequenceModel.load_from_checkpoint(cfg.model.pretrained_path, weights_only=False)
    model.update_cfg(cfg)

    datamodule = model._datamodule

    logger = []
    if cfg.wandb:
        wandb_logger = pl.loggers.WandbLogger(project="ELDA")
        logger.append(wandb_logger)
    logger.append(pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs"))

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    if not getattr(cfg, "diag_tf_only", False):
        trainer.test(model, datamodule)

    if getattr(cfg, "diagnose", False):
        try:
            if getattr(cfg, "diag_tf_only", False):
                diagnose_teacher_forcing_net_prob(model, cfg)
            else:
                diagnose_circuit_generation(model, cfg)
                diagnose_teacher_forcing_net_prob(model, cfg)
        except Exception as e:
            log.exception("Diagnose failed: %s", e)


if __name__ == "__main__":
    main()
