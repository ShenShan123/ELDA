import os
import hashlib
import json
import logging
import platform
import subprocess
import hydra
from pyprojroot import here
import numpy as np
from pathlib import Path

import graph_tool
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from elda.models.seq_models import SequenceModel


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver('eval', eval)

log = logging.getLogger(__name__)


def _sha256_file(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=here(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def write_reproducibility_manifest(cfg):
    """Write provenance only; this function does not alter training behavior."""
    output = Path(str(cfg.logs.path))
    output.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(str(cfg.datamodule.root))
    tracked_inputs = {
        "mapping": getattr(cfg.datamodule, "cell_mapping_path", None),
        "meta": dataset_root / "meta.pt",
        "train_split": dataset_root / "train_source_clean.txt",
        "val_split": dataset_root / "val_source_clean.txt",
        "test_split": dataset_root / "test_source_clean.txt",
        "dataset_manifest": (
            dataset_root / "v61_ablation_dataset_manifest.json"
            if (dataset_root / "v61_ablation_dataset_manifest.json").exists()
            else dataset_root / "v61_dataset_manifest.json"
        ),
    }
    source_files = [
        here() / "train.py",
        here() / "elda/models/seq_models.py",
        here() / "elda/datamodules/graph_dataset.py",
        here() / "elda/datamodules/data/circuit_source_net_v61_schema.py",
        here() / "elda/datamodules/data/circuit_source_net_v61_tokenizer.py",
        here() / "elda/datamodules/data/circuit_source_net_v61_ablation_tokenizers.py",
    ]
    manifest = {
        "manifest_version": "elda_training_reproducibility_v1",
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status": _git_output("status", "--short"),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pytorch_lightning": pl.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "input_sha256": {
            name: _sha256_file(path)
            for name, path in tracked_inputs.items()
            if path
        },
        "source_sha256": {
            str(path.relative_to(here())): _sha256_file(path)
            for path in source_files
            if path.exists()
        },
    }
    (output / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@hydra.main(
    version_base="1.3", config_path=str(here() / "configs"), config_name="train"
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)
    write_reproducibility_manifest(cfg)

    # Pure-Python tokenization emits label-only sequences for labeled graphs.
    # Combined with grammar_mask, this creates incompatible training targets.
    pure_py = os.environ.get("SENT_UTILS_PURE_PY", "0") == "1"
    if pure_py and getattr(cfg.datamodule, "labeled_graph", False) and getattr(cfg.train, "grammar_mask", False):
        raise ValueError(
            "Incompatible settings: SENT_UTILS_PURE_PY=1 with train.grammar_mask=true "
            "on labeled graphs. Use train.grammar_mask=false or SENT_UTILS_PURE_PY=0."
        )

    initialize_path = getattr(cfg.train, "initialize_from_checkpoint", None)
    if initialize_path:
        initialize_path = Path(str(initialize_path))
        if not initialize_path.exists():
            raise FileNotFoundError(f"initialization checkpoint does not exist: {initialize_path}")
        log.info(f"Initializing model weights from {initialize_path} with fresh optimizer state...")
        model = SequenceModel(cfg)
        checkpoint = torch.load(initialize_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"initialization checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
    elif cfg.model.pretrained_path is None:
        model = SequenceModel(cfg)
    else:
        log.info(f"Loading model from {cfg.model.pretrained_path}...")
        model = SequenceModel.load_from_checkpoint(
            cfg.model.pretrained_path,
            model=cfg.model,
            weights_only=False,
        )
        os.symlink(
            Path(cfg.model.pretrained_path).resolve(),
            Path(cfg.logs.path) / "pretrained.ckpt",
        )
        model.update_cfg(cfg)
    datamodule = model._datamodule

    logger = []
    if cfg.wandb:
        wandb_logger = pl.loggers.WandbLogger(project="ELDA", config=OmegaConf.to_container(cfg, resolve=True))
        logger.append(wandb_logger)
    logger.append(pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs"))

    model_ckpt_cls = pl.callbacks.ModelCheckpoint

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        model_ckpt_cls(
            monitor=f'val/{datamodule.val_metric[0]}',
            dirpath=cfg.logs.path,
            filename=cfg.model.model_name,
            mode=f'{datamodule.val_metric[1]}',
        )
    ]
    early_stopping_patience = int(
        getattr(cfg.train, "early_stopping_patience", -1)
    )
    if early_stopping_patience >= 0:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor=f"val/{datamodule.val_metric[0]}",
            mode=f"{datamodule.val_metric[1]}",
            min_delta=float(
                getattr(cfg.train, "early_stopping_min_delta", 0.0)
            ),
            patience=early_stopping_patience,
            check_finite=True,
            check_on_train_epoch_end=False,
            verbose=True,
        )
        initial_best = getattr(cfg.train, "early_stopping_initial_best", None)
        if initial_best is not None:
            early_stopping.best_score = torch.tensor(float(initial_best))
        callbacks.append(early_stopping)
    periodic_steps = int(getattr(cfg.train, "periodic_checkpoint_steps", 0) or 0)
    if periodic_steps > 0:
        callbacks.append(
            model_ckpt_cls(
                dirpath=cfg.logs.path,
                filename=f"{cfg.model.model_name}-periodic-{{step}}",
                every_n_train_steps=periodic_steps,
                save_top_k=1,
                monitor=None,
                save_on_train_epoch_end=False,
            )
        )

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    resume_path = getattr(cfg.train, "resume_from_checkpoint", None)
    resume_path = str(resume_path) if resume_path else None
    if resume_path and not Path(resume_path).exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    # Lightning 2.6 follows PyTorch 2.6's weights_only=True default when this
    # argument is omitted. A full training checkpoint contains trusted local
    # OmegaConf objects plus optimizer/scheduler state, so resume must load the
    # complete checkpoint rather than weights only.
    trainer.fit(model, datamodule, ckpt_path=resume_path, weights_only=False)

    trainer.save_checkpoint(f"{cfg.logs.path}/{cfg.model.model_name}-last.ckpt")

    final_test_num_samples = int(getattr(cfg, "final_test_num_samples", 512))
    if final_test_num_samples != 0:
        # Avoid silently launching a full test-set sampling run after training.
        # Use -1 explicitly when a balanced/full test sampling pass is intended.
        model.cfg.sampling.num_samples = final_test_num_samples
        trainer.test(model, datamodule)
    else:
        log.info("Skipping post-training test because final_test_num_samples=0")


if __name__ == "__main__":
    main()
