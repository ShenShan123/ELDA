# ELDA: Endpoint-Level Demand Assignment for Gate-Level Generation

ELDA generates materializable gate-level subcircuits with explicit Liberty
input-pin demands, typed driver endpoints, source-to-demand assignments, and
state-constrained autoregressive decoding.

This repository contains the model and decoder, training configurations,
dataset-construction code, evaluation scripts, compact result tables, and a
real-sample minimal reproduction.

## Resources

- Source code: [ShenShan123/ELDA](https://github.com/ShenShan123/ELDA)
- Pretrained checkpoints: [yuey1801/ELDA-Checkpoints](https://huggingface.co/yuey1801/ELDA-Checkpoints)
- Evaluation outputs: [yuey1801/ELDA-Artifacts](https://huggingface.co/datasets/yuey1801/ELDA-Artifacts)

The two Hugging Face repositories are currently private. Collaborators can
download them after being granted access, or use an equivalent local copy
transferred by the project owner. Their contents and layout are described in
[ARTIFACTS.md](ARTIFACTS.md).

## What is included

- `elda/`: model, dataset, serializer, and constrained-decoder implementation.
- `circuit_kahypar/` and `tools/`: gate-level subcircuit extraction and data tools.
- `configs/experiment/elda_*.yaml`: reference and R1--R4 configurations.
- `scripts/run_elda_reference_train.sh`: guarded reference-training entry point.
- `data_manifests/elda/`: path-independent frozen split counts and hashes.
- `paper/scripts/` and `paper/reproduce_tables.sh`: supported final-table drivers.
- `release/ELDA_Code_Data_Supplement/`: portable real-sample smoke reproduction
  and compact paper tables.
- `release/verify_release.py`: integrity audit for an installed full artifact bundle.

The public Python namespace is `elda`. `AutoGraph` appears only where it names
the upstream project or a scientifically distinct comparison baseline.

## Installation

The tested full environment is defined by `environment.yaml`:

```bash
git clone https://github.com/ShenShan123/ELDA.git
cd ELDA
conda env create -f environment.yaml
conda activate elda
pip install -e . --no-build-isolation
```

Yosys is required for Verilog read/check. OpenROAD-flow-scripts is required
only for routed-GDS reproduction. The lightweight supplement has its own
environment and can be validated independently.

## Fast public smoke test

This uses one frozen real paper attempt and does not require the full corpus or
checkpoint bundle:

```bash
bash release/ELDA_Code_Data_Supplement/scripts/run_tests.sh
bash release/ELDA_Code_Data_Supplement/scripts/run_minimal_reproduction.sh
```

The second command reconstructs the ELDA object, validates it, emits Verilog,
and runs Yosys when Yosys is available. Expected outputs and checksums are
included in the supplement.

Run the source unit tests with:

```bash
python -m pytest -q tests
```

Data-backed tests skip when the external corpus is absent. To enable them, set
`ELDA_DATA_ROOT` as described below.

## Download checkpoints and evaluation outputs

Collaborators with access to the private Hugging Face repositories can run:

```bash
hf auth login
hf download yuey1801/ELDA-Checkpoints \
  --local-dir /path/to/ELDA-Checkpoints
hf download yuey1801/ELDA-Artifacts \
  --repo-type dataset \
  --local-dir /path/to/ELDA-Artifacts
```

The same directory layout may be copied directly between servers. Checkpoints
are under `ELDA-Checkpoints/{ELDA,R1,R2,R3,R4}/model.ckpt`; frozen generation
attempts and compact results are under `ELDA-Artifacts/{attempts,results}`.

## Data and split identity

Point ELDA at a materialized corpus and Nangate45 Liberty file:

```bash
export ELDA_DATA_ROOT=/path/to/elda_dataset
export ELDA_NANGATE45_LIBERTY=/path/to/NangateOpenCellLibrary_typical.lib
```

The dataset root must contain `meta.pt`, `mapping.txt`, the three source-clean
split lists, and the referenced graph objects. The strict public training view
contains 170,940/9,396/11,574 train/validation/test objects. Reference-dependent
paper metrics use the development-deduplicated 11,390-object test reference.

The current `ELDA-Artifacts` repository contains dataset manifests and
preprocessing instructions, but not the full third-party-derived graph corpus
or Nangate45 library. These inputs must be reconstructed from their licensed
sources or transferred separately within an authorized collaboration.

The selected checkpoint predates the final family audit and retains its
recorded 171,192/9,639/11,574 lineage. Exact counts and split hashes for both
views are recorded in `data_manifests/elda/split_summary.json`; the repository
does not silently rewrite checkpoint provenance.

`OPENROAD_FLOW_ROOT` may be used instead of `ELDA_NANGATE45_LIBERTY` when it
points to an OpenROAD-flow-scripts `flow/` directory containing Nangate45.

## Train ELDA

The entry point is deliberately guarded against accidental long jobs:

```bash
START_ELDA_TRAINING=1 \
ELDA_DATA_ROOT=/path/to/elda_dataset \
ELDA_PYTHON=/path/to/python \
scripts/run_elda_reference_train.sh \
trainer.max_epochs=4 \
trainer.max_steps=85596 \
trainer.accumulate_grad_batches=8 \
train.periodic_checkpoint_steps=0
```

R1--R4 use the corresponding experiment configs and `ELDA_R1_DATA_ROOT`
through `ELDA_R4_DATA_ROOT`. Final paper hyperparameters and decoder-mask
switches are also frozen under `release/ELDA_Code_Data_Supplement/configs/`.

## Reproduce paper tables

The quickest result check uses the compact tables in `ELDA-Artifacts`:

```bash
cp /path/to/ELDA-Artifacts/results/paper_tables/* \
  release/ELDA_Code_Data_Supplement/paper_results/
python release/ELDA_Code_Data_Supplement/scripts/recompute_paper_audits.py
```

For a complete rerun of the supported table drivers, prepare the paper result
overlay described in [ARTIFACTS.md](ARTIFACTS.md), then run:

```bash
ELDA_DATA_ROOT=/path/to/elda_dataset \
ELDA_PROJECTION_DATA_ROOT=/path/to/common_cell_projection \
ELDA_COMMON_DATA_ROOT=/path/to/common_graph_data \
ELDA_ORFS_ROOT=/path/to/OpenROAD-flow-scripts/flow \
ELDA_PYTHON=/path/to/python \
paper/reproduce_tables.sh
```

`ELDA_REPRO_SCOPE=core` reruns attempt-only ablation, repair, and assembly
audits without recomputing dataset-dependent reference metrics. The default
`all` scope rebuilds the supported final tables from frozen outputs and data.

## Reproducibility boundary

The Git repository intentionally excludes multi-gigabyte checkpoints, attempt
directories, generated reports, and machine-specific historical queue scripts.
This prevents opaque binaries and 124,000 generated files from entering source
history. Nothing is deleted from the archival working tree. The compact
supplement provides an immediately executable smoke path; the external artifact
bundle provides exact full-paper regeneration. See [RELEASE_SCOPE.md](RELEASE_SCOPE.md)
and [ARTIFACTS.md](ARTIFACTS.md).

## Upstream attribution

ELDA was developed from the BSD-3-Clause AutoGraph codebase (“Flatten Graphs
as Sequences: Transformers are Scalable Graph Generators”). The original
license is retained. ELDA adds the gate-level representation, corpus pipeline,
stateful circuit constraints, materialization/evaluation tools, ablations, and
publication artifacts described in the paper.
