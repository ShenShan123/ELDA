# ELDA checkpoints and evaluation artifacts

The GitHub repository contains the source code. Pretrained checkpoints and
frozen evaluation outputs are stored in two separate Hugging Face repositories:

- [yuey1801/ELDA-Checkpoints](https://huggingface.co/yuey1801/ELDA-Checkpoints)
- [yuey1801/ELDA-Artifacts](https://huggingface.co/datasets/yuey1801/ELDA-Artifacts)

Both repositories are currently private and available to approved
collaborators. They may also be copied directly between authorized servers.

## Lightweight supplement

`release/ELDA_Code_Data_Supplement/` is committed directly. It contains a real
frozen paper attempt, expected reconstructed objects and Verilog, a negative
case, compact frozen tables, tests, and content checksums. It is the fastest
way to verify the representation, validation, deterministic exporter, and
Yosys path.

## Download

```bash
hf auth login
hf download yuey1801/ELDA-Checkpoints \
  --local-dir /path/to/ELDA-Checkpoints
hf download yuey1801/ELDA-Artifacts \
  --repo-type dataset \
  --local-dir /path/to/ELDA-Artifacts
```

The repositories contain:

- `ELDA-Checkpoints/ELDA/`: selected ELDA checkpoint;
- `ELDA-Checkpoints/R1/` through `R4/`: independently trained representation-ablation checkpoints;
- `ELDA-Artifacts/attempts/`: 14 final 1,024-attempt populations;
- `ELDA-Artifacts/results/`: compact tables and reproduction inputs;
- `ELDA-Artifacts/manifests/`: run, decoder, split, and design manifests;
- `ELDA-Artifacts/data/`: source and preprocessing metadata.

The decoder ablations D1--D6, `ELDA_Unconstrained`, and `ELDA_Syntax` reuse the
ELDA checkpoint and change only the decoding configuration. The frequency
sampler does not use a language-model checkpoint.

## Compact result reproduction

```bash
cd /path/to/ELDA
cp /path/to/ELDA-Artifacts/results/paper_tables/* \
  release/ELDA_Code_Data_Supplement/paper_results/
python release/ELDA_Code_Data_Supplement/scripts/recompute_paper_audits.py
```

`ELDA-Artifacts/MANIFEST.json` maps every attempt population and result file to
its experiment, checkpoint, decoder configuration, seed, and purpose.

## Full table drivers

The final attempts are stored by public experiment name under
`ELDA-Artifacts/attempts/`. The corresponding layouts expected by the paper
drivers are recorded in `release/final_attempts_manifest.json` and the artifact
manifest. Reference-dependent metrics additionally require a materialized ELDA
corpus and common graph projections:

```bash
ELDA_DATA_ROOT=/path/to/elda_dataset \
ELDA_PROJECTION_DATA_ROOT=/path/to/common_cell_projection \
ELDA_COMMON_DATA_ROOT=/path/to/common_graph_data \
ELDA_ORFS_ROOT=/path/to/OpenROAD-flow-scripts/flow \
ELDA_PYTHON=/path/to/python \
paper/reproduce_tables.sh
```

The full graph corpus, third-party RTL sources, Nangate45 Liberty/PDK files,
and OpenROAD installation are not included in the Hugging Face repositories.
See `ELDA-Artifacts/data/` for source and preprocessing instructions.
