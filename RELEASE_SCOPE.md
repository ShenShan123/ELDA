# ELDA V6.1 publication scope

This directory is the independent ELDA publication tree. A byte-preserving
safety copy of the pre-release working tree is retained outside the release.
The GitHub source view excludes large frozen artifacts through `.gitignore`;
no archival artifact is deleted by that source-release boundary.

## Frozen training lineage

The paper checkpoint was produced in two stages:

1. four epochs from scratch with the V6.1 compact-load representation;
2. six additional epochs initialized from the selected four-epoch checkpoint.

The separately distributed full artifact bundle retains selected checkpoints,
resolved Hydra configurations, metric logs, and reproducibility manifests for:

- V6.1 four-epoch scratch training;
- V6.1 best7 selection/continuation;
- final R1--R4 representation-ablation training.

Duplicate `last` and periodic checkpoints may be removed after their selected
counterparts have been hash-verified.

## Dataset views

The frozen checkpoints were trained with the pre-audit split lists:

- train: 171,192;
- validation: 9,639;
- test: 11,574.

The later strict cross-split-audited corpus view contains:

- train: 170,940;
- validation: 9,396;
- test: 11,574.

Both sets of split lists are retained in the data artifact; compact counts and
hashes are retained in the source repository. The historical
checkpoint lineage must use the pre-audit lists. New training defaults to the
strict audited view unless the historical reproduction preset is selected
explicitly.

## Frozen paper evidence

The full artifact bundle contains the final V6.1 evidence needed by the paper:

- 1,024-attempt ELDA generation;
- unconstrained, syntax-only, and field-frequency controls;
- complete decoder and representation ablations;
- 100-design scaffold-guided assembly;
- final generic-baseline decoded outputs used by common-view metrics;
- repair audits, final tables, appendices, and table-generation scripts.

Failed pilots, superseded retries, V6.2 experiments, duplicate checkpoints,
and unrelated pre-V6.1 logs are outside the publication scope.
