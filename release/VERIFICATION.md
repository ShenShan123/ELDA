# ELDA V6.1 release verification

Verification date: 2026-08-28.

## Public source view

- The GitHub candidate contains 193 files and is 2.24 MiB before Git metadata.
- No file exceeds 10 MiB; no checkpoint or full attempt directory enters Git.
- The candidate contains no author-machine absolute path, credential-like
  material, escaping symlink, legacy implementation import, or legacy Hydra
  target.
- A clean publication-only copy builds `elda_netlist-1.0.0` as a wheel.
- The original archival working tree and large frozen artifacts remain outside
  the Git source view; constructing the release did not delete them.

Run the source-boundary audit with:

```bash
python scripts/audit_github_release.py
```

## Code and data-backed tests

- Without external datasets: 21 tests passed and 11 data-backed tests skipped.
- With the frozen V6.1 dataset and Nangate45 Liberty: 28 tests passed, 4 tests
  skipped, and all 4 R1--R4 grammar subtests passed.
- The V6.1 and R1--R4 Hydra configurations compose using only documented
  environment-variable data roots.
- Missing Liberty metadata now fails immediately for endpoint-complete V6.1
  and V6.2 tokenizers instead of silently producing an empty pin vocabulary.
- Training/reproduction shell entry points pass `bash -n`.

## Executable supplement

- Static layout, anonymity, release-identity, and SHA-256 audits pass in a clean
  publication-only copy.
- Supplement tests: 5 passed.
- The real frozen paper sample completes object reconstruction, validation,
  constrained-decoder replay, deterministic Verilog export, and Yosys
  read/check.

Run these checks with:

```bash
bash release/ELDA_Code_Data_Supplement/scripts/run_tests.sh
bash release/ELDA_Code_Data_Supplement/scripts/run_minimal_reproduction.sh
```

## Full frozen artifacts

- Six selected checkpoints match `release/checkpoints.sha256`.
- ELDA, three controls, R1--R4, and D1--D6 each contain exactly the contiguous
  set `attempt_0000` through `attempt_1023`.
- Required frozen paper tables are present in the full artifact bundle.
- `release/checkpoint_namespace_migration.json` records metadata-only namespace
  migration and byte-identical model/optimizer tensors.

Run the complete audit after downloading the artifact bundle:

```bash
python release/verify_release.py --artifact-root /path/to/ELDA-artifacts
```

The final audit returned `status: pass`, checked all six checkpoint payload
hashes, and reported 1,024 attempts for every declared run.

## Environment limitation

The final source-release audit did not execute a new GPU optimization step.
Historical GPU training provenance remains frozen in checkpoint manifests;
CPU package construction, configuration resolution, data/tokenizer tests,
artifact integrity, table inputs, and the Yosys smoke path were all exercised.
