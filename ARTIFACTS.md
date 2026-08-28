# ELDA artifact layout and integrity

The GitHub repository is the source release. Large immutable research outputs
are published separately so that Git history remains reviewable and normal
clones do not require approximately 13 GB of generated content.

## Lightweight supplement

`release/ELDA_Code_Data_Supplement/` is committed directly. It contains a real
frozen paper attempt, expected reconstructed objects and Verilog, a negative
case, compact frozen tables, tests, and content checksums. It is the fastest
way to verify the representation, validation, deterministic exporter, and
Yosys path.

## Full artifact bundle

The full bundle restores these paths relative to an artifact root:

- `checkpoints/`: selected reference and R1--R4 checkpoints;
- `paper/results/`: final one-shot attempts and assembly outputs;
- `paper/reports/`: frozen reference tables and audit reports;
- `data_manifests/elda/`: full split lists and PyTorch metadata.

Download the bundle from the archival location supplied with the paper or the
GitHub release, verify its published archive checksum, and run:

```bash
python release/verify_release.py --artifact-root /path/to/ELDA-artifacts
```

The optional `release/checkpoints.sha256` manifest verifies the selected
checkpoint payloads after download.
`release/final_attempts_manifest.json` fixes every supported run root and its
1,024 attempt population. No retry, resampling, or repaired replacement attempt
is silently substituted by the verifier.

To run `paper/reproduce_tables.sh`, copy or symlink the artifact bundle's
`paper/results` and `paper/reports` directories into the source checkout. This
explicit overlay keeps executable code and immutable outputs separable while
preserving the exact relative paths consumed by the frozen scripts.

The public archive URL/DOI should be inserted in the GitHub release notes once
assigned; no placeholder URL is embedded in executable manifests.
