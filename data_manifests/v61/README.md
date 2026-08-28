# Frozen V6.1 split identity

This source repository intentionally contains only compact, path-independent
metadata for the frozen V6.1 corpus. The complete split lists, `meta.pt`, and
referenced graph objects are data artifacts rather than source code and are
distributed separately.

After materializing that artifact, set `ELDA_DATA_ROOT` to its root. The root
must contain `meta.pt`, `mapping_v61.txt`, and the three source-clean split
lists. `split_summary.json` records the frozen counts and SHA-256 hashes used
to verify those lists without embedding any author-machine paths in Git.

The historical best7 checkpoint used the pre-audit split. New training uses
the strict family-audited split by default. These views are deliberately kept
distinct; checkpoint provenance is never rewritten retroactively.
