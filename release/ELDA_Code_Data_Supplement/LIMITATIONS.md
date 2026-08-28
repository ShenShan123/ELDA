# Limitations and excluded artifacts

This package is a runnable method and audit supplement, not a full archival
snapshot of the research filesystem.

- The full source-clean corpus (191,910 gate-level subcircuits), raw RTL,
  mapped netlists, and the 11,390-object final evaluation reference are omitted
  because of size and source-license constraints.
- The selected 984 MB checkpoint and complete 1,024-attempt directories are
  omitted to keep the archive below 50 MB. Compact frozen aggregate records
  and one real successful paper attempt are included instead.
- Full training is not executable from this archive alone. The exact core
  architecture, optimizer, schedule, generation parameters, production
  serialization, and production decoder grammar are retained. The full data
  loader and checkpoint are intentionally not represented as a smoke model.
- The smoke test uses a locally written `BUF_X1` behavioral model. The paper
  flow used a Nangate45 target library; no PDK/Liberty material is distributed.
- Paper-table CSV files are frozen compact outputs. The included audit script
  recomputes arithmetic closure, but cannot recompute distribution metrics
  without the omitted reference graphs and all generated candidates.

## Frozen paper data convention

All dataset metadata exposed by this supplement follows the final paper
convention: 170,940 training, 9,396 validation, and 11,574 full-test
subcircuits, for 191,910 source-clean objects in total. Reference-dependent
metrics use the 11,390-object test set obtained after removing exact endpoint-
sequence matches to either training or validation data. Historical
intermediate manifests are not part of this compact source package.
