# ELDA paper reproduction

The GitHub source repository contains the supported final evaluators. Frozen
one-shot attempts, design-level outputs, and full-precision reports are
distributed in the separate artifact bundle. Overlay its `paper/results` and
`paper/reports` directories here before invoking the driver.

Use:

```bash
ELDA_REPRO_SCOPE=core ELDA_PYTHON=/path/to/python paper/reproduce_tables.sh
```

for the attempt-only repair, representation-ablation, decoder-ablation, and
100-design assembly tables.

Use the default `ELDA_REPRO_SCOPE=all` with `ELDA_DATA_ROOT`,
`ELDA_PROJECTION_DATA_ROOT`, and `ELDA_COMMON_DATA_ROOT` set to regenerate
reference-dependent graph, endpoint, logic-usefulness, and appendix metrics.

The canonical paper outputs are:

- `reports/final_baseline/phase10_8_endpoint_fidelity_main_table/`
- `results/elda/design_assembly/`
  `random100_medium_n1900_3500_strict_20260724/summary/`
- `reports/final_baseline/appendix_dataset_split_inventory_20260726/`
- `reports/final_baseline/appendix_size_stratified_cell_inventory_20260730/`
- `reports/final_baseline/phase10_9_standalone_logic_usefulness/`

Historical pilots and retry queues are intentionally excluded from the GitHub
source view. The paths above are the supported publication entry points in the
full artifact bundle.
