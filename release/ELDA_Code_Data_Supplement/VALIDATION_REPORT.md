# Validation report

Validation was performed on 2026-07-31 in a fresh temporary directory outside
the research repository. The staging tree was copied there without links and
the following commands were executed with the declared Python environment and
Yosys 0.51+101:

```bash
bash scripts/run_minimal_reproduction.sh
bash scripts/run_tests.sh
python scripts/recompute_paper_audits.py
python scripts/audit_privacy.py .
python scripts/audit_release_identity.py .
python scripts/validate_package.py .
```

Results:

- dependency check: PASS;
- Source--Demand serialize/parse/decode/re-serialize: PASS;
- seven positive validity conditions: PASS;
- seven rule-targeted negative cases: PASS;
- full production-grammar token trace: PASS;
- zero-repair Verilog materialization: PASS;
- Yosys read/check: PASS;
- stable expected-output comparison: PASS;
- pytest: 5 passed (one upstream deprecation warning);
- compact paper-record arithmetic audit: PASS;
- privacy and external-link scan: PASS;
- ELDA release-identity and namespace scan: PASS;
- static size, permissions, required-file, and symlink checks: PASS.

The combined clean-copy command sequence took 21 seconds. The scientific smoke
pipeline itself took approximately 9 seconds on the validation host. No GPU or
network access was used. The only warning came from an upstream PyTorch
Geometric deprecation notice and does not affect the results.

No software, scientific-definition, data-license, or environment blocker was
found for the included smoke workflow. Dataset metadata consistently uses the
final paper corpus and the 11,390-object deduplicated evaluation reference.
