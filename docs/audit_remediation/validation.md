# Checkpoint 1 validation — 8 November 2026

Implemented work packages P0, P1, and P2 from [the plan](../audit_remediation_plan.md). The [ledger](findings.md) distinguishes contained/fixed problems from unresolved scientific work.

## Revisions and preservation

- Source baseline: `51009b8cf7464fe4a56bc40c472bfe8377ed9858`.
- Plan/baseline record: `391de69e` in MalthusJAX.
- Paper safety change: `07700c6` in MalthusJAX_Paper.
- Both working copies use `codex/audit-remediation`; nothing was pushed or merged.
- Compared the original paper working-tree hashes after implementation: only `scripts/ingest_cluster_results.py` changed among the 72 existing files. The other 71 files, including uncommitted manuscript edits, tables, figures, and data, were preserved byte-for-byte. New remediation notes/tests/manifests are separate files.

## Tests

```bash
python3 -m pytest tests/benchmarking tests/stats --no-cov -q
```

201 passed, 14 existing numerical warnings, 37.41 seconds. This includes 27 new tests for scientific extraction and actual generator/export orchestration using deterministic mock experiments. No optional GPU experiment was launched. New helper tests first failed because the required module did not exist; after implementation all 24 helper tests passed. The three generator tests also passed.

```bash
python3 -m unittest discover -s tests -p test_ingest_cluster_results.py
```

Run from the paper repository: six tests passed. A separate replay of the old ingestion script in a temporary directory confirmed that its original dry-run replaced an existing table. The fixed tests check byte preservation on dry-run, missing/malformed/nonfinite inputs, and unsupported real generation.

Ruff checks passed for all changed Python files, including an explicit check of the otherwise-excluded empirical generator and paper script/tests. Python compilation and diff whitespace checks passed.

Environment for source checks: `/opt/anaconda3/bin/python3`, Python 3.12.5, JAX 0.11.0, NumPy 2.5.1, SciPy 1.18.0, pytest 9.0.2. Paper safety checks also run using the standard-library unittest runner. The baseline manifest records the audit environment; it is not a locked publication environment.

## Limits and next work

- No H100, multi-GPU, or full-repository test pass is claimed.
- Original experiment artifacts were neither regenerated nor declared invalid wholesale.
- Statistical helper defaults and inferential decisions still require P4. Dataset 5 exports unadjusted tests with an explicit pending-inference label; these are not validated publication conclusions.
- Backend-native signed QD scores are preserved, but comparative QD inference is unavailable pending P3's shared definition.
- Full paper table generation from cluster analysis remains explicitly unavailable pending P10; H2 dry-run inspection validates structure and finite coefficients only.
- Other legacy table narratives and scientific claims still require P10/A22 review.
- The initial next-batch recommendation was P3/P4. After job 1303 exposed timing contamination, [the revised implementation plan](../audit_remediation_plan.md) prioritizes PT1–PT5 timing repairs and focused hardware validation first, then resumes P3/P4. This checkpoint report does not certify timing isolation.
