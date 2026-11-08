# Remediation ledger

Baseline: `51009b8cf7464fe4a56bc40c472bfe8377ed9858`. Owner for all implementation packages: Codex, with scientific policy decisions reviewed by the author. Scope and sequence: [implementation plan](../audit_remediation_plan.md).

The original audit and probes remain at `audit/2026-11-08/` locally. [Baseline manifest](baseline.json) records pre-remediation source hashes and the existing paper working tree. Passing tests below validate the changed contracts; they do not establish H100 performance or validate historical paper conclusions.

| Finding | Evidence at audit | Status / package | Regression / closure requirement | Potentially affected evidence |
|---|---|---|---|---|
| A01 Forced timing | Reproduced | Fixed in P1 | `tests/benchmarking/test_empirical.py`, `test_empirical_generator.py`: actual slowdown survives extraction and table export | Dataset 5 timing/ratio; historical impact unverified |
| A02 Invalid statistical defaults | Reproduced | Open, P4 | Shared finite-sample validity and unavailable inference | Generator statistical claims |
| A03 Signed metrics/QD meaning | Reproduced/source | Partial, P1; harmonization P3 | Signed observations preserved; QD comparisons marked unavailable; shared definitions still needed | Control returns; Dataset 5 QD/fitness |
| A04 Placeholder ingestion | Source; baseline dry-run overwrite confirmed in disposable directory | Contained in P2; real generation deferred P10 | Paper `tests/test_ingest_cluster_results.py`: no writes on dry-run/error; no mock fallback | H1/H2/H3 ingestion; historical impact unverified |
| A05 Crowding | Reproduced | Open, P5 | Independent front-local reference and survival invariance | MO results, if used |
| A06 Island horizons | Reproduced | Open, P6 | Exact 0/remainder/divisible horizons | Island fitness/budgets |
| A07 Island timing | Source | Open, P6 | Compilation separation/synchronization and hardware validation | Island scaling |
| A08 Corrected decisions | Reproduced | Open, P4 | Corrected p=.06 cannot pass equivalence at .05 | Suite inference |
| A09 Duplicate pairing | Reproduced | Open, P7 | One-to-one joins/explicit replicate identity | Regression sample sizes/precision |
| A10 Decision semantics | Source | Open, P4 | Hypothesis-specific conclusions, margins, uncertainty | Equivalence/zero-overhead claims |
| A11 Optimize API limits | Source | Open, P8 | Implement or reject controls; truthful termination/covariance | Satellite optimization users/results |
| A12 Run validity/timeout | Reproduced/source | Open, P9 | Status/finite metrics; enforce or reject requested timeout | Success counts and downstream analysis |
| A13 Crossover inferred size | Reproduced | Open, P5 | Typed/legacy key/default-size contract | Fast-path callers |
| A14 Standard deviation sign | Source | Open, P3 | Sign-invariant standard deviations | Maximization histories/error bars |
| A15 Gap transform | Source; H1 provenance unresolved | Open, P7/P10 | Explicit estimand/optimum/epsilon and analysis replay | H1/fitness regressions |
| A16 Adjusted R² | Source | Open, P7 | Match fitted reference statistics | Regression reports |
| A17 Unpackaged plugin | Source | Open, P11 | Isolated wheel backend test | Installed TensorNEAT/QD backend |
| A18 Assurance scope | Source | Open, P11 | Include scientific paths and packages in release checks | Coverage/CI claims |
| A19 Provenance/environment | Risk | Partial, P0; full manifests P9/P10 | Baseline captured; per-run immutable manifests/environment replay outstanding | All paper evidence |
| A20 Archive statistics | Source | Open, P3 | Occupancy-masked mean/std | QD metrics |
| A21 Warmup selection | Source | Open, P6/P9 | Execution-phase metadata independent of seed number | Timing comparisons |
| A22 Documentation/operational promises | Source | Partial, P1/P2; remainder P11 | Output/ingestion behavior documented; verify Docker/tracing/other commands | Reproduction workflow |

## First implementation batch

- P0: baseline manifest and this tracked ledger; manuscript fingerprints captured before edits.
- P1: extraction moved to `src/malthusjax/benchmarking/empirical.py`; live optional integrations are loaded only when running experiments. No forced timing, invented missing measurements, absolute-value objective conversions, or automatic manuscript synchronization. JSON uses an explicit schema version and records unresolved audit limitations. Per-seed measurements remain available.
- P2: paper ingestion now validates H2 CSV structure/finite coefficients read-only. Missing input and unsupported real generation produce errors, with no placeholder fallback. Full publication table regeneration is deliberately unavailable until P10.

The main generator's existing statistical helpers remain unresolved under A02/P4. Its metadata and Dataset 5 table state that inference is pending. Other legacy table narratives are still subject to P10/A22 review. These intermediate outputs must not be treated as submission-ready evidence.

## Commands for the new output contract

Prepare a configuration and compute budget before launching a live run. The command below illustrates the explicit destination; it was not launched during remediation:

```bash
python examples/generate_empirical_paper_curves.py --smoke --out results/new-run-name/empirical_curves.json
```

The parent directory must be new or empty. JSON, `tables/`, and `runs/` stay there; existing evidence is refused rather than overwritten. Automatic sibling-paper discovery/synchronization has been removed. Schema version 2 preserves signed instantaneous curves and records separately named best-so-far control curves. Consumers that assumed absolute values or always-monotonic `mean` must be reviewed during P10.

Paper input validation uses an explicit results directory:

```bash
python scripts/ingest_cluster_results.py --results-dir /path/to/results --dry-run
```

This validates only the current H2 CSV schema and finite model coefficients. It does not validate H1/H3, scientific conclusions, or provenance, and never writes a table.
