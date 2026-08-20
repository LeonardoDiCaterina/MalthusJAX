# MalthusJAX Executable Scripts & Utilities

This directory contains executable scripts, benchmarking execution harnesses, statistical analysis pipelines, XLA HLO graph extraction tools, and maintenance utilities for MalthusJAX.

---

## 🚀 Primary Entry Points

### `benchmark_runner.py` — TOML-Driven Benchmark Runner
Unified execution runner for TOML-defined benchmark suites.
- **Functionality**: Parses a `.toml` suite definition, generates Cartesian or Latin Hypercube Sampling (LHS) experimental coordinate grids, and executes pipelines via `Composer`.
- **Cluster Protections**: Automatically restricts GPU visibility (`CUDA_VISIBLE_DEVICES="0"`) to prevent JAX NCCL multi-device rendezvous deadlocks on shared GPU clusters.
- **Usage**:
  ```bash
  python scripts/benchmark_runner.py --toml configs/perf/h1_speed_vs_evosax.toml
  python scripts/benchmark_runner.py --toml configs/perf/h1_speed_vs_evosax.toml --smoke
  ```

### `benchmark_analyzer.py` — TOML-Driven Benchmark Analyzer
Unified statistical analyzer processing raw JSON artifacts produced by `benchmark_runner.py`.
- **Functionality**:
  - **Cartesian Mode**: Executes TOST equivalence testing, Wilcoxon signed-rank tests, and Cohen's $d_z$ effect size calculations via `malthusjax.stats.comparator.StatisticalComparator`.
  - **LHS Mode**: Fits OLS log-log interaction regressions with diagnostic checks via `malthusjax.stats.regression_analyzer.OLSRegressionAnalyzer`.
- **Outputs**: Generates publication-ready LaTeX tables (`summary_table.tex`), Markdown summaries, and automated diagnostic plots.
- **Usage**:
  ```bash
  python scripts/benchmark_analyzer.py --dir results/h1_speed_vs_evosax
  ```

---

## 🔍 Diagnostic & Profiling Tools

### `extract_hlo.py` — XLA HLO Graph Extraction & Comparison
Extracts optimized XLA HLO text for TOML engine pipelines and generates side-by-side comparison tables.
- **Functionality**: JIT-compiles engine pipelines to inspect XLA fusion kernels, while loops, and memory copies. For EvoSAX pipelines, it JIT-compiles native `strategy.ask() + strategy.tell()` steps directly to bypass adapter overhead and reveal the true upstream kernel.
- **Usage**:
  ```bash
  python scripts/extract_hlo.py --toml configs/perf/h1_speed_vs_evosax.toml
  ```

### `trace_pipelines.py` — JAX Profiler & TensorBoard Tracing
Traces execution of TOML pipeline runs using `jax.profiler.start_trace` / `stop_trace`.
- **Functionality**: Traces seed 1 of specified pipelines to prevent redundant compilation overhead, outputting TensorBoard profile files.
- **Usage**:
  ```bash
  python scripts/trace_pipelines.py --toml configs/perf/h1_speed_vs_evosax.toml --out-dir results/perf/traces
  ```

---

## 📊 Reporting & Maintenance Utilities

### `generate_thesis_tables.py` — Thesis Table Generator
Parses JSON result files (`parity_results.json`, `ablation_results.json`, etc.) into `ComparisonResult` objects.
- **Functionality**: Formats mean, standard deviation, and confidence intervals into LaTeX (`comparison_table.tex`), CSV (`comparison_table.csv`), and Markdown (`comparison_table.md`) files inside the `analysis/` directory.
- **Usage**:
  ```bash
  python scripts/generate_thesis_tables.py --dir results/h1_parity_qdax
  ```

### `update_readme_coverage.py` — Coverage Injector
Reads `coverage.md` and injects its contents into `README.md` between `<!-- COVERAGE-START -->` and `<!-- COVERAGE-END -->` comment tags.
- **Usage**:
  ```bash
  python scripts/update_readme_coverage.py
  ```

---

## 🐚 Shell Helpers

### `run_core_baseline.sh` — Core Package Quality Baseline
Shell script running quality checks on `malthusjax.core`:
- Runs `pytest` with coverage report on `src/malthusjax/core`.
- Runs `ruff check` on `src/malthusjax/core`.
- Runs `mypy --strict` on `src/malthusjax/core`.
- Logs timestamped outputs to `tmp/core_baseline_YYYYMMDD_HHMMSS.log`.
- **Usage**:
  ```bash
  ./scripts/run_core_baseline.sh
  ```
