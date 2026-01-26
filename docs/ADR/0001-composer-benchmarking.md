# ADR 0001 — Product-first Composer & Benchmarking

Status: Proposed

Date: 2026-01-22

Author: MalthusJAX maintainers

## Context

Benchmarking currently lives in a top-level `benchmarks/` package and asks users to manually wire engines, operators, and fitnesses. This creates friction for newcomers and slows adoption. We prioritize a product-first approach: fast, opinionated defaults and a delightful quickstart experience, then iterate toward production features (validation, plugin APIs, CI-grade checks).

## Decision

Introduce two components in `src/malthusjax`:
- `malthusjax.composer` — minimal composition primitives (registry, nodes, pipelines), intentionally hidden behind user-facing helpers.
- `malthusjax.benchmarking` — user-facing experiment runner and result utilities optimized for smooth UX.

Keep the top-level `benchmarks/` as a compatibility shim for one release while we migrate users to the new product-focused APIs.

## Product goals
- Ship a one-command quick path to run an experiment with sensible defaults.
- Provide several presets (`quick_demo`, `standard_ga`) for immediate, small, CI-friendly runs.
- Keep the learning curve minimal: one or two high-level functions cover most use cases; provide power APIs for advanced users.
- Deliver discoverable examples and a short quickstart notebook.

## Quickstart — 1-minute path 🚀

examples/quickstart.toml (preset):

```toml
[pipelines.quick_demo.nodes.fitness]
type = "griewank"
params = { dim = 10 }

[pipelines.quick_demo.nodes.engine]
type = "standard_ga"
params = { pop_size = 32, generations = 50 }
inputs = ["fitness"]

[global]
preset = "quick_demo"
output_dir = "results/quick_demo/"
```

CLI:

```bash
malthusjax pipeline run --config examples/quickstart.toml --pipeline quick_demo --seed 42 --repeats 3
```

Python:

```py
from malthusjax.composer import Composer
composer = Composer.load_config("examples/quickstart.toml", "quick_demo")
# quick_run normalizes seeds and runs the experiment with sensible defaults
result = composer.quick_run(seeds=[42,43,44], output_dir="results/quick_demo/")
# `result` is an ExperimentResult; a quick plot is shown by default
```

## Minimal product API (opinionated) 🧭

- Composer.quick_run(config_path: str, pipeline_name: str, *, seeds: Sequence[int] | int | None = None, seed: int | None = None, repeats: int = 1, output_dir: str | None = None, profiler: bool = False) -> ExperimentResult
  - Normalizes seeds (explicit `seeds` OR `seed + repeats`) and runs an experiment bundle.
  - Uses opinionated defaults where nodes are partially specified.
  - Persists `summary.json`, `histories_combined.csv` (includes `seed` column), and per-seed profiler traces if enabled.

- Discovery: registry and Node/Pipeline primitives remain available for advanced users but are not required for quick runs.

## UX features & presets 🧩
- Named presets (e.g., `quick_demo`) in `examples/` with small defaults for quick iteration.
- After a quick_run, display a compact plot and canonical summary to stdout.
- Save artifacts to `output_dir` with a clear layout (per-seed subfolders when needed).

## Result model (product-focused)
- ExperimentResult (minimal):
  - `experiment`, `pipeline`, `master_seed`, `seed_results` (seed + short summary), `canonical_summary` (first seed by convention)
- Persisted files:
  - `summary.json` — human-readable canonical summary
  - `histories_combined.csv` — tidy CSV with `seed` column
  - `seed/<seed>/trace/` — profiler traces if enabled

Note: canonical summary uses the first seed for backward compatibility; aggregated statistics can be added later.

## Implementation plan — what goes into `src/` (concrete)

This ADR now specifies the concrete files and modules to implement under `src/malthusjax`. The intention is to keep the public product surface minimal (`Composer.quick_run`, presets, and CLI) while placing the implementation, tests, and utilities in `src/` so they are packaged, discoverable, and testable.

Planned modules & files (initial phase):

- `src/malthusjax/composer/`
  - `__init__.py` (export convenience APIs)
  - `registry.py` — Registry class and registration helpers
  - `node.py` — Node dataclass + validation helpers
  - `pipeline.py` — Pipeline class + wiring and build logic
  - `config.py` — TOML loader + lightweight schema (pydantic models or dataclasses)
  - `presets.py` — small presets used by quick_run
  - `tests/test_registry.py`, `tests/test_pipeline.py`, `tests/test_config.py`

- `src/malthusjax/benchmarking/`
  - `__init__.py`
  - `runner.py` — `BenchmarkRunner` and `Composer.quick_run` implementation (no heavy profiling by default)
  - `io.py` — serialization helpers: write `summary.json`, `histories_combined.csv`, per-seed folders
  - `results.py` — types for `ExperimentResult` / `RunResult` and helpers to materialize paths
  - `cli.py` — a thin CLI entrypoint (`malthusjax pipeline run`) that calls `Composer.quick_run`
  - `tests/test_runner.py`, `tests/test_io.py`

- `benchmarks/` (compat shim)
  - Keep top-level scripts that import from new packages but mark them deprecated; add small docstrings that point to new APIs.

- `examples/`
  - `examples/quickstart.toml` (preset)
  - `examples/quickstart.ipynb` (quickstart notebook demonstrating `Composer.quick_run` → `results` workflow)

- `results/` (adaption work)
  - Update `results/src/data_loader.py` to add `ExperimentData.from_experiment_dir(path)` and support the new `summary.json` + `histories_combined.csv` format
  - Add tests and example fixture directories under `tests/fixtures/experiment_example/`

Acceptance criteria for each incremental PR

- PR 1: `src/malthusjax/composer` skeleton + unit tests (registry, node, config parsing). Adds `examples/quickstart.toml`.
- PR 2: `src/malthusjax/benchmarking.runner` with `Composer.quick_run` skeleton and `io` helpers; add a small integration test that runs a `quick_demo` pipeline and writes an `ExperimentResult` to `tmpdir`.
- PR 3: Update `results` loader and `ResultAggregator` tests to consume `ExperimentResult` directories; update `results/README.md` with new workflow.
- PR N: Gradual migration of `benchmarks/*` scripts to call the new API and add deprecation notes.

Testing & CI

- Add unit tests for all new modules. Keep integration smoke tests fast (small populations/generations) so they can run in CI.
- Add a single workflow that validates the end-to-end `Composer.quick_run` → `results` flow on a small `quick_demo` example.

Developer notes & constraints

- Keep composition pure and avoid JAX compilation at construction time; JAX tracing and profiler are invoked explicitly when `runner.run()` is called.
- `quick_run` must be opinionated and shallow: it should produce useful defaults and a small, reproducible `ExperimentResult` for demos and CI.
- Maintain backward compatibility during migration: add compatibility helpers in `results` to convert existing CSVs into the `ExperimentResult` layout.

## Next steps (immediate)
1. Create `src/malthusjax/composer` skeleton and unit tests (PR 1). — **ready to implement**
2. Implement `Composer.quick_run` skeleton + `benchmarking.io` write helpers and a small integration smoke test (PR 2).
3. Update `results/src/data_loader.py` to add `from_experiment_dir()` and tests (PR 3).

---

Revision history:
- 2026-01-22: Product-first ADR created
- 2026-01-22: Added concrete `src/` implementation plan and file map