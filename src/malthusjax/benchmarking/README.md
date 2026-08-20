# `malthusjax.benchmarking` — Reference

Scope: `malthusjax.benchmarking.runner`, `malthusjax.benchmarking.results`, `malthusjax.benchmarking.config`, `malthusjax.benchmarking.sampling`, `malthusjax.benchmarking.cli`, `malthusjax.benchmarking.io`, `malthusjax.benchmarking.analysis`, `malthusjax.benchmarking.statistics`, `malthusjax.benchmarking.registry`. Every claim below is traceable directly to source code and docstrings.

---

## Overview & Architecture

The `malthusjax.benchmarking` package provides measurement infrastructure for evolutionary algorithms. It decouples engine execution from result collection, multi-seed aggregation, disk serialization, and statistical comparison.

Key mechanisms:
- **Engine Protocol Contract**: Defines a single method `run_once(key: chex.Array) -> Dict[str, Any]` returning `'history'`, `'summary'`, and `'timings'`.
- **Multi-Seed Orchestration**: `BenchmarkRunner` iterates across PRNG seeds, wrapping execution outcomes into structured `RunResult` records and collating them into `ExperimentResult` objects.
- **Lightweight Memory Management**: `serialize_history=False` allows dropping per-generation history arrays during massive multi-day cluster sweeps to prevent memory growth.
- **Unified CLI (`mjax`)**: Provides standard CLI subcommands (`run`, `parity`, `analyze`, `plot`, `report`, `aggregate`, `catalog`).

---

## `malthusjax.benchmarking.runner`

### `Engine` (Protocol)
Defines the engine execution contract:
```python
class Engine(Protocol):
    def run_once(self, key: chex.Array) -> Dict[str, Any]: ...
```
Returned dictionary entries:
- `'history'`: List of per-generation metric dictionaries.
- `'summary'`: Map of final numerical summary metrics.
- `'timings'`: Map of stage wall-clock timings.

### `BenchmarkRunner`
Drives engine execution over seed sequences:
- `engine: Engine` — Execution engine instance.
- `experiment_name: str` (default `"benchmark_experiment"`).
- `output_dir: Optional[Path]` (default `None`).
- `write_artifacts: bool` (default `True`).
- `prng_impl: Optional[str]` (default `None`).
- `trace_dir: Optional[Path]` (default `None`).
- `serialize_history: bool` (default `True`).

`runner.run(seeds: Sequence[int]) -> ExperimentResult` iterates through seeds, deriving PRNG keys via `create_key` or `jr.PRNGKey`, executing `engine.run_once(key)`, measuring wall-clock duration (`duration_seconds`), and returning an `ExperimentResult`.

---

## `malthusjax.benchmarking.results`

### `RunResult`
Dataclass representing a single seeded execution:
- `seed: int` — PRNG seed.
- `status: str` — Status string (`"success"`, `"error"`, `"failure"`, `"timeout"`).
- `metrics: Dict[str, float]` — Computed metrics dictionary (exposes `.summary` property alias).
- `history: List[Dict[str, Any]]` — List of per-generation metrics dictionaries.
- `artifacts: Dict[str, str]` — Output file paths mapping.
- `duration_seconds: Optional[float]` — Wall-clock execution duration in seconds.
- `timings: Optional[Dict[str, float]]` — Detailed stage timing breakdown.
- `error: Optional[str]` — Error message string if `status != "success"`.
- `created_at: datetime` — UTC timestamp.

Methods: `.to_dict()`, `.to_json()`, `RunResult.from_dict(d)`, `RunResult.from_json(s)`.

### `ExperimentResult`
Dataclass holding multi-seed results for a single pipeline:
- `name: str` — Experiment name.
- `runs: List[RunResult]` — Per-seed `RunResult` objects.
- `metadata: Dict[str, Any]` — Run metadata.
- `schema_version: str` (default `"0.1"`).

Methods:
- `.aggregated_summary(optimum=None)` — Calculates mean, median, stdev, and confidence intervals across seeds.
- `.combined_history(seed_field="seed")` — Flattens per-generation history records across seeds into a single tidy list of dictionaries.

### `ComparisonResult`
Dataclass holding multi-pipeline results:
- `pipelines: Dict[str, ExperimentResult]` — Pipeline name map.
- `shared_config: Dict[str, Any]` — Baseline configuration map.
- `initial_population: Optional[Any]` — Shared initial population PyTree.
- `negate_map: Dict[str, bool]` — Metric sign inversion map.

Methods:
- `.summary_table(latex=False)` — Generates aggregated metrics dictionary or formatted LaTeX `tabular` markup (`latex=True`).
- `.plot_convergence(save_path=None)` — Generates convergence overlay plot.
- `.plot_boxplots(save_path=None)` — Generates metric distribution boxplots.

---

## `malthusjax.benchmarking.io` & Persistence

The `write_experiment_artifacts(experiment, output_dir)` function writes artifacts to disk:

```text
{output_dir}/
├── metadata/
│   └── config_snapshot.toml       # Copy of experiment configuration
├── data/
│   ├── pipeline_{name}/
│   │   ├── seed_1.json            # RunResult serialized to JSON
│   │   ├── seed_2.json
│   │   └── ...
└── analysis/
    ├── {pipeline}_summary.json    # Aggregated metrics summary JSON
    ├── comparison_table.csv       # Summary table in CSV format
    ├── comparison_table.md        # Summary table in Markdown format
    └── comparison_table.tex       # Summary table in LaTeX format
```

---

## `malthusjax.benchmarking.cli` (`mjax`)

CLI module exposing commands via `argparse`:

| Command | Arguments / Flags | Functionality |
|---|---|---|
| `mjax run` | `config` (Path) | Runs multi-seed pipeline execution from a TOML configuration file and saves raw JSON results. |
| `mjax parity` | `config` (Path) | Runs seed-aligned statistical parity execution, enforcing a shared initial population across pipelines. |
| `mjax analyze` | `results_dir` (Path) | Computes summary statistics or statistical parity comparisons and writes summaries/tables to `{results_dir}/analysis`. |
| `mjax plot` | `results_dir` (Path) | Generates diagnostic plots (`convergence.png`, `fitness_distribution.png`, `timings.png`) into `{results_dir}/plots`. |
| `mjax report` | `results_dir` (Path) | Sequentially executes `handle_analyze` and `handle_plot`. |
| `mjax aggregate` | `--out_dir` (Path, required), `results_dirs` (Path, 1+) | Combines multiple experiment result directories into an aggregate `MetaComparison` report grid. |
| `mjax catalog` | — | Lists all registered framework operators. |
