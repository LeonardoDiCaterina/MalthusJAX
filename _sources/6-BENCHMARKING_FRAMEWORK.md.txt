# Benchmarking Framework: Multi-Seed Execution and Results Schema

The `malthusjax.benchmarking` package provides multi-seed experiment execution, result aggregation, disk serialization, and a command-line interface (`mjax` / `python -m malthusjax.benchmarking`).

---

## 6.1. The Engine Protocol & BenchmarkRunner

### 6.1.1. The `Engine` Protocol
Any evolutionary algorithm participating in `BenchmarkRunner` execution must satisfy the `Engine` protocol (`runner.py`), implementing a single required method:

$$\text{run\_once}(\text{key} : \text{chex.Array}) \to \text{Dict}[\text{str}, \text{Any}]$$

The returned dictionary contains:
- `"history"`: List of per-generation metric dictionaries (`generation`, `best_fitness`, `mean_fitness`, etc.).
- `"summary"`: Final aggregated metrics dictionary (`best_fitness`, `final_generation`, `total_evaluations`).
- `"timings"`: Optional execution timing breakdown dictionary.

### 6.1.2. The `BenchmarkRunner` Class
The `BenchmarkRunner` dataclass orchestrates multi-seed experiment execution:

- **Constructor Arguments**:
  - `engine: Engine` — Protocol-compliant engine instance.
  - `experiment_name: str` (default `"benchmark_experiment"`).
  - `output_dir: Optional[Path]` (default `None`).
  - `write_artifacts: bool` (default `True`).
  - `prng_impl: Optional[str]` (default `None`) — PRNG backend name (`"threefry2x32"`, `"philox4x32"`).
  - `trace_dir: Optional[Path]` (default `None`) — Directory for optional JAX trace output.
  - `serialize_history: bool` (default `True`) — If `False`, drops per-generation history arrays to conserve memory during large sweeps.

- **Execution (`run`)**:
  - Iterates through a sequence of random seeds (`seeds: Sequence[int]`).
  - Converts each seed into a JAX PRNG key (`create_key` or `jr.PRNGKey`).
  - Executes `engine.run_once(key)`, measuring wall-clock duration (`duration_seconds`).
  - Returns an `ExperimentResult` containing all `RunResult` records and execution metadata.

---

## 6.2. Results Schema

### 6.2.1. `RunResult`
Immutable dataclass recording a single seeded engine run (`results.py`):
- `seed: int` — PRNG seed value.
- `status: str` — Execution status (`"success"`, `"error"`, `"failure"`, `"timeout"`).
- `metrics: Dict[str, float]` — Computed numeric metrics (exposes `.summary` property alias).
- `history: List[Dict[str, Any]]` — Generation history list.
- `artifacts: Dict[str, str]` — Map of output file paths.
- `duration_seconds: Optional[float]` — Wall-clock execution time.
- `timings: Optional[Dict[str, float]]` — Detailed timing breakdown.
- `error: Optional[str]` — Error message string if `status != "success"`.
- `created_at: datetime` — UTC creation timestamp.

Exposes `.to_dict()`, `.to_json()`, `RunResult.from_dict(d)`, and `RunResult.from_json(s)`.

### 6.2.2. `ExperimentResult`
Dataclass aggregating multi-seed runs for a single pipeline (`results.py`):
- `name: str` — Experiment name.
- `runs: List[RunResult]` — List of per-seed `RunResult` records.
- `metadata: Dict[str, Any]` — Experiment metadata (`seeds`, `total_runs`, `successful_runs`).
- `created_at: datetime` — UTC timestamp.
- `schema_version: str` (default `"0.1"`).

Methods:
- `.aggregated_summary(optimum=None)` — Computes mean, median, stdev, and confidence intervals across seeds for all metrics.
- `.combined_history(seed_field="seed")` — Flattens per-generation history records across seeds into a single tidy list of dictionaries.

### 6.2.3. `ComparisonResult`
Dataclass holding multi-pipeline results (`results.py`):
- `pipelines: Dict[str, ExperimentResult]` — Pipeline name map.
- `shared_config: Dict[str, Any]` — Common configuration parameters.
- `initial_population: Optional[Any]` — Shared initial population PyTree.
- `negate_map: Dict[str, bool]` — Metric sign inversion rules.

Methods:
- `.summary_table(latex=False)` — Returns aggregated metrics table or a formatted LaTeX `tabular` string (`latex=True`).
- `.plot_convergence(save_path=None)` — Generates overlay convergence plots.
- `.plot_boxplots(save_path=None)` — Generates final metric distribution boxplots.

---

## 6.3. Persistence Format

Artifacts are saved to disk via `write_experiment_artifacts` (`io.py`) using the following directory layout:

```text
{output_dir}/
├── metadata/
│   └── config_snapshot.toml       # Copy of the experiment TOML configuration
├── data/
│   ├── pipeline_{name}/
│   │   ├── seed_1.json            # RunResult serialized to JSON
│   │   ├── seed_2.json
│   │   └── ...
└── analysis/
    ├── {pipeline}_summary.json    # Aggregated metrics summary JSON
    ├── comparison_table.csv       # Summary table in CSV format
    ├── comparison_table.md        # Summary table in Markdown format
    └── comparison_table.tex       # Summary table in LaTeX tabular format
```

---

## 6.4. Command-Line Interface (`mjax` / `cli.py`)

The CLI module (`malthusjax.benchmarking.cli`) exposes seven commands via `argparse`:

### 1. `mjax run <config>`
- **Positionals**:
  - `config` (Path): Path to experiment TOML config file.
- **Description**: Loads TOML file, runs multi-seed pipeline execution, and dumps metadata/raw JSON data to `results/{config.stem}`.

### 2. `mjax parity <config>`
- **Positionals**:
  - `config` (Path): Path to parity TOML config file.
- **Description**: Runs seed-aligned statistical parity execution, enforcing a shared initial population across pipelines.

### 3. `mjax analyze <results_dir>`
- **Positionals**:
  - `results_dir` (Path): Directory containing raw JSON data.
- **Description**: Computes summary statistics or statistical parity comparisons and writes summaries/tables to `{results_dir}/analysis`.

### 4. `mjax plot <results_dir>`
- **Positionals**:
  - `results_dir` (Path): Directory containing raw JSON data.
- **Description**: Generates diagnostic plots (`convergence.png`, `fitness_distribution.png`, `timings.png`) into `{results_dir}/plots`.

### 5. `mjax report <results_dir>`
- **Positionals**:
  - `results_dir` (Path): Directory containing raw JSON data.
- **Description**: Sequentially executes `handle_analyze` and `handle_plot`.

### 6. `mjax aggregate --out_dir <out_dir> <results_dirs...>`
- **Required Flags**:
  - `--out_dir` (Path): Output directory for the aggregate suite.
- **Positionals**:
  - `results_dirs` (Path, 1 or more): Experiment result directories to aggregate.
- **Description**: Combines multiple experiment result directories into an aggregate `MetaComparison` report grid (`convergence_grid.png`, `fitness_distribution_grid.png`, `aggregate_summary.json`).

### 7. `mjax catalog`
- **Description**: Instantiates `OperatorCatalog` and lists all registered framework operators to standard output.
