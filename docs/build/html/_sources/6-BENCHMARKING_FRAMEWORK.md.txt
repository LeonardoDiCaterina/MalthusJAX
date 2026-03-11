# Benchmarking Framework: Multi-Seed Experiments and Comparative Analysis

The `malthusjax.benchmarking` module implements infrastructure for multi-seed experiment orchestration, result aggregation, and comparative analysis. The module decouples experiment execution from result interpretation, enabling fair cross-framework comparisons and systematic performance characterization. Multi-seed execution is essential for stochastic algorithms: a single run provides no statistical information about algorithm robustness or convergence reliability.

---

## 6.1. The Engine Protocol and Benchmarking Contract

### 6.1.1. Protocol Definition and Universal Interface

The `Engine` protocol establishes a contract that any evolutionary algorithm must satisfy to participate in benchmarking campaigns. The protocol defines a single required method:

$$\text{run\_once}(\text{key} : \text{chex.Array}) \to \text{Dict}[\text{str}, \text{Any}]$$

The returned dictionary *D* must contain exactly three fields:

- *D*`["history"]`: An ordered list of per-generation summaries. Each summary is a dict with mandatory keys `{"best_fitness", "mean_fitness"}` and optional keys for per-generation metrics (e.g., standard deviation, generation counter)
- *D*`["summary"]`: A final aggregation dict with mandatory key `"final_fitness"` and optional keys like `"evaluations"`, `"wall_time"`
- *D*`["timings"]`: A timing breakdown dict with wall-clock measurements for stages (initialization, evolution, evaluation). May be empty

This protocol decouples BenchmarkRunner from specific engine implementations, permitting any backend (native MalthusJAX, evosax, custom algorithms) to participate in unified comparative workflows. The contract is strictly enforced: missing or malformed fields raise `ValueError`.

### 6.1.2. Stochasticity and Multi-Seed Necessity

Evolutionary algorithms are stochastic processes: the sequence of solutions produced depends on the PRNG seed. A single run provides no information about algorithmic robustness, convergence variability, or statistical significance. Multi-seed execution characterizes the distribution of outcomes across the random seed space.

For a fixed problem and seed sequence *S = [s₁, ..., sₖ]*, invoking `run_once(PRNGKey(sᵢ))` for each *sᵢ* produces *k* independent fitness trajectories. Aggregating these trajectories (element-wise mean, standard deviation) reveals convergence consistency and problem-specific difficulty. The relationship is:

$$\text{Var}[\text{fitness at generation } g] = \mathbb{E}[\text{Var}[\text{fitness} \mid s_i]] + \text{Var}[\mathbb{E}[\text{fitness} \mid s_i]]$$

where the first term is within-seed variance (sampling/drift) and the second is across-seed variance (stochastic outcome heterogeneity).

---

## 6.2. Core Result Data Structures

### 6.2.1. RunResult: Single-Seed Execution Record

A `RunResult` encapsulates the complete output from a single seed execution:

```python
@dataclass
class RunResult:
    seed: int                          # Random seed used for this run
    status: str                        # "success", "failure", "timeout", "error"
    history: List[Dict[str, float]]   # Per-generation summaries
    summary: Dict[str, float]          # Final aggregated metrics
    timings: Dict[str, float]          # Wall-clock breakdown
    duration_seconds: float            # Total wall time
    created_at: str                    # ISO 8601 timestamp
    error_message: Optional[str]       # Error details if status != "success"
    artifacts: Dict[str, str]          # Paths to written seed-level files
    schema_version: str                # Version identifier for format
```

The `status` field encodes execution outcome: `"success"` indicates normal completion; `"failure"` indicates a runtime error (NaN, out-of-bounds); `"timeout"` indicates exceeded wall-clock limit; `"error"` indicates framework errors (memory, file I/O). The `history` field is a chronologically ordered list; index *i* contains metrics for generation *i*. Both `RunResult.to_json()` and `RunResult.from_json()` methods provide serialization.

### 6.2.2. ExperimentResult: Multi-Seed Aggregation

An `ExperimentResult` aggregates *k* `RunResult` objects from a single algorithm configuration:

```python
@dataclass
class ExperimentResult:
    name: str                          # Experiment identifier
    runs: List[RunResult]              # Per-seed results (length k)
    metadata: Dict[str, Any]           # Engine config, hardware, versions
    created_at: str                    # ISO 8601 timestamp
    schema_version: str                # Version identifier for format
    
    def combined_history(self) -> List[Dict[str, float]]:
        # Tidy-format output: one row per (seed, generation) pair
        ...
    
    def aggregated_summary(self) -> Dict[str, float]:
        # Compute mean and stddev of final metrics across seeds
        ...
```

The method `combined_history()` produces a tidy-format list where each dict contains keys `{seed, generation, best_fitness, mean_fitness, ...}`, facilitating downstream visualization and statistical analysis. The method `aggregated_summary()` returns a dict with keys like `{"final_fitness_mean", "final_fitness_std", "evaluations_mean"}`.

### 6.2.3. ComparisonResult: Multi-Algorithm Analysis

A `ComparisonResult` holds multiple `ExperimentResult` objects for comparative analysis:

```python
@dataclass
class ComparisonResult:
    experiments: Dict[str, ExperimentResult]  # name → ExperimentResult
    created_at: str
    
    def summary_table(self) -> pd.DataFrame:
        # Side-by-side comparison: rows=algorithms, cols=metrics
        ...
    
    def convergence_data(self) -> Dict[str, np.ndarray]:
        # Extract per-algorithm mean convergence curves
        ...
```

The `summary_table()` method returns a pandas DataFrame where each row is an algorithm and columns are aggregated metrics (final fitness, best fitness, mean, std). The `convergence_data()` method returns a dict mapping algorithm names to arrays of shape *(nᵍ, m)* where *nᵍ* is generation count and *m* is 2 (mean and std of best fitness per generation).

---

## 6.3. The BenchmarkRunner: Multi-Seed Orchestration

### 6.3.1. Configuration and Initialization

The `BenchmarkRunner` class configures and executes multi-seed campaigns. Initialization accepts:

```python
runner = BenchmarkRunner(
    engine: Engine,                     # Protocol-conformant engine
    experiment_name: str,               # Experiment identifier
    output_dir: Optional[Path] = None,  # Artifact root directory
    write_artifacts: bool = True,       # Whether to persist results
    prng_impl: str = "threefry",       # PRNG backend
)
```

The `prng_impl` parameter selects the PRNG backend (e.g., `"threefry"`, `"philox"`) used for seed-to-key conversion. All seeds are deterministically converted to JAX PRNG keys via the specified backend, ensuring reproducibility across machines.

### 6.3.2. Multi-Seed Execution Loop

The `run()` method executes the engine across a seed sequence:

```python
result: ExperimentResult = runner.run(
    seeds: Sequence[int],
    timeout_seconds: Optional[float] = None,
)
```

Execution proceeds as:

1. Derive PRNG keys: *kᵢ = create_key(sᵢ, impl=prng_impl)* for each seed *sᵢ ∈ seeds*
2. For each key *kᵢ*, invoke `engine.run_once(kᵢ) → Dᵢ`
3. Wrap *Dᵢ* in a `RunResult` (with status, timings, error details)
4. If `write_artifacts=true`, persist seed-level results
5. Collect all `{RunResult}ᵢ` into an `ExperimentResult`

Seed-level failures (timeout, NaN, exception) do not terminate the campaign; they are captured in the `status` field and execution continues. A progress bar (via tqdm) tracks completion. The final `ExperimentResult` contains all outputs, including partial results from failed seeds.

---

## 6.4. Result Persistence and Artifact Management

### 6.4.1. Result Serialization

Results are persisted in two formats:

- **JSON**: `ExperimentResult` is serialized to `summary.json` in the experiment root directory. The format preserves all metadata, per-seed results, and history. Parsing via `read_summary_json()` reconstructs the `ExperimentResult` object.
- **CSV (Tidy Format)**: Per-seed histories are combined into a single `histories.csv` where each row is *(seed, generation, best_fitness, mean_fitness, ...)*. This format is suitable for downstream tools (Python/R analysis, visualization libraries).

Atomic writes (temporary file → rename) ensure crash safety. Schema versioning (versioned in metadata) permits forward compatibility.

### 6.4.2. Seed-Level Directory Organization

When `write_artifacts=true`, results are stored hierarchically:

```
{output_dir}/{experiment_name}/
  summary.json
  histories.csv
  seed_0000/
    result.json
    log.txt
  seed_0001/
    result.json
    log.txt
  ...
```

Each seed directory is isolated, enabling parallelization (multiple processes writing to different seed directories without contention). The function `ensure_seed_folder()` creates and returns the seed-level directory path.

### 6.4.3. Optional Tracing Artifacts

When JAX HLO tracing is enabled, the first seed's `run_once()` execution produces an HLO module dump. This is persisted in the seed directory for later inspection via XLA profiling tools or custom analysis. Tracing incurs significant memory and time overhead; it is optional and disabled by default.

---

## 6.5. Analysis and Post-Processing

### 6.5.1. pytest-benchmark Integration

For micro-benchmarking (operator performance, kernel latency), the `malthusjax.benchmarking.analysis` submodule provides parsers for pytest-benchmark JSON output. The function `load_benchmark_file()` reads a `.benchmarks/*.json` file and `benchmarks_to_records()` converts the internal structure to a list of flat dicts, one per benchmark. Each dict contains timing statistics (name, min, mean, max, stddev) and optional custom fields injected by test setup.

### 6.5.2. Pandas and NumPy Integration

The method `ExperimentResult.to_dataframe()` converts the tidy-format combined history to a pandas DataFrame with columns `{seed, generation, best_fitness, ...}`. This facilitates element-wise statistical operations (groupby, aggregation, pivoting). The module is optional (pandas is not a core dependency).

---

## 6.6. Command-Line Interface

The module provides a CLI entry point: `python -m malthusjax.benchmarking`. The CLI accepts arguments for experiment name, output directory, seed sequence, and problem specification (via Composer integration). Results are written to disk and a summary is printed to stdout. Error messages include failure details and recovery suggestions.

---

## 6.7. File Map

| Module | Responsibility |
|--------|-----------------|
| `runner.py` | `BenchmarkRunner`, `Engine` protocol, seed-loop orchestration |
| `results.py` | `RunResult`, `ExperimentResult`, `ComparisonResult` dataclasses and serialization |
| `io.py` | JSON/CSV I/O, artifact writing, seed-level directory management |
| `analysis.py` | pytest-benchmark parsing, Pandas integration, tidy-format conversion |
| `cli.py` | Command-line argument parsing, integration with Composer |

---

## 6.8. Summary and Integration Points

The benchmarking module provides reproducible multi-seed experiment execution and result aggregation. By implementing the `Engine` protocol (§6.1), BenchmarkRunner enables heterogeneous backend participation without backend-specific knowledge. Result persistence (§6.4) ensures experiments are reproducible and shareable. Integration with Composer (§5.1.2) provides high-level orchestration of experiment specification, orchestration, and benchmarking in a unified workflow.

