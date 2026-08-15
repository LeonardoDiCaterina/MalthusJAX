# Benchmarking Framework: Multi-Seed Experiments and Comparative Analysis

The `malthusjax.benchmarking` module implements infrastructure for multi-seed experiment orchestration, result aggregation, and comparative analysis. The module decouples experiment execution from result interpretation, enabling fair cross-framework comparisons and systematic performance characterization. Multi-seed execution is essential for stochastic algorithms: a single run provides no statistical information about algorithm robustness or convergence reliability.

---

## 6.1. The Engine Protocol and Benchmarking Contract

### 6.1.1. Protocol Definition and Universal Interface

The `Engine` protocol establishes a contract that any evolutionary algorithm must satisfy to participate in benchmarking campaigns. The protocol defines a single required method:

$$\text{run\_once}(\text{key} : \text{chex.Array}) \to \text{Dict}[\text{str}, \text{Any}]$$

The returned dictionary *D* must contain exactly three fields:

- *D*`["history"]`: An ordered list of per-generation summaries. Previously restricted to scalar fitness, this is now relaxed to support Multi-Objective and Quality-Diversity workloads. It may contain metrics like `hypervolume`, `qd_score`, or `archive_coverage` in addition to or instead of `best_fitness`.
- *D*`["summary"]`: A final aggregation dict containing final metrics (e.g. `"final_qd_score"`, `"evaluations"`, `"wall_time"`).
- *D*`["timings"]`: A timing breakdown dict with wall-clock measurements for stages (initialization, evolution, evaluation). May be empty.

This protocol decouples BenchmarkRunner from specific engine implementations, permitting any backend (native MalthusJAX, evosax, external adapters) to participate in unified comparative workflows. 

### 6.1.2. The Negate Map (Sign-Flipping for Minimization)

To ensure fair comparison across frameworks that enforce different maximization/minimization conventions (e.g., evosax maximizes, but a user's BBOB problem minimizes), the BenchmarkRunner supports a `negate_map`.

A `negate_map` is a dictionary (e.g., `{"best_fitness": True, "hypervolume": False}`) passed to the `BenchmarkRunner`. During result aggregation:
1. If a metric is marked `True` in the `negate_map`, its sign is flipped (multiplied by -1) before being written to the output CSVs and JSONs.
2. This ensures that downstream visualization tools (like `EngineComparator`) can always assume "higher is better" or "lower is better" consistently across all algorithms, regardless of their internal representation.

### 6.1.3. Stochasticity and Multi-Seed Necessity

Evolutionary algorithms are stochastic processes. A single run provides no information about algorithmic robustness. Multi-seed execution characterizes the distribution of outcomes across the random seed space.

For a fixed problem and seed sequence *S = [s₁, ..., sₖ]*, invoking `run_once(PRNGKey(sᵢ))` for each *sᵢ* produces *k* independent trajectories. Aggregating these trajectories (element-wise mean, standard deviation) reveals convergence consistency and problem-specific difficulty.

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
```

### 6.2.3. ComparisonResult: Multi-Algorithm Analysis

A `ComparisonResult` holds multiple `ExperimentResult` objects for comparative analysis:

```python
@dataclass
class ComparisonResult:
    experiments: Dict[str, ExperimentResult]  # name → ExperimentResult
    created_at: str
```

---

## 6.3. The BenchmarkRunner: Multi-Seed Orchestration

### 6.3.1. Configuration and Initialization

The `BenchmarkRunner` class configures and executes multi-seed campaigns:

```python
runner = BenchmarkRunner(
    engine: Engine,                     # Protocol-conformant engine
    experiment_name: str,               # Experiment identifier
    output_dir: Optional[Path] = None,  # Artifact root directory
    write_artifacts: bool = True,       # Whether to persist results
    prng_impl: str = "threefry",       # PRNG backend
    negate_map: Optional[Dict[str, bool]] = None, # Metric sign-flipping
)
```

### 6.3.2. Multi-Seed Execution Loop

The `run()` method executes the engine across a seed sequence:

1. Derive PRNG keys deterministically.
2. For each key, invoke `engine.run_once(key)`.
3. Apply `negate_map` to the returned `history` and `summary`.
4. Wrap in a `RunResult`.
5. Persist seed-level results.
6. Collect into an `ExperimentResult`.

---

## 6.4. Result Persistence and Artifact Management

### 6.4.1. Result Serialization

Results are persisted in two formats:
- **JSON**: `ExperimentResult` is serialized to `summary.json`.
- **CSV (Tidy Format)**: Per-seed histories are combined into a single `histories.csv`.

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

---

## 6.5. Analysis and Post-Processing

### 6.5.1. Statistical Testing and Significance

The module provides automated statistical testing routines to compare algorithms rigorously:
- `statistical_fitness_delta`: Computes the Mann-Whitney U test (non-parametric) on the final fitness distributions of two engines. It returns p-values and effect sizes (e.g. Vargha-Delaney A) to assess if one algorithm is significantly better than another.
- `statistical_speedup`: Analyzes runtime distributions from pytest-benchmark to identify statistically significant wall-clock speedups.

For academic publication, the `ComparisonResult.to_latex()` method automatically generates formatted LaTeX tables with statistically significant results bolded, adhering to standard GECCO/PPSN formatting guidelines.

### 6.5.2. DataLoader Utility

For combinatorial and real-world benchmarks, the module provides a `DataLoader` utility. It handles reading static datasets securely:
- **TSPLib**: Parses `.tsp` files into JAX arrays for distance matrices and coordinates.
- **CSV/NPZ**: Standard loading for surrogate models or regression datasets.
The DataLoader ensures data is loaded once and cached in device memory, preventing I/O bottlenecks during the scan loop.

### 6.5.3. pytest-benchmark Integration

For micro-benchmarking, the `malthusjax.benchmarking.analysis` submodule provides parsers for pytest-benchmark JSON output to analyze timing statistics.

### 6.5.4. Pandas and NumPy Integration

The method `ExperimentResult.to_dataframe()` converts the tidy-format combined history to a pandas DataFrame. This facilitates element-wise statistical operations.

---

## 6.6. Command-Line Interface

The CLI entry point `python -m malthusjax.benchmarking` accepts arguments for experiment name, output directory, seed sequence, and problem specification (via Composer integration).

---

## 6.7. Summary and Integration Points

The benchmarking module provides reproducible multi-seed experiment execution and result aggregation. By implementing the `Engine` protocol, BenchmarkRunner enables heterogeneous backend participation without backend-specific knowledge. The addition of the `negate_map` and flexible `history` dictionaries natively supports MO and QD paradigms.
