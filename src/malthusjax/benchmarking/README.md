# malthusjax.benchmarking — Experiment Measurement & Analysis

**Benchmarking** is MalthusJAX's measurement infrastructure layer. It orchestrates evolutionary algorithm execution across multiple seeds, collects results with statistics, persists artifacts atomically, and provides rich analysis tools. It sits between **Composer** (high-level API) and **Engines** (low-level execution).

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [BenchmarkRunner — Orchestration](#2-benchmarkrunner--orchestration)
3. [Engine Protocol & Custom Engines](#3-engine-protocol--custom-engines)
4. [Result Objects & Analysis](#4-result-objects--analysis)
5. [I/O & Persistence](#5-io--persistence)
6. [pytest-benchmark Integration](#6-pytest-benchmark-integration)
7. [Data Registry & Management](#7-data-registry--management)
8. [CLI Usage](#8-cli-usage)
9. [Integration Examples](#9-integration-examples)
10. [Troubleshooting & Best Practices](#10-troubleshooting--best-practices)

---

## 1) Overview & Architecture

### What is Benchmarking?

The benchmarking module provides **reproducible measurement infrastructure** for evolutionary algorithms. It abstracts:

- **Execution orchestration** — Run engines across multiple seeds with progress tracking
- **Result aggregation** — Collect multi-seed statistics (mean, median, stdev)
- **Persistent storage** — Atomic writes (JSON summary, CSV histories)
- **Data management** — Registry for external data sources
- **Analysis tools** — Parse pytest benchmarks, export to pandas

### Role in the Stack

```
┌─────────────────────────────────────┐
│  Composer (High-Level API)          │
│  - quick_run()                       │
│  - from_toml()                       │
│  - compare()                         │
└──────────────┬──────────────────────┘
               │ uses
        ┌──────▼──────────────────────┐
        │  BenchmarkRunner (THIS)      │
        │  - orchestrates seeds        │
        │  - collects results          │
        │  - persists artifacts        │
        └──────┬──────────────────────┘
               │
        ┌──────▼──────────────────────┐
        │  Engine (Abstract)           │
        │  - run_once(key)             │
        │ (MalthusJAX, Evosax, custom) │
        └──────────────────────────────┘
```

### Core Result Types

| Type | Scope | Usage |
|------|-------|-------|
| **RunResult** | Single seeded execution | Returned by engine.run_once() |
| **ExperimentResult** | Multiple seeds aggregated | Primary return from BenchmarkRunner.run() |
| **ComparisonResult** | Multiple pipelines compared | From Composer.compare() |

### Data Flow

```
Loop over seeds:
  1. Create PRNG key from seed
  2. Call engine.run_once(key)
  3. Collect result: {history, summary, timings}
  4. Wrap in RunResult (seed, status, metrics, history)
  
Aggregate:
  5. Combine all RunResult → ExperimentResult
  6. Compute statistics (mean, median, stdev)
  
Persist:
  7. Write summary.json (atomic: temp → rename)
  8. Write histories.csv (flattened + seed labels)
  9. Create seed_XXXX/ directories
```

---

## 2) BenchmarkRunner — Orchestration

**BenchmarkRunner** drives evolutionary engines across multiple seeds, collecting results and optionally persisting artifacts.

### Basic Usage

```python
from malthusjax.benchmarking import BenchmarkRunner, StubEngine

# Create a simple test engine
engine = StubEngine(max_steps=100)

# Initialize runner
runner = BenchmarkRunner(
    engine=engine,
    experiment_name="my_experiment",
    output_dir="results/my_experiment",
    write_artifacts=True,
)

# Run across seeds
result = runner.run(seeds=(1, 2, 3))

# Access results
print(result.aggregated_summary())
```

### Result Structure

The returned `ExperimentResult` contains:

```python
result.runs              # List[RunResult] — one per seed
result.name             # "my_experiment"
result.metadata         # Dict with seeds, run counts, artifact paths
result.created_at       # Timestamp
result.schema_version   # "0.1"
```

### Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | Engine | required | Engine implementing `run_once()` protocol |
| `experiment_name` | str | "benchmark_experiment" | Identifier for results and logs |
| `output_dir` | Path or str | None | Directory for artifacts (JSON, CSV, seed folders) |
| `write_artifacts` | bool | True | Write summary.json and histories.csv |
| `prng_impl` | str | None | PRNG implementation ("jax", "numpy", auto-select if None) |
| `trace_dir` | Path or str | None | Directory for JAX profiler traces (first seed only) |

### Example: With JAX Tracing

```python
from pathlib import Path
from malthusjax.benchmarking import BenchmarkRunner
from malthusjax.composer import Composer

composer = Composer.create_default()

# Build a real engine from Composer
engine = composer._build_real_engine(
    fitness="sphere:dim=10",
    selection="tournament:num_selections=25,tournament_size=3",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1,mutation_strength=0.1",
    genome_type="real",
    pop_size=50,
    generations=20,
    genome_shape=10,
    bounds=(-5.0, 5.0),
    elitism=2,
    maximize=False,
)

runner = BenchmarkRunner(
    engine=engine,
    experiment_name="traced_experiment",
    output_dir="results/traced",
    trace_dir=Path("results/traces"),  # Capture first seed
    write_artifacts=True,
)

result = runner.run(seeds=(42, 43, 44))
print(f"Completed {len(result.runs)} runs")
print(f"Traces written to: results/traces/")
```

---

## 3) Engine Protocol & Custom Engines

### Engine Protocol

Any engine used with `BenchmarkRunner` must implement the **Engine protocol**:

```python
from typing import Any, Dict, Protocol

class Engine(Protocol):
    """Simple protocol for evolutionary engines."""
    
    def run_once(self, key: Array) -> Dict[str, Any]:
        """Execute one run and return results.
        
        Returns:
            Dict with keys:
            - 'history': List[Dict] — per-generation statistics
            - 'summary': Dict — final metrics (best_fitness, etc.)
            - 'timings': Dict (optional) — performance metrics
        """
        ...
```

### Contract Details

**Required return keys**:
- **`history`** — List of generation dicts. Each dict should contain:
  - `'generation'` (int) — Generation number
  - `'best_fitness'` (float) — Best individual fitness
  - `'mean_fitness'` (float) — Population mean fitness
  - Additional metrics as needed

- **`summary`** — Dict of final metrics:
  - `'best_fitness'` (float) — Final best fitness
  - `'mean_fitness'` (float) — Final population mean
  - `'initial_fitness'` (float) — Starting best fitness
  - Additional metrics as needed

- **`timings`** (optional) — Dict of timing data:
  - `'total'` (float) — Total runtime
  - `'evaluation'` (float) — Evaluation time
  - Other breakdown metrics

### StubEngine Example

A minimal engine for testing:

```python
from malthusjax.benchmarking import StubEngine

# Create a stub that runs for 20 generations
stub = StubEngine(generations=20)

result = stub.run_once(key)
print(result.keys())  # dict_keys(['history', 'summary', 'timings'])
print(len(result['history']))  # 20 generations
```

### Custom Engine Implementation

Implement any engine to work with BenchmarkRunner:

```python
from typing import Any, Dict
import jax.random as jr
import jax.numpy as jnp

class MyCustomEngine:
    """Simple custom evolutionary engine."""
    
    def __init__(self, pop_size=50, generations=100):
        self.pop_size = pop_size
        self.generations = generations
    
    def run_once(self, key: jnp.ndarray) -> Dict[str, Any]:
        """Simulate evolution and return result structure."""
        history = []
        
        # Simulate generations
        for gen in range(self.generations):
            # Dummy fitness (random)
            best_fitness = 10.0 - gen * 0.05  # Converge over time
            mean_fitness = best_fitness + 1.0
            
            history.append({
                'generation': gen,
                'best_fitness': float(best_fitness),
                'mean_fitness': float(mean_fitness),
            })
        
        return {
            'history': history,
            'summary': {
                'best_fitness': float(history[-1]['best_fitness']),
                'mean_fitness': float(history[-1]['mean_fitness']),
                'initial_fitness': float(history[0]['best_fitness']),
            },
            'timings': {'total': 0.5},
        }
```

Use it with BenchmarkRunner:

```python
from malthusjax.benchmarking import BenchmarkRunner

engine = MyCustomEngine(pop_size=50, generations=100)

runner = BenchmarkRunner(
    engine=engine,
    experiment_name="custom_engine_test",
    output_dir="results/custom",
)

result = runner.run(seeds=(1, 2, 3))
print(f"Completed {len(result.runs)} runs")
```

---

## 4) Result Objects & Analysis

### RunResult — Single Seed Execution

Represents one seeded run:

```python
from malthusjax.benchmarking import RunResult

run = result.runs[0]

print(f"Seed: {run.seed}")                # Random seed value
print(f"Status: {run.status}")            # 'success', 'failure', 'error', 'timeout'
print(f"Metrics: {run.metrics}")          # Dict of final metrics
print(f"History length: {len(run.history)}")  # Generations
print(f"Duration: {run.duration_seconds}s")   # Wall time
print(f"Error: {run.error}")              # None if success
```

### ExperimentResult — Multi-Seed Aggregation

Aggregates multiple RunResult objects with statistics:

#### `.aggregated_summary()` — Multi-Seed Statistics

```python
result = runner.run(seeds=(42, 43, 44))

# Compute mean, median, stdev across seeds
agg = result.aggregated_summary()

for metric, stats in agg.items():
    mean = stats['mean']
    median = stats['median']
    stdev = stats['stdev']
    print(f"{metric}: {mean:.4f} ± {stdev:.4f} (median: {median:.4f})")
```

Output:
```
initial_fitness: 45.2301 ± 2.1543 (median: 44.9800)
best_fitness: 0.5432 ± 0.0234 (median: 0.5401)
mean_fitness: 5.2103 ± 0.3421 (median: 5.1809)
final_generation: 99.0000 ± 0.0000 (median: 99.0000)
total_evaluations: 5000.0000 ± 0.0000 (median: 5000.0000)
```

#### `.combined_history()` — Flattened Histories

Combine all run histories into a single list with seed labels:

```python
history = result.combined_history(seed_field="seed")

# history = [
#     {'generation': 0, 'best_fitness': 45.2, 'mean_fitness': 48.1, 'seed': 42},
#     {'generation': 1, 'best_fitness': 42.1, 'mean_fitness': 46.2, 'seed': 42},
#     ...
#     {'generation': 0, 'best_fitness': 44.8, 'mean_fitness': 47.9, 'seed': 43},
#     ...
# ]

# Export to pandas DataFrame
import pandas as pd
df = pd.DataFrame(history)
print(df.head())

# Group by seed and plot
grouped = df.groupby('seed')['best_fitness']
for seed, group in grouped:
    print(f"Seed {seed}: final best={group.iloc[-1]:.4f}")
```

#### `.canonical_summary` — First Seed Metrics

Quick reference (metrics from first seed):

```python
canonical = result.canonical_summary
print(f"Best fitness: {canonical['best_fitness']}")
```

### Serialization

Save and load results:

```python
# Save to JSON
result_dict = result.to_dict()
result_json = result.to_json()

# Load from JSON
from malthusjax.benchmarking.io import read_summary_json
loaded_result = read_summary_json("results/summary.json")
```

---

## 5) I/O & Persistence

### Atomic Write Pattern

All writes use **atomic operations** (write to temp file, then rename) to prevent corruption:

```
1. Open: path.json.tmp
2. Write data to temp file
3. Rename: path.json.tmp → path.json

Benefits:
- No partial writes if process crashes
- Safe for concurrent access (on same filesystem)
- Automatic cleanup on error
```

### Write Summary JSON

Persistent experiment snapshot:

```python
from malthusjax.benchmarking.io import write_summary_json, read_summary_json
from pathlib import Path

# Automatically called by runner.run() if write_artifacts=True
write_summary_json(result, Path("results/summary.json"))

# Read back
loaded = read_summary_json(Path("results/summary.json"))
print(loaded.name)  # "my_experiment"
print(len(loaded.runs))  # Number of seeds
```

### Write Histories CSV

Flattened convergence data for analysis:

```python
from malthusjax.benchmarking.io import write_histories_csv
import pandas as pd

write_histories_csv(result, Path("results/histories.csv"))

# Read back with pandas
df = pd.read_csv("results/histories.csv")
print(df.head())
# Columns: seed, generation, best_fitness, mean_fitness, ...
```

### Per-Seed Directories

Organize seed-specific outputs:

```python
from malthusjax.benchmarking.io import ensure_seed_folder

# Create and get path to seed-specific folder
seed_dir = ensure_seed_folder("results/my_exp", seed=42)
# → results/my_exp/seed_0042/

# Use for seed-specific artifacts
trace_path = seed_dir / "trace.pb"
```

### DataLoader — Universal File Reader

Load data from various formats:

```python
from malthusjax.benchmarking.io import DataLoader
from pathlib import Path

# Automatically detects format (CSV, JSON, HDF5)
data = DataLoader.load_any(Path("data/training_set.csv"))

# Explicit format
csv_data = DataLoader.load_csv(Path("data/features.csv"))
json_data = DataLoader.load_json(Path("data/config.json"))
```

### Full Artifact Workflow

```python
from malthusjax.benchmarking.io import write_experiment_artifacts
from pathlib import Path

# BenchmarkRunner calls this internally
written = write_experiment_artifacts(
    experiment_result=result,
    output_dir=Path("results/my_exp"),
    write_csv=True,
    write_json=True,
)

print(written.keys())
# dict_keys(['summary_json', 'histories_csv', 'seed_0001', 'seed_0002', ...])

print(written['summary_json'])  # Path to summary.json
print(written['histories_csv'])  # Path to histories.csv
print(written['seed_0001'])      # Path to seed_0001/ directory
```

---

## 6) pytest-benchmark Integration

### Parse Benchmark Results

Load outputs from `pytest --benchmark-save`:

```python
from malthusjax.benchmarking.analysis import (
    load_benchmark_file,
    benchmarks_to_records,
    to_dataframe,
    compute_grouped_kpis,
)
from pathlib import Path

# Load pytest-benchmark JSON
bench_file = Path(".benchmarks/.json/0001_*.json")
# (Actual filename varies; typically .benchmarks/.json/0001_*.json)

# Read raw benchmark data
data = load_benchmark_file(bench_file)

# Convert to flat records
records = benchmarks_to_records(data)
for rec in records:
    print(f"{rec['name']}: {rec['mean']:.4f}s ± {rec['stddev']:.4f}s")
```

### Export to pandas DataFrame

```python
# Convert to DataFrame for analysis
df = to_dataframe(data)
print(df.head())

# Available columns: group, name, min, max, mean, median, stddev, rounds, etc.

# Filter by group
ga_benchmarks = df[df['group'] == 'genetic_algorithm']
print(ga_benchmarks[['name', 'mean', 'stddev']])
```

### Compute KPI Aggregates

```python
# Group and compute statistics
kpis = compute_grouped_kpis(data)

for (group, name), metrics in kpis.items():
    print(f"{group}/{name}:")
    print(f"  Mean: {metrics['mean']:.4f}s")
    print(f"  Stddev: {metrics['stddev']:.4f}s")
    print(f"  Min: {metrics['min']:.4f}s")
    print(f"  Max: {metrics['max']:.4f}s")
```

---

## 7) Data Registry & Management

### DataRegistry for External Data

Manage reusable data sources:

```python
from malthusjax.benchmarking import DataRegistry

registry = DataRegistry()

# Register file-based data
registry.register(
    "training_data_v1",
    {"source": "file", "path": "data/training_set.csv"}
)

# Register synthetic data config
registry.register(
    "sphere_dim10",
    {"source": "synthetic", "dim": 10, "bounds": (-5.0, 5.0)}
)

# Resolve data
training_data = registry.resolve("training_data_v1")
config = registry.resolve("sphere_dim10")
```

### Passing to Evaluators

```python
# Use data in Composer with evaluators
result = composer.quick_run(
    fitness="custom_evaluator:data_id=training_data_v1",
    selection="tournament:num_selections=25",
    ...,
    data_config=data_registry,
)
```

---

## 8) CLI Usage

### Command-Line Interface

Run experiments directly from terminal:

```bash
# Basic usage
python -m malthusjax.benchmarking

# With custom parameters
python -m malthusjax.benchmarking \
  --seeds 1 2 3 4 5 \
  --generations 200 \
  --name my_experiment \
  --output-dir results/my_exp

# Quiet mode (suppress progress bars)
python -m malthusjax.benchmarking --quiet
```

### Available Arguments

```
--seeds [INT ...]              Random seeds (default: 1 2 3)
--generations INT             Number of generations (default: 10)
--name STR                     Experiment name (default: cli_experiment)
--output-dir PATH              Output directory (default: results/{name})
--quiet                        Suppress output
```

### Python API for CLI

```python
from malthusjax.benchmarking.cli import main

# Programmatic CLI invocation
exit_code = main([
    '--seeds', '42', '43', '44',
    '--generations', '200',
    '--name', 'my_experiment',
])

print(exit_code)  # 0 on success
```

---

## 9) Integration Examples

### Direct BenchmarkRunner Usage

```python
from malthusjax.benchmarking import BenchmarkRunner, StubEngine

engine = StubEngine(max_steps=50)

runner = BenchmarkRunner(
    engine=engine,
    experiment_name="direct_example",
    output_dir="results/direct",
    write_artifacts=True,
)

result = runner.run(seeds=(1, 2, 3))

# Analyze
agg = result.aggregated_summary()
print(f"Best fitness: {agg['best_fitness']['mean']:.4f}")

# Export
history = result.combined_history()
import pandas as pd
df = pd.DataFrame(history)
df.to_csv("results/convergence.csv", index=False)
```

### Through Composer

Composer automatically uses BenchmarkRunner internally:

```python
from malthusjax.composer import Composer

composer = Composer.create_default()

# Composer.quick_run() → uses BenchmarkRunner internally
result = composer.quick_run(
    fitness="sphere:dim=10",
    selection="tournament:num_selections=25,tournament_size=3",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1,mutation_strength=0.1",
    seeds=(42, 43, 44),
    output_dir="results/composer_exp",
)

# Same result analysis API
agg = result.aggregated_summary()
history = result.combined_history()
```

### Multi-Engine Comparison

Compare different engines:

```python
from malthusjax.benchmarking import BenchmarkRunner
from malthusjax.composer import Composer

engines = {}

# Build multiple engines
composer = Composer.create_default()

engines["gaussian_mutation"] = composer._build_real_engine(
    fitness="sphere:dim=10",
    mutation="gaussian:mutation_rate=0.1,mutation_strength=0.1",
    pop_size=50,
    generations=50,
    genome_type="real",
    genome_shape=10,
)

engines["polynomial_mutation"] = composer._build_real_engine(
    fitness="sphere:dim=10",
    mutation="polynomial:mutation_rate=0.1,eta=20",
    pop_size=50,
    generations=50,
    genome_type="real",
    genome_shape=10,
)

# Run each
results = {}
for name, engine in engines.items():
    runner = BenchmarkRunner(
        engine=engine,
        experiment_name=name,
        output_dir=f"results/{name}",
    )
    results[name] = runner.run(seeds=(1, 2, 3))

# Compare
for name, result in results.items():
    agg = result.aggregated_summary()
    print(f"{name}: {agg['best_fitness']['mean']:.4f}")
```

---

## 10) Troubleshooting & Best Practices

### Atomic I/O Failures

**Issue**: "Permission denied" or "Device or resource busy"

**Solution**:
- Ensure output directory exists and is writable
- Use absolute paths instead of relative
- Check disk space

```python
from pathlib import Path

output_dir = Path("results/my_exp").resolve()  # Absolute path
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure exists
```

### Timeout Handling

BenchmarkRunner supports timeouts (future enhancement):

```python
runner = BenchmarkRunner(engine=engine, experiment_name="test")

# Timeout not yet implemented in current version
result = runner.run(seeds=(1, 2), timeout_seconds=None)
```

### Memory Issues with Large Histories

For long-running experiments, histories can become large:

```python
# Use DataLoader to stream CSV instead of loading all into memory
from malthusjax.benchmarking.io import DataLoader

history_df = DataLoader.load_csv("results/histories.csv")

# Process in chunks
chunk_size = 100
for i in range(0, len(history_df), chunk_size):
    chunk = history_df.iloc[i:i+chunk_size]
    # Process chunk
```

### JAX Tracing Overhead

Tracing the first seed adds overhead:

```python
# Minimal trace
runner = BenchmarkRunner(
    engine=engine,
    trace_dir="results/traces",  # Only first seed traced
)

# To profile, use separate non-traced run for timing
```

### Best Practices

✅ **Use atomic I/O pattern** — Let BenchmarkRunner handle writes

✅ **Set experiment names** — Makes results discoverable

✅ **Save to absolute paths** — Avoids relative path issues

✅ **Check artifact_paths metadata** — Know where results are written

✅ **Use combined_history() for pandas** — Easy DataFrame conversion

✅ **Implement Engine protocol carefully** — Ensure history/summary dicts are valid

✅ **Test with StubEngine first** — Validate pipeline before full runs

---

## See Also

- [Composer Documentation](../composer/README.md) — High-level API using BenchmarkRunner
- [Engine Documentation](../engine/README.md) — GeneticEngine implementation
- [Results Analysis](../core/fitness/README.md) — Fitness evaluator patterns

