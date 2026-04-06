# malthusjax.composer — Experiment Orchestration Layer

**Composer** is the high-level orchestration layer for defining, running, and comparing evolutionary algorithms. It abstracts away boilerplate configuration, provides string-based DSL (Domain-Specific Language) for operators, and handles result aggregation across multiple seeds and pipelines.

## Table of Contents

1. [Overview & Core Concepts](#1-overview--core-concepts)
2. [Quick-Start: Interactive Experiments](#2-quick-start-interactive-experiments-with-quick_run)
3. [Reproducible Experiments: TOML Configuration](#3-reproducible-experiments-toml-based-configuration)
4. [Multi-Pipeline Comparison](#4-multi-pipeline-comparison-fair-benchmarking)
5. [String DSL & Operator Catalogs](#5-string-dsl--operator-catalogs)
6. [TOML Configuration Reference](#6-toml-configuration-reference)
7. [Result Objects & Analysis](#7-result-objects--analysis)
8. [Advanced Patterns](#8-advanced-patterns)
9. [Configuration Reference](#9-configuration-reference)

---

## 1) Overview & Core Concepts

### What is Composer?

Composer is MalthusJAX's experiment orchestration layer. Instead of manually configuring engines, operators, and result collection, Composer provides:

- **Interactive API** (`quick_run()`) — Explore algorithms with string-based operator specs
- **Declarative TOML** (`from_toml()`) — Version-controllable, reproducible experiments
- **Multi-pipeline Comparison** (`compare()`) — Statistical benchmarking with fair initialization
- **Result Aggregation** — Automatic multi-seed statistics and convergence history collection

### Three Entry Points

```python
from malthusjax.composer import Composer

composer = Composer.create_default()

# 1. Interactive exploration
result = composer.quick_run(
    fitness="sphere:dim=10",
    selection="tournament:num_selections=25,tournament_size=3",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1,mutation_strength=0.1",
    pop_size=50,
    generations=100,
)

# 2. Reproducible experiments from TOML
result = Composer.from_toml("experiment.toml")

# 3. Multi-algorithm comparison
comparison = composer.compare(
    pipelines={
        "Algorithm A": {"mutation": "gaussian:mutation_rate=0.1,mutation_strength=0.1"},
        "Algorithm B": {"mutation": "polynomial:mutation_rate=0.1,eta=20"},
    },
    fitness="sphere:dim=10",
    pop_size=50,
)
```

### Result Types at a Glance

| Type | Scope | Methods |
|------|-------|---------|
| **ExperimentResult** | Single experiment, multiple seeds | `.aggregated_summary()`, `.combined_history()`, `.canonical_summary` |
| **ComparisonResult** | Multiple pipelines (each with multiple seeds) | `.summary_table()`, `.plot_convergence()`, `.convergence_data()` |
| **RunResult** | Single seeded run | `.to_dict()`, `.from_dict()` (for serialization) |

### String DSL Philosophy

Composer uses declarative string specifications for operators:

```
"operator_name:param1=val1,param2=val2,param3=val3"
```

Examples:
- `"sphere:dim=10"` — Sphere function with dimension 10
- `"tournament:num_selections=25,tournament_size=3"` — Tournament selection
- `"gaussian:mutation_rate=0.1,mutation_strength=0.1"` — Gaussian mutation
- `"blend:alpha=0.5"` — Blend crossover with α=0.5

This design enables composable, reproducible, and discoverable algorithm configurations.

### Architecture Diagram

```
┌─────────────────────────────────────────────┐
│  Composer (quick_run, from_toml, compare)   │
├─────────────────────────────────────────────┤
│  Config Parser (TOML, kwargs):              │
│    - operator specs ("sphere:dim=10")        │
│    - shared & pipeline-specific config      │
├─────────────────────────────────────────────┤
│  Catalogs & Registries:                     │
│    - GenomeCatalog (parse genome specs)      │
│    - OperatorCatalog (parse operator specs)  │
│    - EngineRegistry (factory lookup)         │
├─────────────────────────────────────────────┤
│  Engine Factories:                          │
│    - MalthusJAX engine builder              │
│    - Evosax adapter                         │
├─────────────────────────────────────────────┤
│  BenchmarkRunner:                           │
│    - Run engine across seeds                │
│    - Collect RunResult (seed, metrics, history) │
├─────────────────────────────────────────────┤
│  Results Aggregation:                       │
│    - ExperimentResult (multi-seed)          │
│    - ComparisonResult (multi-pipeline)      │
└─────────────────────────────────────────────┘
```

---

## 2) Quick-Start: Interactive Experiments with `quick_run()`

**When to use**: Ad-hoc exploration, interactive tuning, quick validation.

### Basic Pattern

```python
from malthusjax.composer import Composer

composer = Composer.create_default()

result = composer.quick_run(
    fitness="sphere:dim=10",
    selection="tournament:num_selections=25,tournament_size=3",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1,mutation_strength=0.1",
    pop_size=50,
    generations=100,
    seeds=(42, 43, 44),
)

# Access results
summary = result.aggregated_summary()
print(f"Best fitness: {summary['best_fitness']['mean']:.4f} ± {summary['best_fitness']['stdev']:.4f}")
```

### Example 1: Sphere Optimization (Real-Valued)

```python
result = composer.quick_run(
    fitness="sphere:dim=25",
    selection="tournament:num_selections=25,tournament_size=3",
    crossover="uniform_real",
    mutation="gaussian:mutation_rate=0.5,mutation_strength=0.1",
    pop_size=100,
    generations=200,
    seeds=(1, 2, 3),
    bounds=(-5.0, 5.0),
)

print(result.aggregated_summary())
# Access individual run
first_run = result.runs[0]
print(f"Seed {first_run.seed}: best={first_run.metrics['best_fitness']:.4f}")
```

### Example 2: Rastrigin (Highly Multimodal)

```python
result = composer.quick_run(
    fitness="rastrigin:dim=10",
    selection="roulette:num_selections=30",
    crossover="simulated_binary:eta=20",
    mutation="polynomial:mutation_rate=0.1,eta=20",
    pop_size=100,
    generations=300,
    seeds=(42, 43, 44),
)

summary = result.aggregated_summary()
print(summary)
```

### Example 3: Rastrigin (Challenging Landscape)

```python
result = composer.quick_run(
    fitness="rastrigin:dim=10",
    selection="roulette:num_selections=30",
    crossover="simulated_binary:eta=20",
    mutation="polynomial:mutation_rate=0.1,eta=20",
    pop_size=100,
    generations=100,
    seeds=(1, 2, 3),
)

print(result.aggregated_summary())
```

### Example 4: Evosax Backend (DifferentialEvolution)

```python
result = composer.quick_run(
    backend="evosax",
    evosax_strategy="DifferentialEvolution",
    fitness="sphere:dim=10",
    pop_size=30,
    generations=100,
    seeds=(42, 43, 44),
)

summary = result.aggregated_summary()
print(summary)
```

### Example 5: No Operators (Pipeline Testing)

```python
result = composer.quick_run(
    generations=50,
    seeds=(1, 2),
)
print(result.aggregated_summary())
# Useful for validating pipeline infrastructure without expensive evaluation
```

### Parameters Reference

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `fitness` | str | - | Spec like `"sphere:dim=10"`, `"bbob:fn=3,dims=10"` |
| `selection` | str | - | Spec like `"tournament:num_selections=25,tournament_size=3"` |
| `crossover` | str | - | Spec like `"blend:alpha=0.5"`, `"simulated_binary:eta=20"` |
| `mutation` | str | - | Spec like `"gaussian:mutation_rate=0.1,mutation_strength=0.1"` |
| `seeds` | Sequence[int] | (1, 2, 3) | Random seeds for independent runs |
| `pop_size` | int | 50 | Population size (powers of 2 preferred for GPU) |
| `generations` | int | 100 | Full generational cycles |
| `genome_type` | str | "real" | `"real"`, `"binary"`, `"categorical"` |
| `genome_length` | int | 10 | Dimension (real), length (binary), num items (categorical) |
| `bounds` | Tuple[float, float] | (-5.0, 5.0) | Search space bounds for real genomes |
| `maximize` | bool | False | `True` for maximization, `False` for minimization |
| `backend` | str | "malthusjax" | `"malthusjax"` or `"evosax"` |
| `evosax_strategy` | str | "SimpleGA" | Strategy name (only if `backend="evosax"`) |
| `engine_type` | str | "ga" | Engine type (MalthusJAX only) |
| `elitism` | int | 2 | Best individuals carried forward unmodified |
| `experiment_name` | str | "quick_experiment" | Experiment identifier |
| `output_dir` | Path or str | results/{experiment_name} | Output directory |
| `engine` | optional | None | Pre-configured engine (bypasses operator specs) |
| `prng_impl` | str | None | PRNG implementation (auto-select if None) |
| `trace_dir` | Path or str | results/traces | JAX profiler trace directory |
| `data_config` | Dict | None | Custom fitness data (advanced) |

---

## 3) Reproducible Experiments: TOML-Based Configuration

**When to use**: Production runs, version control, team collaboration, paper reproducibility.

### Basic Pattern

```python
from malthusjax.composer import Composer

# Execute all pipelines in experiment.toml
result = Composer.from_toml("experiment.toml")

# Execute only specific pipelines
result = Composer.from_toml("experiment.toml", pipelines=["blend_gaussian", "sbx"])

print(result.summary_table())
result.plot_convergence(seed_index=0)
```

### TOML Structure

```toml
[experiment]
name = "my_experiment"                    # optional

[experiment.shared]
# Defaults applied to all pipelines
fitness = "sphere:dim=10"
selection = "tournament:num_selections=25,tournament_size=3"
pop_size = 50
generations = 100
seeds = [42, 43, 44]
bounds = [-5.0, 5.0]
maximize = false

# Optional: per-pipeline custom data configs
[data.custom_fitness]
# Data fields here (used by custom evaluators)

[pipelines.blend_gaussian]
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.5,mutation_strength=0.1"

[pipelines.sbx_polynomial]
crossover = "simulated_binary:eta=20"
mutation = "polynomial:mutation_rate=0.1,eta=20"

[pipelines.evosax_de]
backend = "evosax"
evosax_strategy = "DifferentialEvolution"
```

### Example TOML File: Sphere Optimization Comparison

```toml
[experiment]
name = "sphere_ga_comparison"
output_dir = "results/sphere_bench"

[experiment.shared]
fitness = "sphere:dim=25"
selection = "tournament:num_selections=30,tournament_size=3"
pop_size = 100
generations = 200
seeds = [42, 43, 44, 45]
bounds = [-5.0, 5.0]
maximize = false

[pipelines.uniform_crossover]
crossover = "uniform_real"
mutation = "gaussian:mutation_rate=0.5,mutation_strength=0.1"

[pipelines.blend_crossover]
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.5,mutation_strength=0.1"

[pipelines.sbx_crossover]
crossover = "simulated_binary:eta=20"
mutation = "polynomial:mutation_rate=0.1,eta=20"
```

### Example TOML File: Multi-Objective with Evosax

```toml
[experiment]
name = "multimodal_comparison"

[experiment.shared]
fitness = "rastrigin:dim=10"
pop_size = 100
generations = 300
seeds = [1, 2, 3]

[pipelines.ga_gaussian]
selection = "tournament:num_selections=25,tournament_size=3"
crossover = "uniform_real"
mutation = "gaussian:mutation_rate=0.1,mutation_strength=0.1"

[pipelines.ga_polynomial]
selection = "tournament:num_selections=25,tournament_size=3"
crossover = "simulated_binary:eta=20"
mutation = "polynomial:mutation_rate=0.1,eta=20"

[pipelines.evosax_de]
backend = "evosax"
evosax_strategy = "DifferentialEvolution"
```

---

## 4) Multi-Pipeline Comparison: Fair Benchmarking

**When to use**: Comparing algorithms, statistical validation, fair initialization control.

### Basic Pattern

```python
from malthusjax.composer import Composer

composer = Composer.create_default()

comparison = composer.compare(
    pipelines={
        "Blend+Gaussian": {
            "crossover": "blend:alpha=0.5",
            "mutation": "gaussian:mutation_rate=0.5,mutation_strength=0.1",
        },
        "SBX+Polynomial": {
            "crossover": "simulated_binary:eta=20",
            "mutation": "polynomial:mutation_rate=0.1,eta=20",
        },
    },
    fitness="sphere:dim=10",        # shared across all
    selection="tournament:num_selections=25,tournament_size=3",
    pop_size=50,
    generations=100,
    seeds=(42, 43, 44),
)

# Results
print(comparison.summary_table())
comparison.plot_convergence(seed_index=0)
```

### Example 1: Three-Way Algorithm Comparison

```python
comparison = composer.compare(
    pipelines={
        "Uniform+Gaussian": {
            "crossover": "uniform_real",
            "mutation": "gaussian:mutation_rate=0.5,mutation_strength=0.1",
        },
        "Blend+Gaussian": {
            "crossover": "blend:alpha=0.5",
            "mutation": "gaussian:mutation_rate=0.5,mutation_strength=0.1",
        },
        "SBX+Polynomial": {
            "crossover": "simulated_binary:eta=20",
            "mutation": "polynomial:mutation_rate=0.1,eta=20",
        },
    },
    fitness="sphere:dim=10",
    pop_size=100,
    generations=200,
    seeds=(42, 43, 44),
)

table = comparison.summary_table()
for pipeline_name, metrics in table.items():
    print(f"{pipeline_name}: best={metrics['best_fitness']:.4f}")
```

### Example 2: Fair Comparison with Shared Initial Population

```python
comparison = composer.compare(
    pipelines={
        "Algorithm A": {...},
        "Algorithm B": {...},
    },
    fitness="rastrigin:dim=10",
    pop_size=100,
    generations=300,
    seeds=(42, 43, 44),
    shared_initial_population=True,  # All pipelines start from same population
    pop_seed=999,                      # Seed for generating shared population
)

# Now differences are purely algorithmic, not initialization variance
table = comparison.summary_table()
```

### Example 3: Mix Backends (MalthusJAX + Evosax)

```python
comparison = composer.compare(
    pipelines={
        "GA+Blend": {
            "backend": "malthusjax",
            "crossover": "blend:alpha=0.5",
            "mutation": "gaussian:mutation_rate=0.1",
        },
        "GA+SBX": {
            "backend": "malthusjax",
            "crossover": "simulated_binary:eta=20",
            "mutation": "polynomial:mutation_rate=0.1,eta=20",
        },
        "Evosax DE": {
            "backend": "evosax",
            "evosax_strategy": "DifferentialEvolution",
        },
    },
    fitness="sphere:dim=10",
    pop_size=50,
    generations=200,
    seeds=(42, 43, 44),
)

table = comparison.summary_table()
comparison.plot_convergence(seed_index=0)
```

### ComparisonResult Methods

#### `summary_table()`

Returns per-pipeline aggregated metrics (mean across seeds). Fitness is automatically normalized so "lower is better" across all pipelines.

```python
table = comparison.summary_table()
# table = {
#     "GA+Blend": {"best_fitness": 0.123, "mean_fitness": 2.345},
#     "GA+SBX": {"best_fitness": 0.089, "mean_fitness": 1.987},
# }
for pipeline_name, metrics in table.items():
    print(f"{pipeline_name}: {metrics}")
```

#### `plot_convergence(seed_index=0, ax=None)`

Matplotlib visualization of convergence curves for all pipelines.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for seed_idx, ax in enumerate(axes):
    comparison.plot_convergence(seed_index=seed_idx, ax=ax)
    ax.set_title(f"Seed {seed_idx + 1}")
    ax.set_ylabel("Best Fitness (lower is better)")
    ax.set_xlabel("Generation")
    ax.legend(loc="best")

plt.tight_layout()
plt.show()
```

#### `convergence_data(seed_index=0)`

Raw history data for custom plotting.

```python
data = comparison.convergence_data(seed_index=0)
# data = {
#     "GA+Blend": [{"generation": 0, "best_fitness": 10.5}, ...],
#     "GA+SBX": [{"generation": 0, "best_fitness": 10.2}, ...],
# }
```

---

## 5) String DSL & Operator Catalogs

The string DSL format is: `"operator_name:param1=val1,param2=val2,param3=val3"`

All operators are registered in catalogs and instantiated dynamically.

### How Parsing Works

```python
from malthusjax.composer.genome_catalog import GenomeCatalog
from malthusjax.composer.catalog import OperatorCatalog

# Parse genome spec
genome_cat = GenomeCatalog()
genome_type, genome_params = genome_cat.parse_spec("real:dim=10,bounds=(-5.0, 5.0)")
# genome_type = "real"
# genome_params = {"dim": 10, "bounds": (-5.0, 5.0)}

# Parse operator spec
op_cat = OperatorCatalog()
op_name, op_params = op_cat.parse_spec("tournament:num_selections=25,tournament_size=3")
# op_name = "tournament"
# op_params = {"num_selections": 25, "tournament_size": 3}
```

### Built-in Genomes

| Spec | Parameters | Description |
|------|-----------|-------------|
| `"real:dim=INT,bounds=(L,U)"` | dim, bounds | Continuous real-valued |
| `"binary:length=INT"` | length | Binary bit strings |
| `"categorical:shape=TUPLE"` | shape | Categorical/permutation |

Examples:
```
"real:dim=10,bounds=(-5.0, 5.0)"
"binary:length=20"
"categorical:shape=(5,3)"
```

### Built-in Fitness Functions

| Spec | Parameters | Description |
|------|-----------|-------------|
| `"sphere:dim=INT"` | dim | Sphere: $\sum_i x_i^2$ |
| `"rastrigin:dim=INT"` | dim | Rastrigin (multimodal) |
| `"griewank:dim=INT"` | dim | Griewank (multimodal) |
| `"bbob:fn=INT,dims=INT"` | fn (1-24), dims (2-40) | Black-Box Optimization Benchmarking suite |
| `"knapsack:capacity=INT,num_items=INT"` | capacity, num_items | 0/1 knapsack |
| `"binary_sum:length=INT"` | length | Sum of bits |
| `"tsp:file=PATH"` | file | Traveling Salesman Problem |
| `"linear_gp:dim=INT"` | dim | Linear Genetic Programming |

Examples:
```
"sphere:dim=25"
"bbob:fn=3,dims=10"
"knapsack:capacity=100,num_items=50"
"tsp:file=data/tsp/berlin52.txt"
```

### Selection Operators

| Spec | Parameters | Description |
|------|-----------|-------------|
| `"tournament:num_selections=INT,tournament_size=INT"` | num_selections, tournament_size (default 3) | Tournament competition |
| `"roulette:num_selections=INT"` | num_selections | Fitness-proportional (SUS-like) |
| `"elite_pool:num_selections=INT,elite_k=INT"` | num_selections, elite_k | Select from top-k |

Examples:
```
"tournament:num_selections=25,tournament_size=3"
"roulette:num_selections=30"
"elite_pool:num_selections=25,elite_k=10"
```

### Crossover Operators (Real Genomes)

| Spec | Parameters | Description |
|------|-----------|-------------|
| `"uniform_real"` | (none) | 50% per-gene exchange |
| `"blend:alpha=FLOAT"` | alpha (default 0.5) | Blend with expansion factor |
| `"simulated_binary:eta=FLOAT"` | eta (default 20) | SBX (Simulated Binary Crossover) |
| `"binomial:cr=FLOAT"` | cr (default 0.7) | DE-style binomial crossover |

Examples:
```
"uniform_real"
"blend:alpha=0.5"
"simulated_binary:eta=20"
"binomial:cr=0.7"
```

### Crossover Operators (Binary Genomes)

| Spec | Parameters | Description |
|------|-----------|-------------|
| `"uniform_binary"` | (none) | Uniform bit-wise crossover |
| `"single_point"` | (none) | Single-point crossover |

Examples:
```
"uniform_binary"
"single_point"
```

### Mutation Operators (Real Genomes)

| Spec | Parameters | Description |
|------|-----------|-------------|
| `"gaussian:mutation_rate=FLOAT,mutation_strength=FLOAT"` | mutation_rate, mutation_strength (std dev) | Gaussian noise |
| `"ball:mutation_rate=FLOAT"` | mutation_rate | Uniform ball (hypersphere) mutation |
| `"polynomial:mutation_rate=FLOAT,eta=FLOAT"` | mutation_rate, eta (default 20) | Polynomial mutation |

Examples:
```
"gaussian:mutation_rate=0.5,mutation_strength=0.1"
"ball:mutation_rate=0.1"
"polynomial:mutation_rate=0.1,eta=20"
```

### Mutation Operators (Binary Genomes)

| Spec | Parameters | Description |
|------|-----------|-------------|
| `"bitflip:mutation_rate=FLOAT"` | mutation_rate | Flip each bit independently |
| `"scramble"` | (none) | Random reordering (order-based) |
| `"swap"` | (none) | Random bit swap (order-based) |

Examples:
```
"bitflip:mutation_rate=0.05"
"scramble"
"swap"
```

---

## 6) TOML Configuration Reference

### Full TOML Schema

```toml
[experiment]
name = "STRING"                           # optional; default uses experiment_name from quick_run
output_dir = "PATH"                       # optional; default is results/{experiment_name}

[experiment.shared]
# All parameters from quick_run() can go here (except engine, which is not serializable)

# Operators
fitness = "STRING"                        # spec like "sphere:dim=10"
selection = "STRING"                      # spec like "tournament:..."
crossover = "STRING"                      # spec like "blend:alpha=0.5"
mutation = "STRING"                       # spec like "gaussian:..."

# Genome & bounds
genome = "STRING"                         # spec like "real:dim=10,bounds=(-5.0,5.0)"
genome_type = "STRING"                    # "real", "binary", "categorical"
genome_length = INT                       # dimension or length
bounds = [FLOAT, FLOAT]                   # [lower, upper]

# Evolution
pop_size = INT                            # default 50
generations = INT                         # default 100
seeds = [INT, INT, ...]                   # list of seeds; default [1, 2, 3]
elitism = INT                             # default 2 (MalthusJAX only)

# Optimization direction
maximize = BOOL                           # default false

# Backend
backend = "STRING"                        # "malthusjax" (default) or "evosax"
evosax_strategy = "STRING"                # if backend="evosax"; default "SimpleGA"
engine_type = "STRING"                    # if backend="malthusjax"; default "ga"

# Advanced
prng_impl = "STRING"                      # PRNG implementation (optional)

# Optional per-pipeline data config
[data.my_data_config]
# Custom data fields here (used by custom evaluators)

# Pipeline definitions (each overrides [experiment.shared])
[pipelines.pipeline_name_1]
# Parameters here override [experiment.shared]
crossover = "simulated_binary:eta=20"
mutation = "polynomial:mutation_rate=0.1,eta=20"

[pipelines.pipeline_name_2]
backend = "evosax"
evosax_strategy = "DifferentialEvolution"
```

### Data Configuration: `[data.*]` Sections

When fitness evaluators need external data (e.g., TSP distance matrices, knapsack item weights), define `[data.*]` sections in your TOML file.

#### Data Source Types

| Type | Use Case | Configuration |
|------|----------|---------------|
| **synthetic** | Generate data via factory (random matrices, etc.) | Source + generation params |
| **file** | Load data from disk | Source + file path |

#### Example 1: TSP with Synthetic Data

```toml
[experiment]
name = "tsp_synthetic"

[experiment.shared]
fitness = "tsp:data_id=random_cities_50"
pop_size = 100
generations = 200
seeds = [42, 43, 44]
genome_type = "categorical"  # TSP uses permutation genomes

# Synthetic TSP: randomly generated 50-city distance matrix
[data.random_cities_50]
source = "synthetic"
num_cities = 50
random_seed = 999

[pipelines.genetic_algorithm]
mutation = "swap:mutation_rate=0.1"
```

#### Example 2: TSP with File Data

```toml
[experiment]
name = "tsp_berlin"

[experiment.shared]
fitness = "tsp:data_id=berlin52"
pop_size = 100
generations = 300
seeds = [42, 43, 44]
genome_type = "categorical"

# Load TSP from file (expected: TSPLIB format or distance matrix)
[data.berlin52]
source = "file"
path = "data/tsp/berlin52.tsp"

[pipelines.genetic_algorithm]
mutation = "swap:mutation_rate=0.1"
```

#### Example 3: Knapsack with Synthetic Items

```toml
[experiment]
name = "knapsack_0_1"

[experiment.shared]
fitness = "knapsack:data_id=items_50"
pop_size = 80
generations = 100
seeds = [1, 2, 3]
genome_type = "binary"  # Knapsack uses binary (select/not select)

# Define knapsack items: weights, values, capacity
[data.items_50]
source = "synthetic"
num_items = 50
capacity = 500
weight_range = [10, 50]
value_range = [30, 150]
random_seed = 42

[pipelines.uniform_mutation]
mutation = "bitflip:mutation_rate=0.05"
crossover = "uniform_binary"
```

#### Example 4: Multi-Seed TSP (Different Data Per Seed)

For statistical comparison, run same GA on multiple problem instances:

```toml
[experiment]
name = "tsp_multi_instance"

[experiment.shared]
genome_type = "categorical"
pop_size = 100
generations = 200
seeds = [1, 2, 3]

# Instance 1: Random 30-city problem
[data.random_30]
source = "synthetic"
num_cities = 30
random_seed = 101

# Instance 2: Random 50-city problem
[data.random_50]
source = "synthetic"
num_cities = 50
random_seed = 102

# Pipeline A: Test on 30-city
[pipelines.on_30_cities]
fitness = "tsp:data_id=random_30"
mutation = "swap:mutation_rate=0.1"

# Pipeline B: Test on 50-city
[pipelines.on_50_cities]
fitness = "tsp:data_id=random_50"
mutation = "swap:mutation_rate=0.1"
```

---

## Data Flow When Loading from TOML

```
Composer.from_toml("experiment.toml")
  ↓
Parse [experiment.shared], [pipelines.*], [data.*]
  ↓
For each pipeline:
  → Merge shared config + pipeline overrides
  → Extract fitness spec (e.g., "tsp:data_id=berlin52")
  → Call quick_run(..., data_config=extracted_data_configs)
    ↓
    Build DataRegistry from [data.*] sections
      ├─ source="synthetic": Config passed to factory
      └─ source="file": Path loaded via DataLoader
    ↓
    catalog.get(fitness_spec, data_registry=registry)
      → Resolves data_id from spec
      → Fetches data from registry
    ↓
    Create evaluator with data=resolved_data
      → Used during fitness evaluation
```
```

### Inheritance & Override Rules

1. **Shared defaults** — All keys in `[experiment.shared]` apply to all pipelines
2. **Pipeline overrides** — Keys in `[pipelines.NAME]` override shared defaults
3. **Command-line override** — Parameters passed to `from_toml()` override TOML

Example:
```toml
[experiment.shared]
fitness = "sphere:dim=10"
pop_size = 50
generations = 100

[pipelines.blend]
crossover = "blend:alpha=0.5"          # overrides any shared crossover

[pipelines.sbx]
crossover = "simulated_binary:eta=20"  # different crossover
pop_size = 100                          # different pop_size
```

---

## 7) Result Objects & Analysis

### ExperimentResult (Single Experiment, Multiple Seeds)

Returned by `quick_run()`.

#### Attributes

- **`.runs`** — List of RunResult objects (one per seed)
- **`.name`** — Experiment identifier
- **`.metadata`** — Arbitrary metadata dict
- **`.canonical_summary`** — Metrics from first seed (quick reference)

#### Methods

##### `.aggregated_summary()` — Multi-Seed Statistics

```python
result = composer.quick_run(
    fitness="sphere:dim=10",
    seeds=(42, 43, 44),
    ...
)

agg = result.aggregated_summary()
# agg = {
#     'best_fitness': {'mean': 0.123, 'median': 0.110, 'stdev': 0.045},
#     'mean_fitness': {'mean': 2.345, 'median': 2.310, 'stdev': 0.120},
# }

print(f"Best fitness: {agg['best_fitness']['mean']:.4f} ± {agg['best_fitness']['stdev']:.4f}")
```

##### `.combined_history(seed_field="seed")` — Flattened History

Flatten all run histories into a single list with seed labels for pandas/CSV export.

```python
history = result.combined_history(seed_field="seed")
# history = [
#     {'generation': 0, 'best_fitness': 10.5, 'seed': 42},
#     {'generation': 1, 'best_fitness': 9.2, 'seed': 42},
#     ...
#     {'generation': 0, 'best_fitness': 10.8, 'seed': 43},
#     ...
# ]

# Export to pandas
import pandas as pd
df = pd.DataFrame(history)
print(df)

# Export to CSV
import csv
with open("convergence.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)
```

##### `.to_dict() / .to_json()` — Serialization

```python
# Save results
result_dict = result.to_dict()
import json
with open("result.json", "w") as f:
    json.dump(result_dict, f)

# Load results
from malthusjax.benchmarking.results import ExperimentResult
loaded = ExperimentResult.from_dict(result_dict)
```

### ComparisonResult (Multiple Pipelines)

Returned by `compare()` and `from_toml()`.

#### Attributes

- **`.pipelines`** — Dict mapping pipeline name → ExperimentResult
- **`.shared_config`** — Common configuration parameters
- **`.initial_population`** — Shared initial pop array (if used)
- **`.names`** — List of pipeline names
- **`.negate_map`** — Per-pipeline fitness sign-flip flags

#### Methods

##### `.summary_table()` — Per-Pipeline Metrics

```python
comparison = composer.compare(
    pipelines={
        "Algorithm A": {...},
        "Algorithm B": {...},
    },
    fitness="sphere:dim=10",
    seeds=(42, 43, 44),
)

table = comparison.summary_table()
# table = {
#     "Algorithm A": {"best_fitness": 0.123, "mean_fitness": 2.345},
#     "Algorithm B": {"best_fitness": 0.089, "mean_fitness": 1.987},
# }

for pipeline_name, metrics in table.items():
    print(f"{pipeline_name}: best={metrics['best_fitness']:.4f}")
```

##### `.plot_convergence(seed_index=0, ax=None)` — Matplotlib Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for seed_idx, ax in enumerate(axes):
    comparison.plot_convergence(seed_index=seed_idx, ax=ax)
    ax.set_title(f"Seed {seed_idx + 1}")
    ax.set_ylabel("Best Fitness (lower is better)")
    ax.set_xlabel("Generation")
    ax.legend(loc="best")

plt.tight_layout()
plt.show()
```

##### `.convergence_data(seed_index=0)` — Raw History Data

```python
data = comparison.convergence_data(seed_index=0)
# data = {
#     "Algorithm A": [
#         {"generation": 0, "best_fitness": 10.5},
#         {"generation": 1, "best_fitness": 9.2},
#         ...
#     ],
#     "Algorithm B": [
#         {"generation": 0, "best_fitness": 10.8},
#         ...
#     ],
# }
```

### Example: Complete Analysis Workflow

```python
import pandas as pd
import matplotlib.pyplot as plt

# Run comparison
comparison = composer.compare(
    pipelines={
        "GA+Blend": {"crossover": "blend:alpha=0.5", ...},
        "GA+SBX": {"crossover": "simulated_binary:eta=20", ...},
    },
    fitness="sphere:dim=10",
    pop_size=100,
    generations=200,
    seeds=(42, 43, 44),
)

# 1. Summary statistics
print("=== Per-Pipeline Aggregated Metrics ===")
table = comparison.summary_table()
for name, metrics in table.items():
    print(f"{name}: best={metrics['best_fitness']:.4f}")

# 2. Plot all seeds
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, ax in enumerate(axes):
    comparison.plot_convergence(seed_index=i, ax=ax)
    ax.set_title(f"Seed {i + 1}")
plt.tight_layout()
plt.show()

# 3. Detailed per-seed analysis
for pipeline_name, result in comparison.pipelines.items():
    print(f"\n=== {pipeline_name} ===")
    for run in result.runs:
        print(f"  Seed {run.seed}: best={run.metrics['best_fitness']:.4f}, duration={run.duration_seconds:.2f}s")
```

---

## 8) Advanced Patterns

### Data Management in Composer

Composer abstracts data handling via two parallel mechanisms:

1. **`data_config` parameter in `quick_run()`** — Programmatic data passing
2. **`[data.*]` sections in TOML** — Declarative data specification

Both flow through the same **DataRegistry** (from `malthusjax.benchmarking.registry`) which resolves data sources and passes materialized data to evaluator factories.

#### Mechanism 1: Programmatic Data with `data_config`

**Use case**: Interactive exploration, custom Python workflo ws, dynamic data generation.

**Format**: `data_config` is a dict mapping `data_id` → `data_config_dict`:

```python
data_config = {
    "my_tsp_instance": {
        "source": "synthetic",
        "num_cities": 50,
        "random_seed": 42,
    },
    "knapsack_items": {
        "source": "synthetic",
        "num_items": 100,
        "capacity": 500,
    },
}

result = composer.quick_run(
    fitness="tsp:data_id=my_tsp_instance",
    data_config=data_config,
    selection="tournament:num_selections=25,tournament_size=3",
    mutation="swap:mutation_rate=0.1",
    pop_size=100,
    generations=200,
    seeds=(42, 43, 44),
    genome_type="categorical",
)
```

**Data sources**:
- `source="synthetic"` — Evaluator factory generates data on-the-fly
- `source="file"` — DataLoader reads from disk

#### Example 1: Quick TSP with Synthetic 50-City Problem

```python
composer = Composer.create_default()

# Quick exploration: randomly generated TSP
result = composer.quick_run(
    fitness="tsp:data_id=problem_a",
    data_config={
        "problem_a": {
            "source": "synthetic",
            "num_cities": 50,
            "random_seed": 42,
        },
    },
    selection="tournament:num_selections=25,tournament_size=3",
    mutation="swap:mutation_rate=0.1",
    pop_size=100,
    generations=200,
    seeds=(1, 2, 3),
    genome_type="categorical",
)

summary = result.aggregated_summary()
print(f"Best tour length: {summary['best_fitness']['mean']:.2f}")
```

#### Example 2: File-Based TSP (berlin52.tsp)

```python
# Load existing TSP benchmark
result = composer.quick_run(
    fitness="tsp:data_id=berlin",
    data_config={
        "berlin": {
            "source": "file",
            "path": "data/tsp/berlin52.tsp",
        },
    },
    selection="tournament:num_selections=25,tournament_size=3",
    mutation="swap:mutation_rate=0.1",
    pop_size=100,
    generations=300,
    seeds=(42, 43, 44),
    genome_type="categorical",
)

summary = result.aggregated_summary()
print(f"Best known distance: {summary['best_fitness']['mean']:.2f}")
```

#### Example 3: Knapsack with Custom Items

```python
# 0/1 Knapsack: 100 items, capacity 500
result = composer.quick_run(
    fitness="knapsack:data_id=problem_100",
    data_config={
        "problem_100": {
            "source": "synthetic",
            "num_items": 100,
            "capacity": 500,
            "weight_range": [10, 50],
            "value_range": [30, 100],
            "random_seed": 999,
        },
    },
    selection="roulette:num_selections=50",
    crossover="uniform_binary",
    mutation="bitflip:mutation_rate=0.05",
    pop_size=80,
    generations=150,
    seeds=(1, 2, 3),
    genome_type="binary",
)

summary = result.aggregated_summary()
print(f"Best value achieved: {summary['best_fitness']['mean']:.2f}")
```

#### Example 4: Multi-Instance Comparison

Compare algorithm performance across different problem sizes:

```python
comparison = composer.compare(
    pipelines={
        "on_20_cities": {
            "fitness": "tsp:data_id=problem_20",
        },
        "on_50_cities": {
            "fitness": "tsp:data_id=problem_50",
        },
        "on_100_cities": {
            "fitness": "tsp:data_id=problem_100",
        },
    },
    data_config={
        "problem_20": {
            "source": "synthetic",
            "num_cities": 20,
            "random_seed": 101,
        },
        "problem_50": {
            "source": "synthetic",
            "num_cities": 50,
            "random_seed": 102,
        },
        "problem_100": {
            "source": "synthetic",
            "num_cities": 100,
            "random_seed": 103,
        },
    },
    selection="tournament:num_selections=25,tournament_size=3",
    mutation="swap:mutation_rate=0.1",
    pop_size=100,
    generations=200,
    seeds=(42, 43, 44),
    genome_type="categorical",
)

table = comparison.summary_table()
for instance, metrics in table.items():
    print(f"{instance}: best={metrics['best_fitness']:.2f}")
```

#### Mechanism 2: Declarative Data with TOML `[data.*]` Sections

**Use case**: Reproducible experiments, version control, sharing configurations.

When using `Composer.from_toml()`, define data sources in `[data.*]` sections. The structure mirrors `data_config` dicts.

**See also**: [Section 6 — TOML Data Configuration](#data-configuration-data-sections) for complete TOML examples.

#### Quick-Run to TOML Equivalence

These are equivalent:

**Programmatic** (quick_run + data_config):
```python
result = composer.quick_run(
    fitness="tsp:data_id=my_problem",
    data_config={
        "my_problem": {
            "source": "synthetic",
            "num_cities": 50,
            "random_seed": 42,
        },
    },
    pop_size=100,
    generations=200,
    seeds=(1, 2, 3),
)
```

**Declarative** (from_toml):
```toml
[experiment.shared]
fitness = "tsp:data_id=my_problem"
pop_size = 100
generations = 200
seeds = [1, 2, 3]
genome_type = "categorical"

[data.my_problem]
source = "synthetic"
num_cities = 50
random_seed = 42

[pipelines.ga]
mutation = "swap:mutation_rate=0.1"
```

Then:
```python
result = Composer.from_toml("experiment.toml")
```

#### Data Registry Behind the Scenes

Both `quick_run(data_config=...)` and `from_toml()` internally create a **DataRegistry** that handles source resolution:

```python
from malthusjax.benchmarking.registry import DataRegistry

# Internal flow (you typically don't call this directly)
registry = DataRegistry()
for data_id, config in data_config.items():
    registry.register(data_id, config)

# When evaluator needs data:
data = registry.resolve("my_problem")
# If source="synthetic": returns config dict for factory
# If source="file": returns loaded array/dict from DataLoader
```

### Custom Fitness via data_config

For advanced use cases, pass custom data to fitness evaluators:

### Backend-Specific Configuration

#### MalthusJAX Features

- **Mutation schedules** — Decay mutation rate over generations
- **Elitism** — Automatically carry forward top-k individuals
- **Multi-device execution** — Via GSPMD sharding

#### Evosax Features

- **Strategy-specific parameters** — Passed via `**kwargs`
- **Available strategies** — SimpleGA, MR15_GA, DifferentialEvolution

### Extending Catalogs

Register custom operators:

```python
from malthusjax.composer.catalog import OperatorCatalog

catalog = OperatorCatalog()

# Register custom mutation
def my_custom_mutation(key, params, pop, bounds):
    # Custom mutation logic
    return new_pop

catalog.register("my_mutation", my_custom_mutation)

# Use in quick_run
result = composer.quick_run(
    mutation="my_mutation:param1=val1",
    ...
)
```

---

## 9) Configuration Reference

### quick_run() Parameter Defaults

```python
composer.quick_run(
    fitness=None,
    selection=None,
    crossover=None,
    mutation=None,
    seeds=(1, 2, 3),
    generations=100,
    pop_size=50,
    genome_type="real",
    genome_length=10,
    bounds=(-5.0, 5.0),
    maximize=False,
    backend="malthusjax",
    evosax_strategy="SimpleGA",
    engine_type="ga",
    elitism=2,
    experiment_name="quick_experiment",
    output_dir=None,  # defaults to results/{experiment_name}
    engine=None,
    prng_impl=None,
    trace_dir=None,
    data_config=None,
)
```

### from_toml() Parameters

```python
Composer.from_toml(
    path="experiment.toml",
    pipelines=None,  # if None, execute all pipelines
    shared_initial_population=True,
    pop_seed=123,
    trace_dir=None,
)
```

### compare() Parameters

```python
composer.compare(
    pipelines={
        "Name A": {...},
        "Name B": {...},
    },
    seeds=(42, 43, 44),
    shared_initial_population=True,
    pop_seed=123,
    **shared_kwargs,  # e.g., fitness, pop_size, generations, bounds
)
```

---

## Troubleshooting

### Invalid Operator Spec

**Error**: `KeyError: "sphere_2"`

**Solution**: Check operator spelling and parameters. Valid format:
```
"operator_name:param1=val1,param2=val2"
```

### Missing Required Parameter

**Error**: `KeyError: "fitness"` or similar

**Solution**: Ensure all required parameters are specified in `quick_run()`, TOML `[experiment.shared]`, or pipeline overrides.

### Type Mismatches

**Error**: `TypeError: expected float, got str`

**Solution**: TOML and string specs use specific type conversions. Ensure:
- Numbers (dim, eta, alpha) are unquoted: `dim=10` not `dim="10"`
- Bounds are lists: `bounds = [-5.0, 5.0]` not `bounds = "(-5.0, 5.0)"`
- Seeds are lists: `seeds = [42, 43, 44]`

---

## See Also

- [Engine Documentation](../engine/README.md) — GeneticEngine, ResourceMap, key derivation strategies
- [Operators Documentation](../operators/README.md) — Selection, crossover, mutation details
- [Genome Documentation](../core/genome/README.md) — RealGenome, BinaryGenome, CategoricalGenome
- [Fitness Evaluators Documentation](../core/fitness/README.md) — Available evaluators and custom evaluator patterns

