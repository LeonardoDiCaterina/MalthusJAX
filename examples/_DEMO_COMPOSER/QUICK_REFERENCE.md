# Quick Reference: Composer Compare API

## Installation & Setup
```python
from malthusjax.composer import Composer, EngineRegistry
from malthusjax.benchmarking import ComparisonResult
```

## Two Ways to Compare Algorithms

### Option 1: Programmatic (Python Code)
```python
composer = Composer.create_default()

result = composer.compare(
    pipelines={
        "algo1": {"crossover": "blend:alpha=0.5"},
        "algo2": {"crossover": "simulated_binary:eta=2.0"},
    },
    seeds=(42, 43, 44),                    # Run each pipeline 3 times
    shared_initial_population=True,         # Same starting pop for fair comparison
    pop_seed=42,
    fitness="sphere:dim=10",
    pop_size=50,
    generations=100,
)
```

### Option 2: Declarative (TOML Config)
```python
# experiment.toml (see examples/_DEMO_COMPOSER/experiment.toml)
[experiment.shared]
fitness = "sphere:dim=10"
pop_size = 50
generations = 100
seeds = [42, 43, 44]

[pipelines.algo1]
crossover = "blend:alpha=0.5"

[pipelines.algo2]
crossover = "simulated_binary:eta=2.0"

# Load and run:
result = Composer.from_toml("experiment.toml")
```

## Engine Registry

MalthusJAX now supports an **EngineRegistry** — the engine counterpart of
`OperatorCatalog`.  Engines register themselves at import time and can be
resolved from string specs, just like operators.

### Listing Available Engines
```python
from malthusjax.composer import EngineRegistry

registry = EngineRegistry()
print(registry.list_available())   # ['ga']
print(registry.get_help("ga"))     # Docstring + defaults
```

### Using engine_type in quick_run()
```python
# Default (equivalent to engine_type="ga")
result = composer.quick_run(
    fitness="sphere:dim=10",
    selection="tournament:num_selections=25",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1",
)

# Explicit engine_type with spec overrides
result = composer.quick_run(
    engine_type="ga:elitism=4",
    fitness="sphere:dim=10",
    selection="tournament:num_selections=25",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1",
)
```

### Registering a Custom Engine
```python
from malthusjax.composer import EngineRegistry

def my_engine_factory(evaluator, selection, crossover, mutation, **kwargs):
    """Custom engine that wraps GeneticEngine with extra logic."""
    from malthusjax.composer.engine_factory import build_engine
    return build_engine(
        fitness_evaluator=evaluator,
        selection_op=selection,
        crossover_op=crossover,
        mutation_op=mutation,
        **kwargs,
    )

registry = EngineRegistry()
registry.register("my_engine", my_engine_factory, {"pop_size": 200})

# Now usable everywhere
result = composer.quick_run(engine_type="my_engine", ...)
```

### Using EngineRegistry Directly
```python
from malthusjax.composer import EngineRegistry
from malthusjax.composer.catalog import OperatorCatalog

catalog = OperatorCatalog()
registry = EngineRegistry()

engine = registry.get(
    "ga:pop_size=100,elitism=4",
    evaluator=catalog.get("sphere:dim=10"),
    selection=catalog.get("tournament:num_selections=50"),
    crossover=catalog.get("blend:alpha=0.5"),
    mutation=catalog.get("gaussian:mutation_rate=0.1"),
    generations=200,
    genome_length=10,
)

result = engine.run_once(jax.random.PRNGKey(42))
```

## Analyzing Results

```python
# Pipeline names
print(result.names)  # ['algo1', 'algo2']

# Final fitness statistics
summary = result.summary_table()
for pipeline, metrics in summary.items():
    print(f"{pipeline}: {metrics['best_fitness_mean']:.6f}")

# Convergence history for seed 0
conv_data = result.convergence_data(seed_index=0)
for pipeline_name, fitness_history in conv_data.items():
    print(f"{pipeline_name}: {fitness_history[-1]:.6f}")

# Plot convergence
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
result.plot_convergence(seed_index=0, ax=ax, title="Convergence")
plt.show()
```

## Shared Initial Population

When `shared_initial_population=True`:
- All pipelines start from **identical** random population
- Eliminates initialization variance
- Ensures fair cross-backend comparison
- Uses `jax.random.uniform()` with seed `pop_seed`
- Works across malthusjax, evosax, and stub backends

## Configuration Merging

```toml
# [experiment.shared] is base config
fitness = "sphere:dim=10"
pop_size = 50

# [pipelines.X] overrides only specific keys
[pipelines.algo1]
crossover = "blend:alpha=0.5"   # Only this changes
# pop_size still = 50, fitness still = "sphere:dim=10"

[pipelines.algo2]
crossover = "simulated_binary"
mutation = "polynomial:eta=15"  # Multiple overrides OK
```

## Return Type: ComparisonResult

```python
result: ComparisonResult = composer.compare(...)

# Attributes:
result.names                    # List[str] of pipeline names
result.pipelines               # Dict[str, ExperimentResult]
result.shared_config           # Dict[str, Any]
result.initial_population      # Optional jax array

# Methods:
result.summary_table()         # Dict → summary statistics
result.convergence_data(i)     # Dict → fitness histories
result.plot_convergence(i)     # Plots convergence curves
```

## Common Use Cases

### Compare Crossover Operators
```python
result = composer.compare(
    pipelines={
        "uniform": {"crossover": "uniform_real:rate=0.7"},
        "blend": {"crossover": "blend:alpha=0.5"},
        "sbx": {"crossover": "simulated_binary:eta=20"},
    },
    seeds=(42, 43, 44),
    shared_initial_population=True,
    fitness="sphere:dim=10",
    pop_size=100,
    generations=200,
)
```

### Compare Backends (MalthusJAX vs Evosax)
```python
result = composer.compare(
    pipelines={
        "malthusjax_ga": {
            "backend": "malthusjax",
            "crossover": "blend:alpha=0.5",
        },
        "evosax_simplega": {
            "backend": "evosax",
            "evosax_strategy": "SimpleGA",
        },
    },
    seeds=(42, 43, 44),
    shared_initial_population=True,
    fitness="sphere:dim=10",
    pop_size=100,
    generations=200,
)
```

### Compare Engine Types (when multiple engines are registered)
```python
result = composer.compare(
    pipelines={
        "standard_ga": {"engine_type": "ga"},
        "custom_ga": {"engine_type": "ga:elitism=10"},
    },
    seeds=(42, 43, 44),
    shared_initial_population=True,
    fitness="sphere:dim=10",
    selection="tournament:num_selections=50",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1",
    pop_size=100,
    generations=200,
)
```

### Load from TOML with Pipeline Filtering
```python
# Only run specific pipelines from TOML
result = Composer.from_toml(
    "experiment.toml",
    pipelines=["algo1", "algo3"],  # Skip algo2
)
```

## File Locations

- Example config: [examples/_DEMO_COMPOSER/experiment.toml](examples/_DEMO_COMPOSER/experiment.toml)
- Example notebook: [examples/_DEMO_COMPOSER/Composer_Compare_API_Demo.ipynb](examples/_DEMO_COMPOSER/Composer_Compare_API_Demo.ipynb)
- Composer implementation: [src/malthusjax/composer/composer.py](src/malthusjax/composer/composer.py)
- Engine registry: [src/malthusjax/composer/engine_registry.py](src/malthusjax/composer/engine_registry.py)
- Engine catalog: [src/malthusjax/composer/engine_catalog.py](src/malthusjax/composer/engine_catalog.py)
- Operator catalog: [src/malthusjax/composer/catalog.py](src/malthusjax/composer/catalog.py)
- Engine registration: [src/malthusjax/engine/__init__.py](src/malthusjax/engine/__init__.py)
- Documentation: [docs/COMPOSER_CONFIG_DRIVEN_RUNS.md](docs/COMPOSER_CONFIG_DRIVEN_RUNS.md)

## Tips & Tricks

1. **Reproducibility**: Use `pop_seed=42` and `seeds=(42, 43, 44)` for deterministic runs
2. **Fair comparison**: Always use `shared_initial_population=True` when comparing algorithms
3. **Multiple backends**: Mix malthusjax operators with evosax strategies in same comparison
4. **Large experiments**: Use TOML for long experiment definitions, Python for quick prototyping
5. **Visualization**: `negate=True` for maximization problems to flip convergence curves upward
