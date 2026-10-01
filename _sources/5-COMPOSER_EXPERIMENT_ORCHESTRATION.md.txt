# Experiment Orchestration via Composer

The `malthusjax.composer` module provides high-level experiment orchestration. It abstracts low-level engine construction, operator composition, and seed loop iteration into declarative configuration files (TOML), strongly typed Python configuration trees (`ExperimentConfig`), or interactive programmatic entry points (`quick_run()`, `compare()`).

Following the design principles of production-grade frameworks like **Hydra** and **Kedro**, the Composer decouples configuration, instantiation, and execution into three distinct layers:

1. **Configuration Layer**: Immutable typed config dataclasses (`ExperimentConfig`) and TOML loaders.
2. **Instantiation Boundary**: `EngineFactory` and `BackendRegistry`, resolving components via dependency injection.
3. **Execution Orchestration**: Formal `Engine` protocol executed across seeds via `BenchmarkRunner`.

---

## 5.1. Three Ways to Configure an Experiment

MalthusJAX offers three entry points depending on your workflow:

### 5.1.1. Method A: Typed Python Configuration (Recommended)
Using immutable dataclasses provides full IDE autocompletion, type checking via `mypy`, and runtime validation:

```python
from malthusjax.composer.experiment_config import (
    ExperimentConfig, PopulationConfig, ExecutionConfig, MalthusJAXBackendConfig
)
from malthusjax.composer.engine_factory import EngineFactory
from malthusjax.benchmarking.runner import BenchmarkRunner

# 1. Define immutable configuration tree
config = ExperimentConfig(
    population=PopulationConfig(
        size=64,
        genome_length=10,
        bounds=(-5.0, 5.0),
    ),
    execution=ExecutionConfig(
        generations=100,
        seeds=(42, 43, 44),
    ),
    backend=MalthusJAXBackendConfig(
        fitness="sphere:dim=10",
        selection="tournament:tournament_size=3",
        crossover="blend:alpha=0.5",
        mutation="gaussian:mutation_rate=0.1",
        elitism=2,
    ),
)

# 2. Validate configuration
config.validate()

# 3. Instantiate via EngineFactory
engine = EngineFactory.build(config)

# 4. Execute across seeds
runner = BenchmarkRunner(engine=engine, experiment_name=config.output.experiment_name)
result = runner.run(seeds=config.execution.seeds)
print(result.aggregated_summary())
```

### 5.1.2. Method B: Declarative TOML Files
For reproducible research, automated benchmark sweeps, and cluster submission:

```python
from malthusjax.composer import Composer

composer = Composer.create_default()
comparison = composer.from_toml("configs/benchmark.toml")

print(comparison.summary_table())
comparison.plot_convergence()
```

### 5.1.3. Method C: Interactive Prototyping (`quick_run`)
For fast, interactive exploratory data analysis in Jupyter notebooks:

```python
from malthusjax.composer import Composer

composer = Composer.create_default()
result = composer.quick_run(
    fitness="sphere:dim=10",
    selection="tournament:tournament_size=3",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1",
    backend="malthusjax",
    pop_size=64,
    generations=100,
    seeds=[42, 43, 44],
)

print(result.aggregated_summary())
```

---

## 5.2. The Engine Factory Boundary (`engine_factory.py`)

The `EngineFactory` acts as the single boundary separating configuration from instantiation.

```python
from malthusjax.composer.engine_factory import EngineFactory

# Build engine directly from config
engine = EngineFactory.build(config)

# Or inject a pre-resolved custom evaluator callable
engine = EngineFactory.build_with_evaluator(config, evaluator=custom_fitness_fn)
```

### How `EngineFactory` Works
1. **Validation**: Calls `config.validate()`, verifying bounds, non-empty seeds, and positive generation/population counts.
2. **Backend Provider Lookup**: Queries `BackendRegistry.get_backend(config.backend.backend_name)`.
3. **Evaluator Resolution**: Delegates fitness resolution to `provider.resolve_evaluator(spec, **kwargs)`.
4. **Strategy Resolution**: Resolves algorithm hyperparameters via `provider.default_strategy(**kwargs)`.
5. **Engine Construction**: Calls `provider.build_engine(strategy, evaluator, **kwargs)` returning an object conforming to the `Engine` protocol.

---

## 5.3. The `Engine` Protocol & Runtime Verification

All engines in MalthusJAX—whether native 5-phase genetic algorithms, Quality-Diversity engines, or adapted third-party libraries—implement the runtime-checkable `Engine` protocol:

```python
import chex
from typing import Protocol, Dict, Any, runtime_checkable

@runtime_checkable
class Engine(Protocol):
    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        """Execute a single run and return history, summary, and timings."""
        ...
```

The returned dictionary is strictly structured:
- `"history"`: List of per-generation metric dictionaries (`generation`, `best_fitness`, etc.).
- `"summary"`: Final metrics dictionary (`best_fitness`, `final_generation`, `total_evaluations`).
- `"timings"`: Execution duration dictionary (`{"total": elapsed_seconds, "warmup": ..., "execution": ...}`).

---

## 5.4. Declarative TOML Schema

The `load_experiment_config(path)` function parses declarative TOML files into structured dictionary representations, seamlessly converted into `ExperimentConfig`:

### Verified TOML Structure
```toml
[experiment]
name       = "crossover_comparison"
output_dir = "results/crossover_comparison"

[logging]
level        = "INFO"
interval     = 25
nan_watchdog = true

[experiment.shared]
fitness       = "sphere:dim=10"
pop_size      = 50
generations   = 100
genome_length = 10
bounds        = [-5.0, 5.0]
seeds         = [42, 43, 44]
prng_impl     = "threefry2x32"
elitism       = 2
maximize      = false

[pipelines.blend_ga]
backend   = "malthusjax"
engine_type = "ga"
selection = "tournament:num_selections=25,tournament_size=3"
crossover = "blend:alpha=0.5"
mutation  = "gaussian:mutation_rate=0.1,mutation_strength=0.1"

[pipelines.sbx_ga]
backend   = "malthusjax"
engine_type = "ga"
selection = "tournament:num_selections=25,tournament_size=3"
crossover = "simulated_binary:eta=20.0"
mutation  = "polynomial:mutation_rate=0.1,eta=20.0"

[pipelines.evosax_cma]
backend       = "evosax"
strategy_name = "CMA_ES"
```

---

## 5.5. Programmatic Comparison (`compare`)

`Composer.compare(pipelines={...}, fitness=..., pop_size=..., seeds=...)` runs multiple pipeline parameter dictionaries side-by-side using aligned initial populations and seeds.

```python
from malthusjax.composer import Composer

composer = Composer.create_default()

comparison = composer.compare(
    pipelines={
        "blend_ga": {
            "backend": "malthusjax",
            "crossover": "blend:alpha=0.5",
            "mutation": "gaussian:mutation_rate=0.1",
        },
        "sbx_ga": {
            "backend": "malthusjax",
            "crossover": "simulated_binary:eta=20.0",
            "mutation": "polynomial:mutation_rate=0.1",
        },
        "evosax_cma": {
            "backend": "evosax",
            "strategy_name": "CMA_ES",
        },
    },
    fitness="sphere:dim=10",
    pop_size=64,
    generations=100,
    seeds=[42, 43, 44],
)

# Export statistical summary table
print(comparison.summary_table())

# Generate publication-ready convergence plots
comparison.plot_convergence()
```

---

## 5.6. Catalog & String Specification Resolution

MalthusJAX maps human-readable string specifications to instantiated PyTree objects using regex key-value parsing (`key1=val1,key2=val2`).

### 5.6.1. `OperatorCatalog` (`catalog.py`)
Resolves operator string specs:
- Format: `"operator_name:param1=value1,param2=value2"`
- Parameter coercion: Automatically converts numeric strings to `int` or `float` and booleans to `bool`.
- API methods: `get_selection(spec)`, `get_crossover(spec)`, `get_mutation(spec)`.

### 5.6.2. Custom Component Decorators (`decorators.py`)
Users can extend the catalog dynamically using registry decorators:
- `@register_selection(name="...")`
- `@register_crossover(name="...")`
- `@register_mutation(name="...")`
- `@register_fitness(name="...")`
- `@register_backend(name="...")`

---

## 5.7. External Framework Adapters

Composer provides universal adapter builders to run external libraries under the unified `Engine` protocol interface:

- **EvoSAX Adapter (`evosax_adapter.py`)**: Wraps EvoSAX strategies (e.g. `SimpleGA`, `CMA_ES`, `DifferentialEvolution`, `OpenES`) into `UniversalAdapterEngine`.
- **QDAX Adapter (`qdax_adapter.py`)**: Wraps QDAX emitters, repertoires, and metrics into `UniversalAdapterEngine`.
- **TensorNEAT Adapter (`tensorneat_adapter.py`)**: Wraps TensorNEAT algorithm and problem instances into `UniversalAdapterEngine`.
- **Kozax Adapter (`kozax_adapter.py`)**: Wraps Kozax Genetic Programming strategies into `UniversalAdapterEngine`.

All adapted engines share identical JIT warmup isolation, Wall-Clock timing boundaries, and PRNG key lineage.

---

## 5.8. Showcase Demonstration Suite

The complete benchmark and parity verification scripts are located in [`examples/showcase/`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/examples/showcase):

- [`showcase_cross_backend_compare.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/examples/showcase/showcase_cross_backend_compare.py): Multi-pipeline comparison CLI supporting GA ablation (Demo 1), cross-framework battle (Demo 2), Quality Diversity parity (Demo 3 with `--seeds 30` statistical testing), BBOB hard landscapes (Demo 5), and Level 3 Engine parity (Demo 7).
- [`showcase_engine_composer_parity.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/examples/showcase_engine_composer_parity.py): Demonstrates 1:1 bit-exact parity between raw Level 3 `GeneticEngine` and Level 4 `Composer.quick_run()`, as well as runtime registration of custom engines (`@register_engine`, `@register_genome`).

