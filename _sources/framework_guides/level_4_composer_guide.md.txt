# Level 4: The Composer 🎼

The **Composer** (`src/malthusjax/composer/`) is the fourth and highest abstraction layer in the MalthusJAX framework.

While Level 3 allows you to build and control a JAX Engine manually in Python, Level 4 is an orchestration framework that allows you to define an *entire experiment* via typed Python dataclasses or declarative `.toml` files and run it across multiple random seeds without writing execution boilerplate.

Following the design patterns of production-grade frameworks like **Hydra**, **Kedro**, and **PyTorch Lightning**, Level 4 decouples configuration, instantiation, and execution into three distinct layers.

---

## 1. The 3-Layer Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. CONFIGURATION LAYER (Declarative, Type-Safe, Immutable)             │
│    ExperimentConfig (PopulationConfig, ExecutionConfig, BackendConfig) │
│    Validation, to_dict(), from_dict(), from_toml()                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. INSTANTIATION / FACTORY BOUNDARY (Dependency Injection)             │
│    EngineFactory.build(config: ExperimentConfig) -> Engine             │
│    BackendRegistry & BackendProvider Contract Validation               │
│    Centralized Initial Population Sampler (_population_init.py)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. EXECUTION ORCHESTRATION (Hardware-Accelerated Execution)            │
│    @runtime_checkable class Engine(Protocol): run_once(key)            │
│    BenchmarkRunner.run(engine, seeds) -> ExperimentResult              │
│    JIT Warmup Separation & Zero-Overhead Telemetry Watchdogs           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1: Typed Configuration Model (`experiment_config.py`)

All experiment configurations are expressed as immutable, frozen dataclasses:

```python
from malthusjax.composer.experiment_config import (
    ExperimentConfig,
    PopulationConfig,
    ExecutionConfig,
    MalthusJAXBackendConfig,
    LoggingConfig,
    OutputConfig,
)

config = ExperimentConfig(
    population=PopulationConfig(
        size=64,
        genome_type="real",
        genome_length=10,
        bounds=(-5.0, 5.0),
    ),
    execution=ExecutionConfig(
        generations=100,
        seeds=(42, 43, 44),
        maximize=False,
    ),
    backend=MalthusJAXBackendConfig(
        fitness="sphere:dim=10",
        selection="tournament:tournament_size=3",
        crossover="blend:alpha=0.5",
        mutation="gaussian:mutation_rate=0.1",
        elitism=2,
    ),
    logging=LoggingConfig(log_interval=10, log_nan_watchdog=True),
    output=OutputConfig(experiment_name="l4_guide_experiment"),
)

# Semantic validation
config.validate()
```

### Key Advantages
- **Immutability**: Frozen dataclasses prevent accidental state mutations during execution sweeps.
- **Fail-Fast Validation**: `.validate()` catches negative populations, invalid genome lengths, or inverted bounds *before* triggering heavy JAX compilation.
- **Bi-directional Serialization**: Clean `.to_dict()` and `.from_dict()` support for cloud orchestration.

---

## 3. Layer 2: Instantiation Boundary (`engine_factory.py`)

The `EngineFactory` acts as the single dependency-injection boundary. It takes an `ExperimentConfig` and resolves it into an `Engine`:

```python
from malthusjax.composer.engine_factory import EngineFactory

# Build engine directly
engine = EngineFactory.build(config)
```

### Instantiation Pipeline
1. **Config Validation**: Calls `config.validate()`.
2. **Provider Dispatch**: Looks up `config.backend.backend_name` in the centralized `BackendRegistry`.
3. **Evaluator Resolution**: Calls `provider.resolve_evaluator(spec, **kwargs)`.
4. **Strategy Formulation**: Calls `provider.default_strategy(**kwargs)`.
5. **Engine Assembly**: Calls `provider.build_engine(strategy, evaluator, **kwargs) -> Engine`.

### Centralized Population Initialization (`_population_init.py`)
To ensure strict statistical fairness when comparing native MalthusJAX algorithms against third-party libraries (e.g. EvoSAX, QDAX), initial population generation is centralized in `_population_init.py`.

Whether running native GA or EvoSAX CMA-ES, both engines receive identical candidate distributions generated from the same PRNG seeds:
- `generate_initial_population()`: Generates uniform random candidates or BBOB-calibrated candidate distributions.

---

## 4. Layer 3: Execution Protocol (`engine_protocol.py`)

All engines conform to the formal, runtime-checkable `Engine` protocol:

```python
import chex
from typing import Protocol, Dict, Any, runtime_checkable

@runtime_checkable
class Engine(Protocol):
    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        """Execute a single run and return history, summary, and timings."""
        ...
```

### Verification & Testing
Because the protocol is decorated with `@runtime_checkable`, you can verify compliance at runtime:

```python
from malthusjax.composer.engine_protocol import Engine

assert isinstance(engine, Engine)
```

---

## 5. The Benchmark Runner (`BenchmarkRunner`)

Once constructed by `EngineFactory`, the engine is executed by `BenchmarkRunner`:

- **JIT Compilation Isolation**: Enforces strict `jax.block_until_ready()` barriers, separating JIT compilation warmup (`t_warmup`) from device execution (`t_exec`).
- **Data Collection**: Collects generation histories and scalar KPIs.
- **Statistical Aggregation**: Aggregates means, standard deviations, and confidence intervals across seeds (`ExperimentResult.aggregated_summary()`).

---

## 6. Declarative TOML Benchmarks (`Composer.from_toml`)

For reproducible research and version control, experiments can be defined in `.toml` format:

```toml
[experiment]
name = "benchmark_study"
output_dir = "results/study"

[logging]
level = "INFO"
interval = 20

[experiment.shared]
fitness = "sphere:dim=10"
pop_size = 64
generations = 100
genome_length = 10
bounds = [-5.0, 5.0]
seeds = [42, 43, 44]

[pipelines.native_blend]
backend = "malthusjax"
crossover = "blend:alpha=0.5"
mutation = "gaussian:mutation_rate=0.1"

[pipelines.evosax_cma]
backend = "evosax"
strategy_name = "CMA_ES"
```

```python
from malthusjax.composer import Composer

composer = Composer.create_default()
results = composer.from_toml("benchmark_study.toml")

print(results.summary_table())
results.plot_convergence()
```

---

## 7. Unified Logging & Step Telemetry 🪵

Level 4 exposes zero-overhead diagnostic and step telemetry logging across both native engines and external library adapters:

- **Programmatic Control**: Pass `log_interval` (generation callback cadence), `log_level`, and `log_nan_watchdog` (on-device NaN/Inf detection).
- **Trace-Time Pruning**: When `log_interval` is `None`, Python trace-time branching removes all callback nodes from the compiled XLA graph, guaranteeing 0 ns overhead during production benchmarks.

---

## 8. Showcase Benchmarks & Validation Suite 🏆

The repository provides an end-to-end multi-algorithm benchmark and parity verification suite in [`examples/showcase/showcase_cross_backend_compare.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/examples/showcase/showcase_cross_backend_compare.py):

```bash
# Run any individual demo with plotting:
python examples/showcase/showcase_cross_backend_compare.py --demo 1 --plot
python examples/showcase/showcase_cross_backend_compare.py --demo 2 --plot
python examples/showcase/showcase_cross_backend_compare.py --demo 3 --plot --seeds 30
python examples/showcase/showcase_cross_backend_compare.py --demo 5 --plot
python examples/showcase/showcase_cross_backend_compare.py --demo 7
```

### Highlights:

- **Showcase 1 (GA Operator Ablation)**: Compares recombination and mutation operators (Blend, Uniform, Simulated Binary Crossover, Polynomial Mutation) on multimodal 10D Rastrigin.
- **Showcase 2 (Cross-Framework Battle)**: Simultaneously benchmarks MalthusJAX GA, EvoSAX CMA-ES, Sep-CMA-ES, Open-ES, and Native MAP-Elites on 10D Sphere.
- **Showcase 3 (Quality-Diversity Validation)**: Benchmarks Native MalthusJAX MAP-Elites against upstream QDAX MAP-Elites:
  - **Identical Start**: Both algorithms evaluate the identical shared initial population at Gen 0 ($-90.058$).
  - **Parity**: Achieves matching final coverage ($82.8125\%$) and proves statistical equivalence across 30 seeds ($p = 0.3704$ on paired $t$-test; $p = 0.3492$ on Wilcoxon signed-rank).
  - **Performance**: Native MalthusJAX delivers a **2.5x runtime speedup** over QDAX.
- **Showcase 5 (BBOB Landscapes)**: Evaluates optimization trajectories on Rosenbrock valleys (f8) and multimodal Rastrigin (f15).
- **Showcase 7 (Engine Parity Reproduction)**: Validates bit-exact consistency between raw Level 3 `GeneticEngine` and Level 4 `Composer.quick_run()`, plus registration of custom subclassed engines (`DiversityTrackingEngine`).

