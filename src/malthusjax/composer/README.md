# `malthusjax.composer` — Technical Reference

Scope: `malthusjax.composer.composer`, `malthusjax.composer.catalog`, `malthusjax.composer.config`, `malthusjax.composer.strategies`, `malthusjax.composer.evaluator_parser`, `malthusjax.composer.engine_catalog`, `malthusjax.composer.engine_factory`, `malthusjax.composer.genome_catalog`, `malthusjax.composer.decorators`, `malthusjax.composer.adapters`, `malthusjax.composer.composable_evosax_adapter`, `malthusjax.composer.composable_tensor_neat_adapter`, `malthusjax.composer.evosax_adapter`, `malthusjax.composer.qdax_adapter`, `malthusjax.composer.tensorneat_adapter`, `malthusjax.composer.kozax_adapter`, `malthusjax.composer.pipeline`.

---

## 1. Overview & Architecture

The `malthusjax.composer` package provides the top-level declarative and programmatic experiment orchestration layer of **MalthusJAX**. It bridges low-level core genomes, operators, composable evaluators, and evolutionary engines with high-level multi-seed experiment workflows.

```
                  +----------------------------------------------+
                  |              Composer Entry Point            |
                  |   quick_run()  |  from_toml()  |  compare()  |
                  +----------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
+-------------------+         +---------------------+         +---------------------+
| OperatorCatalog   |         | Strategies Layer    |         | EvaluatorParser     |
| - String DSL      |         | - GeneticStrategy   |         | - 4-Axis Parsing    |
| - Param Coercion  |         | - MapElitesStrategy |         | - Dim Auto-wiring   |
| - Type Validation |         | - EvoSAXStrategy    |         | - Composable Envs   |
| - Global Registry |         | - QDAXStrategy      |         |   & Interpreters    |
+-------------------+         | - TensorNEATStrategy|         +---------------------+
         |                    +---------------------+                    |
         +-------------------------------+-------------------------------+
                                         |
                  +----------------------------------------------+
                  |             Universal Engine Layer           |
                  |  - Native Engines: GeneticEngine (5-phase),  |
                  |    MOEngine (NSGA-II), MapElitesEngine (QD)  |
                  |  - Universal Framework Adapters:             |
                  |    EvoSAX, QDAX, TensorNEAT, Kozax           |
                  +----------------------------------------------+
                                         |
                  +----------------------------------------------+
                  |               BenchmarkRunner                |
                  |   Shared PRNG Seeds | Multi-Run Aggregation  |
                  |   ExperimentResult  | ComparisonResult       |
                  +----------------------------------------------+
```

### Key Capabilities
1. **String DSL & Operator Specs**: Parse compact configuration strings (`"operator:param1=val1,param2=val2"`) into validated, JIT-ready dataclass instances.
2. **Algorithmic Strategies (`strategies/`)**: Declarative specifications binding evolutionary engines with operators, emitters, and hyperparameters.
3. **Dynamic Composable Evaluator Parser (`evaluator_parser.py`)**: Instantiate full 4-axis evaluators (`OptimizationEvaluator`, `SupervisedEvaluator`, `RLEvaluator`) from hierarchical dictionaries or TOML blocks with automatic dimension propagation.
4. **Data-Driven Evaluation (Option C)**: Inject problem datasets or synthetic problem generators via `[data.<data_id>]` sections into problem environments (e.g. TSP, Knapsack).
5. **Universal Framework Adapters**: Wrap external evolutionary algorithms (EvoSAX, QDAX, TensorNEAT, Kozax) under a single uniform `Engine` protocol interface (`run_once(key) -> Dict[str, Any]`).
6. **Statistically Fair Benchmarking**: Share exact PRNG seeds, initial populations, and evaluation environments across competing pipelines.

---

## 2. `malthusjax.composer.composer`

The `Composer` class is the central orchestrator, exposing three primary execution workflows:

### `quick_run(...) -> ExperimentResult`
Interactive execution method for running single pipeline sweeps across multiple random seeds.
```python
from malthusjax.composer import Composer

composer = Composer()
result = composer.quick_run(
    fitness="sphere:dim=10,maximize=false",
    selection="tournament:num_selections=50,tournament_size=3",
    crossover="uniform_real:crossover_rate=0.9",
    mutation="gaussian:mutation_rate=0.1,mutation_strength=0.05",
    pop_size=100,
    generations=100,
    seeds=5,
    backend="malthusjax",
)
summary = result.aggregated_summary()
print("Best fitness mean:", summary["best_fitness"]["mean"])
```
- **Backend Selection**: `"malthusjax"` (default), `"evosax"`, `"qdax"`, `"tensorneat"`, `"kozax"`.
- **Data Configuration**: Accepts `data_config={...}` to resolve problem-specific `data_id` references.
- **Seed Handling**: Normalizes integer seed counts or explicit seed sequences via `_normalize_seeds()`.

### `from_toml(path, ...) -> ComparisonResult`
Declarative entry point for loading experiment TOML files via `load_experiment_config`.
```python
comparison = composer.from_toml("configs/examples/tsp_tour_optimization.toml")
table = comparison.summary_table()
print(table)
```
- Parses shared baseline defaults (`[experiment.shared]`) and per-pipeline overrides (`[pipelines.*]`).
- Supports data-driven registries (`[data.*]`).
- Executes all pipelines in sequence across specified random seeds.

### `compare(pipelines, ...) -> ComparisonResult`
Programmatic multi-pipeline benchmarking entry point.
- Accepts a dictionary mapping pipeline names to `quick_run` parameter keyword dictionaries.
- Enforces identical random seed initialization across pipelines for fair statistical comparison.

---

## 3. Algorithmic Strategies (`malthusjax.composer.strategies`)

The `strategies` module provides declarative dataclasses that specify engine topologies and operator configurations:

| Strategy | Description | Key Fields | Target Engine |
| :--- | :--- | :--- | :--- |
| `GeneticStrategy` | Standard 5-phase genetic algorithm | `selection`, `crossover`, `mutation` | `GeneticEngine` (`GeneticFastEngine`) |
| `MapElitesStrategy` | Native Quality-Diversity archive search | `emitter`, `num_descriptors`, `num_centroids`, `mutation_sigma`, `key_derivation`, `maximize` | `MapElitesEngine` |
| `EvoSAXStrategy` | External EvoSAX evolutionary strategy | `algorithm_name`, `algorithm_kwargs` | `UniversalAdapterEngine` (`evosax`) |
| `QDAXStrategy` | External QDAX Quality-Diversity algorithm | `strategy_cls`, `emitter`, `num_descriptors`, `num_centroids`, `metrics_function` | `UniversalAdapterEngine` (`qdax`) |
| `TensorNEATStrategy` | External TensorNEAT neuroevolution algorithm | `algorithm_name`, `genome_name`, `problem_name`, `num_inputs`, `num_outputs` | `UniversalAdapterEngine` (`tensorneat`) |

When multiple operators are passed to `GeneticStrategy` or `MapElitesStrategy`, they are automatically composed via a `MixingEmitter` or executed in the 5-phase engine sequence.

---

## 4. Dynamic Composable Evaluator Parser (`evaluator_parser.py`)

The `evaluator_parser` module translates nested dictionary configurations (e.g. from TOML) into strongly typed, JIT-compatible Composable Evaluators:

```python
from malthusjax.composer.evaluator_parser import parse_evaluator

config = {
    "type": "OptimizationEvaluator",
    "env": {"type": "SphereEnv", "dim": 20},
    "interpreter": {"type": "IdentityInterpreter"},
    "output": {"type": "ScalarOutput", "maximize": False},
    "transform": {"type": "IdentityTransform"},
}

evaluator = parse_evaluator(config)
```

### Automatic Wiring & Special Features
1. **Dimension Auto-Wiring**: If `env` exposes `obs_dim` or `action_dim` (e.g. `BraxEnv`, `GymnaxEnv`), `parse_evaluator` automatically binds them to `interpreter.input_dim` and `interpreter.output_dim` if omitted.
2. **Sensible Defaults**:
   - `interpreter`: Defaults to `IdentityInterpreter`.
   - `output`: Defaults to `ScalarOutput(maximize=False)`.
   - `transform`: Defaults to `IdentityTransform`.
3. **TensorNEAT Integration**: Resolves `type = "TensorNEATProblemWrapper"` by dynamically instantiating problems from `tensorneat.problem` (e.g. `XOR`, `GymProblem`).

---

## 5. `malthusjax.composer.catalog` & Global Registries

### `OperatorCatalog` (`catalog.py`)
Parses and resolves operator string specifications:
- **DSL Syntax**: `"operator_name:param1=val1,param2=val2"`.
- **Automatic Type Coercion**: Converts numerical strings to `int` / `float`, `"true"` / `"false"` to `bool`.
- **Parameter Validation**: Reflects over factory signatures; raises descriptive errors on unrecognized parameters.
- **Data Injection**: Resolves `data_id` references using the registered `data_registry` and passes `_resolved_data` to evaluators.

### Global Component Registries
MalthusJAX uses lightweight catalog registries (`_shared_registry.py`) providing `register`, `register_table`, and `get_registry()`:

- **Operator Registry** (`_registry.py`): Genetic operators (selection, crossover, mutation, emitters, evaluators).
- **Engine Registry** (`engine_registry.py`): Engine factories (`"ga"`, `"mo"`, `"qd"`, `"island"`).
- **Genome Registry** (`_genome_registry.py`): Genome configurations (`"real"`, `"binary"`, `"categorical"`, `"linear"`, `"cartesian"`, `"series"`).

### Component Decorators (`decorators.py`)
Modules use these decorators to register new plugins automatically:
- `@register_selection(name="...", compatible_genomes=[...])`
- `@register_crossover(name="...", compatible_genomes=[...])`
- `@register_mutation(name="...", compatible_genomes=[...])`
- `@register_emitter(name="...", compatible_genomes=[...])`
- `@register_fitness(name="...", compatible_genomes=[...])`
- `@register_engine(name="...")`
- `@register_genome(name="...")`

---

## 6. External Framework Adapters

Composer bridges third-party evolutionary computation frameworks into the standard MalthusJAX `Engine` protocol interface (`run_once(key) -> Dict[str, Any]` returning `"history"`, `"summary"`, `"timings"`).

### Universal Composable Adapters
- **Composable EvoSAX Adapter (`composable_evosax_adapter.py`)**:
  - Wraps any distribution-based or population-based EvoSAX strategy (e.g., `SimpleGA`, `CMA_ES`, `DifferentialEvolution`, `OpenES`).
  - Supports dual evaluation modes (`EvalMode`):
    - `EvalMode.MJX`: Evaluates raw EvoSAX population arrays using MalthusJAX composable evaluators (`_evosax_mjx_eval`).
    - `EvalMode.NATIVE`: Runs native EvoSAX problem instances (`_evosax_native_eval`).
  - Custom metrics specification via `MetricSpec`.
- **Composable TensorNEAT Adapter (`composable_tensor_neat_adapter.py`)**:
  - Bridges TensorNEAT NEAT and HyperNEAT algorithms.
  - Dynamically inspects available algorithms (`list_algorithms`), genomes (`list_genomes`), and problems (`list_problems`).
  - Supports both native problem evaluations and MalthusJAX composable evaluators.

### Standard Factory Adapters
- **EvoSAX Adapter (`evosax_adapter.py`)**: `build_evosax_engine()` wraps legacy and direct EvoSAX pipelines.
- **QDAX Adapter (`qdax_adapter.py`)**: `build_qdax_engine()` wraps QDAX emitters, repertoires, and metrics into `UniversalAdapterEngine`.
- **TensorNEAT Adapter (`tensorneat_adapter.py`)**: `build_tensorneat_engine()` wraps TensorNEAT pipelines.
- **Kozax Adapter (`kozax_adapter.py`)**: `build_kozax_engine()` wraps Kozax Genetic Programming workflows.

---

## 7. TOML Configuration Structure (Option C)

The `load_experiment_config(path)` function parses declarative TOML files:

```toml
[experiment]
name        = "tsp_demo"
output_dir  = "results/tsp_demo"
description = "Benchmark GA on 52-city TSP with blend crossover variants"

[experiment.shared]
fitness       = "tsp:data_id=berlin52_synthetic,maximize=false"
genome_type   = "real"
genome_length = 52
bounds        = [0.0, 1.0]
maximize      = false
pop_size      = 100
generations   = 50
seeds         = [1, 2, 3]
selection     = "tournament:num_selections=50,tournament_size=3"
mutation      = "gaussian:mutation_rate=0.1,mutation_strength=0.2"

# Data-driven registry section
[data.berlin52_synthetic]
source        = "synthetic"
type          = "tsp"
num_cities    = 52
random_seed   = 42

[pipelines.ga_baseline]
description   = "BLX-α with α=0.5"
crossover     = "blend:alpha=0.5"

[pipelines.ga_blend]
description   = "BLX-α with α=0.8"
crossover     = "blend:alpha=0.8"
```

---

## 8. Result Objects & Serialization

Composer connects execution to `BenchmarkRunner` (`benchmarking` package):
- **`RunResult`**: Output of a single seed execution containing history arrays, timing breakdowns, and final population metrics.
- **`ExperimentResult`**: Holds multi-seed `RunResult` records for a pipeline. Provides `.aggregated_summary()`, `.combined_history()`, and confidence intervals (`ci_lower`, `ci_upper`).
- **`ComparisonResult`**: Holds multi-pipeline `ExperimentResult` objects. Supports `.summary_table()` (exportable to Markdown/LaTeX), `.plot_convergence()`, and statistical hypothesis tests.
- **Artifact Serialization**: Automatically writes structured JSON outputs (`metadata/config_snapshot.toml`, `data/<pipeline>/seed_<X>.json`, `analysis/summary.json`).
