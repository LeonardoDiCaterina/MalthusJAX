# Experiment Orchestration via Composer

The `malthusjax.composer` module provides high-level experiment orchestration. It abstracts low-level engine construction, operator composition, and seed loop iteration into declarative configuration files (TOML) or programmatic entry points (`quick_run()`, `compare()`, `from_toml()`).

---

## 5.1. The Composer Entry Points

The `Composer` class (`composer.py`) provides three primary entry points:

### 5.1.1. Interactive Exploration (`quick_run`)
`Composer.quick_run()` runs an evolutionary experiment across multiple random seeds using string-based operator specifications or pre-constructed engines.

Key arguments:
- `fitness: Optional[str]` — Fitness function spec (e.g. `"sphere:dim=10"`, `"bbob:fn_name=sphere,dim=10"`, `"knapsack:capacity=100,num_items=20"`).
- `selection: Optional[str]` — Selection spec (e.g. `"tournament:num_selections=25,tournament_size=3"`).
- `crossover: Optional[str]` — Crossover spec (e.g. `"blend:alpha=0.5"`).
- `mutation: Optional[str]` — Mutation spec (e.g. `"gaussian:mutation_rate=0.1,mutation_strength=0.1"`).
- `backend: str` — Execution backend (`"malthusjax"` [default], `"evosax"`, `"qdax"`, `"tensorneat"`).
- `pop_size: int` (default 50), `generations: int` (default 100), `elitism: int` (default 2).
- `seeds: Sequence[int]` — Random seed list (default `(1, 2, 3)`). Normalized via `_normalize_seeds()`.

Returns an `ExperimentResult` containing per-seed `RunResult` outputs and aggregated statistics (`aggregated_summary()`).

### 5.1.2. TOML-Driven Experiments (`from_toml`)
`Composer.from_toml(path)` parses a declarative TOML configuration via `load_experiment_config(path)` and executes all declared pipelines, preserving shared parameters and random seeds across pipeline comparisons. Returns a `ComparisonResult`.

### 5.1.3. Programmatic Comparison (`compare`)
`Composer.compare(pipelines={...}, fitness=..., pop_size=..., seeds=...)` runs multiple pipeline parameter dictionaries side-by-side using aligned initial populations and seeds. Returns a `ComparisonResult` with statistical tables (`summary_table()`) and convergence plotting tools (`plot_convergence()`).

---

## 5.2. Catalog & String Specification Resolution

Composer maps human-readable string specifications to instantiated PyTree objects using regex key-value parsing (`key1=val1,key2=val2`).

### 5.2.1. `OperatorCatalog` (`catalog.py`)
`OperatorCatalog` resolves operator string specs:
- Regex format: `"operator_name:param1=value1,param2=value2"`
- Parameter coercion: Automagically converts numeric strings to `int` or `float` and booleans to `bool`.
- API methods: `get_selection(spec)`, `get_crossover(spec)`, `get_mutation(spec)`, `list_operators()`.

### 5.2.2. Custom Component Decorators (`decorators.py`)
Users can extend the catalog dynamically using registry decorators:
- `@register_selection(name="...")`
- `@register_crossover(name="...")`
- `@register_mutation(name="...")`
- `@register_fitness(name="...")`
- `@register_engine(name="...")`
- `@register_genome(name="...")`

### 5.2.3. `EngineRegistry` (`engine_catalog.py`)
`EngineRegistry` maps engine type keywords to concrete engine factory builders:
- `"ga"` — Standard `GeneticEngine`.
- `"mo"` — Multi-Objective `MOEngine` (NSGA-II).
- `"qd"` — Quality-Diversity `MapElitesEngine` (requires `qdax`).
- `"island"` — Distributed `BaseIslandModel` (`RingTopologyIsland`, `FullyConnectedIsland`).

---

## 5.3. Declarative TOML Experiment Loading

The `load_experiment_config(path)` function (`config.py`) loads TOML files using `tomllib` (Python 3.11+) or `tomli`.

### Verified TOML Schema
- `[experiment]`: Top-level metadata.
  - `name: str` — Experiment identifier.
  - `output_dir: str` — Directory for saving output JSON files.
- `[experiment.shared]`: Shared baseline configuration merged into all pipelines.
  - `fitness: str`
  - `pop_size: int`
  - `generations: int`
  - `genome_length: int`
  - `bounds: list` (automatically cast to tuple `(min_b, max_b)`)
  - `seeds: list` (automatically cast to tuple)
  - `prng_impl: str`
  - `elitism: int`
  - `maximize: bool`
- `[pipelines.<pipeline_name>]`: Pipeline-specific configuration overrides.
  - `backend: str`
  - `engine_type: str`
  - `selection: str`
  - `crossover: str`
  - `mutation: str`

### Verified Worked TOML Example
```toml
[experiment]
name = "crossover_comparison"
output_dir = "results/crossover_comparison"

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
```

---

## 5.4. External Framework Adapters

Composer provides universal adapter builders to run external libraries under the unified `Engine` benchmark interface (`run_once(key) -> Dict` returning `"history"`, `"summary"`, `"timings"`):

- **EvoSAX Adapter (`evosax_adapter.py`)**: `build_evosax_engine()` wraps EvoSAX strategies (e.g. `SimpleGA`, `CMA_ES`) into `UniversalAdapterEngine`.
- **QDAX Adapter (`qdax_adapter.py`)**: `build_qdax_engine()` wraps QDAX emitters, repertoires, and metrics into `UniversalAdapterEngine`.
- **TensorNEAT Adapter (`tensorneat_adapter.py`)**: `build_tensorneat_engine()` wraps TensorNEAT algorithm and problem instances into `UniversalAdapterEngine`.
- **Kozax Adapter (`kozax_adapter.py`)**: `build_kozax_engine()` wraps Kozax GP strategies into `UniversalAdapterEngine`.

---

## 5.5. Results & Benchmark Integration

Composer connects engines to `BenchmarkRunner` (`benchmarking` package):
- `run_once(key)` Contract: Executes a single seed run and returns a dictionary with:
  - `"history"`: List of per-generation metric dictionaries (`generation`, `best_fitness`, etc.).
  - `"summary"`: Final metrics dictionary (`best_fitness`, `final_generation`, `total_evaluations`).
  - `"timings"`: Execution duration dictionary (`{"total": elapsed_seconds}`).
- `ExperimentResult`: Output container for a single pipeline run across multiple seeds, exposing `.aggregated_summary()`.
- `ComparisonResult`: Output container for multi-pipeline comparisons, exposing `.summary_table()` and `.plot_convergence()`.
- JSON Serialization: Results are saved to disk as JSON files (`metadata/config_snapshot.toml`, `data/<pipeline>/seed_<X>.json`, `analysis/summary.json`).
