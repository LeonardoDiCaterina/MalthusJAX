# `malthusjax.composer` — Technical Reference

Scope: `malthusjax.composer.composer`, `malthusjax.composer.experiment_config`, `malthusjax.composer.engine_factory`, `malthusjax.composer.engine_protocol`, `malthusjax.composer.backend_provider`, `malthusjax.composer.backend_registry`, `malthusjax.composer.backends`, `malthusjax.composer.catalog`, `malthusjax.composer.config`, `malthusjax.composer.strategies`, `malthusjax.composer.evaluator_parser`, `malthusjax.composer.decorators`, `malthusjax.composer.adapters`.

---

## 1. Overview & 3-Layer Architecture

The `malthusjax.composer` package is the top-level orchestration layer of **MalthusJAX**.

Following the architectural standards of production-grade frameworks like **Hydra**, **Kedro**, and **PyTorch Lightning**, the Composer establishes a clean, decoupled **3-Layer Architecture**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. CONFIGURATION LAYER (Declarative, Immutable, Type-Safe)              │
│                                                                         │
│   ExperimentConfig                                                      │
│   ├── PopulationConfig (size, type, length, bounds, spec)               │
│   ├── ExecutionConfig  (generations, seeds, maximize, prng)             │
│   ├── BackendConfig    (MalthusJAX, Evosax, Qdax, Tensorneat, Generic)  │
│   ├── LoggingConfig    (intervals, watchdog, metrics)                   │
│   └── OutputConfig     (name, output_dir, trace_dir)                    │
│                                                                         │
│   Parsers & Loaders:                                                    │
│   • ExperimentConfig.from_dict()                                        │
│   • ExperimentConfig.from_toml()                                        │
│   • ExperimentConfig.from_quick_run_kwargs() (100% backward-compatible) │
│   • config.validate() (structural & semantic verification)              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. INSTANTIATION / FACTORY BOUNDARY (Dependency Injection & Resolution) │
│                                                                         │
│   EngineFactory.build(config: ExperimentConfig) -> Engine               │
│   ├── Step 1: config.validate()                                         │
│   ├── Step 2: BackendRegistry.get_backend(name)                         │
│   ├── Step 3: Provider.resolve_evaluator(spec, **kwargs)                │
│   ├── Step 4: Provider.default_strategy(**kwargs)                       │
│   └── Step 5: Provider.build_engine(strategy, evaluator, **kwargs)      │
│                                                                         │
│   Centralized Contracts & Boundaries:                                   │
│   • BackendRegistry: Strict runtime method validation on registration   │
│   • BackendProvider: Abstract base class for native and third-party libs│
│   • _population_init: Shared uniform & BBOB initial population sampler  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. EXECUTION ORCHESTRATION (Hardware JIT Loop & Multi-Run Benchmarking) │
│                                                                         │
│   @runtime_checkable class Engine(Protocol):                            │
│       run_once(key: chex.Array) -> Dict[str, Any]                       │
│       (Contract output: "history", "summary", "timings")                │
│                                                                         │
│   BenchmarkRunner.run(engine: Engine, seeds: Sequence[int])             │
│   • Native JIT Loops (jax.lax.scan)                                     │
│   • Universal Adapters (UniversalAdapterEngine)                         │
│   • Result Aggregation: ExperimentResult & ComparisonResult             │
│   • Telemetry: Zero-overhead JIT callbacks & NaN/Inf watchdogs          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1: Typed Configuration Model (`experiment_config.py`)

All experiment configurations are defined as frozen dataclasses in `experiment_config.py`. This ensures immutable, hashable, and fully verifiable specifications prior to any JAX compilation.

### The Config Tree
- **`ExperimentConfig`**: Root container aggregating all sub-configurations:
  - `population: PopulationConfig`
  - `execution: ExecutionConfig`
  - `backend: BackendConfig` (discriminated union)
  - `logging: LoggingConfig`
  - `output: OutputConfig`

### Sub-Configuration Schemas
```python
@dataclass(frozen=True)
class PopulationConfig:
    size: int = 50
    genome_type: str = "real"
    genome_length: int = 10
    bounds: Tuple[float, float] = (-5.0, 5.0)
    genome_spec: Optional[str] = None

@dataclass(frozen=True)
class ExecutionConfig:
    generations: int = 100
    seeds: Tuple[int, ...] = (1, 2, 3)
    maximize: bool = False
    prng_impl: Optional[str] = None
    use_history_for_final: bool = False

@dataclass(frozen=True)
class LoggingConfig:
    log_level: Optional[str] = None
    log_interval: Optional[int] = None
    log_nan_watchdog: bool = True
    history_metrics: Optional[Tuple[str, ...]] = None

@dataclass(frozen=True)
class OutputConfig:
    experiment_name: str = "quick_experiment"
    output_dir: Optional[str] = None
    trace_dir: Optional[str] = None
```

### Backend Configurations
- `MalthusJAXBackendConfig`: Parameters for native genetic engines (`selection`, `crossover`, `mutation`, `engine_type`).
- `EvosaxBackendConfig`: Parameters for EvoSAX strategies (`strategy_name`, `strategy_kwargs`).
- `QdaxBackendConfig`: Parameters for QDAX Quality-Diversity algorithms (`emitter`, `num_descriptors`).
- `TensorneatBackendConfig`: Parameters for TensorNEAT neuroevolution (`algorithm_name`, `problem_name`).
- `GenericBackendConfig`: Extensible key-value store for third-party or custom backend providers.

### Validation & Serialization
```python
# Validation
config.validate()  # Validates positive pop_size, valid bounds, non-empty seeds, etc.

# Serialization
data = config.to_dict()
reconstructed = ExperimentConfig.from_dict(data)

# Backward-Compatible Kwargs Conversion
config = ExperimentConfig.from_quick_run_kwargs(
    fitness="sphere:dim=10", pop_size=64, generations=100, seeds=[42, 43]
)

# Declarative TOML Loading
config = ExperimentConfig.from_toml("path/to/experiment.toml")
```

---

## 3. Layer 2: Instantiation Boundary (`engine_factory.py`)

The `EngineFactory` acts as the single entry point for instantiating evolutionary algorithms from declarative configurations.

```python
from malthusjax.composer.engine_factory import EngineFactory

# Build an engine directly from a validated configuration:
engine = EngineFactory.build(config)

# Or pass a pre-resolved custom evaluator:
engine = EngineFactory.build_with_evaluator(config, evaluator=custom_eval_fn)
```

### Dynamic Backend Resolution
`EngineFactory.build()`:
1. Calls `config.validate()`.
2. Queries `BackendRegistry.get_backend(config.backend.backend_name)`.
3. Instantiates the registered [BackendProvider](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/src/malthusjax/composer/backend_provider.py).
4. Invokes `provider.resolve_evaluator(...)` with population config dimensions.
5. Invokes `provider.default_strategy(...)`.
6. Invokes `provider.build_engine(strategy, evaluator, **kwargs) -> Engine`.

### The `BackendProvider` Contract
```python
class BackendProvider(ABC):
    @abstractmethod
    def build_engine(self, strategy: Any, evaluator: Any, **kwargs: Any) -> Engine: ...

    @abstractmethod
    def resolve_evaluator(self, evaluator_spec: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def default_strategy(self, **kwargs: Any) -> Any: ...

    def generate_initial_population(self, key: chex.Array, config: Any, evaluator: Any = None) -> Any: ...
```

### Centralized Population Initialization (`backends/_population_init.py`)
To prevent divergent random population sampling across backends, `_population_init.py` provides shared, deterministic population generators:
- `generate_initial_population(key, pop_size, genome_type, genome_length, bounds, fitness_spec, evaluator)`
- `sample_population_bbob(...)`: Calibrated sampling for BBOB benchmarking functions.

---

## 4. Layer 3: Execution Protocol & Orchestration

### The `Engine` Protocol (`engine_protocol.py`)
MalthusJAX formalizes execution using a `@runtime_checkable` Python Protocol:

```python
@runtime_checkable
class Engine(Protocol):
    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        """Execute a single seed run.
        
        Returns:
            Dict containing:
            - 'history': Sequence[Dict[str, Any]] (per-generation metrics)
            - 'summary': Dict[str, Any] (aggregated final metrics)
            - 'timings': Dict[str, float] (wall-clock / warmup durations)
        """
        ...
```

### The Orchestrator (`composer.py`)
The `Composer` class wires the 3 layers together:

```python
class Composer:
    def quick_run(self, **kwargs) -> ExperimentResult:
        # Step 1: Configure
        config = ExperimentConfig.from_quick_run_kwargs(**kwargs)
        # Step 2: Resolve & Build
        engine = self.factory.build(config)
        # Step 3: Execute
        runner = BenchmarkRunner(engine=engine, experiment_name=config.output.name)
        return runner.run(config.execution.seeds)

    def from_toml(self, path: str) -> ComparisonResult:
        ...

    def compare(self, pipelines: Dict[str, Dict[str, Any]], **kwargs) -> ComparisonResult:
        ...
```

---

## 5. Catalog & Dynamic Registries

MalthusJAX retains rich string DSL parsing for rapid interactive experimentation:

### `OperatorCatalog` (`catalog.py`)
- Syntax: `"operator_name:param1=val1,param2=val2"`.
- Automatic Type Coercion: Automatically casts `"0.1"` to float, `"10"` to int, `"true"` to bool.
- Registry Decorators (`decorators.py`):
  - `@register_selection(name="...")`
  - `@register_crossover(name="...")`
  - `@register_mutation(name="...")`
  - `@register_fitness(name="...")`
  - `@register_backend(name="...")`

---

## 6. External Framework Adapters (`adapters/`)

External libraries are wrapped under the unified `Engine` protocol using `UniversalAdapterEngine`:

- **EvoSAX (`evosax_adapter.py`)**: Adapts `SimpleGA`, `CMA_ES`, `DifferentialEvolution`, `OpenES`. Supports native EvoSAX problems or MalthusJAX PyTree evaluators.
- **QDAX (`qdax_adapter.py`)**: Adapts MAP-Elites, emitters, and repertoires.
- **TensorNEAT (`tensorneat_adapter.py`)**: Adapts NEAT and HyperNEAT algorithms and problem wrappers.
- **Kozax (`kozax_adapter.py`)**: Adapts Kozax genetic programming trees.

All adapters inherit:
- Standardized `t_warmup` and `t_exec` timing using `jax.block_until_ready()`.
- Standardized JIT step telemetry callbacks via `log_interval`.
- Exact PRNG seed alignment.

---

## 7. Declarative TOML Schema

The Composer loads reproducible experiment files via `Composer.from_toml(path)`:

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
elitism       = 2
maximize      = false

[pipelines.blend_ga]
backend   = "malthusjax"
selection = "tournament:tournament_size=3"
crossover = "blend:alpha=0.5"
mutation  = "gaussian:mutation_rate=0.1"

[pipelines.evosax_cma]
backend       = "evosax"
strategy_name = "CMA_ES"
```

---

## 8. Submodule File Directory

| File | Role |
| :--- | :--- |
| `experiment_config.py` | Typed dataclass tree (`ExperimentConfig`, `PopulationConfig`, etc.) |
| `engine_factory.py` | Factory boundary resolving configs into `Engine` instances |
| `engine_protocol.py` | Runtime-checkable `Engine` protocol and metric contracts |
| `backend_provider.py` | Abstract base class for engine backend providers |
| `backend_registry.py` | Central backend registry with contract enforcement |
| `backends/_population_init.py` | Centralized population initialization utilities |
| `backends/malthusjax.py` | Native MalthusJAX backend provider |
| `backends/evosax.py` | EvoSAX backend provider |
| `backends/qdax.py` | QDAX backend provider |
| `backends/tensorneat.py` | TensorNEAT backend provider |
| `composer.py` | High-level `Composer` API (`quick_run`, `compare`, `from_toml`) |
| `config.py` | TOML experiment configuration loader |
| `catalog.py` | String DSL operator parser |
| `decorators.py` | Registry decorators (`@register_selection`, etc.) |
| `adapters/` | Universal adapters for external libraries |

---

## 9. Showcase Demonstrations & Cross-Backend Benchmarks

The suite in [`examples/showcase/showcase_cross_backend_compare.py`](file:///Users/leonardodicaterina/Documents/GitHub/MalthusJAX/examples/showcase/showcase_cross_backend_compare.py) demonstrates cross-framework benchmarking and parity validation:

```bash
# Run any individual demo or all of them:
python examples/showcase/showcase_cross_backend_compare.py --demo 1 --plot
python examples/showcase/showcase_cross_backend_compare.py --demo 2 --plot
python examples/showcase/showcase_cross_backend_compare.py --demo 3 --plot --seeds 30
python examples/showcase/showcase_cross_backend_compare.py --demo 5 --plot
python examples/showcase/showcase_cross_backend_compare.py --demo 7
```

### Showcase Highlights:

1. **Showcase 1: GA Operator Ablation** (Multimodal 10D Rastrigin):
   - Compares Uniform + Gaussian vs. Blend + Gaussian vs. SBX + Polynomial mutation.
   - Demonstrates declarative operator mix-and-match in pure MalthusJAX.

2. **Showcase 2: Cross-Framework Battle** (10D Sphere):
   - Direct head-to-head comparison between **MalthusJAX GA**, **EvoSAX CMA-ES**, **EvoSAX Sep-CMA-ES**, **EvoSAX Open-ES**, and **MalthusJAX Native MAP-Elites**.
   - Proves unified telemetry and fair evaluation across heterogeneous algorithm paradigms.

3. **Showcase 3: Quality-Diversity Cross-Validation** (Native MAP-Elites vs QDAX):
   - Evaluates on 5D BBOB Rastrigin with CVT Voronoi centroids (64 cells).
   - Validates **100% initial population parity** (Gen 0 fitness identical at $-90.058$).
   - Validates **final archive coverage parity** ($82.8125\%$ for both across 1,920 evaluations).
   - Proves **statistical parity** across 30 seeds ($p = 0.3704 > 0.05$ on paired $t$-test; $p = 0.3492$ on Wilcoxon signed-rank).
   - Demonstrates **2.5x runtime speedup** for Native MalthusJAX over upstream QDAX.

4. **Showcase 5: Hard BBOB Landscape Evaluation**:
   - Assesses algorithms across ill-conditioned Rosenbrock valleys (f8) and highly multimodal Rastrigin landscapes (f15).

5. **Showcase 7: Level 3 Engine Parity Reproduction**:
   - Demonstrates 1:1 parity between raw Level 3 `GeneticEngine` and Level 4 `Composer.quick_run()`.
   - Proves seamless registration and execution of custom subclassed engines (`DiversityTrackingEngine`) via `@register_engine` and `@register_genome`.

