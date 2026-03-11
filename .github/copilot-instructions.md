# MalthusJAX AI Coding Assistant Instructions

> Last updated: 2026-02-24

## Project Overview
MalthusJAX is a JAX-based evolutionary computation framework with a multi-level
hierarchical architecture optimized for JIT compilation and GPU acceleration.
It includes a **Composer** for config-driven experiments, a **Benchmarking**
module with an `Engine` protocol, evosax interop wrappers, and a
**Visualization** layer for single-run / multi-run analysis.

---

## Repository Layout

```
MalthusJAX/
├── src/malthusjax/          # Main package
│   ├── core/                # Level 1 — Genomes, fitness evaluators, base abstractions
│   │   ├── base.py          #   BaseGenome, BasePopulation, spawn_offspring
│   │   ├── random.py        #   PRNG utilities
│   │   ├── genome/          #   BinaryGenome, RealGenome, CategoricalGenome, LinearGenome
│   │   └── fitness/         #   BaseEvaluator + concrete evaluators (see below)
│   ├── operators/           # Level 2 — Selection, crossover, mutation operators
│   │   ├── base.py          #   BaseMutation, BaseCrossover, BaseSelection
│   │   ├── base_injection.py#   Injection-mode operator base
│   │   ├── base_ablation.py #   Ablation decorator utilities
│   │   ├── selection/       #   TournamentSelection, RouletteSelection, ElitePoolSelection
│   │   ├── crossover/       #   Real, Binary, + evosax wrapper (see below)
│   │   └── mutation/        #   Real, Binary, + evosax wrapper (see below)
│   ├── engine/              # Level 3 — Evolution engines
│   │   ├── base.py          #   AbstractEngine, AbstractEvolutionState, AbstractEngineParams, AbstractGenerationOutput
│   │   ├── genetic_fastengine.py  #   GeneticEngine, GeneticEngineParams, GeneticGenerationOutput
│   │   └── resource_mapper.py     #   ResourceMap, KeyDerivationStrategy, ShardingManager
│   ├── composer/            # Level 3.5 — Config-driven experiment orchestration
│   │   ├── composer.py      #   Composer (quick_run, catalog-based builds)
│   │   ├── catalog.py       #   OperatorCatalog (string -> operator registry)
│   │   ├── config.py        #   load_experiment_config (TOML loader)
│   │   ├── engine_factory.py#   build_engine_from_catalog
│   │   ├── evosax_adapter.py#   EvosaxEngineAdapter, build_evosax_engine, list_strategies
│   │   ├── pipeline.py      #   Pipeline composition
│   │   ├── node.py          #   Pipeline node abstraction
│   │   └── registry.py / _registry.py  #  Catalog registration helpers
│   ├── benchmarking/        # Level 3.5 — Multi-seed benchmarking infrastructure
│   │   ├── runner.py        #   BenchmarkRunner, Engine (Protocol), StubEngine
│   │   ├── results.py       #   RunResult, ExperimentResult, ComparisonResult
│   │   ├── io.py            #   Artifact I/O (JSON, CSV, seed folders)
│   │   ├── cli.py           #   CLI interface
│   │   └── __main__.py      #   python -m malthusjax.benchmarking
│   └── visualization/       # Level 4 — Plotting & analysis
│       ├── base.py          #   AbstractVisualizer, VisualizationConfig
│       ├── single_run.py    #   EvolutionVisualizer, GeneticAlgorithmVisualizer
│       └── multi_run.py     #   EngineComparator, FunctionalDataAnalyzer
├── tests/                   # Mirrors src/ structure
│   ├── conftest.py          #   Shared fixtures (PRNGKey, sample data)
│   ├── core/                #   Genome, fitness evaluator tests
│   ├── engine/              #   Engine, PRNG, resource mapper tests (18 files)
│   ├── operators/           #   Selection, crossover, mutation tests
│   ├── composer/            #   Catalog, adapter, integration tests (12 files)
│   ├── benchmarking/        #   Runner, results, IO tests
│   └── benchmarks/          #   Snapshot benchmark suite (pytest-benchmark, evosax comparison)
├── benchmarks/              # External benchmark configs & CLI runners
│   ├── framework/           #   adapters.py, registry.py, runner.py (low-level timing/HLO)
│   └── *.toml               #   Experiment configs (smoke_test, gecco_benchmark, etc.)
├── examples/
│   ├── _DEMO_LV_1/          #   Level 1 demos (genome, fitness)
│   ├── _DEMO_LV_2/          #   Level 2 demos (operators, evosax comparison)
│   ├── _DEMO_LV_3/          #   Level 3 demos (full engine evolution)
│   └── _DEMO_COMPOSER/      #   Composer API demos
├── docs/
│   ├── INEFFICIENCIES_AND_SOLUTIONS.md   #   26-item issue catalogue
│   ├── FIX_PLAN.md                       #   9-phase fix plan
│   ├── ADR/                              #   Architecture Decision Records
│   └── source/                           #   Sphinx documentation source
├── results/                 # Experiment outputs, CSV data, figures
├── Makefile                 # Dev workflow commands
└── pyproject.toml           # Project metadata & dependencies
```

---

## Core Architecture Principles

### Multi-Level Hierarchy
| Level | Location | Responsibility |
|-------|----------|----------------|
| **1** | `src/malthusjax/core/` | Genomes, fitness evaluators, base population abstractions |
| **2** | `src/malthusjax/operators/` | Selection, crossover, mutation operators |
| **3** | `src/malthusjax/engine/` | Complete evolutionary algorithms (scan-based evolution loop) |
| **3.5** | `src/malthusjax/composer/` | Config-driven experiment orchestration & evosax interop |
| **3.5** | `src/malthusjax/benchmarking/` | Multi-seed execution, `Engine` protocol, result comparison |
| **4** | `src/malthusjax/visualization/` | Plotting & analysis (single-run, multi-run) |

### JAX-First Design Pattern
All components follow the **@struct.dataclass factory pattern** for JIT compilation:
```python
# @struct.dataclass + __call__ pattern
operator = TournamentSelection(num_selections=4, tournament_size=3)
selected_indices = operator(key, fitness_values)  # Direct callable interface
jit_operator = jax.jit(operator)                  # JIT the entire operator
```

### Pure Function Signatures (Critical)
All genetic operators follow the **batch-first paradigm** with unified signatures:

**Selection Operators**:
```python
(key: jax.Array, fitness_values: jax.Array) -> selected_indices: jax.Array
# fitness_values: (pop_size,) -> selected_indices: (num_selections,)
```

**Crossover Operators**:
```python
(key: jax.Array, parent1: Genome, parent2: Genome, config: Config) -> offspring: Genome
# Output shape: (num_offspring, ...genome_shape) -- BATCH-FIRST!
```

**Mutation Operators**:
```python
(key: jax.Array, genome: Genome, config: Config) -> mutated_genome: Genome
# Output shape: (num_offspring, ...genome_shape) -- BATCH-FIRST!
```

### Tensorization Requirements
- All core classes use **BaseGenome / BasePopulation** architecture in `core/base.py`
- Implement **@struct.dataclass** with flax for immutability and JIT compatibility
- Use `jax.vmap()` for population-level operations
- Automatic vectorization built into operator `__call__` methods

---

## Key Patterns & Conventions

### Genome Implementations
- Config classes are **@dataclass(frozen=True)** with PyTree registration
- Use `random_init(key, config)` static method for JIT-compiled initialization
- **4 genome types**: `BinaryGenome`, `RealGenome`, `CategoricalGenome`, `LinearGenome`
- Each has a matching `*Config` and `*Population` class
- Population classes provide vectorized operations: `distance_matrix()`, slicing
- `RealGenomeConfig` uses `shape: Tuple[int, ...]` (NOT `length`) plus `bounds: Tuple[float, float]`

### Fitness Evaluators
- Base: `BaseEvaluator` / `BaseEvaluatorConfig` in `core/fitness/base.py`
- Implement `get_tensor_fitness_function()` returning a pure JAX function
- Batch evaluation via `evaluate_batch()` using `jax.vmap()`
- **Concrete evaluators**:
  - Binary: `BinarySumEvaluator`, `KnapsackEvaluator`
  - Real: `SphereEvaluator`, `GriewankEvaluator`, `BoxEvaluator`
  - BBOB: `BBOBEvaluator` (wraps `evosax.problems.BBOBProblem`; factory `BBOBEvaluator.create(config)`)
  - GP: `LinearGPEvaluator`
- Catalog string names (for Composer): `"sphere"`, `"rastrigin"`, `"griewank"`, `"binary_sum"`, `"knapsack"`, `"bbob"`, etc.

### Genetic Operators

**Abstract bases** in `operators/base.py`: `BaseMutation`, `BaseCrossover`, `BaseSelection`
- Also `base_injection.py` for injection-mode variants, `base_ablation.py` for ablation decorators

**Selection** (`operators/selection/`):
| Operator | Catalog Name |
|----------|-------------|
| `TournamentSelection` | `"tournament"` |
| `RouletteSelection` | `"roulette"` |
| `ElitePoolSelection` | `"elite_pool"` |

**Crossover** (`operators/crossover/`):
| Operator | Catalog Name | Notes |
|----------|-------------|-------|
| `SinglePointCrossover` | `"single_point"` | Binary |
| `BinaryUniformCrossover` | `"uniform_binary"` | Binary |
| `RealUniformCrossover` | `"uniform_real"` | Real |
| `BlendCrossover` | `"blend"` | Real (BLX-alpha) |
| `BinomialCrossover` | `"binomial"` | Real (DE-style) |
| `SimulatedBinaryCrossover` | `"simulated_binary"` | Real (SBX) |
| `*_injection` variants | `"*_injection"` | Injection-mode versions |
| `EvosaxUniformCrossoverWrapper` | `"evosax_uniform"` | Wraps evosax SimpleGA crossover |

**Mutation** (`operators/mutation/`):
| Operator | Catalog Name | Notes |
|----------|-------------|-------|
| `BitFlipMutation` | `"bitflip"` | Binary |
| `ScrambleMutation` | `"scramble"` | Binary |
| `SwapMutation` | `"swap"` | Binary |
| `GaussianMutation` | `"gaussian"` | Real |
| `BallMutation` | `"ball"` | Real |
| `PolynomialMutation` | `"polynomial"` | Real |
| `*_injection` variants | `"*_injection"` | Injection-mode versions |
| `EvosaxGaussianWrapper` | `"evosax_gaussian"` | Wraps evosax SimpleGA mutation |

**Pattern:**
```python
@struct.dataclass
class BitFlipMutation(BaseMutation):
    num_offspring: int = 1
    mutation_rate: float = 0.1
    
    def __call__(self, key, genome, config):
        return _bitflip_impl(key, genome, config, self.mutation_rate, self.num_offspring)
```

### Evolution Engines
- **Abstract base**: `AbstractEngine` in `engine/base.py`
- **Implementation**: `GeneticEngine` in `engine/genetic_fastengine.py` (NOT `basic_engine.py`)
- **State**: `AbstractEvolutionState` -> `GeneticEvolutionState` (population, best_genome, generation, best_fitness, stagnation_counter, rng_key)
- **Params**: `AbstractEngineParams` -> `GeneticEngineParams` (pop_size, num_generations, elitism, unroll_num, key_derivation, prng_impl, mutation_strength_schedule)
- **Output**: `AbstractGenerationOutput` -> `GeneticGenerationOutput` (best_fitness, mean_fitness, generation, random_key)
- **Resource allocation**: `ResourceMap`, `KeyDerivationStrategy` (`SPLIT` / `FOLD`), `ShardingManager` in `resource_mapper.py`
- JIT-compile evolution loops with `jax.lax.scan` and XLA compilation caching

### Composer (Config-Driven Experiments)
- `Composer` class with `quick_run()` method -- one-call experiment execution
- Supports `backend="malthusjax"` or `backend="evosax"` (via `EvosaxEngineAdapter`)
- `OperatorCatalog` resolves string names -> operator instances
- `build_engine_from_catalog()` constructs engines from catalog specs
- `load_experiment_config()` reads TOML experiment definitions
- Returns `ExperimentResult` from the benchmarking module

### Benchmarking Infrastructure
- `Engine` protocol: `run_once(key) -> Dict[str, Any]` returning `{history, summary, timings}`
- `BenchmarkRunner`: iterates seeds, wraps `engine.run_once()`, builds `ExperimentResult`
- `RunResult` -> `ExperimentResult` (multi-seed) -> `ComparisonResult` (multi-pipeline)
- `ComparisonResult` provides `summary_table()`, `convergence_data()`, `plot_convergence()`
- IO: `write_summary_json`, `write_histories_csv`, `ensure_seed_folder`
- CLI: `python -m malthusjax.benchmarking`

### Visualization
- `AbstractVisualizer` base with `VisualizationConfig`
- `EvolutionVisualizer`, `GeneticAlgorithmVisualizer` for single-run analysis
- `EngineComparator`, `FunctionalDataAnalyzer` for multi-run comparison

---

## Development Workflow

### Environment Setup
```bash
make install-dev  # Install with dev dependencies
```

### Code Quality Pipeline
```bash
make check-all    # Runs lint, format, type-check, test
make test         # pytest with coverage (80% minimum)
make lint         # Ruff linting (replaces flake8, black, isort)
make format       # Ruff formatting
make type-check   # mypy with strict settings
make docs         # Build Sphinx documentation
```

### Benchmark Suite
```bash
# Snapshot benchmarks (pytest-benchmark, compares MalthusJAX vs evosax)
pytest tests/benchmarks/test_snapshot_benchmark.py -v --benchmark-only
pytest tests/benchmarks/test_snapshot_benchmark.py -v --benchmark-save=baseline
pytest tests/benchmarks/test_snapshot_benchmark.py -v --benchmark-compare=0001_baseline

# Convergence parity tests (uses BenchmarkRunner + ComparisonResult)
pytest tests/benchmarks/test_snapshot_benchmark.py -v -k "test_fitness_parity"
```

### Testing Strategy
- Test structure mirrors `src/` in `tests/`
- Use `conftest.py` for shared fixtures (random keys, sample data)
- JAX random keys from `jr.PRNGKey(42)` fixture
- Mark slow tests with `@pytest.mark.slow`
- **Benchmark tests** in `tests/benchmarks/` -- 6 groups: single-step latency, multi-gen throughput, JIT compilation time, operator microbenchmarks, convergence parity, scaling sweep

---

## Critical Implementation Details

### Random Key Management
```python
# In JAXTensorizable base class
@property
def random_key(self):
    if self.jax_random_key is not None:
        self.jax_random_key, subkey = jar.split(self.jax_random_key)
        return subkey  # Auto-splits to prevent reuse
```

### JIT Compilation Strategy
- Level 1: JIT individual functions (fitness, initialization)
- Level 2: JIT operator functions directly with `jax.jit(operator)`
- Level 3: JIT entire evolution step in engines via `jax.lax.scan`

### Dependencies & Tooling
- **Core**: JAX 0.4+, jaxlib 0.4+, numpy 1.21+, flax 0.7+, chex 0.1.7+, scikit-learn 1.0+, Python 3.8+
- **Build**: Hatchling 1.8+
- **Evosax interop**: evosax (for `BBOBEvaluator`, crossover/mutation wrappers, `EvosaxEngineAdapter`)
- **Dev tools**: Ruff (linting + formatting), mypy, pytest, pytest-cov, pytest-xdist, pre-commit
- **Docs**: Sphinx with autodoc, sphinx-rtd-theme, myst-parser
- **Examples**: matplotlib, seaborn, jupyter

---

## Known Issues & Active Fix Plan

> See `docs/INEFFICIENCIES_AND_SOLUTIONS.md` (26 items) and `docs/FIX_PLAN.md` (9 phases).

| Phase | Name | Issues | Status |
|-------|------|--------|--------|
| **0** | Safety net -- snapshot baselines | -- | Complete |
| **1** | Correctness hot-fixes (no API change) | ~~CV-1~~, ~~CV-2~~, ~~JR-1~~, ~~JR-3~~ | ✅ Done |
| **2** | Mutation schedule redesign | ~~CV-3~~, ~~JR-4~~ | ✅ Done |
| **3** | Crossover fusion & operator hot-path | ~~FB-1~~, ~~FB-2~~, ~~FB-5~~ | ✅ Done |
| **4** | Engine merge & HOF rewrite | ~~FB-3~~, ~~FB-4~~ | ✅ Done |
| **5** | Population API hardening | CV-4, CV-5, MB-1 | Not started |
| **6** | Config dispatch & init cleanup | ~~JR-2~~, ~~AR-1~~, ~~AR-3~~ | ✅ Done |
| **7** | Memory-bound operators | ~~MB-2~~ | ✅ Done |
| **8** | Micro-optimizations & polish | MO-1 to MO-6, AR-2, AR-4 | Not started |

### Key Issues to Be Aware Of (affect how you write code)
- **CV-4**: `BasePopulation.__iter__` triggers O(N) host-device sync. Will be gated or removed.
- **`engine.run()` donates buffers**: After calling `engine.run(state)`, the `state` buffers are consumed. Always re-init state for repeated runs.

---

## Common Gotchas
- **CRITICAL: DO NOT USE `__post_init__` in JAX/Flax dataclasses!** It runs during tracing and causes `TracerBoolConversionError` if it contains checks on traced values (like `if self.capacity > 0`). Validation should happen outside the JIT boundary or be omitted for performance.
- JAX arrays are immutable -- always return new arrays
- Use `jax.vmap()` for population operations, not Python loops
- Config classes must be registered as PyTrees for JIT compatibility
- **Operator signature order matters**: Always `(key, ...inputs)` for consistency
- Use `@partial(jax.jit, static_argnames=[...])` for functions with static arguments
- Random keys auto-split in `JAXTensorizable` -- don't reuse
- Genetic operators don't implement tensor serialization
- **Crossover n_outputs**: Shape is `(n_outputs, ...genome_shape)`, not flattened
- **`RealGenomeConfig`**: Uses `shape=(dims,)`, NOT `length=dims`
- **`engine.run()` donates buffers**: After calling `engine.run(state)`, the `state` buffers are consumed. Always re-init state for repeated runs.

---

## File Navigation
- Core abstractions: `src/malthusjax/core/base.py`
- Genome types: `src/malthusjax/core/genome/` (`real_genome.py`, `binary_genome.py`, `categorical_genome.py`, `linear_genome.py`)
- Fitness evaluators: `src/malthusjax/core/fitness/` (`bbob_evaluator.py`, `real_evaluators.py`, `binary_evaluators.py`, `linear_gp_evaluator.py`)
- Operator bases: `src/malthusjax/operators/base.py`, `base_injection.py`
- Selection: `src/malthusjax/operators/selection/` (`tournament.py`, `roulette.py`, `elite_pool.py`)
- Crossover: `src/malthusjax/operators/crossover/` (`real.py`, `binary.py`, `evosax_crossover.py`)
- Mutation: `src/malthusjax/operators/mutation/` (`real.py`, `binary.py`, `evosax_mutation.py`)
- Engine abstractions: `src/malthusjax/engine/base.py`
- Engine implementation: `src/malthusjax/engine/genetic_fastengine.py`
- Resource mapper: `src/malthusjax/engine/resource_mapper.py`
- Composer: `src/malthusjax/composer/composer.py`
- Benchmarking runner: `src/malthusjax/benchmarking/runner.py`
- Benchmarking results: `src/malthusjax/benchmarking/results.py`
- Visualization: `src/malthusjax/visualization/` (`single_run.py`, `multi_run.py`)
- Snapshot benchmarks: `tests/benchmarks/test_snapshot_benchmark.py`
- Example notebooks: `examples/_DEMO_LV_1/`, `_DEMO_LV_2/`, `_DEMO_LV_3/`, `_DEMO_COMPOSER/`
- Issue catalogue: `docs/INEFFICIENCIES_AND_SOLUTIONS.md`
- Fix plan: `docs/FIX_PLAN.md`
- Test fixtures: `tests/conftest.py`
