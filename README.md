# MalthusJAX: A JAX-Native Evolutionary Computation Framework

[![JAX](https://img.shields.io/badge/JAX-0.4+-blue.svg)](https://github.com/google/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](http://mypy-lang.org/)
[![Coverage](https://img.shields.io/badge/coverage-80%25+-brightgreen.svg)](https://github.com/pytest-dev/pytest-cov)

MalthusJAX is a composable, type-safe evolutionary algorithm framework built on JAX and XLA for high-performance population-based optimization and evolutionary computation research. The framework uses a strict 3-level hierarchical architecture with static resource budgeting, functional purity, and explicit compilation boundaries to enable JIT-friendly code without sacrificing algorithm clarity or extensibility.

## Core Design Principles

**Immutability & PyTree Compatibility**: All state uses Flax `struct.dataclass` (immutable) to enable safe JIT compilation and transparent tracing. Configuration classes are marked `pytree_node=False` to avoid tracing static metadata.

**Static Resource Budgeting**: PRNG key allocation is computed once at initialization via `ResourceMap`, eliminating dynamic key splitting inside traced loops. All operators declare their RNG requirements upfront (`num_keys_per_atomic_operation`), enabling pre-allocated key buffers and deterministic allocation. This removes host-device synchronization as a bottleneck.

**Three-Tier Operator Architecture**: All genetic operators decompose into three layers—*atomic arithmetic* (Tier 1: pure per-genome logic), *noise generation* (Tier 2: RNG consumption), and *population-level vectorization* (Tier 3: nested vmap + output flattening). This separation enables XLA kernel fusion (Tier 1+2 merged into monolithic kernels) without sacrificing testability or reproducibility.

**Struct-of-Arrays (SoA) Design**: Genomes are immutable PyTrees where a single genome has shape `(d,)` and a batched population has shape `(N, d)`. This design makes JAX transformations (`vmap`, `jit`) transparent and enables efficient memory layouts on accelerators.

**Type-Safe Generics**: Generic type parameters (`BasePopulation[G]`, `BaseEvaluator[G, C, D]`) enforce compile-time compatibility between genome types, evaluators, and operators, reducing runtime type errors in heterogeneous algorithms.

## Architecture: Three-Level Hierarchy

MalthusJAX decomposes evolutionary computation into three independent levels, each with strict input/output contracts:

### Level 1: Core Primitives (Genomes, Populations, Fitness)

**Genomes**: Immutable, JAX-friendly representations for candidate solutions.

- `RealGenome` / `RealPopulation[RealGenome]` — continuous real-valued vectors with configurable bounds and dtype
- `BinaryGenome` / `BinaryPopulation[BinaryGenome]` — bit strings with Hamming distance and efficient bit operations
- `CategoricalGenome` — discrete choice vectors for multi-choice problems
- `LinearGenome` — variable-length linear programs for genetic programming

**Key Properties**:
- **Struct-of-Arrays (SoA)**: Single genome has shape `(d,)`, population batches to `(N, d)` via immutable `BasePopulation[G]` container
- **Distance Metrics**: Polymorphic `distance(other: BaseGenome, metric: str) -> Numeric` enables diversity computation and analysis
- **PyTree Compatible**: All genomes are Flax PyTrees; populations are batched PyTrees with leading dimension `N`

**Populations**: Generic containers `BasePopulation[G]` provide:

- Immutable slicing and indexing operations
- Vectorized distance matrix computation for diversity analysis
- Configuration-based initialization via `init_random(key, config, size)`

**Fitness Evaluators**: Pure, batched evaluation functions with three interface levels:

1. **Per-Individual**: `evaluate(genome: G) -> Numeric` — scalar fitness for one genome
2. **Batched Population**: `evaluate_population(pop: BasePopulation[G]) -> BasePopulation[G]` — automatic `jax.vmap` batching
3. **Tensor Interface**: `get_tensor_fitness_function() -> (genes: Array[N, d]) -> Array[N]` — for external/third-party evaluators (BBOB, custom batch APIs)

**Evaluator Config Requirements**:
- All configs mandate `maximize: bool` for unambiguous fitness semantics (eliminate sign-flip errors)
- Mark static data with `pytree_node=False` to prevent tracing large arrays
- Tensor functions receive batched arrays in batch-first order `(N, ...)`

**Concrete Examples**:
- **Analytical**: `SphereEvaluator`, `GriewankEvaluator` — classic continuous test functions
- **Combinatorial**: `KnapsackEvaluator` — discrete optimization with constraint penalties
- **Adapters**: `BBOBEvaluator` — high-performance wrapper for evosax BBOB problems
- **Specialized**: `LinearGPEvaluator` — genetic programming with symbiotic instruction selection

---

### Level 2: Genetic Operators

All operators follow a unified **three-tier decomposition** and factory pattern with the `@struct.dataclass` callable interface:

```python
# Factory pattern: instantiate operator with hyperparameters, then call as a pure function
operator = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)
mutated_pop = operator(key, population, config)  # Direct callable interface
jitted_op = jax.jit(operator)  # JIT the entire operator
```

**Tier 1: Arithmetic Kernel** — Pure, deterministic operation on a single individual.
- `_mutate_one(genome, noise_data, config) -> genome`
- `_recombine_one(p1, p2, noise_data, config) -> genome`
- `_select(keys, fitness, config) -> indices`
- No Python control flow; use `jnp.where`, `jax.lax.select` for branching

**Tier 2: Noise Generation** — RNG consumption with deterministic entropy production.
- `_generate_noise(keys: Array[K, 2], config) -> noise_data` where `K = num_keys_per_atomic_operation`
- Returns shaped arrays matching `config.shape` (or tuples for multi-component operations)
- Explicit dtype handling: `dtype=config.dtype` to prevent implicit promotion

**Tier 3: Population-Level Vectorization** — Nested `jax.vmap` orchestration.
- `BaseMutation.__call__`, `BaseCrossover.__call__`, `BaseSelection.__call__` handle key reshaping, nested vmap, output flattening
- Inherited by all subclasses; developers implement only Tiers 1 and 2
- Automatically handles offspring flattening to **offspring-major** order

**XLA Kernel Fusion**: Each operator's `_fused()` method combines Tier 2 (RNG) + Tier 1 (arithmetic) into a single expression tree. XLA compiles this as a monolithic kernel, minimizing per-invocation device overhead.

**Key Derivation**: Each operator declares RNG needs via `num_keys_per_atomic_operation`:
- Selection: depends on algorithm (0 for elite, ≥1 for tournament/roulette)
- Crossover: typically 1 (uniform mask) or 2+ (blend parameters)
- Mutation: typically 2 (mask + noise)

**Output Convention**: Nested vmap produces intermediate shape `(num_parents_or_pairs, num_offspring, *shape)`. Final transpose + reshape flattens to `(num_offspring * num_parents_or_pairs, *shape)`, placing all offspring 0 first, then offspring 1, etc. This **offspring-major** order is enforced by `spawn_offspring()` for consistency across all operators.

**Available Operators**:

| Category | Operators | Notes |
|----------|-----------|-------|
| **Selection** | Tournament, Roulette Wheel, Elite Pool | Per-individual & bulk injection modes |
| **Real Crossover** | Uniform, Blend (BLX-α), Simulated Binary (SBX), Binomial | All support configurable offspring count |
| **Binary Crossover** | Uniform, Single-Point | Efficient bit-wise operations |
| **Real Mutation** | Gaussian, Ball, Polynomial | Adaptive & scheduled variants available |
| **Binary Mutation** | Bit-Flip, Scramble, Swap | Complement/toggle operations |

---

### Level 3: Evolution Engines

**Abstraction**: `AbstractEngine` defines the evolutionary loop contract.
- `init_state(key) -> state` — one-time initialization with compilation setup
- `step(state) -> (new_state, output)` — single generation execution
- `run(state, num_gens=...) -> (final_state, history)` — wrapper using `jax.lax.scan` for loop fusion

**Concrete Implementation**: `GeneticEngine` — standard genetic algorithm with resource budgeting and flexible composition.

**Initialization Phase (`init_state`)**: 

1. **Compute ResourceMap**: Queries each operator for RNG requirements via `num_keys()`. Computes total budget: `sum(operator.num_keys(pop_size))`. Produces per-operator slices for deterministic key allocation.
2. **Freeze Operators**: Calls `set_input_length()` on selection, crossover, mutation to lock population size. This enables XLA to compile a single kernel per operator per generation (no recompilation).
3. **Initialize Population**: Creates initial genomes via `init_random()`. Evaluates fitness via `BaseEvaluator.evaluate_population()` with `jax.vmap`.
4. **Create State**: Bundles population, best genome, generation counter, RNG key, ResourceMap, frozen operator state into immutable `GeneticEvolutionState`.

Result: All compilation happens once. `step()` reuses kernels across all generations with zero compilation overhead.

**Evolution Step (`step`)** — Six phases (methods decorated with `@traceable` for HLO profiling):

1. **`_allocate_entropy()`** — Slice pre-derived RNG keys for each operator via `ResourceMap.get_keys()`. Key derivation strategy (SPLIT vs FOLD) controlled by `GeneticEngineParams.key_derivation`.
2. **`_selection_phase()`** — Extract elites (top `elitism` individuals). Select parents via selection operator. Returns parent indices.
3. **`_reproduction_phase()`** — Gather parents into subpopulations. Apply crossover to produce offspring. Apply mutation to offspring. Frozen operator sizes ensure stable key buffers.
4. **`_merge()`** — Combine elites + top mutants to fill population of size `pop_size`. Trivial population assembly with no fitness re-evaluation.
5. **`_evaluate()`** — Compute fitness on merged population via `BaseEvaluator.evaluate_population()`.
6. **`_update_hof()`** — Track best genome and increment stagnation counter. Reset stagnation when new best found.

**Resource Allocation Mechanics**:
- Total keys per generation: `sum(operator.num_keys(pop_size))` across all operators
- Keys pre-derived once per generation via `ResourceMap.get_keys()` using strategy (SPLIT or FOLD)
- Each operator receives a fixed-size key slice (no dynamic allocation inside traced regions)
- Eliminates host-device synchronization from RNG management

**RNG Derivation Strategies** (controlled by `GeneticEngineParams.key_derivation`):

| Strategy | Method | Trade-off |
|----------|--------|-----------|
| **SPLIT** | Sequential `jax.random.split()` | Uncorrelated key streams; bottleneck in parallelization |
| **FOLD** | Parallel `jax.random.fold_in()` | Deterministic counter-advancing; better for multi-device |

Both produce statistically equivalent results; choice affects RNG topology and reproducibility semantics.

**Extensibility**: Use template method pattern to override selection/reproduction/evaluation logic with **full access to evolution state**.

All component methods receive `(key, state, params)` and have access to:
- `state.population` — current individuals and fitness
- `state.generation` — current generation number
- `state.stagnation_counter` — generations without improvement
- `state.best_fitness`, `state.best_genome` — evolution progress

Override methods:
- **`_select_parents(key, state, params)`** → returns `BasePopulation` of selected parents
- **`_select_elites(key, state, params)`** → returns elite genes (ArrayTree, not Population)
- **`_create_offspring(key, parents, state, params)`** → returns offspring genes for crossover + mutation

**Adaptive Algorithms Example**: Stagnation-aware mutation rate:
```python
@struct.dataclass
class AdaptiveEngine(GeneticEngine):
    base_mutation_rate: float = struct.field(default=0.01, pytree_node=False)
    
    def _create_offspring(self, key, parents, state, params):
        # Increase mutation when stagnating
        adaptive_rate = self.base_mutation_rate * (1 + 0.2 * state.stagnation_counter)
        adaptive_mutation = self.mutation.replace(mutation_rate=adaptive_rate)
        return adaptive_mutation(key, parents.genes, params)
```

**Mutation Strength Scheduling**: Optional time-dependent schedule applied each generation:
```python
def schedule(generation: int, num_generations: int) -> float:
    return 0.5 * (1.0 - generation / num_generations)  # linear decay

engine_params = GeneticEngineParams(..., mutation_strength_schedule=schedule)
```

**Ask/Tell Interface**: For decoupled evaluation (external simulators, distributed evaluation):
```python
engine_with_entropy, population = engine.ask(state)  # Get population to evaluate
new_population = external_evaluator(population)      # Evaluate outside JAX
next_state = engine.tell(state, new_population)      # Complete evolution step
```

**GSPMD Sharding**: `ShardingManager` provides multi-device placement specs:
- Population matrices: `NamedSharding(mesh, P('batch', None))` — batch-sharded
- Fitness vectors: `NamedSharding(mesh, P('batch'))` — batch-sharded
- Scalars: `NamedSharding(mesh, P())` — replicated across devices

## Installation

```bash
# Clone the repository
git clone https://github.com/LeonardoDiCaterina/MalthusJAX.git
cd MalthusJAX

# Install with development dependencies
make install-dev
```

## Extensibility: Full-Access Component Architecture

MalthusJAX engines use the **Template Method pattern** with **full state visibility**, enabling sophisticated adaptive algorithms without breaking JIT compilation. Every overridable method receives the complete evolution state, allowing context-dependent operator behavior.

**Why Full Access Matters**: 
- Adaptive algorithms (mutation rate scheduling based on convergence)
- Quality-diversity objectives (access to population diversity metrics)
- Multi-objective optimization (compute auxiliary fitness from full state)
- Custom evolution strategies (age-layered populations, niching, migration)

All while maintaining:
- Full JIT compilation of the evolution loop
- Static RNG allocation (no dynamic key splits)
- Deterministic reproducibility
- Type safety via generics

## Example Usage

```python
import jax.random as jar
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams

# Setup random key
from malthusjax.core import create_key, PRNGImpl
# Prefer explicit PRNG implementation control; PHILOX is GPU-optimized
key = create_key(42, impl=PRNGImpl.PHILOX)

# 1. Define genome: real-valued vectors in [-5, 5]
genome_config = RealGenomeConfig(length=10, bounds=(-5.0, 5.0))

# 2. Setup fitness evaluator (BBOB benchmark function)
bbob_config = BBOBConfig(fn_name="sphere", num_dims=10, maximize=False)
evaluator = BBOBEvaluator.create(bbob_config)

# 3. Configure engine parameters
engine_params = GeneticEngineParams(
    pop_size=100,
    elitism=5,
    num_generations=50
)

# 4. Create genetic operators
selection = ElitePoolSelection(num_selections=100, elite_k=10)
crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
mutation = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)

# 5. Assemble engine
engine = GeneticEngine(
    engine_params=engine_params,
    genome_config=genome_config,
    evaluator=evaluator,
    selection=selection,
    crossover=crossover,
    mutation=mutation
)

# 6. Initialize and run evolution
state = engine.init_state(key)
final_state, history = engine.run(state)
print(f"Best fitness: {final_state.best_fitness}")
```

## Implementation Details

### Level 1: Genomes & Fitness

**Genomes** implement immutable PyTree structures via Flax `struct.dataclass`:
- All fields are read-only; mutation uses `.replace(field=new_value)` pattern
- Distance computation is polymorphic: `distance(other: BaseGenome, metric: str) -> Numeric`
- Single-individual math (e.g., `add_noise`) are pure functions suitable for `jax.vmap`

**Populations** batch genomes into SoA structure:
- `BasePopulation[G].genes` is a batched genome PyTree with leading dimension `(N, ...)`
- Indexing and slicing return sub-populations with correct type preservation
- `init_random(key, config, size)` creates initial populations with JIT-compatible initialization

**Fitness Evaluators** use composition to avoid code duplication:
- Per-individual `evaluate(genome)` implements the core logic
- `evaluate_population(pop)` uses `jax.vmap(self.evaluate)` for batched evaluation
- Tensor interface `get_tensor_fitness_function()` bridges to external libraries (BBOB, JAX-free code)
- Return type `chex.Numeric` (not Python float) ensures JAX tracer compatibility

**Evaluator Config Pattern**:
```python
@struct.dataclass(frozen=True)
class CustomEvaluatorConfig(BaseEvaluatorConfig):
    maximize: bool = struct.field(pytree_node=False)  # Explicit direction
    custom_param: float = struct.field(pytree_node=False)  # Config only, not traced
```

### Level 2: Operators (Three-Tier Architecture)

**Why Three Tiers Exist**:
1. Tier 1 (arithmetic) is pure and testable in isolation
2. Tier 2 (noise) allows deterministic replay of randomness by freezing the seed
3. Tier 3 (vmap) orchestrates batching without touching algorithm logic

**Tier 1: Arithmetic Kernel** — Implement only this tier; Tiers 2/3 are inherited:

```python
def _mutate_one(self, genome: RealGenome, noise_data: chex.Array, config: RealGenomeConfig) -> RealGenome:
    """Pure function: genome + noise -> mutated genome."""
    mutated = genome.values + noise_data
    if self.clip:
        mutated = jnp.clip(mutated, config.bounds[0], config.bounds[1])
    return genome.replace(values=mutated)
```

**Tier 2: Noise Generation** — Declare RNG requirements and produce entropy:

```python
@property
def num_keys_per_atomic_operation(self) -> int:
    """Each mutation needs: 1 key for mask, 1 key for noise = 2 total."""
    return 2

def _generate_noise(self, keys: chex.Array, config: RealGenomeConfig) -> chex.Array:
    """Consume exactly 2 keys, produce noise shaped to config.shape."""
    k_mask, k_noise = keys[0], keys[1]
    mask = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=config.shape)
    noise = jax.random.normal(k_noise, shape=config.shape, dtype=config.dtype)
    return noise * self.mutation_strength * mask.astype(config.dtype)
```

**Tier 3: Vectorization** — Automatically handled by `BaseMutation.__call__`:
1. Reshape keys: `(num_keys,) -> (pop_size, num_offspring, keys_per_op, 2)`
2. Outer vmap: iterate over pop_size parents
3. Inner vmap: iterate over num_offspring per parent
4. Apply `_mutate_fused()` (Tier 1 + 2 combined for XLA fusion)
5. Flatten: `(pop_size, num_offspring, ...) -> (num_offspring * pop_size, ...)`

**Output Convention**: Offspring-major order is critical for downstream processing:
```
Input:  4 parents, 2 offspring each  -> shape (4, 2, d)
Output: 8 offspring, ordered as     [offs0_par0, offs0_par1, offs0_par2, offs0_par3,
                                     offs1_par0, offs1_par1, offs1_par2, offs1_par3]
                                    -> shape (8, d)
```

**Static Key Budgeting**: The ResourceMap pre-computes exact RNG needs:
- Calls `operator.num_keys(input_shape)` once at initialization
- Slices pre-allocated key buffer: no dynamic splits inside evolution loop
- Enables single JIT compilation with guaranteed buffer sizes

### Level 3: Engines (Execution & Composition)

**Initialization (`init_state`)** is where compilation setup happens:
```python
# Pseudo-code showing key phases
def init_state(self, key):
    # 1. ResourceMap computes total RNG budget
    rmap = compute_resource_map(self.selection, self.crossover, self.mutation, pop_size)
    
    # 2. Freeze operator sizes (enables XLA single-kernel compilation)
    frozen_sel = self.selection.set_input_length(pop_size)
    frozen_cross = self.crossover.set_input_length(num_pairs)
    frozen_mut = self.mutation.set_input_length(pop_size)
    
    # 3. Initialize population & evaluate
    pop = initialize_population(key, self.genome_config, pop_size)
    pop = self.evaluator.evaluate_population(pop)
    
    # 4. Create state (all immutable)
    return GeneticEvolutionState(
        population=pop,
        rng_key=key,
        resource_map=rmap,
        operators=OperatorState(frozen_sel, frozen_cross, frozen_mut),
        ...
    )
```

**Evolution Loop (`step`)** never recompiles:
```python
def step(self, state):
    # All operators already frozen; keys pre-allocated
    # Inner loop never touches JIT boundaries
    key = state.rng_key
    
    # Phase 1: allocate entropy (deterministic slicing)
    all_keys = state.resource_map.get_keys(key)
    
    # Phase 2-6: standard genetic algorithm phases
    # All use frozen operators with pre-allocated buffers
    ...
    
    return new_state, metrics
```

**Extensibility via Template Methods**: Override only what you need:

```python
@struct.dataclass
class CustomEngine(GeneticEngine):
    def _select_parents(self, key, state, params):
        # Full access to state enables context-dependent logic
        # Example: diversity-aware selection
        pop = state.population
        dist_matrix = pop.distance_matrix(metric="euclidean")
        diversity_bonus = compute_diversity_bonus(dist_matrix)
        combined_fitness = pop.fitness + 0.3 * diversity_bonus
        
        indices = self.selection(key, combined_fitness, params)
        return pop[indices]
```

### Static Metadata & JIT Compatibility

**Mark Static Fields**:
```python
@struct.dataclass
class MyOperator(BaseMutation):
    num_offspring: int = struct.field(pytree_node=False)  # Static
    mutation_rate: float = struct.field(pytree_node=False)  # Static
    dynamic_param: chex.Array = struct.field(pytree_node=True)  # Traced
```

**Why This Matters**:
- Static fields are never traced; JAX receives concrete values at compile-time
- Enables operator-specific optimizations (buffer pre-allocation, loop unrolling)
- Zero overhead for configuration parameters
- One JIT compilation per operator per population size

---

## Composer: Config-Driven Experiment Orchestration

The **Composer** sits at Level 3.5 of the architecture and provides a product-first API for assembling, running, and comparing evolutionary experiments. It unifies the MalthusJAX engine pipeline with evosax interoperability, multi-seed benchmarking, and TOML-based configuration — all through a single entry point.

### Quick Start

```python
from malthusjax.composer import Composer

composer = Composer.create_default()

# One-call experiment: specify operators as string specs
result = composer.quick_run(
    fitness="sphere:dim=10",
    selection="tournament:num_selections=25,tournament_size=3",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1",
    pop_size=100,
    generations=200,
    seeds=(42, 43, 44),
)

print(result.aggregated_summary())
```

### Architecture Overview

The Composer brings together four subsystems:

| Component | Module | Role |
|-----------|--------|------|
| **OperatorCatalog** | `composer/catalog.py` | Resolves operator string specs → instances |
| **EngineRegistry** | `composer/engine_catalog.py` | Resolves engine string specs → configured engines |
| **EvosaxAdapter** | `composer/evosax_adapter.py` | Wraps evosax ask/tell strategies for interop |
| **BenchmarkRunner** | `benchmarking/runner.py` | Multi-seed execution with artifact I/O |

Data flow:

```
String specs  →  OperatorCatalog  →  operator instances  ─┐
                 EngineRegistry   →  engine factory       ─┼→ BenchmarkRunner → ExperimentResult
                                                           │
TOML config   →  load_experiment_config  → shared+pipeline ┘
```

### String Specification Format

All operators and engines use a unified spec format: `"name:param1=value1,param2=value2"`

**Operator Specs** (resolved by `OperatorCatalog`):

| Category | Examples |
|----------|----------|
| **Selection** | `"tournament:num_selections=50,tournament_size=3"`, `"roulette"`, `"elite_pool"` |
| **Crossover** | `"blend:alpha=0.5"`, `"simulated_binary:eta=2.0"`, `"binomial"`, `"uniform_real"` |
| **Mutation** | `"gaussian:mutation_rate=0.1"`, `"polynomial:mutation_rate=0.05"`, `"ball"` |
| **Fitness** | `"sphere:dim=10"`, `"bbob:fn_name=rastrigin,dim=5"`, `"griewank"`, `"knapsack"` |

**Engine Specs** (resolved by `EngineRegistry`):

| Engine | Spec | Description |
|--------|------|-------------|
| **GA** | `"ga"`, `"ga:pop_size=200,elitism=4"` | Standard genetic algorithm (`GeneticEngine`) |

Custom engines can be registered at runtime:

```python
from malthusjax.composer import EngineRegistry

reg = EngineRegistry()
reg.register("nsga2", my_nsga2_factory, defaults={"pop_size": 100})
print(reg.list_available())  # ['ga', 'nsga2']
```

### Composer.quick_run()

The primary entry point. Supports two backends:

**MalthusJAX backend** (default) — builds a `GeneticEngine` from operator specs:

```python
result = composer.quick_run(
    backend="malthusjax",       # default
    engine_type="ga",           # resolved via EngineRegistry
    fitness="sphere:dim=10",
    selection="tournament:num_selections=25,tournament_size=3",
    crossover="blend:alpha=0.5",
    mutation="gaussian:mutation_rate=0.1",
    pop_size=100,
    generations=200,
    genome_length=10,
    bounds=(-5.0, 5.0),
    elitism=2,
    prng_impl="threefry",       # or "philox" for GPU
    seeds=(42, 43, 44),
)
```

**Evosax backend** — wraps evosax population-based strategies (ask/tell):

```python
result = composer.quick_run(
    backend="evosax",
    evosax_strategy="SimpleGA",   # or "MR15_GA", "DifferentialEvolution"
    fitness="sphere:dim=10",
    pop_size=100,
    generations=200,
)
```

Both backends return an `ExperimentResult` with identical structure, enabling direct cross-framework comparison.

### Composer.compare()

Run multiple pipelines with aligned seeds and optional shared initial population:

```python
cmp = composer.compare(
    pipelines={
        "Blend+Gaussian": dict(
            crossover="blend:alpha=0.5",
            mutation="gaussian:mutation_rate=0.1",
        ),
        "SBX+Polynomial": dict(
            crossover="simulated_binary:eta=2",
            mutation="polynomial:mutation_rate=0.1",
        ),
        "Evosax SimpleGA": dict(
            backend="evosax",
            evosax_strategy="SimpleGA",
        ),
    },
    # Shared across all pipelines:
    fitness="sphere:dim=10",
    pop_size=50,
    generations=100,
    seeds=(42, 43),
)

# Comparison analysis
cmp.summary_table()        # Per-pipeline aggregated metrics
cmp.convergence_data()     # Per-generation history for plotting
cmp.plot_convergence()     # Matplotlib overlay plot
```

`ComparisonResult` automatically normalises fitness values across backends (MalthusJAX uses a maximisation convention internally; evosax uses minimisation). The `negate_map` ensures all displayed values follow a unified "lower is better" convention.

### TOML-Driven Experiments

Define experiments declaratively in TOML and run with a single call:

```toml
# experiment.toml
[experiment]
name = "crossover_comparison"
output_dir = "results/crossover_comparison"

[experiment.shared]
fitness       = "sphere:dim=10"
selection     = "tournament:num_selections=25,tournament_size=3"
mutation      = "gaussian:mutation_rate=0.1"
engine_type   = "ga"
pop_size      = 50
generations   = 100
genome_length = 10
bounds        = [-5.0, 5.0]
seeds         = [42, 43, 44]
prng_impl     = "threefry"

[pipelines.blend_ga]
backend   = "malthusjax"
crossover = "blend:alpha=0.5"

[pipelines.sbx_ga]
backend   = "malthusjax"
crossover = "simulated_binary:eta=2.0"

[pipelines.evosax_simple]
backend         = "evosax"
evosax_strategy = "SimpleGA"
```

```python
result = Composer.from_toml("experiment.toml")
result.plot_convergence()
```

`from_toml()` merges `[experiment.shared]` defaults with per-pipeline overrides, generates a shared initial population for fair comparison, and returns a `ComparisonResult`.

### Result Objects

The Composer returns structured result objects from the benchmarking module:

| Class | Contents |
|-------|----------|
| `RunResult` | Single-seed data: status, metrics, per-generation history, timings |
| `ExperimentResult` | Multi-seed: list of `RunResult`, aggregated summary, combined history |
| `ComparisonResult` | Multi-pipeline: dict of `ExperimentResult`, summary table, convergence plotting |

```python
# ExperimentResult API
result.aggregated_summary()    # {metric: {mean, median, stdev}}
result.combined_history()      # Flattened history across all seeds
result.canonical_summary       # First run's metrics

# ComparisonResult API
cmp.summary_table()            # {pipeline: {metric: value}}
cmp.convergence_data()         # {pipeline: [{generation, best_fitness, ...}]}
cmp.plot_convergence()         # Matplotlib convergence overlay
cmp.names                      # Pipeline names in insertion order
```

Artifacts (JSON summaries, CSV histories) are written automatically to the output directory when `write_artifacts=True`.

### EngineRegistry

The `EngineRegistry` provides a catalog pattern for engine types, paralleling the `OperatorCatalog` for operators. Engines self-register at import time via `engine/__init__.py`.

```python
from malthusjax.composer import EngineRegistry

reg = EngineRegistry()

# Introspection
reg.list_available()           # ['ga']
reg.get_help("ga")             # Docstring + defaults
reg.parse_spec("ga:pop_size=200")  # ('ga', {'pop_size': 200})

# Direct instantiation (requires operator instances)
engine = reg.get(
    "ga:pop_size=100",
    evaluator=sphere_eval,
    selection=tournament,
    crossover=blend,
    mutation=gaussian,
    generations=200,
)

# Runtime registration
def my_nsga2_factory(evaluator, selection, crossover, mutation, **kw):
    return MyNSGA2Engine(evaluator=evaluator, ...)

reg.register("nsga2", my_nsga2_factory, defaults={"pop_size": 100})
```

The `engine_type` parameter in `quick_run()` routes through the `EngineRegistry`:

```python
# Use registered engine type
result = composer.quick_run(
    engine_type="ga:elitism=4",
    fitness="sphere:dim=10",
    ...
)
```

---

## 📚 Layer-Specific Technical Documentation

Each layer provides detailed technical specifications for developers implementing or extending components:

### Level 1: Genomes & Fitness
- [Genome Architecture](src/malthusjax/core/genome/README.md) — Struct-of-Arrays design, extension patterns, distance metrics
- [Fitness Evaluators](src/malthusjax/core/fitness/README.md) — Per-individual & batch evaluation, tensor interfaces, config patterns

### Level 2: Operators  
- [Selection Operators](src/malthusjax/operators/selection/README.md) — Static RNG budgeting, index generation, GSPMD sharding
- [Mutation Operators](src/malthusjax/operators/mutation/README.md) — Three-tier architecture (Tier 1/2/3), XLA fusion, offspring-major flattening
- [Crossover Operators](src/malthusjax/operators/crossover/README.md) — Recombination kernels, mask conventions, nested vmap orchestration

### Level 3: Engines
- [Engine Architecture](src/malthusjax/engine/README.md) — Execution model, ResourceMap contract, key derivation strategies (SPLIT vs FOLD), init-phase compilation

### Level 3.5: Composer & Benchmarking
- [Composer Demo](examples/_DEMO_COMPOSER/) — Interactive notebooks, TOML config examples, quick reference
- [Benchmarking Infrastructure](src/malthusjax/benchmarking/) — Multi-seed runner, result objects, artifact I/O

---

## 🔬 Benchmarking & Performance Analysis

MalthusJAX includes comprehensive benchmarking tools for dispatch overhead, parameter tuning, and algorithm comparison.

### 1. JAX Dispatch Timing Analysis

Analyze JIT compilation costs and per-operator timing:

```bash
# Quick test (single task, small dimensions)
python benchmarks/cli_dispatch.py benchmarks/dispatch_timing.toml --quick

# Full comparative analysis (MalthusJAX vs Evosax)
python benchmarks/cli_dispatch.py benchmarks/dispatch_timing.toml --framework both
```

**Key Metrics**:
- **Cold compile time**: Initial JIT compilation (includes tracing + XLA compilation)
- **Warm dispatch time**: Amortized per-step cost after first compilation
- **Unroll factor impact**: How many steps needed to amortize compilation overhead
- **Per-operator breakdown**: Individual timing for selection, crossover, mutation, evaluation

**Output**: CSV files, operator timing breakdown, optimization recommendations saved to `results/dispatch_timing/`

### 2. Fitness Landscape Benchmark

Evolutionary algorithm performance on BBOB test functions:

```bash
# Quick smoke test
python benchmarks/cli_fitness.py benchmarks/fitness_tuning_quick.toml --quick

# Full tuning benchmark with hyperparameter sweeps
python benchmarks/cli_fitness.py benchmarks/fitness_tuning.toml --plot
```

### 3. Configuration Generation

Generate benchmark configs with warp-aligned population sizes for GPU stress testing:

```bash
# Generate GECCO benchmark configs for all BBOB tasks
python generate_configs.py

# Creates configs for: sphere, rosenbrock, ellipsoidal, rastrigin, schaffers_f7
# with dimensions [10, 50, 100] and population sizes [32→1025]
```

### Makefile Shortcuts

```bash
make install-dev    # Install with dev dependencies
make check-all      # Run all quality checks (lint, format, type-check, test)
make test           # Run pytest with coverage (minimum 80%)
make lint           # Ruff linting
make format         # Ruff code formatting
make type-check     # mypy with strict settings
```

---

## Development & Contributing

**Quality Standards**:
- **Test Coverage**: Minimum 80% coverage enforced via pytest-cov
- **Type Safety**: Full mypy strict mode compliance (no `Any` escapes without justification)
- **Code Style**: Ruff for linting and formatting (replaces black, flake8, isort)
- **Pre-commit**: All checks must pass before merge (`make check-all`)

**Testing Strategy**:
- Test structure mirrors `src/` in `tests/`
- Shared fixtures in `conftest.py` (random keys, sample genomes)
- Mark slow tests with `@pytest.mark.slow`
- JAX random keys from `jr.PRNGKey(42)` fixture for determinism

**Implementation Checklist** (operators, evaluators, engines):
- [ ] Implement as `@struct.dataclass` with `pytree_node=False` for static fields
- [ ] Declare RNG requirements via `num_keys_per_atomic_operation`
- [ ] Use `jax.lax.select` for branching (not Python `if` in traced regions)
- [ ] Return `chex.Numeric` (not Python float) from traced functions
- [ ] Add unit tests with shape/dtype validation
- [ ] Document config fields with type annotations and descriptions

---

## 🚧 In-Progress Components

These features are currently under development:

- **Visualization Tools**: HLO graph profiling, evolution trace analysis, population diversity visualization
- **Island-Model Evolution**: Multi-population migration topologies via Composer pipelines
- **Additional Engine Types**: NSGA-II, CMA-ES, and other strategies in the EngineRegistry

See [docs/](docs/) for design documents and progress tracking.

---
## License

[MIT License](LICENSE)