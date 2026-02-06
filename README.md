# MalthusJAX

**A JAX-Based Framework for Evolutionary Computation**

MalthusJAX is an evolutionary computation framework built on JAX, designed with a modular 3-level architecture that emphasizes composability, type safety, and JIT compilation compatibility. The framework provides a principled approach to evolutionary algorithm design through strict separation of concerns and functional programming patterns.

## Design Philosophy

MalthusJAX follows three core principles:

1. **Compositional Architecture**: Components at each level are independently developed and compose cleanly through well-defined interfaces
2. **Functional Purity**: All operations are pure functions operating on immutable data structures, enabling reliable JIT compilation
3. **Type Safety**: Generic type parameters ensure compile-time verification of component compatibility

## Architecture Overview

### Level 1: Core Components
- **Genome Representations**: Immutable genome types (`BinaryGenome`, `RealGenome`, `CategoricalGenome`, `LinearGenome`) implemented as Flax `struct.dataclass` for JAX compatibility
- **Population Containers**: Type-safe population wrappers (`BasePopulation[G]`) providing vectorized operations
- **Fitness Evaluators**: Generic evaluators (`BaseEvaluator[G, C, D]`) with automatic vectorization via `jax.vmap()`

### Level 2: Genetic Operators
- **Selection Operators**: Parent selection strategies (`TournamentSelection`, `RouletteWheelSelection`)
- **Crossover Operators**: Recombination operators with batch-first output (`UniformCrossover`, `SimulatedBinaryCrossover`)
- **Mutation Operators**: Variation operators supporting multiple offspring (`BitFlipMutation`, `GaussianMutation`)

All operators follow a unified factory pattern using `@struct.dataclass` with `__call__` methods, enabling direct JIT compilation.

### Level 3: Evolution Engines
- **Abstract Engine Interface**: `AbstractEngine` defines the evolutionary loop contract
- **Genetic Engine Implementation**: `GeneticEngine` provides a standard genetic algorithm with pluggable components
- **Template Method Pattern**: Engines expose overridable methods (`_select_parents`, `_select_elites`, `_create_offspring`) for custom evolutionary strategies
- **State Management**: Immutable `AbstractEvolutionState` enables JIT compilation via `jax.lax.scan`
- **Extensibility**: Subclass engines to incorporate custom selection strategies (e.g., diversity preservation, novelty search)

## Installation

```bash
# Clone the repository
git clone https://github.com/LeonardoDiCaterina/MalthusJAX.git
cd MalthusJAX

# Install with development dependencies
make install-dev
```

## Extensible Architecture

Unlike rigid GA libraries that hide internal state, MalthusJAX uses a **"Full Access" component design** that gives you complete control over the evolutionary process. Every internal method in `GeneticEngine` receives the full evolution state, enabling you to:

- **Adaptive Algorithms**: Modify operator behavior based on convergence metrics (e.g., increase mutation rate when `state.stagnation_counter` is high)
- **Multi-Objective Selection**: Access `state.population` directly to compute auxiliary metrics (diversity, novelty, age) and combine them with fitness
- **Stateful Evolution**: Track custom metrics across generations (e.g., lineage, speciation, niching) by subclassing `AbstractEvolutionState`
- **Context-Aware Operators**: Make operator decisions based on `state.generation`, `state.best_fitness`, or population statistics

### Full Access Method Signature

All component methods follow a unified, strongly-typed signature. Below are the primary methods you are expected to override and their exact signatures used throughout the codebase:

```python
from typing import Tuple
import jax
import jax.numpy as jnp

# Parent selection: returns a `BasePopulation` (selected parents)
def _select_parents(
    self,
    key: jax.Array,                      # PRNG key (jax.random.PRNGKey)
    state: AbstractEvolutionState,      # Full evolution state (population, metrics)
    params: GeneticEngineParams         # Engine configuration (pop_size, elitism, ...)
) -> BasePopulation:
    """Select parents for reproduction.

    Full access to `state` enables computing auxiliary metrics (distance matrices,
    novelty, age) and implementing adaptive behaviour based on `state.generation`
    or `state.stagnation_counter`.
    """
    ...

# Elite selection: returns genes (ArrayTree) representing elite individuals
def _select_elites(
    self,
    key: jax.Array,
    state: AbstractEvolutionState,
    params: GeneticEngineParams
) -> jax.Array:
    """Return elite genes (not a population object)."""
    ...

# Offspring creation: returns genes (ArrayTree) for offspring after crossover/mutation
def _create_offspring(
    self,
    key: jax.Array,
    parents: BasePopulation,
    state: AbstractEvolutionState,
    params: GeneticEngineParams
) -> jax.Array:
    """Create offspring from parents; supports adaptive operators using `state`."""
    ...
```

This architecture enables **rapid prototyping** of evolutionary strategies without rewriting the main JIT-compiled evolution loop. Override just the methods you need—the framework handles state management, JIT compilation, and performance optimization.

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
key = jar.PRNGKey(42)

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

The 3-level architecture enforces separation of concerns:

**Level 1 (Core)**: Genome types implement immutable data structures using Flax `struct.dataclass`. Population containers provide vectorized operations via JAX transformations. Fitness evaluators define pure functions with automatic batching through `jax.vmap()`.

**Level 2 (Operators)**: All operators are stateless callables implementing standard interfaces (`BaseMutation`, `BaseCrossover`, `BaseSelection`). The batch-first convention ensures outputs have shape `(num_offspring, ...genome_shape)`, enabling efficient vectorization.

**Level 3 (Engines)**: The `AbstractEngine` interface defines `init_state()`, `step()`, and `run()` methods. Evolution loops use `jax.lax.scan` for efficient iteration with compile-time loop fusion. State objects are immutable PyTrees containing population, generation counter, and algorithm-specific data.

### Engine Extensibility: Full Access Component Methods

MalthusJAX engines use the **Template Method pattern** with **Full Access signatures**, giving every component method complete visibility into the evolution state. This enables sophisticated adaptive algorithms without breaking JIT compilation.

**Key Overridable Methods** (all receive `key`, `state`, `params`):
- **`_select_parents(key, state, params)`**: Customize parent selection with access to population, generation, stagnation
- **`_select_elites(key, state, params)`**: Control elite preservation with context-aware logic
- **`_create_offspring(key, parents, state, params)`**: Implement adaptive variation (e.g., mutation rate scheduling)
- **`_merge_and_evaluate(key, elites, offspring, state, params)`**: Custom population assembly and fitness evaluation

**Example: Diversity-Aware Selection with Full State Access**
```python
from flax import struct
from malthusjax.engine.genetic_fastengine import GeneticEngine

@struct.dataclass
class DiversityAwareEngine(GeneticEngine):
    diversity_weight: float = struct.field(default=0.3, pytree_node=False)
    
    def _select_parents(self, key, state, params):
        # Full access to state enables computing auxiliary metrics
        population = state.population
        
        # Compute crowding distance using distance matrix
        dist_matrix = population.distance_matrix(metric="euclidean")
        crowding = self._compute_crowding_scores(dist_matrix)
        
        # Combine fitness and diversity into selection criterion
        diversity_fitness = (
            (1 - self.diversity_weight) * population.fitness + 
            self.diversity_weight * crowding
        )
        
        # Use standard selection operator with diversity-aware fitness
        indices = self.selection(key, diversity_fitness).flatten()
        return population[indices]
```

**Example: Adaptive Mutation Based on Stagnation**
```python
from flax import struct
from malthusjax.engine.genetic_fastengine import GeneticEngine

@struct.dataclass
class AdaptiveMutationEngine(GeneticEngine):
    base_mutation_rate: float = struct.field(default=0.01, pytree_node=False)
    
    def _create_offspring(self, key, parents, state, params):
        # Increase mutation rate when evolution stagnates
        stagnation_factor = 1 + 0.2 * state.stagnation_counter
        adaptive_rate = self.base_mutation_rate * stagnation_factor
        
        # Create modified mutation operator with adaptive rate
        adaptive_mutation = self.mutation.replace(mutation_rate=adaptive_rate)
        
        # Apply adaptive mutation to parents
        offspring_genes = adaptive_mutation(key, parents.genes, params)
        
        # Construct offspring population
        from malthusjax.core.genome.real_genome import RealPopulation
        return RealPopulation(genes=offspring_genes)
```

This **Full Access architecture** enables researchers to experiment with:
- **Quality-Diversity algorithms** (MAP-Elites, Novelty Search)
- **Multi-objective optimization** (NSGA-II, SPEA2)
- **Adaptive parameter control** (self-adaptive mutation, learning rate schedules)
- **Age-layered population models** (ALPS)
- **Island models** with migration strategies

All while maintaining **full JIT compilation compatibility** and **functional purity**.

## Statically Allocated Entropy & Operator Design

MalthusJAX uses a **static entropy allocation** strategy to maximize JIT compilation efficiency. Rather than splitting random keys dynamically within operators (which breaks JIT-ability), all random keys are pre-allocated by a resource manager and passed directly to operators.

### How Operators Declare Key Requirements

Each operator declares exactly how many random keys it needs via the **`num_keys()` contract**:

```python
@struct.dataclass
class GaussianMutation(BaseMutation):
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Each mutation needs 2 keys: one for mask, one for noise."""
        return 2
    
    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """Total keys = Population × Offspring × Keys-per-op"""
        pop_size = input_shape[0]
        return pop_size * self.num_offspring * self.num_keys_per_atomic_operation
```

**Key calculation example:**
- Population size: 100
- Offspring per parent: 1
- Keys per atomic operation: 2
- **Total keys needed: 100 × 1 × 2 = 200 keys**

The Resource Allocator computes the maximum across all operators and splits a single PRNG key into the required number of independent keys upfront. This enables:
- ✅ Pure functional operations (no side effects)
- ✅ Full JIT compilation of the evolution loop
- ✅ Deterministic key allocation (no dynamic control flow)
- ✅ Zero overhead for key management

### RNG Derivation Strategies: User Control Over Key Generation

MalthusJAX provides **two RNG derivation strategies** for generating the static key budget. You can choose which strategy best fits your use case:

| Strategy | Method | Best For | Characteristics |
|----------|--------|----------|------------------|
| **SPLIT** (default) | `jax.random.split()` | Multi-device, distributed optimization | Independent key streams, reduced correlation, optimal for GPU farms |
| **FOLD** | `jax.random.fold_in()` | Reproducibility-focused, single-device | Deterministic counter-advancing sequences, seed-stable behavior |

**Example: Choosing a Strategy**
```python
from malthusjax.engine.resource_mapper import KeyDerivationStrategy
from malthusjax.engine.genetic_fastengine import GeneticEngineParams

# Use FOLD for strict reproducibility
engine_params = GeneticEngineParams(
    pop_size=100,
    num_generations=50,
    key_derivation=KeyDerivationStrategy.FOLD  # Deterministic sequences
)

# Or use SPLIT (default) for multi-device setups
engine_params = GeneticEngineParams(
    pop_size=100,
    num_generations=50,
    key_derivation=KeyDerivationStrategy.SPLIT  # Independent streams
)
```

Both strategies produce **statistically equivalent results**; the choice affects RNG stream topology and reproducibility semantics. For detailed information, see [Engine Architecture Documentation](src/malthusjax/engine/README.md#key-derivation-strategies).

### Ablation Operators: Benchmarking Key Allocation Overhead

To quantify the performance impact of static key allocation vs. dynamic splitting, MalthusJAX includes **ablation study decorators** (`@ablation_single_key_mutation`, `@ablation_single_key_crossover`) that reduce any operator to single-key allocation:

```python
from malthusjax.operators.base_ablation import ablation_single_key_mutation
from malthusjax.operators.mutation.real import GaussianMutation

# Wrap standard operator with ablation decorator
@ablation_single_key_mutation
class GaussianMutation_ablation(GaussianMutation):
    pass

# Use ablation operator for benchmarking
standard_op = GaussianMutation(num_offspring=1, mutation_rate=0.1)
ablation_op = GaussianMutation_ablation(num_offspring=1, mutation_rate=0.1)

# Both implement identical arithmetic, but differ in RNG topology:
# - standard_op.num_keys(100) → 200 (pre-allocated)
# - ablation_op.num_keys(100) → 1 (dynamic fold_in internally)
```

**Ablation decorator behavior:**
- **Standard operator**: `num_keys() = pop_size × offspring × keys_per_op` → static allocation via ResourceMap
- **Ablation operator**: `num_keys() = 1` → keys generated internally using `jax.random.fold_in()` on-the-fly

**Benchmark use case:**
```bash
# Compare ResourceMap pre-allocation vs. dynamic key generation
python benchmarks/cli_dispatch.py config.toml --framework malthus

# Standard results: Dispatch + Allocation + Operator overhead
# Ablation results: Dispatch + Dynamic-splitting + Operator overhead
# Difference = static allocation framework efficiency gain (or loss!)
```

**Available ablation decorators** in [src/malthusjax/operators/base_ablation.py](src/malthusjax/operators/base_ablation.py):
- `@ablation_single_key_mutation` — Convert any `BaseMutation` to single-key allocation
- `@ablation_single_key_crossover` — Convert any `BaseCrossover` to single-key allocation

For detailed ablation study methodology, see:
- [Mutation Operator Ablation Study Mode](src/malthusjax/operators/mutation/README.md#ablation-study-mode-)
- [Crossover Operator Ablation Study Mode](src/malthusjax/operators/crossover/README.md#ablation-study-mode-)
- [Engine Resource Mapping & Key Derivation](src/malthusjax/engine/README.md#resource-mapping--cascade-data-flow)

This enables **precise measurement of framework overhead vs. implementation benefit trade-offs**.

---

## 📚 Comprehensive Documentation

MalthusJAX provides detailed technical documentation for each layer:

### Level 3: Evolution Engines
- **[Engine Architecture & Execution Model](src/malthusjax/engine/README.md)** (417 lines)
  - 6-phase evolution step execution with named calls for HLO profiling
  - ResourceMap contract and static RNG budgeting
  - KeyDerivationStrategy (SPLIT vs FOLD) detailed explanation
  - Operator baking, scheduled mutation, ask/tell interface
  - GSPMD sharding for single/multi-device optimization
  - Extension points and custom engine development patterns

### Level 2: Genetic Operators
- **[Selection Operators](src/malthusjax/operators/selection/README.md)** — Parent selection strategies
  - Atomic logic separation and vectorized slicing patterns
  - Mode A vs Mode D (bulk injection) trade-offs
  - Developer checklist for implementing custom selection

- **[Mutation Operators](src/malthusjax/operators/mutation/README.md)** (Tier 1/2/3 architecture)
  - Three-tier design: arithmetic kernel → noise generation → vectorized wrapper
  - BaseMutation (per-individual) vs BaseMutation_injection (bulk) modes
  - ResourceMap integration and KeyDerivationStrategy impact
  - Ablation study decorators for performance benchmarking

- **[Crossover Operators](src/malthusjax/operators/crossover/README.md)** (Tier 1/2/3 architecture)
  - Recombination kernels and mask conventions
  - Mode A (per-pair sampling) vs Mode D (bulk injection)
  - Offspring-major flattening for consistency across modes
  - Ablation study methodology and benchmarking patterns

### Level 1: Core Components
- **Genomes**: Immutable genome representations with automatic vectorization
- **Fitness Evaluators**: Batch evaluation with VMAP, population-level metrics

### Benchmarking & Evaluation
- **[Dispatch Timing Analysis](benchmarks/)** — JAX dispatch overhead and operator profiling
- **[Fitness Landscape Analysis](benchmarks/)** — Hyperparameter tuning on BBOB functions

---

## Testing

```bash
# Run tests with coverage
pytest

# Run specific test categories
pytest -m integration   # Integration tests
pytest -m slow          # Slow/comprehensive tests
pytest tests/core/      # Test specific module
```

## Benchmarking

MalthusJAX includes comprehensive benchmarking tools to analyze performance, dispatch overhead, and algorithm behavior.

### Generating Benchmark Configurations

Use the config generator to create GECCO benchmark configurations with warp-aligned population sizes:

```bash
# Generate benchmark configs for all BBOB tasks
python generate_configs.py

# This creates configs/run_<task>.toml for:
# - sphere, rosenbrock, ellipsoidal_rotated, rastrigin, schaffers_f7
```

The generator creates configurations that test:
- **Warp boundary stress testing**: Population sizes 32→1025 (tests GPU resource utilization)
- **Multiple dimensions**: 10, 50, 100
- **Statistical significance**: 30 repeats with different seeds
- **Long evolution**: 2000 generations per trial

Generated configs automatically set output directories: `results/gecco_final/{task}/`

### 1. JAX Dispatch Timing Analysis

Analyze JAX dispatch overhead, JIT compilation costs, and per-operator timing:

```bash
# Quick test (single task, small dimensions)
python benchmarks/cli_dispatch.py benchmarks/dispatch_timing.toml --quick

# Full comparative analysis (MalthusJAX vs Evosax)
python benchmarks/cli_dispatch.py benchmarks/dispatch_timing.toml --framework both

# MalthusJAX only
python benchmarks/cli_dispatch.py benchmarks/dispatch_timing.toml --framework malthus

# Evosax analysis with specific strategy
python benchmarks/cli_dispatch.py benchmarks/dispatch_timing.toml --framework evosax --evosax-strategy SimpleGA

# Custom configuration
python benchmarks/cli_dispatch.py your_config.toml --framework both --evosax-strategy DifferentialEvolution
```

**Outputs:**
- CSV files with unroll factor analysis (dispatch amortization curves)
- Per-operator timing breakdown (ask, evaluate, tell)
- Text reports with optimization recommendations
- Results saved to `results/dispatch_timing/`

**Key Metrics:**
- **Cold compile time**: Initial JIT compilation overhead (includes tracing)
- **Warm dispatch time**: Amortized dispatch cost after first compilation
- **Unroll factor impact**: How many steps to amortize compilation overhead
- **Per-operator breakdown**: Individual timing for selection, crossover, mutation, evaluation

### 2. Fitness Benchmark (Quick)

Quick evolutionary algorithm benchmark on BBOB functions:

```bash
# Quick smoke test (instant execution)
python benchmarks/cli_fitness.py benchmarks/fitness_tuning_quick.toml --quick

# Full quick benchmark
python benchmarks/cli_fitness.py benchmarks/fitness_tuning_quick.toml
```

### 3. Fitness Tuning Benchmark

Comprehensive hyperparameter tuning and fitness landscape analysis:

```bash
# Full fitness tuning benchmark
python benchmarks/cli_fitness.py benchmarks/fitness_tuning.toml

# Generate comparison plots
python benchmarks/cli_fitness.py benchmarks/fitness_tuning.toml --plot
```

### 4. GECCO Benchmark Suite

Benchmark configuration for academic paper experiments:

```bash
python benchmarks/cli.py --config benchmarks/gecco_benchmark.toml
```

### 5. Smoke Test (Minimal Validation)

Quick validation that everything works:

```bash
# Run minimal smoke test
python benchmarks/cli.py --config benchmarks/smoke_test.toml

# Quick validation with dispatch analysis
python benchmarks/cli_dispatch.py benchmarks/smoke_test.toml --quick --framework both
```

### Configuration File Format

All benchmarks use TOML configuration files. Example structure:

```toml
[experiment]
name = "My_Benchmark"
output_dir = "results/my_benchmark"

[grid]
tasks = ["sphere", "rosenbrock"]        # BBOB problem names
dimensions = [10, 50, 100]               # Problem dimensions
pop_sizes = [64, 256, 1024]              # Population sizes
seeds = [42, 43, 44]                     # Random seeds

[dispatch]  # For dispatch_timing.toml
unroll_factors = [1, 2, 4, 8, 16, 32]   # Unroll factors to test
warmup_runs = 5                          # Warmup iterations
timed_runs = 20                          # Timed iterations

[hyperparam]
mutation_rate = 0.1
sigma = 0.1
crossover_rate = 0.9
elite_ratio = 0.1
tournament_size = 3
```

### Results Analysis

Benchmark results are saved to `results/dispatch_timing/`, organized as:

```
results/dispatch_timing/
├── sphere_d10_p32_s42_malthus/
│   ├── unroll_analysis.csv          # Dispatch overhead vs unroll factor
│   ├── operator_breakdown.csv       # Per-operator timing
│   └── dispatch_report.txt          # Detailed analysis
├── rosenbrock_d10_p32_s42_evosax_SimpleGA/
│   ├── unroll_analysis.csv
│   ├── operator_breakdown.csv
│   └── dispatch_report.txt
└── summary.csv                       # Aggregate results
```

### Makefile Shortcuts

```bash
# Run all quality checks
make check-all

# Run only tests
make test

# Code formatting with Ruff
make format

# Type checking
make type-check

# Linting
make lint
```

## License

[MIT License](LICENSE)
