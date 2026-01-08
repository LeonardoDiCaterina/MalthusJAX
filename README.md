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
import malthusjax as mjx
import jax.random as jar

# Define problem: optimize a 100-bit binary string
genome_config = mjx.BinaryGenomeConfig(length=100)

# Configure evolutionary parameters
params = mjx.StandardEngineParams(
    pop_size=1000,
    num_generations=50,
    elitism=5
)

# Initialize population
key = jar.PRNGKey(42)
key, k_pop = jar.split(key)
initial_pop = mjx.BinaryPopulation.init_random(k_pop, genome_config, params.pop_size)

# Assemble engine with desired operators
engine = mjx.StandardGeneticEngine(
    evaluator=mjx.BinarySumEvaluator(mjx.BinarySumConfig(maximize=True)),
    selection=mjx.selection.Tournament(num_selections=params.pop_size, tournament_size=3),
    crossover=mjx.crossover.Uniform(num_offspring=2, crossover_rate=0.8),
    mutation=mjx.mutation.BitFlip(num_offspring=1, mutation_rate=0.01)
)

# Initialize and run evolution
key, k_init = jar.split(key)
state = engine.init_state(k_init, initial_pop)
final_state, history, elapsed_time = engine.run(state, params, time_it=True)
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
from malthusjax.engine import GeneticEngine

@struct.dataclass
class DiversityAwareEngine(GeneticEngine):
    diversity_weight: float = struct.field(default=0.3, pytree_node=False)
    
    def _select_parents(self, key, state, params):
        # Full access to state enables computing auxiliary metrics
        population = state.population
        
        # Compute crowding distance using distance matrix
        dist_matrix = population.distance_matrix(metric="hamming")
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
@struct.dataclass
class AdaptiveMutationEngine(GeneticEngine):
    base_mutation_rate: float = struct.field(default=0.01, pytree_node=False)
    
    def _create_offspring(self, key, parents, state, params):
        # Increase mutation rate when evolution stagnates
        adaptive_rate = self.base_mutation_rate * (1 + 0.2 * state.stagnation_counter)
        
        # Create modified mutation operator with adaptive rate
        adaptive_mutation = self.mutation.replace(mutation_rate=adaptive_rate)
        
        # Use parent implementation with modified operator
        # (simplified example - actual implementation may vary)
        return super()._create_offspring(key, parents, state, params)
```

This **Full Access architecture** enables researchers to experiment with:
- **Quality-Diversity algorithms** (MAP-Elites, Novelty Search)
- **Multi-objective optimization** (NSGA-II, SPEA2)
- **Adaptive parameter control** (self-adaptive mutation, learning rate schedules)
- **Age-layered population models** (ALPS)
- **Island models** with migration strategies

All while maintaining **full JIT compilation compatibility** and **functional purity**.

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
