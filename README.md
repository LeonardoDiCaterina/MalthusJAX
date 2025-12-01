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
- **State Management**: Immutable `AbstractEvolutionState` enables JIT compilation via `jax.lax.scan`

## Installation

```bash
# Clone the repository
git clone https://github.com/LeonardoDiCaterina/MalthusJAX.git
cd MalthusJAX

# Install with development dependencies
make install-dev
```

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

## License

[MIT License](LICENSE)
