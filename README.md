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

All component methods follow a unified signature:
```python
def _select_parents(self, key: PRNGKey, state: AbstractEvolutionState, params: EngineParams) -> BasePopulation:
    # Full access to:
    # - state.population (current population with fitness)
    # - state.generation (current generation number)
    # - state.stagnation_counter (generations without improvement)
    # - state.best_fitness, state.best_genome (hall of fame)
    # - params (pop_size, elitism, num_generations, etc.)
    pass
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

## License

[MIT License](LICENSE)
