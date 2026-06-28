# Kozax Framework Mechanics

This document breaks down the internal mechanics of the Kozax Genetic Programming framework.

## 1. Initialization Protocol
- **Primary Object**: `kozax.genetic_programming.GeneticProgramming`
- **Init Method**: `population = gp.initialize_population(key)`
- **State Object**: Kozax is fundamentally **stateless**. The object returned by initialization is merely a flat JAX `Array` of shape `(num_populations * population_size, num_trees, max_nodes, 4)`. This array contains the numerical representations of expression trees.

## 2. Execution Loop (Step Protocol)
Kozax uses a monolithic update approach.
- **Architecture Type**: Monolithic Step
- **Signature**: `next_population = gp.evolve_population(population, fitness, key)`
- **Behavior**: The framework consumes the previous population tensor, its fitnesses, and a PRNGKey, and produces a new population tensor directly via genetic operators (crossover, mutation). It does not use split `ask` and `tell` methods.

## 3. PRNG Management
PRNG is passed directly to the `evolve_population` monolithic method. External orchestrators must maintain the key state and split it before passing it to Kozax.

## 4. Metrics & Logging
- Kozax assumes **Minimization** by default (assigning `1e8` or higher numerical penalties to failed bounds/evaluations).
- Kozax does not return any metrics from its step function. The MalthusJAX Universal adapter will calculate `min_fitness` dynamically inside its scan loop since the `@adapter` must track generation progress natively.
