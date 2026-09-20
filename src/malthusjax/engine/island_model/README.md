# `malthusjax.engine.island_model` — Reference

Scope: `malthusjax.engine.island_model.base`, `malthusjax.engine.island_model.topologies`. Traceable directly to source code, tests, and docstrings.

---

## Overview & Architecture

The `malthusjax.engine.island_model` module provides parallel distributed evolutionary algorithms through **vectorized island models**. It wraps any native single-population engine (e.g. `GeneticEngine`) into a multi-deme meta-engine executed via `jax.vmap` across the island axis.

Each island evolves its local sub-population independently for a configurable `migration_interval` number of generations, after which individuals migrate between islands according to a structured communication topology.

```
                      ┌──────────────────────────────────────────────┐
                      │    Vectorized 2D Population (M Islands)      │
                      │       shape: (num_islands, island_size, ...) │
                      └──────────────────────┬───────────────────────┘
                                             │
                       jax.vmap(lax.scan)    │ (evolve locally for interval)
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    Evolved 2D Sub-Populations                │
                      └──────────────────────┬───────────────────────┘
                                             │
                       migrate(key, pop)     │ (exchange migrants across demes)
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    Migrated 2D State (Next Cycle)            │
                      └──────────────────────────────────────────────┘
```

Key architectural mechanisms:
- **Zero Host Re-dispatch Inside Intervals**: Local island generations run inside a compiled `jax.lax.scan` loop mapped over islands via `jax.vmap`, maximizing GPU compute utilization and SIMD vectorization.
- **Topological Migration Policies**: Concrete subclasses implement the `migrate()` method to execute deterministic index permutations or random shuffles.
- **Universal Engine Compatibility**: Upgrades any `AbstractEngine` subclass into an island model without altering the underlying engine's step logic.

---

## Class Reference

### `BaseIslandModel[Generic[E]]` (`base.py`)
Abstract meta-engine wrapping an underlying engine `E`:

| Property / Method | Description |
| :--- | :--- |
| `engine: E` | The underlying single-population engine instance. |
| `num_islands: int` | Number of parallel isolated demes. |
| `migration_interval: int` | Number of local generations executed between migration events. |
| `num_migrants: int` | Number of individuals exchanged per island during migration. |
| `maximize` | Derives optimization direction from the underlying engine or evaluator. |
| `init_state(key)` | Splits key into `num_islands` subkeys and initializes a 2D multi-island state via `jax.vmap(engine.init_state)`. |
| `_island_loop(state)` | Runs `engine.step()` inside `jax.lax.scan` for `migration_interval` generations. |
| `migrate(key, multi_pop)` | Abstract method applying the topological migration exchange. |
| `step(multi_state)` | Orchestrates one macro-generation: vmapped local evolution followed by topological migration. |

---

## Topologies (`topologies.py`)

### `RingTopologyIsland`
Classic ring topology for genetic algorithms and genetic programming:
- **Mechanism**: Extracts the top `num_migrants` elite individuals from each island, shifts their positions one island to the right via `jnp.roll(axis=0)`, and injects them into the worst fitness spots of the target island.
- **Characteristics**: Low diffusion rate, preserves regional diversity, prevents premature global convergence.

### `FullyConnectedIsland`
Maximum-diffusion topology:
- **Mechanism**: Extracts the top `num_migrants` elites across all islands into a unified pool, applies a global PRNG permutation via `jax.random.permutation`, and redistributes migrants uniformly across all islands to replace their worst individuals.
- **Characteristics**: High communication throughput, rapid propagation of high-quality genetic material.

---

## Usage Example

```python
import jax.random as jr
from malthusjax.core.genome import RealGenomeConfig
from malthusjax.core.fitness.composable.evaluators import create_sphere_evaluator
from malthusjax.operators.selection import TournamentSelection
from malthusjax.operators.crossover import UniformCrossover
from malthusjax.operators.mutation import GaussianMutation
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.island_model.topologies import RingTopologyIsland

# 1. Base components
config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
evaluator = create_sphere_evaluator()
selection = TournamentSelection(num_selections=16, tournament_size=3)
crossover = UniformCrossover(crossover_rate=0.8)
mutation = GaussianMutation(mutation_rate=0.1, mutation_strength=0.1)

# 2. Local engine (island_size = 20)
base_engine = GeneticEngine(
    genome_config=config,
    fitness_evaluator=evaluator,
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    engine_params=GeneticEngineParams(pop_size=20, num_generations=10),
)

# 3. Island model wrapper (5 islands, migrate every 10 generations, 2 migrants)
island_model = RingTopologyIsland(
    engine=base_engine,
    num_islands=5,
    migration_interval=10,
    num_migrants=2,
)

key = jr.PRNGKey(42)
multi_state = island_model.init_state(key)
multi_state, history = island_model.step(multi_state)
print("Island model step completed successfully.")
```
