# `malthusjax.engine.qd` — Reference

Scope: `malthusjax.engine.qd.map_elites`. Traceable directly to source code, tests, and docstrings.

---

## Overview & Architecture

The `malthusjax.engine.qd` module provides native Quality-Diversity (QD) evolution via the **MAP-Elites** (Multi-dimensional Archive of Phenotypic Elites) algorithm. It illuminates complex behavioral feature spaces by partitioning phenotypic descriptors into discrete grid cells (centroids) and retaining the highest-fitness individual discovered in each cell.

```
                      ┌─────────────────────────────────┐
                      │    MapElitesRepertoire (QDAX)   │
                      │  Centroids × Descriptors Grid   │
                      └────────────────┬────────────────┘
                                       │
                       emitter.ask()   │ (sample parents from archive)
                                       ▼
                      ┌─────────────────────────────────┐
                      │    Offspring Population (λ)     │
                      └────────────────┬────────────────┘
                                       │
                       evaluator       ▼ (score fitness + descriptors)
                      ┌─────────────────────────────────┐
                      │    Evaluated Offspring (λ)      │
                      │  fitness: (λ,), descriptors: (λ,d)
                      └────────────────┬────────────────┘
                                       │
                       repertoire.add()│ (update grid cells if better)
                                       ▼
                      ┌─────────────────────────────────┐
                      │      Updated Repertoire         │
                      │  QD Score & Coverage Tracking   │
                      └─────────────────────────────────┘
```

> **Dependency Requirement**:
> `MapElitesEngine` uses `qdax.core.containers.mapelites_repertoire.MapElitesRepertoire` for its internal $n$-dimensional Voronoi or grid archive. To use this engine, install QDAX via `pip install qdax`.

---

## Class Reference

### `MapElitesEngineParams`
Dataclass configuration (`@struct.dataclass`):
- `pop_size: int` (default 100) — Batch size of offspring emitted per generation.
- `num_generations: int` (default 50) — Number of QD generational cycles.
- `key_derivation: str` (default `"fold_in"`) — Key derivation strategy: `"fold_in"`, `"split"`, or `"qdax_replica"`.
- `maximize: bool` (default False) — Optimization sense (True for maximization, False for minimization).

### `MapElitesState[G, P]`
Carry state PyTree passed through scan loops:
- `repertoire: MapElitesRepertoire` — QDAX archive containing cell genotypes, fitnesses, descriptors, and centroids.
- `emitter_state: Optional[EmitterState]` — Internal state of the attached emitter.
- `population: BasePopulation` — Most recently evaluated batch of candidate offspring.
- `best_genome: G` — Global elite genome discovering highest fitness across all cells.
- `generation: int` — Completed generational count.
- `best_fitness: chex.Numeric` — True un-flipped raw fitness of the global elite.
- `rng_key: chex.Array` — PRNG key for subsequent generation.

### `QDGenerationOutput`
KPI metrics returned per generation step:
- `qd_score: chex.Array` — Sum of fitnesses of all occupied repertoire cells.
- `coverage: chex.Array` — Percentage of repertoire centroids currently filled ($N_{\text{filled}} / N_{\text{total}}$).
- `best_fitness: chex.Numeric` — Global best fitness in the archive.
- `generation: int` — Completed generation index.

### `MapElitesEngine[G, P]`
Flagship Quality-Diversity engine:
- `emitter: BaseEmitter` — Offspring generator implementing the `ask`/`tell` protocol.
- `evaluator: BaseQDEvaluator` — Evaluator returning fitness scores and behavioral descriptors (`info["descriptors"]`).
- `engine_params: MapElitesEngineParams` — Engine configuration parameters.
- **`init_state(rng_key, initial_population, centroids) -> MapElitesState`**: Evaluates initial population, initializes the QDAX repertoire and emitter, and returns starting `MapElitesState`.
- **`step(state) -> Tuple[MapElitesState, QDGenerationOutput]`**: Executes one complete MAP-Elites generation cycle.

---

## Key Derivation Modes

`MapElitesEngine` provides three PRNG key derivation strategies:

1. **`"fold_in"` (Default)**: Uses `jax.random.fold_in` across an integer index range. Fast, parallel, and compiles cleanly with zero host-to-device synchronization.
2. **`"split"`**: Standard sequential `jax.random.split` chaining.
3. **`"qdax_replica"`**: Exactly reproduces QDAX's nested key-splitting sequence (`split(randkey) -> split(subkey1) -> split(subkey2)`). Used for bit-exact cross-framework statistical parity validation against upstream QDAX runs.

---

## Fitness Sign Handling

In MalthusJAX, minimization tasks report lower raw fitnesses (lower is better). However, QDAX's `MapElitesRepertoire` internally assumes maximization. `MapElitesEngine` automatically handles this:
- When `maximize=False`, fitnesses are negated before being passed to `repertoire.add()`.
- Returned metrics (`best_fitness`, `QDGenerationOutput`) are automatically re-scaled so that reported metrics always reflect the user's intended objective.

---

## Usage Example

```python
import jax.random as jr
from malthusjax.core.genome import RealGenomeConfig, RealPopulation
from malthusjax.operators.mutation import GaussianMutation
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter
from malthusjax.engine.qd.map_elites import MapElitesEngine, MapElitesEngineParams

# 1. Configuration & Setup
config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
key = jr.PRNGKey(42)
k_init, k_run = jr.split(key)

# 2. Components
init_pop = RealPopulation.init_random(k_init, config, size=128)
mutation = GaussianMutation(mutation_rate=0.2, mutation_strength=0.1)
emitter = GeneticMutationEmitter(mutation=mutation, genome_config=config, _batch_size=128)

# 3. Engine setup (assuming centroids shape (num_centroids, num_descriptors))
engine = MapElitesEngine(
    emitter=emitter,
    evaluator=my_qd_evaluator,
    engine_params=MapElitesEngineParams(pop_size=128, num_generations=100, maximize=True),
)

state = engine.init_state(k_run, init_pop, centroids=centroids)
state, output = engine.step(state)
print(f"Coverage: {output.coverage:.2f}%, QD Score: {output.qd_score:.2f}")
```
