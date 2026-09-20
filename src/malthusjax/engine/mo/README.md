# `malthusjax.engine.mo` — Reference

Scope: `malthusjax.engine.mo.mo_engine`. Traceable directly to source code, tests, and docstrings.

---

## Overview & Architecture

The `malthusjax.engine.mo` module provides native multi-objective evolutionary optimization in pure JAX, implementing the classic **NSGA-II** ($\mu + \lambda$) non-dominated sorting elitism paradigm.

```
                  ┌───────────────────────────────┐
                  │    Current MOPopulation (μ)   │
                  └──────────────┬────────────────┘
                                 │
                 emitter.ask()   │ (sample parents)
                                 ▼
                  ┌───────────────────────────────┐
                  │      Offspring Population (λ) │
                  └──────────────┬────────────────┘
                                 │
                   evaluator     ▼ (multi-objective scoring)
                  ┌───────────────────────────────┐
                  │     Evaluated Offspring (λ)   │
                  └──────────────┬────────────────┘
                                 │
            pop.merge(eval_pop)  ▼ (combined μ + λ pool)
                  ┌───────────────────────────────┐
                  │    Combined Pool (μ + λ)      │
                  │   Non-Dominated Sort (Ranks)  │
                  │   Crowding Distance Sorting   │
                  └──────────────┬────────────────┘
                                 │
                 pop.truncate(μ) ▼ (preserve best μ)
                  ┌───────────────────────────────┐
                  │     Next MOPopulation (μ)     │
                  └───────────────────────────────┘
```

Key architectural mechanisms:
- **`MOPopulation` Container**: Specialized Struct-of-Arrays (SoA) population carrying multi-dimensional fitness tensors `(pop_size, num_objectives)`, non-dominated `pareto_rank` arrays, and `crowding_distance` metrics.
- **Vectorized $(\mu + \lambda)$ Elitism**: Merging parent and offspring populations (`.merge()`) and truncating down to population size (`.truncate(pop_size)`) using JAX-compiled non-dominated front sorting and crowding distance tie-breaking.
- **Decoupled Variation via `BaseEmitter`**: Emits candidate offspring batches using any compatible Level 2 emitter (e.g., `GeneticMutationEmitter`, `GeneticMixingEmitter`), allowing modular reuse of crossover and mutation operators.
- **Deterministic Static PRNG Budgeting**: Pre-calculates exact key requirements per generation (`emitter_keys + 3`), allocating subkeys for `ask`, `eval`, `tell`, and scan state progression without dynamic allocation inside `jax.lax.scan`.

---

## Class Reference

### `MOEngineParams`
Dataclass configuration (`@struct.dataclass`) for multi-objective evolution:
- `pop_size: int` (default 100) — Population size ($\mu$).
- `num_generations: int` (default 50) — Number of generational scan cycles.
- `key_derivation: str` (default `"fold_in"`) — PRNG key derivation strategy (`"fold_in"` or `"split"`).

### `MOState[G, P]`
Carry state PyTree passed through scan loops:
- `population: MOPopulation` — Current evaluated population with Pareto ranks and crowding distances.
- `best_genome: G` — Genome of the representative individual in the first Pareto front.
- `generation: int` — Current generational counter.
- `best_fitness: chex.Numeric` — Objective-0 fitness value of the first Pareto front individual.
- `rng_key: chex.Array` — PRNG key for the subsequent generation.
- `emitter_state: Optional[EmitterState]` — Internal state of the attached emitter.

### `MOGenerationOutput`
KPI metrics returned per generation step:
- `best_fitness: chex.Numeric` — Objective-0 fitness value of the first individual in the rank-0 Pareto front.
- `mean_fitness: chex.Numeric` — Mean fitness across objective 0.
- `std_fitness: chex.Numeric` — Standard deviation across objective 0.
- `generation: int` — Completed generation index.
- `num_pareto_optimal: chex.Numeric` — Count of individuals currently residing in Pareto front rank 0 (`jnp.sum(pareto_rank == 0)`).
- `max_crowding_distance: chex.Numeric` — Maximum crowding distance observed in the population (`jnp.max(crowding_distance)`).

### `MOEngine[G, P]`
Flagship multi-objective evolutionary engine:
- `emitter: BaseEmitter` — Offspring generator implementing the `ask`/`tell` protocol.
- `evaluator: BaseMOEvaluator` — Vectorized evaluator producing multi-objective fitness scores and metadata.
- `engine_params: MOEngineParams` — Configuration dataclass.
- **`init_state(rng_key, initial_population) -> MOState`**: Evaluates initial population, initializes emitter state, and packages initial `MOState`.
- **`step(state) -> Tuple[MOState, MOGenerationOutput]`**: Executes one NSGA-II generation cycle.

---

## Generational Step Pipeline

During each `step()` invocation:
1. **PRNG Slicing**: Budgets `emitter.num_keys() + 3` keys via `fold_in` or `split`, yielding `k_ask`, `k_eval`, `k_tell`, and `k_next`.
2. **Offspring Emission (`ask`)**: Calls `emitter.ask()` to sample parents and generate a batch of offspring.
3. **Multi-Objective Evaluation**: Calls `dispatch_evaluate_population(evaluator, offspring_pop, k_eval)`.
4. **$(\mu + \lambda)$ Survival**: Executes `state.population.merge(eval_pop).truncate(pop_size)` to sort Pareto fronts and maintain high diversity via crowding distance.
5. **Emitter Feedback (`tell`)**: Optionally informs the emitter of offspring fitness for adaptive step sizes.
6. **KPI Extraction**: Gathers `num_pareto_optimal`, `max_crowding_distance`, and updates `MOState`.

---

## Usage Example

```python
import jax.random as jr
from malthusjax.core.genome import RealGenomeConfig, RealPopulation
from malthusjax.core.fitness.mo.evaluator import BaseMOEvaluator
from malthusjax.operators.mutation import GaussianMutation
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter
from malthusjax.engine.mo.mo_engine import MOEngine, MOEngineParams

# 1. Configuration
config = RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))
key = jr.PRNGKey(42)
k_init, k_run = jr.split(key)

# 2. Components
init_pop = RealPopulation.init_random(k_init, config, size=100)
mutation = GaussianMutation(mutation_rate=0.1, mutation_strength=0.1)
emitter = GeneticMutationEmitter(mutation=mutation, genome_config=config, _batch_size=100)

# 3. Engine setup (assuming custom multi-objective evaluator)
engine = MOEngine(
    emitter=emitter,
    evaluator=my_mo_evaluator,
    engine_params=MOEngineParams(pop_size=100, num_generations=50),
)

state = engine.init_state(k_run, init_pop)
state, output = engine.step(state)
print(f"Pareto optimal individuals: {output.num_pareto_optimal}")
```
