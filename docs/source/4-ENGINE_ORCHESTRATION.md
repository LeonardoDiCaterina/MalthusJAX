# The Engine: Orchestration and Compilation

The `malthusjax.engine` module provides the execution layer of MalthusJAX. It orchestrates population containers (`BasePopulation`), objective evaluators (`BaseEvaluator`), and genetic operators (`BaseSelection`, `BaseCrossover`, `BaseMutation`, `BaseEmitter`) into stateful execution loops using JAX transformations (`jax.lax.scan`, `jax.vmap`, `jax.jit`).

All engine configurations use Flax `@struct.dataclass` PyTrees with static configuration fields marked `pytree_node=False`.

---

## 4.1. Execution State & Parameter Validation

### Engine Parameters (`AbstractEngineParams`, `GeneticEngineParams`)
Engine execution is governed by static dataclass configurations:
- `pop_size: int` — Population size ($N > 0$).
- `elitism: int` — Number of elite individuals preserved across generations ($0 \le \text{elitism} < \text{pop\_size}$).
- `num_generations: int` — Total generational steps to execute ($N_\text{gen} > 0$).
- `unroll_num: int` — Scan unroll factor (defaults to `1`; `compute_unroll_num()` returns `1` as scan unrolling is deprecated to prevent linear XLA IR growth).
- `track_best: TrackBest` — IntEnum controlling Hall-of-Fame state tracking inside the scan carry:
  - `TrackBest.NONE` (0): Zero extra ops in the scan carry per step; `best_genome` and `best_fitness` are populated post-scan from the final population.
  - `TrackBest.LIGHT` (1, default): Tracks monotonic `best_fitness` in the carry using fusible `jnp.max` and `jnp.maximum` operations; `best_genome` is extracted post-scan via `jnp.argmax`.
  - `TrackBest.FULL` (2): Tracks both `best_fitness` and `best_genome` in the carry every step using `jnp.max`, `jnp.argmax`, `Gather`, and element-wise `jnp.where`.
- `track_metrics: TrackMetrics` — IntEnum controlling metric calculation (`NONE`, `BASIC`, `ALL`).

Parameter boundaries are validated outside JIT context by `validate_engine_params(params)`, raising `ValueError` if constraints are violated.

### Evolution State (`AbstractEvolutionState`, `GeneticEvolutionState`)
State is threaded through `jax.lax.scan` as an immutable PyTree carrying:
- `population`: Current `BasePopulation[G]` container.
- `best_genome`: `BaseGenome` PyTree of the best solution found so far.
- `best_fitness`: Scalar array tracking best fitness score.
- `generation`: Integer step counter.
- `rng_key`: Master 2-element PRNG key for key derivation.
- `resource_map`: Pre-computed `ResourceMap` execution plan (`GeneticEvolutionState`).
- `operators`: Frozen `OperatorState` dataclass containing initialized operators (`GeneticEvolutionState`).

Deep copying of state PyTrees (to avoid JAX buffer donation conflicts across multi-seed runs) is provided via `state.copy()`.

---

## 4.2. PRNG Key Allocation & `ResourceMapper`

Key budgeting and data-flow shapes are pre-calculated during `init_state` by `compute_resource_map(...)`.

### `ResourceMap`
The `ResourceMap` pre-allocates contiguous slices inside a flat master key tensor for each pipeline stage (`selection`, `crossover`, `mutation`, `evaluation`, `next_key`), preventing dynamic key allocation during scan execution.

Key derivation strategies (`KeyDerivationStrategy`):
- `SPLIT` (`KeyDerivationStrategy.SPLIT`): Derives keys sequentially via `jax.random.split(master_key, total_rng_budget)`.
- `FOLD` (`KeyDerivationStrategy.FOLD`): Derives keys in parallel by mapping `jax.random.fold_in` over integer indices (`jnp.arange(total_rng_budget)`) via `jax.vmap`.
  > **Backend Constraint**: `KeyDerivationStrategy.FOLD` is incompatible with `rbg` and `unsafe_rbg` PRNG backends and raises a `ValueError` if selected with those backends.

### `ShardingManager`
Multi-device and layout management is handled by `ShardingManager`, which constructs a JAX `Mesh` along a designated `batch` axis. It applies `NamedSharding` with `PartitionSpec("batch", None)` to shard population matrices across devices via `jax.device_put`.

---

## 4.3. Generational Loop Architecture (`GeneticEngine`)

`GeneticEngine` executes standard evolutionary optimization. Each call to `step(state)` executes 5 distinct phases:

1. **Phase 0 — Entropy Allocation (`_allocate_entropy`)**:
   Slices the pre-allocated master PRNG key buffer using `ResourceMap.get_key_slice` into subkeys for selection, crossover, mutation, evaluation, and the next-generation seed.
2. **Phase 1 — Selection (`_selection_phase`)**:
   Invokes `selection(key, population)` to compute `parent_indices` (length `num_selections`) and `elite_indices` (length `elitism`).
3. **Phase 2 — Reproduction (`_reproduction_phase`)**:
   Calls `crossover(k_cross, p1_pop, p2_pop, config, generation)` to generate recombined offspring, followed by `mutation(k_mut, offspring_pop, config, generation)`. Dynamic mutation strength scheduling is computed via `compute_scheduled_strength` (`ScheduleType`: `CONSTANT`, `LINEAR_DECAY`, `COSINE_ANNEAL`, `EXPONENTIAL_DECAY`).
4. **Phase 3 — Merge (`_merge_phase`)**:
   When `elitism > 0`, combines elite parent genomes with mutated offspring genomes into a unified population container of size `pop_size` using array concatenation (`jnp.concatenate` / `tree_map`).
5. **Phase 4 — Evaluation & State Update (`_evaluation_phase`)**:
   Scores the population via `dispatch_evaluate_population(evaluator, population, k_eval)` and updates the Hall-of-Fame state according to `TrackBest`.

### Execution Modes
- **Scan-based Loop (`run`)**: Wraps `step` inside `jax.lax.scan` for in-process execution over `num_generations` steps.
- **Ask/Tell Interface (`ask`, `tell`)**:
  - `ask(state)`: Allocates entropy, increments the generation counter, and returns `(state_with_entropy, population.genes)` for external evaluation.
  - `tell(state, evaluated_population)`: Receives externally evaluated fitness scores, updates population fitness, and performs Hall-of-Fame state tracking.

---

## 4.4. Multi-Objective Engine (`MOEngine`)

`MOEngine` implements Multi-Objective optimization under the NSGA-II paradigm:
- **Architecture**: Operates via `BaseEmitter`, `BaseMOEvaluator`, and `MOPopulation`.
- **Pareto Elitism**: `MOPopulation` maintains Pareto ranks and crowding distances across combined parent ($\mu$) and offspring ($\lambda$) pools.
- **KPI Output (`MOGenerationOutput`)**: Reports `num_pareto_optimal` (number of rank-0 non-dominated solutions) and `max_crowding_distance`. `best_fitness` reports the objective-0 score of the primary Pareto individual.

---

## 4.5. Quality-Diversity Engine (`MapElitesEngine`)

`MapElitesEngine` orchestrates MAP-Elites Quality-Diversity optimization:
- **Grid Container**: Integrates `BaseEmitter`, `BaseQDEvaluator`, and QDAX's `MapElitesRepertoire` grid structure.
- **External Dependency**: Requires the `qdax` Python package (`from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire`). Raises an `ImportError` if `qdax` is not installed in the Python environment.
- **KPI Output (`QDGenerationOutput`)**: Reports `qd_score` (sum of fitnesses across all filled archive cells) and `coverage` (fraction of filled grid cells).

---

## 4.6. Distributed Island Models (`BaseIslandModel`)

`BaseIslandModel` upgrades any 1D engine into a 2D multi-island model by wrapping engine methods in `jax.vmap` across an island dimension of size `num_islands`.

### Host Synchronization Execution Pattern
`BaseIslandModel.step()` executes using a hybrid JAX/Python loop:
1. **Local Evolution**: Runs `self.engine.step` across all islands in parallel for `migration_interval` generations using `jax.vmap` over an inner `jax.lax.scan`.
2. **Host Dropout & Migration**: Drops out of JAX execution back to the Python host to execute `migrate(key, multi_pop)`.
3. **Re-injection**: Re-injects the migrated population PyTree into the island state and launches the next JAX execution kernel.

### Topological Migration Policies (`topologies.py`)
- **`RingTopologyIsland`**: Sorts fitness per island, extracts the top `num_migrants` elites, shifts elite tensors to the adjacent island via `jnp.roll(arr, shift=1, axis=0)`, and overwrites the worst `num_migrants` individuals in the destination island.
- **`FullyConnectedIsland`**: Extracts top elites across all islands into a flat pool of size `num_islands * num_migrants`, shuffles them globally using `jax.random.permutation`, and redistributes them back into the worst slots of each island.
