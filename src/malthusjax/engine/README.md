# `malthusjax.engine` — Reference

Scope: `malthusjax.engine.base`, `malthusjax.engine.resource_mapper`, `malthusjax.engine.schedules`, `malthusjax.engine.genetic_fastengine`, `malthusjax.engine.mo.*`, `malthusjax.engine.qd.*`, `malthusjax.engine.island_model.*`. Every claim below is traceable directly to source code and docstrings.

---

## Overview & Architecture

The `malthusjax.engine` package orchestrates population containers (`BasePopulation`), fitness evaluators (`BaseEvaluator`), and genetic operators (`BaseSelection`, `BaseCrossover`, `BaseMutation`, `BaseEmitter`) into stateful evolution loops.

Key architectural mechanisms:
- **Init-Phase JIT Compilation**: `init_state` calculates an execution plan (`ResourceMap`) and freezes operator input dimensions. Subsequent calls to `step` or `run` use the cached plan without recompilation overhead.
- **State Threading**: `AbstractEvolutionState` PyTrees are carried through `jax.lax.scan` to compile generational loops into fused XLA kernels.
- **Resource Budgeting**: `ResourceMap` pre-calculates exact per-operator PRNG key slices during `init_state`, eliminating dynamic key allocation during scan execution.

---

## `malthusjax.engine.base`

### `AbstractEngineParams`
Base dataclass configuration for evolution engines (`pytree_node=False`):
- `pop_size: int` (default 100) — Population size ($N > 0$).
- `elitism: int` (default 0) — Number of elite individuals preserved each generation ($0 \le \text{elitism} < \text{pop\_size}$).
- `num_generations: int` (default 50) — Number of evolution steps.
- `unroll_num: int` (default 1) — Scan unroll factor. Note: `compute_unroll_num()` always returns `1` (unrolling was deprecated to prevent linear XLA IR growth).
- `track_metrics: bool` (default True) — Controls whether evaluation metrics are collected.

### `validate_engine_params(params)`
Validates configuration constraints outside JIT context, raising `ValueError` if `pop_size <= 0`, `num_generations <= 0`, or `elitism` violates $0 \le \text{elitism} < \text{pop\_size}$.

### `AbstractEvolutionState[G, P]`
Mutable scan carry PyTree storing `population`, `best_genome`, `generation`, `best_fitness`, and `rng_key`. Supports deep copying via `state.copy()` to avoid JAX buffer donation errors across runs.

### `AbstractEngine[G, P]`
Abstract base class hashable via `id(self)` for JIT `static_argnums`. Enforces standard engine interface: `maximize` property, `init_state(rng_key)`, `step(state)`, and debug methods (`debug_step`, `debug_run`).

---

## `malthusjax.engine.resource_mapper`

### `KeyDerivationStrategy`
Enum controlling how per-generation master PRNG keys are derived:
- `SPLIT` (`KeyDerivationStrategy.SPLIT`): Sequential hash chain using `jax.random.split`.
- `FOLD` (`KeyDerivationStrategy.FOLD`): Parallel key derivation mapping `jax.random.fold_in` over integer indices (`jnp.arange(total_rng_budget)`).
  > **Constraint**: `FOLD` is incompatible with `rbg` and `unsafe_rbg` PRNG backends and raises a `ValueError` if selected with those backends.

### `ShardingManager`
Constructs a JAX `Mesh` along axis `batch` and applies `NamedSharding` with `PartitionSpec("batch", None)` for population placement on multi-device or GPU layouts.

### `ResourceMap` & `compute_resource_map`
Pre-calculates total PRNG key budget and per-stage key slice ranges (`selection`, `crossover`, `mutation`, `evaluation`, `next_key`) for a generation.

---

## `malthusjax.engine.schedules`

### `ScheduleType` (IntEnum)
Mutation strength schedules evaluated inside JAX loops via `compute_scheduled_strength`:
- `CONSTANT` (0): No schedule (static strength).
- `LINEAR_DECAY` (1): Linear decay from `initial_strength` to `final_strength`.
- `COSINE_ANNEAL` (2): Cosine annealing decay.
- `EXPONENTIAL_DECAY` (3): Exponential decay.

### `TrackBest` (IntEnum)
Controls Hall-of-Fame tracking in scan carry:
- `NONE` (0): Zero extra ops per step in carry; `best_genome` and `best_fitness` populated post-scan.
- `LIGHT` (1, default): Tracks monotonic `best_fitness` in carry via `jnp.max`/`jnp.maximum`; `best_genome` populated post-scan via `jnp.argmax`.
- `FULL` (2): Tracks both `best_fitness` and `best_genome` in carry every step using `jnp.max`, `jnp.argmax`, `Gather`, and element-wise `jnp.where`.

---

## `malthusjax.engine.genetic_fastengine`

### `GeneticEngine`
Standard Genetic Algorithm engine implementing a 5-phase generational step:
1. **Phase 0 (Entropy Allocation)**: Slices master key into stage subkeys via `ResourceMap`.
2. **Phase 1 (Selection)**: Calls `selection(key, population)` returning parent and elite index arrays.
3. **Phase 2 (Reproduction)**: Calls `crossover` and `mutation` with scheduled mutation strength.
4. **Phase 3 (Merge)**: Combines elites (when `elitism > 0`) and mutants via array concatenation (`jnp.concatenate` / `tree_map`).
5. **Phase 4 (Evaluation & Tracking)**: Evaluates population fitness via `dispatch_evaluate_population` and updates `TrackBest` state metrics.

### Execution Entry Points
- `run(initial_state)`: Executes `step` inside `jax.lax.scan` over `num_generations`.
- `ask(state)`: Returns `(state_with_entropy, population.genes)` for external evaluation loops.
- `tell(state, evaluated_population)`: Updates population, fitness scores, and state metrics after external evaluation.

---

## Specialized & Meta-Engines

### Multi-Objective Engine (`malthusjax.engine.mo`)
- `MOEngine`: Orchestrates NSGA-II non-dominated sorting elitism using `BaseEmitter`, `BaseMOEvaluator`, and `MOPopulation`.
- `MOGenerationOutput`: Reports `num_pareto_optimal` and `max_crowding_distance`. `best_fitness` reports objective-0 of the primary Pareto front individual.

### Quality-Diversity Engine (`malthusjax.engine.qd`)
- `MapElitesEngine`: Integrates `BaseEmitter`, `BaseQDEvaluator`, and QDAX's `MapElitesRepertoire`.
- **Dependency**: Requires external `qdax` package (`from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire`); raises `ImportError` if `qdax` is absent.
- `QDGenerationOutput`: Reports `qd_score` and `coverage`.

### Distributed Island Models (`malthusjax.engine.island_model`)
- `BaseIslandModel`: Meta-engine distributing any 1D `AbstractEngine` across `num_islands` via `jax.vmap`.
- **Host Synchronization Architecture**: Runs local evolution for `migration_interval` steps inside `jax.vmap` of `jax.lax.scan`, then **drops out to the Python host** to execute `migrate(key, multi_pop)` before dispatching the next JAX kernel.
- `RingTopologyIsland`: Shifts top `num_migrants` elites 1 island right via `jnp.roll(arr, shift=1, axis=0)` and overwrites worst slots.
- `FullyConnectedIsland`: Flattens elites across all islands, shuffles globally via `jax.random.permutation`, and redistributes to worst slots.
