# Selection Operators — Architecture & Implementation

This document explains how selection operators are implemented in MalthusJAX, and how they interact with the static ResourceMapper and sharding. It targets developers extending or optimizing selection stages in the engine.

---

## High-level Summary

- Selection consumes a population and fitness values, producing an index array of selected individuals.
- The system uses **static RNG budgeting** via `ResourceMap` to ensure deterministic key allocation and predictable shapes for JIT compilation.
- Selection implements a clear separation between **atomic logic** (a pure `_select()` method) and **population-level slicing** (`BaseSelection.__call__`) that produces a batched `jnp.ndarray` of indices.

---

## 1) Static RNG Budgeting & Resource Mapping

- The `ResourceMapper` queries each operator to compute the exact number of keys required:
  - `sel_keys_needed = num_keys_per_atomic_operation` (single key, shared across all selections via internal vectorization).
  - `compute_resource_map(...)` stores these allocations in an `OperatorAllocation` for selection.
- The engine allocates a single master key buffer of size `total_rng_budget` and slices this buffer for selection using `ResourceMap.get_key_slice("selection")`.

Table — Key budgeting formula

| Symbol | Meaning |
|--------|----------|
| N | population size |
| S | num_selections (per call) |
| K | num_keys_per_atomic_operation | 
| sel_keys_needed | K (not S * K) |

Why static budgeting matters:
- Enables one-time host-side split and deterministic slicing.
- Avoids dynamic allocations and repeated `jax.random.split` calls inside the inner loop, minimizing host-device traffic.
- **Key insight**: Selection uses a single PRNG key for all `num_selections` because JAX vectorizes sampling operations. A single key is internally split/reused to generate all `num_selections` indices simultaneously (e.g., `jax.random.randint(..., shape=(num_selections, ...))`).

---

## 2) Integration with Engine Operations

- Selection produces **two parallel index arrays**:
  - **Parent indices** (`parent_idx`): Shape `(num_selections,)`, used for gathering parents for crossover/mutation
  - **Elite indices** (`elite_idx`): Shape `(n_elites,)`, used for elite preservation in next generation
- These indices feed downstream: parents to crossover, elites directly to next generation (or to elite pool)
- The ResourceMapper computes total key budgets for all operators so shapes are determined at initialization time.

---

## 2.5) Elite Preservation Mechanism

**When Elitism is Enabled** (`elitism > 0`), the engine preserves the top-performing individuals:
1. Engine calls `selection = selection.set_n_elites(params.elitism)` at initialization (note: returns new instance)
2. During each generation, selection's `__call__()` returns both parent and elite indices
3. Elite indices identify the top `n_elites` individuals (guaranteed to be highest fitness)
4. Engine directly carries these individuals to the next generation (bypass crossover/mutation)

**IMPORTANT**: `set_n_elites()` is immutable—it returns a new operator instance with the elite count set. Assign the result:
```python
selection = TournamentSelection(num_selections=10, tournament_size=3)
selection = selection.set_n_elites(2)  # Must assign the result
parent_idx, elite_idx = selection(keys, population, None)
# elite_idx now contains indices of top 2 individuals
```

**Key Advantage**: Elite identification leverages sorting/partitioning already performed during selection:
- **Tournament/Roulette**: Default `get_elite_indices()` uses O(N) `jnp.argpartition` (separate pass)
- **ElitePoolSelection**: Fuses both parent selection AND elite identification in a single `argpartition` pass (one-pass optimization)

**Elite Index Properties**:
- Always top-k individuals by fitness (highest fitness first)
- Distinct indices (no duplicates)
- Shape `(n_elites,)`, can be empty `(0,)` when `n_elites=0`
- Guaranteed different from parent indices (elites not re-selected, directly preserved)

---

## 3) Operator Interface & Functional Logic

- **Atomic Logic (`_select()` method)**
  - A pure function that consumes PRNG keys and a fitness array and returns selected indices.
  - Signature: `_select(keys: chex.Array, fitness: chex.Array, config: Optional[C]) -> indices: chex.Array`
  - Returns indices matching `num_selections`, dtype int32/int64.
  - Internal only; subclasses implement selection-specific logic here (tournament sampling, softmax, argpartition)

- **Population-Level Selection (BaseSelection.__call__)**
  - **Dual Output**: `BaseSelection.__call__(keys, population, config) -> (parent_idx, elite_idx)`
  - Accepts either a Population object (extracts `.fitness`) or a fitness array directly via `getattr(population, "fitness", population)`
  - Calls `_select()` to produce parent indices, then separately produces elite indices via `get_elite_indices()`
  - Returns tuple of two integer index arrays: `(parent_idx: (num_selections,), elite_idx: (n_elites,))`

- **Elite Identification (`get_elite_indices()` method)**
  - Base implementation: O(N) `jnp.argpartition` to find top `n_elites` individuals
  - Subclasses (e.g., ElitePoolSelection) may override to fuse this with parent selection for optimization
  - Always returns highest-fitness individuals (monotonic with fitness rank)

Benefits of returning indices (not genomes):
- Decouples selection logic from memory movement
- Allows engine to perform single gather operation, minimizing copies and improving fusion
- Elite preservation happens at index level, avoiding duplicate data in next generation

---

## 4) Hardware-Specific Optimization & Sharding

- `ShardingManager` defines named sharding specs (`pop_sharding`, `vector_sharding`) and can be used to `device_put` selection's index buffer so that it is already sharded across devices.
- Fitness arrays should be sharded consistently with population layout so selection index computation remains local to devices where possible, minimizing cross-device moves.
- When running on many devices, `split_key_sharded` helps distribute RNG keys across the mesh to ensure per-device independent streams.

Example: place selection indices on `pop_sharding` so subsequent gather operations are local to each device.

---

## 5) Index Type Requirements

- Selection indices MUST be integer dtypes (int32 or int64). Create indices explicitly using `jax.random.randint(..., dtype=jnp.int32)` or similar functions.
- Avoid floating-point operations on indices; cast to integers immediately after any sampling.

---

## 6) Key Features

- **Deterministic vs Stochastic**: Selection operators can be deterministic (best, truncation) by setting `num_keys_per_atomic_operation = 0`, or stochastic (tournament, rank-based) requiring PRNG keys.
- **Static Key Budgeting**: The `set_input_length()` method freezes the population size for static key allocation; `num_keys()` returns the total keys needed (just `K`, shared across all selections).
- **RNG Flexibility**: The engine pre-allocates and slices keys from a master buffer, ensuring no dynamic allocation during the evolution loop.
- **PRNG Format Handling** (`typed_keys`): Operators branch on `typed_keys` flag to extract RNG key correctly:
  - `typed_keys=True`: New-style JAX PRNG (simple scalar keys), extract via `keys if keys.ndim == 0 else keys[0]`
  - `typed_keys=False`: Legacy uint32[2] pairs, extract via `keys if keys.ndim <= 1 else keys[0]`
  - Engine sets this at init based on PRNG backend, operators must handle both formats
- **Population Fallback**: `__call__()` accepts Population OR raw fitness array:
  - If argument has `.fitness` attribute: use it
  - If argument is an array: use directly
  - Enables flexible API (users can pass population OR pre-extracted fitness)

---

## 7) Engine Integration

- The engine calls `compute_resource_map()` once at initialization to determine total RNG budget and shape requirements.
- For selection, the engine slices keys from the master buffer and passes them to the selection operator.
- The selection operator returns indices, which the engine uses for gathering parent genomes via a single batched operation.
- All key allocations and buffer shapes are fixed at initialization, enabling zero dynamic allocation in the inner evolution loop.

---

## 7.5) Built-in Selection Operators — Optimizations & Behavior

### TournamentSelection
- **Strategy**: Randomly sample `tournament_size` candidates per tournament, return winner (highest fitness)
- **Determinism**: Stochastic (requires 1 PRNG key)
- **Fitness Requirements**: Works with any fitness range (positive, negative, mixed)
- **Computational Complexity**: O(num_selections × tournament_size)
- **Selection Pressure**: Controlled by tournament_size:
  - tournament_size=2: High diversity, mild exploitation
  - tournament_size=3: Balanced (default, recommended for most problems)
  - tournament_size=7+: High exploitation, lower diversity
- **Elite Handling**: Uses default `get_elite_indices()` (separate O(N) argpartition when elitism on)
- **When to Use**: General-purpose problems, when fitness range is unknown, when diversity is important

### RouletteSelection
- **Strategy**: Select indices proportional to fitness using softmax transformation (temperature-controlled)
- **Determinism**: Stochastic (requires 1 PRNG key)
- **Fitness Requirements**: ⚠️ **Non-negative only** (negative fitness causes NaN in softmax)
- **Computational Complexity**: 
  - With Gumbel-Max (full replacement): O(N log N) due to sorting
  - With categorical: O(N + num_selections)
- **Temperature Parameter**: Controls selection pressure:
  - temperature=0.1: Very aggressive (exploitation)
  - temperature=1.0: Balanced fitness-proportional (default)
  - temperature=5.0: Uniform-ish (exploration)
- **Gumbel-Max Optimization**: 
  - Only active when `use_gumbel_trick=True` AND `num_selections == pop_size`
  - Condition: Full population replacement (standard EA survivor selection)
  - Falls back to categorical sampling if num_selections < pop_size (slow path)
  - `chunk_size` parameter: Memory vs speed tradeoff for large populations
- **Elite Handling**: Uses default `get_elite_indices()` (separate pass when elitism on)
- **When to Use**: Well-characterized fitness landscapes, fitness-weighted selection desired, all fitness values guaranteed non-negative

### ElitePoolSelection
- **Strategy**: Select top `elite_k` individuals, then sample parents uniformly from this pool
- **Determinism**: Stochastic (requires 1 PRNG key)
- **Fitness Requirements**: Works with any fitness range (ranks only)
- **Computational Complexity**: O(N + num_selections) where elite identification is O(N) argpartition
- **Elite Fusion Optimization**: 
  - **Key advantage**: When elitism is enabled, fuses both parent selection AND elite identification into single `argpartition` call
  - **How it works**: `k = min(max(elite_k, n_elites), pop_size)`, then single argpartition on combined k
  - **Performance**: Avoids second O(N) scan that tournament/roulette require when elitism is on
  - **Interaction**: Most beneficial when `elite_k >= n_elites` (no nested sorting needed)
  - **Fallback**: When `elite_k != n_elites`, performs nested sort within top-k (still better than two separate passes)
- **Selection Pressure**: Very high (only top elite_k contribute genes)
- **When to Use**: Late optimization phase, fine-tuning near optima, hybrid approaches (mix with tournament), small elite_k (5-10%)

---

## 8) Technical Summary

- **Input/Output Contract**: 
  - **Input**: Fitness values (either from Population object or raw array) + optional PRNG keys
  - **Output**: Tuple `(parent_idx, elite_idx)` where:
    - `parent_idx`: shape `(num_selections,)`, indices for parents (may have duplicates)
    - `elite_idx`: shape `(n_elites,)`, indices for preserved elites (no duplicates, top-k)
- **Key Budgeting**: Total keys needed = `num_keys_per_atomic_operation` (typically 1 for stochastic, 0 for deterministic).
- **Decoupled Slicing**: Selection returns indices (not reordered genomes) to minimize memory overhead and enable single gather operation in engine.
- **Type Safety**: Indices are explicitly int32/int64 to prevent accidental promotion during downstream indexing.
- **Elite Properties**: Elite indices guaranteed to correspond to highest-fitness individuals; if `n_elites=0`, elite_idx is empty array `jnp.zeros(0, dtype=jnp.int32)`.

---

## 10) Developer Checklist — Implementing a Selection Operator

### Core Requirements
- [ ] Define `num_keys_per_atomic_operation` (0 for deterministic, ≥1 for stochastic).
- [ ] Implement `_select(keys: chex.Array, fitness: chex.Array, config: Optional[C]) -> indices` as a pure function.
- [ ] Return an integer `jnp.ndarray` of indices with shape `(num_selections,)` and dtype int32/int64.
- [ ] Handle both PRNG formats in `_select()`:
  ```python
  if self.typed_keys:
      rng = keys if keys.ndim == 0 else keys[0]  # New-style: scalar
  else:
      rng = keys if keys.ndim <= 1 else keys[0]  # Legacy: (2,) pair
  ```
- [ ] Document the selection logic and any required config attributes.
- [ ] Add unit tests for correctness, shape contracts, and both PRNG formats.
- [ ] Verify `num_keys()` returns the correct total key budget (typically just `K`, not `S * K`).

### Optional Optimizations
- [ ] **Elite Fusion** (if applicable): Override `__call__()` to fuse parent and elite selection into single pass (like `ElitePoolSelection` does with `argpartition`). Document the complexity savings.
- [ ] **Conditional Behavior** (if applicable): Implement fast paths for common cases (e.g., Roulette's Gumbel-Max when `num_selections == pop_size`).
- [ ] **Elite Preservation Notes**: If operator interacts with elitism, document how elite indices are determined and any performance implications.

### Documentation
- [ ] Include comprehensive docstring with:
  - Selection strategy explanation
  - Parameter descriptions with valid ranges
  - Fitness requirements (e.g., must be non-negative for Roulette)
  - When to use this operator (use cases, problem types)
  - Computational complexity with and without elitism
  - Example string specification format
- [ ] Cross-reference with BaseSelection if overriding methods
- [ ] Document shape contracts for both parent_idx output (when used alone) and elite_idx (when elitism is on)

---

## Practical Usage Examples

### Basic Tournament Selection

```python
import jax.random as jr
import jax.numpy as jnp
from malthusjax.operators.selection import TournamentSelection
from malthusjax.core.genome import RealPopulation, RealGenomeConfig

# Setup: 10 individuals, select 4 parents
key = jr.PRNGKey(42)
config = RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0))
pop = RealPopulation.init_random(key, config, size=10)

# Assign fitness values (higher is better)
fitness = jnp.array([1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 6.0, 8.0, 0.5])
pop = RealPopulation(genes=pop.genes, fitness=fitness, config=config)

# Create selection operator
selection = TournamentSelection(num_selections=4, tournament_size=3)
selection = selection.set_input_length(10)

# Allocate PRNG key and select
num_keys = selection.num_keys((10,))
keys = jr.split(key, num_keys)

parent_idx, elite_idx = selection(keys, pop, None)
print(f"Selected parents: {parent_idx}")  # Indices of 4 chosen parents
print(f"Elite: {elite_idx}")               # Empty (no elitism set)
```

### With Elite Preservation

```python
# Same setup as above, but enable elitism
selection = TournamentSelection(num_selections=4, tournament_size=3)
selection = selection.set_input_length(10)
selection = selection.set_n_elites(2)  # Preserve top 2 individuals

# Run selection
keys = jr.split(key, selection.num_keys((10,)))
parent_idx, elite_idx = selection(keys, pop, None)

print(f"Selected parents: {parent_idx}")  # 4 parent indices
print(f"Elite: {elite_idx}")               # Indices of top 2 (e.g., [3, 8])
# elite_idx[0] is highest fitness (9.0), elite_idx[1] is second highest (8.0)
```

### Population Fallback: Fitness Array Input

```python
# You can pass fitness array directly (without Population object)
fitness_array = jnp.array([1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 6.0, 8.0, 0.5])

parent_idx, elite_idx = selection(keys, fitness_array, None)
# Works identically to population-based selection above
```

### Roulette Selection with Temperature Control

```python
from malthusjax.operators.selection import RouletteSelection

# High temperature = more exploration (uniform-like)
selection_explore = RouletteSelection(num_selections=6, temperature=5.0)
selection_explore = selection_explore.set_input_length(10)

# Low temperature = more exploitation (peaks on highest fitness)
selection_exploit = RouletteSelection(num_selections=6, temperature=0.1)
selection_exploit = selection_exploit.set_input_length(10)

keys_explore = jr.split(key, selection_explore.num_keys((10,)))
keys_exploit = jr.split(key, selection_exploit.num_keys((10,)))

idx_explore, _ = selection_explore(keys_explore, pop, None)
idx_exploit, _ = selection_exploit(keys_exploit, pop, None)

# idx_explore will have more uniform distribution across all individuals
# idx_exploit will heavily favor individuals 3 and 8 (high fitness)
```

### Elite Pool Selection

```python
from malthusjax.operators.selection import ElitePoolSelection

# Select parents from top 3 individuals
selection = ElitePoolSelection(num_selections=5, elite_k=3)
selection = selection.set_input_length(10)

keys = jr.split(key, selection.num_keys((10,)))
parent_idx, elite_idx = selection(keys, pop, None)

# All 5 parents come from the top 3 individuals (sampled with replacement)
# Fastest convergence (high exploitation)
```

---

## References

### Core Architecture
- `malthusjax.operators.base.BaseSelection` — Abstract base class and `__call__` interface
- `malthusjax.engine.resource_mapper.ResourceMap` — Key budgeting and allocation
- `malthusjax.engine.resource_mapper.ShardingManager` — Multi-device sharding utilities

### Built-in Implementations
- `malthusjax.operators.selection.TournamentSelection` — Balanced selection via competitive tournaments
- `malthusjax.operators.selection.RouletteSelection` — Fitness-proportional selection with temperature control
- `malthusjax.operators.selection.ElitePoolSelection` — High-exploitation elite pool selection with fusion optimization

### Related Documentation
- **Fitness Module**: See `malthusjax/core/fitness/README.md` for objective function design
- **Genome Module**: See `malthusjax/core/genome/README.md` for population structures
- **Engine Integration**: See `malthusjax/engine/README.md` for full evolution loop details