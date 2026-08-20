# `malthusjax.operators` — Reference

Scope: `malthusjax.operators.base`, `malthusjax.operators.base_ablation`, `malthusjax.operators.base_injection`, `malthusjax.operators.selection.*`, `malthusjax.operators.crossover.*`, `malthusjax.operators.mutation.*`, `malthusjax.operators.emitters.*`. Every claim below is traceable directly to source code and unit tests.

---

## Overview & 3-Tier Architecture

In MalthusJAX, operators are pure functions (or `@struct.dataclass` PyTrees) designed for XLA kernel fusion and stateless GPU execution. Operators decouple domain math from vectorization via a **3-tier hierarchy**:

- **Tier 1 (Pure Arithmetic)**: `_mutate_one(genome, noise, config)` or `_recombine_one(p1, p2, noise, config)` operates on a single structural `Genome` PyTree instance `G`. Array-native equivalents (`_apply_noise`, `_apply_mask`) operate directly on flat `jax.Array` leaves for engine scan carries.
- **Tier 2 (RNG Generation)**: `_generate_noise(keys, config, generation=0)` consumes pre-allocated PRNG keys and produces a noise PyTree (Bernoulli masks, Gaussian noise, random indices).
- **Tier 3 (Population Vectorization)**: `__call__` orchestrates JAX `vmap` calls over batched `BasePopulation[G]` containers, treating `G` as an opaque PyTree without assuming a contiguous `.values` array.

---

## `malthusjax.operators.base`

### `BaseMutation[G, C]`
**Purpose:** Base class for vectorized mutation operators with pre-allocated key budgeting.

**Public API:**
- `__call__(all_keys, population, config, generation=0) -> BasePopulation[G]` — Tier-3 entry point; applies flat `vmap` (when `num_offspring == 1`) or nested `vmap` (when `num_offspring > 1`), returning offspring stacked behind parents.
- `apply_fastpath(all_keys, flat_values, config, generation=0) -> chex.Array` — bypasses PyTree overhead by mapping `_apply_noise` across flat arrays.
- `num_keys_per_atomic_operation` — abstract property (keys per individual/offspring pair).
- `num_keys(input_shape) -> int` — total key count required for pre-allocation by `ResourceMapper`.
- `set_input_length(length)` / `set_typed_keys(typed)` / `set_max_generations(n)` — returns an updated PyTree dataclass copy.

### `BaseCrossover[G, C]`
**Purpose:** Base class for vectorized crossover operators operating on paired parent populations.

**Public API:**
- `__call__(all_keys, p1_pop, p2_pop, config, generation=0) -> BasePopulation[G]` — receives two parent populations `p1_pop` and `p2_pop` and recombines corresponding pairs.
- `cross_single_pair(key, p1, p2, config, generation=0) -> G` — recombines a single parent pair outside `__call__` by splitting `key`.
- `apply_fastpath(all_keys, p1_values, p2_values, config, generation=0) -> chex.Array` — array-native fast path for flat parent tensors.

### `BaseSelection[P, C]`
**Purpose:** Stateless selection operator for fitness-based parent and elite index sampling.

**Public API:**
- `__call__(keys, population, config=None) -> Tuple[chex.Array, chex.Array]` — returns `(parent_indices, elite_indices)` with shapes `(num_selections,)` and `(n_elites,)`.
- `get_elite_indices(fitness) -> chex.Array` — extracts indices of the `n_elites` best individuals (lowest fitness) using O(N) `jnp.argpartition`.
- `set_n_elites(n)` / `set_input_length(length)` — static configuration helpers.

---

## Operational Variants

### Ablation Mode (`base_ablation.py`)
- `@ablation_single_key_mutation` / `@ablation_single_key_crossover`: Class decorators that modify operators to consume a single PRNG key and split it internally. Used to benchmark dynamic key-splitting overhead vs `ResourceMapper` static key pre-allocation.

### Injection Mode (`base_injection.py`)
- `BaseMutation_injection` / `BaseCrossover_injection`: Single-key mode operators where `_generate_noise(single_key, config)` generates and materializes full noise tensors up front. Enables exact noise replay and deterministic trajectory testing.

---

## `malthusjax.operators.selection`

| Class | Algorithm | Complexity | Notes |
|---|---|---|---|
| `TournamentSelection` | Sample `tournament_size` (default 3) candidates with replacement; pick minimum fitness. | O(num_selections × tournament_size) | Robust across all fitness ranges (positive/negative). |
| `RouletteSelection` | Fitness-proportional selection via softmax logits + Gumbel-Max trick. | O(N log N) or O(N + num_selections) | **Requires non-negative fitness values**. `temperature` controls selection pressure. |
| `ElitePoolSelection` | Filter top `elite_k` (default 10) individuals via `jnp.argpartition`; sample uniformly. | O(N + num_selections) | High exploitation, lower population diversity. |
| `EvoSaxMimicSelection` | Mimics EvoSAX selection semantics via `jnp.argsort` and `jax.random.choice`. | O(N log N) | Used for exact parity verification with EvoSAX. |

---

## `malthusjax.operators.crossover`

### Binary (`crossover/binary.py`)
- `UniformCrossover`: Per-bit independent selection from parents via Bernoulli mask (`crossover_rate` default 0.5).
- `SinglePointCrossover`: Selects a random crossover point in `[1, N-1)` via `jax.random.randint` and swaps binary segments.

### Real-Valued (`crossover/real.py`)
- `UniformCrossover`: Gene-wise Bernoulli mask selection.
- `BlendCrossover` (BLX-α): Expands parent interval `[min(p1,p2), max(p1,p2)]` by `±alpha * |p1 - p2|` (default `alpha=0.5`, `crossover_rate=0.9`) and samples uniformly from the expanded region.
- `SBXCrossover`: Simulated Binary Crossover simulating single-point binary crossover in continuous space (`eta` default 20.0).

---

## `malthusjax.operators.mutation`

### Binary (`mutation/binary.py`)
- `BitFlipMutation`: Inverts bits with probability `mutation_rate` (default 0.05).

### Real-Valued (`mutation/real.py`)
- `GaussianMutation`: Adds Gaussian noise $N(0, \sigma)$ to genes with probability `mutation_rate` (default 0.1, `mutation_strength` 0.1). Supports strength schedules (`LINEAR_DECAY`, `EXPONENTIAL_DECAY`) and optional bound clipping.
- `PolynomialMutation`: Real-valued polynomial perturbation using distribution index `eta` (default 20.0).

### Categorical (`mutation/categorical.py`)
- `SwapMutation`: Swaps two random positions in a permutation genome.
- `ScrambleMutation`: Randomly shuffles a random slice of positions in a permutation genome.

---

## `malthusjax.operators.emitters`

**Purpose**: Quality-Diversity (QD) emitter interface for topological, grid-based, or complex variation.

**Core Classes (`emitters/base.py`)**:
- `BaseEmitter`: Defines QD emitter contract (`batch_size`, `num_keys`, `init`, `ask`, `tell`).
- `AtomicEmitter`: Enforces single-consumer 3-tier architecture for atomic QD variation.
- `GeneticEmitter` (`genetic.py`): Combines selection, crossover, and mutation into a QD emitter.
- `MixingEmitter` (`mixing.py`): Combines multiple sub-emitters with custom selection probabilities.
- `QDAXReplicaEmitter` (`qdax_replica.py`): Wraps QDAX emitters to run inside MalthusJAX engines.
- `TensorNEATEmitter` (`tensorneat_emitter.py`): Wraps TensorNEAT algorithms for evolving neural topologies.
