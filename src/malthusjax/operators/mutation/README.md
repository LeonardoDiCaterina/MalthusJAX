# Mutation Operators — Architecture & Implementation

This document describes the three-tier mutation architecture in MalthusJAX and implementation best practices for developing JAX-native mutation operators.

---

## Overview

MalthusJAX mutation operators implement stochastic perturbation, introducing variability into candidate solutions. The design follows a three-tier separation:

- **Tier 1**: Arithmetic kernel (pure deterministic application of mutations)
- **Tier 2**: Noise generation (RNG, produces random perturbations or masks)
- **Tier 3**: Vectorized wrapper (population-level vmap orchestration)

This separation enables static resource budgeting, reproducibility, and XLA kernel fusion.

---

## Tier 1: Arithmetic Kernel (Pure Deterministic Perturbation)

- **Contract**: `_mutate_one(genome: G, noise_data: Any, config: C) -> G`
- **Responsibilities**:
  - Implement the arithmetic for mutating one individual using masks and noise (e.g., `genome.values + (noise * mask)`).
  - Avoid Python branching in hot paths to maximize XLA fusion.
  - Explicitly set dtype for numeric arrays (e.g., `dtype=config.dtype`) to avoid implicit type promotion.
  - Return a new, immutable genome dataclass (e.g., `genome.replace(values=new_values)`).

**Example (GaussianMutation)**:
```python
def _mutate_one(self, genome: RealGenome, noise_data: chex.Array, config: RealGenomeConfig, **kwargs: Any) -> RealGenome:
    mutated_values = genome.values + noise_data
    if self.clip:
        min_val, max_val = config.bounds
        mutated_values = jnp.clip(mutated_values, min_val, max_val)
    return cast(RealGenome, cast(Any, genome).replace(values=mutated_values))
```

---

## Tier 2: Noise Generation (RNG, Produces Entropy)

- **Contract**: `_generate_noise(keys: chex.Array, config: C) -> noise_data`
- **Responsibilities**:
  - Consume a fixed slice of PRNG keys (determined by `num_keys_per_atomic_operation`).
  - Produce masks (Bernoulli), Gaussian/Uniform noise, or domain-specific stochastic values shaped to `config.shape`.
  - Explicitly set dtype for all numeric arrays (e.g., `dtype=config.dtype`) to avoid implicit type promotion.
  - Return noise shaped to `config.shape` (vmapped by Tier 3).

**Example (GaussianMutation)**:
```python
def _generate_noise(self, keys: chex.Array, config: RealGenomeConfig) -> chex.Array:
    k_mask, k_noise = keys[0], keys[1]
    dtype = config.dtype
    
    # Bernoulli mask (which components to mutate)
    mask_bool = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=config.shape)
    mask_val = mask_bool.astype(dtype)
    
    # Gaussian noise scaled by strength
    raw_noise = jax.random.normal(k_noise, shape=config.shape, dtype=dtype)
    strength = jnp.array(self.mutation_strength, dtype=dtype)
    
    return raw_noise * strength * mask_val
```

---

## Tier 3: Vectorized Wrapper (Population-Level Orchestration)

`BaseMutation.__call__` orchestrates the population-level mutation via nested vmap:

- **Responsibilities**:
  - Reshape pre-allocated keys to `(pop_size, num_offspring, num_keys_per_atomic_operation, 2)`.
  - Use nested vmap: outer over parents, inner over offspring per parent.
  - Each offspring gets its own deterministic key block.
  - Fuse RNG + arithmetic via `_mutate_fused()` for XLA kernel fusion.
  - Flatten offspring from `(pop_size, num_offspring, ...)` to `(pop_size * num_offspring, ...)` via direct `reshape` (no transpose).
  - Return new population via `spawn_offspring()`.

**Key Reshaping**:

```python
all_keys.reshape(pop_size, num_offspring, num_keys_per_atomic_operation, 2)
```

**Pair-Major Flattening**:

```python
def flatten_fn(x: chex.Array) -> chex.Array:
    # x shape: (pop_size, num_offspring, ...)
    # Direct reshape — no transpose. Same rationale as crossover (see FB-1).
    return x.reshape((-1,) + x.shape[2:])  # -> (pop_size * num_offspring, ...)
```

**Result**: All offspring for parent 0 come first, then all offspring for parent 1, etc. This is the **pair-major** order (consistent with crossover).

---

## Resource Mapping & Static RNG Budget

The `ResourceMap` (computed by `compute_resource_map()`) pre-calculates RNG budget:

1. Calls `operator.num_keys(input_shape)` for each mutation operator.
2. Computes `total_rng_budget = pop_size * num_offspring * num_keys_per_atomic_operation`.
3. Produces per-operator `OperatorAllocation` with start/end indices for key slices.

**Benefits**:
- Static memory allocation for key buffers.
- Deterministic key slicing across all operators.
- Predictable device placement.

---

## Key Features

**Nested vmap structure** ensures efficient vectorization:
- Outer vmap iterates over parents (count: `pop_size`).
- Inner vmap iterates over offspring per parent (count: `num_offspring`).
- Each offspring gets a unique key block for reproducibility.

**XLA kernel fusion** combines RNG generation and mutation:
- `_mutate_fused()` calls `_generate_noise()` then `_mutate_one()` in sequence.
- XLA merges both operations into a single compiled kernel.
- Result: minimal device overhead and maximal throughput on accelerators.

---

## Complete Example: GaussianMutation

Assume: population of 4 genomes (shape 10,), 2 offspring per parent, mutation_rate=0.5, strength=0.3, clip=True.

**Step 1: Resource Allocation**
```python
op = GaussianMutation(num_offspring=2, mutation_rate=0.5, mutation_strength=0.3).set_input_length(4)
# ResourceMapper: total_keys = 4 * 2 * 2 = 16
all_keys = jr.split(master_key, 16).reshape((4, 2, 2, 2))
# Shape: (pop_size=4, num_offspring=2, num_keys_per_atomic_operation=2, key_dim=2)
```

**Step 2: Nested vmap (Tier 3)**
```python
keys_reshaped = all_keys.reshape((4, 2, 2, 2))
nested_offspring = jax.vmap(
    lambda k_block, genome: jax.vmap(
        lambda k: self._mutate_fused(k, genome, config)
    )(k_block)
)(keys_reshaped, population.genes)
# Output shape: (4, 2, 10) — 4 parents × 2 offspring × 10 genes
```

**Step 3: Tier 2 — Generate Noise**
```python
def _generate_noise(self, keys, config):
    k_mask, k_noise = keys[0], keys[1]
    mask = jax.random.bernoulli(k_mask, p=0.5, shape=(10,)).astype(jnp.float32)
    noise = jax.random.normal(k_noise, shape=(10,), dtype=jnp.float32) * 0.3 * mask
    return noise  # Shape (10,)
```

**Step 4: Tier 1 — Mutate Individual**
```python
def _mutate_one(self, genome, noise_data, config):
    mutated = genome.values + noise_data
    if self.clip:
        mutated = jnp.clip(mutated, -5.0, 5.0)
    return genome.replace(values=mutated)
```

**Step 5: Pair-Major Flattening**
```python
# nested shape: (4, 2, 10)
flattened = nested_offspring.reshape((8, 10))  # -> (pop_size * num_offspring, 10)
# Result: [parent0_offspring0, parent0_offspring1,
#          parent1_offspring0, parent1_offspring1,
#          parent2_offspring0, parent2_offspring1,
#          parent3_offspring0, parent3_offspring1]
```

---

## Static Metadata & JIT Compatibility

- Mark static attributes as `pytree_node=False` (e.g., `mutation_rate`, `num_offspring`).
- JAX receives concrete shapes at compile-time, enabling optimization.
- Use `@struct.dataclass` for all operator definitions (Flax PyTree registration).

---

## Testing

The codebase provides tests that validate:

- **Tier 2 shape/dtype**: `_generate_noise` returns correct shape and dtype.
- **Tier 1 correctness**: `_mutate_one` produces expected arithmetic results (e.g., zero noise → no change).
- **Statistical properties**: Mean/std of generated noise match expected distributions.
- **Dtype safety**: No implicit type promotion with `config.dtype=jnp.bfloat16`.
- **Pair-major ordering**: Flattened offspring match expected layout (direct `reshape`, no transpose).

---

## Performance & XLA Fusion

XLA merges RNG generation and arithmetic operations into single kernels:

- `_mutate_fused()` calls `_generate_noise()` then `_mutate_one()`.
- XLA fuses both into one HLO kernel, minimizing device overhead.
- Result: minimal memory transfers and maximal throughput on accelerators.

---

## Developer Checklist

When implementing a new mutation operator:

**Core Implementation**
- [ ] Define `num_keys_per_atomic_operation` (e.g., 1 for BitFlip, 2 for Gaussian + mask).
- [ ] Implement `_generate_noise(keys, config)`:
  - Extract individual keys: `k1, k2 = keys[0], keys[1]`.
  - Return array shaped to `config.shape` with explicit dtype.
- [ ] Implement `_mutate_one(genome, noise_data, config)`:
  - Pure arithmetic only (no Python branching in hot path).
  - Use masked operations: `value + noise * mask`.
  - Return single genome (not tuple).

**Inheritance** (Automatic)
- [ ] Inherit from `BaseMutation` to get Tier 3 vmap support automatically.

**Testing**
- [ ] Add tests in `tests/operators/mutation/`:
  - Verify shape/dtype of `_generate_noise()`.
  - Test zero-noise case (genome unchanged).
  - Verify mutation_rate empirically over 100+ samples.
  - Check pair-major ordering.
  - Check no dtype promotion.

---

## Available Mutation Operators

**Binary Mutations** (for BinaryGenome)
- **BitFlipMutation**: Independently flip each bit with probability `mutation_rate`.
- **ScrambleMutation**: Randomly permute bit positions with probability `mutation_rate`.
- **SwapMutation**: Randomly swap two positions with probability `mutation_rate`.

**Real-Valued Mutations** (for RealGenome)
- **GaussianMutation**: Add Gaussian noise to components selected by Bernoulli mask.
- **BallMutation**: Sample uniformly within an n-dimensional ball of fixed radius.
- **PolynomialMutation**: Use polynomial distribution parameterized by `eta`.
- **EvosaxGaussianWrapper**: Wrapper around Evosax native Gaussian mutation.

## Architecture References

- [base.py](./base.py) — `BaseMutation` (Tier 3 vmap orchestration)
- [binary.py](./binary.py) — Binary mutation implementations
- [real.py](./real.py) — Real-valued mutation implementations
- [evosax_mutation.py](./evosax_mutation.py) — Evosax wrapper
- [tests/operators/mutation/](../../tests/operators/mutation/) — Test suite