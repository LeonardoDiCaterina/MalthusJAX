# Crossover Operators — Architecture & Implementation

This document describes the three-tier crossover architecture in MalthusJAX and implementation best practices for developing JAX-native crossover operators.

---

## Overview

MalthusJAX crossover operators implement genetic recombination, combining genetic material from two parent genomes to create offspring. The design follows a three-tier separation:

- **Tier 1**: Recombination kernel (pure deterministic selection)
- **Tier 2**: Noise/mask generation (RNG, produces crossover masks or indices)
- **Tier 3**: Vectorized wrapper (population-level vmap orchestration)

This separation enables static resource budgeting, reproducibility, and XLA kernel fusion.

---

## Tier 1: Recombination Kernel (Pure Deterministic Selection)

- **Contract**: `_recombine_one(p1: G, p2: G, noise_data: Any, config: C) -> G`
- **Responsibilities**:
  - Implement the arithmetic for combining one parent pair.
  - Use deterministic selection (e.g., `jnp.where(mask, p2.values, p1.values)`).
  - Avoid Python branching in hot paths to maximize XLA fusion.
  - Explicitly set dtype for numeric arrays (e.g., `dtype=config.dtype`) to avoid implicit type promotion.
  - Return a single offspring genome (not a tuple) — vectorization is handled by Tier 3.

**Mask Convention**:
- `mask=False` → select from Parent 1
- `mask=True` → select from Parent 2
- Example: `offspring = jnp.where(mask, p2.values, p1.values)`

---

## Tier 2: Noise/Mask Generation (RNG, Produces Entropy)

- **Contract**: `_generate_noise(keys: chex.Array, config: C) -> noise_data`
- **Responsibilities**:
  - Consume a fixed slice of PRNG keys (determined by `num_keys_per_atomic_operation`).
  - Produce masks (Bernoulli), crossover points (randint), or blend parameters (uniform floats).
  - Explicitly set dtype for numeric arrays (e.g., `dtype=config.dtype`).
  - Return noise shaped to `config.shape` (vmapped by Tier 3).

**Implementation Examples**:

UniformCrossover (uses 1 key for Bernoulli mask):
```python
def _generate_noise(self, keys: chex.PRNGKey, config: RealGenomeConfig) -> chex.Array:
    return jax.random.bernoulli(keys[0], p=self.crossover_rate, shape=config.shape)
```

BlendCrossover (uses 2 keys — one for decision, one for blend values):
```python
def _generate_noise(self, keys: chex.PRNGKey, config: RealGenomeConfig) -> Tuple[chex.Array, chex.Array]:
    k_do, k_val = keys[0], keys[1]
    should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)
    random_samples = jax.random.uniform(k_val, shape=config.shape, dtype=config.dtype)
    return should_cross, random_samples
```

---

## Tier 3: Vectorized Wrapper (Population-Level Orchestration)

`BaseCrossover.__call__` orchestrates the population-level crossover via nested vmap:

- **Responsibilities**:
  - Reshape pre-allocated keys to `(num_pairs, num_offspring, num_keys_per_atomic_operation, 2)`.
  - Use nested vmap: outer over parent pairs, inner over offspring per pair.
  - Each offspring gets its own deterministic key block.
  - Fuse RNG + arithmetic via `_cross_fused()` for XLA kernel fusion.
  - Transpose and flatten offspring from pair-major `(num_pairs, num_offspring, ...)` to offspring-major `(num_offspring * num_pairs, ...)`.
  - Return new population via `spawn_offspring()` with reset fitness.

**Key Reshaping**:

```python
all_keys.reshape(num_pairs, num_offspring, num_keys_per_atomic_operation, 2)
```

**Offspring-Major Flattening**:

```python
def flatten_fn(x: chex.Array) -> chex.Array:
    # x shape: (num_pairs, num_offspring, ...)
    transposed = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))  # -> (num_offspring, num_pairs, ...)
    return transposed.reshape((-1,) + transposed.shape[2:])  # -> (num_offspring * num_pairs, ...)
```

**Result**: All offspring 0 from all pairs come first, then offspring 1, etc. This is the **offspring-major** order used throughout MalthusJAX.

---

## Resource Mapping & Static RNG Budget

The `ResourceMap` (computed by `compute_resource_map()`) pre-calculates RNG budget:

1. Calls `operator.num_keys(input_shape)` for each crossover operator.
2. Computes `total_rng_budget = num_pairs * num_offspring * num_keys_per_atomic_operation`.
3. Produces per-operator `OperatorAllocation` with start/end indices for key slices.

**Benefits**:
- Static memory allocation for key buffers.
- Deterministic key slicing across all operators.
- Predictable device placement.

---

## Key Features

**Nested vmap structure** ensures efficient vectorization:
- Outer vmap iterates over parent pairs (count: `num_pairs`).
- Inner vmap iterates over offspring per pair (count: `num_offspring`).
- Each offspring gets a unique key block for reproducibility.

**XLA kernel fusion** combines RNG generation and recombination:
- `_cross_fused()` calls `_generate_noise()` then `_recombine_one()` in sequence.
- XLA merges both operations into a single compiled kernel.
- Result: minimal device overhead and maximal throughput on accelerators.

---

## Complete Example: UniformCrossover

Assume: 4 parent pairs, 2 offspring per pair, crossover_rate=0.7, genome shape (5,).

**Step 1: Resource Allocation**
```python
op = UniformCrossover(num_offspring=2, crossover_rate=0.7).set_input_length(4)
# ResourceMapper: total_keys = 4 * 2 * 1 = 8
all_keys = jr.split(master_key, 8).reshape((4, 2, 1, 2))
# Shape: (num_pairs=4, num_offspring=2, num_keys_per_atomic_operation=1, key_dim=2)
```

**Step 2: Nested vmap (Tier 3)**
```python
keys_reshaped = all_keys.reshape((4, 2, 1, 2))
nested_offspring = jax.vmap(
    lambda k_block, g1, g2: jax.vmap(
        lambda k: self._cross_fused(k, g1, g2, config)
    )(k_block)
)(keys_reshaped, p1_pop.genes, p2_pop.genes)
# Output shape: (4, 2, 5) — 4 pairs × 2 offspring × 5 genes
```

**Step 3: Tier 2 — Generate Bernoulli Mask**
```python
def _generate_noise(self, keys, config):
    return jax.random.bernoulli(keys[0], p=0.7, shape=(5,))
    # Returns: boolean array shape (5,)
```

**Step 4: Tier 1 — Per-Gene Selection**
```python
def _recombine_one(self, p1, p2, noise_data, config):
    mask = noise_data  # Shape (5,), dtype bool
    offspring = jnp.where(mask, p2.values, p1.values)
    return p1.replace(values=offspring)
```

**Step 5: Offspring-Major Flattening**
```python
# nested shape: (4, 2, 5)
transposed = jnp.transpose(nested_offspring, (1, 0, 2))  # -> (2, 4, 5)
flattened = transposed.reshape((8, 5))  # -> (num_offspring * num_pairs, 5)
# Result: [offspring0_pair0, offspring0_pair1, offspring0_pair2, offspring0_pair3,
#          offspring1_pair0, offspring1_pair1, offspring1_pair2, offspring1_pair3]
```

---

## Mask Semantics

All crossover operators follow the convention:
- `mask=False` → inherit from Parent 1
- `mask=True` → inherit from Parent 2

**Implementation pattern**:
```python
offspring = jnp.where(mask, p2.values, p1.values)  # True -> p2, False -> p1
```

**Tests verify**:
- All-False mask → offspring == p1
- All-True mask → offspring == p2
- Mixed mask → correct per-gene inheritance

---

## Static Metadata & JIT Compatibility

- Mark static attributes as `pytree_node=False` (e.g., `crossover_rate`, `num_offspring`).
- JAX receives concrete shapes at compile-time, enabling optimization.
- Use `@struct.dataclass` for all operator definitions (Flax PyTree registration).

---

## Testing

The codebase provides tests that validate:

- **Tier 2 shape/dtype**: `_generate_noise` returns correct shape and dtype.
- **Tier 1 correctness**: 
  - All-False mask → offspring == p1
  - All-True mask → offspring == p2
  - Mixed mask → correct per-gene inheritance
- **Crossover rate**: Empirical rate matches parameter (within ±10%).
- **Dtype safety**: No implicit type promotion with `config.dtype=jnp.bfloat16`.
- **Offspring-major ordering**: Flattened offspring match expected layout.

---

## Performance & XLA Fusion

XLA merges RNG generation and arithmetic operations into single kernels:

- `_cross_fused()` calls `_generate_noise()` then `_recombine_one()`.
- XLA fuses both into one HLO kernel, minimizing device overhead.
- Result: minimal memory transfers and maximal throughput on accelerators.

---

## Developer Checklist

When implementing a new crossover operator:

**Core Implementation**
- [ ] Define `num_keys_per_atomic_operation` (e.g., 1 for Bernoulli, 2 for blend + decision).
- [ ] Implement `_generate_noise(keys, config)`:
  - Extract individual keys: `k1, k2 = keys[0], keys[1]`.
  - Return array shaped to `config.shape` with explicit dtype.
- [ ] Implement `_recombine_one(p1, p2, noise_data, config)`:
  - Pure arithmetic only (no Python branching in hot path).
  - Use `jnp.where(mask, p2.values, p1.values)` for mask semantics.
  - Return single genome (not tuple).

**Inheritance** (Automatic)
- [ ] Inherit from `BaseCrossover` to get Tier 3 vmap support automatically.

**Testing**
- [ ] Add tests in `tests/operators/crossover/`:
  - Verify shape/dtype of `_generate_noise()`.
  - Test all-False → p1, all-True → p2 mask cases.
  - Verify crossover rate empirically.
  - Check offspring-major ordering.

---

## Available Crossover Operators

**Binary Crossovers** (for BinaryGenome)
- **UniformCrossover**: Independently select each bit from Parent 1 or Parent 2 with probability `crossover_rate`.
- **SinglePointCrossover**: Select random crossover point and swap segments.

**Real-Valued Crossovers** (for RealGenome)
- **UniformCrossover**: Independently select each gene from Parent 1 or Parent 2 with probability `crossover_rate`.
- **BlendCrossover (BLX-α)**: Sample uniformly from extended interval around parents.
- **SimulatedBinaryCrossover (SBX)**: Polynomial distribution-based crossover with parameter `eta`.
- **BinomialCrossover**: Differential evolution style selection.
- **EvosaxUniformCrossoverWrapper**: Wrapper around Evosax native uniform crossover.

---

## Operator Selection Guide

| Problem Type | Recommended Operator | Parameters | Notes |
|--------------|---------------------|------------|-------|
| Binary / Combinatorial | Uniform Crossover | rate=0.5 | High disruption, good exploration |
| Binary / Building Blocks | Single-Point | default | Preserves adjacency |
| Real / Independent Genes | Uniform Crossover | rate=0.6 | Simple per-gene mixing |
| Real / Exploration | Blend (BLX-α) | α=0.5 | Explores outside parental range |
| Real / Exploitation | SBX | η=20-30 | Parent-centric, adaptive |
| Differential Evolution | Binomial | rate=0.5 | Directional, mutant-biased |

---

## Architecture References

- [base.py](../base.py) — `BaseCrossover` (Tier 3 vmap orchestration)
- [binary.py](./binary.py) — Binary crossover implementations
- [real.py](./real.py) — Real-valued crossover implementations
- [evosax_crossover.py](./evosax_crossover.py) — Evosax wrapper
- [tests/operators/crossover/](../../tests/operators/crossover/) — Test suite

---

## Quick Summary: 3-Tier Paradigm

1. **Tier 1 (Pure Arithmetic)**: `_recombine_one(p1, p2, noise_data, config) -> offspring`
   - Deterministic selection: `jnp.where(mask, p2, p1)`
   - No randomness or Python branching

2. **Tier 2 (RNG)**: `_generate_noise(keys, config) -> noise_data`
   - Produces Bernoulli masks, uniform samples, or crossover points
   - Shapes match `config.shape`, dtypes explicit

3. **Tier 3 (Vectorization)**: `BaseCrossover.__call__(all_keys, p1_pop, p2_pop, config) -> offspring_pop`
   - Nested vmaps over pairs and offspring
   - Offspring-major flattening for consistency

Together: **JIT compilation**, **reproducibility**, **static budgeting**, and **XLA kernel fusion**.
