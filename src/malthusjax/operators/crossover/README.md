# Crossover Operators — Architecture & Implementation (Operators / Crossover) 🔀

This document describes the three-tier crossover architecture in MalthusJAX, the dual RNG modes (Mode A vs Mode D), and implementation best practices for developing high-performance, JAX-native crossover operators.

---

## Overview

MalthusJAX crossover operators implement genetic recombination, combining genetic material from two parent genomes to create offspring. The design follows the same three-tier separation as mutation operators, enabling static resource budgeting, provable correctness, and maximal HLO fusion.

- **Tier 1 — Recombination Kernel** (pure, deterministic arithmetic)
- **Tier 2 — Noise/Mask Generation** (RNG, consumes pre-allocated key budget)
- **Tier 3 — Vectorized Wrapper** (`BaseCrossover.__call__`, ResourceMapper integration)

This separation allows swapping RNG topologies (per-pair sampling vs bulk injection) without changing the recombination logic.

---

## Tier 1 — Recombination Kernel (Pure & Promotion-Free) ⚖️

- Contract: `_recombine_one(p1: G, p2: G, noise_data: Any, config: C, **kwargs) -> G`
- Responsibilities:
  - Implement the exact arithmetic for one pair using masks, indices, or blend parameters.
  - Use deterministic selection (e.g., `jnp.where(mask, p2.values, p1.values)`).
  - Avoid Python branching to maximize XLA fusion.
  - **Vaccinate** constants with `jnp.array(..., dtype=config.dtype)` to prevent implicit promotion.
  - Return a single offspring genome (not a tuple) — Tier 3 handles repetition via `num_offspring`.

**Mask Convention (Critical):**
- `mask=False` → select from Parent 1 (p1)
- `mask=True` → select from Parent 2 (p2)
- Example: `offspring = jnp.where(mask, p2.values, p1.values)`

**Example (UniformCrossover):**
```python
def _recombine_one(self, p1: RealGenome, p2: RealGenome, noise_data: chex.Array, 
                   config: RealGenomeConfig, **kwargs: Any) -> RealGenome:
    mask = noise_data  # Boolean array of same shape as genome
    # Convention: mask=True -> take from p2, False -> take from p1
    offspring_values = jnp.where(mask, p2.values, p1.values)
    return cast(RealGenome, cast(Any, p1).replace(values=offspring_values))
```

Why:
- Keeping this tier pure and stateless makes it trivially `vmap`-able and enables both Mode A and Mode D (bulk injection).

---

## Tier 2 — Noise/Mask Generation (Entropy Producers) 🎲

- Contract: `_generate_noise(keys: chex.Array, config: C) -> noise_data`
- Responsibilities:
  - Consume an exact slice of PRNG keys (fixed by `num_keys_per_atomic_operation`).
  - Produce masks (Bernoulli), crossover points (randint), or blend parameters (uniform floats).
  - **Ensure dtype correctness**: `dtype=config.dtype` for all numeric arrays.
  - Return noise shaped to `config.shape` (will be vmapped by Tier 3).

Notes:
- This tier is **agnostic** to how keys are sourced — both `BaseCrossover` (Mode A) and `BaseCrossover_injection` (Mode D) use identical implementations.
- For masks, use Bernoulli sampling; for blend parameters, use Uniform.

**Example (UniformCrossover):**
```python
def _generate_noise(self, keys: chex.PRNGKey, config: RealGenomeConfig) -> chex.Array:
    # Bernoulli mask: True with probability crossover_rate
    return jax.random.bernoulli(keys[0], p=self.crossover_rate, shape=config.shape)
```

**Example (BlendCrossover):**
```python
def _generate_noise(self, keys: chex.PRNGKey, config: RealGenomeConfig) -> Tuple[chex.Array, chex.Array]:
    k_do, k_val = keys[0], keys[1]
    dtype = config.dtype
    
    should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)
    random_samples = jax.random.uniform(k_val, shape=config.shape, dtype=dtype)
    
    return should_cross, random_samples
```

---

## Tier 3 — Vectorized Wrapper (Resource-aware Lifting) 🚀

- `BaseCrossover.__call__(all_keys, p1_pop, p2_pop, config, **kwargs)` is the vectorized lifting layer (Mode A).
- `BaseCrossover_injection.__call__(single_key, p1_pop, p2_pop, config, **kwargs)` is the bulk injection variant (Mode D).

### Mode A: Per-Pair Keys (BaseCrossover)
- Responsibilities:
  - Reshape `all_keys` into `(num_pairs, num_offspring, num_keys_per_atomic_operation, 2)`.
  - Use **nested vmap**: outer vmap over parent pairs, inner vmap over offspring per pair.
  - Each offspring uses its own deterministic key block.
  - Fuse RNG + arithmetic in `_cross_fused(keys, p1, p2, config)` for XLA kernel fusion.
  - Flatten offspring in **offspring-major** order: `(num_offspring, num_pairs) -> (-1,)` (all offspring0 across pairs, then offspring1, ...).
  - Return `p1_pop.spawn_offspring(new_genes)` with reset fitness.

### Mode D: Bulk Injection (BaseCrossover_injection)
- Responsibilities:
  - Accept a single `key` and generate one bulk noise tensor of shape `(num_pairs * num_offspring, ...)`.
  - Internally split the key into `num_pairs * num_offspring` subkeys.
  - Pass bulk noise to nested vmap of `_recombine_one` operations.
  - Flatten offspring in **offspring-major** order (same as Mode A for consistency).
  - Enables maximal XLA fusion: RNG generation + per-element arithmetic merged into single kernel.

### Key Reshaping Details
- Mode A: `all_keys.reshape(num_pairs, num_offspring, num_keys_per_atomic_operation, 2)`
- Mode D: internally generates `(num_pairs * num_offspring, num_keys_per_atomic_operation, 2)` from single key.
- Both use Threefry counters, but Mode A uses isolated subkeys while Mode D advances counters per batch index.

### Offspring-Major Flattening (Critical for Consistency)
The fused `BaseCrossover` transposes before flattening to ensure **offspring-major** ordering:

```python
def flatten_fn(x: chex.Array) -> chex.Array:
    # x shape: (num_pairs, num_offspring, ...)
    transposed = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))  # -> (num_offspring, num_pairs, ...)
    return transposed.reshape((-1,) + transposed.shape[2:])  # -> (num_offspring * num_pairs, ...)
```

**Result:** All `num_offspring` from pair 0 come first, then all `num_offspring` from pair 1, etc. This matches injection mode and ensures test parity.

---

## Resource Mapping & Static RNG Budget (ResourceMapper) 🗺️

- The `ResourceMap` (computed by `compute_resource_map(...)`) pre-calculates total RNG budget:
  - Calls `operator.num_keys(input_shape)` for each crossover operator.
  - Computes `total_rng_budget = num_pairs * num_offspring * num_keys_per_atomic_operation`.
  - Produces per-operator `OperatorAllocation` (start/end indices).
- Benefits:
  - Static memory allocation for key buffers.
  - Deterministic slicing across all operators.
  - Predictable device placement and minimal host-device transfers.

---

## PRNG Topologies & Implementation Modes 🔁

### Mode A — Per-Pair Sampling (BaseCrossover)
- **Implementation:** `BaseCrossover.__call__` in [base.py](../base.py#L180)
- **Key topology:** Each of `num_pairs * num_offspring` crossover operations receives a unique PRNG key block.
- **Code flow:**
  ```python
  keys_reshaped = all_keys.reshape(num_pairs, num_offspring, num_keys_per_atomic_operation, 2)
  nested = jax.vmap(
      lambda k_block, g1, g2: jax.vmap(
          lambda k: self._cross_fused(k, g1, g2, config)
      )(k_block)
  )(keys_reshaped, p1_pop.genes, p2_pop.genes)
  # Output: (num_pairs, num_offspring, ...genome_shape)
  
  # Flatten with offspring-major ordering
  transposed = jnp.transpose(nested, (1, 0, 2))
  flattened = transposed.reshape((-1,) + transposed.shape[2:])
  ```
- **Pros:** Conceptually clear, bitwise reproducible with same per-pair keys.
- **Cons:** Multiple small host-side splits reduce XLA fusion potential.

### Mode D — Bulk Injection (BaseCrossover_injection)
- **Implementation:** `BaseCrossover_injection.__call__` in [base_injection.py](../base_injection.py#L220)
- **Key topology:** Single `key` generates one bulk noise tensor `(num_pairs * num_offspring, ...config.shape)` in one RNG call.
- **Code flow:**
  ```python
  noise = self._generate_noise(key, config)  # Returns (N, ...shape) where N = num_pairs * num_offspring
  
  def reshape_noise(x):
      return x.reshape((num_pairs, num_offspring) + x.shape[1:])
  
  noise = jax.tree_util.tree_map(reshape_noise, noise)
  
  nested = jax.vmap(
      lambda noise_block, g1, g2: jax.vmap(
          lambda n: self._recombine_one(g1, g2, n, config)
      )(noise_block)
  )(noise, p1_pop.genes, p2_pop.genes)
  # Output: (num_pairs, num_offspring, ...genome_shape)
  
  # Flatten with offspring-major ordering (same as Mode A)
  ```
- **Pros:** Single RNG call enables XLA to fuse RNG generation + arithmetic into one HLO kernel.
- **Cons:** Bitwise outputs differ from Mode A (counter topology). Tests validate statistical equivalence.

### When to use which mode
- **Mode A (BaseCrossover):** Standard EA use cases where reproducibility with specific keys matters.
- **Mode D (BaseCrossover_injection):** Large-scale optimization on modern accelerators (H100/A100) where kernel fusion and throughput dominate.

### Statistical Equivalence vs Bitwise Divergence
- Both modes produce noise from the same distribution (e.g., Bernoulli(p)).
- **Exact sequences differ** because Mode A uses isolated Threefry counters per pair, while Mode D advances counters within a bulk call.
- Tests verify: distribution properties match, L2 divergence > 0 (confirming different sequences), no dtype promotion.

---

## Complete Example: UniformCrossover (Tier Walkthrough) 🌊

Assume: 4 parent pairs, 2 offspring per pair, crossover_rate=0.7, genome shape (5,).

### Mode A (BaseCrossover) Flow

**Step 1: Tier 3 — Resource Allocation & Reshaping**
```python
op = UniformCrossover(num_offspring=2, crossover_rate=0.7).set_input_length(4)

# ResourceMapper computes: total_keys = 4 * 2 * 1 = 8 (pairs * offspring * keys_per_op)
all_keys = jr.split(master_key, 8).reshape((4, 2, 1, 2))
# Shape: (num_pairs=4, num_offspring=2, num_keys_per_atomic_operation=1, key_dim=2)
```

**Step 2: Tier 3 — Nested vmap orchestration**
```python
keys_reshaped = all_keys.reshape((4, 2, 1, 2))
nested_offspring = jax.vmap(
    lambda k_block, g1, g2: jax.vmap(
        lambda k: self._cross_fused(k, g1, g2, config),
        in_axes=(0,)  # vmap over offspring keys
    )(k_block),
    in_axes=(0, 0, 0)  # vmap over pairs
)(keys_reshaped, p1_pop.genes, p2_pop.genes)
# Output shape: (4, 2, 5) — 4 pairs × 2 offspring × 5 genes
```

**Step 3: Tier 2 — _generate_noise (for each offspring)**
```python
def _generate_noise(self, keys, config):
    return jax.random.bernoulli(keys[0], p=0.7, shape=(5,))
    # Returns shape (5,), dtype bool

# keys shape: (1, 2) — [k0] unpacked
```

**Step 3a: Tier 1 — _recombine_one (for each offspring)**
```python
def _recombine_one(self, p1, p2, noise_data, config):
    mask = noise_data  # Shape (5,), dtype bool
    # Convention: False -> p1, True -> p2
    offspring = jnp.where(mask, p2.values, p1.values)
    return p1.replace(values=offspring)
```

**Step 4: Offspring-Major Flattening**
```python
# nested shape: (4, 2, 5)
transposed = jnp.transpose(nested_offspring, (1, 0, 2))  # -> (2, 4, 5)
flattened = transposed.reshape((8, 5))  # -> (num_offspring * num_pairs, 5)
# Order: offspring0[pair0], offspring0[pair1], offspring0[pair2], offspring0[pair3],
#        offspring1[pair0], offspring1[pair1], offspring1[pair2], offspring1[pair3]

return p1_pop.spawn_offspring(RealGenome(values=flattened))
```

### Mode D (BaseCrossover_injection) Flow

**Same setup, but single key:**
```python
op = UniformCrossover_injection(num_offspring=2, crossover_rate=0.7).set_input_length(4)
key = jr.PRNGKey(42)  # Single key

# _generate_noise internally splits into (4*2, 1, 2) = (8, 1, 2)
noise_bulk = self._generate_noise(key, config)  # Returns (8, 5) in one RNG call

# Reshape to (4, 2, 5)
noise_reshaped = noise_bulk.reshape((4, 2, 5))

# vmap over all pairs and offspring
nested = jax.vmap(
    lambda noise_block, g1, g2: jax.vmap(
        lambda n, g1_single, g2_single: self._recombine_one(g1_single, g2_single, n, config)
    )(noise_block, jnp.expand_dims(g1, 0), jnp.expand_dims(g2, 0))
)(noise_reshaped, p1_pop.genes, p2_pop.genes)

# Flatten with offspring-major ordering (same as Mode A)
```

**XLA fusion benefit:** Bernoulli generation (`jr.bernoulli(key, (8, 5))`) + per-element selection merged into single kernel.

---

## Mask Semantics & Convention (Critical) 🎯

**All crossover operators follow the convention:**
- `mask=False` or `False` bits → inherit from Parent 1 (p1)
- `mask=True` or `True` bits → inherit from Parent 2 (p2)

**Implementation pattern:**
```python
offspring = jnp.where(mask, p2.values, p1.values)  # True -> p2, False -> p1
```

**Why consistent semantics matter:**
- Tests verify mask correctness: all-False mask → offspring == p1, all-True mask → offspring == p2.
- Ensures reproducibility across operator implementations and modes (A vs D).
- Enables easy inversion logic for symmetric crossovers (e.g., both offspring inherit equally from both parents).

---

## Static Meta-data & JIT Compatibility 🧩

- Mark static attributes as `pytree_node=False` (e.g., `crossover_rate`, `num_offspring`). JAX gets concrete shapes at compile-time.
- Precomputing `total_rng_budget` allows deterministic slicing and static `all_keys` shapes.
- Use `@struct.dataclass` for all operator definitions (enables Flax PyTree registration).

---

## Testing Guarantees: Correctness & Equivalence ✅

- The codebase provides tests that prove:
  - **Mask semantics:** All-False → p1, all-True → p2, mixed → correct per-gene inheritance.
  - **Bitwise divergence:** Mode A vs Mode D have non-zero L2 difference (expected due to Threefry topology).
  - **Statistical properties:** Crossover rate is respected (empirical frequency ≈ parameter).
  - **Promotion safety:** Operations using `config.dtype` remain in expected dtype.
  - **Offspring-major ordering:** Flattened offspring match expected layout (injection mode parity).

---

## Performance Rationale: Hardware Fusion 💡

- Bulk injection (Mode D) unlocks XLA fusion across:
  - RNG generation (`jax.random.bernoulli` or `jax.random.uniform`)
  - Arithmetic operations (`jnp.where`, `jnp.clip`, etc.)
  - Into single HLO kernel, minimizing host overhead and maximizing device throughput on large GPUs.

---

## Developer Checklist — Adding a New Crossover Operator 🛠️

### Core Implementation
- [ ] Define `num_keys_per_atomic_operation`:
  - Example: UniformCrossover uses `1` (one key for Bernoulli mask).
  - Example: BlendCrossover uses `2` (one for crossover decision, one for uniform sampling).
- [ ] Implement `_generate_noise(keys: chex.Array, config: C) -> noise_data`:
  - Extract individual keys: `k1, k2, ... = keys[0], keys[1], ...`
  - Generate noise with **explicit dtype** where applicable.
  - Return array shaped to `config.shape`.
- [ ] Implement `_recombine_one(p1: G, p2: G, noise_data: Any, config: C) -> G` (returns **single** offspring, not tuple):
  - Pure arithmetic only (no Python `if`/`else` in hot path).
  - Follow **mask semantics:** `mask=False -> p1`, `mask=True -> p2`.
  - Use `jnp.where(mask, p2.values, p1.values)` for masked selection.
  - **Vaccinate** constants: `jnp.array(constant, dtype=config.dtype)`.
  - Return `p1.replace(values=new_values)` (immutable copy).

### Mode Support (Automatic via Inheritance)
- [ ] Mode A (BaseCrossover) support is **automatic** — inherit from `BaseCrossover` and implement Tier 1 & 2.
- [ ] Mode D (BaseCrossover_injection) support is **automatic** — also create a `YourCrossover_injection` variant:
  - Inherit from `BaseCrossover_injection`.
  - Use same `_generate_noise` and `_recombine_one` implementations.
  - Handle bulk noise shape in `_generate_noise`: expect to generate `(num_pairs * num_offspring, ...config.shape)`.

### Testing
- [ ] Add tests in [tests/operators/crossover/test_crossover_inner_methods.py](../../tests/operators/crossover/test_crossover_inner_methods.py):
  - **Tier 2 shape/dtype:** Verify `_generate_noise` returns correct shape and dtype.
  - **Tier 1 correctness:** 
    - All-False mask → offspring == p1.
    - All-True mask → offspring == p2.
    - Mixed mask → correct per-gene inheritance.
  - **Crossover rate:** Generate 100+ samples, verify empirical rate ≈ parameter (within ±10%).
  - **Promotion safety:** Ensure no implicit upcasting with `config.dtype=jnp.bfloat16`.
  - **Mode parity (optional):** If implementing Mode D, verify Mode A and Mode D produce statistically similar results.
- [ ] Add regression tests for:
  - Default `num_offspring` value.
  - Mask semantics consistency across fused and injection variants.
  - Offspring-major flattening ordering.
- [ ] Run with `pytest tests/operators/crossover/ -v` to validate.

---

## Available Crossover Operators 📚

### Binary Crossovers (for BinaryGenome)
- **UniformCrossover** — Independently select each bit from Parent 1 or Parent 2 with probability `crossover_rate`. High disruption, excellent exploration.
- **SinglePointCrossover** — Select a random crossover point and swap segments. Preserves building blocks, more conservative.

### Real-Valued Crossovers (for RealGenome)
- **UniformCrossover** — Independently select each gene from Parent 1 or Parent 2 with probability `crossover_rate`. For independent genes.
- **BlendCrossover (BLX-α)** — Sample uniformly from extended interval around parents. Supports both exploration (high α) and exploitation (low α).
- **SimulatedBinaryCrossover (SBX)** — Polynomial distribution-based crossover. Self-adaptive spread, parent-centric. Parameter `eta` controls distribution width.
- **BinomialCrossover** — Differential evolution style. Selects genes from mutant or target vector. Used in DE algorithms.
- **EvosaxUniformCrossoverWrapper** — Wrapper around Evosax's native uniform crossover for benchmarking.

### Injection-Mode Variants
- **UniformCrossover_injection** — Mode D version of UniformCrossover (both binary and real).
- **BlendCrossover_injection** — Mode D version of BlendCrossover.
- Custom operators can support Mode D by creating a `YourCrossover_injection` subclass.

---

## Operator Selection Guide 📊

| Problem Type | Recommended Operator | Parameters | Characteristics |
|--------------|---------------------|------------|------------------|
| **Binary/Combinatorial** | Uniform Crossover | rate=0.5 | High disruption, excellent exploration |
| **Binary/Building Blocks** | Single-Point | default | Low disruption, preserves adjacency |
| **Real/Independent Genes** | Real Uniform | rate=0.6 | Gene independence, simple mixing |
| **Real/Exploration** | Blend (BLX-α) | α=0.5 | Explores outside parental range |
| **Real/Exploitation** | SBX | η=20-30 | Parent-centric, adaptive spread |
| **Differential Evolution** | Binomial | rate=0.5 | Directional, mutant-biased |

---

## Architecture References 🔗

- [base.py](../base.py) — `BaseCrossover` (Tier 3, Mode A per-pair)
- [base_injection.py](../base_injection.py) — `BaseCrossover_injection` (Tier 3, Mode D bulk)
- [binary.py](./binary.py) — Binary crossover implementations
- [real.py](./real.py) — Real-valued crossover implementations
- [evosax_crossover.py](./evosax_crossover.py) — Evosax wrapper (supports both modes)
- [tests/operators/crossover/test_regress_crossover.py](../../tests/operators/crossover/test_regress_crossover.py) — Regression tests (mask semantics, offspring-major order)
- [tests/operators/crossover/test_crossover_inner_methods.py](../../tests/operators/crossover/test_crossover_inner_methods.py) — Tier 1/2 correctness tests

---

## Quick Recap: 3-Tier Paradigm 🎯

1. **Tier 1 (Pure Arithmetic):** `_recombine_one(p1, p2, noise_data, config) -> offspring`
   - Deterministic selection: `jnp.where(mask, p2, p1)`
   - No randomness, no Python branching

2. **Tier 2 (Noise/Masks):** `_generate_noise(keys, config) -> noise_data`
   - Produces Bernoulli masks, uniform samples, or crossover points
   - Shapes match `config.shape`, dtypes explicit

3. **Tier 3 (Vectorization):** `BaseCrossover.__call__(all_keys, p1_pop, p2_pop, config) -> offspring_pop`
   - Nested vmaps over pairs and offspring
   - Offspring-major flattening for consistency
   - Automatic support for Mode A and Mode D via inheritance

Together, these tiers enable **JIT compilation**, **deterministic reproducibility**, **static resource budgeting**, and **maximal XLA kernel fusion**.
