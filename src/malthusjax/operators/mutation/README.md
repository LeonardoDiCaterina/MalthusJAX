# Mutation Operators — Architecture & Implementation (Operators / Mutation) 🔧

This document describes the three-tier mutation architecture in MalthusJAX, the PRNG ablation experiment (Mode A vs Mode D), and implementation best practices for developing high-performance, JAX-native mutation operators.

---

## Overview

MalthusJAX mutation operators are engineered for JAX/XLA and modern accelerators (H100/A100). The design separates concerns into three tiers to enable provable correctness, static resource budgeting, and maximal HLO fusion.

- **Tier 1 — Arithmetic Kernel** (pure, promotion-free)
- **Tier 2 — Noise Generation** (RNG, consumes pre-allocated key budget)
- **Tier 3 — Vectorized Wrapper** (`BaseMutation.__call__`, ResourceMapper integration)

This separation allows swapping RNG topologies (per-individual sampling vs bulk injection) without changing the arithmetic kernel.

---

## Tier 1 — Arithmetic Kernel (Pure & Promotion-Free) ⚖️

- Contract: `_mutate_one(genome: G, noise_data: Any, config: C, **kwargs) -> G`
- Responsibilities:
  - Implement the exact arithmetic for one individual using masked operations (e.g., `genome.values + (noise * mask)`).
  - Avoid Python branching (no `if`/`else` inside math) to maximize XLA fusion.
  - **Vaccinate** constants and intermediate scalars with `jnp.array(..., dtype=config.dtype)` to prevent BF16 -> FP32 implicit promotion.
  - Return a new, immutable genome dataclass (e.g., `genome.replace(values=new_values)`).

**Example (GaussianMutation):**
```python
def _mutate_one(self, genome: RealGenome, noise_data: chex.Array, config: RealGenomeConfig, **kwargs: Any) -> RealGenome:
    # noise_data = raw_noise * mutation_strength * bernoulli_mask (from Tier 2)
    mutated_values = genome.values + noise_data
    if self.clip:
        min_val, max_val = config.bounds
        mutated_values = jnp.clip(mutated_values, min_val, max_val)
    return cast(RealGenome, cast(Any, genome).replace(values=mutated_values))
```

Why:
- Keeping this tier pure and stateless makes it trivially `vmap`-able and JIT-friendly. It also helps enable bulk-injection (Mode D) where precomputed noise is injected into this kernel.

---

## Tier 2 — Noise Generation (Entropy Producers) 🎲

- Contract: `_generate_noise(keys: chex.Array, config: C, ...) -> noise_data`
- Responsibilities:
  - Consume an exact slice of PRNG keys (the block length is fixed by `num_keys_per_atomic_operation`).
  - Produce masks, Gaussian/Uniform noise, or any domain-specific stochastic values shaped to `config.shape`.
  - **Ensure dtype correctness** for all generated arrays (use `dtype=config.dtype` to prevent implicit promotion).

Notes:
- This tier is **agnostic** to how keys are sourced — whether per-individual (`Mode A`) or via bulk injection (`Mode D`).
- Both `BaseMutation` and `BaseMutation_injection` subclasses use the same `_generate_noise` implementation.

**Example (GaussianMutation):**
```python
def _generate_noise(self, keys: chex.Array, config: RealGenomeConfig) -> chex.Array:
    k_mask, k_noise = keys[0], keys[1]
    dtype = config.dtype
    
    # Tier 2.1: Bernoulli mask (which components to mutate)
    mask_bool = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=config.shape)
    mask_val = mask_bool.astype(dtype)  # Vaccinate: preserve dtype
    
    # Tier 2.2: Gaussian noise scaled by strength
    raw_noise = jax.random.normal(k_noise, shape=config.shape, dtype=dtype)
    strength = jnp.array(self.mutation_strength, dtype=dtype)  # Vaccinate
    
    # Combine: noise_data = (noise * strength * mask)
    return raw_noise * strength * mask_val
```

---

## Tier 3 — Vectorized Wrapper (Resource-aware Lifting) 🚀

- `BaseMutation.__call__(all_keys, population, config, **kwargs)` is the vectorized lifting layer (Mode A — per-individual sampling).
- `BaseMutation_injection.__call__(single_key, population, config, **kwargs)` is the bulk injection variant (Mode D — counter-advancing sampling).

### Mode A: Per-Individual Keys (BaseMutation)
- Responsibilities:
  - Reshape `all_keys` into `(pop_size, num_offspring, num_keys_per_atomic_operation, 2)` for per-individual PRNG blocks.
  - Use **nested vmap**: outer vmap over parent pairs, inner vmap over offspring per pair.
  - Each offspring uses its own deterministic key block, enabling reproducibility and independent randomness.
  - Fuses RNG + arithmetic in `_mutate_fused(keys, genome, config)` for XLA kernel fusion.
  - Return `population.spawn_offspring(new_genes)` with reset fitness (`NaN`).

### Mode D: Bulk Injection (BaseMutation_injection)
- Responsibilities:
  - Accept a single `key` and use it to generate **one bulk noise tensor** of shape `(pop_size * num_offspring, ...)`.
  - Internally splits the key into `pop_size * num_offspring` subkeys and samples all noise in one RNG call.
  - Pass bulk noise to nested vmap of `_mutate_one` operations.
  - Enables maximal XLA fusion: RNG generation + per-element arithmetic merged into single kernel.

### Key Reshaping Details
- Mode A: `all_keys.reshape(pop_size, num_offspring, num_keys_per_atomic_operation, 2)`
- Mode D: internally generates `(pop_size * num_offspring, num_keys_per_atomic_operation, 2)` from single key.
- Both use Threefry counters, but Mode A uses isolated subkeys (counter 0) while Mode D advances counters per batch index.

---

## Resource Mapping & Static RNG Budget (ResourceMapper) 🗺️

- The `ResourceMap` (computed by `compute_resource_map(...)`) pre-calculates the total RNG budget for the full pipeline:
  - Calls `operator.num_keys(input_shape)` for selection, crossover, and mutation.
  - Produces `total_rng_budget` and per-operator `OperatorAllocation` slices (start/end indices)
- Benefits:
  - Static memory allocation for key buffers, enabling predictable device placement and host-device transfer minimization.
  - Ability to request a single master split and slice it deterministically for each operator.

---

## PRNG Topologies & Implementation Modes 🔁

### Mode A — Per-Individual Sampling (BaseMutation)
- **Implementation:** `BaseMutation.__call__` in [base.py](./base.py#L79)
- **Key topology:** Each of `pop_size * num_offspring` individuals receives a unique PRNG key block from pre-split `all_keys`.
- **Code flow:**
  ```python
  keys_reshaped = all_keys.reshape(pop_size, num_offspring, num_keys_per_atomic_operation, 2)
  nested = jax.vmap(
      lambda k_block, genome: jax.vmap(
          lambda k: self._mutate_fused(k, genome, config)
      )(k_block)
  )(keys_reshaped, population.genes)
  ```
- **Pros:** Conceptually simple, bitwise reproducible with same per-individual keys.
- **Cons:** Multiple small host-side splits + multiple RNG calls reduce fusion potential.

### Mode D — Bulk Injection (BaseMutation_injection)
- **Implementation:** `BaseMutation_injection.__call__` in [base_injection.py](../base_injection.py#L110)
- **Key topology:** Single `key` is used to generate one bulk noise tensor `(pop_size * num_offspring, ...config.shape)` in a single `jr.normal` or `jr.split` call.
- **Code flow:**
  ```python
  noise = self._generate_noise(key, config)  # Returns (N, ...shape)
  mutated = jax.vmap(
      lambda n, g: self._mutate_one(g, n, config)
  )(noise, population.genes)
  ```
- **Pros:** Single RNG call enables XLA to fuse RNG generation + arithmetic into one HLO kernel (significant throughput gains on H100/A100).
- **Cons:** Bitwise outputs differ from Mode A (counter topology). Tests validate statistical equivalence, not bitwise identity.

### When to use which mode
- **Mode A (BaseMutation):** Standard GA use cases where reproducibility with specific keys matters more than peak throughput.
- **Mode D (BaseMutation_injection):** Large-scale optimization on modern accelerators where kernel fusion and throughput dominate.

### Statistical Equivalence vs Bitwise Divergence
- Both modes produce noise from the same distribution (e.g., Normal(0, σ²)).
- **Exact sequences differ** because Mode A uses `Threefry(K_i, 0)` for each individual, while Mode D uses `Threefry(K, i)` counter advancement.
- Tests in [tests/operators/mutation/](../../tests/operators/mutation/) verify: mean/std match, L2 divergence > 0 (confirming different sequences), and no dtype promotion.

---

## Complete Example: GaussianMutation (Tier Walkthrough) 🌊

Assume: population of 4 genomes (shape 10,), 2 offspring per parent, mutation_rate=0.5, strength=0.3, clip=True.

### Mode A (BaseMutation) Flow

**Step 1: Tier 3 — Resource Allocation & Reshaping**
```python
engine.population = RealPopulation(..., size=4)
op = GaussianMutation(num_offspring=2, mutation_rate=0.5, mutation_strength=0.3).set_input_length(4)

# ResourceMapper computes: total_keys = 4 * 2 * 2 = 16 (pop * offspring * keys_per_op)
all_keys = jr.split(master_key, 16).reshape((4, 2, 2, 2))
# Shape: (pop_size=4, num_offspring=2, num_keys_per_atomic_operation=2, key_dim=2)
```

**Step 2: Tier 3 — Nested vmap orchestration**
```python
keys_reshaped = all_keys.reshape((4, 2, 2, 2))
nested_offspring = jax.vmap(
    lambda k_block, genome: jax.vmap(
        lambda k: self._mutate_fused(k, genome, config),  # Fused RNG + arithmetic
        in_axes=(0,)  # vmap over offspring keys
    )(k_block),
    in_axes=(0, 0)  # vmap over parents
)(keys_reshaped, population.genes)
# Output shape: (4, 2, 10) — 4 parents × 2 offspring × 10 genes
```

**Step 3: Tier 2 — _mutate_fused (for each offspring)**
```python
def _mutate_fused(self, keys, genome, config):
    noise = self._generate_noise(keys, config)  # Tier 2
    return self._mutate_one(genome, noise, config)  # Tier 1

# keys = (2, 2) — [k_mask, k_noise] unpacked
```

**Step 3a: Tier 2 — _generate_noise**
```python
k_mask, k_noise = keys[0], keys[1]
mask = jnp.bernoulli(k_mask, 0.5, shape=(10,)).astype(jnp.float32)  # [0 or 1]
noise = jnp.normal(k_noise, shape=(10,), dtype=jnp.float32) * 0.3 * mask
# Returns shape (10,), dtype float32
```

**Step 3b: Tier 1 — _mutate_one**
```python
mutated = genome.values + noise  # (10,) + (10,) = (10,)
if self.clip:
    mutated = jnp.clip(mutated, -5.0, 5.0)  # Bounds from config
return genome.replace(values=mutated)
```

**Step 4: Finalization**
```python
new_genes = RealGenome(values=nested_offspring.reshape((8, 10)))  # Flatten to (8, 10)
return population.spawn_offspring(new_genes)  # Returns RealPopulation with 8 genomes, fitness=NaN
```

### Mode D (BaseMutation_injection) Flow

**Same setup, but single key:**
```python
op = GaussianMutation_injection(num_offspring=2, mutation_rate=0.5, mutation_strength=0.3).set_input_length(4)
key = jr.PRNGKey(42)  # Single key

# _generate_noise internally splits into (4*2, 2, 2) = (8, 2, 2)
noise_bulk = self._generate_noise(key, config)  # Returns (8, 10) in one RNG call

# vmap over all 8 offspring
mutated = jax.vmap(
    lambda n, g: self._mutate_one(g, n, config)
)(noise_bulk, population.genes.reshape((8, 10)))
```

**XLA fusion benefit:** RNG generation (`jr.normal(key, (8, 10))`) + per-element arithmetic (`+ noise * mask`) merged into single kernel.

---

## Static Meta-data & JIT Compatibility 🧩

- Mark static config attributes as `pytree_node=False` (e.g., `RealGenomeConfig.length`). This gives JAX concrete shapes at compile-time and avoids dynamic-shape tracing.
- Precomputing `total_rng_budget` allows deterministic slicing and static `all_keys` shapes.

---

## Testing guarantees: Sequential Equivalence & Statistical Integrity ✅

- The codebase provides tests that prove:
  - **Bitwise divergence**: Mode A vs Mode D will have non-zero L2 difference (expected due to Threefry topology). Tests assert divergence > 0.
  - **Statistical equivalence**: Mean and standard deviation of injected noise match expectations within tolerances.
  - **Promotion safety**: Operations using `config.dtype` remain in expected dtype (e.g., BF16) and do not upcast unexpectedly.

---

## Performance rationale: Hardware Fusion 💡

- Bulk injection (Mode D) unlocks XLA fusion across RNG generation and elementary arithmetic (mask * noise + value), minimizing host overhead and maximizing device throughput on large GPUs.

---

## Developer Checklist — Adding a New Mutation Operator 🛠️

### Core Implementation
- [ ] Define `num_keys_per_atomic_operation` (counts atomic PRNG keys needed per genome).
  - Example: GaussianMutation uses `2` (one for Bernoulli mask, one for Gaussian noise).
- [ ] Implement `_generate_noise(keys: chex.Array, config: C) -> noise_data`:
  - Extract individual keys: `k1, k2, ... = keys[0], keys[1], ...`
  - Generate noise with **explicit dtype:** `jnp.random.normal(..., dtype=config.dtype)`
  - Return numpy-like array (will be vmapped by Tier 3).
- [ ] Implement `_mutate_one(genome: G, noise_data: Any, config: C) -> G`:
  - Pure arithmetic only (no Python `if`/`else` in hot path).
  - Use masked operations: `value * mask + noise`.
  - **Vaccinate** constants: `jnp.array(constant, dtype=config.dtype)`.
  - Return `genome.replace(values=new_values)` (immutable copy).

### Mode Support (Automatic via Inheritance)
- [ ] Mode A (BaseMutation) support is **automatic** — inherit from `BaseMutation` and implement Tier 1 & 2.
- [ ] Mode D (BaseMutation_injection) support is **automatic** — also create a `YourMutation_injection` variant:
  - Inherit from `BaseMutation_injection`.
  - Use same `_generate_noise` and `_mutate_one` implementations.
  - Handle bulk noise shape in `_generate_noise`: expect to generate `(input_length * num_offspring, ...config.shape)`.

### Testing
- [ ] Add tests in [tests/operators/mutation/test_mutation_inner_methods.py](../../tests/operators/mutation/test_mutation_inner_methods.py):
  - **Tier 2 shape/dtype:** Verify `_generate_noise` returns correct shape and dtype.
  - **Tier 1 correctness:** Verify `_mutate_one` produces expected arithmetic results (e.g., all-zero noise → no change).
  - **Statistical properties:** Generate 100+ samples and check mean/std match expected distributions.
  - **Promotion safety:** Ensure no implicit BF16 → FP32 upcasting with `config.dtype=jnp.bfloat16`.
  - **Mode equivalence:** Verify Mode A and Mode D produce statistically similar (not bitwise identical) results.
- [ ] Run with `pytest tests/operators/mutation/ -v` to validate.
- [ ] Support both per-individual (Mode A) and bulk injection (Mode D) invocation paths at Tier 3.
- [ ] Add tests: statistical equivalence (mean/std), bitwise divergence (L2 > 0), and promotion-safety (no undesired upcasts).

---

## Available Mutation Operators 📚

### Binary Mutations (for BinaryGenome)
- **BitFlipMutation** — Independently flip each bit with probability `mutation_rate`. Simple and fast.
- **ScrambleMutation** — Randomly permute bit positions with probability `mutation_rate`. Good for order-dependent problems.
- **SwapMutation** — Randomly swap two positions with probability `mutation_rate`. Local, neighborhood-preserving search.

### Real-Valued Mutations (for RealGenome)
- **GaussianMutation** — Add Gaussian noise to components selected by Bernoulli mask. Standard for continuous optimization.
- **BallMutation** — Sample uniformly within an n-dimensional ball of fixed radius. Rotationally invariant, maintains volume-uniform density.
- **PolynomialMutation** — Use polynomial distribution (parameterized by `eta`) for adaptive spread. Popular in evolutionary strategies.
- **EvosaxGaussianWrapper** — Wrapper around Evosax's native Gaussian mutation for benchmarking/ablation studies.

### Injection-Mode Variants
- **GaussianMutation_injection** — Mode D version of GaussianMutation; generates bulk noise from single key.
- **BallMutation_injection** — Mode D version of BallMutation.

## Architecture References 🔗

- [base.py](./base.py) — `BaseMutation` (Tier 3, Mode A per-individual)
- [base_injection.py](../base_injection.py) — `BaseMutation_injection` (Tier 3, Mode D bulk)
- [binary.py](./binary.py) — Binary mutation implementations
- [real.py](./real.py) — Real-valued mutation implementations
- [evosax_mutation.py](./evosax_mutation.py) — Evosax wrapper (supports both modes)
- [tests/operators/mutation/test_mutation_inner_methods.py](../../tests/operators/mutation/test_mutation_inner_methods.py) — Tier 1/2 test suite