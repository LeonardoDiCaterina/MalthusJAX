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
- **BitFlipMutation**: Independently flip each bit with probability `mutation_rate`. (1 key)
- **ScrambleMutation**: Randomly permute bit positions with probability `mutation_rate`. (2 keys)
- **SwapMutation**: Randomly swap two bit positions with probability `mutation_rate`. (3 keys)

**Real-Valued Mutations** (for RealGenome)
- **GaussianMutation**: Add Gaussian noise to components selected by Bernoulli mask. (2 keys, with injection variant)
- **BallMutation**: Sample uniformly within an n-dimensional ball of fixed radius using Muller's method. (3 keys, with injection variant)
- **PolynomialMutation**: Use polynomial distribution parameterized by shape parameter `eta`. (2 keys, with injection variant)
- **EvosaxGaussianWrapper**: Wrapper around Evosax native Gaussian mutation for benchmarking. (1 key, injection mode)

---

## Operator Selection Guide

Choose the mutation operator based on your problem type and genome dimension:

| Problem Type | Recommended Operator | Parameters | Use When | Notes |
|--------------|---------------------|------------|----------|-------|
| **Binary / Small** | BitFlipMutation | rate=0.1 | Default binary optimization | Simple, effective per-bit disruption |
| **Binary / Structured** | ScrambleMutation | rate=0.05 | Permutation-based problems | Preserves connectivity patterns |
| **Binary / Rare Events** | SwapMutation | rate=0.1 | Combinatorial/routing | Localized position changes |
| **Real / Small (d<20)** | GaussianMutation | rate=0.3, std=0.1 | General-purpose, exploration | Standard recommendation |
| **Real / Medium (d=20-100)** | GaussianMutation | rate=0.1, std=0.05 | Balanced exploration/exploitation | Recommended default |
| **Real / Large (d>100)** | BallMutation | radius=0.1 | High-dimensional spaces | Uniform distribution avoids bias |
| **Real / Multimodal** | PolynomialMutation | rate=0.1, eta=20 | NSGA-II compatibility, fine-tuning | Parent-centric, good for MOO |
| **Benchmarking** | EvosaxGaussianWrapper | std_dev=0.1 | Comparing MalthusJAX vs Evosax | Direct Evosax compatibility check |

### Tuning Recommendations by Problem Phase

| Phase | Mutation Rate | Gaussian Strength | Ball Radius | Notes |
|-------|---|---|---|---|
| **Exploration (Gen 0-30%)** | Higher (0.2-0.3) | Higher (0.1-0.2) | Larger (0.15-0.25) | Encourage diversity, escape local optima |
| **Balanced (Gen 30-70%)** | Moderate (0.1) | Moderate (0.1) | Moderate (0.1) | Standard settings work well |
| **Exploitation (Gen 70-100%)** | Lower (0.05-0.1) | Lower (0.01-0.05) | Smaller (0.01-0.05) | Fine-tune near solution |
| **With Scheduling** | Constant | Use decay schedule | Use decay schedule | See `schedule_type` parameter |

---

## Mutation Modes: Standard vs Injection

MalthusJAX supports two mutation paradigms for real-valued operators:

### Standard Mode (Default)

**Architecture**: Full 3-tier with pre-allocated key buffers.

```python
gaussian = GaussianMutation(
    mutation_rate=0.1,
    mutation_strength=0.1
)

# Tier 3 pre-allocates keys: shape (population_size * num_offspring, num_keys_per_atomic_op, 2)
# Tier 2 generates noise for each individual independently
# Tier 1 applies arithmetic: genome + noise
```

**Characteristics**:
- ✅ Pre-allocated key budget (predictable memory)
- ✅ Static XLA shapes (better compilation)
- ✅ Per-individual key determinism
- ⚠️ Materialized noise arrays

**Best For**: Multi-population workflows, complex compositions, XLA optimization.

### Injection Mode (Single-Key Alternative)

Available for: `GaussianMutation_injection`, `BallMutation_injection`, `PolynomialMutation_injection`

```python
gaussian_inj = GaussianMutation_injection(
    mutation_rate=0.1,
    mutation_strength=0.1
)

# Tier 3 receives single key, splits internally to (population_size * num_offspring * num_keys_per_atomic_op)
# All noise generated upfront via nested vmap
# Single-seed determinism for reproducibility
```

**Characteristics**:
- ✅ Minimal key pre-allocation (1 key total)
- ✅ Single-seed reproducibility
- ✅ Simpler integration pattern
- ⚠️ All noise materialized at once (higher memory)

**Best For**: Single-population workflows, simpler evolution loops, one-shot evaluations.

**Trade-off Matrix**:

| Criterion | Standard | Injection |
|-----------|----------|-----------|
| Key pre-allocation | `n * k * m * 2` (high) | `1` (low) |
| Memory use | Streaming | Materialized |
| XLA optimization | ✅ Best | ✅ Good |
| Reproducibility | Per-individual | Single-seed |
| Compilation time | Slower (complex shape) | Faster (simpler) |

---

## Evosax Integration

MalthusJAX provides **EvosaxGaussianWrapper** to integrate Evosax's mutation operators while leveraging MalthusJAX's ecosystem.

### Quick Integration

```python
from malthusjax.operators.mutation import EvosaxGaussianWrapper
import evosax

# Create wrapper
wrapper = EvosaxGaussianWrapper(
    mutation_strength=0.1,
    injection_mode=True  # Single-key pattern
)

# Use in evolution loop
mutated = wrapper(keys, population, config)
```

### Evosax API Compatibility

**evosax 0.1.6 (PyPI)**:
- Provides `evosax.mutation(key, solution, std) -> mutated`
- ✅ Supported via compatibility layer (`malthusjax.compat.evosax_mimic`)

**evosax GitHub main**:
- Modern ask/tell interface (not used directly by MalthusJAX)
- ⚠️ Can be installed separately for advanced workflows

### Use Cases

1. **Benchmarking**: Compare MalthusJAX-composed GA vs pure Evosax
   ```python
   # MalthusJAX approach
   ga_malthus = EvolutionLoop(mutation_fn=EvosaxGaussianWrapper(...))
   
   # Direct Evosax approach (for comparison)
   ga_evosax = evosax.SimpleGA(...)
   
   # Results should be numerically equivalent
   ```

2. **Ablation Studies**: Replace MalthusJAX mutation with Evosax baseline
   ```python
   baseline = EvolutionLoop(mutation_fn=EvosaxGaussianWrapper(...))
   custom = EvolutionLoop(mutation_fn=GaussianMutation(...))
   # Compare performance
   ```

3. **Algorithm Swapping**: Drop-in replacement for experimentation
   ```python
   # Dynamic injection
   mutator = EvosaxGaussianWrapper if use_baseline else GaussianMutation
   ```

### Architecture: Why Wrapping Succeeds

1. **Evosax mutation already pure**: No internal RNG state, just `key, solution, std -> mutant`
2. **MalthusJAX manages keys**: Wrapper provides PRNG stream through `EvosaxGaussianWrapper`
3. **Type compatibility**: MalthusJAX `RealGenome` ↔ Evosax flat arrays (transparent conversion)
4. **Injection mode natural fit**: Single key → split internally (matches Evosax single-key interface)

### Troubleshooting Evosax Integration

| Problem | Solution |
|---------|----------|
| `ImportError: evosax` | `pip install evosax>=0.1.6` |
| `evosax.mutation not found` | Use compatibility layer: `malthusjax.compat.evosax_mimic.mutation` |
| Shape mismatch | Ensure `genome.values.shape` matches Evosax expectation (flat array) |
| Different results | Check PRNG seeding; Evosax may differ slightly due to floating-point order |

---

## Performance & XLA Fusion Considerations

### Hierarchy of Performance

From fastest to slowest (all in same XLA compilation):

1. **BitFlipMutation** (1 key): Simplest, minimal RNG
2. **Polynomial/Gaussian** (2 keys): Standard choice
3. **Ball** (3 keys): More RNG, but still fast
4. **Injection variants**: Materialization cost, but simpler XLA traces
5. **EvosaxGaussianWrapper**: External function call (slight overhead)

### Memory Footprint

- **Standard mode**: O(pop_size * num_offspring * dimension)
- **Injection mode**: O(pop_size * num_offspring * dimension) (same, but clustered)
- **Binary mutations**: O(pop_size * num_offspring * genome_bits)

### Compilation Profile

| Operator | First Run | Subsequent | Notes |
|----------|-----------|-----------|-------|
| BitFlip | 2-3s | <50ms | Very simple XLA graph |
| Gaussian | 3-5s | 50-100ms | Standard cost |
| Ball | 4-6s | 100-150ms | More RNG operations |
| Injection | 2-4s | 50-100ms | Simpler vmap structure |

---

## Troubleshooting

### Issue: "Shape mismatch in _generate_noise"

**Symptom**: `ValueError: operands could not be broadcast together with shapes (10,) (5,)`

**Solution**:
```python
# Check genome configuration
print(f"Config shape: {config.shape}")
print(f"Genome shape: {genome.values.shape}")

# Ensure consistency
assert config.shape == genome.values.shape
```

### Issue: "Type promotion to float64"

**Symptom**: Unexpected `dtype=float64` in outputs despite setting `dtype=float32`

**Solution**:
```python
# Ensure explicit casting throughout
dtype = config.dtype
raw_noise = jax.random.normal(..., dtype=dtype)  # ✓ Explicit
result = raw_noise.astype(dtype)  # ✓ Safe cast

# Avoid implicit promotion
# result = raw_noise + 1e-8  # ✗ May promote to float64
result = raw_noise + jnp.array(1e-8, dtype=dtype)  # ✓ Same dtype
```

### Issue: "Inconsistent results between runs"

**Symptom**: Same seed produces different mutations

**Solution**:
```python
# Always use explicit PRNG keys
key = jax.random.PRNGKey(42)
seeds = jax.random.split(key, population_size)

# Never use implicit global state
# for _ in range(...):  # ✗ Wrong
#     mutate_direct()

# Use ResourceMapper for deterministic key allocation
op.set_input_length(population_size)
```

### Issue: "Mutation not affecting population"

**Symptom**: Offspring identical to parents

**Solution**:
```python
# Check mutation_rate isn't too low
if mutation_rate < 1e-3:
    print("Warning: mutation_rate very low, may see no changes")

# Verify num_offspring > 0
assert num_offspring >= 1

# Check noise actually generated (zero mutations possible but rare)
noise = mutation._generate_noise(keys, config)
print(f"Noise norm: {jnp.linalg.norm(noise)}")
```

---

## Advanced Usage

### Scheduling Mutation Strength (Gaussian & Ball)

Decay mutation strength over generations:

```python
from malthusjax.engine.schedules import ScheduleType

gaussian = GaussianMutation(
    mutation_rate=0.1,
    mutation_strength=0.2,  # Initial strength
    schedule_type=ScheduleType.LINEAR_DECAY,  # Decay schedule
    final_strength=0.01,  # Final strength at max_generations
    max_generations=1000
)

# In evolution loop:
for gen in range(max_generations):
    mutated = gaussian(keys, population, config, generation=gen)
    # Strength automatically decays: 0.2 → 0.01 over 1000 generations
```

### Custom Mutation Kernels (Template)

```python
from flax import struct
from malthusjax.operators.base import BaseMutation

@struct.dataclass
class MyCustomMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """Custom mutation: Cauchy distribution (heavy tails, rare large perturbations)."""
    
    mutation_rate: float = 0.1
    scale: float = 0.1
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2  # Bernoulli mask + Cauchy distribution
    
    def _generate_noise(self, keys, config, generation=0):
        k_mask, k_cauchy = keys[0], keys[1]
        dtype = config.dtype
        
        mask = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=config.shape).astype(dtype)
        standard_cauchy = jnp.tan(jnp.pi * (jax.random.uniform(k_cauchy, shape=config.shape) - 0.5))
        cauchy_noise = (standard_cauchy / 100.0) * jnp.array(self.scale, dtype=dtype)  # Normalized
        
        return cauchy_noise * mask
    
    def _mutate_one(self, genome, noise_data, config, **kwargs):
        mutated = genome.values + noise_data
        # No clipping for heavy-tailed distribution
        return genome.replace(values=mutated)
```

---

## Architecture References

- [base.py](./base.py) — `BaseMutation` (Tier 3 vmap orchestration)
- [binary.py](./binary.py) — Binary mutation implementations
- [real.py](./real.py) — Real-valued mutation implementations
- [evosax_mutation.py](./evosax_mutation.py) — Evosax wrapper
- [tests/operators/mutation/](../../tests/operators/mutation/) — Test suite
- [MUTATION_INTEGRATION.md](../../MUTATION_INTEGRATION.md) — Comprehensive integration guide
- [MUTATION_QUICK_REFERENCE.md](../../MUTATION_QUICK_REFERENCE.md) — Developer quick reference