# Mutation Integration Guide

**Comprehensive documentation for selecting, using, and extending MalthusJAX mutation operators.**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Operator Deep Dives](#operator-deep-dives)
4. [Mode Selection](#mode-selection)
5. [Evosax Integration](#evosax-integration)
6. [Advanced Patterns](#advanced-patterns)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Overview

**Mutation operators** introduce variability into evolutionary populations through stochastic perturbation. MalthusJAX provides 10+ operators across:

- **Binary mutations**: BitFlip, Scramble, Swap
- **Real-valued mutations**: Gaussian, Ball, Polynomial (each with standard + injection variants)
- **Integration layer**: EvosaxGaussianWrapper for Evosax compatibility

All follow the **3-tier architecture** (Tier 1: arithmetic, Tier 2: RNG, Tier 3: vectorization) for XLA compilation and JAX optimization.

### Design Principles

1. **Determinism**: Fixed key budgets enable reproducibility
2. **Vectorization**: Nested vmap over populations and offspring
3. **Type Safety**: Explicit dtype handling prevents JAX promotion bugs
4. **Modularity**: Operators compose with selection, crossover, and evaluation
5. **Acceleration**: XLA kernel fusion for single-kernel execution

---

## Quick Start

### Installation

```bash
pip install malthus-jax>=0.1.6
```

### Minimal Example: Gaussian Mutation

```python
import jax
import jax.numpy as jnp
from malthusjax.operators.mutation import GaussianMutation
from malthusjax.core.genome import RealGenome, RealGenomeConfig

# Create operator
mutation = GaussianMutation(
    mutation_rate=0.1,     # 10% of genes mutated per event
    mutation_strength=0.1  # Gaussian std dev
)

# Setup
key = jax.random.PRNGKey(0)
config = RealGenomeConfig(shape=(10,), dtype=jnp.float32)
population = RealGenome(values=jax.random.normal(key, (50, 10)))

# Apply mutation
mutation_keys = jax.random.split(key, 50)
mutated = mutation(mutation_keys, population, config)

print(f"Original shape: {population.values.shape}")
print(f"Mutated shape: {mutated.values.shape}")
```

### Minimal Example: Binary Mutation

```python
from malthusjax.operators.mutation import BitFlipMutation
from malthusjax.core.genome import BinaryGenome, BinaryGenomeConfig

# Create operator
bit_flip = BitFlipMutation(mutation_rate=0.05)

# Setup
config = BinaryGenomeConfig(shape=(100,), dtype=jnp.uint8)
population = BinaryGenome(values=jax.random.bernoulli(key, 0.5, (50, 100)).astype(jnp.uint8))

# Apply mutation
mutated = bit_flip(mutation_keys, population, config)
```

---

## Operator Deep Dives

### Real-Valued Mutations

#### GaussianMutation — The Standard Choice

**Best For**: General-purpose optimization, any dimension, all problem types.

**Theory**: Independent Gaussian perturbation with per-gene selection:
$$x'_i = \begin{cases} x_i + \mathcal{N}(0, \sigma^2) & \text{with prob } p_m \\ x_i & \text{otherwise} \end{cases}$$

**Key Parameters**:

```python
GaussianMutation(
    mutation_rate=0.1,      # p_m: typically 1/d (where d = dimension)
    mutation_strength=0.1,  # σ: scale relative to genome bounds
    clip=False,             # Clip to bounds after mutation?
    schedule_type=None      # Optional: decay strength over generations
)
```

**Parameter Tuning**:
- `mutation_rate`: 
  - Small genomes (d<20): 0.2-0.5
  - Medium (d=20-100): 0.05-0.15
  - Large (d>100): 0.01-0.05
- `mutation_strength`: 
  - Exploration phase: 0.1-0.2
  - Balanced: 0.05-0.1
  - Fine-tuning: 0.01-0.05

**Example with Scheduling**:
```python
from malthusjax.engine.schedules import ScheduleType

mutation = GaussianMutation(
    mutation_rate=0.1,
    mutation_strength=0.2,        # Start strong
    schedule_type=ScheduleType.LINEAR_DECAY,
    final_strength=0.01,          # Decay to fine-tuning
    max_generations=1000
)

for gen in range(1000):
    mutated = mutation(keys, pop, config, generation=gen)
    # Strength automatically: 0.2 → 0.01
```

**Strengths** ✅:
- Simple, effective, well-understood
- Works across dimensions
- Excellent empirical performance
- Natural interaction with selection pressure

**Weaknesses** ❌:
- Can concentrate mutations near Gaussian mean (symmetric around parent)
- For high-dimensional spaces, may miss local structure

---

#### BallMutation — For High Dimensions

**Best For**: High-dimensional problems (d > 100), constrained optimization.

**Theory**: Uniform perturbation within $L_2$ ball of radius $r$:
$$\|\Delta x\|_2 \leq r, \quad \Delta x \text{ uniformly distributed}$$

**Implementation**: Muller's method ensures uniform volume distribution:

$$\delta = u^{1/d} \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|} \cdot r$$

where $\mathbf{g} \sim \mathcal{N}(0, I)$ (Gaussian direction) and $u \sim \text{Uniform}(0,1)$ (power-law magnitude).

**Key Parameters**:

```python
BallMutation(
    radius=0.1,             # Ball radius
    mutation_rate=1.0,      # Apply to all dimensions by default
    clip=False,
    schedule_type=None      # Optional: decay radius
)
```

**Parameter Tuning**:
- `radius` should be ~5-10% of search space width to avoid excessive disruption
- For constrained problems, set radius based on feasible region size
- `mutation_rate` typically 1.0 (apply perturbation to whole ball)

**Why Muller's Method**:
- **Naive approach** (just Gaussian direction + uniform scaling):
  - Mutations concentrated near sphere surface (curse of dimensionality)
  - Doesn't explore interior uniformly
- **Muller's method** (power-law $u^{1/d}$):
  - Uniform distribution throughout ball volume
  - Scales correctly with dimension
  - Critical for d > 50

**Example**:
```python
ball = BallMutation(
    radius=0.15,
    mutation_rate=1.0
)

# For dimension d=200, this ensures:
# - Uniform perturbations within ball of r=0.15
# - No concentration at sphere surface
# - Better exploration in high dimensions
```

**Strengths** ✅:
- Mathematically optimal for uniform distribution
- Scales correctly to high dimensions
- No directional bias (unlike Gaussian)

**Weaknesses** ❌:
- 3 keys (higher RNG cost)
- Slower XLA compilation
- May over-explore for low dimensions (use Gaussian instead)

---

#### PolynomialMutation — For Fine-Tuning & MOO

**Best For**: Multi-objective optimization (NSGA-II), later generations (fine-tuning).

**Theory**: Polynomial distribution parameterized by shape `eta`:

$$\delta_q = \begin{cases} 
(2u)^{1/(\eta+1)} - 1 & \text{if } u \leq 0.5 \\
1 - (2(1-u))^{1/(\eta+1)} & \text{if } u > 0.5
\end{cases}$$

where $u \sim \text{Uniform}(0,1)$ and $\delta_q$ scaled by bound range.

**Key Parameters**:

```python
PolynomialMutation(
    mutation_rate=0.1,  # Per-gene probability
    eta=20.0,           # Shape parameter: smaller = heavier tails
    clip=False          # Clip to bounds?
)
```

**Parameter Tuning**:
- `eta`: 
  - `eta=5-10`: Heavy tails, larger perturbations (exploration)
  - `eta=20-30`: Standard (balanced, recommended)
  - `eta=50+`: Light tails, smaller perturbations (fine-tuning)
- Symmetric around parent by construction

**Why Two Branches**:
```python
# Two symmetric branches ensure:
# - If u < 0.5: pull toward lower bound slightly
# - If u > 0.5: pull toward upper bound slightly  
# - Symmetric distribution → no bias toward bounds
```

**Example: NSGA-II Hereditary Algorithm**:
```python
# NSGA-II standard settings
mutation = PolynomialMutation(
    mutation_rate=1.0 / genome_dimension,  # Standard probability
    eta=20.0                                 # Standard shape
)

# This is the industry-standard MOO mutation
```

**Strengths** ✅:
- Standard in multi-objective algorithms (NSGA-II, MOEA/D)
- Symmetric distribution avoids bound-seeking bias
- Parent-centric (good for fine-tuning)
- Only 2 keys

**Weaknesses** ❌:
- More complex (students sometimes struggle with eta tuning)
- Less intuitive than Gaussian
- Heavier computational cost (power operations)

---

### Binary Mutations

#### BitFlipMutation — Universal Binary Choice

**Best For**: All binary optimization problems, universal default.

**Theory**: Independent bit flip with probability `mutation_rate`:
$$x'_i = \begin{cases} \neg x_i & \text{with prob } p_m \\ x_i & \text{otherwise} \end{cases}$$

**Implementation**: XOR with Bernoulli mask (handles bool and numeric dtypes).

**Key Parameters**:
```python
BitFlipMutation(mutation_rate=0.1)  # Typically 1/genome_bits
```

**Example**:
```python
bit_flip = BitFlipMutation(mutation_rate=1.0/100)  # 1% per bit

# For 100-bit genome: ~1 bit flipped per generation on average
```

---

#### ScrambleMutation — For Permutation Problems

**Best For**: Combinatorial/routing problems with sequence structure.

**Theory**: Random permutation with probability `mutation_rate`:
$$\text{Permute}(\text{subset of positions})$$

**Implementation**: Branchless via `jax.random.permutation` and `jax.lax.select`.

**Key Parameters**:
```python
ScrambleMutation(mutation_rate=0.1)  # Probability of scrambling
```

**Why This Helps**:
- Preserves run-length structure (clusters of same values)
- Better for tree/graph encodings
- Example: TSP edge order matters

---

#### SwapMutation — For Local Search

**Best For**: Problems where local position swaps are meaningful (e.g., scheduling).

**Theory**: Swap two random positions with probability `mutation_rate`:
$$x' = \text{Swap}(x_{i_1}, x_{i_2})$$

**Implementation**: Functional immutable swap via `.at[idx].set()` chaining.

**Key Parameters**:
```python
SwapMutation(mutation_rate=0.1)  # Probability of swapping
```

**Example**:
```python
# Job scheduling problem: swap job order
swap = SwapMutation(mutation_rate=0.2)
```

---

### Evosax Wrapper

#### EvosaxGaussianWrapper — External Integration

**Best For**: Benchmarking, ablation studies, comparing MalthusJAX vs Evosax.

**Theory**: Drop-in wrapper around `evosax.mutation(key, solution, std)`.

**Key Parameters**:
```python
EvosaxGaussianWrapper(
    mutation_strength=0.1,
    injection_mode=True  # Single-key pattern
)
```

**Example: Direct Comparison**:
```python
# MalthusJAX Gaussian
malthus_mut = GaussianMutation(mutation_rate=0.1, mutation_strength=0.1)

# Evosax wrapper (for comparison)
evosax_mut = EvosaxGaussianWrapper(mutation_strength=0.1)

# Results should be similar (not identical—order of operations differs)
```

---

## Mode Selection

### Standard Mode (Default)

**When to Use**: Multi-operator compositions, complex pipelines.

```python
gaussian = GaussianMutation(mutation_rate=0.1, mutation_strength=0.1)
# Tier 3 pre-allocates keys automatically
```

**Characteristics**:
- Keys pre-allocated: O(pop_size * num_offspring * num_keys_per_op)
- Static XLA shapes (better compilation)
- Streaming noise generation

### Injection Mode

**When to Use**: Single-operator workflows, simple evolution loops.

```python
gaussian_inj = GaussianMutation_injection(mutation_rate=0.1, mutation_strength=0.1)
# Single key provided, split internally
```

**Characteristics**:
- Keys: 1 (split internally)
- All noise materialized at once
- Simpler integration

**Trade-off Decision Tree**:

```
Complex multi-operator pipeline?
  → YES: Use Standard
  → NO: Could use either

Single-population workflow?
  → YES: Injection mode OK
  → NO: Prefer Standard

XLA optimization critical?
  → YES: Likely Standard (pre-allocated shapes)
  → NO: Either works

Memory-constrained?
  → YES: Injection (smaller key buffers)
  → NO: Either works
```

---

## Evosax Integration

### Setup

```bash
# Install evosax (stable, PyPI)
pip install evosax>=0.1.6

# Optional: GitHub version for advanced features
pip install git+https://github.com/rjbruin/evosax.git@main
```

### Direct Evosax Mutation

```python
import evosax

# Simple usage:
key = jax.random.PRNGKey(0)
solution = jnp.array([1.0, 2.0, 3.0])
mutated = evosax.mutation(key, solution, std=0.1)
```

### Via MalthusJAX Wrapper

```python
from malthusjax.operators.mutation import EvosaxGaussianWrapper

wrapper = EvosaxGaussianWrapper(mutation_strength=0.1)
# Use in EvolutionLoop or other composers
```

### Benchmarking Pattern

```python
# Setup both implementations
malthus_mut = GaussianMutation(mutation_rate=0.1, mutation_strength=0.1)
evosax_mut = EvosaxGaussianWrapper(mutation_strength=0.1)

# Create parallel evolution loops
ga_malthus = EvolutionLoop(..., mutation_fn=malthus_mut)
ga_evosax = EvolutionLoop(..., mutation_fn=evosax_mut)

# Compare results over generations
for gen in range(100):
    results_m = ga_malthus.step(gen)
    results_e = ga_evosax.step(gen)
    
    # Results won't be identical (PRNG order differs)
    # but should show similar convergence patterns
```

---

## Advanced Patterns

### Pattern 1: Multi-Operator Mutation

Use different mutations for different genome regions:

```python
import jax
import jax.numpy as jnp

gaussian = GaussianMutation(mutation_rate=0.1, mutation_strength=0.5)
ball = BallMutation(radius=0.1)

def hybrid_mutate(key, genome, config):
    """50% Gaussian, 50% Ball perturbation"""
    k1, k2 = jax.random.split(key)
    
    # Apply both
    mutated_g = gaussian(jax.random.split(k1, len(genome)), genome, config)
    mutated_b = ball(jax.random.split(k2, len(genome)), genome, config)
    
    # Randomly select one per individual
    selector = jax.random.bernoulli(key, 0.5, (len(genome),))
    
    final = RealGenome(
        values=jnp.where(selector[:, None], mutated_g.values, mutated_b.values)
    )
    return final
```

### Pattern 2: Adaptive Mutation Strength

Based on population fitness progress:

```python
def adaptive_strength(fitness_history, gen):
    """Increase strength if stalled, decrease if improving"""
    recent = fitness_history[-10:] if len(fitness_history) > 10 else fitness_history
    
    # Compute improvement trend
    if len(recent) > 1:
        improvement = recent[-1] - jnp.mean(recent[:-1])
        if improvement < 0.01:  # Stalled
            return 0.2  # Increase exploration
        else:
            return 0.05  # Exploitation
    return 0.1

# Usage in loop:
for gen in range(max_gen):
    strength = adaptive_strength(fitness_hist, gen)
    mutation.mutation_strength = strength
    mutated = mutation(...)
```

### Pattern 3: Problem-Specific Custom Mutation

Cauchy distribution example (heavy-tailed for rare large jumps):

```python
from flax import struct
from malthusjax.operators.base import BaseMutation

@struct.dataclass
class CauchyMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """Heavy-tailed mutation for escaping local minima."""
    
    mutation_rate: float = 0.1
    scale: float = 0.1
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2  # Mask + Cauchy
    
    def _generate_noise(self, keys, config, generation=0):
        k_mask, k_cauchy = keys[0], keys[1]
        dtype = config.dtype
        
        mask = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=config.shape).astype(dtype)
        
        # Cauchy distribution: heavy tails
        u = jax.random.uniform(k_cauchy, shape=config.shape)
        cauchy = jnp.tan(jnp.pi * (u - 0.5))
        
        # Normalize to avoid overflow
        cauchy_safe = jnp.clip(cauchy, -100, 100) * jnp.array(self.scale, dtype=dtype)
        
        return cauchy_safe * mask
    
    def _mutate_one(self, genome, noise_data, config, **kwargs):
        mutated = genome.values + noise_data  # No clipping for heavy tails
        return genome.replace(values=mutated)
```

---

## Performance Tuning

### Memory Optimization

```python
# High population (10,000+)?
# Use Injection mode to reduce key allocation

mutation = GaussianMutation_injection(...)  # ✅ 1 key total
# NOT: GaussianMutation  # ❌ 10,000*num_offspring*2 keys
```

### Compilation Speed

```python
# Slow first run?
# Simpler operators compile faster

fast_compile = BitFlipMutation(...)        # ~2s (simple)
normal_compile = GaussianMutation(...)     # ~4s (standard)
slow_compile = BallMutation(...)           # ~5s (multiple keys)

# Subsequent runs: all <100ms, XLA caches
```

### Runtime Performance

```python
# Profiling:
import time

key = jax.random.PRNGKey(0)
mutation = GaussianMutation(mutation_rate=0.1)

population = RealGenome(values=jnp.ones((1000, 100)))
config = RealGenomeConfig(shape=(100,), dtype=jnp.float32)

# Warm up
_ = mutation(jax.random.split(key, 1000), population, config)

# Time it
start = time.time()
for _ in range(100):
    _ = mutation(jax.random.split(key, 1000), population, config)
elapsed = time.time() - start

print(f"100 iterations: {elapsed:.3f}s ({elapsed/100*1000:.2f}ms per generation)")
```

---

## Troubleshooting

### "Shape mismatch" Error

```python
# ❌ Problem
config = RealGenomeConfig(shape=(10,))
genome = RealGenome(values=jnp.ones((20,)))  # Wrong shape!

# ✅ Solution
assert config.shape == genome.values.shape[1:]
```

### "Type promotion bug"

```python
# ❌ Problem: implicit float32 → float64 promotion
dtype = config.dtype  # float32
result = raw_noise + 1e-8  # 1e-8 is float64, promotes result!

# ✅ Solution: explicit casting
dtype = config.dtype
result = raw_noise + jnp.array(1e-8, dtype=dtype)
```

### "Inconsistent results"

```python
# ❌ Problem: forgotten generation parameter
mutated = mutation(keys, pop, config)  # Missing generation=gen

# ✅ Solution: always pass generation for scheduled mutations
for gen in range(max_gen):
    mutated = mutation(keys, pop, config, generation=gen)
```

### "No visible mutation"

```python
# ❌ Possible problem: mutation_rate too low
if mutation_rate < 1e-4:
    print("WARNING: mutation_rate very low, may not see changes")

# ✅ Debug
noise = mutation._generate_noise(keys, config)
print(f"Noise RMS: {jnp.sqrt(jnp.mean(noise**2))}")
# Should be >> 0 for typical parameters
```

---

## References

- **MalthusJAX Mutation README**: [src/malthusjax/operators/mutation/README.md](./src/malthusjax/operators/mutation/README.md)
- **Quick Reference**: [MUTATION_QUICK_REFERENCE.md](./MUTATION_QUICK_REFERENCE.md)
- **Crossover Guide**: [CROSSOVER_INTEGRATION.md](./EVOSAX_INTEGRATION.md) (similar architecture)
- **Evosax Docs**: https://evosax.readthedocs.io
- **Tournament Selection**: [src/malthusjax/core/selection/](./src/malthusjax/core/selection/)
