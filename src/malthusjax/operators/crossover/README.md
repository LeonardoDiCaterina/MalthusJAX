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

**Mask Convention** (for Per-Element Selection Operators):
- `mask=False` → select from Parent 1
- `mask=True` → select from Parent 2
- Example: `offspring = jnp.where(mask, p2.values, p1.values)`
- Applies to: UniformCrossover, SinglePointCrossover

**Gate-Based Selection** (Alternative Pattern for Adaptive Operators):
- Some operators use masks/decisions as **gates**, not parent selectors
- Pattern: `offspring = jnp.where(gate, computed_value, p1.values)`
- True → use computed offspring; False → return parent 1 unchanged
- Applies to: BlendCrossover, SimulatedBinaryCrossover, BinomialCrossover
- Example: SBX applies `jnp.where(should_cross, sbx_child, p1.values)` where `should_cross` is a gating decision, not a per-gene mask

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
  - Flatten offspring from `(num_pairs, num_offspring, ...)` to `(num_pairs * num_offspring, ...)` via direct `reshape` (no transpose).
  - Return new population via `spawn_offspring()`.

**Key Reshaping**:

```python
all_keys.reshape(num_pairs, num_offspring, num_keys_per_atomic_operation, 2)
```

**Pair-Major Flattening**:

```python
def flatten_fn(x: chex.Array) -> chex.Array:
    # x shape: (num_pairs, num_offspring, ...)
    # Direct reshape — no transpose needed. Output ordering is irrelevant
    # to downstream mutation/merge/evaluation, and avoiding the transpose
    # eliminates a physical data copy that would break XLA fusion (see FB-1).
    return x.reshape((-1,) + x.shape[2:])  # -> (num_pairs * num_offspring, ...)
```

**Result**: All offspring for pair 0 come first, then all offspring for pair 1, etc. This is the **pair-major** order.

> **Design Evolution (Phase 3 Optimization)**: Earlier versions used an offspring-major
> ordering via `jnp.transpose`. This forced XLA to materialize a physical data copy,
> creating a fusion barrier between crossover and downstream mutation. Since the engine's
> merge phase treats all offspring identically (truncation via `[:num_mutants]`),
> the ordering is semantically irrelevant. Removing the transpose eliminates the copy
> and enables XLA to fuse crossover → mutation into a single kernel, improving throughput.

---

## Tier 3 Variants: Standard vs Injection Mode

**Standard Mode** (`BaseCrossover` class):
- Operator consumes pre-allocated keys from ResourceMapper
- Keys reshaped to `(num_pairs, num_offspring, num_keys_per_atomic_operation, 2)`
- Tier 2 (`_generate_noise`) called once per (pair, offspring) inside nested vmap
- Lazy generation: noise generated on-demand per kernel call
- Memory efficient: only stores keys, not all noise arrays
- Best for: Most operators; memory-constrained scenarios

**Injection Mode** (`BaseCrossover_injection` class):
- Operator receives single key from engine
- Internally splits key into `(n_pairs * n_offspring)` subkeys
- Tier 2 (`_generate_noise`) materializes **all** noise upfront
- Returns full array: shape `(n_pairs * n_offspring, ...)` or tuple thereof
- Memory cost: O(n_pairs × n_offspring × noise_shape) upfront
- Trade-off: More memory for full determinism + explicit noise control
- Available variants: `UniformCrossover_injection`, `BlendCrossover_injection`, `SimulatedBinaryCrossover_injection`
- Best for: Debugging; exact reproducibility requirements; ablation studies

**When to Choose**:
- **Standard (default)**: Production runs, large populations, memory-constrained hardware
- **Injection**: Debugging, validation, small populations, when full noise materialization is beneficial

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

### User-Facing API Usage

```python
import jax.random as jr
from malthusjax.operators.crossover import RealUniformCrossover
from malthusjax.core.genome import RealGenomeConfig, RealPopulation

# Setup: 4 parent pairs, 2 offspring per pair, crossover_rate=0.7
key = jr.PRNGKey(42)
config = RealGenomeConfig(shape=(5,), bounds=(0.0, 1.0))

# Create two parent populations
p1_pop = RealPopulation.init_random(key, config, size=4)
p2_pop = RealPopulation.init_random(jr.fold_in(key, 1), config, size=4)

# Create and configure crossover operator
crossover = RealUniformCrossover(num_offspring=2, crossover_rate=0.7)
crossover = crossover.set_input_length(4)  # 4 parent pairs

# Allocate PRNG keys
num_keys = crossover.num_keys((4,))  # Returns: 4 * 2 * 1 = 8
all_keys = jr.split(key, num_keys)

# Perform crossover (Tier 3 handles internal vmap orchestration)
offspring_pop = crossover(all_keys, p1_pop, p2_pop, config)
# Result: offspring_pop has 8 individuals (4 pairs × 2 offspring each)
```

### Internal Implementation Details (Developers Only)

The above user call internally executes these Tier-3 (vmap) nested iterations:

**Tier 2 — Generate Bernoulli Mask** (called per pair per offspring):
```python
def _generate_noise(self, keys, config):
    return jax.random.bernoulli(keys[0], p=0.7, shape=(5,))
    # Returns: boolean array shape (5,) per individual
```

**Tier 1 — Per-Gene Selection** (pure arithmetic):
```python
def _recombine_one(self, p1, p2, noise_data, config):
    mask = noise_data  # Shape (5,), dtype bool
    offspring = jnp.where(mask, p2.values, p1.values)
    return p1.replace(values=offspring)
```

**Pair-Major Flattening** (Tier 3 final reshape):
```python
# Before flatten: shape (4, 2, 5) = (pairs, offspring_per_pair, genes)
# After direct reshape: (8, 5) = (total_offspring, genes)
# Result ordering: [pair0_offspring0, pair0_offspring1,
#                   pair1_offspring0, pair1_offspring1,
#                   pair2_offspring0, pair2_offspring1,
#                   pair3_offspring0, pair3_offspring1]
```

---

## Mask Semantics & Operator Patterns

### Per-Element Selection (UniformCrossover, SinglePointCrossover)

These operators use **per-gene mask selection**:
- `mask=False` → inherit from Parent 1
- `mask=True` → inherit from Parent 2
- **Implementation pattern**:
  ```python
  offspring = jnp.where(mask, p2.values, p1.values)  # True -> p2, False -> p1
  ```
- **Tests verify**:
  - All-False mask → offspring == p1
  - All-True mask → offspring == p2
  - Mixed mask → correct per-gene inheritance

### Gate-Based Selection (BlendCrossover, SBX, BinomialCrossover)

These operators use **gating decisions**, not per-gene masks:
- `decision=False` → return Parent 1 unchanged
- `decision=True` → return computed/blended offspring
- **Implementation pattern**:
  ```python
  offspring = jnp.where(decision, computed_values, p1.values)  # True -> computed, False -> p1
  ```
- **Key difference**: The mask is a 0-D boolean decision, not a per-gene selection vector
- **Example (SBX)**: `jnp.where(should_cross, sbx_child, p1.values)` where `should_cross` is a scalar boolean
- **Example (Blend)**: `jnp.where(should_cross, blended_values, p1.values)` where `should_cross` gates the blend operation
- **Tests verify**:
  - False gate → offspring == p1
  - True gate → offspring == computed_values (not p2)

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

> **Note**: As of Phase 3, crossover uses **pair-major** ordering (direct
> `reshape`). Mutation already used direct `reshape` and was not affected.

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
- [ ] Define `num_keys_per_atomic_operation` (e.g., 1 for Bernoulli, 2 for blend + decision, 3 for SBX).
- [ ] Implement `_generate_noise(keys, config)`:
  - Extract individual keys: `k1, k2 = keys[0], keys[1]`.
  - Return array(s) shaped to `config.shape` with explicit dtype.
  - For multi-element noise, return tuple: `(decision, values)` or `(decision, values, swap_mask)`
- [ ] Implement `_recombine_one(p1, p2, noise_data, config)`:
  - Pure arithmetic only (no Python branching in hot path).
  - **Return single genome** (not tuple) — Tier 3 calls this `num_offspring` times
  - Decide mask semantics: per-element selection OR gate-based (see below)

**Mask Semantics (Choose One Pattern)**
- [ ] **Per-Element Selection** (UniformCrossover, SinglePointCrossover pattern):
  - Use `jnp.where(mask, p2.values, p1.values)` for per-gene parent selection
  - Test: all-False mask → offspring==p1, all-True mask → offspring==p2
- [ ] **Gate-Based Selection** (BlendCrossover, SBX, BinomialCrossover pattern):
  - Use `jnp.where(gate, computed_values, p1.values)` where gate is 0-D boolean
  - Test: gate=False → offspring==p1, gate=True → offspring==computed_values
  - Document that this differs from per-element semantics

**PRNG Handling**
- [ ] Handle `typed_keys` parameter for PRNG format (engine sets based on backend):
  - For key extraction inside `_generate_noise`: keys are already extracted by Tier 3
  - For custom `_cross_fused` overrides (like EvosaxUniformCrossoverWrapper):
    ```python
    if self.typed_keys:
        prng_key = keys.reshape(-1)[0]  # New-style: scalar
    else:
        prng_key = keys.reshape((-1, keys.shape[-1]))[0]  # Legacy: (2,) pair
    ```

**Offspring Semantics**
- [ ] Understand `num_offspring` in Tier 3 context:
  - Each `_recombine_one` call produces **one** offspring per parent pair
  - Tier 3 calls `_recombine_one` `num_offspring` times (controlled by outer vmap)
  - Keys are pre-allocated: `(num_pairs, num_offspring, num_keys_per_atomic, 2)`
  - Each vmap iteration gets unique key block → reproducible but distinct offspring

**Inheritance & Modes**
- [ ] Inherit from `BaseCrossover` for standard mode (pre-allocated keys)
- [ ] Optional: Inherit from `BaseCrossover_injection` for single-key injection mode
  - Override `_generate_noise` to materialize all `(n_pairs * n_offspring)` noise upfront
  - Trade-off: More memory for full determinism + replay capability
- [ ] Consider custom `_cross_fused` override for special cases (e.g., EvosaxUniformCrossoverWrapper)

**Testing**
- [ ] Add tests in `tests/operators/crossover/`:
  - Test shape/dtype of `_generate_noise()`
  - Test boundary conditions (all-False, all-True masks or gates)
  - Verify crossover rate/parameters empirically
  - Verify pair-major offspring ordering
  - If supporting both standard + injection: test both code paths
  - Test with both legacy (uint32[2]) and new-style (typed) PRNG keys

---

## Available Crossover Operators

**Binary Crossovers** (for BinaryGenome)
- **UniformCrossover**: Independently select each bit from Parent 1 or Parent 2 with probability `crossover_rate`.
- **SinglePointCrossover**: Select random crossover point and swap segments.

**Real-Valued Crossovers** (for RealGenome)
- **UniformCrossover** (+ `UniformCrossover_injection`): Independently select each gene from Parent 1 or Parent 2 with probability `crossover_rate`.
- **BlendCrossover (BLX-α)** (+ `BlendCrossover_injection`): Sample uniformly from extended interval around parents.
- **SimulatedBinaryCrossover (SBX)** (+ `SimulatedBinaryCrossover_injection`): Polynomial distribution-based crossover with parameter `eta`.
- **BinomialCrossover** (+ `BinomialCrossover_injection`): Differential evolution style selection.
- **EvosaxUniformCrossoverWrapper**: Direct wrapper around Evosax crossover with both standard and injection modes (see below).

**Mode Availability**:
- All real-valued operators have standard and injection variants (registered with `_injection` suffix)
- EvosaxUniformCrossoverWrapper supports both modes via `injection_mode` parameter

---

## Operator Selection Guide

| Problem Type | Recommended Operator | Parameters | Notes |
|--------------|---------------------|------------|-------|
| Binary / Combinatorial | Uniform Crossover | rate=0.5 | High disruption, good exploration |
| Binary / Building Blocks | Single-Point | default | Preserves adjacency |
| Real / Independent Genes | Uniform Crossover | rate=0.6 | Simple per-gene mixing |
| Real / Exploration | Blend (BLX-α) | α=0.5 | Explores outside parental range |
| Real / Exploitation | SBX | η=20-30, num_offspring≤2 | Parent-centric, distribution-aware (default 2 offspring, configurable) |
| Differential Evolution | Binomial | rate=0.5 | Directional, mutant-biased |

---

## Evosax Integration

MalthusJAX provides **EvosaxUniformCrossoverWrapper**, a high-compatibility wrapper around Evosax's crossover operators. This enables direct use of Evosax algorithms while leveraging MalthusJAX's infrastructure (xmap, state management, result tracking).

### Compatibility Context

**API Generations**:
- **evosax 0.1.6** (PyPI, current stable): Lower-level fitness evaluation, no ask/tell algos
- **evosax GitHub main**: Ask/tell algorithm interface, modern `BBOBProblem` API
- **MalthusJAX 0.1.6+**: Dual-mode support via compatibility layer

**Compatibility Layer** (`src/malthusjax/compat/evosax_mimic.py`):
- Pure JAX implementation of Evosax mutation/crossover
- Enables MalthusJAX to work with evosax 0.1.6 without external dependencies
- Used internally by EvosaxUniformCrossoverWrapper for robust operation

### EvosaxUniformCrossoverWrapper: Core Integration API

**Purpose**: Direct wrapper enabling Evosax crossover functions within MalthusJAX workflows.

**Invocation**:
```python
from malthusjax.operators.crossover import EvosaxUniformCrossoverWrapper

wrapper = EvosaxUniformCrossoverWrapper(
    evosax_crossover_fn=evosax.crossover,  # or your custom crossover function
    crossover_rate=0.5,
    injection_mode=False,   # True for injection-style semantics
    dtype=jnp.float32
)

offspring = wrapper(keys, parents1, parents2, config)
```

**Modes**:

1. **Standard Mode** (`injection_mode=False`):
   - Standard Tier-1 semantics: pure deterministic selection using masks
   - Contract: `offspring = jnp.where(mask, p2.values, p1.values)`
   - Best for: Standard crossover or swapping with external EAs

2. **Injection Mode** (`injection_mode=True`):
   - Gate-based semantics: computed offspring OR parent unchanged
   - Contract: `offspring = jnp.where(should_cross, computed, p1.values)`
   - Best for: Adaptive operators (BlendCrossover, SBX) where not crossing is meaningful

**Configuration**:
```python
# Both modes registered in MalthusJAX ecosystem
config_standard = CrossoverConfig(
    rate=YOUR_RATE,
    injection_mode=False
)

config_injection = CrossoverConfig(
    rate=YOUR_RATE,
    injection_mode=True
)
```

### Integration with Evosax Algorithms

**Direct Composition**:
```python
from malthusjax.operators.crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.mutation import EvosaxMutationWrapper
import evosax

# Create wrappers
crossover = EvosaxUniformCrossoverWrapper(
    evosax_crossover_fn=evosax.crossover,
    crossover_rate=0.7
)
mutation = EvosaxMutationWrapper(
    evosax_mutation_fn=evosax.mutation,
    std_dev=0.1
)

# Use in evolve loops (Tier 3 vmaps handle population aggregation)
```

**State Management Considerations**:
- Evosax ask/tell algorithms maintain internal state (population, fitness cache)
- MalthusJAX loops manage PRNG key streams explicitly
- For bidirectional integration, use `EvosaxAdapter` (see `src/malthusjax/composer/evosax_adapter.py`)

### Fitness Evaluators with Evosax

**BBOBFitness Integration**:
```python
# 0.1.6 compatible
from evosax.problems import BBOBFitness

fitness = BBOBFitness(num_dims=10, function_id=1)
R, Q = fitness.get_rotation_matrices(key)
scores = fitness.rollout(key, solutions, R, Q)
```

See [core/fitness/bbob_evaluator.py](../../core/fitness/bbob_evaluator.py) for full integration pattern and `BBOB_NAME_ALIASES` normalization.

### Architecture Pattern: Why Wrapping Works

1. **Tier 1 Absorption**:
   - Evosax crossover functions are already deterministic (no RNG inside)
   - EvosaxUniformCrossoverWrapper absorbs them as pure Tier-1 logicCrossovers

2. **Key Stream Management**:
   - MalthusJAX provides PRNG keys via `_generate_noise()`
   - Evosax operators consume keys internally or MalthusJAX provides randomness

3. **Genome Type Translation**:
   - Wrapper converts MalthusJAX genomes (`RealGenome`, etc.) to numpy arrays
   - Applies Evosax operations
   - Converts results back to MalthusJAX genome type

### Troubleshooting Evosax Integration

| Problem | Solution |
|---------|----------|
| `ImportError: 'evosax.algorithms'` | Ensure evosax ≥0.1.6 from PyPI; use compat layer |
| `BBOBProblem not found` | Use `BBOBFitness` (0.1.6) or wrap GitHub evosax separately |
| Dimension mismatch | Verify genome shape matches evosax expectation (flat arrays) |
| Mode confusion | Standard = per-element selection; Injection = gate-based |
| Key consumption mismatch | Check `num_keys_per_atomic_operation` alignment |

### Extension: Adding New Evosax Operators

**Recipe** (for adding future Evosax operators):

1. **Wrapper class**:
   ```python
   class MyEvosaxCrossoverWrapper(BaseCrossover):
       def __init__(self, evosax_fn, param1, ...):
           self.evosax_fn = evosax_fn
           ...
       
       def _recombine_one(self, p1, p2, noise_data, config):
           # Convert to array, call evosax_fn, convert back
           ...
       
       def _generate_noise(self, keys, config):
           # Prepare parameters needed by evosax_fn
           ...
   ```

2. **Register** in [__init__.py](.//__init__.py) export list

3. **Test** against parity with direct Evosax calls (see `test_evosax_crossover_parity.py`)

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
   - Pair-major flattening (direct `reshape`, no transpose) for XLA fusion

Together: **JIT compilation**, **reproducibility**, **static budgeting**, and **XLA kernel fusion**.
