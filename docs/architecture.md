# MalthusJAX JAX Execution Architecture

This document provides a technical guide to the execution model, PyTree data structures, and compilation boundaries in **MalthusJAX**.

---

## 1. High-Level System Design

MalthusJAX is structured into four distinct layers that decouple high-level experiment configuration from hardware-accelerated XLA compilation:

```text
              Python Configuration / DSL
                         │
                         ▼
                   Composer Layer
                         │
                         ▼
                   Engine Layer
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Selection      Crossover       Mutation
          │              │              │
          └──────────────┼──────────────┘
                         ▼
               Evolution State (PyTree)
                         │
                         ▼
                   jax.lax.scan
                         │
                         ▼
                    XLA Kernel
```

- **Composer Layer**: Parses TOML configurations or Python specs, resolves registered operators/evaluators, and constructs the appropriate engine instance.
- **Engine Layer**: Manages the 5-phase generational loop (`entropy`, `selection`, `reproduction`, `merge`, `evaluation`).
- **Operators Layer**: Contains vectorized genetic operators implemented across progressive vectorization tiers.
- **Core Layer**: Implements immutable PyTree structures (`BaseGenome`, `BasePopulation`, `GeneticEngineState`) and deterministic PRNG key management.

---

## 2. Compiling Generational Loops with `jax.lax.scan`

Traditional evolutionary frameworks loop over generations in Python, incurring host-device context-switching overhead on every step. MalthusJAX instead encapsulates the entire evolutionary state inside a single JAX PyTree and uses `jax.lax.scan` to lower the generational loop into one compiled XLA program.

```python
import jax
import jax.numpy as jnp

def step_fn(state, _):
    # Pure state transition executing on GPU/TPU VRAM
    next_state = engine.step(state)
    return next_state, next_state.best_fitness

# Compile the entire 1,000-generation evolution into ONE XLA kernel
compiled_evolution = jax.jit(
    lambda init_state: jax.lax.scan(step_fn, init_state, None, length=1000)
)
```

---

## 3. Core Principles & JAX Performance Engineering

### Struct-of-Arrays (SoA) PyTrees
Populations are stored as Struct-of-Arrays rather than Array-of-Structs. A population of 10,000 individuals with 10-dimensional continuous genomes is stored as a single JAX array of shape `(10000, 10)` inside a `RealPopulation` PyTree node. This layout maximizes SIMD memory alignment on GPU tensor cores.

### Deterministic PRNG Resource Allocation
Generating entropy inside `jax.jit` functions requires managing pseudo-random keys without side effects. Engine states maintain a deterministic PRNG counter/key inside `GeneticEngineState`. At the beginning of each step (`allocate_entropy`), keys are split deterministically for selection, crossover, and mutation without returning control to Python.

### Buffer Donation Semantics (`clone_buffers()`)
To avoid memory allocation churn across thousands of generations, JAX supports buffer donation (`donate_argnums`). MalthusJAX PyTrees provide `clone_buffers()` methods to guarantee clean buffer ownership separation before handing buffers to JAX trace environments.

---

## 4. Multi-Tier Operator API

MalthusJAX does not force every genetic operator into a single fixed vectorization model. Instead, it offers a **progressive operator API** that allows users to select the appropriate semantic level:

| Tier | Semantic Level | Target Signature | Lifting & Framework Handling | Control |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | Single Genome | `mutate(genome, key)` | Framework lifts to population via `jax.vmap` | High Convenience |
| **Tier 2** | Controlled Noise | `mutate(genome, key, noise)` | Pre-allocates noise tensors, lifts via `vmap` | Medium |
| **Tier 3** | Population Kernel | `mutate(population, key)` | User owns population-wide vectorization & interactions | High Control |

### Design Guidance
> **Start at Tier 1 (Genome Level).** Write the natural mathematical algorithm for a single individual. Drop to **Tier 3 (Population Level)** only when your algorithm intrinsically requires population-wide interactions (e.g. covariance matrix adaptation, population normalization) or custom batching strategies.

---

## 5. Summary of Key Invariants

1. **Pure Functional State Transitions**: Engine `step(state)` functions are pure functions returning `(next_state, metrics)`.
2. **Static vs. Dynamic Boundaries**: Configuration fields (bounds, population size, genome length) are static PyTree metadata, while population genes and fitness arrays are dynamic JAX tracers.
3. **Seed Alignment**: Multi-pipeline benchmarking uses seed-aligned initial populations to eliminate PRNG noise when comparing evolutionary operators.
