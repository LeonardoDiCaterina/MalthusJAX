# malthusjax.core.genome — Technical Overview

This document describes the design, extension patterns, and JAX integration for the `malthusjax.core.genome` package. It's intended for developers who will implement, extend, or use genome types and population containers in MalthusJAX.

---

## Overview

- **Purpose**: Provide immutable, JAX-compatible genome primitives and population containers that are easy to JIT/VMap and to extend for algorithm-specific behaviors.
- **Key classes**: `BaseGenome`, `BasePopulation` (generic `BasePopulation[G]`), distance metric utilities (`DistanceMetric`, `RealDistanceMetric`), and concrete implementations such as `RealGenome`, `RealGenomeConfig`, and `RealPopulation`.

---

## The Struct-of-Arrays (SoA) Paradigm

**Concept**: The codebase follows a Struct-of-Arrays (SoA) architecture: a `BaseGenome` represents the logic and structure for a *single* individual, while a `BasePopulation` holds a "lifted" (batched) version where every leaf in the PyTree has a leading population dimension `N`.

- Single individual:
  - `RealGenome.values` → 1-D array `(length,)`
- Population container:
  - `RealPopulation.genes.values` → 2-D array `(N, length)` (each leaf gains a leading `N` dimension)

Why this matters:
- JAX-friendly vectorization is straightforward because population-level arrays are the same PyTree structure but with an added leading axis.
- Many algorithms operate at the array level (e.g., `jax.vmap`) over that leading axis for speed and JIT-compatibility.

---

## Extension Pattern

How to extend a genome or population safely and in a type-safe way:

- Extend `BaseGenome` for a domain-specific representation (e.g., `RealGenome`). Provide static `random_init` and any convenience methods for per-individual operations (normalize, add_noise, etc.).
- Extend `BasePopulation[G]` to preserve the concrete genome type in the population container (e.g., `RealPopulation(BasePopulation[RealGenome])`). This preserves the specific type `G` for type-checkers and generic algorithms.

Best practices:
- Keep the `distance(self, other: BaseGenome, ...)` signature in your `BaseGenome` subclasses. This ensures **Liskov Substitutability** for algorithm code that calls `distance` polymorphically.
- Inside `distance`, cast the incoming `other` to your concrete type: `other_real = cast(RealGenome, other)` — this avoids narrowing the public signature while letting you access subclass internals.
- For population-level factories, provide `init_random(cls, key, config, size)` or helpers that build batched genomes (see `RealPopulation.init_random`).

---

## JAX Integration (vmap / jit / immutability)

- Genomes are implemented as **immutable PyTrees** (via `flax.struct.dataclass`). This is important for JIT compilation and predictable behavior under tracing.
- Use `jax.vmap` to apply per-individual pure functions over populations where each leaf has shape `(N, ...)`.
- Use `jax.jit` to JIT-compile per-individual and batched operators.

Mutation / transformation pattern:
- Because `flax.struct.dataclass` instances are frozen, use the `.replace(...)` factory to create modified instances.
- To keep typing and static-checkers happy, use the `cast(Any, self).replace(...)` pattern and then cast back to the concrete type: e.g.

```py
# inside a RealGenome method
return cast(RealGenome, cast(Any, self).replace(values=some_new_values))
```

- When mapping a per-individual method over a population, prefer using `jax.vmap` directly on the PyTree:

```py
# keys: shape (N,)
# pop.genes: RealGenome with values shaped (N, length)
mutate_vmap = jax.vmap(lambda k, g: g.add_noise(k, 0.05), in_axes=(0, 0))
new_genes = mutate_vmap(keys, pop.genes)
# then update the population immutably
new_pop = cast(RealPopulation, cast(Any, pop).replace(genes=new_genes))
```

Notes:
- Avoid `__post_init__` with behavior that relies on run-time values that will be traced by JAX — this can break tracing.
- Keep methods pure (no side effects) so they are safe under JIT and vmap.

---

## Standard vs. Extended Metrics

- `DistanceMetric` (in `core/base.py`) defines standard metrics and the canonical `distance(self, other: BaseGenome, ...)` signature used across genome types (e.g. Hamming, Euclidean).
- Domain-specific metric classes (e.g. `RealDistanceMetric`) can extend `DistanceMetric` to add more specialized metrics (cosine, normalized Lp, etc.).
- Implementations may choose to expose convenience wrappers that default to sensible metrics (e.g., `RealGenome.distance(..., metric="euclidean")`).

**Important**: Algorithms should call `BaseGenome.distance(...)` polymorphically and should not rely on concrete type internals; concrete subclasses cast `other` internally when needed.

---

## Usage Example

Table: `RealGenomeConfig` fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `length` | int | — | Dimensionality of the real vector |
| `bounds` | Tuple[float,float] | `(-inf, inf)` | (min, max) bounds applied by `autocorrect` |
| `dtype` | `jnp.dtype` | `jnp.float32` | Numeric dtype used by random initialization |

Note: `BinaryGenomeConfig` also supports a legacy `length` keyword that is treated as
`shape=(length,)`. `BinaryGenomeConfig.shape` defaults to `(1,)` to avoid accidental
scalar genomes when a shape is omitted.

Example — initialize and mutate a population using `vmap`:

```py
import jax
import jax.numpy as jnp
from typing import Any, cast

from malthusjax.core.genome.real_genome import (
    RealGenomeConfig,
    RealPopulation,
    RealPopulation,
)

# RNG and config
key = jax.random.PRNGKey(0)
config = RealGenomeConfig(length=10, bounds=(-1.0, 1.0), dtype=jnp.float32)

# Create a population of 32 individuals (batched 'genes' inside)
pop = RealPopulation.init_random(key, config, size=32)

# Prepare keys for per-individual mutation
subkeys = jax.random.split(key, pop.genes.values.shape[0] + 1)[1:]

# Apply per-individual mutation via vmap (RealGenome.add_noise is per-individual)
mutate_vmap = jax.vmap(lambda k, g: g.add_noise(k, noise_std=0.05), in_axes=(0, 0))
new_genes = mutate_vmap(subkeys, pop.genes)

# Replace genes immutably in the population
new_pop = cast(RealPopulation, cast(Any, pop).replace(genes=new_genes))
```

---

## Generics: Why `BasePopulation[G]` matters

- `BasePopulation` is generic in the genome type (`G`). This preserves the concrete genome type across APIs and keeps static typing precise for engines and operators.
- Example: `BasePopulation[RealGenome]` signals to type-checkers and implementers that `.genes` is a `RealGenome` PyTree and that static helpers (like `GENOME_CLS`) match that type.
- This reduces the need for ad-hoc casts across the codebase and improves IDE/autocomplete/analysis accuracy.

---

## Implementation Checklist & Best Practices

- Preserve the `distance(self, other: BaseGenome, ...)` signature in all genome subclasses.
- Use `cast(ConcreteGenome, other)` inside `distance` for specific computations.
- Use `cast(Any, self).replace(...)` to return mutated/factory-changed instances.
- Prefer pure, functional-style methods to remain compatible with `jax.jit` and `jax.vmap`.
- Keep configuration as an immutable `@struct.dataclass` that is `pytree_node=False` for non-array fields.


