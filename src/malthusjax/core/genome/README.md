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

## Distance Metrics & Polymorphic Comparisons

- `DistanceMetric` (in `core/base.py`) defines standard metrics: `HAMMING`, `EUCLIDEAN`, `MANHATTAN`.
- All genome types implement the `distance(self, other: BaseGenome, metric: str)` signature for polymorphic distance computation.
- Each genome type defaults to a sensible metric:
  - **RealGenome**: `metric="euclidean"` (L2 norm, default)
  - **BinaryGenome**: `metric="hamming"` (bitwise mismatch count, default)
  - **CategoricalGenome**: `metric="hamming"` (category mismatch count, default)

**Important**: 
- Algorithms should call `BaseGenome.distance(...)` polymorphically without assuming concrete type internals.
- Concrete subclasses cast `other` internally (`other_real = cast(RealGenome, other)`) to access subclass-specific data.
- When implementing a new genome, provide sensible defaults that match the domain (metrics for permutations differ from continuous spaces).

---

## Usage Example

### RealGenomeConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `shape` | Tuple[int, ...] | `()` | Dimensionality of genome (e.g., `(10,)` for 10-D vector, `(5, 3)` for 5×3 matrix) |
| `bounds` | Tuple[float,float] | `(-inf, inf)` | (min, max) bounds applied by `autocorrect` |
| `dtype` | `jnp.dtype` | `jnp.float32` | Numeric dtype for values |

**Note on Legacy `length` Parameter**: 
- **BinaryGenomeConfig only** supports a legacy `length` keyword, which is converted to `shape=(length,)`
- **RealGenomeConfig does NOT support `length`** — always use explicit `shape=` parameter
- `BinaryGenomeConfig.shape` defaults to `(1,)` to prevent accidental scalar genomes

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
config = RealGenomeConfig(shape=(10,), bounds=(-1.0, 1.0), dtype=jnp.float32)

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
- Document the default distance metric for your genome type (users should not be surprised by defaults).
- For multi-dimensional genomes (e.g., `shape=(5, 3)`), ensure `size` and `shape` properties are correctly defined.

---

## Common Genome Operations

All genomes support standard Python conventions via `BaseGenome`:

```py
# Length queries
len(genome)  # → int (number of elements)
genome.size  # → int (same as len, but property)
genome.shape  # → tuple (e.g., (10,) for 10-D vector, (5, 3) for matrix)

# Indexing and slicing (requires `subscriptable=True`, default for most)
first_element = genome[0]
slice_vals = genome[1:5]
for val in genome:  # iterate over values
    print(val)
```

**Batched population initialization via vmap**:
```py
# Manual approach (rarely needed; BasePopulation.init_random is preferred)
keys = jax.random.split(key, pop_size)
batched_pop = jax.vmap(
    lambda k: RealGenome.random_init(k, config),
    in_axes=0
)(keys)
```

Equivalent to:
```py
batched_pop = RealGenome.create_population(key, config, pop_size)
```

---

## Population Operations

### Slicing and Indexing

```py
# Integer indexing returns a single unwrapped genome
individual = pop[0]  # → RealGenome (not wrapped in population)
assert isinstance(individual, RealGenome)

# Slice/fancy indexing returns a sub-population
sub_pop = pop[10:20]  # → RealPopulation with 10 individuals
mask = fitness > threshold
fitted_pop = pop[mask]  # → filtered population
```

### Distance Matrix

```py
# Compute all pairwise distances in population (uses nested vmap)
dist_matrix = pop.distance_matrix(metric="euclidean")  # shape: (N, N)

# Check diversity via distance statistics
off_diag = jnp.triu_indices_from(dist_matrix, k=1)
avg_diversity = jnp.mean(dist_matrix[off_diag])
```

### Batched Corrections

```py
# Apply autocorrect to all individuals in parallel
corrected_pop = pop.autocorrect(config)
# Ensures all genomes satisfy domain constraints (e.g., bounds for RealGenome)
```

### Iteration

```py
# Iterate over population (Python-side, slow; prefer vmap for hot paths)
for individual in pop:
    print(individual.size)

# For computation, vmap is preferred:
mutated_genes = jax.vmap(
    lambda g: g.add_noise(jax.random.PRNGKey(0), 0.1)
)(pop.genes)
```

---

## Categorical Genomes (Discrete Sequences)

**Use cases**: Permutation problems (TSP), categorical choices, SAT, job scheduling.

`CategoricalGenome` represents discrete sequences where each value is an integer index in `[0, num_categories)`.

### Key Methods

```py
from malthusjax.core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig

# Create a categorical genome (e.g., permutation of 10 cities)
config = CategoricalGenomeConfig(num_categories=10, shape=(10,))
genome = CategoricalGenome.random_init(rng_key, config)

# Check if it's a valid permutation (all values unique in [0, num_categories))
is_valid_perm = genome.is_permutation()  # → JAX scalar boolean

# Convert any sequence to a permutation via argsort (deterministic, JIT-safe)
perm_genome = genome.to_permutation(config)  # → CategoricalGenome with unique values

# Swap two positions (returns new genome)
swapped = genome.swap_positions(pos1=2, pos2=5)

# Count occurrences of a specific category
count = genome.count_category(category=7)  # → JAX scalar
```

### Distance Metrics for Categorical

```py
# Hamming: count differing positions (default)
dist_hamming = genome1.distance(genome2, metric="hamming")

# Euclidean: L2 norm of differences
dist_euclidean = genome1.distance(genome2, metric="euclidean")

# Manhattan: L1 norm of differences
dist_manhattan = genome1.distance(genome2, metric="manhattan")
```

---

## `spawn_offspring` — Population Factory Method

`BasePopulation.spawn_offspring(new_genes, fitness=None)` creates a new population
from a gene PyTree, optionally with a pre-set fitness array.

```py
# Default: NaN fitness (signals pending evaluation)
offspring_pop = parent_pop.spawn_offspring(new_genes)
assert jnp.all(jnp.isnan(offspring_pop.fitness))

# With explicit fitness (e.g., from precomputation)
fitness_values = jnp.zeros(len(new_genes))
offspring_pop = parent_pop.spawn_offspring(new_genes, fitness=fitness_values)
```

**When to use each form**:
- **Without `fitness`** (default): Use in operator-level code where fitness will be computed later by the engine's `_evaluate` phase. The NaN sentinel acts as a safety flag: any downstream code that accidentally uses unevaluated fitness will fail fast rather than silently operating on invalid values.
- **With `fitness`**: Use when fitness is immediately available or will be overwritten in the next step. This avoids allocating a temporary NaN array that will be discarded.


