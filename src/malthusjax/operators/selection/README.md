# Selection Operators — Architecture & Implementation

This document explains how selection operators are implemented in MalthusJAX, and how they interact with the static ResourceMapper and sharding. It targets developers extending or optimizing selection stages in the engine.

---

## High-level Summary

- Selection consumes a population and fitness values, producing an index array of selected individuals.
- The system uses **static RNG budgeting** via `ResourceMap` to ensure deterministic key allocation and predictable shapes for JIT compilation.
- Selection implements a clear separation between **atomic logic** (a pure `_select()` method) and **population-level slicing** (`BaseSelection.__call__`) that produces a batched `jnp.ndarray` of indices.

---

## 1) Static RNG Budgeting & Resource Mapping

- The `ResourceMapper` queries each operator to compute the exact number of keys required:
  - `sel_keys_needed = num_selections * num_keys_per_atomic_operation` (adjusted for input shapes and `num_selections`).
  - `compute_resource_map(...)` stores these allocations in an `OperatorAllocation` for selection.
- The engine allocates a single master key buffer of size `total_rng_budget` and slices this buffer for selection using `ResourceMap.get_key_slice("selection")`.

Table — Key budgeting formula

| Symbol | Meaning |
|--------|---------|
| N | population size |
| S | num_selections (per call) |
| K | num_keys_per_atomic_operation | 
| sel_keys_needed | S * K |

Why static budgeting matters:
- Enables one-time host-side split and deterministic slicing.
- Avoids dynamic allocations and repeated `jax.random.split` calls inside the inner loop, minimizing host-device traffic.

---

## 2) Integration with Engine Operations

- Selection produces index arrays via `num_selections`, which the engine uses to gather parent genomes.
- These indices feed downstream to crossover and mutation operators.
- The ResourceMapper computes total key budgets for all operators so shapes are determined at initialization time.

---

## 3) Operator Interface & Functional Logic

- **Atomic Logic (`_select()` method)**
  - A pure function that consumes PRNG keys and a fitness array and returns selected indices.
  - Signature: `_select(keys: chex.Array, fitness: chex.Array, config: Optional[C]) -> indices: chex.Array`
  - Returns indices matching `num_selections`, dtype int32/int64.

- **Population-Level Selection (BaseSelection.__call__)**
  - `BaseSelection.__call__(keys, population, config)` accepts either a Population object (extracts `.fitness`) or a fitness array directly.
  - Calls the abstract `_select()` method and returns an integer index array.
  - The operator returns indices (not reordered genomes) to keep operations lightweight and enable efficient downstream gathering.

Benefits of returning indices:
- Decouples selection logic from memory movement.
- Allows the engine to perform a single gather operation, minimizing copies and improving fusion.

---

## 4) Hardware-Specific Optimization & Sharding

- `ShardingManager` defines named sharding specs (`pop_sharding`, `vector_sharding`) and can be used to `device_put` selection's index buffer so that it is already sharded across devices.
- Fitness arrays should be sharded consistently with population layout so selection index computation remains local to devices where possible, minimizing cross-device moves.
- When running on many devices, `split_key_sharded` helps distribute RNG keys across the mesh to ensure per-device independent streams.

Example: place selection indices on `pop_sharding` so subsequent gather operations are local to each device.

---

## 5) Index Type Requirements

- Selection indices MUST be integer dtypes (int32 or int64). Create indices explicitly using `jax.random.randint(..., dtype=jnp.int32)` or similar functions.
- Avoid floating-point operations on indices; cast to integers immediately after any sampling.

---

## 6) Key Features

- **Deterministic vs Stochastic**: Selection operators can be deterministic (best, truncation) by setting `num_keys_per_atomic_operation = 0`, or stochastic (tournament, rank-based) requiring PRNG keys.
- **Static Key Budgeting**: The `set_input_length()` method freezes the population size for static key allocation; `num_keys()` returns the total keys needed.
- **RNG Flexibility**: The engine pre-allocates and slices keys from a master buffer, ensuring no dynamic allocation during the evolution loop.

---

## 7) Engine Integration

- The engine calls `compute_resource_map()` once at initialization to determine total RNG budget and shape requirements.
- For selection, the engine slices keys from the master buffer and passes them to the selection operator.
- The selection operator returns indices, which the engine uses for gathering parent genomes via a single batched operation.
- All key allocations and buffer shapes are fixed at initialization, enabling zero dynamic allocation in the inner evolution loop.

---

## 8) Technical Summary

- **Input/Output Contract**: Selection accepts fitness values (either from a Population object or directly) and outputs integer indices of shape `(num_selections,)`.
- **Key Budgeting**: Total keys needed = `num_keys_per_atomic_operation`, computed by `selection.num_keys(input_shape)`.
- **Decoupled Slicing**: Selection returns indices (not reordered genomes) to minimize memory overhead.
- **Type Safety**: Indices are explicitly int32/int64 to prevent accidental promotion during downstream indexing.

---

## 9) Developer Checklist — Implementing a Selection Operator

- [ ] Define `num_keys_per_atomic_operation` (0 for deterministic, ≥1 for stochastic).
- [ ] Implement `_select(keys: chex.Array, fitness: chex.Array, config: Optional[C]) -> indices` as a pure function.
- [ ] Return an integer `jnp.ndarray` of indices with shape `(num_selections,)` and dtype int32/int64.
- [ ] Document the selection logic and any required config attributes.
- [ ] Add unit tests for correctness and shape contracts.
- [ ] Verify `num_keys()` returns the correct total key budget.

---

## References

- `malthusjax.operators.base.BaseSelection` — Abstract base class and `__call__` interface
- `malthusjax.engine.resource_mapper.ResourceMap` — Key budgeting and allocation
- `malthusjax.engine.resource_mapper.ShardingManager` — Multi-device sharding utilities