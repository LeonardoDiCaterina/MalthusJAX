# Selection Operators — Architecture & Implementation 🎯

This document explains how selection operators are implemented in MalthusJAX, and how they interact with the static ResourceMapper, sharding, and PRNG topologies. It targets developers extending or optimizing selection stages in the engine.

---

## 🔍 High-level Summary

- Selection is the **entry point of the Cascade**: it consumes the population size `N` and produces an index buffer `P` (parent indices) which becomes the input to Crossover and then Mutation.
- The system uses **static RNG budgeting** via `ResourceMap` to ensure zero or minimal allocation during the main loop and predictable shapes for compilation.
- Selection implements a clear separation between **atomic logic** (a pure `_select_one`) and **vectorized slicing** (`BaseSelection.__call__`) that produces a batched `jnp.ndarray` of indices.

---

## 1) Static RNG Budgeting & Resource Mapping 🧮

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

## 2) The Cascade Effect — Selection → Crossover → Mutation 🔁

- Selection determines how many parent indices are needed (`sel_output_count`) based on `num_selections` and `input_shape`.
- These indices are used to gather parent genomes for crossover. The number of parents determines the number of parent pairs, which in turn determines crossover output and subsequent mutation input sizes.
- The ResourceMapper computes the complete cascade so each operator knows how many keys and buffer shapes will be used downstream.

---

## 3) Operator Interface & Functional Logic 🧩

- **Atomic Logic (`_select_one`)**
  - A pure function that consumes PRNG key slices and the population fitness array and returns a single integer index.
  - Example signature:

```py
def _select_one(keys: chex.Array, fitness: chex.Array, config: C) -> jnp.int32:
    # pure JAX code, no side-effects
```

- **Vectorized Slicing (BaseSelection.__call__)**
  - `BaseSelection.__call__(keys, population, config)` reshapes the master key slice into per-selection blocks and vmaps `_select_one` across those blocks.
  - Returns a batched `jnp.ndarray` of parent indices with dtype `jnp.int32` or `jnp.int64` depending on platform and expected indexing range.
  - The selection layer intentionally returns indices (not a sliced population) to keep the operator lightweight; gathering can be done once by the Engine using a single `jax.vmap` or `jax.lax.dynamic_slice` to minimize copies.

Benefits of returning indices:
- Delays memory movement until the engine-level gather, enabling better fusion with downstream ops.
- Keeps the selection operator focused and testable.

---

## 4) Hardware-Specific Optimization & Sharding 🖥️

- `ShardingManager` defines named sharding specs (`pop_sharding`, `vector_sharding`) and can be used to `device_put` selection's index buffer so that it is already sharded across devices.
- Fitness arrays should be sharded consistently with population layout so selection index computation remains local to devices where possible, minimizing cross-device moves.
- When running on many devices, `split_key_sharded` helps distribute RNG keys across the mesh to ensure per-device independent streams.

Example: place selection indices on `pop_sharding` so subsequent gather operations are local to each device.

---

## 5) Promotion-Free Indices & Type Hygiene 🔢

- Selection indices MUST be integer dtypes. Use `jnp.int32` or `jnp.int64` explicitly when creating indices to prevent accidental upcasting during downstream arithmetic or indexing operations.
- Avoid any FP ops on indices; cast to integers immediately after sampling (e.g., sampling uniform integers via `jax.random.randint(..., dtype=jnp.int32)`).

---

## 6) Ablation Context — Mode A vs Mode D for Selection 🧪

- **Mode A (Standard)**: For each selection event, use per-event keys (e.g., via `split`) and call atomic selection logic. This is straightforward and bitwise reproducible under the same key slicing.

- **Mode D (Bulk / Injection)**: Generate a single bulk index tensor (e.g., `jr.randint(master_key, shape=(S,), minval=0, maxval=N, dtype=jnp.int32)` or by bulk-shuffling indices) in one call. This creates a single HLO that produces all parent indices in one pass.

Trade-offs:
- Mode A gives exact per-sample reproducibility but incurs many small RNG operations that prevent HLO fusion.
- Mode D reduces the "hashing tax" and enables the XLA compiler to fuse the index-generation code, improving throughput, especially in multi-device settings.

Ablation tests should measure: index generation throughput, host-device transfers, and the divergence (bitwise and statistical) between Mode A and Mode D.

---

## 7) Zero-Allocation Loop & Data Flow Integrity ✅

- The engine calls `compute_resource_map(...)` once to get `ResourceMap` containing exact start/end key indices and required shapes.
- For selection, the engine slices `all_keys[sel_slice]` and hands that slice to the selection operator.
- `BaseSelection.__call__` returns an indices array which the engine uses to perform a single batched gather (e.g., `jax.vmap(lambda idx: population.genes[idx])`) or an efficient `jax.lax.gather` using the sharded indices.
- Because all shapes and key budgets are precomputed, the inner loop can operate with zero dynamic allocation—keys and index buffers are pre-allocated and reused across generations.

---

## 8) Technical Summary (For Records) 📝

- **Input/Output Contract**: Selection accepts `pop_size` implicitly (via `population`) and outputs `sel_output_count` indices computed from `num_selections`.
- **Key Budgeting**: `sel_keys_needed` computed via `selection.num_keys((pop_size,))` where `num_keys` uses `num_keys_per_atomic_operation`.
- **Decoupled Slicing**: Selection returns indices (not new populations) to minimize early memory movement.
- **PRNG Topology**: Supports both Case A (split-per-selection) and Case D (bulk RNG for indices), with expected bitwise divergence but comparable statistical properties.

---

## 9) Developer Checklist — Implementing a Selection Operator ✅

- [ ] Define `num_keys_per_atomic_operation` precisely.
- [ ] Implement `_select_one(keys_slice: chex.Array, fitness: chex.Array, config: C) -> jnp.Int` as a pure function.
- [ ] Ensure `__call__` reshapes `all_keys` into `(num_selections, num_keys_per_atomic_operation, 2)` and vmaps `_select_one` over these blocks.
- [ ] Return an integer `jnp.ndarray` of indices and document expected shape and dtype.
- [ ] Add ablation tests comparing Mode A and Mode D for throughput and correctness.
- [ ] Ensure tight sharding placement via `ShardingManager` when running on multiple devices.

---

## References (key files)

- `malthusjax.operators.base.BaseSelection` — interface & `__call__`
- `malthusjax.engine.resource_mapper` — `compute_resource_map` and `ResourceMap`
- `malthusjax.engine.resource_mapper.ShardingManager` — sharding utilities

---

If you'd like, I can add a minimal example selection implementation (e.g., TournamentSelection) in `src/malthusjax/operators/selection/` and wire an ablation test that compares Mode A vs Mode D. Would you like that? ✨