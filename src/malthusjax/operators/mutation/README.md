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
  - Vaccinate constants and intermediate scalars with `jnp.array(..., dtype=config.dtype)` to prevent BF16 -> FP32 implicit promotion.
  - Return a new, immutable genome dataclass (e.g., `genome.replace(values=new_values)`).

Why:
- Keeping this tier pure and stateless makes it trivially `vmap`-able and JIT-friendly. It also helps enable bulk-injection (Mode D) where precomputed noise is injected into this kernel.

---

## Tier 2 — Noise Generation (Entropy Producers) 🎲

- Contract: `_generate_noise(keys: chex.Array, config: C, ...) -> noise_data`
- Responsibilities:
  - Consume an exact slice of PRNG keys (the block length is fixed by `num_keys_per_atomic_operation`).
  - Produce masks, Gaussian/Uniform noise, or any domain-specific stochastic values shaped to `config.shape`.
  - Ensure dtype correctness for all generated arrays (use `dtype=config.dtype`).

Notes:
- This tier is **agnostic** to how keys are sourced — whether per-individual (`Mode A`) or via bulk injection (`Mode D`).

---

## Tier 3 — Vectorized Wrapper (Resource-aware Lifting) 🚀

- `BaseMutation.__call__(all_keys, population, config, **kwargs)` is the vectorized lifting layer.
- Responsibilities:
  - Reshape `all_keys` according to `pop_size`, `num_offspring`, and `num_keys_per_atomic_operation` and distribute them to per-individual invocations.
  - Use `jax.vmap` to lift `_mutate_with_keys` or `_mutate_one` across the population efficiently.
  - Use `population.spawn_offspring(new_genes)` (or `.replace`) to return a properly-typed `BasePopulation[G]` with reset fitness metadata.

Shape contract example:
- `all_keys` is expected to be shaped so it can be reshaped as `(pop_size, num_offspring, num_keys_per_atomic_operation, 2)` for two-word Threefry keys.

---

## Resource Mapping & Static RNG Budget (ResourceMapper) 🗺️

- The `ResourceMap` (computed by `compute_resource_map(...)`) pre-calculates the total RNG budget for the full pipeline:
  - Calls `operator.num_keys(input_shape)` for selection, crossover, and mutation.
  - Produces `total_rng_budget` and per-operator `OperatorAllocation` slices (start/end indices)
- Benefits:
  - Static memory allocation for key buffers, enabling predictable device placement and host-device transfer minimization.
  - Ability to request a single master split and slice it deterministically for each operator.

---

## PRNG Topologies & the Ablation Study (Mode A vs Mode D) 🔁

### Mode A — Standard (Evosax / Per-Individual Sampling)
- Each individual uses a unique subkey from a host-split call: `K_i = split(master_key)[i]` and the sampler internally uses `Threefry(K_i, 0)`.
- Pros: conceptually simple, bitwise reproducible with same per-sample keys.
- Cons: multiple small RNG calls (split/hash) add "hashing tax" and reduce kernel fusion opportunities.

### Mode D — Bulk Injection (Counter-Advancing Threefry)
- Generate one bulk noise tensor with a single RNG call: `noise = jr.normal(master_key, shape=(N, dims))`. This uses changed counter topology (`Threefry(K, i)`).
- Pros: Enables XLA to fuse RNG generation and per-element arithmetic in one kernel (significant throughput gains on H100/A100).
- Cons: Bitwise outputs differ from Mode A (Threefry mapping changes). Tests must validate statistical equivalence rather than bitwise identity.

### Mathematical divergence
- Both methods yield statistically equivalent distributions, but exact sequences differ because `Mode A` uses isolated subkeys with counter 0 while `Mode D` advances counters per batch index. This is a deterministic, expected divergence in Threefry-based PRNGs.

---

## GaussianMutation — Example Flow (Tier Walkthrough) 🌊

**Tier 3** (`BaseMutation.__call__`)
- Engine provides `all_keys` (slice from `ResourceMap`) and `population`.
- `all_keys` is reshaped into per-individual/per-offspring blocks.
- Lift with `jax.vmap` into the per-individual processing pipeline.

**Tier 2** (`_generate_noise`)
- For each atomic operation (or once in bulk):
  - `mask = jax.random.bernoulli(k_mask, p, shape=config.shape).astype(config.dtype)`
  - `noise = jax.random.normal(k_noise, shape=config.shape, dtype=config.dtype) * jnp.array(mutation_strength, dtype=config.dtype)`
- In Mode D, `k_mask` and `k_noise` are used to sample whole `(N, *config.shape)` arrays in one call.

**Tier 1** (`_mutate_one`)
- `new_values = genome.values + (noise * mask)`
- Optionally `jnp.clip(new_values, min, max)` with typed bounds.
- `return genome.replace(values=new_values)`

**Finalization**
- `spawn_offspring(new_genes)` returns the mutated `BasePopulation` with `fitness` reset.

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

- [ ] Define `num_keys_per_atomic_operation` (used by ResourceMapper).
- [ ] Implement `_generate_noise(keys, config)` with explicit `dtype=config.dtype` for outputs.
- [ ] Implement `_mutate_one(genome, noise_data, config)` pure and promotion-free.
- [ ] Ensure Tier 1 uses masked arithmetic and returns `genome.replace(...)`.
- [ ] Support both per-individual (Mode A) and bulk injection (Mode D) invocation paths at Tier 3.
- [ ] Add tests: statistical equivalence (mean/std), bitwise divergence (L2 > 0), and promotion-safety (no undesired upcasts).

---

## References (key symbols & files)

- `malthusjax.operators.base.BaseMutation` — Tier 3 lifting logic
- `malthusjax.operators.mutation.real` — concrete `GaussianMutation`, `BallMutation`, `PolynomialMutation`
- `malthusjax.operators.mutation.ablation_mutation` — Mode D bulk injection implementations
- `malthusjax.engine.resource_mapper` — `compute_resource_map` and `ResourceMap` allocation