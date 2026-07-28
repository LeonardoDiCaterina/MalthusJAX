# The Engine: Orchestration and Compilation

The `malthusjax.engine` module implements Tier 2 of the framework. This tier orchestrates populations, fitness evaluators, and genetic operators into a fully compilable evolution loop using a **scan-based stateful architecture**. 

The core design principle is **immutable state threading**: the evolution loop carries one opaque state object through `jax.lax.scan`, guaranteeing JAX traceability and JIT compilability to a single XLA kernel.

The framework provides two execution modes: `step()` for iterative in-process evolution, and `ask()`/`tell()` for external fitness evaluation (e.g., remote simulations). Both modes preserve identical state semantics.

## 4.1. The ResourceMapper

In the v2 architecture, the Engine heavily delegates PRNG key management to the `ResourceMapper`. 

At initialization, the `ResourceMapper` computes a deterministic resource allocation plan. It queries every operator in the pipeline for its `num_keys` requirement, and pre-computes where in the flat key array each operator's slice is located.

This pre-computation enables **zero-overhead key slicing** during each generation: the key derivation is done once, and operators receive their slices as pure inputs without dynamic lookup or recursion. The resource map is marked metadata (not traced), avoiding JAX overhead.

Two strategies are available for generating the per-generation key block:
- **SPLIT**: Sequential hash chain (`jax.random.split`), lower memory but sequential parallelism.
- **FOLD**: Fully parallel (`vmap(fold_in)`), higher memory overhead but deterministic and seekable.

## 4.2. Generational Loop Architecture: The Baseline GA

The baseline `GeneticEngine` orchestrates a standard generational loop. Each call to `step(state)` executes one generation in five phases:

### 4.2.1. Phase 0 — Entropy Allocation
The PRNG key is partitioned using the pre-computed `ResourceMapper` table. Operators receive ready-to-use key blocks.

### 4.2.2. Phase 1 — Selection
Selection operators read the Population's fitness vector and return parent and elite index arrays. Crucially, elitism creates a structural partition in the batch dimension.

### 4.2.3. Phase 2 — Reproduction
The `Population` is bridged to the Tier-3 operator mechanism (`tree_leaves`). Parent pairs are gathered; crossover and mutation apply unbounded noise. The payload is snapped back to valid bounds via `autocorrect()`, and a new offspring Population is spawned.

### 4.2.4. Phase 3a — Merge
Elite genes and mutated offspring are combined into the next-generation population using JAX's `dynamic_update_slice`. This permits XLA to reuse the old population buffer in-place through buffer donation.

### 4.2.5. Phase 3b — Evaluate
The merged population is evaluated, updating the `fitness` vector.

## 4.3. Multi-Objective Orchestration (NSGA-II)

The `MOEngine` orchestrates Multi-Objective optimization. It differs from the baseline GA by overriding the `Selection` and `Merge` phases.

In MO, fitness is a vector of objectives. The engine uses **Non-Dominated Sorting** and **Crowding Distance** to rank the population.

1. **Merge (Elitism)**: Instead of a strict elite partition, the MO engine merges the entire parent population ($N$) and the offspring population ($N$) into a combined pool of size $2N$.
2. **Ranking**: It computes Pareto ranks and crowding distances for the $2N$ pool.
3. **Truncation**: It truncates the pool back to size $N$ by selecting the best Pareto fronts, using crowding distance as a tie-breaker for the final front.

This state tracking (ranks, distances) is securely stored in the specialized `MOPopulation` container.

## 4.4. Quality-Diversity Orchestration (MAP-Elites)

The `QDEngine` orchestrates Quality-Diversity search, specifically MAP-Elites. Instead of a linear population, it maintains an **Archive** of behaviorally diverse elites.

1. **Evaluation**: Evaluators return both a `fitness` scalar and a `descriptor` vector.
2. **Mapping**: The engine maps the `descriptor` to a discrete cell index in the Archive.
3. **Archive Update**: If the new individual maps to an empty cell, or has better fitness than the existing occupant, it replaces them. The Archive is maintained as a static-sized buffer using `dynamic_update_slice`.
4. **Selection**: Parents are sampled uniformly from occupied cells in the Archive.

The `QDPopulation` container safely tracks `descriptors` and `cell_indices` during this process.

## 4.5. The Island Model (Distributed Topology)

The `IslandModelEngine` orchestrates parallel evolution across multiple sub-populations ("islands"). It introduces a structural **Migration** phase.

At fixed intervals (the migration epoch), islands exchange individuals based on a specific topology (e.g., Ring, Fully Connected) and a migration policy (e.g., Best replaces Worst, Random replaces Random). 

To ensure GSPMD compatibility and avoid dynamic routing overhead, migration is implemented using deterministic permutations (`jax.numpy.take` with static indices) across the island dimension. This allows XLA to optimize cross-device communication via collective communication primitives (like `AllGather` or `Permute`).

## 4.6. Operator-Level Personalization and Scheduling

Operators accept an optional generation counter as a parameter, enabling time-dependent behavior (e.g., mutation rate decay, crossover bias modulation). Scheduling functions are pure JAX, compiled into the kernel, and safe inside `jax.lax.scan`. 

Schedule types include:
- **CONSTANT**: Time-independent (baseline)
- **LINEAR_DECAY**: Strength decreases linearly
- **COSINE_ANNEAL**: Cosine annealing schedule
- **EXPONENTIAL_DECAY**: Exponential decay schedule
