# Resource Mapper in MalthusJAX## 📚 Overview
- The **Resource Mapper** (`src/malthusjax/engine/resource_mapper.py`) is Step 3 of the Optimization Roadmap.
- Its primary job is to pre-calculate exact Random Number Generation (RNG) requirements and data shapes for the entire population pipeline before any JIT compilation occurs.
- **Engine Initialization**: The `ResourceMapper` computes these allocations precisely at `Engine.__init__` time. By doing this upfront, the XLA graph treats the total PRNG dimensionality as a strict constant during JIT compilation, entirely avoiding dynamic shape issues.
- This allows MalthusJAX to allocate one massive static array of PRNG keys and route exact slices to each operator, guaranteeing complete determinism and XLA compatibility without dynamic allocations.

---

## 1️⃣ Key Derivation Strategies
The `KeyDerivationStrategy` determines how the engine derives thousands of keys from a single `master_key` per generation.

- **`SPLIT`**: Uses `jax.random.split` sequentially. This is the statistical gold standard (guarantees strictly uncorrelated keys), but can be slower and is strictly single-threaded.
- **`FOLD`**: Uses `jax.random.fold_in` with sequential integer indices inside a `vmap`. This is highly parallelizable and scales much better to massive key budgets on multi-device setups, though it relies on deterministic hashing rather than true PRNG splitting.

---

## 2️⃣ Sharding Manager (GSPMD)
The `ShardingManager` is responsible for handling population-level data parallelism across multiple devices (e.g., TPUs or multi-GPU).
- It creates a Mesh layout using `NamedSharding`.
- The `matrix_sharding` partitions the `(pop_size, genome_dim)` arrays along the `pop_size` axis.
- The `vector_sharding` partitions the `(pop_size,)` fitness array.
- Using `alloc_population`, it ensures that zero-initialized populations are placed securely on the device mesh *before* the evolution loop starts, preventing host-to-device transfer bottlenecks.

---

## 3️⃣ Operator Allocations & The `ResourceMap`
Instead of operators randomly calling `jax.random.split` on the fly, the engine calculates a `ResourceMap` upfront.

For each operator (Selection, Crossover, Mutation, Evaluation), it calculates an `OperatorAllocation`:
- `num_keys`: How many exact PRNG keys are required.
- `start_idx` & `end_idx`: Where in the master key array this operator's keys live.
- `input_count`: How many individuals enter this stage.
- `output_count`: How many individuals exit this stage.

---

## 4️⃣ The Cascade Logic (`compute_resource_map`)
This function simulates the data flow of the entire generation mathematically to budget resources. 

Given `pop_size`, `elitism`, and the configured operators:
1. **Selection**: Requires `pop_size` inputs, outputs `parents_needed` indices.
2. **Crossover**: Requires `parents_needed` inputs, outputs `offspring_count` genomes.
3. **Mutation**: Requires `offspring_count` inputs, outputs `mutant_count` genomes.
4. **Next Key**: Allocates 1 key to seed the next generation.

*Note*: If crossover over-produces (e.g., you need 15 offspring but it produces in pairs of 2, generating 16), the resource mapper explicitly logs a warning but allows it, advising the user to tweak `pop_size` to minimize waste.

---

## 5️⃣ Interaction with Emitters (Quality-Diversity)
In Quality-Diversity (QD) algorithms (like MAP-Elites), standard selection and crossover don't apply linearly. Instead, MalthusJAX uses **Emitters**.

Emitters act as both the compositional interface and the ResourceMapper endpoint:
- **`num_keys_per_atomic_operation`**: Emitters declare exactly how many keys they need per offspring.
- **`batch_size`**: Emitters declare how many offspring they generate per generation.
- **`num_keys()`**: The ResourceMapper calls this to allocate the total budget (`batch_size * num_keys_per_atomic_operation + sampling_keys`).

### The `ask()` Endpoint (Tier 3 Orchestration)
The `Emitter.ask()` method receives the pre-allocated flat buffer of keys from the ResourceMapper.
It is responsible for:
1. Slicing out a key to sample parents from the QD Repertoire (Tier 2).
2. Reshaping the rest of the key buffer into `(batch_size, atomic_keys)` to cleanly pass into a `jax.vmap` that executes the pure atomic genetic modification (Tier 1).

---

## 6️⃣ Debugging & Visualization
The Resource Mapper provides two incredible helper functions to visualize what XLA is doing under the hood:

- **`get_resource_summary(rmap)`**: Prints the exact RNG budget and input/output counts for every stage.
- **`get_step_dimension_flow(rmap)`**: Generates a symbolic, exact phase-by-phase dimension flow (e.g., `p1_pop: (p, d) -> crossover out: (p * 2, d)`). This is invaluable for debugging custom operators to ensure your shapes align perfectly with the Engine's expectations.

```python
from malthusjax.engine.resource_mapper import compute_resource_map, get_step_dimension_flow

rmap = compute_resource_map(selection, crossover, mutation, pop_size=100)
print(get_step_dimension_flow(rmap))
```
