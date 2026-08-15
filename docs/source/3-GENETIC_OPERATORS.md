# Functional Genetic Operators

This section covers the `malthusjax.operators` module. In MalthusJAX, operators (Selection, Crossover, Mutation, and Emitters) are strictly pure functions (or stateless PyTree structs) that transform an input Population into a new Population without mutating the original arrays in place.

## 3.1. Operator Base Architecture (malthusjax.operators.base)

### 3.1.1. The Pure Operator Contract

Operators enforce stateless, reproducible genetic operations through a standardized signature: `new_population = operator(keys, population, config, generation=0)`. The signature explicitly accepts a batched `Population` object and returns a newly spawned `Population`. 

Operators use the `generation` argument to adjust behavior over time—for example, mutation strength or crossover radius according to a schedule—without modifying the operator's PyTree fields. Any required state must be explicitly passed and returned; internal state updates are prohibited.

### 3.1.2. Operators as PyTrees

Operators are registered as PyTrees, permitting them to be compiled within `jax.jit` functions and passed seamlessly through the Evolutionary Engines. Differentiation between hyper-parameters requiring dynamic updates and static configuration flags is enforced in the PyTree flattening process, with static fields marked `pytree_node=False`.

## 3.2. Selection Mechanisms (malthusjax.operators.selection)

Selection operators filter the population based on fitness scores. Unlike crossover and mutation, selection **does not touch the genome payload**. It only examines the vectors (e.g., `fitness`, `pareto_rank`) stored at the Population level and returns integer indices of chosen individuals. This minimizes memory movement.

Every selection operator returns two complementary index arrays: **parent indices** for breeding and **elite indices** for preservation. The efficient primitive underlying all elite extraction is `jnp.argpartition`, which identifies the top-k elements in O(N) time.

### 3.2.1. Vectorized Tournament Selection
Tournament selection randomly samples `tournament_size` individuals and selects the one with the highest fitness. Adjustable selection pressure is achieved by varying the tournament size.

### 3.2.2. Roulette Wheel / Proportionate Selection
Roulette selection uses the **Gumbel-Max Trick** to sample without replacement in O(1) parallel time, producing a permutation of population indices with the correct distribution efficiently on GPUs.

### 3.2.3. Elite Pool Selection
Elite pool restricts the mating pool to the top `elite_k` individuals. This is the default in evosax's SimpleGA due to its speed (O(N) argpartition), though it severely restricts exploration.

### 3.2.4. EvoSAX Mimic Selection (Parity Operators)
To ensure theoretical fidelity and fair benchmarking against external frameworks, MalthusJAX provides selection operators that perfectly mimic the selection semantics of EvoSAX and other libraries.

## 3.3. The Array-Family Variation Mechanism

For array-based genomes (Real, Binary, Categorical, Series), MalthusJAX employs a highly optimized **three-tier architecture**. This architecture completely isolates domain math (Tier-1) from batching and tree-traversal logic (Tier-3).

### 3.3.1. Tier 3: Population-Level Traversal (`__call__`)

The Tier-3 method is the public entry point. It receives a batched `Population` container and must return a new one. To do this generically without knowing the genome's fields, it performs the following:

1. **Extraction**: Calls `jax.tree_util.tree_leaves(population.genes)` to extract the raw, batched arrays.
2. **Noise Generation**: Calls Tier-2 to generate corresponding noise arrays based on the shapes of the extracted arrays.
3. **Vectorization**: Uses `jax.vmap` to map the Tier-1 scalar kernel over the batch dimension of the raw arrays and noise.
4. **Reconstruction**: Repacks the mutated arrays using `jax.tree_util.tree_unflatten`.
5. **Spawning**: Calls `population.spawn_offspring(new_genes)` to return a pristine batched container.

This allows a single `UniformCrossover` or `GaussianMutation` Tier-3 wrapper to work universally across `RealGenome`, `BinaryGenome`, and `SeriesGenome`.

### 3.3.2. Tier 2: Noise Generation (`_generate_noise`)

Noise generators receive PRNG keys, data types, and **actual array shapes** (not arbitrary config shapes). They output auxiliary data: a Boolean flip mask, a Gaussian perturbation tensor, etc. The output noise must match the shape requirements of the Tier-1 kernel.

### 3.3.3. Tier 1: Pure Unbatched Kernels (`_mutate_one`, `_recombine_one`)

Tier-1 kernels represent pure arithmetic. They receive a single, unbatched array and a corresponding noise array. They do not know about Genomes, PyTrees, or Populations. 

*Example: `GaussianMutation._mutate_one(arr, noise)` simply returns `arr + noise`.*

Because domain logic like clipping is deferred to the genome's `autocorrect()` method, Tier-1 operators remain perfectly generic.

## 3.4. Emitters: Complex & Topological Variation

While the Array-Family Mechanism relies on structurally static PyTrees and scalar `vmap`s, some representations undergo complex topological changes—such as adding nodes or edges to a neural network (`TensorNEATGenome`). 

For these, MalthusJAX introduces **Emitters**. 

Emitters are a specialized class of operators that replace the standard Crossover/Mutation pipeline. They receive the entire population and maintain their own internal state (if necessary) to manage topological innovations, historical markers, or complex distributional sampling (e.g., CMA-ES covariance updates). Emitters often bypass the standard `tree_leaves` mapping, interacting directly with the complex Genome structure to safely add or prune parameters.

## 3.5. Randomness and PRNG Management

Key budgeting is central to the operator design. Each concrete operator exposes a `num_keys` or `num_keys_per_atomic_operation` property. The `ResourceMapper` in the Engine tier reads this property and pre‑allocates a contiguous block of keys of the proper shape. Operators simply reshape and consume these pre-allocated blocks, ensuring exact determinism and GSPMD compatibility.

### 3.5.1. Injection and Regressions (base_injection.py)

Injection-mode operators consume a single PRNG key and accept externally generated noise/masks instead of budgeted subkeys. They trade memory for determinism and external control, making them ideal for regression testing (e.g., verifying that evolution produces exact trajectories using pre-specified noise arrays).
