# Level 2: Operators & Emitters – Guide

## 📚 Overview
If Level 1 is about defining *what* a Genome is, **Level 2** (`src/malthusjax/operators/`) is about defining *how* it changes. 

By stepping up to Level 2, you stop writing your own manual `jax.vmap` loops for genetic variations. Instead, you subclass MalthusJAX's highly structured operator base classes (`BaseMutation`, `BaseCrossover`, `BaseSelection`, `AtomicEmitter`). 

Using Level 1 + Level 2 means you are still writing your own `jax.lax.scan` evolutionary loop (opting out of the Engine and Resource Mapper), but you are taking advantage of MalthusJAX's robust 3-Tier Architecture for generating noise, applying math, and batched orchestration.

---

## 1️⃣ The 3-Tier Operator Architecture

When you build an operator at Level 2, you are forced to adhere to a strict 3-Tier separation of concerns. This guarantees maximum XLA efficiency and prevents subtle RNG bugs.

### Tier 1: Pure Math (`_mutate_one` / `_recombine_one`)
This is the absolute lowest level. You write a purely mathematical function that takes a single unbatched `BaseGenome` and a tuple of pre-generated noise arrays, and outputs a modified `BaseGenome`. There are **no random keys** here.

### Tier 2: Pure Randomness (`_generate_noise`)
This tier is exclusively responsible for RNG. It receives an array of perfectly sized PRNG keys and generates a tuple of noise tensors (e.g., Gaussian noise, crossover masks, index pointers). It never touches the genome.

### Tier 3: Batched Orchestration (`__call__`)
You almost never write this! The `BaseMutation` and `BaseCrossover` classes implement `__call__` for you. When you call an operator, Tier 3 automatically uses `jax.vmap` to combine your Tier 2 noise generation with your Tier 1 mathematical application, mapping it seamlessly across the entire `BasePopulation`.

---

## 2️⃣ The Workflow (Level 1 + Level 2)

If you are writing your own custom training loop (ignoring the Engine/ResourceMapper), you interact with Level 2 like this:

1. **Instantiate the Operator**: e.g., `mutator = GaussianMutation(mutation_rate=0.1)`.
2. **Query the PRNG Budget**: Ask the operator how many keys it needs per individual via `mutator.num_keys(pop_size)`.
3. **Split Keys**: Manually split your master key to satisfy this budget.
4. **Execute (Tier 3)**: Call the operator `mutated_pop = mutator(keys, population, config)`. 

---

## 3️⃣ Pros and Cons of Level 1 + Level 2

> [!TIP]
> **Pros**:
> - **Incredible Reusability**: You can swap operators instantly because they all share the exact same `__call__` signature.
> - **Performance**: The 3-Tier separation ensures your operators fuse perfectly in XLA without dynamic shape recompilations.
> - **Diagnostic Logging**: Operators emit debug messages under `"malthusjax.operators"` when locking population length or budgeting keys.
> - **Still Flexible**: You still have complete freedom to write wild custom loops (like co-evolution or RL).

> [!WARNING]
> **Cons**:
> - **Manual RNG Budgeting**: Because you are not using the Engine's `ResourceMapper`, you still have to manually query `operator.num_keys(pop_size)` and meticulously split your PRNG keys before passing them into the operators. If you get the shapes wrong, JAX will crash!

