# Level 1: The Core – Guide

## 📚 Overview
MalthusJAX is designed with a strict hierarchical architecture. **Level 1** represents the absolute foundation: `src/malthusjax/core/`.

If you decide to use *only* Level 1, you are opting out of the automated Resource Mapper, the Operator Registry, and the Engine's `lax.scan` loop. Instead, you are using MalthusJAX simply as a highly optimized, type-safe PyTree structuring library to manage your evolutionary state while you manually write the JAX training loops yourself.

This is highly recommended for researchers who want to prototype wild new algorithmic structures (e.g., crazy nested RL loops) where the standard Genetic Engine feels too restrictive.

---

## 1️⃣ The Three Pillars of Level 1

At this level, you only interact with three base classes:

### 1. `BaseGenome` and `BaseGenomeConfig`
You define the mathematical structure of a single individual. By inheriting from `BaseGenome` (which is a `flax.struct.dataclass`), you guarantee that your individual is a valid JAX PyTree.

### 2. `BasePopulation`
You define how a batch of genomes is stored. The `BasePopulation` inherently knows how to handle "Lifted Struct-of-Arrays". You never deal with lists of genomes; instead, the population holds a single `BaseGenome` where every internal array has a leading batch dimension `(pop_size, ...)`.

### 3. `BaseEvaluator`
You define how an individual (or population) is scored. `dispatch_evaluate_population` uses `jax.vmap` under the hood to automatically vectorize your single-individual evaluation logic across the entire population matrix.

---

## 2️⃣ The Workflow (Manual Orchestration)

When working strictly at Level 1, your workflow looks like this:

1. **Initialize**: Call `config.init_population(key, pop_size)` to get your starting `BasePopulation`.
2. **Evaluate**: Call `dispatch_evaluate_population` to calculate initial fitness.
3. **The Loop**: Write a standard Python `for` loop (or your own `jax.lax.scan`).
4. **Manual RNG**: You are fully responsible for calling `jax.random.split` to manage PRNG keys.
5. **Manual VMAP**: You write your own mutation/crossover logic and explicitly wrap it in `jax.vmap` to apply it to the population's gene arrays.

---

## 3️⃣ Pros and Cons of Level 1 Only

> [!TIP]
> **Pros**:
> - Absolute freedom. You can write custom meta-learning loops, hybrid RL/Evolution loops, or completely non-standard algorithms without fighting the framework's Engine.
> - Perfect for rapid prototyping in Jupyter Notebooks.

> [!WARNING]
> **Cons**:
> - You lose the **Resource Mapper**. You must manually ensure you don't leak RNG keys or accidentally reuse them, which can cause silent, disastrous correlations in JAX.
> - You lose **Init-Phase Compilation**. If you aren't careful with your manual `vmap` and `jnp.where` logic, your code might suffer from massive dynamic shape recompilations.
> - You cannot use the TOML **Composer** to load pipelines dynamically.

Check out the accompanying script for a fully functional, self-contained example!
