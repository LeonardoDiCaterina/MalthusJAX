# Level 1: The Core – Guide

## 📚 Overview
MalthusJAX is designed with a strict hierarchical architecture. **Level 1** represents the absolute foundation: `src/malthusjax/core/`.

If you decide to use *only* Level 1, you are opting out of the automated Resource Mapper, the Operator Registry, and the Engine's `lax.scan` loop. Instead, you are using MalthusJAX as a highly optimized, type-safe PyTree structuring library to manage your evolutionary state while you manually write the JAX training loops yourself.

This is highly recommended for researchers who want to prototype wild new algorithmic structures (e.g., crazy nested RL loops, custom meta-evolution) where the standard Genetic Engine feels too restrictive.

---

## 1️⃣ The Four Pillars of Level 1

At this level, you interact with four base abstractions:

### 1. `BaseGenome` and `BaseGenomeConfig`
Defines the mathematical structure of a single individual. By inheriting from `BaseGenome` (a `flax.struct.dataclass`), your genome is a valid JAX PyTree and participates in `jax.jit`, `jax.vmap`, and `jax.lax.scan` automatically.

### 2. `BasePopulation`
Holds a batch of genomes in **Struct-of-Arrays (SoA)** layout. You never deal with Python lists of genomes — the population holds a single `BaseGenome` where every internal array has a leading batch dimension `(pop_size, ...)`.

### 3. The Composable Evaluator (`Environment × Transform × Interpreter × Output`)
The new core paradigm for defining fitness. Instead of subclassing a single monolithic evaluator, you **compose** four independent axes:

| Axis | Role | Examples |
|---|---|---|
| **Environment** | The problem definition | `SphereEnv`, `TSPEnv`, `SklearnEnv`, `BraxEnv` |
| **Transform** | Genotype → phenotype mapping | `IdentityTransform`, `TensorNeatTransform` |
| **Interpreter** | How the genome produces outputs | `IdentityInterpreter`, `MLPInterpreter` |
| **Output** | Fitness aggregation & sign convention | `ScalarOutput`, `QDOutput`, `MOOutput` |

These plug into one of the three evaluator shells: `OptimizationEvaluator`, `SupervisedEvaluator`, or `RLEvaluator`.

> [!NOTE]
> For simple optimization tasks (minimizing/maximizing a function of real vectors), the stack collapses to just `SphereEnv + IdentityInterpreter + ScalarOutput` — three lines total. You only need to add more axes when your problem is more complex (e.g., using a neural network policy, or a QD descriptor).

### 4. `BaseEvaluatorConfig`
Holds cross-cutting configuration (e.g., `maximize: bool`). For most cases this is managed internally by `ScalarOutput`.

### 5. Unified Logging Subsystem (`malthusjax.core.logger`)
Level 1 includes a zero-dependency, hierarchical logging framework. You can control logging verbosity, enable ANSI-colored output, route logs to files, or configure `StepLoggingConfig` for your own custom `lax.scan` loops:

```python
from malthusjax.core import get_logger, configure_logging

configure_logging(level="INFO", format_type="color")
logger = get_logger("my_custom_loop")

logger.info("Starting manual evolutionary loop...")
```

---

## 2️⃣ The Workflow (Manual Orchestration)

When working strictly at Level 1, your workflow looks like this:

1. **Define your genome** — inherit from `BaseGenome` and `BaseGenomeConfig`.
2. **Compose your evaluator** — pick an `Environment`, pair it with an `Interpreter` and `ScalarOutput`, plug into `OptimizationEvaluator`.
3. **Initialize** — call `config.init_population(key, pop_size)` to get your starting `BasePopulation`.
4. **The Loop** — write a standard Python `for` loop or your own `jax.lax.scan`.
5. **Manual RNG** — you are fully responsible for calling `jax.random.split` to manage PRNG keys.
6. **Manual VMAP** — write your own mutation/crossover logic and explicitly wrap it in `jax.vmap`.

---

## 3️⃣ Pros and Cons of Level 1 Only

> [!TIP]
> **Pros**:
> - Absolute freedom. Write custom meta-learning loops, hybrid RL/Evolution loops, or completely non-standard algorithms without fighting the framework's Engine.
> - The composable evaluator stack is still fully JIT-compatible — you get the benefits of the new architecture without needing the Engine.
> - Full access to zero-dependency Level 1 logging (`malthusjax.core.logger`).
> - Built-in runtime stabilization and native crash diagnostics (`malthusjax.core.diagnostics`) protecting against OpenMP thread collisions and GPU preallocation crashes.
> - Perfect for rapid prototyping in Jupyter Notebooks.

> [!WARNING]
> **Cons**:
> - You lose the **Resource Mapper**. You must manually ensure you don't leak RNG keys or accidentally reuse them, which can cause silent, disastrous correlations in JAX.
> - You lose **Init-Phase Compilation**. If you aren't careful with your manual `vmap` and `jnp.where` logic, your code might suffer from massive dynamic shape recompilations.
> - You cannot use the TOML **Composer** to load pipelines dynamically.

Check out the accompanying script for a fully functional, self-contained example!

