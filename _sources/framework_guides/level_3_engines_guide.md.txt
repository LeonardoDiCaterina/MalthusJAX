# Level 3: Engines & Resource Mapper – Guide

## 📚 Overview
Welcome to **Level 3** (`src/malthusjax/engine/`). If you've been following the Level 1 and Level 2 guides, you know that manually writing `jax.lax.scan` loops and budgeting `PRNGKeys` for operators is tedious and prone to silent errors.

Level 3 solves this completely by introducing the **Engine** and its internal **Resource Mapper**.

By using Level 1, 2, and 3 together, you are using the full programmatic power of MalthusJAX. You construct your custom Genomes, Evaluators, and Operators, and then hand them off to a pre-built Engine (like `GeneticEngine`). The Engine handles all compilation, key splitting, population merging, and history tracking automatically.

---

## 1️⃣ The Resource Mapper (Init-Phase Compilation)
When you call `engine.init_state()`, the Engine secretly calls the `ResourceMapper`. 
- The mapper queries all your Level 2 operators (e.g., `mutator.num_keys_per_atomic_operation`).
- It calculates exactly how many PRNG keys the entire generational step will need.
- It bakes these shapes directly into the XLA graph *before* evolution starts.

As a result, your operators are fed massive, statically sized arrays of PRNG keys every generation, completely eliminating dynamic shape recompilations.

---

## 2️⃣ The `GeneticEngine` Orchestrator
The `GeneticEngine` is the flagship orchestrator. It strictly enforces a 5-phase generation inside its `step()` method:
1. **Entropy Allocation**: The Resource Mapper slices the master key.
2. **Selection**: Parents are sampled.
3. **Reproduction**: Crossover and Mutation are applied (via your Level 2 operators).
4. **Merge**: The new offspring are merged with any elites.
5. **Evaluation**: Your Level 1 Evaluator is batched over the new population.

---

## 3️⃣ The Workflow (Level 1 + 2 + 3)

1. **Build Components**: Define your Genome, Evaluator, and Operators (Levels 1 & 2).
2. **Configure Engine**: Instantiate `GeneticEngineParams` (e.g., pop size, elitism, generations).
3. **Instantiate Engine**: Create the `GeneticEngine` passing in your components.
4. **Initialize State**: Call `state = engine.init_state(master_key)`.
5. **Run**: Call `final_state, history, _ = engine.run(state)`.

*That's it! You no longer write the `lax.scan` loop yourself.*

---

## 4️⃣ Zero-Overhead JIT Telemetry & NaN Watchdog

MalthusJAX engines support host-device telemetry through `StepLoggingConfig`:

```python
from malthusjax.core.logger import StepLoggingConfig, configure_logging

configure_logging(level="INFO")

# Enable step progress every 10 generations and non-finite watchdog
step_cfg = StepLoggingConfig(log_interval=10, log_nan_watchdog=True)

# Option A: Attach to engine params
params = GeneticEngineParams(pop_size=100, num_generations=100, step_logging=step_cfg)

# Option B: Pass directly to run()
final_state, history, _ = engine.run(state, step_logging=step_cfg)
```

- **Trace-Time Pruning**: If `step_logging=None`, zero callback instructions exist in the lowered StableHLO graph.
- **On-Device Anomaly Detection**: `jnp.isnan` / `jnp.isinf` evaluates on the accelerator, firing a `CRITICAL` log only when numerical instability occurs.

---

## 5️⃣ Pros and Cons of Level 3

> [!TIP]
> **Pros**:
> - **Zero Boilerplate**: You don't have to write tracking metrics, `lax.scan` unrolling, or RNG key management.
> - **Maximum Safety**: The Resource Mapper guarantees that keys are never reused, avoiding catastrophic correlations in evolution.
> - **Insane Speed**: The entire 5-phase loop is perfectly fused into a single XLA kernel.
> - **Native Observability**: Integrated device-to-host logging with zero overhead when disabled.

> [!WARNING]
> **Cons**:
> - **Less Flexibility**: You are forced into the 5-phase `(Selection -> Crossover -> Mutation -> Merge -> Evaluate)` pipeline. If your algorithm requires evaluating *before* merging, or mutating *twice* per generation, you have to write a custom Engine subclass.
> - **No TOML Yet**: You are still writing Python glue code to wire the components together (Level 4 Composer solves this).
