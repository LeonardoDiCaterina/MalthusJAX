# Deep Dive: GeneticEngine in MalthusJAX

> [!TIP]
> **Who is this for?** Developers who want to understand the inner workings of MalthusJAX's flagship engine, particularly those who need to distribute evaluations across a cluster using the `ask()` and `tell()` loop.

The `GeneticEngine` (`src/malthusjax/engine/genetic_fastengine.py`) is the flagship, highly-optimized standard genetic algorithm engine in MalthusJAX. 

It is designed to run in two distinct modes: **Synchronous** (the entire loop is JIT-compiled) and **Asynchronous** (you evaluate individuals in Python or across a cluster, then feed fitness back into JAX).

---

## 1. The Synchronous Monolith (`step` and `run`)

The simplest way to use MalthusJAX is to let XLA handle the entire generation loop. This requires your fitness evaluator to be fully JAX-compatible.

### `init_state`
Before any evolution starts, `init_state(rng_key)` is called. It acts as the "Init-Phase Compiler." 
1. It queries the operators for their shape outputs.
2. It calls the `ResourceMapper` to compute the exact static PRNG key budget for the entire generation.
3. It initializes the population and returns a strictly typed `GeneticEvolutionState`.

### `step`
The `step(state)` method executes exactly one generation. It strictly orchestrates the 5 phases:
1. **Entropy Allocation**: Slices the master key.
2. **Selection**: Samples parents using the selection operator.
3. **Reproduction**: Applies crossover and mutation.
4. **Merge**: Replaces the old population with elites and new offspring.
5. **Evaluation**: Calls the JIT-compiled evaluator over the entire new population matrix.

### `run`
The `run()` method simply wraps `step()` inside a `jax.lax.scan`. XLA unrolls this scan into a single massive GPU kernel, running thousands of generations without ever returning to the Python interpreter.

---

## 2. The Asynchronous Interface (`ask` and `tell`)

Sometimes, your evaluation function **cannot** be JIT-compiled (e.g., you are querying a REST API, running a Unity physics simulation, or distributing gym environment rollouts across a CPU cluster). 

For this, `GeneticEngine` provides the `ask()` and `tell()` interface.

### The Contract
The `ask` and `tell` loop splits the standard `step()` function exactly at Phase 5 (Evaluation).

> [!IMPORTANT]
> **The Entropy Contract:** When you call `ask(state)`, it consumes PRNG keys to generate the new population. Because `ask()` is detached from `tell()`, it must temporarily store these consumed keys to maintain deterministic consistency.
> 
> Therefore, `ask()` returns a *new, slightly modified engine instance* alongside the population. **You MUST use this returned engine to call `tell()`**. Do not use the original engine.

### Example: Single Batch Evaluation
```python
# 1. Ask the engine for a new population to evaluate
engine_for_tell, population = engine.ask(state)

# 2. Evaluate the population externally (e.g., in standard Python)
fitness_scores = run_complex_cpu_simulation(population) 

# 3. Inject the fitness scores back into the population
evaluated_pop = population.replace(fitness=fitness_scores)

# 4. Tell the specific engine instance the results
new_state = engine_for_tell.tell(state, evaluated_pop)
```

### Example: Massive Cluster Distribution
If you have a massive population and 4 separate compute nodes, you can `ask` for multiple batches in parallel, distribute them, and aggregate before calling `tell`.

```python
batches = 4
engines = [None] * batches
populations = [None] * batches

# Generate 4 distinct batches using the same starting state
for i in range(batches):
    engines[i], populations[i] = engine.ask(state)

# Distribute populations[i] to different GPUs/processes
# e.g., results = ray.map(evaluate_fn, populations)
results = parallel_map(evaluate_fn, populations)

# Aggregate results into the first population
aggregated_pop = populations[0].replace(fitness=results_aggregated)

# Tell the first engine the aggregated results
new_state = engines[0].tell(state, aggregated_pop)
```

---

## 3. Tracing and XLA HLO Profiling

By default, the 5 phases of `step()` are invisible to XLA profilers because XLA fuses them together into one giant block for maximum performance.

However, if you are debugging a custom operator and want to see exactly how much time Selection takes vs. Mutation in the TensorBoard profiler, you can use the `@traceable` decorator system.

At the top of your script, call:
```python
from malthusjax.engine.genetic_fastengine import enable_tracing

enable_tracing()
```
This forces JAX to wrap the internal engine phases in `jax.named_call`, which creates explicit boundaries in the HLO graph so you can profile them individually!
