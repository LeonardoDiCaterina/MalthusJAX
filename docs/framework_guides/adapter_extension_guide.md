# Composer Adapters in MalthusJAX## 📚 Overview
- In MalthusJAX, the true **Adapters** (`src/malthusjax/composer/*_adapter.py`) are strictly used for **Ablation and Benchmarking**. 
- They allow you to "borrow" entire evolutionary objects and strategies from external libraries (like `evosax`, `qdax`, `TensorNEAT`, or `Kozax`), and wrap them so they perfectly implement the `BenchmarkRunner.Engine` protocol.
- This ensures that when you run a benchmark comparing MalthusJAX to other frameworks, the `lax.scan` compilation loop, Wall-Clock Timing, and Metric Tracking are standardized and perfectly unbiased.

---

## 1️⃣ The Universal Adapter Core
Found in `src/malthusjax/composer/adapters/base.py`, this module provides the `UniversalAdapterEngine` and the magical `@adapter` decorator.

It standardizes:
- **JIT Compilation**: Wrapping the external framework's initialization and step functions in a standardized `jax.lax.scan`.
- **Timing Methodology**: Measuring `t_warmup_start`, `t_exec_start`, and `t_exec_end` consistently using JAX's `block_until_ready()`.
- **Fitness Evaluation Mapping**: Decoupling the evaluation loop so external engines can be tested against native MalthusJAX evaluators, or vice versa.

---

## 2️⃣ External Framework Adapters

### Evosax (`evosax_adapter.py`)
- Adapts `evosax`'s `population_based_algorithms` and `distribution_based_algorithms` (like CMA-ES).
- Provides dual evaluation modes:
  - `NATIVE`: Uses the `evosax_problem` directly.
  - `MALTHUSJAX`: Dynamically wraps the raw `evosax` NumPy arrays into `RealGenome` PyTrees, passing them through MalthusJAX's batched evaluators.

### QDAX (`qdax_adapter.py`)
- Wraps `qdax` Quality-Diversity engines (like their native MAP-Elites).
- Standardizes the metric output (e.g., extracting `qd_score` and `coverage`) so it can be plotted directly alongside MalthusJAX's native `MapElitesEngine`.

### TensorNEAT (`tensorneat_adapter.py`) & Kozax (`kozax_adapter.py`)
- Adapts specialized neuroevolution and genetic programming frameworks to conform to the standard `run_once(key)` contract, ensuring fair speed and fitness comparisons against MalthusJAX's native equivalents.

---

## 3️⃣ Internal Engine Adapters
Even MalthusJAX's own specialized engines need to be adapted for the benchmarking protocol!

### Multi-Objective Factory (`mo_factory.py`)
- Wraps the native `MOEngine` via the `MOEngineAdapter`.
- Since MO engines return complex Pareto fronts and crowding distances, this adapter standardizes how the final metrics (like `num_pareto_optimal` or `max_crowding_distance`) are extracted and serialized for the `BenchmarkRunner`.

```python
from malthusjax.composer.adapters import adapter

@adapter(
    framework="my_lib",
    state_mapping={"init": "my_init", "step": "my_step"}
)
class MyLibraryEngineAdapter:
    pass
```

## 4️⃣ Building Your Own Adapter
If you want to benchmark a completely new external library against MalthusJAX:
1. Define the framework's native metrics via `MetricSpec`.
2. Write a native evaluation wrapper and a MalthusJAX evaluation wrapper (to handle PyTree vs Array conversions).
3. Decorate a class with `@adapter(framework="my_lib", state_mapping={"init": "my_init", "step": "my_step"})`.
4. The `@adapter` generates a `UniversalAdapterEngine` that you can immediately plug into the `BenchmarkRunner`!
