# malthusjax.core.fitness — Technical Specification & Architecture

This document describes how MalthusJAX handles candidate evaluation in a JAX-native way. It targets developers who will implement new evaluators, adapt external benchmarks, or integrate domain-specific data into fitness pipelines.

---

## 1) The Abstract Evaluator Paradigm

### Core idea
- The package exposes an abstract `BaseEvaluator[G, C, D]` that separates a single-individual evaluation from population-level execution.
- `G` is a `BaseGenome` subtype, `C` is a `BaseEvaluatorConfig` subtype, and `D` is an arbitrary data payload (e.g., `RegressionData`).

### Functional split
- `evaluate(self, genome: G) -> chex.Numeric` — implement the per-individual math here. It receives a single (non-batched) genome instance.
- `evaluate_population(self, population: BasePopulation[G]) -> BasePopulation[G]` — implemented in `BaseEvaluator` and uses `jax.vmap(self.evaluate)` to run the per-individual function across the leading axis of the population `genes` PyTree.

Why this design:
- Developers only write the scalar/vector math for one genome and get a highly-optimized, batched evaluation for free via `vmap`.
- This keeps implementations small, testable, and JIT-friendly.

---

## 2) Type-Safe Configuration

`BaseEvaluatorConfig` is a minimal Flax dataclass used by all evaluators. Important points:

- **maximize: bool** is mandatory and explicit in every evaluator config.
  - This single boolean makes the optimization direction explicit (Maximization vs Minimization).
  - Engines and selection operators rely on this to compare and sort candidates consistently.

Table — common config fields

| Field | Type | Description |
|-------|------|-------------|
| `maximize` | bool | If True, higher fitness is better. Must be set by the problem author. |
| `seed`, `fn_name`, `target`, ... | domain-dependent | Other fields should be annotated with `pytree_node=False` if they are static/config-only.

---

## 3) Handling Optimization Direction (Avoiding Python control flow)

- Inside `evaluate` implementations prefer `jax.lax.select(condition, true_val, false_val)` to flip the sign or choose between outcomes depending on `config.maximize`.
- Do **not** use Python `if/else` on `config.maximize` inside JIT-traced functions — that would create separate XLA graphs and can cause recompilation or runtime errors.

Example:

```py
# prefer this inside evaluate()
val = compute_objective(...)
return jax.lax.select(self.config.maximize, val, -val)

# avoid this pattern inside a JIT-traced function
if self.config.maximize:
    return val
else:
    return -val
```

Reason: `jax.lax.select` composes into the single XLA graph, keeping JIT compilation stable and the runtime efficient.

---

## 4) Static Configuration and Data Storage

All evaluators follow a consistent pattern for handling configuration and problem-specific data:

- **Config field (`C`)**: Stored as a `@struct.dataclass` with `pytree_node=False` to mark it as static. This prevents JAX from treating config as trainable state and avoids embedding it repeatedly into JIT traces.
- **Data field (`D`)**: Stored with `pytree_node=False` annotation. This ensures large or immutable problem data (training sets, distance matrices, lookup tables) remains static across `vmap` and `jit` operations.

Example pattern:
```py
@struct.dataclass
class MyEvaluatorConfig(BaseEvaluatorConfig):
    param1: int = struct.field(pytree_node=False)
    param2: str = struct.field(pytree_node=False)

@struct.dataclass
class MyEvaluator(BaseEvaluator[G, MyEvaluatorConfig, D]):
    config: MyEvaluatorConfig
    data: D = struct.field(pytree_node=False)  # Mark large data as static
```

This pattern ensures:
- Configuration changes don't require recompilation of JIT code
- Large datasets aren't copied unnecessarily during vectorization
- Evaluators remain pure functions from the JAX perspective

---

## 5) Specialized Evaluator Implementations

The repository includes several evaluator categories with canonical implementations:

- **Analytical Evaluators** (e.g., `SphereEvaluator`, `GriewankEvaluator`)
  - Operate directly on `RealGenome.values` and implement classic continuous test functions.
  - Return a scalar `chex.Numeric` and use `jax.lax.select` to apply `maximize` consistently.

- **Combinatorial Evaluators** (e.g., `KnapsackEvaluator`)
  - Work on discrete genomes like `BinaryGenome`.
  - Use vectorized linear algebra (dot products) and linear penalty terms for constraint violations:
    - Penalty is constructed using differentiable-friendly ops such as `jnp.maximum(0, excess)` so the evaluator remains JIT-friendly.

- **BBOB Adapter** (`BBOBEvaluator`) — wrapping external packages (evosax)
  - Uses evosax `BBOBProblem` under the hood and leverages its high-performance (batch) API for population evaluation.
  - Handles sign convention internally: MalthusJAX engines always maximize, so for `maximize=False` problems, the evaluator negates the raw evosax score (turning minimization into maximization of the negated objective).
  - Stores problem instance and state as static (`pytree_node=False`) non-PyTree fields to avoid recompilation.
  - Example pattern: `fitness_scores, state, info = self.evosax_problem.eval(keys, X, self.evosax_state)` and then update population with `.replace(fitness=fitness_scores)`.

- **TSP (Traveling Salesman Problem)** (`TSPEvaluator`)
  - Solves the classic TSP using a distance matrix and a real-valued genome encoding (Random Key permutation).
  - **Encoding**: Uses `RealGenome` values where `argsort(genome.values)` produces a city tour permutation. This "random key" encoding avoids explicit permutation genomes while leveraging continuous operators.
  - **Distance matrix**: Stored as evaluator data; can be generated synthetically `create_synthetic(num_cities, seed)` or loaded from file `create_from_data(config, distance_matrix)`.
  - **Computation**: Computes total tour distance via `distance_matrix[tour, roll(tour)]` and returns distance (for minimization) or negated distance (for maximization).

---

## 6) Data Management

- Static problem data such as training sets, distance matrices, and lookup tables should be stored in the `data` field of your evaluator and marked with `pytree_node=False`.
- `RegressionData` is a type alias: `RegressionData = Tuple[chex.Array, chex.Array]` representing `(X, y)`.
- For deterministic or static datasets, store them as `chex.Array` objects to be carried efficiently in JIT traces.
- Example: `TSPEvaluator` stores a distance matrix; `BBOBEvaluator` stores the evosax problem state.

---

## 7) Implementation Best Practices — Developer's Checklist

When adding a new evaluator, follow this checklist to ensure compatibility with MalthusJAX engines and JAX tracing:

1. Use `chex.Numeric` as the return type for `evaluate` to avoid tracer-vs-float conflicts.
2. Make evaluators pure and free of side effects. Avoid random draws inside `evaluate` unless you accept non-determinism; prefer passing RNGs explicitly where needed.
3. Use `jax.lax.select` for decisions depending on `config.maximize` so the code is compatible with `jit` and produces a single XLA graph.
4. Handle numerical stability explicitly (examples):
   - Protected division: `jnp.where(jnp.abs(y) < eps, 0.0, x / y)`
   - Log-domain math: `jnp.where(x > 0, jnp.log(x), fallback)`
   - Use `jnp.nan_to_num` when operations (e.g., division, trigonometric functions) can produce NaNs/inf.
5. Type-safe population updates: When returning a new population with updated `fitness`, use the Flax `.replace` pattern while satisfying MyPy and Flax typing:

```py
from typing import Any, cast

return cast(BasePopulation[G], cast(Any, population).replace(fitness=fitness_scores))
```

6. If you need to dispatch among many small functions (e.g., tensor GP op-codes), prefer `jax.lax.switch` to keep control flow inside the XLA graph.
7. Mark truly static config values with `pytree_node=False` to avoid embedding large arrays into the JIT trace as mutable nodes.
8. Add a factory method for complex third-party adapters (e.g., `BBOBEvaluator.create(config)`) so initialization (problem state, rotations, seeds) is explicit and separable from JIT-time evaluation.

---

## 8) Quick Examples

- Vectorized evaluation using an existing evaluator:

```py
# given: evaluator: BaseEvaluator and population: BasePopulation[G]
# vectorized (and JIT-able) evaluation is a single call
new_pop = evaluator.evaluate_population(population)
# new_pop.fitness now contains a batched fitness array
```

- Creating and evaluating a knapsack problem

```py
from malthusjax.core.fitness.binary_evaluators import KnapsackEvaluator
from malthusjax.core.fitness.binary_evaluators import KnapsackConfig

# create problem
config = KnapsackEvaluator.create_random_problem(jax.random.PRNGKey(0), n_items=20)
knapsack = KnapsackEvaluator(config=config)

# evaluate population
pop = ...  # some BasePopulation[BinaryGenome]
pop = knapsack.evaluate_population(pop)
```

- Creating and evaluating a TSP instance

```py
from malthusjax.core.fitness.tsp_evaluator import TSPEvaluator

# create synthetic TSP with 50 cities
tsp = TSPEvaluator.create_synthetic(num_cities=50, seed=42)

# evaluate population
pop = ...  # some BasePopulation[RealGenome] with 50 dimensions
pop = tsp.evaluate_population(pop)
```

---

## 9) Final Notes

- The package emphasizes single-responsibility evaluator implementations (single-individual math) combined with the SoA structure and `vmap` for efficient batched execution.
- Following the developer checklist will keep new evaluators robust, JIT-friendly, and consistent with engine semantics (maximize/minimize).

