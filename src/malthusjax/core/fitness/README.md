# malthusjax.core.fitness — Technical Specification & Architecture ✅

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

## 4) Specialized Evaluator Implementations

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
  - Maintains internal type-safety and flips optimization direction as needed using `jax.lax.select`.
  - Example pattern: `fitness_scores, state, info = self.evosax_problem.eval(keys, X, self.evosax_state)` and then update population with `.replace(fitness=fitness_scores)`.

- **Linear GP & Symbiotic Selection** (`LinearGPEvaluator`)
  - Executes each program instruction as a candidate output and selects the best-performing instruction ("symbiotic selection").
  - Uses `jax.lax.switch` to implement opcode dispatch (stable under JIT) and `jax.lax.scan` to simulate program execution across instructions.
  - Stores training data as `RegressionData` and evaluates instructions across the dataset via batched predictions (`vmap` over X).

---

## 5) Data Management (RegressionData)

- `RegressionData` is a type alias: `RegressionData = Tuple[chex.Array, chex.Array]` representing `(X, y)`.
- `BaseEvaluator.data` may hold static data (training sets, environment parameters) which are accessible in `evaluate` and should be structured as PyTree-friendly objects.
- Example usage in `LinearGPEvaluator`: the evaluator stores `(X, y)` and uses `jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)` to produce predictions for every input row and then computes per-instruction MSE.
- For deterministic or static datasets, store them as `chex.Array` and mark them as PyTree nodes to be carried in JIT traces.

---

## 6) Implementation Best Practices — Developer's Checklist ✅

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

## Quick Examples

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

- Regression data usage in GP

```py
from malthusjax.core.fitness.linear_gp_evaluator import LinearGPEvaluator, RegressionData

X = jax.random.normal(jax.random.PRNGKey(0), (100, 5))
y = jnp.sin(jnp.sum(X, axis=1))
reg_data: RegressionData = (X, y)

config = LinearGPEvaluatorConfig(num_inputs=5, length=32, maximize=False)
eval = LinearGPEvaluator(config=config, data=reg_data)

# vectorized over population
pop = eval.evaluate_population(population)
```

---

## Final Notes

- The package emphasizes single-responsibility evaluator implementations (single-individual math) combined with the SoA structure and `vmap` for efficient batched execution.
- Following the developer checklist will keep new evaluators robust, JIT-friendly, and consistent with engine semantics (maximize/minimize).
