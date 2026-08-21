# Evaluation Bridge (Kozax)

MalthusJAX's universal `@adapter` supports dual-mode fitness evaluation.

## 1. Native Framework Evaluation (`EvalMode.NATIVE`)
When initialized with `eval_mode=EvalMode.NATIVE`, the adapter uses native Kozax symbolic regression or control problems.

**Mechanism**:
The `@adapter` decorator utilizes the native translator:
```python
lambda gp, pop, data, key: gp.evaluate_population(pop, data, key)[0]
```
- `gp` is the `GeneticProgramming` instance injected during Engine initialization.
- `pop` is the flat population tensor representing trees.
- `data` represents offline training data (like X, Y points for symbolic regression).
- Kozax's `evaluate_population` returns a tuple of `(fitnesses, evaluated_pop)`. MalthusJAX extracts `[0]` to only retrieve the evaluated fitness array, passing it back into the step function.

## 2. MalthusJAX Evaluation (`EvalMode.MALTHUSJAX`)
When initialized with `eval_mode=EvalMode.MALTHUSJAX`, the adapter intercepts the raw population tensor and pipes it into a native MalthusJAX Evaluator instance.

**Mechanism**:
The `@adapter` utilizes the MJX translator:
```python
lambda mjx_eval, pop: mjx_eval.evaluate_batch(pop)
```
- `mjx_eval` is expected to decipher the Kozax tree encoding `(num_populations * population_size, num_trees, max_nodes, 4)` and evaluate its performance against MalthusJAX native environments.
- The evaluated fitnesses are injected back into the Kozax `evolve_population` phase.
