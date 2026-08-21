# Evaluation Bridge (Evosax)

MalthusJAX's universal `@adapter` supports dual-mode fitness evaluation. This ensures that Evosax can be benchmarked purely natively without interference, or fully integrated into the MalthusJAX ecosystem.

## 1. Native Framework Evaluation (`EvalMode.NATIVE`)
When initialized with `eval_mode=EvalMode.NATIVE`, the adapter uses the native Evosax problem formulation (e.g. `BBOB_Fitness` or any other native Evosax task).

**Mechanism**:
The `@adapter` decorator utilizes the native translator:
```python
lambda problem, state, pop_raw, keys, forward_fn: problem.evaluate(keys, pop_raw)
```
- `problem` refers to an Evosax problem class instance injected during the Engine initialization.
- The `keys` argument provides PRNG sequences to support stochastic tasks.
- The output strictly respects Evosax's format (a flat JAX array of fitnesses).

## 2. MalthusJAX Evaluation (`EvalMode.MALTHUSJAX`)
When initialized with `eval_mode=EvalMode.MALTHUSJAX`, the adapter intercepts the raw genotypes output by the Evosax `ask` phase and pipes them into a native MalthusJAX Evaluator instance.

**Mechanism**:
The `@adapter` utilizes the MJX translator:
```python
lambda mjx_eval, pop_raw: mjx_eval.evaluate_batch(pop_raw)
```
- `mjx_eval` refers to a `BaseEvaluator` subclass conforming to the MalthusJAX ecosystem.
- `evaluate_batch` encapsulates the logic to construct the necessary PyTrees (e.g. `RealGenome`) required by the MalthusJAX problem.
- The adapter unpacks the evaluated fitnesses and feeds them directly back into the Evosax `tell` phase.
