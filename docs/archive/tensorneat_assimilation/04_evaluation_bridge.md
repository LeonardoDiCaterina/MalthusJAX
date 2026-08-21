# Evaluation Bridge (TensorNEAT)

MalthusJAX's universal `@adapter` supports dual-mode fitness evaluation.

## 1. Native Framework Evaluation (`EvalMode.NATIVE`)
When initialized with `eval_mode=EvalMode.NATIVE`, the adapter uses native TensorNEAT problem environments.

**Mechanism**:
The `@adapter` decorator utilizes the native translator:
```python
lambda problem, state, pop_transformed, keys, forward_fn: (
    jax.vmap(problem.evaluate, in_axes=(None, 0, None, 0))(state, keys, forward_fn, pop_transformed)
)
```
- `problem` refers to a native TensorNEAT problem instance.
- `pop_transformed` is the output of `algorithm.transform()`.
- `forward_fn` is `algorithm.forward()`, an inference function that accepts the transformed topology to compute network activations.
- We rely on `jax.vmap` because TensorNEAT expects the user to handle parallelization over the batch.

## 2. MalthusJAX Evaluation (`EvalMode.MALTHUSJAX`)
When initialized with `eval_mode=EvalMode.MALTHUSJAX`, the adapter intercepts the `transformed` network parameters and pipes them into a native MalthusJAX Evaluator instance.

**Mechanism**:
The `@adapter` utilizes the MJX translator:
```python
lambda mjx_eval, pop_transformed: mjx_eval.evaluate_batch(pop_transformed)
```
- `mjx_eval` evaluates the neural network parameters directly against a MalthusJAX environment.
- The evaluated fitnesses (Array) are fed back into the TensorNEAT `tell` phase.
