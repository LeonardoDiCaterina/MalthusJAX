# Evaluators: MalthusJAX Implementation Details

This document details exactly how MalthusJAX Evaluators extend the core `BaseEvaluator` and `StochasticEvaluator` generics to translate the theoretical properties of Genetic Programming (CGP, MEP, dCGP) into highly optimized JAX execution pipelines.

## The Extension Pattern
Every evaluator in MalthusJAX extends the `BaseEvaluator[G, C, D]` generic (or one of its stochastic/multi-objective variants). The generic parameters strictly define:
- **`G`**: The Genome type (e.g., `CartesianGenome`).
- **`C`**: The Configuration type.
- **`D`**: Auxiliary return data.

To implement an evaluator, the class must define `predict_one(self, genome, x_input)` to handle the forward pass of a single graph, and `evaluate(self, genome, rng)` to handle batched data processing and loss computation.

---

## 1. `CartesianGPEvaluator` (Discrete CGP)

**What it extends:**
```python
class CartesianGPEvaluator(StochasticEvaluator[CartesianGenome, CartesianGPEvaluatorConfig, RegressionData]):
```

**How it achieves the CGP Goal:**
It fulfills its requirement by implementing `predict_one(self, genome, x_input)`.
1. **The Forward Pass (DAG Execution):** It creates a `jnp.zeros` memory buffer and injects `x_input` at the start. It defines an internal `step` function that reads `genome.ops` and `genome.args`, pulls from memory via `jnp.take`, and routes the math via `jax.lax.switch`. It wraps this step in `jax.lax.scan` to compile the entire graph into one GPU kernel.
2. **The Output Readout (Neutral Drift):** After the scan finishes, it explicitly enforces CGP's output routing by returning only the nodes designated by the integer array:
   ```python
   outputs = jnp.take(final_mem, genome.out_nodes)
   return outputs
   ```
3. **The Loss:** In its `evaluate()` function, it uses `jax.vmap(self.predict_one)` to run the graph across all data points, and computes the loss strictly on those output nodes, ensuring disconnected nodes accumulate **Neutral Drift** (silent mutations).

---

## 2. `LinearGPEvaluator` (Discrete MEP)

**What it extends:**
```python
class LinearGPEvaluator(StochasticEvaluator[LinearGenome, LinearGPEvaluatorConfig, RegressionData]):
```

**How it achieves the MEP Goal:**
It uses the exact same `jax.lax.scan` forward pass as CGP, but the readout and evaluation completely diverge to implement SIP (Strong Implicit Parallelism).
1. **The Readout:** Instead of slicing `out_nodes`, the `predict_one` function returns the *entire* memory buffer (the outputs of every single instruction).
2. **Dynamic Output Routing:** In the `evaluate()` function, it maps the loss function across *all* returned nodes simultaneously:
   ```python
   # final_memory shape: [num_nodes, batch_size]
   all_node_losses = jax.vmap(compute_mse, in_axes=(0, None))(final_memory, targets)
   ```
3. **Best Gene Wins:** It dynamically routes the fitness to the best-performing node:
   ```python
   return jnp.min(all_node_losses)
   ```

---

## 3. Custom Differentiable Evaluators (Pure dCGP & ODE Solvers)

To implement "Pure dCGP" (or even use dCGPANN to solve differential equations), we must extract the exact analytical derivative of the graph output with respect to its inputs ($\frac{\partial y}{\partial x}$). 

In MalthusJAX, this requires zero C++ overhead or custom dual-number algebra. Because the base evaluators are written in pure JAX `lax` primitives, the entire execution graph is already natively differentiable.

To solve an ODE (e.g., $y' = -y$), you simply define a custom evaluator that wraps the forward pass in `jax.grad`:
```python
# 1. Define the scalar forward pass (e.g., using a CartesianGenome)
def predict_scalar(t):
    final_mem, _ = jax.lax.scan(eval_node, memory, jnp.arange(N, N + num_nodes))
    return final_mem[genome.out_nodes][0]

# 2. Extract the analytical derivative using JAX AutoDiff!
grad_predict = jax.grad(predict_scalar)

# 3. Compute the PDE residual fitness natively
def compute_residual(t):
    y_pred = predict_scalar(t)
    dy_dt = grad_predict(t)
    
    # Mathematical ODE residual: dy/dt + y = 0
    return (dy_dt + y_pred)**2

# Map across the time domain T
mean_res = jnp.mean(jax.vmap(compute_residual)(T))
```
By simply applying `jax.grad`, a standard symbolic regression evaluator is instantly transformed into a Physics-Informed Differential Equation solver.
