# User Guide 2: Advanced Differential Evolution Tuning

Differential Evolution (Storn & Price, 1997) is one of the most robust metaheuristic global optimizers for non-differentiable, multimodal continuous landscapes. `malthus-optimize` brings native XLA scan acceleration to this algorithm.

---

## 1. Mathematical Mechanics

In each generation $g$, for each target vector $\mathbf{x}_{i, g}$ in population of size $N_p$:
1. **Mutation**: A mutant donor vector $\mathbf{v}_{i, g}$ is formed by adding scaled difference vectors:
   $$\mathbf{v}_{i, g} = \mathbf{x}_{r_1, g} + F \cdot (\mathbf{x}_{r_2, g} - \mathbf{x}_{r_3, g})$$
   where $F \in (0, 2]$ is the differential weight (`mutation` parameter).
2. **Crossover**: A trial vector $\mathbf{u}_{i, g}$ is generated using binomial or exponential crossover with probability $C_r$ (`recombination` parameter):
   $$u_{j, i, g} = \begin{cases} v_{j, i, g} & \text{if } \text{rand}_j(0, 1) \le C_r \text{ or } j = j_{\text{rand}} \\ x_{j, i, g} & \text{otherwise} \end{cases}$$
3. **Selection**: Greedy replacement if $f(\mathbf{u}_{i, g}) \le f(\mathbf{x}_{i, g})$.

---

## 2. Parameter Tuning Matrix

| Parameter | SciPy Default | Malthus Recommended | Effect & Best Practice |
| :--- | :--- | :--- | :--- |
| `popsize` | `15` | `15` - `50` | In SciPy, total population size is $N_p = \text{popsize} \times D$. In `malthus-optimize`, if you pass an explicit integer, $N_p$ is bounded cleanly. Increase for highly multimodal landscapes (e.g. Rastrigin, Ackley). |
| `maxiter` | `1000` | `50` - `500` | Number of evolutionary generations. Because all $N_p$ candidates are evaluated in parallel on GPU/TPU, you can run more generations in less wall-clock time. |
| `mutation` | `(0.5, 1)` | `(0.5, 1.0)` or `0.8` | Mutation scale factor $F$. Tuple enables dithering between lower and upper bounds across generations. |
| `recombination` | `0.7` | `0.7` - `0.9` | Crossover rate $C_r$. Higher values ($0.8 - 0.9$) accelerate convergence on separable functions; lower values ($0.3 - 0.5$) help with correlated dimensions. |
| `seed` | `None` | Integer (e.g. `42`) | PRNG seed for deterministic JAX PRNGKey splitting. |

---

## 3. High-Dimensional Code Example

```python
import jax.numpy as jnp
from malthus_optimize import differential_evolution

# 20-Dimensional Ackley Benchmark
def ackley_20d(x):
    d = len(x)
    sum_sq = jnp.sum(x**2)
    sum_cos = jnp.sum(jnp.cos(2.0 * jnp.pi * x))
    return -20.0 * jnp.exp(-0.2 * jnp.sqrt(sum_sq / d)) - jnp.exp(sum_cos / d) + 20.0 + jnp.e

bounds = [(-32.768, 32.768)] * 20

res = differential_evolution(
    func=ackley_20d,
    bounds=bounds,
    popsize=30,
    maxiter=250,
    mutation=0.8,
    recombination=0.9,
    seed=123,
)

print(f"Optimal Value: {res.fun:.6e}")
print(f"Iterations:    {res.nit}")
print(f"Total Evals:   {res.nfev}")
```

---

## 4. Boundary Enforcement

Search space bounds are strictly preserved using continuous clipping and normalization:
- Candidates generated outside $[0, 1]$ during mutation/crossover are projected back into the valid hypercube before fitness evaluation.
- No invalid solutions will ever be evaluated by your objective function.
