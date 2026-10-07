# User Guide 1: Zero-Code-Change SciPy Drop-In Migration

`malthus-optimize` is engineered to be an immediate, drop-in replacement for the most widely used global optimization and curve fitting functions in `scipy.optimize`.

---

## 1. The Migration Pattern

In legacy scientific workflows, global optimization with SciPy is structured like this:

```python
# Before: SciPy CPU-bound optimization
from scipy.optimize import differential_evolution, minimize, curve_fit

res = differential_evolution(objective, bounds=[(-5.0, 5.0), (-5.0, 5.0)], seed=42)
print(res.x, res.fun)
```

To migrate to `malthus-optimize`, change only the import statement:

```python
# After: Malthus-Optimize JAX-accelerated optimization
from malthus_optimize import differential_evolution, minimize, curve_fit

res = differential_evolution(objective, bounds=[(-5.0, 5.0), (-5.0, 5.0)], seed=42)
print(res.x, res.fun)
```

---

## 2. API Parity Checklist

| SciPy Symbol | Malthus-Optimize Equivalent | Supported Arguments | Behavior |
| :--- | :--- | :--- | :--- |
| `differential_evolution` | `malthus_optimize.differential_evolution` | `func`, `bounds`, `args`, `strategy`, `maxiter`, `popsize`, `mutation`, `recombination`, `seed`, `tol`, `atol`, `callback` | JIT-compiled Differential Evolution via Composer. |
| `minimize` | `malthus_optimize.minimize` | `fun`, `x0`, `args`, `method`, `bounds`, `options` | Dispatches to DE, CMA-ES, PSO, GA, or OpenES. |
| `curve_fit` | `malthus_optimize.curve_fit` | `f`, `xdata`, `ydata`, `p0`, `sigma`, `absolute_sigma`, `bounds`, `method`, `max_generations`, `pop_size`, `seed` | Weighted non-linear regression with full covariance $P_{\text{cov}}$ computation. |
| `OptimizeResult` | `malthus_optimize.OptimizeResult` | Attributes (`.x`, `.fun`, `.nfev`, `.nit`, `.success`, `.message`) and dict indexing (`res['x']`). | 100% SciPy structure parity. |

---

## 3. Pure JAX vs. NumPy Objectives

`malthus-optimize` automatically detects whether your objective function uses pure JAX or standard NumPy:

### Scenario A: Pure JAX Objective (Maximum Throughput)
```python
import jax.numpy as jnp
from malthus_optimize import differential_evolution

def rastrigin(x):
    return 10.0 * len(x) + jnp.sum(x**2 - 10.0 * jnp.cos(2.0 * jnp.pi * x))

# Completely vectorized and executed inside the GPU/TPU kernel
res = differential_evolution(rastrigin, bounds=[(-5.12, 5.12)] * 10, maxiter=200, seed=42)
```

### Scenario B: Standard NumPy Objective (Transparent Fallback)
```python
import numpy as np
from malthus_optimize import differential_evolution

def legacy_blackbox(x):
    # Calls external routines, NumPy calculations, or legacy C extensions
    return np.sum(np.sin(x) * np.exp(x / 10.0))

# Runs transparently via jax.pure_callback without tracer leakage
res = differential_evolution(legacy_blackbox, bounds=[(0.0, 10.0)] * 5, maxiter=100, seed=42)
```

---

## 4. Returned Result Contract

The returned object is an instance of `malthus_optimize.OptimizeResult`. You can access fields both as properties and as dictionary keys:

```python
res = differential_evolution(objective, bounds=[(-5, 5), (-5, 5)])

# Attribute access
best_parameters = res.x
minimum_value   = res.fun
eval_count      = res.nfev
generation_count= res.nit
converged       = res.success
status_msg      = res.message

# Dictionary access (for legacy codebases)
assert res["x"] is res.x
assert res["fun"] == res.fun
```
