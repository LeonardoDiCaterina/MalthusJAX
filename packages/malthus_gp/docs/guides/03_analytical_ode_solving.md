# Guide 3: Closed-Form Analytical ODE Solving via AutoDiff

This guide explains how to discover analytical, closed-form solutions to Ordinary Differential Equations (ODEs) using **`CGPODESolver`** and JAX automatic differentiation (`jax.grad`).

---

## 1. The Physics-Informed Symbolic Formulation

Traditional numerical differential equation solvers (e.g. Runge-Kutta 4th order, Euler) produce discrete trajectory tables at specific time steps $\Delta t$. In contrast, **symbolic ODE solving** discovers an explicit algebraic function $\hat{y}(t)$ that satisfies the differential equation across continuous time.

Consider the general initial value problem:
$$\frac{dy}{dt} = f(t, y), \quad \text{with initial condition } y(t_0) = y_0$$

### The AutoDiff Residual Loss

Rather than relying on numerical finite-difference approximations ($\frac{y(t+\Delta t) - y(t)}{\Delta t}$), `malthus-gp` calculates the **exact analytical derivative** $\frac{d\hat{y}}{dt}$ at any query point $t$ using JAX's reverse-mode differentiation (`jax.grad`):

$$\left. \frac{d\hat{y}}{dt} \right|_t = \text{jax.grad}(\hat{y})(t)$$

The fitness function evaluates the mean squared residual error over a continuous time domain $T$, penalized by boundary condition violations:

$$\mathcal{L}(\hat{y}) = \frac{1}{|T|} \sum_{t \in T} \left( \left. \frac{d\hat{y}}{dt} \right|_t - f(t, \hat{y}(t)) \right)^2 + \lambda_{\text{bc}} \cdot \left( \hat{y}(t_0) - y_0 \right)^2$$

Where:
- The first term enforces the differential equation dynamics across $T$.
- The second term strictly anchors the solution to the initial boundary value $(t_0, y_0)$ with penalty weight $\lambda_{\text{bc}}$ (default `10.0`).

---

## 2. Practical Example: Exponential Decay

Solve the initial value problem:
$$\frac{dy}{dt} = -y, \quad y(0) = 1.0 \implies \text{Analytical Solution: } y(t) = e^{-t}$$

### Code Implementation

```python
import numpy as np
from malthus_gp.learn import CGPODESolver

# 1. Define the differential equation dy/dt = f(t, y)
def decay_ode(t, y):
    return -y

# 2. Configure the Symbolic ODE Solver
solver = CGPODESolver(
    ode_fn=decay_ode,
    initial_condition=(0.0, 1.0),            # y(0) = 1
    t_domain=np.linspace(0.0, 2.5, 50),     # Evaluation grid T
    n_cols=25,                               # Computational nodes
    pop_size=5,                              # (1 + 4) ES
    max_generations=400,
    bc_weight=10.0,                          # Initial condition penalty
    random_state=42,
)

# 3. Fit the symbolic solution
solver.fit()

# 4. Inspect the discovered closed-form solution
print("Discovered ODE Formula:")
print(solver.to_formula())

# 5. Evaluate accuracy against true analytical solution
t_test = np.linspace(0.0, 2.5, 100)
y_pred = solver.predict(t_test)
y_true = np.exp(-t_test)

max_error = np.max(np.abs(y_pred - y_true))
print(f"Max Absolute Error across [0, 2.5]: {max_error:.4e}")
```

---

## 3. Advanced Applications

1. **Stiff Differential Equations**: Standard Runge-Kutta solvers require extremely small step sizes for stiff systems. Symbolic discovery operates without discretization grids, avoiding stiffness instability.
2. **Coupled ODE Systems**: Multi-output Cartesian grids (`num_outputs > 1`) can simultaneously solve coupled dynamical systems (e.g. Lotka-Volterra predator-prey equations).
