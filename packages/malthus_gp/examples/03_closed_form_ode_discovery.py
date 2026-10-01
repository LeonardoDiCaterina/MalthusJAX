"""03_closed_form_ode_discovery.py.

Demonstrates closed-form analytical ordinary differential equation (ODE) discovery
using CGPODESolver and exact JAX AutoDiff derivatives (jax.grad).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from malthus_gp.learn import CGPODESolver


def run_ode_showcase(output_plot: Path) -> None:
    print("=" * 70)
    print("  Malthus-GP Showcase 3: Closed-Form Analytical ODE Discovery")
    print("=" * 70)

    # Initial Value Problem: dy/dt = -y, y(0) = 1.0 -> True Solution: y(t) = exp(-t)
    def decay_ode(t, y):
        return -y

    print("\nTarget Differential Equation: dy/dt = -y")
    print("Initial Boundary Condition:   y(0.0) = 1.0")
    print("Exact Analytical Solution:    y(t) = exp(-t)")

    t_domain = np.linspace(0.0, 2.5, 50)
    y_analytical = np.exp(-t_domain)

    t0 = time.time()
    solver = CGPODESolver(
        ode_fn=decay_ode,
        initial_condition=(0.0, 1.0),
        t_domain=t_domain,
        n_cols=25,
        pop_size=5,
        max_generations=400,
        random_state=42,
    )
    solver.fit()
    elapsed = time.time() - t0

    y_pred = solver.predict(t_domain)
    max_err = float(np.max(np.abs(y_pred - y_analytical)))
    mse_err = float(np.mean((y_pred - y_analytical) ** 2))

    print(f"\nElapsed Time:             {elapsed:.3f} s")
    print(f"Max Absolute Error:       {max_err:.6f}")
    print(f"Mean Squared Error:       {mse_err:.6e}")
    print(f"Discovered Closed-Form:   {solver.to_formula()}")

    # Plot Comparison and Residual
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Solution curve
    dense_t = np.linspace(0.0, 2.5, 200)
    dense_pred = solver.predict(dense_t)
    dense_true = np.exp(-dense_t)

    axes[0].plot(dense_t, dense_true, "k--", lw=2, label="Analytical Truth: exp(-t)")
    axes[0].plot(dense_t, dense_pred, color="#1f77b4", lw=2, label="Discovered ODE Solution")
    axes[0].scatter([0.0], [1.0], color="red", s=70, zorder=5, label="Boundary Condition y(0)=1")
    axes[0].set_title(f"Analytical ODE Solution\nFormula: {solver.to_formula()}", fontsize=11)
    axes[0].set_xlabel("Time (t)")
    axes[0].set_ylabel("y(t)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Residual error
    residuals = np.abs(dense_pred - dense_true)
    axes[1].plot(dense_t, residuals, color="#d62728", lw=2)
    axes[1].set_yscale("log")
    axes[1].set_title(
        f"Absolute Residual Error |y_pred - y_true|\nMax Error: {max_err:.4e}", fontsize=11
    )
    axes[1].set_xlabel("Time (t)")
    axes[1].set_ylabel("Error (log scale)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved ODE discovery diagnostic plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "03_ode_discovery.png"
    run_ode_showcase(plot_file)
