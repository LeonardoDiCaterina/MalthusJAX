"""02_dcgp_memetic_learning.py.

Demonstrates Differentiable Cartesian Genetic Programming (dCGP) and the
LSMF (Learn, Select, Mutate, Forget) memetic algorithm.

Compares classical discrete CGP vs. dCGP on a polynomial with non-trivial
floating-point coefficients: y = 2.45*x^3 - 1.82*x^2 + 0.65*x - 0.35.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from malthus_gp.learn import CGPRegressor, DCGPRegressor


def run_dcgp_showcase(output_plot: Path) -> None:
    print("=" * 70)
    print("  Malthus-GP Showcase 2: Differentiable CGP (dCGP) & LSMF Memetic Search")
    print("=" * 70)

    # Continuous coefficient polynomial
    # Classical GP struggles to find constants like 2.45 or -1.82 without bloat
    np.random.seed(42)
    X = np.linspace(-1.5, 1.5, 50).reshape(-1, 1)
    y = 2.45 * (X[:, 0] ** 3) - 1.82 * (X[:, 0] ** 2) + 0.65 * X[:, 0] - 0.35
    # Add minor noise
    y_noisy = y + np.random.normal(0, 0.05, size=y.shape)

    print("\nTarget: y = 2.45*x³ - 1.82*x² + 0.65*x - 0.35 (+ small noise)")

    # 1. Classical Discrete CGP (Must approximate real constants via subtrees)
    print("\n--- 1. Running Classical Discrete CGP (500 Generations) ---")
    t0 = time.time()
    cgp = CGPRegressor(
        n_rows=1,
        n_cols=30,
        pop_size=5,
        max_generations=500,
        random_state=42,
    )
    cgp.fit(X, y_noisy)
    t_cgp = time.time() - t0
    r2_cgp = cgp.score(X, y)
    print(f"Classical CGP Time:    {t_cgp:.3f} s")
    print(f"Classical CGP R² Score: {r2_cgp:.5f}")
    print(f"Formula:                {cgp.to_formula()}")

    # 2. Differentiable Memetic CGP (Optimizes continuous weights with Optax Adam)
    print("\n--- 2. Running Differentiable CGP (dCGP with LSMF Adam Epochs) ---")
    t0 = time.time()
    dcgp = DCGPRegressor(
        n_rows=1,
        n_cols=20,
        pop_size=5,
        max_generations=150,
        lsmf_epochs=15,
        learning_rate=0.03,
        random_state=42,
    )
    dcgp.fit(X, y_noisy)
    t_dcgp = time.time() - t0
    r2_dcgp = dcgp.score(X, y)
    print(f"dCGP (Memetic) Time:    {t_dcgp:.3f} s")
    print(f"dCGP R² Score:          {r2_dcgp:.5f}")
    print(f"Formula (with weights): {dcgp.to_formula()}")

    # Generate Comparison Plot
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    dense_x = np.linspace(-1.5, 1.5, 200).reshape(-1, 1)
    dense_true = (
        2.45 * (dense_x[:, 0] ** 3) - 1.82 * (dense_x[:, 0] ** 2) + 0.65 * dense_x[:, 0] - 0.35
    )
    dense_pred_cgp = cgp.predict(dense_x)
    dense_pred_dcgp = dcgp.predict(dense_x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        X[:, 0], y_noisy, color="black", alpha=0.6, label="Training Data (with noise)", zorder=3
    )
    ax.plot(dense_x[:, 0], dense_true, "k--", lw=2, label="Ground Truth Analytical", zorder=2)
    ax.plot(
        dense_x[:, 0],
        dense_pred_cgp,
        color="#d62728",
        lw=2,
        label=f"Classical CGP (R²={r2_cgp:.4f})",
    )
    ax.plot(
        dense_x[:, 0],
        dense_pred_dcgp,
        color="#1f77b4",
        lw=2.5,
        label=f"dCGP + LSMF Memetic (R²={r2_dcgp:.4f})",
    )

    ax.set_title(
        "Classical CGP vs. Differentiable CGP (dCGP) on Real-Valued Coefficients", fontsize=12
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved dCGP comparison plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "02_dcgp_memetic.png"
    run_dcgp_showcase(plot_file)
