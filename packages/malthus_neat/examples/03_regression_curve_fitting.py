"""03_regression_curve_fitting.py.

Demonstrates non-linear continuous curve fitting using NEATRegressor,
visualizing the learned functional approximation and the underlying evolved neural network topology.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from malthus_neat.learn import NEATRegressor


def run_regression_showcase(output_plot: Path) -> None:
    print("=" * 70)
    print("  Malthus-NEAT Showcase 3: Non-Linear Function Fitting")
    print("=" * 70)

    # 1. Non-linear target function: y = sin(2.5 * x) + 0.5 * x
    np.random.seed(42)
    n_samples = 60
    X = np.linspace(-2.0, 2.0, n_samples).reshape(-1, 1).astype(np.float32)
    y = (np.sin(2.5 * X[:, 0]) + 0.5 * X[:, 0]).astype(np.float32)
    y_noisy = y + np.random.normal(0, 0.05, size=y.shape).astype(np.float32)

    print(f"Target: y = sin(2.5*x) + 0.5*x over [-2.0, 2.0] ({n_samples} points)")

    # 2. Configure NEATRegressor
    t0 = time.time()
    reg = NEATRegressor(
        pop_size=60,
        max_generations=120,
        max_nodes=16,
        max_conns=32,
        elitism=2,
        random_state=42,
    )
    reg.fit(X, y_noisy)
    elapsed = time.time() - t0

    r2_score = reg.score(X, y)
    active_nodes, active_conns = reg.best_genome_.get_active_mask()

    print(f"\nEvolution Time:  {elapsed:.3f} s")
    print(f"Final R² Score:  {r2_score:.4f}")
    print(f"Active Nodes:    {int(np.sum(active_nodes))}")
    print(f"Active Synapses: {int(np.sum(active_conns))} / {reg.max_conns}")

    # 3. Render 2-panel figure: Curve Approximation + Evolved Network
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    dense_x = np.linspace(-2.0, 2.0, 200).reshape(-1, 1).astype(np.float32)
    dense_true = np.sin(2.5 * dense_x[:, 0]) + 0.5 * dense_x[:, 0]
    dense_pred = reg.predict(dense_x)

    # Panel A: Fitted Function Curve
    axes[0].scatter(X[:, 0], y_noisy, color="black", alpha=0.6, label="Training Data", zorder=3)
    axes[0].plot(dense_x[:, 0], dense_true, "k--", lw=2, label="Analytical Truth: sin(2.5x) + 0.5x")
    axes[0].plot(
        dense_x[:, 0],
        dense_pred,
        color="#1f77b4",
        lw=2.5,
        label=f"Evolved NEAT Network (R² = {r2_score:.3f})",
    )
    axes[0].set_title("Non-Linear Function Approximation", fontsize=11)
    axes[0].set_xlabel("Input x")
    axes[0].set_ylabel("Output y")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Panel B: Network Topology Diagram
    reg.plot_network(ax=axes[1])
    axes[1].set_title(
        f"Evolved Neural Network Architecture ({int(np.sum(active_nodes))} nodes, {int(np.sum(active_conns))} synapses)",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved regression evolution plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "03_regression_curve_fitting.png"
    run_regression_showcase(plot_file)
