"""01_xor_minimal_topology.py.

Demonstrates evolving a minimal neural network topology to solve the non-linear
XOR benchmark from scratch using NEATClassifier and visualizing the evolved wiring.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from malthus_neat.learn import NEATClassifier


def run_xor_showcase(output_plot: Path) -> None:
    print("=" * 70)
    print("  Malthus-NEAT Showcase 1: Minimal Topology XOR Evolution")
    print("=" * 70)

    # 1. XOR Dataset
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)

    print("Task: Non-linear XOR Classification")
    for xi, yi in zip(X, y):
        print(f"  Input: {xi} -> Target: {yi}")

    # 2. Configure NEATClassifier
    t0 = time.time()
    clf = NEATClassifier(
        pop_size=60,
        max_generations=100,
        max_nodes=12,
        max_conns=24,
        elitism=2,
        random_state=42,
    )
    clf.fit(X, y)
    elapsed = time.time() - t0

    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    acc = float(np.mean(preds == y))
    active_nodes, active_conns = clf.best_genome_.get_active_mask()

    print(f"\nEvolution Time:     {elapsed:.3f} s")
    print(f"Classification Acc: {acc * 100:.1f}%")
    print(f"Active Nodes:       {int(np.sum(active_nodes))} (including inputs/outputs)")
    print(f"Active Synapses:    {int(np.sum(active_conns))} / {clf.max_conns}")
    print(f"Predicted Probs:    {probs[:, 1].round(4).tolist()}")

    # 3. Render 2-panel figure: Decision boundary + Neural Wiring
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: 2D Decision Boundary Contour
    grid_res = 100
    gx = np.linspace(-0.2, 1.2, grid_res)
    gy = np.linspace(-0.2, 1.2, grid_res)
    xx, yy = np.meshgrid(gx, gy)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_probs = clf.predict_proba(grid_points)[:, 1].reshape(xx.shape)

    contour = axes[0].contourf(xx, yy, grid_probs, levels=20, cmap="coolwarm", alpha=0.8)
    fig.colorbar(contour, ax=axes[0], label="P(y = 1)")
    axes[0].scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color="blue",
        s=120,
        edgecolors="black",
        lw=1.5,
        label="Class 0",
        zorder=5,
    )
    axes[0].scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color="red",
        s=120,
        edgecolors="black",
        lw=1.5,
        label="Class 1",
        zorder=5,
    )
    axes[0].set_title(f"XOR Decision Boundary (Acc = {acc * 100:.1f}%)", fontsize=11)
    axes[0].set_xlabel("Input x0")
    axes[0].set_ylabel("Input x1")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Panel B: Evolved Network Wiring
    clf.plot_network(ax=axes[1])
    axes[1].set_title(
        f"Evolved Neural Topology ({int(np.sum(active_nodes))} nodes, {int(np.sum(active_conns))} synapses)",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved XOR evolution plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "01_xor_minimal_topology.png"
    run_xor_showcase(plot_file)
