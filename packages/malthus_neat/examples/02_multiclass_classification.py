"""02_multiclass_classification.py.

Demonstrates multi-class non-linear classification using NEATClassifier,
visualizing decision regions and the evolved multi-output neural architecture.
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


def make_spiral_data(
    n_samples_per_class: int = 50, n_classes: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Generates 2D multi-class spiral dataset."""
    np.random.seed(42)
    X = []
    y = []
    for j in range(n_classes):
        r = np.linspace(0.1, 1.0, n_samples_per_class)
        t = (
            np.linspace(j * 4.0, (j + 1) * 4.0, n_samples_per_class)
            + np.random.randn(n_samples_per_class) * 0.15
        )
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append([j] * n_samples_per_class)
    return np.vstack(X).astype(np.float32), np.concatenate(y).astype(np.int32)


def run_multiclass_showcase(output_plot: Path) -> None:
    print("=" * 70)
    print("  Malthus-NEAT Showcase 2: Multi-Class Decision Region Evolution")
    print("=" * 70)

    # 1. Generate multi-class spiral dataset
    X, y = make_spiral_data(n_samples_per_class=40, n_classes=3)
    n_classes = len(np.unique(y))
    print(f"Dataset: {len(X)} samples, 2 input features [x0, x1], {n_classes} classes")

    # 2. Configure NEATClassifier for multi-output classification
    t0 = time.time()
    clf = NEATClassifier(
        pop_size=60,
        max_generations=120,
        max_nodes=18,
        max_conns=36,
        elitism=2,
        random_state=42,
    )
    clf.fit(X, y)
    elapsed = time.time() - t0

    preds = clf.predict(X)
    acc = float(np.mean(preds == y))
    active_nodes, active_conns = clf.best_genome_.get_active_mask()

    print(f"\nEvolution Time:     {elapsed:.3f} s")
    print(f"Classification Acc: {acc * 100:.1f}%")
    print(f"Active Nodes:       {int(np.sum(active_nodes))}")
    print(f"Active Synapses:    {int(np.sum(active_conns))} / {clf.max_conns}")

    # 3. Render 2-panel plot: Multi-class Decision Surface + Evolved Architecture
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: 2D Decision Regions
    grid_res = 120
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    gx = np.linspace(x_min, x_max, grid_res)
    gy = np.linspace(y_min, y_max, grid_res)
    xx, yy = np.meshgrid(gx, gy)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_preds = clf.predict(grid_points).reshape(xx.shape)

    axes[0].contourf(
        xx, yy, grid_preds, alpha=0.35, cmap="tab10", levels=np.arange(n_classes + 1) - 0.5
    )
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", edgecolors="black", s=50, lw=0.8)
    axes[0].set_title(f"3-Class Spiral Decision Regions (Acc = {acc * 100:.1f}%)", fontsize=11)
    axes[0].set_xlabel("Feature x0")
    axes[0].set_ylabel("Feature x1")
    axes[0].grid(True, alpha=0.3)

    # Panel B: Evolved Network Wiring
    clf.plot_network(ax=axes[1])
    axes[1].set_title(
        f"Evolved Multi-Output Architecture ({int(np.sum(active_nodes))} nodes, {int(np.sum(active_conns))} synapses)",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved multi-class evolution plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "02_multiclass_classification.png"
    run_multiclass_showcase(plot_file)
