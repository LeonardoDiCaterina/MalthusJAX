"""04_qd_neat_repertoire.py.

Demonstrates Quality-Diversity neuroevolution (QD-NEAT / MAP-Elites) using NEATRegressor.
Explores the dual-objective behavioral descriptor space:
- BD0: Connection Density (structural complexity)
- BD1: Output Variance (prediction behavior diversity)
Visualizes the resulting MAP-Elites repertoire archive alongside the elite network topology.
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


def run_qd_neat_showcase(output_plot: Path) -> None:
    print("=" * 70)
    print("  Malthus-NEAT Showcase 4: Quality-Diversity MAP-Elites Neuroevolution")
    print("=" * 70)

    # 1. Non-linear multi-frequency dataset: y = sin(3x) + 0.5 * cos(7x)
    np.random.seed(42)
    n_samples = 80
    X = np.linspace(-1.5, 1.5, n_samples).reshape(-1, 1).astype(np.float32)
    y = (np.sin(3.0 * X[:, 0]) + 0.5 * np.cos(7.0 * X[:, 0])).astype(np.float32)

    num_centroids = 40
    print(f"Dataset: Multi-frequency oscillation ({n_samples} points in [-1.5, 1.5])")
    print(f"Engine:  MAP-Elites (QD-NEAT) with {num_centroids} centroids")

    # 2. Fit with MAP-Elites engine
    t0 = time.time()
    qd_reg = NEATRegressor(
        pop_size=60,
        max_generations=40,
        max_nodes=16,
        max_conns=32,
        engine="map_elites",
        engine_params={"num_centroids": num_centroids},
        random_state=42,
    )
    qd_reg.fit(X, y)
    elapsed = time.time() - t0

    rep = qd_reg.get_repertoire()
    centroids = np.array(rep.centroids)  # (C, 2)
    fitnesses = np.array(rep.fitnesses).squeeze()  # (C,)
    descriptors = np.array(rep.descriptors)  # (C, 2)

    occupied_mask = ~np.isinf(fitnesses) & ~np.isnan(fitnesses)
    num_occupied = int(np.sum(occupied_mask))
    coverage = (num_occupied / num_centroids) * 100.0

    print(f"\nEvolution Time:     {elapsed:.3f} s")
    print(f"Repertoire Cells:   {num_occupied} / {num_centroids} ({coverage:.1f}% coverage)")
    print(f"Best Elite Fitness: {qd_reg.best_fitness_:.4f}")

    active_nodes, active_conns = qd_reg.best_genome_.get_active_mask()
    print(f"Elite Active Nodes: {int(np.sum(active_nodes))}")
    print(f"Elite Synapses:     {int(np.sum(active_conns))} / {qd_reg.max_conns}")

    # 3. Create 2-panel diagnostic figure
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # Panel A: MAP-Elites Repertoire in Behavioral Descriptor Space
    # Draw all centroids as empty markers
    axes[0].scatter(
        centroids[~occupied_mask, 0],
        centroids[~occupied_mask, 1],
        color="lightgray",
        marker="x",
        s=30,
        alpha=0.6,
        label="Empty Centroid",
        zorder=1,
    )

    # Draw occupied cells color-coded by fitness
    occ_x = descriptors[occupied_mask, 0]
    occ_y = descriptors[occupied_mask, 1]
    occ_fit = fitnesses[occupied_mask]

    sc = axes[0].scatter(
        occ_x,
        occ_y,
        c=occ_fit,
        cmap="viridis",
        s=90,
        edgecolors="black",
        linewidths=0.8,
        label=f"Occupied Elite ({num_occupied})",
        zorder=2,
    )
    cbar = plt.colorbar(sc, ax=axes[0], pad=0.02)
    cbar.set_label("Elite Fitness (Higher is Better)", fontsize=10)

    # Highlight global best
    best_idx = np.argmax(np.where(occupied_mask, fitnesses, -np.inf))
    best_desc = descriptors[best_idx]
    axes[0].scatter(
        [best_desc[0]],
        [best_desc[1]],
        color="#e74c3c",
        edgecolors="black",
        s=220,
        marker="*",
        label="Global Best Elite",
        zorder=4,
    )

    axes[0].set_title(
        f"MAP-Elites Behavioral Archive ({num_occupied}/{num_centroids} Cells, Coverage: {coverage:.1f}%)",
        fontsize=11,
    )
    axes[0].set_xlabel("Behavioral Descriptor 0: Connection Density (Synapses / Max)", fontsize=10)
    axes[0].set_ylabel("Behavioral Descriptor 1: Output Variance (tanh(std))", fontsize=10)
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left", fontsize=9)

    # Panel B: Best Elite Evolved Topology
    qd_reg.plot_network(ax=axes[1])
    axes[1].set_title(
        f"Global Best Elite Network Topology ({int(np.sum(active_nodes))} nodes, {int(np.sum(active_conns))} synapses)",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved QD-NEAT repertoire plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "04_qd_neat_repertoire.png"
    run_qd_neat_showcase(plot_file)
