"""03_sensor_trilateration.py.

Demonstrates geometric global optimization for 2D sensor network localization (trilateration).
Given noisy range/distance measurements from multiple fixed anchor stations (e.g. GPS,
radar, or Wi-Fi beacons), finds the unknown 2D coordinates (x, y) of an emitter.

Visualizes:
1. The physical anchor network with noisy measurement range circles.
2. The 2D non-linear loss contour landscape with recovered target position.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from malthus_optimize import differential_evolution, minimize


def run_trilateration_showcase(output_plot: Path) -> None:
    print("=" * 75)
    print("  Malthus-Optimize Showcase 3: 2D Sensor Network Trilateration")
    print("=" * 75)

    # 1. Define anchor beacon coordinates in a 10m x 10m workspace
    anchors = np.array(
        [
            [1.0, 1.5],  # Anchor 1 (SW)
            [9.0, 1.0],  # Anchor 2 (SE)
            [8.5, 9.0],  # Anchor 3 (NE)
            [1.5, 8.5],  # Anchor 4 (NW)
            [5.0, 5.5],  # Anchor 5 (Center)
        ],
        dtype=np.float32,
    )

    # True unknown transmitter position
    true_target = np.array([6.8, 3.6], dtype=np.float32)

    # Compute true Euclidean distances and add measurement noise
    np.random.seed(42)
    true_distances = np.linalg.norm(anchors - true_target, axis=1)
    noise_sigma = 0.15  # 15 cm sensor noise
    noisy_distances = true_distances + np.random.normal(0.0, noise_sigma, size=len(anchors))

    print(f"Anchors ({len(anchors)} stations):\n{anchors}")
    print(f"True Target Position: x={true_target[0]:.2f}, y={true_target[1]:.2f}")
    print(f"Noisy Range Measurements: {np.round(noisy_distances, 3)} m (noise σ={noise_sigma} m)")

    # 2. Formulate least-squares positioning objective: sum_i ( ||pos - anchor_i|| - d_i )^2
    def trilateration_loss(pos: np.ndarray) -> float:
        pred_dist = np.linalg.norm(anchors - pos, axis=1)
        return float(np.sum((pred_dist - noisy_distances) ** 2))

    # 3. Solve with Differential Evolution and Particle Swarm Optimization
    bounds = [(0.0, 10.0), (0.0, 10.0)]

    t0 = time.perf_counter()
    res_de = differential_evolution(
        trilateration_loss,
        bounds=bounds,
        popsize=25,
        maxiter=60,
        seed=42,
    )
    t_de = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_pso = minimize(
        trilateration_loss,
        bounds=bounds,
        method="pso",
        options={"popsize": 30, "maxiter": 60, "seed": 42},
    )
    t_pso = time.perf_counter() - t0

    err_de = np.linalg.norm(res_de.x - true_target)
    err_pso = np.linalg.norm(res_pso.x - true_target)

    print("\n[Optimization Results]")
    print(
        f"  Differential Evolution : ({res_de.x[0]:.3f}, {res_de.x[1]:.3f}) | Error: {err_de * 100:.1f} cm | Time: {t_de:.3f} s"
    )
    print(
        f"  Particle Swarm (PSO)   : ({res_pso.x[0]:.3f}, {res_pso.x[1]:.3f}) | Error: {err_pso * 100:.1f} cm | Time: {t_pso:.3f} s"
    )

    # 4. Generate 2-panel publication figure
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Physical Network Geometry
    axes[0].set_xlim(-0.5, 10.5)
    axes[0].set_ylim(-0.5, 10.5)
    axes[0].set_aspect("equal")

    # Draw range circles from each anchor
    for i, (anchor, d) in enumerate(zip(anchors, noisy_distances)):
        circle = plt.Circle(
            anchor, d, color="#3498db", fill=False, linestyle="--", alpha=0.45, lw=1.3
        )
        axes[0].add_patch(circle)
        axes[0].scatter(anchor[0], anchor[1], marker="^", s=110, color="#e74c3c", zorder=5)
        axes[0].text(
            anchor[0] + 0.25,
            anchor[1] + 0.2,
            f"A{i + 1}",
            fontsize=9,
            fontweight="bold",
            color="#c0392b",
        )

    # Draw true target and estimated target
    axes[0].scatter(
        true_target[0],
        true_target[1],
        marker="*",
        s=200,
        color="#27ae60",
        edgecolors="black",
        lw=1.0,
        label=f"True Target ({true_target[0]:.2f}, {true_target[1]:.2f})",
        zorder=6,
    )
    axes[0].scatter(
        res_de.x[0],
        res_de.x[1],
        marker="o",
        s=80,
        facecolors="none",
        edgecolors="#2980b9",
        lw=2.2,
        label=f"DE Estimate ({res_de.x[0]:.2f}, {res_de.x[1]:.2f})",
        zorder=7,
    )

    axes[0].set_title("2D Sensor Network Geometry & Range Circles", fontsize=11)
    axes[0].set_xlabel("X Coordinate (meters)", fontsize=10)
    axes[0].set_ylabel("Y Coordinate (meters)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.4)
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel B: Loss Surface Contour Map
    grid_x = np.linspace(0.0, 10.0, 100)
    grid_y = np.linspace(0.0, 10.0, 100)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])

    # Evaluate loss over 2D grid
    grid_losses = np.array([trilateration_loss(pt) for pt in grid_points]).reshape(gx.shape)

    contour = axes[1].contourf(gx, gy, grid_losses, levels=30, cmap="viridis_r", alpha=0.85)
    cbar = plt.colorbar(contour, ax=axes[1])
    cbar.set_label("Residual Error $\\sum (d_{pred} - d_{obs})^2$", fontsize=9)

    axes[1].scatter(
        true_target[0],
        true_target[1],
        marker="*",
        s=220,
        color="#2ecc71",
        edgecolors="black",
        lw=1.0,
        label="Global Minimum",
        zorder=5,
    )
    axes[1].scatter(
        res_de.x[0],
        res_de.x[1],
        marker="x",
        s=100,
        color="red",
        lw=2.5,
        label="Optimizer Solution",
        zorder=6,
    )

    axes[1].set_title("2D Trilateration Loss Landscape Contours", fontsize=11)
    axes[1].set_xlabel("X Coordinate (meters)", fontsize=10)
    axes[1].set_ylabel("Y Coordinate (meters)", fontsize=10)
    axes[1].set_aspect("equal")
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved trilateration plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "03_sensor_trilateration.png"
    run_trilateration_showcase(plot_file)
