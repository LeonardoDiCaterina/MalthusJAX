"""02_global_optimization_benchmarks.py.

Compares multiple global optimization algorithms (Differential Evolution, CMA-ES, PSO, GA)
on classic multi-modal landscapes (Rastrigin & Rosenbrock).
Visualizes generational convergence trajectories across all strategies.
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


def rastrigin(x: np.ndarray) -> float:
    """Multi-modal Rastrigin benchmark with many local minima: global min at x=0, f=0."""
    return float(10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """Non-convex Rosenbrock banana valley: global min at x=1, f=0."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def run_benchmark_showcase(output_plot: Path) -> None:
    print("=" * 75)
    print("  Malthus-Optimize Showcase 2: Global Optimization Benchmark Comparison")
    print("=" * 75)

    dim = 5
    generations = 80
    pop_size = 40
    seed = 42

    bounds_rast = [(-5.12, 5.12)] * dim
    bounds_rosen = [(-3.0, 3.0)] * dim

    methods = [
        ("de", "Differential Evolution", "#2980b9"),
        ("cmaes", "CMA-ES", "#e74c3c"),
        ("pso", "Particle Swarm (PSO)", "#27ae60"),
        ("ga", "Genetic Algorithm (GA)", "#8e44ad"),
    ]

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Rastrigin Function
    print(f"\nEvaluating {dim}D Rastrigin Function ({generations} gens, {pop_size} pop):")
    for method_key, label, color in methods:
        t0 = time.perf_counter()
        if method_key == "de":
            res = differential_evolution(
                rastrigin,
                bounds=bounds_rast,
                popsize=pop_size // dim,
                maxiter=generations,
                seed=seed,
            )
        else:
            res = minimize(
                rastrigin,
                bounds=bounds_rast,
                method=method_key,
                options={"popsize": pop_size, "maxiter": generations, "seed": seed},
            )
        elapsed = time.perf_counter() - t0
        print(f"  {label:<25} -> Best Fun: {res.fun:.6e} ({elapsed:.3f} s)")

        history = [h["best_fitness"] for h in res.history]
        axes[0].plot(
            range(1, len(history) + 1),
            history,
            label=f"{label} ({res.fun:.2e})",
            color=color,
            lw=2.0,
        )

    axes[0].set_yscale("log")
    axes[0].set_title("5D Rastrigin Convergence (Multi-Modal)", fontsize=11)
    axes[0].set_xlabel("Generation / Iteration", fontsize=10)
    axes[0].set_ylabel("Best Objective Value (log scale)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=9)

    # Panel B: Rosenbrock Function
    print(f"\nEvaluating {dim}D Rosenbrock Function ({generations} gens, {pop_size} pop):")
    for method_key, label, color in methods:
        t0 = time.perf_counter()
        if method_key == "de":
            res = differential_evolution(
                rosenbrock,
                bounds=bounds_rosen,
                popsize=pop_size // dim,
                maxiter=generations,
                seed=seed,
            )
        else:
            res = minimize(
                rosenbrock,
                bounds=bounds_rosen,
                method=method_key,
                options={"popsize": pop_size, "maxiter": generations, "seed": seed},
            )
        elapsed = time.perf_counter() - t0
        print(f"  {label:<25} -> Best Fun: {res.fun:.6e} ({elapsed:.3f} s)")

        history = [h["best_fitness"] for h in res.history]
        axes[1].plot(
            range(1, len(history) + 1),
            history,
            label=f"{label} ({res.fun:.2e})",
            color=color,
            lw=2.0,
        )

    axes[1].set_yscale("log")
    axes[1].set_title("5D Rosenbrock Convergence (Banana Valley)", fontsize=11)
    axes[1].set_xlabel("Generation / Iteration", fontsize=10)
    axes[1].set_ylabel("Best Objective Value (log scale)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved benchmark trajectories plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "02_global_optimization_benchmarks.png"
    run_benchmark_showcase(plot_file)
