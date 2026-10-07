"""01_classical_cgp_benchmarks.py.

Demonstrates classical Cartesian Genetic Programming (CGP) on standard symbolic
regression benchmarks (Koza-1 and Nguyen-7).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure packages/malthus_gp/src is on the import path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from malthus_gp.core.functions import EXTENDED_FUNCTIONS, EXTENDED_NAMES
from malthus_gp.learn import CGPRegressor


def run_benchmark_suite(output_plot: Path) -> None:
    print("=" * 70)
    print("  Malthus-GP Showcase 1: Classical Cartesian GP Benchmarks")
    print("=" * 70)

    # 1. Koza-1 Benchmark: y = x^4 + x^3 + x^2 + x on [-1, 1]
    print("\n--- Problem 1: Koza-1 Quartic Polynomial ---")
    print("Target Function: y = x^4 + x^3 + x^2 + x")
    X_koza = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y_koza = (X_koza[:, 0] ** 4 + X_koza[:, 0] ** 3 + X_koza[:, 0] ** 2 + X_koza[:, 0]).flatten()

    t0 = time.time()
    reg_koza = CGPRegressor(
        n_rows=1,
        n_cols=30,
        levels_back=-1,
        pop_size=5,
        max_generations=500,
        mutation_rate=0.04,
        random_state=42,
    )
    reg_koza.fit(X_koza, y_koza)
    t_koza = time.time() - t0

    r2_koza = reg_koza.score(X_koza, y_koza)
    active_mask_koza = reg_koza.get_active_mask()
    active_ratio_koza = float(np.mean(active_mask_koza))

    print(f"Elapsed Time:       {t_koza:.3f} s")
    print(f"Final R^2 Score:    {r2_koza:.5f}")
    print(
        f"Active Nodes:       {int(np.sum(active_mask_koza))} / {len(active_mask_koza)} ({active_ratio_koza * 100:.1f}%)"
    )
    print(f"Evolved Formula:    {reg_koza.to_formula()}")

    # 2. Nguyen-7 Benchmark: y = ln(x + 1) + ln(x^2 + 1) on [0, 2]
    print("\n--- Problem 2: Nguyen-7 Transcendental Function ---")
    print("Target Function: y = ln(x + 1) + ln(x^2 + 1)")
    X_nguyen = np.linspace(0.0, 2.0, 40).reshape(-1, 1)
    y_nguyen = np.log(X_nguyen[:, 0] + 1.0) + np.log(X_nguyen[:, 0] ** 2 + 1.0)

    t0 = time.time()
    reg_nguyen = CGPRegressor(
        n_rows=1,
        n_cols=35,
        levels_back=-1,
        pop_size=5,
        max_generations=600,
        functions=EXTENDED_FUNCTIONS,
        function_names=EXTENDED_NAMES,
        random_state=42,
    )
    reg_nguyen.fit(X_nguyen, y_nguyen)
    t_nguyen = time.time() - t0

    r2_nguyen = reg_nguyen.score(X_nguyen, y_nguyen)
    active_mask_nguyen = reg_nguyen.get_active_mask()
    active_ratio_nguyen = float(np.mean(active_mask_nguyen))

    print(f"Elapsed Time:       {t_nguyen:.3f} s")
    print(f"Final R^2 Score:    {r2_nguyen:.5f}")
    print(
        f"Active Nodes:       {int(np.sum(active_mask_nguyen))} / {len(active_mask_nguyen)} ({active_ratio_nguyen * 100:.1f}%)"
    )
    print(f"Evolved Formula:    {reg_nguyen.to_formula()}")

    # Plot results
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Koza-1 Plot
    axes[0].scatter(X_koza[:, 0], y_koza, color="black", label="Ground Truth", zorder=3)
    dense_x_koza = np.linspace(-1.0, 1.0, 200).reshape(-1, 1)
    dense_pred_koza = reg_koza.predict(dense_x_koza)
    axes[0].plot(
        dense_x_koza[:, 0],
        dense_pred_koza,
        color="#1f77b4",
        lw=2,
        label=f"CGP Fit (R²={r2_koza:.4f})",
    )
    axes[0].set_title(
        f"Koza-1: y = x⁴ + x³ + x² + x\nFormula: {reg_koza.to_formula()}", fontsize=11
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Nguyen-7 Plot
    axes[1].scatter(X_nguyen[:, 0], y_nguyen, color="black", label="Ground Truth", zorder=3)
    dense_x_nguyen = np.linspace(0.0, 2.0, 200).reshape(-1, 1)
    dense_pred_nguyen = reg_nguyen.predict(dense_x_nguyen)
    axes[1].plot(
        dense_x_nguyen[:, 0],
        dense_pred_nguyen,
        color="#2ca02c",
        lw=2,
        label=f"CGP Fit (R²={r2_nguyen:.4f})",
    )
    axes[1].set_title(
        f"Nguyen-7: y = ln(x+1) + ln(x²+1)\nFormula: {reg_nguyen.to_formula()}", fontsize=11
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved benchmark diagnostic plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "01_cgp_benchmarks.png"
    run_benchmark_suite(plot_file)
