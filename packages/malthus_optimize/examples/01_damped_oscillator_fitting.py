"""01_damped_oscillator_fitting.py.

Demonstrates non-linear curve fitting on a classic damped harmonic oscillator signal:
y(t) = A * exp(-gamma * t) * cos(omega * t + phi) + C

Highlights the advantage of global evolutionary curve fitting over gradient-based
methods (like standard Levenberg-Marquardt), which frequently get trapped in local
minima due to phase wrapping and frequency harmonic ripples.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from malthus_optimize import curve_fit


def damped_oscillator(
    t: np.ndarray,
    amplitude: float,
    decay_rate: float,
    omega: float,
    phase: float,
    offset: float,
) -> np.ndarray:
    """Damped harmonic oscillator model."""
    return amplitude * np.exp(-decay_rate * t) * np.cos(omega * t + phase) + offset


def run_oscillator_showcase(output_plot: Path) -> None:
    print("=" * 75)
    print("  Malthus-Optimize Showcase 1: Damped Oscillator Non-Linear Curve Fitting")
    print("=" * 75)

    # 1. Synthesize noisy damped oscillation data
    np.random.seed(42)
    t = np.linspace(0.0, 8.0, 120).astype(np.float32)

    # Ground truth parameters: [A, gamma, omega, phi, C]
    true_params = [5.0, 0.4, 4.5, 0.8, 1.2]
    param_names = ["Amplitude (A)", "Decay (γ)", "Freq (ω)", "Phase (φ)", "Offset (C)"]

    y_clean = damped_oscillator(t, *true_params)
    noise_sigma = 0.3
    noise = np.random.normal(0.0, noise_sigma, size=len(t)).astype(np.float32)
    y_observed = y_clean + noise

    print(f"Data: {len(t)} points over t ∈ [0.0, 8.0] s with noise σ = {noise_sigma}")
    print(
        f"Ground Truth: A={true_params[0]}, γ={true_params[1]}, ω={true_params[2]}, φ={true_params[3]}, C={true_params[4]}"
    )

    # 2. Define search bounds
    bounds = (
        [1.0, 0.05, 1.0, -np.pi, -2.0],  # Lower bounds
        [10.0, 1.5, 8.0, np.pi, 4.0],  # Upper bounds
    )

    # 3. Fit via malthus_optimize.curve_fit (Global Evolutionary Search)
    t0 = time.perf_counter()
    popt, pcov, info = curve_fit(
        damped_oscillator,
        t,
        y_observed,
        bounds=bounds,
        pop_size=40,
        max_generations=100,
        seed=42,
        full_output=True,
    )
    t_global = time.perf_counter() - t0

    residuals_global = y_observed - damped_oscillator(t, *popt)
    r2_global = 1.0 - (
        np.sum(residuals_global**2) / np.sum((y_observed - np.mean(y_observed)) ** 2)
    )
    perr_global = np.sqrt(np.diag(pcov))

    print("\n[Malthus-Optimize Global Search]")
    print(f"  Time: {t_global:.3f} s ({info['nfev']} evals)")
    print(f"  Fit Quality R²: {r2_global:.5f}")
    for name, val, err, truth in zip(param_names, popt, perr_global, true_params):
        print(f"  {name:<15}: {val:.3f} ± {err:.3f} (Truth: {truth:.3f})")

    # 4. Compare with local SciPy curve_fit without a fine-tuned initial guess
    scipy_fit_curve = None
    r2_local = 0.0
    try:
        import scipy.optimize as sp_opt

        # Local gradient descent with generic p0
        p0_generic = [1.0, 1.0, 1.0, 0.0, 0.0]
        popt_local, _ = sp_opt.curve_fit(
            damped_oscillator, t, y_observed, p0=p0_generic, maxfev=2000
        )
        residuals_local = y_observed - damped_oscillator(t, *popt_local)
        r2_local = 1.0 - (
            np.sum(residuals_local**2) / np.sum((y_observed - np.mean(y_observed)) ** 2)
        )
        scipy_fit_curve = damped_oscillator(t, *popt_local)
        print("\n[SciPy Local Search (Generic p0)]")
        print(f"  Trapped in local minimum with R²: {r2_local:.5f}")
    except Exception as exc:
        print(f"\n[SciPy Local Search] Skipped: {exc}")

    # 5. Generate 2-panel figure
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    dense_t = np.linspace(0.0, 8.0, 300)
    fit_curve_global = damped_oscillator(dense_t, *popt)
    truth_curve = damped_oscillator(dense_t, *true_params)

    # Panel A: Fits vs Data
    axes[0].scatter(
        t, y_observed, color="#2c3e50", alpha=0.5, s=20, label="Observed Noisy Data", zorder=2
    )
    axes[0].plot(dense_t, truth_curve, "k--", lw=1.5, alpha=0.7, label="Ground Truth", zorder=3)
    axes[0].plot(
        dense_t,
        fit_curve_global,
        color="#2980b9",
        lw=2.2,
        label=f"Malthus-Optimize Fit (R² = {r2_global:.4f})",
        zorder=4,
    )

    if scipy_fit_curve is not None:
        dense_local = damped_oscillator(dense_t, *popt_local)
        axes[0].plot(
            dense_local,
            color="#e74c3c",
            lw=1.8,
            linestyle=":",
            label=f"Local Gradient Fit (R² = {r2_local:.2f})",
            zorder=3,
        )

    axes[0].set_title("Damped Oscillator Non-Linear Curve Fitting", fontsize=11)
    axes[0].set_xlabel("Time t (seconds)", fontsize=10)
    axes[0].set_ylabel("Signal Amplitude y(t)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper right")

    # Panel B: Residuals Distribution
    axes[1].axhline(0.0, color="black", linestyle="-", lw=1.0, alpha=0.6)
    axes[1].axhline(
        2 * noise_sigma, color="gray", linestyle="--", lw=1.0, alpha=0.5, label="±2σ Expected Noise"
    )
    axes[1].axhline(-2 * noise_sigma, color="gray", linestyle="--", lw=1.0, alpha=0.5)
    axes[1].scatter(
        t, residuals_global, color="#2980b9", alpha=0.7, s=22, label="Malthus Fit Residuals"
    )

    axes[1].set_title(f"Fit Residuals (RMS Error = {np.std(residuals_global):.3f})", fontsize=11)
    axes[1].set_xlabel("Time t (seconds)", fontsize=10)
    axes[1].set_ylabel("Residual (Observed - Fitted)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved damped oscillator plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "01_damped_oscillator_fitting.png"
    run_oscillator_showcase(plot_file)
