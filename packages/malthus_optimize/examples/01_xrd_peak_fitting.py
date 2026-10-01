"""01_xrd_peak_fitting.py.

Demonstrates accelerated multi-peak fitting on overlapping XRD 2theta-omega diffraction scans
using malthus_optimize.curve_fit and differential_evolution.
Visualizes individual peak deconvolution and cumulative fit residuals.
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


def pseudo_voigt(x: np.ndarray, amp: float, cen: float, fwhm: float, eta: float) -> np.ndarray:
    """Pseudo-Voigt profile: linear combination of Gaussian and Lorentzian peak profiles."""
    sig = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    gauss = np.exp(-((x - cen) ** 2) / (2.0 * sig**2))
    lorentz = 1.0 / (1.0 + ((x - cen) / gamma) ** 2)
    return amp * (eta * lorentz + (1.0 - eta) * gauss)


def doublet_model(
    x: np.ndarray,
    a1: float,
    c1: float,
    w1: float,
    eta1: float,
    a2: float,
    c2: float,
    w2: float,
    eta2: float,
    bg_slope: float,
    bg_intercept: float,
) -> np.ndarray:
    """Two overlapping Bragg peaks with a linear background baseline."""
    p1 = pseudo_voigt(x, a1, c1, w1, eta1)
    p2 = pseudo_voigt(x, a2, c2, w2, eta2)
    bg = bg_slope * (x - x.min()) + bg_intercept
    return p1 + p2 + bg


def run_xrd_peak_fitting_showcase(output_plot: Path) -> None:
    print("=" * 75)
    print("  Malthus-Optimize Showcase 1: XRD Multi-Peak Deconvolution")
    print("=" * 75)

    # 1. Synthesize overlapping XRD diffraction scan in 2theta range [36.0, 42.0] degrees
    np.random.seed(42)
    two_theta = np.linspace(36.0, 42.0, 150).astype(np.float32)

    # True structural peak parameters: Peak 1 (substrate/matrix), Peak 2 (thin film/epitaxial phase)
    true_params = [
        1200.0,
        38.4,
        0.45,
        0.35,  # Peak 1: [amp, center, fwhm, eta]
        850.0,
        39.6,
        0.60,
        0.50,  # Peak 2: [amp, center, fwhm, eta]
        -15.0,
        120.0,  # Linear background: [slope, intercept]
    ]

    clean_intensity = doublet_model(two_theta, *true_params)
    noise = np.random.normal(0.0, 30.0, size=len(two_theta)).astype(np.float32)
    measured_intensity = np.maximum(clean_intensity + noise, 0.0)

    print(
        f"Data: 150 diffraction points across 2θ ∈ [{two_theta.min():.1f}°, {two_theta.max():.1f}°]"
    )
    print("Model: Doublet Pseudo-Voigt with linear baseline (10 parameters)")

    # 2. Define global parameter search bounds
    bounds = [
        (100.0, 2500.0),
        (37.5, 39.0),
        (0.1, 1.2),
        (0.0, 1.0),  # Peak 1
        (100.0, 2000.0),
        (39.0, 40.5),
        (0.1, 1.5),
        (0.0, 1.0),  # Peak 2
        (-100.0, 100.0),
        (0.0, 300.0),  # Background
    ]

    # 3. Fit via malthus_optimize.curve_fit
    t0 = time.perf_counter()
    popt, pcov, info = curve_fit(
        doublet_model,
        two_theta,
        measured_intensity,
        bounds=bounds,
        full_output=True,
        pop_size=50,
        max_generations=120,
        seed=42,
    )
    elapsed = time.perf_counter() - t0

    # Extract deconvolved components
    fit_curve = doublet_model(two_theta, *popt)
    residuals = measured_intensity - fit_curve
    r2_score = 1.0 - (
        np.sum(residuals**2) / np.sum((measured_intensity - np.mean(measured_intensity)) ** 2)
    )

    peak1_fit = pseudo_voigt(two_theta, *popt[0:4])
    peak2_fit = pseudo_voigt(two_theta, *popt[4:8])
    bg_fit = popt[8] * (two_theta - two_theta.min()) + popt[9]

    print(f"\nOptimization Time: {elapsed:.3f} s ({info['nfev']} evaluations)")
    print(f"Goodness of Fit R²: {r2_score:.5f}")
    print(
        f"Fitted Peak 1: Center = {popt[1]:.3f}° (Truth = {true_params[1]:.2f}°), FWHM = {popt[2]:.3f}°"
    )
    print(
        f"Fitted Peak 2: Center = {popt[5]:.3f}° (Truth = {true_params[5]:.2f}°), FWHM = {popt[6]:.3f}°"
    )

    # 4. Generate 2-panel publication figure
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Main spectrum plot
    ax_main.scatter(
        two_theta,
        measured_intensity,
        color="#2c3e50",
        alpha=0.6,
        s=18,
        label="Measured XRD Data",
        zorder=2,
    )
    ax_main.plot(
        two_theta,
        fit_curve,
        color="#e74c3c",
        lw=2.2,
        label=f"Global Fit (R² = {r2_score:.4f})",
        zorder=3,
    )
    ax_main.plot(
        two_theta,
        peak1_fit + bg_fit,
        "--",
        color="#2980b9",
        lw=1.8,
        label=f"Phase 1: 2θ={popt[1]:.2f}°",
    )
    ax_main.plot(
        two_theta,
        peak2_fit + bg_fit,
        "--",
        color="#27ae60",
        lw=1.8,
        label=f"Phase 2: 2θ={popt[5]:.2f}°",
    )
    ax_main.plot(two_theta, bg_fit, ":", color="#7f8c8d", lw=1.5, label="Linear Background")

    ax_main.set_ylabel("Diffraction Intensity (counts)", fontsize=11)
    ax_main.set_title("XRD 2θ-ω Scan Peak Deconvolution via Malthus-Optimize", fontsize=12, pad=10)
    ax_main.grid(True, linestyle="--", alpha=0.3)
    ax_main.legend(loc="upper right", fontsize=9.5)

    # Residuals plot
    ax_res.axhline(0, color="gray", linestyle="--", lw=1)
    ax_res.scatter(two_theta, residuals, color="#e74c3c", alpha=0.7, s=14)
    ax_res.set_ylabel("Residuals", fontsize=10)
    ax_res.set_xlabel("Diffraction Angle 2θ (degrees)", fontsize=11)
    ax_res.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved XRD peak fitting plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "01_xrd_peak_fitting.png"
    run_xrd_peak_fitting_showcase(plot_file)
