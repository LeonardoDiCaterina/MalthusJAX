"""03_kinetic_modeling.py.

Demonstrates non-linear parameter estimation for chemical kinetics and Arrhenius activation energy
using malthus_optimize.curve_fit.
Visualizes temperature-dependent concentration curves and the extracted Arrhenius rate behavior.
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


def arrhenius_decay(
    t: np.ndarray,
    c0: float,
    pre_exp: float,
    ea_over_r: float,
    temperature: float,
) -> np.ndarray:
    """Concentration profile C(t) = C0 * exp(-k(T) * t) with k(T) = A * exp(-Ea / (R * T))."""
    k = pre_exp * np.exp(-ea_over_r / temperature)
    return c0 * np.exp(-k * t)


def run_kinetic_showcase(output_plot: Path) -> None:
    print("=" * 75)
    print("  Malthus-Optimize Showcase 3: Non-Linear Kinetic & Arrhenius Parameter Estimation")
    print("=" * 75)

    np.random.seed(42)
    temperatures = [300.0, 320.0, 340.0, 360.0]  # Kelvin
    t_points = np.linspace(0.0, 60.0, 30).astype(np.float32)  # minutes

    # True physical kinetic constants
    true_c0 = 10.0  # Initial concentration (mol/L)
    true_pre_exp = 5.0e4  # Pre-exponential factor A (1/min)
    true_ea_over_r = 3800.0  # Ea / R (Kelvin)

    print(f"Temperatures: {temperatures} K")
    print(
        f"True Parameters: C0={true_c0} mol/L, A={true_pre_exp:.1e} 1/min, Ea/R={true_ea_over_r:.0f} K"
    )

    # Generate synthetic multi-temperature observations
    all_times = []
    all_temps = []
    all_concentrations = []

    for temp in temperatures:
        c_clean = arrhenius_decay(t_points, true_c0, true_pre_exp, true_ea_over_r, temp)
        c_noisy = np.maximum(c_clean + np.random.normal(0, 0.25, size=len(t_points)), 0.0)
        all_times.append(t_points)
        all_temps.append(np.full_like(t_points, temp))
        all_concentrations.append(c_noisy)

    time_concat = np.concatenate(all_times).astype(np.float32)
    temp_concat = np.concatenate(all_temps).astype(np.float32)
    conc_concat = np.concatenate(all_concentrations).astype(np.float32)

    # Multi-variable model wrapper: xdata is 2D (time, temp)
    def multi_temp_model(x_data: np.ndarray, c0: float, a: float, ea_r: float) -> np.ndarray:
        t_arr, t_kelvin = x_data[:, 0], x_data[:, 1]
        k = a * np.exp(-ea_r / t_kelvin)
        return c0 * np.exp(-k * t_arr)

    x_input = np.column_stack([time_concat, temp_concat])
    bounds = [(5.0, 15.0), (1.0e3, 5.0e5), (1000.0, 8000.0)]

    # Fit parameters via curve_fit
    t0 = time.perf_counter()
    popt, pcov = curve_fit(
        multi_temp_model,
        x_input,
        conc_concat,
        bounds=bounds,
        pop_size=40,
        max_generations=100,
        seed=42,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nOptimization Time: {elapsed:.3f} s")
    print(f"Fitted C0:   {popt[0]:.2f} mol/L (Truth = {true_c0:.2f})")
    print(f"Fitted A:    {popt[1]:.2e} 1/min (Truth = {true_pre_exp:.2e})")
    print(f"Fitted Ea/R: {popt[2]:.1f} K (Truth = {true_ea_over_r:.1f})")

    # Render 2-panel publication figure
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel A: Concentration Decay Curves
    colors = ["#2980b9", "#27ae60", "#e67e22", "#e74c3c"]
    dense_t = np.linspace(0.0, 60.0, 100)

    for i, temp in enumerate(temperatures):
        mask = temp_concat == temp
        axes[0].scatter(time_concat[mask], conc_concat[mask], color=colors[i], alpha=0.6, s=20)
        c_pred = arrhenius_decay(dense_t, popt[0], popt[1], popt[2], temp)
        axes[0].plot(dense_t, c_pred, color=colors[i], lw=2.0, label=f"T = {temp:.0f} K")

    axes[0].set_title("Multi-Temperature Concentration Decay Kinetics", fontsize=11)
    axes[0].set_xlabel("Time (minutes)", fontsize=10)
    axes[0].set_ylabel("Concentration C(t) (mol/L)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=9.5)

    # Panel B: Arrhenius Linearization Plot (ln k vs 1000 / T)
    inv_t = 1000.0 / np.array(temperatures)
    k_true = true_pre_exp * np.exp(-true_ea_over_r / np.array(temperatures))
    k_fit = popt[1] * np.exp(-popt[2] / np.array(temperatures))

    axes[1].plot(inv_t, np.log(k_true), "k--", lw=1.5, label="Analytical Arrhenius Slope")
    axes[1].scatter(
        inv_t,
        np.log(k_fit),
        color="#8e44ad",
        s=60,
        zorder=3,
        label=f"Fitted k(T) (Ea/R={popt[2]:.0f} K)",
    )
    axes[1].set_title("Arrhenius Activation Energy Linearization", fontsize=11)
    axes[1].set_xlabel("Inverse Temperature 1000 / T (1/K)", fontsize=10)
    axes[1].set_ylabel("ln(Rate Constant k) [1/min]", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\n[+] Saved chemical kinetics plot to: {output_plot}")


if __name__ == "__main__":
    plot_file = ROOT_DIR / "examples" / "plots" / "03_kinetic_modeling.png"
    run_kinetic_showcase(plot_file)
