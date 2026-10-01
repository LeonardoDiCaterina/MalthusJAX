"""Tests for curve_fit API on non-linear parameter estimation."""

from __future__ import annotations

import numpy as np

from malthus_optimize import curve_fit


def test_curve_fit_gaussian_peak() -> None:
    """Tests fitting a single Gaussian peak model."""

    def gaussian(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    xdata = np.linspace(-4, 4, 60)
    true_params = [4.0, 0.5, 0.8]
    y_clean = gaussian(xdata, *true_params)
    np.random.seed(42)
    y_noisy = y_clean + np.random.normal(0, 0.03, size=len(xdata))

    popt, pcov = curve_fit(
        gaussian,
        xdata,
        y_noisy,
        bounds=[(0.0, 8.0), (-2.0, 2.0), (0.2, 2.0)],
        pop_size=30,
        max_generations=50,
        seed=42,
    )

    assert popt.shape == (3,)
    assert pcov.shape == (3, 3)
    assert np.allclose(popt, true_params, atol=0.1)


def test_curve_fit_full_output() -> None:
    """Tests full_output=True returning convergence dictionary."""

    def linear_decay(x: np.ndarray, k: float, b: float) -> np.ndarray:
        return np.exp(-k * x) + b

    xdata = np.linspace(0, 5, 40)
    ydata = linear_decay(xdata, 0.8, 0.2)

    popt, pcov, info = curve_fit(
        linear_decay,
        xdata,
        ydata,
        bounds=[(0.1, 2.0), (0.0, 1.0)],
        full_output=True,
        pop_size=20,
        max_generations=30,
        seed=42,
    )

    assert "nfev" in info
    assert "nit" in info
    assert "history" in info
    assert np.allclose(popt, [0.8, 0.2], atol=0.05)
