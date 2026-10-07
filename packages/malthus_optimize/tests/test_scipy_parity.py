"""Tests for exact API and structural parity with scipy.optimize."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds as ScipyBounds

from malthus_optimize import OptimizeResult, differential_evolution, minimize


def test_optimize_result_parity() -> None:
    """Verifies OptimizeResult behaves identically to SciPy's OptimizeResult."""
    res = OptimizeResult(x=np.array([1.0, 2.0]), fun=0.0, success=True, nit=10, nfev=150)
    assert res.x[0] == 1.0
    assert res["x"][0] == 1.0
    assert res.fun == 0.0
    assert res["fun"] == 0.0
    assert res.success is True
    assert "x:" in repr(res)


def test_scipy_bounds_object_support() -> None:
    """Tests differential_evolution with scipy.optimize.Bounds object."""

    def quad(x: np.ndarray) -> float:
        return float(np.sum((x - 1.0) ** 2))

    bounds = ScipyBounds(lb=[-2.0, -2.0], ub=[4.0, 4.0])
    res = differential_evolution(quad, bounds=bounds, popsize=10, maxiter=25, seed=42)

    assert res.success is True
    assert np.allclose(res.x, [1.0, 1.0], atol=1e-2)


def test_minimize_signature_parity() -> None:
    """Tests minimize accepts the standard SciPy options dict and returns valid result."""

    def quad(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    res = minimize(
        quad,
        x0=np.array([1.0, -1.0]),
        method="de",
        options={"maxiter": 20, "popsize": 15, "seed": 42},
    )

    assert isinstance(res, OptimizeResult)
    assert hasattr(res, "x")
    assert hasattr(res, "fun")
    assert hasattr(res, "success")
    assert hasattr(res, "nit")
    assert hasattr(res, "nfev")
