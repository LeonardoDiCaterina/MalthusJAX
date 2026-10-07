"""Tests for minimize API across multiple evolutionary strategies."""

from __future__ import annotations

import numpy as np

from malthus_optimize import OptimizeResult, minimize


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def test_minimize_cmaes() -> None:
    """Tests minimize with CMA-ES strategy."""
    res = minimize(
        sphere,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        method="cmaes",
        options={"popsize": 25, "maxiter": 40, "seed": 42},
    )
    assert isinstance(res, OptimizeResult)
    assert res.success is True
    assert np.allclose(res.x, 0.0, atol=1e-3)


def test_minimize_pso() -> None:
    """Tests minimize with Particle Swarm Optimization (PSO)."""
    res = minimize(
        sphere,
        bounds=[(-4.0, 4.0), (-4.0, 4.0)],
        method="pso",
        options={"popsize": 30, "maxiter": 50, "seed": 42},
    )
    assert isinstance(res, OptimizeResult)
    assert res.success is True
    assert np.allclose(res.x, 0.0, atol=1e-2)


def test_minimize_with_x0_only() -> None:
    """Tests minimize when only x0 initial guess is provided without explicit bounds."""
    res = minimize(
        sphere,
        x0=np.array([2.0, -3.0], dtype=np.float32),
        method="cmaes",
        options={"popsize": 25, "maxiter": 40, "seed": 42},
    )
    assert res.success is True
    assert np.allclose(res.x, 0.0, atol=1e-2)
