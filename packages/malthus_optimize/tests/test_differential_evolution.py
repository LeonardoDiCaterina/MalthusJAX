"""Tests for differential_evolution API and convergence on standard benchmark surfaces."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from malthus_optimize import OptimizeResult, differential_evolution


def test_differential_evolution_sphere() -> None:
    """Tests DE optimization on the 3D Sphere function."""

    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    res = differential_evolution(
        sphere,
        bounds=[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
        popsize=15,
        maxiter=40,
        seed=42,
    )

    assert isinstance(res, OptimizeResult)
    assert res.success is True
    assert res.nit == 40
    assert res.nfev == 40 * (15 * 3)
    assert np.allclose(res.x, 0.0, atol=1e-3)
    assert res.fun < 1e-4


def test_differential_evolution_jax_function() -> None:
    """Tests pure JAX objective function vectorization without callback overhead."""

    def jax_sphere(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x**2)

    res = differential_evolution(
        jax_sphere,
        bounds=[(-3.0, 3.0), (-3.0, 3.0)],
        popsize=10,
        maxiter=30,
        seed=123,
    )

    assert res.success is True
    assert np.allclose(res.x, 0.0, atol=1e-3)


def test_differential_evolution_with_args() -> None:
    """Tests passing additional extra arguments (*args) to the objective."""

    def shifted_sphere(x: np.ndarray, offset: float, scale: float) -> float:
        return float(scale * np.sum((x - offset) ** 2))

    res = differential_evolution(
        shifted_sphere,
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        args=(2.5, 3.0),
        popsize=20,
        maxiter=50,
        seed=42,
    )

    assert np.allclose(res.x, [2.5, 2.5], atol=1e-2)
    assert res.fun < 1e-3


def test_differential_evolution_rosenbrock() -> None:
    """Tests DE on the non-convex 2D Rosenbrock function."""

    def rosenbrock(x: np.ndarray) -> float:
        return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)

    res = differential_evolution(
        rosenbrock,
        bounds=[(-3.0, 3.0), (-3.0, 3.0)],
        popsize=25,
        maxiter=80,
        seed=42,
    )

    assert np.allclose(res.x, [1.0, 1.0], atol=5e-2)
    assert res.fun < 0.1
