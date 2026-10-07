"""Validation test comparing malthus_gp against Kozax Genetic Programming."""

import numpy as np
import pytest

# Skip this entire module if kozax is not installed
pytest.importorskip("kozax")

from kozax.fitness_functions.SR_fitness_function import SymbolicRegressionFitnessFunction
from kozax.genetic_programming import GeneticProgramming
from malthus_gp.learn import CGPRegressor, DCGPRegressor


def test_kozax_vs_cgp_data_contract():
    """Verify that both Kozax and malthus_gp conform to standard input/output shapes."""
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = (X[:, 0] ** 2 + X[:, 0]).flatten()

    # 1. malthus_gp CGPRegressor
    cgp = CGPRegressor(n_rows=1, n_cols=10, pop_size=5, max_generations=10, random_state=42)
    cgp.fit(X, y)
    cgp_preds = cgp.predict(X)
    assert cgp_preds.shape == y.shape
    assert np.all(np.isfinite(cgp_preds))
    assert isinstance(cgp.to_formula(), str)

    # 2. Kozax GP
    fitness_fn = SymbolicRegressionFitnessFunction()
    kozax_gp = GeneticProgramming(
        num_generations=10,
        population_size=10,
        tournament_size=4,
        fitness_function=fitness_fn,
        max_nodes=15,
    )
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(42)
    kozax_gp.fit(
        key, (jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)), verbose=0
    )
    pareto_fitness, pareto_solutions = kozax_gp.pareto_front
    best_candidate = pareto_solutions[0]

    kozax_preds = jax.vmap(kozax_gp.tree_evaluator, in_axes=[None, 0])(
        best_candidate, jnp.asarray(X, dtype=jnp.float32)
    )[:, 0]
    kozax_preds_np = np.asarray(kozax_preds)
    assert kozax_preds_np.shape == y.shape
    assert np.all(np.isfinite(kozax_preds_np))


def test_kozax_vs_cgp_quartic_benchmark():
    """Compare CGP and Kozax on Koza-1 quartic polynomial: y = x^4 + x^3 + x^2 + x."""
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = (X[:, 0] ** 4 + X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0]).flatten()

    # Fit CGP
    cgp = CGPRegressor(n_rows=1, n_cols=25, pop_size=5, max_generations=150, random_state=42)
    cgp.fit(X, y)
    cgp_mse = float(np.mean((y - cgp.predict(X)) ** 2))

    # Fit Kozax
    fitness_fn = SymbolicRegressionFitnessFunction()
    kozax_gp = GeneticProgramming(
        num_generations=50,
        population_size=20,
        tournament_size=5,
        fitness_function=fitness_fn,
        max_nodes=20,
    )
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(42)
    kozax_gp.fit(
        key, (jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)), verbose=0
    )
    pareto_fitness, pareto_solutions = kozax_gp.pareto_front
    best_candidate = pareto_solutions[0]
    kozax_preds = jax.vmap(kozax_gp.tree_evaluator, in_axes=[None, 0])(
        best_candidate, jnp.asarray(X, dtype=jnp.float32)
    )[:, 0]
    kozax_mse = float(np.mean((y - np.asarray(kozax_preds)) ** 2))

    # Both should achieve reasonable non-divergent regression error
    assert np.isfinite(cgp_mse)
    assert np.isfinite(kozax_mse)
    assert cgp_mse < 2.0
    assert kozax_mse < 2.0


def test_dcgp_continuous_optimization():
    """Verify DCGPRegressor handles continuous real-valued polynomial coefficients via AutoDiff."""
    X = np.linspace(-1.0, 1.0, 25).reshape(-1, 1)
    y = (2.5 * X[:, 0] ** 2 - 1.3 * X[:, 0] + 0.7).flatten()

    dcgp = DCGPRegressor(
        n_rows=1,
        n_cols=20,
        pop_size=5,
        max_generations=80,
        lsmf_epochs=10,
        learning_rate=0.03,
        random_state=42,
    )
    dcgp.fit(X, y)
    preds = dcgp.predict(X)
    mse = float(np.mean((y - preds) ** 2))

    assert np.isfinite(mse)
    assert mse < 1.0
