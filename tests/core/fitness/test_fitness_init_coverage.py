"""Targeted coverage tests for malthusjax.core.fitness factories."""

import jax.numpy as jnp
import pytest

from malthusjax.core.fitness import (
    _create_bbob_evaluator,
    _create_binary_sum_evaluator,
    _create_knapsack_evaluator,
    _create_tsp_evaluator,
)


def test_create_knapsack_evaluator():
    ev1 = _create_knapsack_evaluator(_resolved_data={"source": "synthetic"})
    assert ev1 is not None

    ev2 = _create_knapsack_evaluator(weights=[1.0, 2.0], values=[3.0, 4.0], capacity=5.0)
    assert ev2 is not None


def test_create_binary_sum_evaluator():
    ev = _create_binary_sum_evaluator(maximize=True)
    assert ev is not None


def test_create_bbob_evaluator():
    ev = _create_bbob_evaluator(fn_name="sphere", dim=5, seed=42)
    assert ev is not None


def test_create_tsp_evaluator():
    # 1. Direct distance matrix
    dmat = jnp.zeros((4, 4))
    ev1 = _create_tsp_evaluator(distance_matrix=dmat)
    assert ev1 is not None

    # 2. Resolved data with distance_matrix dict
    ev2 = _create_tsp_evaluator(_resolved_data={"distance_matrix": [[0, 1], [1, 0]]})
    assert ev2 is not None

    # 3. Resolved data with synthetic
    ev3 = _create_tsp_evaluator(
        _resolved_data={"source": "synthetic", "num_cities": 5, "random_seed": 123}
    )
    assert ev3 is not None

    # 4. Fallback num_cities and seed
    ev4 = _create_tsp_evaluator(num_cities=6, seed=999)
    assert ev4 is not None
