import jax.numpy as jnp
import numpy as np
import pytest

from malthusjax.core.genome.mo.sorting import (
    compute_crowding_distance,
    compute_dominance_matrix,
    compute_pareto_ranks,
)


@pytest.fixture
def sample_fitness_min():
    """Sample fitness for a 2D minimization problem."""
    return jnp.array(
        [
            [1.0, 5.0],  # A: Non-dominated
            [2.0, 4.0],  # B: Non-dominated
            [3.0, 3.0],  # C: Non-dominated
            [5.0, 1.0],  # D: Non-dominated
            [4.0, 6.0],  # E: Dominated by B (2,4 < 4,6), C (3,3 < 4,6), A (1,5 < 4,6)
            [10.0, 10.0],  # F: Dominated by everyone
        ]
    )


def test_compute_dominance_matrix(sample_fitness_min):
    # Default is maximize=False (minimization)
    D = compute_dominance_matrix(sample_fitness_min)

    # A(0) should dominate E(4) and F(5)
    assert D[0, 4]
    assert D[0, 5]
    assert not D[0, 1]  # A does not dominate B

    # E(4) should not dominate anyone but F(5)
    assert not D[4, 0]
    assert D[4, 5]

    # F(5) dominates no one
    assert not jnp.any(D[5, :])


def test_compute_pareto_ranks(sample_fitness_min):
    D = compute_dominance_matrix(sample_fitness_min)
    ranks = compute_pareto_ranks(D)

    # Front 0: A, B, C, D
    assert ranks[0] == 0
    assert ranks[1] == 0
    assert ranks[2] == 0
    assert ranks[3] == 0

    # Front 1: E
    assert ranks[4] == 1

    # Front 2: F
    assert ranks[5] == 2


def test_compute_crowding_distance(sample_fitness_min):
    D = compute_dominance_matrix(sample_fitness_min)
    ranks = compute_pareto_ranks(D)
    cd = compute_crowding_distance(sample_fitness_min, ranks)

    # Normalize within Front 0 only: both objective ranges are 5 - 1 = 4.
    # Other fronts must not affect these distances.
    assert jnp.isinf(cd[0]) and jnp.isinf(cd[3])
    assert jnp.isinf(cd[4]) and jnp.isinf(cd[5])
    np.testing.assert_allclose(cd[1], 1.0, rtol=1e-5)
    np.testing.assert_allclose(cd[2], 1.5, rtol=1e-5)
