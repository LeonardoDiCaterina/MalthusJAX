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

    # f_min = [1, 1], f_max = [10, 10], range = [9, 9]

    # A (0) and D (3) are boundary points of Front 0
    assert jnp.isinf(cd[0])
    assert jnp.isinf(cd[3])

    # E (4) and F (5) are the only points in their fronts, so they are both left and right boundaries
    assert jnp.isinf(cd[4])
    assert jnp.isinf(cd[5])

    # B (1) crowding distance:
    # Sorted Front 0 by obj 0: A(1,5), B(2,4), C(3,3), D(5,1)
    # obj 0 diff for B: C[0] - A[0] = 3 - 1 = 2 -> norm: 2 / 9
    # obj 1 diff for B: A[1] - C[1] = 5 - 3 = 2 -> norm: 2 / 9
    # Total for B: 4/9 ≈ 0.444
    expected_b = 4.0 / 9.0
    np.testing.assert_allclose(cd[1], expected_b, rtol=1e-5)

    # C (2) crowding distance:
    # obj 0 diff for C: D[0] - B[0] = 5 - 2 = 3 -> norm: 3 / 9
    # obj 1 diff for C: B[1] - D[1] = 4 - 1 = 3 -> norm: 3 / 9
    # Total for C: 6/9 ≈ 0.666
    expected_c = 6.0 / 9.0
    np.testing.assert_allclose(cd[2], expected_c, rtol=1e-5)
