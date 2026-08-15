import jax.numpy as jnp

from malthusjax.core.genome.mo.sorting import (
    compute_dominance_matrix,
    compute_pareto_ranks,
)


def test_identical_fitnesses():
    """Test that identical fitness vectors do not dominate each other."""
    fitness = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    # In minimization, A dominates B if A <= B on all objs AND A < B on at least one.
    # Therefore, identical vectors DO NOT dominate each other.
    dom_matrix = compute_dominance_matrix(fitness, maximize=False)
    assert not jnp.any(dom_matrix)

    ranks = compute_pareto_ranks(dom_matrix)
    # They should all be rank 0
    assert jnp.all(ranks == 0)


def test_negative_fitnesses():
    """Ensure it works cleanly with negative values and infinity."""
    fitness = jnp.array([[-10.0, -10.0], [-5.0, -15.0], [jnp.inf, jnp.inf]])

    # Maximize = True
    # A = [-10, -10], B = [-5, -15]
    # A vs B: A is worse than B on obj0 (-10 < -5), A is better than B on obj1 (-10 > -15)
    # So they are non-dominated.
    # C is +inf, which dominates everyone if maximize=True.

    dom_matrix = compute_dominance_matrix(fitness, maximize=True)
    # C dominates A and B
    assert dom_matrix[2, 0]
    assert dom_matrix[2, 1]

    # A and B do not dominate each other
    assert not dom_matrix[0, 1]
    assert not dom_matrix[1, 0]

    ranks = compute_pareto_ranks(dom_matrix)
    # C is front 0, A and B are front 1
    assert ranks[2] == 0
    assert ranks[0] == 1
    assert ranks[1] == 1


def test_single_objective_fallback():
    """Test if the math falls back cleanly to 1D equivalent ranking."""
    fitness = jnp.array([[10.0], [20.0], [30.0], [40.0]])

    # Maximize = True
    dom_matrix = compute_dominance_matrix(fitness, maximize=True)
    ranks = compute_pareto_ranks(dom_matrix)

    # D(40) dominates everyone. C(30) dominates A,B. etc.
    # Ranks should be D:0, C:1, B:2, A:3
    assert ranks[3] == 0
    assert ranks[2] == 1
    assert ranks[1] == 2
    assert ranks[0] == 3
