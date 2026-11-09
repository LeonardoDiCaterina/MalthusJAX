"""P5: front-local crowding against a small independent NumPy reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.mo.population import MOPopulation
from malthusjax.core.genome.mo.sorting import compute_crowding_distance
from malthusjax.core.genome.real_genome import RealGenome


def reference(fitness, ranks):
    f = np.asarray(fitness, dtype=np.float64)
    distances = np.zeros(len(f))
    for rank in np.unique(ranks):
        front = np.flatnonzero(ranks == rank)
        if len(front) <= 2:
            distances[front] = np.inf
            continue
        for objective in range(f.shape[1]):
            order = front[np.argsort(f[front, objective], kind="stable")]
            extent = f[order[-1], objective] - f[order[0], objective]
            if extent == 0:
                continue
            distances[order[[0, -1]]] = np.inf
            for i in range(1, len(order) - 1):
                distances[order[i]] += (
                    f[order[i + 1], objective] - f[order[i - 1], objective]
                ) / extent
    return distances


def test_audit_counterexample_is_front_local():
    fitness = jnp.array([[0.0, 10.0], [1.0, 6.0], [4.0, 3.0], [10.0, 0.0], [1000.0, 20.0]])
    ranks = jnp.array([0, 0, 0, 0, 1])
    result = compute_crowding_distance(fitness, ranks)
    np.testing.assert_allclose(result[1:3], [1.1, 1.5], rtol=1e-6)
    np.testing.assert_allclose(result, reference(fitness, np.asarray(ranks)))


@pytest.mark.parametrize("compiled", [False, True])
def test_dominated_front_cannot_change_actual_survivors(compiled):
    front = jnp.array([[0.0, 10.0], [1.0, 6.0], [4.0, 3.0], [10.0, 0.0]])

    def survivors(f):
        pop = BasePopulation(
            genes=RealGenome(values=jnp.arange(len(f))[:, None]), fitness=f, config=None, info={}
        )
        return MOPopulation.from_evaluated(pop).truncate(3).genes.values[:, 0]

    run = jax.jit(survivors) if compiled else survivors
    assert set(np.asarray(run(front))) == {0, 2, 3}
    assert set(np.asarray(run(jnp.concatenate([front, jnp.array([[1000.0, 20.0]])])))) == {0, 2, 3}


@pytest.mark.parametrize(
    "fitness,ranks",
    [
        (np.empty((0, 2)), np.array([], dtype=np.int32)),
        ([[2.0, 3.0]], [0]),
        ([[2.0, 3.0], [2.0, 3.0]], [0, 0]),
        ([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [0, 0, 0, 0]),
        ([[0.0, 4.0], [1.0, 4.0], [2.0, 4.0], [3.0, 4.0]], [0, 0, 0, 0]),
        (
            [[0.0, 3.0], [1.0, 2.0], [1.0, 2.0], [3.0, 0.0], [4.0, 4.0], [5.0, 5.0]],
            [0, 0, 0, 0, 7, 12],
        ),
    ],
)
def test_front_sizes_constants_duplicates_and_gapped_labels(fitness, ranks):
    fitness = jnp.asarray(fitness, dtype=jnp.float32)
    ranks = jnp.asarray(ranks, dtype=jnp.int32)
    expected = reference(fitness, np.asarray(ranks))
    for fn in (compute_crowding_distance, jax.jit(compute_crowding_distance)):
        np.testing.assert_allclose(fn(fitness, ranks), expected, rtol=1e-6)


def test_random_fronts_match_reference_and_rescaling():
    rng = np.random.default_rng(123)
    fitness = rng.normal(size=(21, 3)).astype(np.float32)
    ranks = np.repeat([0, 2, 8], 7).astype(np.int32)
    rng.shuffle(ranks)
    fn = jax.jit(compute_crowding_distance)
    expected = reference(fitness, ranks)
    np.testing.assert_allclose(fn(jnp.asarray(fitness), jnp.asarray(ranks)), expected, rtol=2e-6)
    np.testing.assert_allclose(
        fn(jnp.asarray(fitness * 1e-10), jnp.asarray(ranks)), expected, rtol=2e-6
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_are_unavailable_not_boundary_rewards(bad):
    fitness = jnp.array([[0.0, 2.0], [1.0, bad], [2.0, 0.0]])
    assert np.isnan(
        np.asarray(jax.jit(compute_crowding_distance)(fitness, jnp.zeros(3, dtype=jnp.int32)))
    ).all()


def test_negative_ranks_are_unavailable():
    assert np.isnan(
        np.asarray(compute_crowding_distance(jnp.ones((3, 2)), jnp.array([0, -1, 0])))
    ).all()


@pytest.mark.parametrize(
    "fitness,ranks",
    [
        (jnp.ones(3), jnp.zeros(3, dtype=jnp.int32)),
        (jnp.ones((3, 0)), jnp.zeros(3, dtype=jnp.int32)),
        (jnp.ones((3, 2)), jnp.zeros(2, dtype=jnp.int32)),
        (jnp.ones((3, 2)), jnp.zeros(3)),
    ],
)
def test_invalid_shapes_and_rank_dtype_raise(fitness, ranks):
    with pytest.raises(ValueError):
        jax.jit(compute_crowding_distance)(fitness, ranks)


def test_large_finite_ranges_do_not_overflow_normalization():
    high = np.finfo(np.float32).max
    fitness = jnp.array([[-high, high], [0.0, 0.0], [high, -high]], dtype=jnp.float32)
    ranks = jnp.zeros(3, dtype=jnp.int32)
    result = jax.jit(compute_crowding_distance)(fitness, ranks)
    np.testing.assert_allclose(result, reference(fitness, np.asarray(ranks)), rtol=1e-6)
    assert result[1] == 2.0
