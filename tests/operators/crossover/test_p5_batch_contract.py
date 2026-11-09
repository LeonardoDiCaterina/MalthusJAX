"""P5: inferred pair counts and exact key budgets in both batch paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.operators.crossover.real import BlendCrossover, UniformCrossover


def parents(n=3):
    return jnp.arange(n * 4, dtype=jnp.float32).reshape(n, 4), -jnp.ones((n, 4))


@pytest.mark.parametrize("operator", [UniformCrossover, BlendCrossover])
@pytest.mark.parametrize(
    "typed,configured,compiled",
    [(False, False, False), (True, False, True), (False, True, True), (True, True, False)],
)
def test_default_and_configured_fast_path_matches_population_path(
    operator, typed, configured, compiled
):
    left, right = parents()
    cfg = RealGenomeConfig(shape=(4,), bounds=(-20.0, 20.0))
    op = operator(input_length=3 if configured else -1)
    key = jax.random.key(42) if typed else jax.random.PRNGKey(42)
    keys = jax.random.split(key, op.num_keys((3,)))
    if configured:
        keys = keys.reshape((3, 1, op.num_keys_per_atomic_operation) + (() if typed else (2,)))

    def population(x):
        return BasePopulation(genes=RealGenome(values=x), fitness=jnp.zeros(3), config=cfg, info={})

    expected = op(keys, population(left), population(right), cfg).genes.values

    def fn(a, b, k):
        return op.apply_fastpath(k, a, b, cfg)

    result = (jax.jit(fn) if compiled else fn)(left, right, keys)
    if operator is UniformCrossover:
        np.testing.assert_array_equal(result, expected)
    else:
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("path", ["fast", "population"])
@pytest.mark.parametrize("problem", ["parents", "configured", "keys", "dtype"])
def test_bad_parent_counts_shapes_and_key_budgets_raise(path, problem):
    left, right = parents()
    cfg = RealGenomeConfig(shape=(4,))
    op = UniformCrossover(input_length=4 if problem == "configured" else -1)
    keys = jax.random.split(jax.random.key(1), 2 if problem == "keys" else 3)
    if problem == "parents":
        right = jnp.ones((2, 4))
    if problem == "dtype":
        keys = jnp.zeros((3, 2), dtype=jnp.float32)

    def run(a, b, k):
        if path == "fast":
            return op.apply_fastpath(k, a, b, cfg)

        def pop(x):
            return BasePopulation(
                genes=RealGenome(values=x), fitness=jnp.zeros(len(x)), config=cfg, info={}
            )

        return op(k, pop(a), pop(b), cfg)

    with pytest.raises(ValueError):
        jax.jit(run)(left, right, keys)


def test_empty_batch_and_multiple_offspring():
    cfg = RealGenomeConfig(shape=(4,))
    op = UniformCrossover()
    empty = jnp.empty((0, 4))
    keys = jax.random.split(jax.random.key(0), 0)
    assert jax.jit(lambda a, k: op.apply_fastpath(k, a, a, cfg))(empty, keys).shape == (0, 4)
    left, right = parents()
    op = UniformCrossover(num_offspring=2)
    keys = jax.random.split(jax.random.key(0), op.num_keys((3,)))

    def pop(x):
        return BasePopulation(genes=RealGenome(values=x), fitness=jnp.zeros(3), config=cfg, info={})

    result = jax.jit(lambda k: op(k, pop(left), pop(right), cfg))(keys)
    assert result.genes.values.shape == (6, 4)
    with pytest.raises(NotImplementedError):
        op.apply_fastpath(keys, left, right, cfg)


@pytest.mark.parametrize("typed", [False, True])
def test_structured_genomes_and_zero_key_operations(typed):
    from flax import struct

    from malthusjax.operators.base import BaseCrossover

    @struct.dataclass
    class Genome:
        first: object
        second: object

    @struct.dataclass
    class MeanCrossover(BaseCrossover):
        @property
        def num_keys_per_atomic_operation(self):
            return 0

        def _generate_noise(self, keys, config, generation=0):
            return None

        def _recombine_one(self, first, second, noise, config, **kwargs):
            return jax.tree_util.tree_map(lambda a, b: (a + b) / 2, first, second)

    left = Genome(first=jnp.arange(6.0).reshape(3, 2), second=jnp.ones((3, 1)))
    right = jax.tree_util.tree_map(lambda a: a + 2, left)

    def pop(genes):
        return BasePopulation(genes=genes, fitness=jnp.zeros(3), config=None, info={})

    key = jax.random.key(7) if typed else jax.random.PRNGKey(7)
    keys = jax.random.split(key, 0)
    op = MeanCrossover(num_offspring=2)
    run = jax.jit(lambda a, b: op(keys, pop(a), pop(b), None))
    result = run(left, right)
    for source, actual in zip(
        jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(result.genes)
    ):
        np.testing.assert_array_equal(actual, jnp.repeat(source + 1, 2, axis=0))
    with pytest.raises(ValueError, match="shapes|batch"):
        run(left.replace(second=jnp.ones((4, 1))), right.replace(second=jnp.ones((4, 1))))
    with pytest.raises(ValueError, match="PyTree"):
        run(left, RealGenome(values=jnp.ones((3, 2))))


def test_parent_width_and_excess_keys_are_rejected():
    op = UniformCrossover()
    cfg = RealGenomeConfig(shape=(4,))
    left, right = parents()
    keys = jax.random.split(jax.random.key(9), 3)
    with pytest.raises(ValueError, match="shapes"):
        op.apply_fastpath(keys, left, right[:, :3], cfg)
    with pytest.raises(ValueError, match="exactly 3 keys"):
        op.apply_fastpath(jax.random.split(jax.random.key(9), 4), left, right, cfg)


def test_matching_size_numeric_arrays_are_not_typed_prng_keys():
    left, right = parents()
    cfg = RealGenomeConfig(shape=(4,))
    with pytest.raises(ValueError, match="typed PRNG keys"):
        UniformCrossover().apply_fastpath(jnp.zeros(3), left, right, cfg)
