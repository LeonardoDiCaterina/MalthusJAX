import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


def test_binary_genome_index_iter_len():
    key = jr.PRNGKey(0)
    cfg = BinaryGenomeConfig(shape=(8,), dtype=jnp.bool_)
    g = BinaryGenome.random_init(key, cfg)

    assert len(g) == 8
    assert int(g[0]) == int(g.values[0])
    assert [int(x) for x in g] == [int(x) for x in g.values]

    pop = BinaryPopulation.init_random(jr.PRNGKey(1), cfg, size=4)
    first = pop[0]
    assert isinstance(first, BinaryGenome)
    assert int(first[0]) == int(pop.genes.values[0, 0])


def test_real_genome_index_iter_len():
    key = jr.PRNGKey(1)
    cfg = RealGenomeConfig(shape=(5,), bounds=(-1.0, 1.0))
    g = RealGenome.random_init(key, cfg)

    assert len(g) == 5
    assert jnp.isclose(g[0], g.values[0])
    assert [float(x) for x in g] == [float(x) for x in g.values]

    pop = RealPopulation.init_random(jr.PRNGKey(2), cfg, size=3)
    first = pop[0]
    assert isinstance(first, RealGenome)
    assert jnp.allclose(first[0], pop.genes.values[0, 0])


def test_disable_subscriptable_raises():
    key = jr.PRNGKey(3)
    cfg = RealGenomeConfig(shape=(3,), bounds=(-1.0, 1.0))
    g = RealGenome.random_init(key, cfg)
    g_disabled = g.replace(subscriptable=False)

    with pytest.raises(TypeError):
        _ = g_disabled[0]

    with pytest.raises(TypeError):
        for _ in g_disabled:
            pass
