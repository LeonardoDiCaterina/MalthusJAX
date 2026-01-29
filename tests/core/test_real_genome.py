import jax
import jax.numpy as jnp
from jax import random

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


def test_random_init_shape_and_bounds():
    cfg = RealGenomeConfig(shape=(5,), bounds=(-1.0, 2.0), dtype=jnp.float32)
    key = random.PRNGKey(0)

    g = RealGenome.random_init(key, cfg)
    assert g.values.shape == (5,)
    assert jnp.all(g.values >= cfg.bounds[0]) and jnp.all(g.values <= cfg.bounds[1])
    assert g.values.dtype == cfg.dtype


def test_create_population_and_init_random():
    cfg = RealGenomeConfig(shape=(3,), bounds=(-2.0, 2.0))
    key = random.PRNGKey(1)

    pop_genes = RealGenome.create_population(key, cfg, pop_size=7)
    assert pop_genes.values.shape == (7, 3)

    pop = RealPopulation.init_random(key, cfg, size=7)
    assert pop.genes.values.shape == (7, 3)
    assert pop.fitness.shape == (7,)
