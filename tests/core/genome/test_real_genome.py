import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.base import DistanceMetric
from malthusjax.core.genome.real_genome import RealGenome, RealPopulation


def test_real_genome_init(rng_key, real_genome_config):
    """Verifies single genome initialization and shapes."""
    genome = RealGenome.random_init(rng_key, real_genome_config)
    assert isinstance(genome, RealGenome)
    assert genome.values.shape == (real_genome_config.length,)
    assert jnp.all(genome.values >= real_genome_config.bounds[0])
    assert jnp.all(genome.values <= real_genome_config.bounds[1])


def test_real_population_soa(real_population, real_genome_config):
    """Verifies that the population follows the Struct-of-Arrays pattern."""
    # Ensure real_population is actually a RealPopulation, not a legacy tuple
    if isinstance(real_population, tuple):
        pytest.fail(
            "conftest.py is still returning a tuple for real_population. "
            "Please ensure the RealPopulation.init_random version is saved."
        )

    assert isinstance(real_population, RealPopulation)
    assert real_population.genes.values.shape == (10, real_genome_config.length)
    assert real_population.fitness.shape == (10,)


def test_real_indexing(real_population):
    """Verifies that indexing returns a single Genome vs a Sub-Population."""
    # If the fixture is a tuple from the old conftest, we handle it or fail
    if isinstance(real_population, tuple):
        values, config = real_population
        real_population = RealPopulation(
            genes=RealGenome(values=values), fitness=jnp.zeros(values.shape[0]), config=config
        )

    # Indexing an integer should return a single RealGenome
    individual = real_population[0]
    assert isinstance(individual, RealGenome)
    assert individual.values.ndim == 1

    # Indexing a slice should return a RealPopulation sub-batch
    sub_pop = real_population[0:3]
    assert isinstance(sub_pop, RealPopulation)
    assert len(sub_pop) == 3


def test_real_distance_jit(real_population):
    """Verifies that the distance method is JIT-compatible."""
    if isinstance(real_population, tuple):
        values, config = real_population
        real_population = RealPopulation(
            genes=RealGenome(values=values), fitness=jnp.zeros(values.shape[0]), config=config
        )

    g1 = real_population[0]
    g2 = real_population[1]

    @jax.jit
    def calc_dist(a, b):
        return a.distance(b, metric=DistanceMetric.EUCLIDEAN)

    dist = calc_dist(g1, g2)
    assert dist >= 0
    assert not jnp.isnan(dist)


def test_real_autocorrect_vectorization(real_population, real_genome_config):
    """Verifies autocorrect works across an entire population batch."""
    if isinstance(real_population, tuple):
        values, config = real_population
        real_population = RealPopulation(
            genes=RealGenome(values=values), fitness=jnp.zeros(values.shape[0]), config=config
        )

    # Create manually broken values (all 100.0)
    broken_values = jnp.full_like(real_population.genes.values, 100.0)
    # Reconstruct population with broken values
    broken_pop = real_population.replace(genes=RealGenome(values=broken_values))

    # Corrected should be within bounds (-5.0, 5.0)
    corrected_pop = broken_pop.autocorrect(real_genome_config)
    assert jnp.all(corrected_pop.genes.values <= 5.0)


if __name__ == "__main__":
    pytest.main()
