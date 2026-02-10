import jax.numpy as jnp

from malthusjax.core.base import DistanceMetric
from malthusjax.core.genome.categorical_genome import (
    CategoricalGenome,
    CategoricalGenomeConfig,
    CategoricalPopulation,
)


def test_categorical_genome_init(rng_key):
    """Verifies single categorical genome initialization and category constraints."""
    config = CategoricalGenomeConfig(shape=(10,), num_categories=5)
    genome = CategoricalGenome.random_init(rng_key, config)

    assert isinstance(genome, CategoricalGenome)
    assert genome.values.shape == config.shape
    assert jnp.all(genome.values >= 0)
    assert jnp.all(genome.values < config.num_categories)


def test_categorical_population_soa(rng_key):
    """Verifies the SoA batching for categorical labels."""
    pop_size = 12
    config = CategoricalGenomeConfig(shape=(8,), num_categories=10)
    population = CategoricalPopulation.init_random(rng_key, config, size=pop_size)

    assert isinstance(population, CategoricalPopulation)
    assert population.genes.values.shape == (pop_size, 8)
    assert population.fitness.shape == (pop_size,)


def test_categorical_swap_positions(rng_key):
    """Tests the functional position-swapping logic."""
    config = CategoricalGenomeConfig(shape=(5,), num_categories=10)
    genome = CategoricalGenome.random_init(rng_key, config)

    pos1, pos2 = 0, 3
    val1, val2 = genome.values[pos1], genome.values[pos2]

    swapped_genome = genome.swap_positions(pos1, pos2)

    assert swapped_genome.values[pos1] == val2
    assert swapped_genome.values[pos2] == val1
    assert genome.values[pos1] == val1


def test_categorical_permutation_logic(rng_key):
    """Verifies permutation check and conversion logic."""
    config = CategoricalGenomeConfig(shape=(4,), num_categories=4)
    values = jnp.array([0, 1, 1, 2])
    genome = CategoricalGenome(values=values)

    assert not bool(genome.is_permutation())

    perm_genome = genome.to_permutation(config)
    assert bool(perm_genome.is_permutation())
    assert jnp.all(jnp.sort(perm_genome.values) == jnp.arange(4))


def test_categorical_distance_metrics(rng_key):
    """Tests Hamming vs Euclidean distance for discrete categories."""
    config = CategoricalGenomeConfig(shape=(5,), num_categories=10)
    g1 = CategoricalGenome.random_init(rng_key, config)
    g2_values = g1.values.at[0].set((g1.values[0] + 1) % config.num_categories)
    g2 = CategoricalGenome(values=g2_values)

    hamming_dist = g1.distance(g2, metric=DistanceMetric.HAMMING)
    euclidean_dist = g1.distance(g2, metric=DistanceMetric.EUCLIDEAN)

    assert int(hamming_dist) == 1
    assert euclidean_dist > 0


def test_categorical_autocorrect_clipping():
    """Verifies that out-of-range categories are corrected."""
    num_categories = 5
    config = CategoricalGenomeConfig(shape=(4,), num_categories=num_categories)

    broken_values = jnp.array([-1, 2, 5, 4])
    genome = CategoricalGenome(values=broken_values)

    corrected_genome = genome.autocorrect(config)

    expected = jnp.array([0, 2, 4, 4])
    assert jnp.all(corrected_genome.values == expected)
