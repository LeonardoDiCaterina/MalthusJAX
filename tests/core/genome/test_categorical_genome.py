import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.base import DistanceMetric
from malthusjax.core.genome.categorical_genome import CategoricalGenome, CategoricalPopulation


def test_categorical_genome_init(rng_key, categorical_genome_config):
    """Verifies single categorical genome initialization and category bounds."""
    genome = CategoricalGenome.random_init(rng_key, categorical_genome_config)
    assert isinstance(genome, CategoricalGenome)
    assert genome.categories.shape == (categorical_genome_config.length,)
    # Verify values are within [0, num_categories - 1]
    assert jnp.all(genome.categories >= 0)
    assert jnp.all(genome.categories < categorical_genome_config.num_categories)


def test_categorical_population_soa(categorical_population, categorical_genome_config):
    """Verifies the SoA batching for categorical choice genomes."""
    assert isinstance(categorical_population, CategoricalPopulation)
    # Batch dimension (10, 8) based on conftest fixtures
    assert categorical_population.genes.categories.shape == (10, categorical_genome_config.length)
    assert categorical_population.fitness.shape == (10,)


def test_categorical_distance_hamming(categorical_population):
    """Tests Hamming distance (mismatch count) for categorical labels."""
    g1 = categorical_population[0]
    g2 = categorical_population[1]

    # Standard categorical distance is the number of differing elements
    dist = g1.distance(g2, metric=DistanceMetric.HAMMING)

    manual_dist = jnp.sum(g1.categories != g2.categories)
    assert float(dist) == pytest.approx(float(manual_dist))


def test_permutation_logic_jit(rng_key, categorical_genome_config):
    """Verifies that permutation checks and conversions are JIT-stable."""
    # Create a guaranteed permutation [0, 1, 2, ...]
    length = categorical_genome_config.length
    perm_values = jnp.arange(length)
    genome = CategoricalGenome(categories=perm_values)

    @jax.jit
    def check_perm(g):
        return g.is_permutation(), g.to_permutation(categorical_genome_config)

    is_p, new_p = check_perm(genome)
    assert is_p
    assert jnp.all(new_p.categories == perm_values)


def test_categorical_swap_positions(categorical_population):
    """Tests the functional swapping logic used in combinatorial search."""
    genome = categorical_population[0]
    val_at_0 = genome.categories[0]
    val_at_1 = genome.categories[1]

    # Swap first two positions
    swapped = genome.swap_positions(0, 1)
    assert swapped.categories[0] == val_at_1
    assert swapped.categories[1] == val_at_0

    # Ensure original remains unchanged (immutability)
    assert genome.categories[0] == val_at_0


def test_categorical_autocorrect_wrapping(categorical_population, categorical_genome_config):
    """Verifies that out-of-range labels are clipped back to valid categories."""
    # Inject values beyond num_categories
    broken_cats = jnp.full_like(categorical_population.genes.categories, 99)
    broken_pop = categorical_population.replace(genes=CategoricalGenome(categories=broken_cats))

    corrected_pop = broken_pop.autocorrect(categorical_genome_config)
    assert jnp.all(corrected_pop.genes.categories == categorical_genome_config.num_categories - 1)
