import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.base import DistanceMetric
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


def test_real_genome_init(rng_key, real_genome_config):
    """Verifies single real genome initialization and bounds constraints."""
    genome = RealGenome.random_init(rng_key, real_genome_config)
    assert isinstance(genome, RealGenome)
    # Verify shape matches config
    assert genome.values.shape == real_genome_config.shape

    # Verify values are within configured bounds
    min_val, max_val = real_genome_config.bounds
    assert jnp.all(genome.values >= min_val)
    assert jnp.all(genome.values <= max_val)


def test_real_population_soa(real_population, real_genome_config):
    """Verifies the SoA batching for continuous real-valued vectors."""
    assert isinstance(real_population, RealPopulation)
    # Leading dimension should match the population size
    pop_size = len(real_population)
    assert real_population.genes.values.shape == (pop_size,) + real_genome_config.shape
    assert real_population.fitness.shape == (pop_size,)


def test_real_normalize_jit(real_population):
    """Verifies that vector normalization is JIT-stable."""
    genome = real_population[0]

    @jax.jit
    def normalize_genome(g):
        return g.normalize()

    normalized_genome = normalize_genome(genome)
    # The L2 norm should now be approximately 1.0
    norm = jnp.sqrt(jnp.sum(jnp.square(normalized_genome.values)))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_real_add_noise_immutability(rng_key, real_population):
    """Verifies that adding jitter maintains functional immutability."""
    genome = real_population[0]
    original_values = jnp.copy(genome.values)

    # Apply Gaussian noise
    noisy_genome = genome.add_noise(rng_key, noise_std=0.1)

    # Ensure values have actually changed
    assert not jnp.allclose(noisy_genome.values, genome.values)
    # Original genome must remain unchanged (immutable dataclass)
    assert jnp.all(genome.values == original_values)


def test_real_distance_metrics(real_population):
    """Tests Euclidean and Manhattan distances in continuous space."""
    g1 = real_population[0]
    g2 = real_population[1]

    euclidean_dist = g1.distance(g2, metric=DistanceMetric.EUCLIDEAN)
    manhattan_dist = g1.distance(g2, metric=DistanceMetric.MANHATTAN)

    assert euclidean_dist >= 0
    assert manhattan_dist >= 0

    # Verify Euclidean distance math (L2 norm)
    expected_l2 = jnp.sqrt(jnp.sum(jnp.square(g1.values - g2.values)))
    assert float(euclidean_dist) == pytest.approx(float(expected_l2))

    # Verify Manhattan distance math (L1 norm)
    expected_l1 = jnp.sum(jnp.abs(g1.values - g2.values))
    assert float(manhattan_dist) == pytest.approx(float(expected_l1))


def test_real_autocorrect_clipping(real_population, real_genome_config):
    """Verifies that out-of-bounds values are clipped correctly."""
    # Define specific testing bounds
    min_bound, max_bound = -1.0, 1.0
    temp_config = RealGenomeConfig(shape=(5,), bounds=(min_bound, max_bound))

    # Create values that far exceed the bounds (pop_size 10, length 5)
    broken_values = jnp.full((10, 5), 10.0)

    # Reconstruct population with broken values
    broken_pop = real_population.replace(genes=RealGenome(values=broken_values))

    # Apply autocorrect which uses jnp.clip internally
    corrected_pop = broken_pop.autocorrect(temp_config)

    assert jnp.all(corrected_pop.genes.values >= min_bound)
    assert jnp.all(corrected_pop.genes.values <= max_bound)
    assert jnp.all(corrected_pop.genes.values == 1.0)
