import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.base import DistanceMetric
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation


def test_binary_genome_init(rng_key, binary_genome_config):
    """Verifies single binary genome initialization and bit constraints."""
    genome = BinaryGenome.random_init(rng_key, binary_genome_config)
    assert isinstance(genome, BinaryGenome)
    assert genome.bits.shape == (binary_genome_config.length,)
    # Verify values are only 0 or 1
    assert jnp.all((genome.bits == 0) | (genome.bits == 1))


def test_binary_population_soa(binary_population, binary_genome_config):
    """Verifies the SoA batching for binary strings."""
    # Ensure binary_population is actually a BinaryPopulation
    assert isinstance(binary_population, BinaryPopulation)
    # Batch dimension should match the fixture (size 10, length 10)
    assert binary_population.genes.bits.shape == (10, binary_genome_config.length)
    assert binary_population.fitness.shape == (10,)


def test_binary_to_int_jit(binary_population):
    """Verifies that decimal conversion is JIT-stable."""
    genome = binary_population[0]

    @jax.jit
    def get_val(g):
        return g.to_int()

    val = get_val(genome)
    assert val >= 0
    # For length 10, max value is 1023
    assert val < 1024


def test_binary_flip_bit(binary_population):
    """Tests the functional bit-flipping logic."""
    genome = binary_population[0]
    original_bit = genome.bits[0]

    # Flip the first bit
    flipped_genome = genome.flip_bit(0)
    assert flipped_genome.bits[0] == 1 - original_bit

    # Verify immutability (original shouldn't change)
    assert genome.bits[0] == original_bit


def test_binary_distance_metrics(binary_population):
    """Tests Hamming vs Euclidean distance in binary space."""
    g1 = binary_population[0]
    g2 = binary_population[1]

    hamming_dist = g1.distance(g2, metric=DistanceMetric.HAMMING)
    euclidean_dist = g1.distance(g2, metric=DistanceMetric.EUCLIDEAN)

    assert hamming_dist >= 0
    assert euclidean_dist >= 0
    # In binary space, Hamming is often larger than or equal to L2 squared
    assert float(hamming_dist) == pytest.approx(float(jnp.sum(jnp.square(g1.bits - g2.bits))))


def test_binary_autocorrect_clipping(binary_population, binary_genome_config):
    """Verifies that non-binary values are corrected to [0, 1]."""
    # Create manually broken bits with float values that should be clipped
    broken_bits = jnp.array([[2, -1, 0.5, 1, 0]] * 10)

    # Update a temp config to match the broken shape (length 5)
    temp_config = BinaryGenomeConfig(length=5)

    # Reconstruct population with broken bits
    broken_pop = binary_population.replace(genes=BinaryGenome(bits=broken_bits))

    corrected_pop = broken_pop.autocorrect(temp_config)
    assert jnp.all((corrected_pop.genes.bits == 0) | (corrected_pop.genes.bits == 1))
