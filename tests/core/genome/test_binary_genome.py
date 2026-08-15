import chex
import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.core.base import DistanceMetric
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig


def test_binary_genome_bit_constraints(rng_key, binary_genome_config):
    """Validates binary value constraints and probability parameter."""
    genome = BinaryGenome.random_init(rng_key, binary_genome_config)

    chex.assert_shape(genome.values, binary_genome_config.shape)
    chex.assert_trees_all_close(
        jnp.logical_or(genome.values == 0, genome.values == 1), True, rtol=0.0, atol=0.0
    )


def test_binary_genome_probability_density():
    """Tests that initialization p parameter controls bit density."""
    key = jr.PRNGKey(42)
    shape = (1000,)

    for p in [0.1, 0.5, 0.9]:
        config = BinaryGenomeConfig(shape=shape, p=p)
        genome = BinaryGenome.random_init(key, config)

        empirical_p = jnp.mean(genome.values)
        chex.assert_trees_all_close(empirical_p, p, rtol=0.05, atol=0.05)


def test_binary_population_soa_integrity(binary_population, binary_genome_config):
    """Validates SoA batch structure and type consistency."""
    pop_size = len(binary_population)
    expected_shape = (pop_size,) + binary_genome_config.shape

    chex.assert_shape(binary_population.genes.values, expected_shape)
    chex.assert_shape(binary_population.fitness, (pop_size,))

    all_values_binary = jnp.logical_or(
        binary_population.genes.values == 0, binary_population.genes.values == 1
    )
    assert jnp.all(all_values_binary)


def test_binary_to_int_conversion_accuracy():
    """Tests decimal conversion accuracy and boundary cases."""
    # Known test vectors (MSB first by default)
    test_cases = [
        ([0, 0, 0, 0], 0),  # 0000 = 0
        ([0, 0, 0, 1], 1),  # 0001 = 1
        ([0, 0, 1, 0], 2),  # 0010 = 2
        ([0, 0, 1, 1], 3),  # 0011 = 3
        ([1, 1, 1, 1], 15),  # 1111 = 15
    ]

    for bits, expected_val in test_cases:
        genome = BinaryGenome(values=jnp.array(bits))
        actual_val = genome.to_int()
        chex.assert_trees_all_close(actual_val, expected_val, rtol=0.0, atol=0.0)

    single_bit_0 = BinaryGenome(values=jnp.array([0]))
    single_bit_1 = BinaryGenome(values=jnp.array([1]))
    assert single_bit_0.to_int() == 0
    assert single_bit_1.to_int() == 1


def test_binary_to_int_jit_equivalence(binary_population):
    """Verifies JIT vs non-JIT conversion equivalence."""
    genome = binary_population[0]

    @jax.jit
    def jit_to_int(g):
        return g.to_int()

    non_jit_result = genome.to_int()
    jit_result = jit_to_int(genome)

    chex.assert_trees_all_close(non_jit_result, jit_result, rtol=0.0, atol=0.0)


def test_binary_flip_bit_comprehensive():
    """Tests bit flipping across positions and boundary cases."""
    original = jnp.array([0, 1, 0, 1, 0])
    genome = BinaryGenome(values=original)

    for pos in range(len(original)):
        flipped = genome.flip_bit(pos)
        expected = original.at[pos].set(1 - original[pos])

        chex.assert_trees_all_close(flipped.values, expected, rtol=0.0, atol=0.0)
        chex.assert_trees_all_close(genome.values, original, rtol=0.0, atol=0.0)

    single_genome = BinaryGenome(values=jnp.array([1]))
    flipped_single = single_genome.flip_bit(0)
    chex.assert_trees_all_close(flipped_single.values, jnp.array([0]), rtol=0.0, atol=0.0)


def test_binary_distance_metric_formulas(binary_population):
    """Tests distance metrics against mathematical definitions."""
    g1, g2 = binary_population[0], binary_population[1]

    self_hamming = g1.distance(g1, metric=DistanceMetric.HAMMING)
    chex.assert_trees_all_close(self_hamming, 0.0, rtol=0.0, atol=0.0)

    hamming_dist = g1.distance(g2, metric=DistanceMetric.HAMMING)
    expected_hamming = jnp.sum(jnp.not_equal(g1.values, g2.values))
    chex.assert_trees_all_close(hamming_dist, expected_hamming, rtol=0.0, atol=0.0)

    euclidean_dist = g1.distance(g2, metric=DistanceMetric.EUCLIDEAN)
    expected_euclidean = jnp.sqrt(jnp.sum(jnp.square(g1.values - g2.values)))
    chex.assert_trees_all_close(euclidean_dist, expected_euclidean, rtol=1e-6, atol=1e-7)


def test_binary_autocorrect_boundary_handling():
    """Tests autocorrect with various out-of-bounds scenarios."""
    config = BinaryGenomeConfig(shape=(5,), p=0.5)

    test_cases = [
        ([-1.0, 0.0, 0.5, 1.0, 2.0], [0, 0, 0, 1, 1]),
        ([0.49, 0.51], [0, 0]),
        ([10.0, -10.0], [1, 0]),
        ([0.0, 1.0], [0, 1]),
    ]

    for input_vals, expected in test_cases:
        input_array = jnp.array(input_vals[: config.shape[0]])
        expected_array = jnp.array(expected[: config.shape[0]])

        genome = BinaryGenome(values=input_array)
        corrected = genome.autocorrect(config)

        chex.assert_trees_all_close(corrected.values, expected_array, rtol=0.0, atol=0.0)


def test_binary_genome_subscriptable_indexing():
    """Tests Pythonic indexing and iteration when enabled."""
    genome = BinaryGenome(values=jnp.array([1, 0, 1, 0, 1]), subscriptable=True)

    assert genome[0] == 1
    assert genome[-1] == 1
    assert jnp.array_equal(genome[1:3], jnp.array([0, 1]))

    collected = [bit for bit in genome]
    chex.assert_trees_all_close(jnp.array(collected), genome.values, rtol=0.0, atol=0.0)

    assert len(genome) == 5


def test_binary_genome_dtype_preservation():
    """Tests that binary operations preserve specified dtypes."""
    for dtype in [jnp.bool_, jnp.int8, jnp.int32]:
        config = BinaryGenomeConfig(shape=(5,), dtype=dtype, p=0.5)
        key = jr.PRNGKey(42)
        genome = BinaryGenome.random_init(key, config)

        assert genome.values.dtype == dtype

        flipped = genome.flip_bit(0)
        assert flipped.values.dtype == dtype

        corrected = genome.autocorrect(config)
        assert corrected.values.dtype == dtype


def test_binary_genome_edge_case_empty():
    """Tests edge case with empty binary genome."""
    empty_config = BinaryGenomeConfig(shape=(0,), p=0.5)
    key = jr.PRNGKey(42)

    empty_genome = BinaryGenome.random_init(key, empty_config)
    chex.assert_shape(empty_genome.values, (0,))

    assert empty_genome.to_int() == 0

    self_dist = empty_genome.distance(empty_genome, metric=DistanceMetric.HAMMING)
    chex.assert_trees_all_close(self_dist, 0.0, rtol=0.0, atol=0.0)
