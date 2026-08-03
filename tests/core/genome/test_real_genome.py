import chex
import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.core.base import DistanceMetric
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig


def test_real_genome_bounds_contract(rng_key, real_genome_config):
    """Validates bounds enforcement and shape consistency at initialization."""
    genome = RealGenome.random_init(rng_key, real_genome_config)

    chex.assert_shape(genome.values, real_genome_config.shape)
    min_val, max_val = real_genome_config.bounds
    chex.assert_trees_all_close(
        jnp.clip(genome.values, min_val, max_val), genome.values, rtol=0.0, atol=0.0
    )


def test_real_genome_dtype_consistency():
    """Verifies dtype preservation in float32 operations (JAX default)."""
    key = jr.PRNGKey(42)

    # Test with explicit float32 (JAX default without X64 mode)
    config = RealGenomeConfig(shape=(5,), bounds=(-1.0, 1.0), dtype=jnp.float32)
    genome = RealGenome.random_init(key, config)

    assert genome.values.dtype == jnp.float32

    # Operations should preserve dtype
    key, subkey = jr.split(key)
    noisy = genome.add_noise(subkey, noise_std=0.1)
    assert noisy.values.dtype == jnp.float32

    normalized = genome.normalize()
    assert normalized.values.dtype == jnp.float32


def test_real_population_soa_invariants(real_population, real_genome_config):
    """Validates SoA batch structure and PyTree consistency."""
    pop_size = len(real_population)
    expected_shape = (pop_size,) + real_genome_config.shape

    chex.assert_shape(real_population.genes.values, expected_shape)
    chex.assert_shape(real_population.fitness, (pop_size,))

    # PyTree structure preserved under vmap
    def extract_first_element(g):
        return g.values[0]

    vmapped_extract = jax.vmap(extract_first_element)
    first_elements = vmapped_extract(real_population.genes)
    chex.assert_shape(first_elements, (pop_size,))


def test_real_normalize_boundary_cases():
    """Tests normalization edge cases: zero vectors, unit vectors, numerical stability."""
    # Zero vector case
    zero_genome = RealGenome(values=jnp.zeros(5))
    normalized_zero = zero_genome.normalize()
    # Should handle gracefully (NaN or zero vector)
    assert jnp.isfinite(normalized_zero.values).any() or jnp.allclose(normalized_zero.values, 0.0)

    # Unit vector case (already normalized)
    unit_genome = RealGenome(values=jnp.array([1.0, 0.0, 0.0]))
    normalized_unit = unit_genome.normalize()
    chex.assert_trees_all_close(
        normalized_unit.values, jnp.array([1.0, 0.0, 0.0]), rtol=1e-6, atol=1e-7
    )

    # Large magnitude vector
    large_genome = RealGenome(values=jnp.array([1e6, 1e6]))
    normalized_large = large_genome.normalize()
    norm = jnp.linalg.norm(normalized_large.values)
    chex.assert_trees_all_close(norm, 1.0, rtol=1e-5, atol=1e-6)


def test_real_normalize_jit_equivalence(real_population):
    """Verifies JIT compilation preserves normalization accuracy."""
    genome = real_population[0]

    @jax.jit
    def jit_normalize(g):
        return g.normalize()

    non_jit_result = genome.normalize()
    jit_result = jit_normalize(genome)

    chex.assert_trees_all_close(non_jit_result.values, jit_result.values, rtol=1e-6, atol=1e-7)


def test_real_add_noise_statistics(real_population):
    """Validates noise magnitude and Gaussian properties."""
    genome = real_population[0]
    key = jr.PRNGKey(123)
    noise_std = 0.1

    # Generate many noisy samples
    keys = jr.split(key, 1000)
    noisy_samples = jax.vmap(lambda k: genome.add_noise(k, noise_std).values)(keys)

    # Noise should be approximately Gaussian with correct std
    noise_vectors = noisy_samples - genome.values
    empirical_std = jnp.std(noise_vectors, axis=0)

    # Allow 10% tolerance on standard deviation
    chex.assert_trees_all_close(
        empirical_std, jnp.full_like(empirical_std, noise_std), rtol=0.1, atol=0.01
    )

    # Test zero noise case
    zero_noise = genome.add_noise(key, noise_std=0.0)
    chex.assert_trees_all_close(zero_noise.values, genome.values, rtol=0.0, atol=0.0)


def test_real_distance_metric_properties(real_population):
    """Tests distance metric mathematical properties and edge cases."""
    g1, g2 = real_population[0], real_population[1]

    # Distance to self is zero
    chex.assert_trees_all_close(
        g1.distance(g1, metric=DistanceMetric.EUCLIDEAN), 0.0, rtol=1e-6, atol=1e-7
    )

    # Symmetry: d(a,b) = d(b,a)
    d_ab = g1.distance(g2, metric=DistanceMetric.EUCLIDEAN)
    d_ba = g2.distance(g1, metric=DistanceMetric.EUCLIDEAN)
    chex.assert_trees_all_close(d_ab, d_ba, rtol=1e-6, atol=1e-7)

    # Verify formulas
    euclidean_expected = jnp.linalg.norm(g1.values - g2.values)
    euclidean_actual = g1.distance(g2, metric=DistanceMetric.EUCLIDEAN)
    chex.assert_trees_all_close(euclidean_actual, euclidean_expected, rtol=1e-6, atol=1e-7)

    manhattan_expected = jnp.sum(jnp.abs(g1.values - g2.values))
    manhattan_actual = g1.distance(g2, metric=DistanceMetric.MANHATTAN)
    chex.assert_trees_all_close(manhattan_actual, manhattan_expected, rtol=1e-6, atol=1e-7)


def test_real_autocorrect_comprehensive():
    """Tests autocorrect clipping in all boundary configurations."""
    config = RealGenomeConfig(shape=(4,), bounds=(-1.0, 1.0))

    # Test all boundary cases
    test_cases = [
        ([-2.0, -1.0, 0.0, 1.0, 2.0], [-1.0, -1.0, 0.0, 1.0, 1.0]),  # Mixed boundaries
        ([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]),  # All upper overflow
        ([-2.0, -2.0, -2.0], [-1.0, -1.0, -1.0]),  # All lower overflow
        ([0.0, 0.5, -0.5], [0.0, 0.5, -0.5]),  # All within bounds
    ]

    for input_vals, expected_vals in test_cases:
        # Adjust shapes to match config
        input_array = jnp.array(input_vals[:4])
        expected_array = jnp.array(expected_vals[:4])

        genome = RealGenome(values=input_array)
        corrected = genome.autocorrect(config)

        chex.assert_trees_all_close(corrected.values, expected_array, rtol=0.0, atol=0.0)


def test_real_genome_immutability_chain():
    """Verifies functional immutability through chained operations."""
    key = jr.PRNGKey(456)
    genome = RealGenome(values=jnp.array([1.0, 2.0, 3.0]))
    original_values = genome.values.copy()

    # Chain multiple operations
    key1, key2 = jr.split(key)
    modified = genome.add_noise(key1, 0.1).normalize().add_noise(key2, 0.05)

    # Original should be unchanged
    chex.assert_trees_all_close(genome.values, original_values, rtol=0.0, atol=0.0)

    # Each step should create new instance
    assert modified is not genome
    assert not jnp.array_equal(modified.values, genome.values)


def test_real_genome_single_element_edge_case():
    """Tests edge case with shape=(1,) genomes."""
    key = jr.PRNGKey(789)
    config = RealGenomeConfig(shape=(1,), bounds=(-1.0, 1.0))
    genome = RealGenome.random_init(key, config)

    # Should work with single-element operations
    normalized = genome.normalize()
    chex.assert_trees_all_close(jnp.linalg.norm(normalized.values), 1.0, rtol=1e-5, atol=1e-6)

    # Distance to self
    self_dist = genome.distance(genome, metric=DistanceMetric.EUCLIDEAN)
    chex.assert_trees_all_close(self_dist, 0.0, rtol=1e-6, atol=1e-7)
