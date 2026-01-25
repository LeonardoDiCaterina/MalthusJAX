"""
Tests for core genome implementations (Level 1).

Tests binary, real, and categorical genomes for correctness,
JAX compatibility, and tensorization.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig

from ..conftest import (
    assert_valid_binary_genome,
    assert_valid_categorical_genome,
    assert_valid_real_genome,
)


class TestBinaryGenome:
    """Test binary genome implementation."""

    def test_init_from_bits(self, binary_genome_config):
        """Test manual initialization from bits."""
        bits = jnp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        genome = BinaryGenome(bits=bits)
        assert_valid_binary_genome(genome, binary_genome_config)
        assert jnp.array_equal(genome.bits, bits)

    def test_random_init(self, rng_key, binary_genome_config):
        """Test random initialization."""
        genome = BinaryGenome.random_init(rng_key, binary_genome_config)
        assert_valid_binary_genome(genome, binary_genome_config)

    def test_random_init_deterministic(self, rng_key, binary_genome_config):
        """Test that random init is deterministic given same key."""
        genome1 = BinaryGenome.random_init(rng_key, binary_genome_config)
        genome2 = BinaryGenome.random_init(rng_key, binary_genome_config)
        assert jnp.array_equal(genome1.bits, genome2.bits)

    def test_random_init_different_keys(self, rng_key, binary_genome_config):
        """Test that different keys produce different genomes."""
        key1, key2 = jr.split(rng_key)
        genome1 = BinaryGenome.random_init(key1, binary_genome_config)
        genome2 = BinaryGenome.random_init(key2, binary_genome_config)
        # Very unlikely to be identical for length > 5
        if binary_genome_config.length > 5:
            assert not jnp.array_equal(genome1.bits, genome2.bits)

    @pytest.mark.jit
    def test_jit_compatibility(self, rng_key, binary_genome_config):
        """Test that random_init can be JIT compiled."""
        # NEW paradigm: JIT compile the class method
        jit_init = jax.jit(BinaryGenome.random_init, static_argnames=['config'])

        genome = jit_init(rng_key, binary_genome_config)
        assert_valid_binary_genome(genome, binary_genome_config)

    def test_str_representation(self, binary_genome):
        """Test string representation."""
        str_repr = str(binary_genome)
        assert "BinaryGenome(" in str_repr
        assert f"len={len(binary_genome.bits)}" in str_repr

    def test_different_sizes(self, rng_key, binary_size_config):
        """Test genomes of different sizes."""
        genome = BinaryGenome.random_init(rng_key, binary_size_config)
        assert_valid_binary_genome(genome, binary_size_config)


class TestRealGenome:
    """Test real genome implementation."""

    def test_init_from_values(self, real_genome_config):
        """Test manual initialization from values."""
        values = jnp.array([1.0, -2.0, 3.5, -1.5, 0.0])
        genome = RealGenome(values=values)
        assert_valid_real_genome(genome, real_genome_config)
        assert jnp.array_equal(genome.values, values)

    def test_random_init(self, rng_key, real_genome_config):
        """Test random initialization."""
        genome = RealGenome.random_init(rng_key, real_genome_config)
        assert_valid_real_genome(genome, real_genome_config)

    def test_random_init_deterministic(self, rng_key, real_genome_config):
        """Test that random init is deterministic given same key."""
        genome1 = RealGenome.random_init(rng_key, real_genome_config)
        genome2 = RealGenome.random_init(rng_key, real_genome_config)
        assert jnp.allclose(genome1.values, genome2.values)

    def test_random_init_different_keys(self, rng_key, real_genome_config):
        """Test that different keys produce different genomes."""
        key1, key2 = jr.split(rng_key)
        genome1 = RealGenome.random_init(key1, real_genome_config)
        genome2 = RealGenome.random_init(key2, real_genome_config)
        # Very unlikely to be identical
        assert not jnp.allclose(genome1.values, genome2.values, atol=1e-6)

    def test_bounds_respected(self, rng_key, constrained_real_genome_config):
        """Test that bounds are respected in random initialization."""
        # Generate many samples to test bounds
        for i in range(10):
            key = jr.fold_in(rng_key, i)
            genome = RealGenome.random_init(key, constrained_real_genome_config)
            assert_valid_real_genome(genome, constrained_real_genome_config)

    @pytest.mark.jit
    def test_jit_compatibility(self, rng_key, real_genome_config):
        """Test that random_init can be JIT compiled."""
        # NEW paradigm: JIT compile the class method
        jit_init = jax.jit(RealGenome.random_init, static_argnames=['config'])

        genome = jit_init(rng_key, real_genome_config)
        assert_valid_real_genome(genome, real_genome_config)

    def test_str_representation(self, real_genome):
        """Test string representation."""
        str_repr = str(real_genome)
        assert "RealGenome(" in str_repr
        assert f"len={len(real_genome.values)}" in str_repr

    def test_different_sizes(self, rng_key, real_size_config):
        """Test genomes of different sizes."""
        genome = RealGenome.random_init(rng_key, real_size_config)
        assert_valid_real_genome(genome, real_size_config)


class TestCategoricalGenome:
    """Test categorical genome implementation."""

    def test_init_from_categories(self, categorical_genome_config):
        """Test manual initialization from categories."""
        categories = jnp.array([0, 2, 1, 3, 0, 1, 2, 3])
        genome = CategoricalGenome(categories=categories)
        assert_valid_categorical_genome(genome, categorical_genome_config)
        assert jnp.array_equal(genome.categories, categories)

    def test_random_init(self, rng_key, categorical_genome_config):
        """Test random initialization."""
        genome = CategoricalGenome.random_init(rng_key, categorical_genome_config)
        assert_valid_categorical_genome(genome, categorical_genome_config)

    def test_random_init_deterministic(self, rng_key, categorical_genome_config):
        """Test that random init is deterministic given same key."""
        genome1 = CategoricalGenome.random_init(rng_key, categorical_genome_config)
        genome2 = CategoricalGenome.random_init(rng_key, categorical_genome_config)
        assert jnp.array_equal(genome1.categories, genome2.categories)

    def test_random_init_different_keys(self, rng_key, categorical_genome_config):
        """Test that different keys produce different genomes."""
        key1, key2 = jr.split(rng_key)
        genome1 = CategoricalGenome.random_init(key1, categorical_genome_config)
        genome2 = CategoricalGenome.random_init(key2, categorical_genome_config)
        # Very unlikely to be identical for length > 5
        if categorical_genome_config.length > 5:
            assert not jnp.array_equal(genome1.categories, genome2.categories)

    def test_category_bounds_respected(self, rng_key, categorical_genome_config):
        """Test that category values are within valid range."""
        # Generate many samples to test bounds
        for i in range(10):
            key = jr.fold_in(rng_key, i)
            genome = CategoricalGenome.random_init(key, categorical_genome_config)
            assert_valid_categorical_genome(genome, categorical_genome_config)

    @pytest.mark.jit
    def test_jit_compatibility(self, rng_key, categorical_genome_config):
        """Test that random_init can be JIT compiled."""
        # NEW paradigm: JIT compile the class method
        jit_init = jax.jit(CategoricalGenome.random_init, static_argnames=['config'])

        genome = jit_init(rng_key, categorical_genome_config)
        assert_valid_categorical_genome(genome, categorical_genome_config)

    def test_str_representation(self, categorical_genome):
        """Test string representation."""
        str_repr = str(categorical_genome)
        assert "CategoricalGenome(" in str_repr
        assert f"len={len(categorical_genome.categories)}" in str_repr

    def test_different_sizes(self, rng_key, categorical_size_config):
        """Test genomes of different sizes."""
        genome = CategoricalGenome.random_init(rng_key, categorical_size_config)
        assert_valid_categorical_genome(genome, categorical_size_config)


class TestGenomeConfigs:
    """Test genome configuration objects."""

    def test_binary_config_frozen(self):
        """Test that binary config is frozen (immutable)."""
        config = BinaryGenomeConfig(length=5)
        with pytest.raises(AttributeError):
            config.length = 10  # Should fail - frozen dataclass

    def test_real_config_frozen(self):
        """Test that real config is frozen (immutable)."""
        config = RealGenomeConfig(length=5, bounds=(-1.0, 1.0))
        with pytest.raises(AttributeError):
            config.length = 10  # Should fail - frozen dataclass

    def test_categorical_config_frozen(self):
        """Test that categorical config is frozen (immutable)."""
        config = CategoricalGenomeConfig(length=5, num_categories=3)
        with pytest.raises(AttributeError):
            config.length = 10  # Should fail - frozen dataclass

    def test_real_config_invalid_bounds(self):
        """Test that invalid bounds raise an error."""
        from malthusjax.core.genome.real_genome import validate_real_config

        invalid_config = RealGenomeConfig(length=5, bounds=(5.0, -5.0))  # min > max
        with pytest.raises(ValueError):
            validate_real_config(invalid_config)

    def test_categorical_config_validation(self):
        """Test categorical config validation."""
        from malthusjax.core.genome.categorical_genome import validate_categorical_config

        # Valid config should work
        config = CategoricalGenomeConfig(length=5, num_categories=3)
        assert config.length == 5
        assert config.num_categories == 3
        validate_categorical_config(config)  # Should not raise

        # Invalid configs should raise errors
        invalid_config = CategoricalGenomeConfig(length=0, num_categories=3)
        with pytest.raises(ValueError):
            validate_categorical_config(invalid_config)


@pytest.mark.slow
class TestGenomePerformance:
    """Performance tests for genome operations."""

    def test_large_binary_genome_creation(self, rng_key, large_binary_genome_config):
        """Test creation of large binary genomes."""
        genome = BinaryGenome.random_init(rng_key, large_binary_genome_config)
        assert_valid_binary_genome(genome, large_binary_genome_config)

    @pytest.mark.jit
    def test_batched_genome_creation(self, rng_key, binary_genome_config):
        """Test JIT-compiled batch creation of genomes."""
        batch_size = 100

        @jax.jit
        def create_population(keys):
            # NEW paradigm: vmap the class method
            return jax.vmap(BinaryGenome.random_init, in_axes=(0, None))(
                keys, binary_genome_config)

        keys = jr.split(rng_key, batch_size)
        population = create_population(keys)

        # Extract bits for validation
        population_bits = population.bits
        assert population_bits.shape == (batch_size, binary_genome_config.length)
        assert jnp.all((population_bits == 0) | (population_bits == 1))
