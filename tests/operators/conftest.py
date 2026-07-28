"""
Shared pytest fixtures for operator tests.

This module provides reusable fixtures for setting up genomes, populations,
and operator configurations for consistent test execution across the test suite.
"""

import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation


@pytest.fixture
def prng_key():
    """Provides a consistent PRNG key for deterministic testing."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def real_genome_config():
    """Standard real genome configuration for testing."""
    return RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))


@pytest.fixture
def binary_genome_config():
    """Standard binary genome configuration for testing."""
    return BinaryGenomeConfig(shape=(10,), p=0.5)


@pytest.fixture
def real_population(real_genome_config):
    """A standard real population of size 20 for testing."""
    pop_size = 20
    key = jax.random.PRNGKey(123)
    genes = jax.random.uniform(
        key,
        shape=(pop_size,) + real_genome_config.shape,
        minval=real_genome_config.bounds[0],
        maxval=real_genome_config.bounds[1],
    )
    fitness = jnp.zeros(pop_size)
    return RealPopulation(genes=genes, fitness=fitness)


@pytest.fixture
def binary_population(binary_genome_config):
    """A standard binary population of size 20 for testing."""
    pop_size = 20
    key = jax.random.PRNGKey(123)
    genes = jax.random.bernoulli(
        key, p=binary_genome_config.p, shape=(pop_size,) + binary_genome_config.shape
    ).astype(jnp.int32)
    fitness = jnp.zeros(pop_size)
    return BinaryPopulation(genes=genes, fitness=fitness)
