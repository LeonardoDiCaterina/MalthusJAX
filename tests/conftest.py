"""
Shared pytest fixtures and test utilities for MalthusJAX.

This module provides common fixtures and utilities used across all tests,
including random keys, genome configurations, and test data.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.core.fitness.binary_evaluators import KnapsackConfig, KnapsackEvaluator
from malthusjax.core.random import PRNGImpl, create_key
from malthusjax.engine.resource_mapper import KeyDerivationStrategy

# Import genome types and configurations
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


@pytest.fixture
def knapsack_config(rng_key) -> KnapsackConfig:
    """
    Standard Knapsack configuration fixture.
    Creates a problem with 10 items and a 50% capacity ratio.
    """
    return KnapsackEvaluator.create_random_problem(
        rng_key, n_items=10, capacity_ratio=0.5, maximize=True
    )


@pytest.fixture
def knapsack_evaluator(knapsack_config) -> KnapsackEvaluator:
    """Initialized KnapsackEvaluator with standard config."""
    return KnapsackEvaluator(config=knapsack_config, data=None)


@pytest.fixture
def knapsack_data() -> tuple[jax.Array, jax.Array, float]:
    """
    Manual Knapsack data for unit testing exact math.
    Items: [W: 2, V: 10], [W: 4, V: 20], [W: 6, V: 30]
    Capacity: 7
    """
    weights = jnp.array([2.0, 4.0, 6.0])
    values = jnp.array([10.0, 20.0, 30.0])
    capacity = 7.0
    return weights, values, capacity



@pytest.fixture
def real_population(rng_key, real_genome_config) -> RealPopulation:
    """Typed RealPopulation fixture."""
    return RealPopulation.init_random(rng_key, real_genome_config, size=10)


@pytest.fixture
def binary_population(rng_key, binary_genome_config) -> BinaryPopulation:
    """Typed BinaryPopulation fixture."""
    return BinaryPopulation.init_random(rng_key, binary_genome_config, size=10)


@pytest.fixture
def key_fixture():
    return jax.random.PRNGKey(42)


def get_batch_shape(pytree_obj):
    """
    Generic way to get the shape of a genome batch.
    Works for Binary and Real values.
    """
    # Get the first array (leaf) found in the structure
    leaves = jax.tree_util.tree_leaves(pytree_obj)
    if not leaves:
        return (0,)
    # Return the shape of that first array
    return leaves[0].shape


@pytest.fixture
def rng_key() -> jax.Array:
    """Base random key for deterministic tests."""
    return jr.PRNGKey(42)


@pytest.fixture
def binary_genome_config() -> BinaryGenomeConfig:
    """Standard binary genome configuration."""
    return BinaryGenomeConfig(shape=(10,), p=0.5)


@pytest.fixture
def small_binary_genome_config() -> BinaryGenomeConfig:
    """Small binary genome for quick tests."""
    return BinaryGenomeConfig(shape=(5,), p=0.5)


@pytest.fixture
def large_binary_genome_config() -> BinaryGenomeConfig:
    """Large binary genome for performance tests."""
    return BinaryGenomeConfig(shape=(100,), p=0.5)


@pytest.fixture
def real_genome_config() -> RealGenomeConfig:
    """Standard real genome configuration."""
    return RealGenomeConfig(shape=(10,), bounds=(-10.0, 10.0))


@pytest.fixture
def constrained_real_genome_config() -> RealGenomeConfig:
    """Real genome with tight bounds."""
    return RealGenomeConfig(shape=(3,), bounds=(-1.0, 1.0))

@pytest.fixture
def binary_genome(rng_key, binary_genome_config) -> BinaryGenome:
    """Sample binary genome for testing."""
    return BinaryGenome.random_init(rng_key, binary_genome_config)

@pytest.fixture
def real_genome(rng_key, real_genome_config) -> RealGenome:
    """Sample real genome for testing."""
    return RealGenome.random_init(rng_key, real_genome_config)

@pytest.fixture
def fitness_values() -> jax.Array:
    """Sample fitness values for selection testing."""
    return jnp.array([0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.5, 0.6, 0.4, 0.95])

@pytest.fixture
def low_fitness_values() -> jax.Array:
    """All low fitness values for edge case testing."""
    return jnp.array([0.01, 0.02, 0.01, 0.03, 0.01])


@pytest.fixture
def high_fitness_values() -> jax.Array:
    """All high fitness values for edge case testing."""
    return jnp.array([0.95, 0.98, 0.97, 0.99, 0.96])


# Utility functions for test assertions


def assert_valid_binary_genome(genome: BinaryGenome, config: BinaryGenomeConfig) -> None:
    """Assert that a binary genome is valid."""
    assert isinstance(genome, BinaryGenome)
    assert genome.bits.shape == (config.length,)
    assert jnp.all((genome.bits == 0) | (genome.bits == 1))


def assert_valid_binary_genome(genome: BinaryGenome, config: BinaryGenomeConfig) -> None:
    """Assert that a binary genome is valid."""
    assert isinstance(genome, BinaryGenome)
    assert genome.bits.shape == (config.length,)
    assert jnp.all((genome.bits == 0) | (genome.bits == 1))


def assert_valid_real_genome(genome: RealGenome, config: RealGenomeConfig) -> None:
    """Assert that a real genome is valid."""
    assert isinstance(genome, RealGenome)
    assert genome.values.shape == (config.length,)
    assert jnp.all(genome.values >= config.bounds[0])
    assert jnp.all(genome.values <= config.bounds[1])



def assert_valid_binary_genome_batch(genome_batch, config: BinaryGenomeConfig) -> None:
    """Assert that a batch of binary genomes is valid (NEW paradigm)."""
    # genome_batch should be BinaryGenome with batch-first shape (batch_size, length)
    assert hasattr(genome_batch, "bits")
    batch_size = genome_batch.bits.shape[0]
    assert genome_batch.bits.shape == (batch_size, config.length)
    assert jnp.all((genome_batch.bits == 0) | (genome_batch.bits == 1))


def assert_valid_real_genome_batch(genome_batch, config: RealGenomeConfig) -> None:
    """Assert that a batch of real genomes is valid (NEW paradigm)."""
    # genome_batch should be RealGenome with batch-first shape (batch_size, length)
    assert hasattr(genome_batch, "values")
    batch_size = genome_batch.values.shape[0]
    assert genome_batch.values.shape == (batch_size, config.length)
    assert jnp.all(genome_batch.values >= config.bounds[0])
    assert jnp.all(genome_batch.values <= config.bounds[1])


def assert_jit_compilable(func, *args) -> None:
    """Assert that a function is JIT compilable."""
    try:
        jit_func = jax.jit(func)
        result = jit_func(*args)
        assert result is not None
    except Exception as e:
        pytest.fail(f"Function failed JIT compilation: {e}")


def assert_deterministic(func, *args) -> None:
    """Assert that a function produces deterministic results with same inputs."""
    result1 = func(*args)
    result2 = func(*args)
    if hasattr(result1, "shape"):
        assert jnp.allclose(result1, result2), "Function should be deterministic"
    else:
        assert result1 == result2, "Function should be deterministic"


# Parameterized fixtures for different genome sizes
@pytest.fixture(params=[5, 10, 20])
def binary_size_config(request) -> BinaryGenomeConfig:
    """Binary genome configs of different sizes."""
    return BinaryGenomeConfig(shape=(request.param,), p=0.5)


@pytest.fixture(params=[3, 5, 10])
def real_size_config(request) -> RealGenomeConfig:
    """Real genome configs of different sizes."""
    return RealGenomeConfig(shape=(request.param,), bounds=(-5.0, 5.0))


# Performance testing markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "jit: marks tests that specifically test JIT compilation")
    config.addinivalue_line("markers", "integration: marks integration tests")


# ------------------------
# PRNG / Key Fixtures
# ------------------------
@pytest.fixture(params=list(PRNGImpl))
def prng_impl(request) -> PRNGImpl:
    """Parametrized PRNG implementation fixture.

    Skips implementations that are not available in the current JAX build.
    Use in tests as `prng_impl` to automatically iterate over supported impls.
    """
    impl: PRNGImpl = request.param
    try:
        # sanity-check: attempt to construct a key with this impl
        create_key(0, impl=impl)
    except ValueError:
        pytest.skip(f"PRNGImpl {impl} not supported by this JAX build")
    return impl


@pytest.fixture(params=[KeyDerivationStrategy.SPLIT, KeyDerivationStrategy.FOLD])
def key_derivation(request) -> KeyDerivationStrategy:
    """Parametrized key derivation strategy fixture."""
    return request.param


@pytest.fixture
def master_prng_key(prng_impl: PRNGImpl) -> jax.Array:
    """Create a deterministic master key for the requested PRNG impl."""
    return create_key(1234, impl=prng_impl)


@pytest.fixture
def engine_with_prng(prng_impl: PRNGImpl, key_derivation: KeyDerivationStrategy):
    """Construct a small GeneticEngine configured with the requested PRNG/key-derivation.

    Useful for PRNG-focused tests that need a baked operator set in `state.operators`.
    """
    from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
    from malthusjax.core.genome.real_genome import RealGenomeConfig
    from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
    from malthusjax.operators.selection.elite_pool import ElitePoolSelection
    from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
    from malthusjax.operators.mutation.real import GaussianMutation

    params = GeneticEngineParams(
        pop_size=32,
        elitism=1,
        num_generations=1,
        key_derivation=key_derivation,
        prng_impl=prng_impl,
    )

    genome_config = RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0))
    bbob = BBOBEvaluator.create(BBOBConfig(fn_name="sphere", num_dims=5, maximize=False))

    engine = GeneticEngine(
        engine_params=params,
        genome_config=genome_config,
        evaluator=bbob,
        selection=ElitePoolSelection(num_selections=32, elite_k=2),
        crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
        mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.1),
    )
    return engine
