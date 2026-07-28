"""
Shared pytest fixtures for engine tests.

This module provides reusable fixtures for setting up genetic engines,
genomes, and operators for consistent test execution across the test suite.
"""

import jax.random as jar
import pytest

from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
)
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


@pytest.fixture
def prng_key():
    """Provides a consistent PRNG key for deterministic testing."""
    return jar.PRNGKey(42)


@pytest.fixture
def genome_config():
    """Standard real genome configuration for testing."""
    return RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0))


@pytest.fixture
def engine_params():
    """Standard engine parameters for testing."""
    return GeneticEngineParams(
        pop_size=100,
        elitism=2,
        num_generations=10,
    )


@pytest.fixture
def bbob_evaluator():
    """BBOB Sphere evaluator for fitness testing."""
    bbob_config = BBOBConfig(fn_name="sphere", num_dims=10, maximize=False)
    return BBOBEvaluator.create(bbob_config)


@pytest.fixture
def selection_operator():
    """Elite pool selection operator."""
    return ElitePoolSelection(num_selections=100, elite_k=10)


@pytest.fixture
def crossover_operator():
    """Simulated binary crossover operator."""
    return SimulatedBinaryCrossover(num_offspring=2, eta=15.0)


@pytest.fixture
def mutation_operator():
    """Gaussian mutation operator."""
    return GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)


@pytest.fixture
def genetic_engine(
    engine_params,
    genome_config,
    bbob_evaluator,
    selection_operator,
    crossover_operator,
    mutation_operator,
):
    """
    Fully configured genetic engine for testing.

    Uses standard parameters and operators suitable for most tests.
    """
    return GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=bbob_evaluator,
        selection=selection_operator,
        crossover=crossover_operator,
        mutation=mutation_operator,
        enable_progress_bar=False,
    )


@pytest.fixture
def initialized_state(genetic_engine, prng_key):
    """
    Provides an initialized engine state ready for testing.

    This is the post-baking state after `init_state()` completes.
    """
    return genetic_engine.init_state(prng_key)
