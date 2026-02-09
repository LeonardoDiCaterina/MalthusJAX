"""Integration tests for PRNG engine entry points."""

import jax
import jax.random as jr
import pytest

from malthusjax.core.random import create_key, PRNGImpl, is_new_style_key
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
    GeneticEvolutionState,
)
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator


@pytest.fixture
def small_engine():
    pop_size = 20
    genome_shape = (5,)
    engine_params = GeneticEngineParams(pop_size=pop_size, elitism=1, num_generations=1)
    genome_config = RealGenomeConfig(shape=genome_shape, bounds=(-5.0, 5.0))

    bbob_config = BBOBConfig(fn_name="sphere", num_dims=genome_shape[0], maximize=False)
    evaluator = BBOBEvaluator.create(bbob_config)

    selection = ElitePoolSelection(num_selections=pop_size, elite_k=2)
    crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
    mutation = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.1)

    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
    )

    return engine


def test_init_state_accepts_int_seed(small_engine):
    state = small_engine.init_state(42)
    assert isinstance(state, GeneticEvolutionState)
    # created key should be new-style (created via create_key)
    assert is_new_style_key(state.rng_key)


def test_init_state_accepts_key_object(small_engine):
    key = create_key(123, PRNGImpl.THREEFRY)
    state = small_engine.init_state(key)
    assert isinstance(state, GeneticEvolutionState)
    assert is_new_style_key(state.rng_key)


def test_init_state_warns_on_legacy_key(small_engine):
    legacy = jr.PRNGKey(42)
    with pytest.warns(DeprecationWarning):
        small_engine.init_state(legacy)
