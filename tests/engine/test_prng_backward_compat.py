"""Tests ensuring legacy PRNGKey compatibility and warnings."""

import jax.random as jr
import pytest

from malthusjax.engine.genetic_fastengine import GeneticEngine
from malthusjax.core.random import is_new_style_key
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator


@pytest.fixture
def small_engine():
    from malthusjax.engine.genetic_fastengine import GeneticEngineParams

    engine_params = GeneticEngineParams(pop_size=8, elitism=1, num_generations=2)
    engine = GeneticEngine(
        engine_params=engine_params,
        genome_config=RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0)),
        evaluator=BBOBEvaluator.create(BBOBConfig(fn_name="sphere", num_dims=5, maximize=False)),
        selection=ElitePoolSelection(num_selections=8, elite_k=2),
        crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
        mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.1),
    )
    return engine


def test_legacy_prngkey_still_works(small_engine):
    legacy = jr.PRNGKey(42)
    # Should not raise
    state = small_engine.init_state(legacy)
    assert state is not None


def test_legacy_key_reproducible_runs(small_engine):
    legacy = jr.PRNGKey(123)
    s1 = small_engine.init_state(legacy)
    f1, _, _ = small_engine.run(s1, compile=False)

    legacy2 = jr.PRNGKey(123)
    s2 = small_engine.init_state(legacy2)
    f2, _, _ = small_engine.run(s2, compile=False)

    assert f1.population.genes.values.shape == f2.population.genes.values.shape


def test_is_new_style_key_detects_legacy():
    legacy = jr.PRNGKey(42)
    assert not is_new_style_key(legacy)
