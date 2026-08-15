import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.fitness.real_evaluators import SphereConfig, SphereEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine,
    GeneticEngineParams,
    disable_tracing,
    enable_tracing,
)
from malthusjax.operators.crossover.real import UniformCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.tournament import TournamentSelection


def test_tracing_enabled():
    enable_tracing()
    engine = GeneticEngine(
        genome_config=RealGenomeConfig(shape=(2,)),
        evaluator=SphereEvaluator(SphereConfig(maximize=False)),
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        engine_params=GeneticEngineParams(
            pop_size=4, elitism=1, num_generations=2, debug_tracing=True
        ),
    )
    key = jax.random.PRNGKey(0)
    state = engine.init_state(key)
    # This will hit line 78
    engine.step(state)
    disable_tracing()


def test_debug_step_coverage():
    engine = GeneticEngine(
        genome_config=RealGenomeConfig(shape=(2,)),
        evaluator=SphereEvaluator(SphereConfig(maximize=False)),
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        engine_params=GeneticEngineParams(pop_size=4, elitism=1, num_generations=2),
    )
    key = jax.random.PRNGKey(0)
    state = engine.init_state(key)
    engine.debug_step(state)


class DummyConfig:
    dtype = jnp.float32


def test_no_init_population():
    engine = GeneticEngine(
        genome_config=DummyConfig(),
        evaluator=SphereEvaluator(SphereConfig(maximize=False)),
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        engine_params=GeneticEngineParams(pop_size=4, elitism=1, num_generations=2),
    )
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="Unsupported genome config"):
        engine.init_state(key)


def test_ask_tell_with_key():
    engine = GeneticEngine(
        genome_config=RealGenomeConfig(shape=(2,)),
        evaluator=SphereEvaluator(SphereConfig(maximize=False)),
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        engine_params=GeneticEngineParams(pop_size=4, elitism=1, num_generations=2),
    )
    key = jax.random.PRNGKey(0)
    state = engine.init_state(key)

    engine, pop = engine.ask_with_key(state, jax.random.PRNGKey(1))
    # mock fitness
    pop = pop.replace(fitness=jnp.ones(4))
    state = engine.tell_with_key(state, pop, jax.random.PRNGKey(2))
    assert state.generation == 1


def test_enforce_layout_1d():
    engine = GeneticEngine(
        genome_config=RealGenomeConfig(shape=()),
        evaluator=SphereEvaluator(SphereConfig(maximize=False)),
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        engine_params=GeneticEngineParams(pop_size=4, elitism=1, num_generations=2),
    )
    key = jax.random.PRNGKey(0)
    state = engine.init_state(key)
    assert state.population.genes.values.ndim == 1


def test_forward_presplit_keys_true():
    engine = GeneticEngine(
        genome_config=RealGenomeConfig(shape=(2,)),
        evaluator=SphereEvaluator(SphereConfig(maximize=False)),
        selection=TournamentSelection(num_selections=4, tournament_size=2),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(),
        engine_params=GeneticEngineParams(
            pop_size=4, elitism=1, num_generations=2, forward_presplit_keys=True
        ),
    )
    key = jax.random.PRNGKey(0)
    state = engine.init_state(key)
    # The default key allocations from ResourceMap won't match what the operators expect
    # when forward_presplit_keys=True in all branches, so this will trigger the fallbacks.
    engine.step(state)
