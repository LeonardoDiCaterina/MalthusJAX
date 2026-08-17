"""Tests for the BraxEvaluator."""

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("brax")

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.rl.brax_evaluator import BraxEvaluator, BraxEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome


@pytest.fixture
def evaluator_and_genome_size():
    config = BraxEvaluatorConfig(env_name="ant", max_steps=10)
    evaluator = BraxEvaluator.create(config)
    flat_params, _ = jax.tree_util.tree_flatten(
        evaluator.policy.init(jax.random.PRNGKey(0), jnp.zeros((1, evaluator.env.observation_size)))
    )
    genome_size = sum(p.size for p in flat_params)
    return evaluator, genome_size


def test_brax_evaluator_creation(evaluator_and_genome_size):
    evaluator, _ = evaluator_and_genome_size
    assert evaluator is not None


def test_brax_evaluator_evaluate(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    genome = RealGenome(values=jnp.zeros(genome_size))
    fitness = evaluator.evaluate(genome, jax.random.PRNGKey(0))

    assert fitness.shape == ()


def test_brax_evaluator_evaluate_population(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    pop_size = 5
    rng = jax.random.PRNGKey(42)
    genes_values = jax.random.normal(rng, (pop_size, genome_size))
    genomes = RealGenome(values=genes_values)
    population = BasePopulation(genes=genomes, fitness=jnp.zeros(pop_size), config=None)

    evaluated_pop = evaluator.evaluate_population(population, jax.random.PRNGKey(0))

    assert evaluated_pop.fitness.shape == (pop_size,)


def test_brax_evaluator_jittable(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    genome = RealGenome(values=jnp.zeros(genome_size))
    jitted_eval = jax.jit(evaluator.evaluate)

    fitness = jitted_eval(genome, jax.random.PRNGKey(0))
    assert fitness.shape == ()


def test_brax_evaluator_maximize_sign_polarity(evaluator_and_genome_size):
    _, genome_size = evaluator_and_genome_size

    config_max = BraxEvaluatorConfig(env_name="ant", max_steps=10, maximize=True)
    eval_max = BraxEvaluator.create(config_max)

    config_min = BraxEvaluatorConfig(env_name="ant", max_steps=10, maximize=False)
    eval_min = BraxEvaluator.create(config_min)

    genome = RealGenome(values=jnp.zeros(genome_size))
    key = jax.random.PRNGKey(0)

    fit_max = eval_max.evaluate(genome, key)
    fit_min = eval_min.evaluate(genome, key)

    # Negated for maximize=True vs positive for maximize=False
    assert jnp.allclose(fit_max, -fit_min)


def test_brax_evaluator_end_to_end_elitism(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    from malthusjax.core.genome.real_genome import RealGenomeConfig
    from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
    from malthusjax.operators.crossover.real import BlendCrossover
    from malthusjax.operators.mutation.real import GaussianMutation
    from malthusjax.operators.selection.tournament import TournamentSelection

    genome_config = RealGenomeConfig(shape=(genome_size,), bounds=(-1.0, 1.0))
    engine_params = GeneticEngineParams(pop_size=10, num_generations=5, elitism=2)

    engine = GeneticEngine(
        genome_config=genome_config,
        evaluator=evaluator,
        selection=TournamentSelection(num_selections=10, tournament_size=2),
        crossover=BlendCrossover(),
        mutation=GaussianMutation(mutation_rate=0.1),
        engine_params=engine_params,
    )

    state = engine.init_state(rng_key=42)
    final_state, history, _ = engine.run(state)

    # Monotonic non-increasing fitness across generations (lower is better for stored fitness)
    history_best = history.best_fitness
    for i in range(len(history_best) - 1):
        assert history_best[i + 1] <= history_best[i] + 1e-5

