"""Tests for the JumanjiEvaluator."""

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("jumanji")

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.rl.jumanji_evaluator import JumanjiEvaluator, JumanjiEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome


@pytest.fixture
def evaluator_and_genome_size():
    config = JumanjiEvaluatorConfig(env_name="Snake-v1", max_steps=10)
    evaluator = JumanjiEvaluator.create(config)
    _, dummy_timestep = evaluator.env.reset(jax.random.PRNGKey(0))
    flat_params, _ = jax.tree_util.tree_flatten(
        evaluator.policy.init(jax.random.PRNGKey(0), dummy_timestep.observation)
    )
    genome_size = sum(p.size for p in flat_params)
    return evaluator, genome_size


def test_jumanji_evaluator_creation(evaluator_and_genome_size):
    evaluator, _ = evaluator_and_genome_size
    assert evaluator is not None


def test_jumanji_evaluator_evaluate(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    genome = RealGenome(values=jnp.zeros(genome_size))
    fitness = evaluator.evaluate(genome, jax.random.PRNGKey(0))

    assert fitness.shape == ()


def test_jumanji_evaluator_evaluate_population(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    pop_size = 5
    rng = jax.random.PRNGKey(42)
    genes_values = jax.random.normal(rng, (pop_size, genome_size))
    genomes = RealGenome(values=genes_values)
    population = BasePopulation(genes=genomes, fitness=jnp.zeros(pop_size), config=None)

    evaluated_pop = evaluator.evaluate_population(population, jax.random.PRNGKey(0))

    assert evaluated_pop.fitness.shape == (pop_size,)


def test_jumanji_evaluator_jittable(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    genome = RealGenome(values=jnp.zeros(genome_size))
    jitted_eval = jax.jit(evaluator.evaluate)

    fitness = jitted_eval(genome, jax.random.PRNGKey(0))
    assert fitness.shape == ()


def test_jumanji_evaluator_maximize_sign_polarity(evaluator_and_genome_size):
    _, genome_size = evaluator_and_genome_size

    config_max = JumanjiEvaluatorConfig(env_name="Snake-v1", max_steps=10, maximize=True)
    eval_max = JumanjiEvaluator.create(config_max)

    config_min = JumanjiEvaluatorConfig(env_name="Snake-v1", max_steps=10, maximize=False)
    eval_min = JumanjiEvaluator.create(config_min)

    genome = RealGenome(values=jnp.zeros(genome_size))
    key = jax.random.PRNGKey(0)

    fit_max = eval_max.evaluate(genome, key)
    fit_min = eval_min.evaluate(genome, key)

    assert jnp.allclose(fit_max, -fit_min)

