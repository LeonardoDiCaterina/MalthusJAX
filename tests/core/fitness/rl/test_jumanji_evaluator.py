"""Tests for the JumanjiEvaluator."""

import jax
import jax.numpy as jnp
import pytest

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
    fitness = evaluator.evaluate(genome)

    assert fitness.shape == ()


def test_jumanji_evaluator_evaluate_population(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    pop_size = 5
    rng = jax.random.PRNGKey(42)
    genes_values = jax.random.normal(rng, (pop_size, genome_size))
    genomes = RealGenome(values=genes_values)
    population = BasePopulation(genes=genomes, fitness=jnp.zeros(pop_size), config=None)

    evaluated_pop = evaluator.evaluate_population(population)

    assert evaluated_pop.fitness.shape == (pop_size,)


def test_jumanji_evaluator_jittable(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    genome = RealGenome(values=jnp.zeros(genome_size))
    jitted_eval = jax.jit(evaluator.evaluate)

    fitness = jitted_eval(genome)
    assert fitness.shape == ()
