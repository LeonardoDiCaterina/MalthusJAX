"""Tests for the GymnaxEvaluator."""

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("gymnax")

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.rl.gymnax_evaluator import GymnaxEvaluator, GymnaxEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome


@pytest.fixture
def evaluator_and_genome_size():
    config = GymnaxEvaluatorConfig(env_name="CartPole-v1", max_steps=10)
    evaluator = GymnaxEvaluator.create(config)
    flat_params, _ = jax.tree_util.tree_flatten(
        evaluator.policy.init(
            jax.random.PRNGKey(0),
            jnp.zeros((1,) + evaluator.env.observation_space(evaluator.env_params).shape),
        )
    )
    genome_size = sum(p.size for p in flat_params)
    return evaluator, genome_size


def test_gymnax_evaluator_creation(evaluator_and_genome_size):
    evaluator, _ = evaluator_and_genome_size
    assert evaluator is not None


def test_gymnax_evaluator_evaluate(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    genome = RealGenome(values=jnp.zeros(genome_size))
    fitness = evaluator.evaluate(genome, jax.random.PRNGKey(0))

    assert fitness.shape == ()


def test_gymnax_evaluator_evaluate_population(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    pop_size = 5
    rng = jax.random.PRNGKey(42)
    genes_values = jax.random.normal(rng, (pop_size, genome_size))
    genomes = RealGenome(values=genes_values)
    population = BasePopulation(genes=genomes, fitness=jnp.zeros(pop_size), config=None)

    evaluated_pop = evaluator.evaluate_population(population, jax.random.PRNGKey(0))

    assert evaluated_pop.fitness.shape == (pop_size,)
    # Cartpole random weights often hit max step immediately if they do the same action,
    # but with random normals, some might differ. We just check shape.


def test_gymnax_evaluator_jittable(evaluator_and_genome_size):
    evaluator, genome_size = evaluator_and_genome_size

    genome = RealGenome(values=jnp.zeros(genome_size))
    jitted_eval = jax.jit(evaluator.evaluate)

    fitness = jitted_eval(genome, jax.random.PRNGKey(0))
    assert fitness.shape == ()


def test_gymnax_evaluator_maximize_sign_polarity(evaluator_and_genome_size):
    _, genome_size = evaluator_and_genome_size

    config_max = GymnaxEvaluatorConfig(env_name="CartPole-v1", max_steps=10, maximize=True)
    eval_max = GymnaxEvaluator.create(config_max)

    config_min = GymnaxEvaluatorConfig(env_name="CartPole-v1", max_steps=10, maximize=False)
    eval_min = GymnaxEvaluator.create(config_min)

    genome = RealGenome(values=jnp.zeros(genome_size))
    key = jax.random.PRNGKey(0)

    fit_max = eval_max.evaluate(genome, key)
    fit_min = eval_min.evaluate(genome, key)

    assert jnp.allclose(fit_max, -fit_min)

