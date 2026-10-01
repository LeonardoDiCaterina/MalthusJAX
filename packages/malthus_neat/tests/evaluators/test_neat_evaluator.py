"""Tests for NeatEvaluator and NeatQDEvaluator."""

import jax
import jax.numpy as jnp

from malthus_neat.evaluators.neat_evaluator import NeatEvaluator, NeatEvaluatorConfig
from malthus_neat.evaluators.neat_qd_evaluator import NeatQDEvalConfig, NeatQDEvaluator
from malthus_neat.genomes.neat_genome import NeatGenome, NeatGenomeConfig, NeatPopulation
from malthus_neat.interpreters.neat_interpreter import NeatInterpreter


def test_neat_supervised_evaluator_mse():
    X = jnp.linspace(-1.0, 1.0, 20).reshape(-1, 2)
    y = jnp.sum(X**2, axis=-1, keepdims=True)

    config = NeatGenomeConfig(num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16)
    interpreter = NeatInterpreter(genome_config=config)
    eval_cfg = NeatEvaluatorConfig(genome_config=config, loss_function="mse", maximize=False)
    evaluator = NeatEvaluator(config=eval_cfg, data=(X, y), interpreter=interpreter)

    key = jax.random.PRNGKey(42)
    genome = NeatGenome.random_init(key, config)

    loss = evaluator.evaluate(genome)
    assert jnp.isfinite(loss)
    assert loss >= 0.0

    # Test population evaluation
    pop = NeatPopulation.init(key, config, size=5)
    evaluated_pop = evaluator.evaluate_population(pop)
    assert evaluated_pop.fitness.shape == (5,)
    assert jnp.all(jnp.isfinite(evaluated_pop.fitness))


def test_neat_qd_evaluator_descriptors():
    X = jnp.linspace(-1.0, 1.0, 20).reshape(-1, 2)
    y = jnp.sin(X[:, 0:1])

    config = NeatGenomeConfig(num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16)
    interpreter = NeatInterpreter(genome_config=config)
    eval_cfg = NeatQDEvalConfig(genome_config=config, loss_function="mse", maximize=False)
    evaluator = NeatQDEvaluator(config=eval_cfg, data=(X, y), interpreter=interpreter)

    key = jax.random.PRNGKey(101)
    genome = NeatGenome.random_init(key, config)

    loss, descriptors = evaluator.evaluate_qd(genome)
    assert jnp.isfinite(loss)
    assert descriptors.shape == (2,)
    assert 0.0 <= float(descriptors[0]) <= 1.0  # Connection density
    assert 0.0 <= float(descriptors[1]) <= 1.0  # Prediction spread

    # Test population evaluation
    pop = NeatPopulation.init(key, config, size=5)
    evaluated_pop = evaluator.evaluate_population(pop)
    assert evaluated_pop.fitness.shape == (5,)
    assert "descriptors" in evaluated_pop.info
    assert evaluated_pop.info["descriptors"].shape == (5, 2)
