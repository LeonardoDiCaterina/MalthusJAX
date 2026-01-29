import jax
import jax.numpy as jnp
from jax import random

import chex
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


def test_bbob_single_evaluate_type_and_scalar():
    # Note: BaseEvaluatorConfig requires `maximize` explicitly
    cfg = BBOBConfig(fn_name="sphere", num_dims=3, seed=42, maximize=False)
    evaluator = BBOBEvaluator.create(cfg)

    # Single genome (zeros) - should produce a scalar numeric fitness
    genome = RealGenome(values=jnp.zeros((3,)))
    fit = evaluator.evaluate(genome)

    # Fitness should be a scalar (0-d array) and finite
    assert jnp.ndim(jnp.asarray(fit)) == 0
    assert jnp.isfinite(jnp.asarray(fit))


def test_bbob_evaluate_population_shapes_and_finite():
    cfg = BBOBConfig(fn_name="sphere", num_dims=4, seed=123, maximize=False)
    evaluator = BBOBEvaluator.create(cfg)

    # Create a random population compatible with the evaluator
    key = random.PRNGKey(0)
    pop = RealPopulation.init_random(key, RealGenomeConfig(shape=(4,), bounds=(-5.0, 5.0)), size=5)

    evaluated = evaluator.evaluate_population(pop)

    # fitness should be (pop_size,) and finite
    assert evaluated.fitness.shape == (5,)
    assert jnp.all(jnp.isfinite(evaluated.fitness))

    # genes should not be modified in shape
    assert evaluated.genes.values.shape == (5, 4)


def test_bbob_maximize_flag_flips_sign():
    """Ensure that the `maximize` flag flips the reported fitness sign."""
    cfg_min = BBOBConfig(fn_name="sphere", num_dims=2, seed=0, maximize=False)
    cfg_max = BBOBConfig(fn_name="sphere", num_dims=2, seed=0, maximize=True)

    eval_min = BBOBEvaluator.create(cfg_min)
    eval_max = BBOBEvaluator.create(cfg_max)

    # Construct a small deterministic population
    X = jnp.array([[0.1, 0.2], [0.4, -0.5]])
    pop = RealPopulation(genes=RealGenome(values=X), fitness=jnp.zeros((2,)), config=RealGenomeConfig(shape=(2,)))

    pop_min = eval_min.evaluate_population(pop)
    pop_max = eval_max.evaluate_population(pop)

    # maximize output should be negation of minimization output
    assert jnp.allclose(pop_min.fitness, -pop_max.fitness)
