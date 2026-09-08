"""Tests for BBOBAXEvaluator using the real bbobax package.

Since bbobax is now available in the test environment, these tests use the
real implementation rather than mocks.
"""
import jax.numpy as jnp
import pytest

try:
    from malthusjax.core.fitness.bbobax_evaluator import BBOBAXConfig, BBOBAXEvaluator
    from malthusjax.core.genome.real_genome import RealGenome
    HAS_BBOBAX = True
except ImportError:
    HAS_BBOBAX = False


@pytest.mark.skipif(not HAS_BBOBAX, reason="bbobax is not installed")
def test_bbobax_evaluator_sphere_minimize():
    """Evaluating sphere with maximize=False returns the raw fitness."""
    config = BBOBAXConfig(fn_name="sphere", num_dims=3, seed=0, maximize=False)
    evaluator = BBOBAXEvaluator.create(config)
    genome = RealGenome(values=jnp.zeros(3))
    fitness = evaluator.evaluate(genome)
    # At the origin the sphere value is small but not necessarily zero due to x_opt shift
    assert jnp.isfinite(fitness)
    assert fitness >= 0.0


@pytest.mark.skipif(not HAS_BBOBAX, reason="bbobax is not installed")
def test_bbobax_evaluator_sphere_maximize():
    """Maximizing negates the fitness so larger is better."""
    config_min = BBOBAXConfig(fn_name="sphere", num_dims=3, seed=0, maximize=False)
    config_max = BBOBAXConfig(fn_name="sphere", num_dims=3, seed=0, maximize=True)

    eval_min = BBOBAXEvaluator.create(config_min)
    eval_max = BBOBAXEvaluator.create(config_max)

    genome = RealGenome(values=jnp.ones(3))

    fitness_min = eval_min.evaluate(genome)
    fitness_max = eval_max.evaluate(genome)

    # maximize=True negates the value
    assert jnp.isclose(fitness_min, -fitness_max)


@pytest.mark.skipif(not HAS_BBOBAX, reason="bbobax is not installed")
def test_bbobax_evaluator_invalid_function():
    """Creating an evaluator with an unknown fn_name raises ValueError."""
    config = BBOBAXConfig(fn_name="does_not_exist", num_dims=3)
    with pytest.raises(ValueError, match="Unknown function 'does_not_exist'"):
        BBOBAXEvaluator.create(config)
