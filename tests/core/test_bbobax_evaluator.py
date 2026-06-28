import jax.numpy as jnp
from jax import random

from malthusjax.core.fitness.bbobax_evaluator import BBOBAXConfig, BBOBAXEvaluator
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


def test_bbobax_single_evaluate_type_and_scalar():
    """Ensure single genome evaluation returns a finite scalar."""
    cfg = BBOBAXConfig(fn_name="sphere", num_dims=3, seed=42, maximize=False)
    evaluator = BBOBAXEvaluator.create(cfg)

    # Single genome (zeros)
    genome = RealGenome(values=jnp.zeros((3,)))
    fit = evaluator.evaluate(genome)

    # Fitness should be a scalar (0-d array) and finite
    assert jnp.ndim(jnp.asarray(fit)) == 0
    assert jnp.isfinite(jnp.asarray(fit))


def test_bbobax_evaluate_population_shapes_and_finite():
    """Ensure vectorized population evaluation returns correct shapes."""
    cfg = BBOBAXConfig(fn_name="sphere", num_dims=4, seed=123, maximize=False)
    evaluator = BBOBAXEvaluator.create(cfg)

    # Create a random population compatible with the evaluator
    key = random.PRNGKey(0)
    pop = RealPopulation.init_random(key, RealGenomeConfig(shape=(4,), bounds=(-5.0, 5.0)), size=5)

    evaluated = evaluator.evaluate_population(pop)

    # fitness should be (pop_size,) and finite
    assert evaluated.fitness.shape == (5,)
    assert jnp.all(jnp.isfinite(evaluated.fitness))

    # genes values should remain unchanged in shape
    assert evaluated.genes.values.shape == (5, 4)


def test_bbobax_maximize_flag_flips_sign():
    """Ensure that the `maximize` flag correctly flips the reported fitness sign."""
    # Sphere function is always >= 0.
    # Minimizing sphere (maximize=False) -> we return -fitness (e.g., -10.0)
    # Maximizing sphere (maximize=True) -> we return fitness (e.g., 10.0)

    cfg_min = BBOBAXConfig(fn_name="sphere", num_dims=2, seed=0, maximize=False)
    cfg_max = BBOBAXConfig(fn_name="sphere", num_dims=2, seed=0, maximize=True)

    eval_min = BBOBAXEvaluator.create(cfg_min)
    eval_max = BBOBAXEvaluator.create(cfg_max)

    # Construct a small deterministic solution
    genome = RealGenome(values=jnp.array([1.0, 1.0]))

    fit_min = eval_min.evaluate(genome)
    fit_max = eval_max.evaluate(genome)

    # They should be opposites
    assert jnp.isclose(fit_min, -fit_max)
    # Since sphere at [1,1] is 2.0, fit_max should be 2.0, fit_min should be -2.0
    # (Assuming the task hasn't shifted the optimum away from 0,0 significantly)
    assert fit_max > fit_min


def test_bbobax_different_functions_produce_different_results():
    """Ensure fn_name correctly routes to different bbobax logic."""
    cfg_sphere = BBOBAXConfig(fn_name="sphere", num_dims=2, seed=1, maximize=True)
    cfg_rastrigin = BBOBAXConfig(fn_name="rastrigin", num_dims=2, seed=1, maximize=True)

    eval_s = BBOBAXEvaluator.create(cfg_sphere)
    eval_r = BBOBAXEvaluator.create(cfg_rastrigin)

    genome = RealGenome(values=jnp.array([2.0, 2.0]))

    fit_s = eval_s.evaluate(genome)
    fit_r = eval_r.evaluate(genome)

    # Sphere and Rastrigin should give different values for the same point
    assert not jnp.isclose(fit_s, fit_r)


def test_bbobax_seed_consistency():
    """Ensure same seed produces same instance (x_opt, f_opt)."""
    cfg1 = BBOBAXConfig(fn_name="ellipsoidal", num_dims=2, seed=42, maximize=True)
    cfg2 = BBOBAXConfig(fn_name="ellipsoidal", num_dims=2, seed=42, maximize=True)

    eval1 = BBOBAXEvaluator.create(cfg1)
    eval2 = BBOBAXEvaluator.create(cfg2)

    genome = RealGenome(values=jnp.array([0.5, -0.5]))

    assert jnp.isclose(eval1.evaluate(genome), eval2.evaluate(genome))
