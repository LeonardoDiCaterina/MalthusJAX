import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.fitness.real_evaluators import (
    BoxConfig,
    BoxEvaluator,
    GriewankConfig,
    GriewankEvaluator,
    SphereConfig,
    SphereEvaluator,
)


def test_sphere_evaluator_equivalence(real_population):
    """Verifies that vectorized evaluation matches individual evaluation."""
    config = SphereConfig(maximize=False)
    evaluator = SphereEvaluator(config=config, data=None)

    # 1. Individual evaluations
    individual_results = []
    for i in range(len(real_population)):
        res = evaluator.evaluate(real_population[i])
        individual_results.append(res)
    manual_fitness = jnp.array(individual_results)

    # 2. Population (vectorized) evaluation
    pop_with_fitness = evaluator.evaluate_population(real_population)

    assert jnp.allclose(manual_fitness, pop_with_fitness.fitness)


def test_sphere_optimization_direction(real_genome):
    """Verifies that maximize=True/False correctly flips the fitness sign."""
    # Sphere value is sum of squares (always positive)
    val = jnp.sum(jnp.square(real_genome.values))

    eval_min = SphereEvaluator(config=SphereConfig(maximize=False), data=None)
    eval_max = SphereEvaluator(config=SphereConfig(maximize=True), data=None)

    assert float(eval_min.evaluate(real_genome)) == pytest.approx(-float(val))
    assert float(eval_max.evaluate(real_genome)) == pytest.approx(float(val))


def test_griewank_evaluator_jit(real_population):
    """Verifies that the multimodal Griewank evaluator is JIT-compilable."""
    config = GriewankConfig(maximize=False)
    evaluator = GriewankEvaluator(config=config, data=None)

    @jax.jit
    def fast_eval(pop):
        return evaluator.evaluate_population(pop)

    result_pop = fast_eval(real_population)
    assert not jnp.any(jnp.isnan(result_pop.fitness))


def test_box_constraint_penalty(rng_key):
    """Verifies that genomes outside the box bounds receive a heavy penalty."""
    target = jnp.array([0.0, 0.0])
    bounds = (jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))

    config = BoxConfig(
        target_point=target, box_bounds=bounds, penalty_factor=1000.0, maximize=False
    )
    evaluator = BoxEvaluator(config=config, data=None)

    from malthusjax.core.genome.real_genome import RealGenome

    # Genome inside bounds at (0.5, 0.5)
    inner_genome = RealGenome(values=jnp.array([0.5, 0.5]))
    # Genome outside bounds at (2.0, 2.0)
    outer_genome = RealGenome(values=jnp.array([2.0, 2.0]))

    inner_fit = evaluator.evaluate(inner_genome)
    outer_fit = evaluator.evaluate(outer_genome)

    # Outer genome should have a much lower (more negative) fitness due to penalty
    assert outer_fit < inner_fit
