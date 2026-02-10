import chex
import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.core.fitness.real_evaluators import (
    BoxConfig,
    BoxEvaluator,
    GriewankConfig,
    GriewankEvaluator,
    SphereConfig,
    SphereEvaluator,
)


def test_sphere_evaluator_mathematical_correctness(real_population):
    """Validates Sphere function implementation against analytical formula."""
    config = SphereConfig(maximize=False)
    evaluator = SphereEvaluator(config=config, data=None)

    from malthusjax.core.genome.real_genome import RealGenome

    origin_genome = RealGenome(values=jnp.zeros(3))
    origin_fitness = evaluator.evaluate(origin_genome)
    chex.assert_trees_all_close(origin_fitness, 0.0, rtol=1e-6, atol=1e-7)

    unit_genome = RealGenome(values=jnp.array([1.0, 0.0, 0.0]))
    unit_fitness = evaluator.evaluate(unit_genome)
    chex.assert_trees_all_close(unit_fitness, -1.0, rtol=1e-6, atol=1e-7)

    test_genome = real_population[0]
    expected_value = -jnp.sum(jnp.square(test_genome.values))
    actual_value = evaluator.evaluate(test_genome)
    chex.assert_trees_all_close(actual_value, expected_value, rtol=1e-6, atol=1e-7)


def test_sphere_optimization_direction_precision(real_genome):
    """Tests maximize/minimize flag with precise numerical validation."""
    sphere_value = jnp.sum(jnp.square(real_genome.values))

    eval_min = SphereEvaluator(config=SphereConfig(maximize=False), data=None)
    eval_max = SphereEvaluator(config=SphereConfig(maximize=True), data=None)

    min_result = eval_min.evaluate(real_genome)
    max_result = eval_max.evaluate(real_genome)

    chex.assert_trees_all_close(min_result, -sphere_value, rtol=1e-6, atol=1e-7)
    chex.assert_trees_all_close(max_result, sphere_value, rtol=1e-6, atol=1e-7)
    chex.assert_trees_all_close(min_result, -max_result, rtol=1e-6, atol=1e-7)


def test_sphere_vmap_population_consistency(real_population):
    """Validates individual vs population evaluation mathematical equivalence."""
    config = SphereConfig(maximize=False)
    evaluator = SphereEvaluator(config=config, data=None)

    # Individual evaluations
    individual_results = []
    for i in range(len(real_population)):
        res = evaluator.evaluate(real_population[i])
        individual_results.append(res)
    manual_fitness = jnp.array(individual_results)

    # Population evaluation via vmap
    pop_with_fitness = evaluator.evaluate_population(real_population)

    chex.assert_trees_all_close(manual_fitness, pop_with_fitness.fitness, rtol=1e-6, atol=1e-7)


def test_sphere_nan_inf_handling():
    """Tests behavior with NaN/Inf inputs."""
    config = SphereConfig(maximize=False)
    evaluator = SphereEvaluator(config=config, data=None)

    from malthusjax.core.genome.real_genome import RealGenome

    nan_genome = RealGenome(values=jnp.array([1.0, jnp.nan, 2.0]))
    nan_result = evaluator.evaluate(nan_genome)
    assert jnp.isnan(nan_result)

    inf_genome = RealGenome(values=jnp.array([jnp.inf, 0.0]))
    inf_result = evaluator.evaluate(inf_genome)
    assert jnp.isneginf(inf_result)


def test_griewank_evaluator_range_validation(real_population):
    """Validates Griewank function mathematical properties and range."""
    config = GriewankConfig(maximize=False)
    evaluator = GriewankEvaluator(config=config, data=None)

    pop_with_fitness = evaluator.evaluate_population(real_population)

    assert jnp.all(pop_with_fitness.fitness <= 0.0)
    assert not jnp.any(jnp.isnan(pop_with_fitness.fitness))
    assert not jnp.any(jnp.isinf(pop_with_fitness.fitness))

    from malthusjax.core.genome.real_genome import RealGenome

    origin = RealGenome(values=jnp.zeros(5))
    origin_fitness = evaluator.evaluate(origin)
    chex.assert_trees_all_close(origin_fitness, 0.0, rtol=1e-5, atol=1e-6)


def test_griewank_jit_stability(real_population):
    """Verifies JIT compilation preserves Griewank evaluation accuracy."""
    config = GriewankConfig(maximize=False)
    evaluator = GriewankEvaluator(config=config, data=None)

    @jax.jit
    def jit_eval_population(pop):
        return evaluator.evaluate_population(pop)

    non_jit_result = evaluator.evaluate_population(real_population)
    jit_result = jit_eval_population(real_population)

    chex.assert_trees_all_close(non_jit_result.fitness, jit_result.fitness, rtol=1e-6, atol=1e-7)


def test_box_constraint_penalty_magnitude(rng_key):
    """Validates penalty calculation precision and boundary behavior."""
    target = jnp.array([0.0, 0.0])
    bounds = (jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))
    penalty_factor = 100.0

    config = BoxConfig(
        target_point=target, box_bounds=bounds, penalty_factor=penalty_factor, maximize=False
    )
    evaluator = BoxEvaluator(config=config, data=None)

    from malthusjax.core.genome.real_genome import RealGenome

    inner_genome = RealGenome(values=jnp.array([0.5, 0.0]))
    inner_fitness = evaluator.evaluate(inner_genome)
    expected_inner = -0.5 
    chex.assert_trees_all_close(inner_fitness, expected_inner, rtol=1e-6, atol=1e-7)

    boundary_genome = RealGenome(values=jnp.array([1.0, 1.0]))  # dist = sqrt(2)
    boundary_fitness = evaluator.evaluate(boundary_genome)
    expected_boundary = -jnp.sqrt(2.0)
    chex.assert_trees_all_close(boundary_fitness, expected_boundary, rtol=1e-6, atol=1e-7)

    outer_genome = RealGenome(values=jnp.array([2.0, 0.0]))  # 1 unit outside
    outer_fitness = evaluator.evaluate(outer_genome)
    expected_outer = -(2.0 + 100.0)
    chex.assert_trees_all_close(outer_fitness, expected_outer, rtol=1e-6, atol=1e-7)


def test_box_evaluator_edge_cases():
    """Tests Box evaluator with edge case configurations."""
    config_no_penalty = BoxConfig(
        target_point=jnp.array([0.0]),
        box_bounds=(jnp.array([-1.0]), jnp.array([1.0])),
        penalty_factor=0.0,
        maximize=False,
        )
    evaluator_no_penalty = BoxEvaluator(config=config_no_penalty, data=None)

    from malthusjax.core.genome.real_genome import RealGenome

    outside_genome = RealGenome(values=jnp.array([2.0]))
    fitness = evaluator_no_penalty.evaluate(outside_genome)
    chex.assert_trees_all_close(fitness, -2.0, rtol=1e-6, atol=1e-7)


def test_evaluator_population_size_scaling():
    """Tests evaluator behavior across different population sizes."""
    config = SphereConfig(maximize=False)
    evaluator = SphereEvaluator(config=config, data=None)

    from malthusjax.core.genome.real_genome import RealGenomeConfig, RealPopulation

    genome_config = RealGenomeConfig(shape=(3,), bounds=(-2.0, 2.0))

    for pop_size in [1, 5, 50]:
        key = jr.PRNGKey(42 + pop_size)
        population = RealPopulation.init_random(key, genome_config, size=pop_size)

        result_pop = evaluator.evaluate_population(population)
        chex.assert_shape(result_pop.fitness, (pop_size,))
        assert not jnp.any(jnp.isnan(result_pop.fitness))

        # Verify PyTree structure preserved
        assert hasattr(result_pop, "genes")
        assert hasattr(result_pop, "fitness")
        chex.assert_shape(result_pop.genes.values, (pop_size, 3))
