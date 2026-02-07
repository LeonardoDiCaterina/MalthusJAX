import chex
import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.core.fitness.binary_evaluators import (
    BinarySumConfig,
    BinarySumEvaluator,
    KnapsackConfig,
    KnapsackEvaluator,
)
from malthusjax.core.genome.binary_genome import BinaryGenome


def test_binary_sum_mathematical_correctness(binary_population):
    """Validates OneMax implementation against bit counting formula."""
    config = BinarySumConfig(maximize=True)
    evaluator = BinarySumEvaluator(config=config, data=None)

    # Test known cases
    test_cases = [
        ([0, 0, 0, 0], 0),
        ([1, 1, 1, 1], 4),
        ([1, 0, 1, 0], 2),
        ([1], 1),
        ([], 0),  # Empty case
    ]
    
    for bits, expected_count in test_cases:
        if len(bits) > 0:
            genome = BinaryGenome(values=jnp.array(bits))
        else:
            genome = BinaryGenome(values=jnp.array([], dtype=jnp.int32))
        
        actual_fitness = evaluator.evaluate(genome)
        chex.assert_trees_all_close(
            actual_fitness, 
            expected_count, 
            rtol=0.0, atol=0.0
        )


def test_binary_sum_vmap_consistency(binary_population):
    """Validates individual vs population evaluation equivalence."""
    config = BinarySumConfig(maximize=True)
    evaluator = BinarySumEvaluator(config=config, data=None)

    # Manual individual calculation
    individual_results = [
        evaluator.evaluate(binary_population[i]) 
        for i in range(len(binary_population))
    ]
    manual_fitness = jnp.array(individual_results)

    # Population evaluation
    pop_with_fitness = evaluator.evaluate_population(binary_population)

    chex.assert_trees_all_close(
        manual_fitness, 
        pop_with_fitness.fitness, 
        rtol=0.0, atol=0.0
    )


def test_knapsack_penalty_formula_precision(binary_genome_config):
    """Tests penalty calculation against exact mathematical formula."""
    # Precise test case
    weights = jnp.array([2.0, 3.0, 4.0])
    values = jnp.array([1.0, 2.0, 3.0])
    capacity = 5.0
    penalty_factor = 10.0

    config = KnapsackConfig(
        maximize=True, 
        weights=weights, 
        values=values, 
        capacity=capacity, 
        penalty_factor=penalty_factor
    )
    evaluator = KnapsackEvaluator(config=config, data=None)

    # Test cases: (selection, total_weight, total_value, penalty, expected_fitness)
    test_cases = [
        ([1, 0, 0], 2.0, 1.0, 0.0, 1.0),  # Under capacity
        ([1, 1, 0], 5.0, 3.0, 0.0, 3.0),  # At capacity
        ([1, 1, 1], 9.0, 6.0, 40.0, -34.0),  # Over capacity: 6 - (9-5)*10 = -34
        ([0, 0, 1], 4.0, 3.0, 0.0, 3.0),  # Under capacity
    ]
    
    for selection, expected_weight, expected_value, expected_penalty, expected_fitness in test_cases:
        genome = BinaryGenome(values=jnp.array(selection))
        actual_fitness = evaluator.evaluate(genome)
        
        # Verify intermediate calculations
        actual_weight = jnp.sum(genome.values * weights)
        actual_value = jnp.sum(genome.values * values)
        actual_penalty = jnp.maximum(0.0, actual_weight - capacity) * penalty_factor
        
        chex.assert_trees_all_close(actual_weight, expected_weight, rtol=1e-6, atol=1e-7)
        chex.assert_trees_all_close(actual_value, expected_value, rtol=1e-6, atol=1e-7)
        chex.assert_trees_all_close(actual_penalty, expected_penalty, rtol=1e-6, atol=1e-7)
        
        # Final fitness
        chex.assert_trees_all_close(
            actual_fitness, 
            expected_fitness, 
            rtol=1e-6, atol=1e-7
        )


def test_knapsack_capacity_edge_cases():
    """Tests knapsack behavior with extreme capacity values."""
    weights = jnp.array([1.0, 2.0, 3.0])
    values = jnp.array([10.0, 20.0, 30.0])
    penalty_factor = 100.0
    
    # Zero capacity - everything gets penalty
    config_zero = KnapsackConfig(
        maximize=True,
        weights=weights,
        values=values,
        capacity=0.0,
        penalty_factor=penalty_factor
    )
    evaluator_zero = KnapsackEvaluator(config=config_zero, data=None)
    
    # Take one item
    one_item = BinaryGenome(values=jnp.array([1, 0, 0]))
    fitness_one = evaluator_zero.evaluate(one_item)
    # Value=10, penalty=1*100=100, fitness=10-100=-90
    chex.assert_trees_all_close(fitness_one, -90.0, rtol=1e-6, atol=1e-7)
    
    # Infinite capacity - no penalties
    config_inf = KnapsackConfig(
        maximize=True,
        weights=weights,
        values=values,
        capacity=jnp.inf,
        penalty_factor=penalty_factor
    )
    evaluator_inf = KnapsackEvaluator(config=config_inf, data=None)
    
    # Take everything
    all_items = BinaryGenome(values=jnp.array([1, 1, 1]))
    fitness_all = evaluator_inf.evaluate(all_items)
    # No penalty, just sum of values
    expected_total_value = jnp.sum(values)
    chex.assert_trees_all_close(fitness_all, expected_total_value, rtol=1e-6, atol=1e-7)


def test_knapsack_factory_determinism_and_properties(rng_key):
    """Validates random problem generation and capacity ratio accuracy."""
    n_items = 50
    capacity_ratio = 0.3
    
    # Generate problem
    config = KnapsackEvaluator.create_random_problem(
        rng_key, 
        n_items=n_items, 
        capacity_ratio=capacity_ratio
    )

    # Validate structure
    chex.assert_shape(config.weights, (n_items,))
    chex.assert_shape(config.values, (n_items,))
    assert config.maximize is True
    assert config.capacity > 0
    
    # Validate capacity ratio approximation
    total_weight = jnp.sum(config.weights)
    actual_ratio = config.capacity / total_weight
    chex.assert_trees_all_close(
        actual_ratio, 
        capacity_ratio, 
        rtol=0.01, atol=0.01  # Allow small deviation in ratio
    )
    
    # Test determinism
    config2 = KnapsackEvaluator.create_random_problem(
        rng_key, 
        n_items=n_items, 
        capacity_ratio=capacity_ratio
    )
    
    chex.assert_trees_all_close(config.weights, config2.weights, rtol=0.0, atol=0.0)
    chex.assert_trees_all_close(config.values, config2.values, rtol=0.0, atol=0.0)
    chex.assert_trees_all_close(config.capacity, config2.capacity, rtol=0.0, atol=0.0)


def test_knapsack_zero_weight_items():
    """Tests behavior with zero-weight or zero-value items."""
    # Items with zero weight (should always be selected if positive value)
    weights = jnp.array([0.0, 1.0, 2.0])
    values = jnp.array([5.0, 10.0, 15.0])
    capacity = 2.0
    
    config = KnapsackConfig(
        maximize=True,
        weights=weights,
        values=values,
        capacity=capacity,
        penalty_factor=100.0
    )
    evaluator = KnapsackEvaluator(config=config, data=None)
    
    # Take zero-weight item + one regular item
    selection = BinaryGenome(values=jnp.array([1, 1, 0]))  # weight=1, value=15
    fitness = evaluator.evaluate(selection)
    expected_fitness = 5.0 + 10.0  # No penalty since weight ≤ capacity
    chex.assert_trees_all_close(fitness, expected_fitness, rtol=1e-6, atol=1e-7)
    
    # Zero-value items
    weights_zv = jnp.array([1.0, 1.0])
    values_zv = jnp.array([0.0, 10.0])
    
    config_zv = KnapsackConfig(
        maximize=True,
        weights=weights_zv,
        values=values_zv,
        capacity=1.5,
        penalty_factor=100.0
    )
    evaluator_zv = KnapsackEvaluator(config=config_zv, data=None)
    
    # Take zero-value item only
    zero_value_selection = BinaryGenome(values=jnp.array([1, 0]))
    fitness_zv = evaluator_zv.evaluate(zero_value_selection)
    chex.assert_trees_all_close(fitness_zv, 0.0, rtol=1e-6, atol=1e-7)


def test_binary_evaluators_jit_consistency(binary_population):
    """Validates JIT vs non-JIT equivalence for binary evaluators."""
    config = BinarySumConfig(maximize=True)
    evaluator = BinarySumEvaluator(config=config, data=None)

    @jax.jit
    def jit_eval_population(pop):
        return evaluator.evaluate_population(pop)

    non_jit_result = evaluator.evaluate_population(binary_population)
    jit_result = jit_eval_population(binary_population)
    
    chex.assert_trees_all_close(
        non_jit_result.fitness, 
        jit_result.fitness, 
        rtol=0.0, atol=0.0
    )


def test_knapsack_population_scaling_invariance(knapsack_evaluator):
    """Tests knapsack evaluation across different population sizes."""
    from malthusjax.core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
    
    # Use config from knapsack_evaluator to determine genome size
    n_items = len(knapsack_evaluator.config.weights)
    genome_config = BinaryGenomeConfig(shape=(n_items,), p=0.5)
    
    for pop_size in [1, 10, 100]:
        key = jr.PRNGKey(42 + pop_size)
        population = BinaryPopulation.init_random(key, genome_config, size=pop_size)
        
        result_pop = knapsack_evaluator.evaluate_population(population)
        chex.assert_shape(result_pop.fitness, (pop_size,))
        assert not jnp.any(jnp.isnan(result_pop.fitness))
        
        # Verify PyTree structure preserved
        assert hasattr(result_pop, 'genes')
        assert hasattr(result_pop, 'fitness')
        chex.assert_shape(result_pop.genes.values, (pop_size, n_items))
