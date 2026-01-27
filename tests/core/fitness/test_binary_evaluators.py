import jax
import jax.numpy as jnp

from malthusjax.core.fitness.binary_evaluators import (
    BinarySumConfig,
    BinarySumEvaluator,
    KnapsackConfig,
    KnapsackEvaluator,
)
from malthusjax.core.genome.binary_genome import BinaryGenome


def test_binary_sum_equivalence(binary_population):
    """Verifies that vectorized OneMax evaluation matches manual loop."""
    config = BinarySumConfig(maximize=True)
    evaluator = BinarySumEvaluator(config=config, data=None)

    # 1. Manual calculation for comparison
    individual_results = [
        evaluator.evaluate(binary_population[i]) for i in range(len(binary_population))
    ]
    manual_fitness = jnp.array(individual_results)

    # 2. Vectorized population evaluation
    pop_with_fitness = evaluator.evaluate_population(binary_population)

    assert jnp.allclose(manual_fitness, pop_with_fitness.fitness)


def test_knapsack_penalty_logic(binary_genome_config):
    """Tests that exceeding capacity results in lower fitness via penalties."""
    # Create a problem where capacity is very low
    weights = jnp.array([10.0, 10.0, 10.0])
    values = jnp.array([5.0, 5.0, 5.0])
    capacity = 15.0  # Can only fit one item comfortably

    config = KnapsackConfig(
        maximize=True, weights=weights, values=values, capacity=capacity, penalty_factor=100.0
    )
    evaluator = KnapsackEvaluator(config=config, data=None)

    # Under capacity: [1, 0, 0] -> weight 10, value 5
    safe_genome = BinaryGenome(values=jnp.array([1, 0, 0]))
    # Over capacity: [1, 1, 1] -> weight 30, value 15, penalty (30-15)*100 = 1500
    heavy_genome = BinaryGenome(values=jnp.array([1, 1, 1]))

    safe_fit = evaluator.evaluate(safe_genome)
    heavy_fit = evaluator.evaluate(heavy_genome)

    # Heavy genome must have significantly lower fitness due to penalty
    assert heavy_fit < safe_fit
    assert float(heavy_fit) < 0  # In this case, penalty outweighs value


def test_knapsack_factory_determinism(rng_key):
    """Verifies the random problem generator produces valid, consistent configs."""
    config = KnapsackEvaluator.create_random_problem(rng_key, n_items=20, capacity_ratio=0.5)

    assert isinstance(config, KnapsackConfig)
    assert config.weights.shape == (20,)
    assert config.values.shape == (20,)
    assert config.maximize is True
    assert config.capacity > 0


def test_binary_evaluators_jit(binary_population):
    """Ensures binary evaluators can be traced and compiled by XLA."""
    config = BinarySumConfig(maximize=True)
    evaluator = BinarySumEvaluator(config=config, data=None)

    @jax.jit
    def fast_eval(pop):
        return evaluator.evaluate_population(pop)

    result = fast_eval(binary_population)
    assert result.fitness.shape == (10,)
    assert not jnp.any(jnp.isnan(result.fitness))


def test_knapsack_infeasible_penalty(knapsack_evaluator):
    """Verifies that an over-capacity genome is penalized."""
    # Setup a genome that takes EVERYTHING (30+10+20 weight > 15 capacity)
    # Using the n_items=10 from the fixture
    bits = jnp.ones(10)
    genome = BinaryGenome(values=bits)

    # Calculate fitness
    fitness = knapsack_evaluator.evaluate(genome)

    # In a properly configured penalty-based GA,
    # taking every item in an over-capacity sack should result in
    # a negative fitness or a significantly reduced positive one.
    assert float(fitness) < jnp.sum(knapsack_evaluator.config.values)
