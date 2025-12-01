"""
Tests for fitness evaluators (Level 1).

Tests the new evaluator paradigm with evaluate() and evaluate_population() methods.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig, KnapsackEvaluator, KnapsackConfig
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation

from tests.conftest import assert_jit_compilable


class TestBinarySumFitnessEvaluator:
    """Test binary sum fitness evaluator with new paradigm."""

    def test_single_evaluation(self):
        """Test evaluation of a single binary genome."""
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumEvaluator(config=config, data=None)
        
        genome = BinaryGenome(bits=jnp.array([1, 0, 1, 1, 0]))  # Sum = 3
        
        fitness = evaluator.evaluate(genome)
        assert fitness == 3.0

    def test_batch_evaluation(self, rng_key):
        """Test batch evaluation of binary genomes."""
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumEvaluator(config=config, data=None)
        
        # Create batch of binary genomes
        batch_size = 5
        length = 10
        genome_config = BinaryGenomeConfig(length=length)
        
        # Create population using NEW paradigm
        population = BinaryPopulation.init_random(rng_key, genome_config, size=batch_size)
        
        # Use evaluate_population which returns population with fitness
        evaluated_pop = evaluator.evaluate_population(population)
        
        assert evaluated_pop.fitness.shape == (batch_size,)
        assert jnp.all(evaluated_pop.fitness >= 0)
        assert jnp.all(evaluated_pop.fitness <= length)

    def test_known_values(self):
        """Test evaluator on known binary patterns."""
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumEvaluator(config=config, data=None)
        
        # All zeros
        all_zeros = BinaryGenome(bits=jnp.zeros(5, dtype=jnp.int32))
        assert evaluator.evaluate(all_zeros) == 0.0
        
        # All ones
        all_ones = BinaryGenome(bits=jnp.ones(5, dtype=jnp.int32))
        assert evaluator.evaluate(all_ones) == 5.0
        
        # Mixed pattern
        mixed = BinaryGenome(bits=jnp.array([1, 0, 1, 0, 1]))
        assert evaluator.evaluate(mixed) == 3.0

    @pytest.mark.jit
    def test_jit_compatibility(self):
        """Test that evaluator functions are JIT compilable."""
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumEvaluator(config=config, data=None)
        
        # Test single evaluation JIT
        genome = BinaryGenome(bits=jnp.array([1, 0, 1, 1, 0]))
        jit_single = jax.jit(evaluator.evaluate)
        fitness = jit_single(genome)
        assert fitness == 3.0
        
        # Test batch evaluation JIT with population
        genome_config_batch = BinaryGenomeConfig(length=3)
        population = BinaryPopulation.init_random(jax.random.PRNGKey(42), genome_config_batch, size=3)
        # Override with known values for testing
        population = population.replace(
            genes=population.genes.replace(
                bits=jnp.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
            )
        )
        jit_batch = jax.jit(evaluator.evaluate_population)
        evaluated_pop = jit_batch(population)
        expected = jnp.array([2.0, 1.0, 3.0])
        assert jnp.allclose(evaluated_pop.fitness, expected)

    def test_pure_function_interface(self):
        """Test the NEW paradigm batch evaluation interface."""
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumEvaluator(config=config, data=None)
        
        # Create a small population for batch evaluation
        genome_config = BinaryGenomeConfig(length=5)
        population = BinaryPopulation.init_random(jax.random.PRNGKey(42), genome_config, size=2)
        # Override with known values
        population = population.replace(
            genes=population.genes.replace(
                bits=jnp.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
            )
        )
        
        evaluated_pop = evaluator.evaluate_population(population)
        expected = jnp.array([3.0, 2.0])
        assert jnp.allclose(evaluated_pop.fitness, expected)
        
        # Should be JIT compilable
        jit_batch = jax.jit(evaluator.evaluate_population)
        evaluated_pop_jit = jit_batch(population)
        assert jnp.allclose(evaluated_pop.fitness, evaluated_pop_jit.fitness)

    def test_minimize_mode(self):
        """Test that minimize mode works correctly."""
        config = BinarySumConfig(maximize=False)  # Minimize = count zeros
        evaluator = BinarySumEvaluator(config=config, data=None)
        
        # All zeros should give best fitness (5.0)
        all_zeros = BinaryGenome(bits=jnp.zeros(5, dtype=jnp.int32))
        assert evaluator.evaluate(all_zeros) == 5.0
        
        # All ones should give worst fitness (0.0)
        all_ones = BinaryGenome(bits=jnp.ones(5, dtype=jnp.int32))
        assert evaluator.evaluate(all_ones) == 0.0


class TestKnapsackEvaluator:
    """Test knapsack problem evaluator."""

    def test_basic_knapsack(self):
        """Test basic knapsack evaluation."""
        # Simple knapsack: 3 items
        weights = jnp.array([2.0, 3.0, 4.0])
        values = jnp.array([3.0, 4.0, 5.0])
        capacity = 5.0
        
        config = KnapsackConfig(
            weights=weights,
            values=values,
            capacity=capacity,
            maximize=True
        )
        evaluator = KnapsackEvaluator(config=config, data=None)
        
        # Select items 0 and 1 (weight=5, value=7) - exactly at capacity
        genome = BinaryGenome(bits=jnp.array([1, 1, 0]))
        fitness = evaluator.evaluate(genome)
        assert fitness == 7.0  # Should get full value


@pytest.mark.slow
class TestFitnessPerformance:
    """Performance tests for fitness evaluators."""

    @pytest.mark.jit
    def test_large_batch_evaluation(self, rng_key):
        """Test evaluation on large batches."""
        batch_size = 100  # Smaller for faster tests
        length = 20
        
        # Binary sum evaluator
        config = BinarySumConfig(maximize=True)
        binary_evaluator = BinarySumEvaluator(config=config, data=None)
        
        # Create proper population
        genome_config = BinaryGenomeConfig(length=length)
        population = BinaryPopulation.init_random(rng_key, genome_config, size=batch_size)
        
        evaluated_pop = binary_evaluator.evaluate_population(population)
        assert evaluated_pop.fitness.shape == (batch_size,)
        assert jnp.all(evaluated_pop.fitness >= 0)
        assert jnp.all(evaluated_pop.fitness <= length)

    @pytest.mark.jit
    def test_vmap_compatibility(self, rng_key):
        """Test that evaluators work with vmap."""
        # Test binary evaluator with vmap
        config = BinarySumConfig(maximize=True)
        binary_evaluator = BinarySumEvaluator(config=config, data=None)
        
        # Create population
        genome_config = BinaryGenomeConfig(length=10)
        population = BinaryPopulation.init_random(rng_key, genome_config, size=5)
        
        # Use evaluate_population (which uses vmap internally)
        evaluated_pop = binary_evaluator.evaluate_population(population)
        
        # Manual vmap should give same results
        vmap_fitness = jax.vmap(binary_evaluator.evaluate)(population.genes)
        
        assert jnp.allclose(evaluated_pop.fitness, vmap_fitness)

    def test_evaluator_consistency(self, rng_key):
        """Test that different evaluation methods give consistent results."""
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumEvaluator(config=config, data=None)
        
        # Create test data
        genome_config = BinaryGenomeConfig(length=10)
        genome = BinaryGenome.random_init(rng_key, config=genome_config)
        
        # Single evaluation
        fitness1 = evaluator.evaluate(genome)
        
        # Batch evaluation with single item
        population = BinaryPopulation.init_random(rng_key, genome_config, size=1)
        population = population.replace(genes=population.genes.replace(bits=genome.bits.reshape(1, -1)))
        evaluated_pop = evaluator.evaluate_population(population)
        fitness2 = evaluated_pop.fitness[0]
        
        assert jnp.isclose(fitness1, fitness2)
