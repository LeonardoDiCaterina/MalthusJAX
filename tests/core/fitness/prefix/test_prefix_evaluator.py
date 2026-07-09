"""Tests for BasePrefixEvaluator and LinearGPPrefixEvaluator."""

import jax
import jax.numpy as jnp

from malthusjax.core.fitness.prefix.linear_gp_prefix_evaluator import (
    LinearGPPrefixEvaluator,
    LinearGPPrefixEvaluatorConfig,
)
from malthusjax.core.genome.prefix.genome import BasePrefixAwareGenome, PrefixGenomeConfig
from malthusjax.core.genome.prefix.population import PrefixPopulation


def test_linear_gp_prefix_evaluator_shape():
    """Test that the prefix evaluator returns the correct shape (L,)."""
    key = jax.random.PRNGKey(42)
    length = 5
    num_inputs = 1
    
    config = PrefixGenomeConfig(length=length, num_inputs=num_inputs, num_ops=2, max_arity=2)
    eval_config = LinearGPPrefixEvaluatorConfig(length=length, num_inputs=num_inputs)
    
    # Simple data: y = 2x
    X = jnp.array([[1.0], [2.0], [3.0]])
    y = jnp.array([2.0, 4.0, 6.0])
    data = (X, y)
    
    evaluator = LinearGPPrefixEvaluator(config=eval_config, data=data)
    genome = BasePrefixAwareGenome.random_init(key, config)
    
    # Test evaluate_all_prefixes
    prefix_fitness = evaluator.evaluate_all_prefixes(genome)
    assert prefix_fitness.shape == (length,)
    
    # Test scalar Liskov evaluate (should be the minimum of prefix_fitness)
    scalar_fitness = evaluator.evaluate(genome)
    assert scalar_fitness.shape == ()
    assert jnp.isclose(scalar_fitness, jnp.min(prefix_fitness))


def test_prefix_evaluator_population():
    """Test that evaluate_population populates the PrefixPopulation fields correctly."""
    key = jax.random.PRNGKey(42)
    pop_size = 4
    length = 5
    num_inputs = 1
    
    config = PrefixGenomeConfig(length=length, num_inputs=num_inputs, num_ops=2, max_arity=2)
    eval_config = LinearGPPrefixEvaluatorConfig(length=length, num_inputs=num_inputs)
    
    X = jnp.array([[1.0], [2.0]])
    y = jnp.array([2.0, 4.0])
    data = (X, y)
    
    evaluator = LinearGPPrefixEvaluator(config=eval_config, data=data)
    pop = PrefixPopulation.init_random(key, config, size=pop_size)
    
    # Before evaluation
    assert pop.prefix_fitness is None
    assert pop.winning_prefix_idx is None
    assert jnp.all(pop.fitness == -jnp.inf)
    
    # Evaluate
    evaluated_pop = evaluator.evaluate_population(pop)
    
    # Check fields
    assert evaluated_pop.prefix_fitness.shape == (pop_size, length)
    assert evaluated_pop.winning_prefix_idx.shape == (pop_size,)
    assert evaluated_pop.fitness.shape == (pop_size,)
    
    # Check consistency
    for i in range(pop_size):
        win_idx = evaluated_pop.winning_prefix_idx[i]
        assert jnp.isclose(
            evaluated_pop.fitness[i], 
            evaluated_pop.prefix_fitness[i, win_idx]
        )
        assert jnp.isclose(
            evaluated_pop.fitness[i],
            jnp.min(evaluated_pop.prefix_fitness[i])
        )
