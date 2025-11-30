"""Fitness evaluators for binary genomes.

This module provides fitness functions specifically designed for binary genomes,
including classic problems like BinarySum (OneMax) and Knapsack optimization.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from ..base import BaseGenome
from ..genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation


@struct.dataclass
class BinarySumConfig:
    """Configuration for BinarySum (OneMax) fitness evaluator.
    
    The OneMax problem is to maximize the number of 1s in a binary string.
    This is a classic benchmark problem in evolutionary computation.
    """
    # Marked as static (pytree_node=False) to allow use in Python control flow within JIT
    maximize: bool = struct.field(pytree_node=False, default=True)
    

@struct.dataclass
class BinarySumEvaluator:
    """BinarySum (OneMax) fitness evaluator.
    
    Evaluates binary genomes by counting the number of 1s (or 0s).
    This is a simple but important benchmark problem for testing
    evolutionary algorithms on binary representations.
    """
    
    config: BinarySumConfig
        
    def evaluate_single(self, genome: BinaryGenome) -> float:
        """Evaluate a single binary genome.
        
        Args:
            genome: BinaryGenome to evaluate
            
        Returns:
            Fitness value (number of ones or zeros)
        """
        ones_count = jnp.sum(genome.bits).astype(jnp.float32)
        if self.config.maximize:
            return ones_count
        else:
            length = jnp.array(len(genome.bits), dtype=jnp.float32)
            return length - ones_count
            
    def evaluate_batch(self, population: BinaryPopulation) -> jnp.ndarray:
        """Evaluate a population of binary genomes using NEW paradigm."""
        # Use vmap for efficient batch evaluation - return JAX array for JIT compatibility
        return jax.vmap(self.evaluate_single)(population.genes)


@struct.dataclass 
class KnapsackConfig:
    """Configuration for Knapsack problem fitness evaluator.
    
    The 0/1 Knapsack problem: given items with weights and values,
    select a subset that maximizes value while staying within weight capacity.
    """
    weights: jnp.ndarray  # Item weights, shape (n_items,)
    values: jnp.ndarray   # Item values, shape (n_items,)
    capacity: float       # Maximum weight capacity
    penalty_factor: float = 1000.0  # Penalty for exceeding capacity
    maximize: bool = struct.field(pytree_node=False, default=True)

@struct.dataclass
class KnapsackEvaluator:
    """Knapsack problem fitness evaluator.
    
    Evaluates binary genomes where each bit indicates whether
    to include the corresponding item in the knapsack.
    Maximizes value while penalizing weight constraint violations.
    """
    
    config: KnapsackConfig
        
    def evaluate_single(self, genome: BinaryGenome) -> float:
        """Evaluate a single binary genome for knapsack fitness.
        
        Args:
            genome: BinaryGenome representing item selection
            
        Returns:
            Fitness value (total value minus capacity penalty)
        """
        if len(genome.bits) != len(self.config.weights):
            raise ValueError(f"Genome length {len(genome.bits)} != number of items {len(self.config.weights)}")
            
        # Calculate total weight and value
        selected_weights = genome.bits * self.config.weights
        selected_values = genome.bits * self.config.values
        
        total_weight = jnp.sum(selected_weights)
        total_value = jnp.sum(selected_values)
        
        # Apply penalty for exceeding capacity
        # Use jnp.where for JIT compatibility instead of if/else
        penalty = (total_weight - self.config.capacity) * self.config.penalty_factor
        
        # If weight > capacity, return value - penalty, else return value
        # We use jnp.where to handle the conditional logic in a trace-safe way
        is_over = total_weight > self.config.capacity
        fitness = jnp.where(is_over, total_value - penalty, total_value)
        
        return fitness
            
    def evaluate_batch(self, population: BinaryPopulation) -> jnp.ndarray:
        """Evaluate a population of binary genomes."""
        fitness_fn = self.get_tensor_fitness_function()
        fitness_values = fitness_fn(population.genes.bits)
        return fitness_values
            
    def get_tensor_fitness_function(self):
        """Get pure JAX function for batch evaluation."""
        
        def _knapsack_fitness(bits_batch: jnp.ndarray) -> jnp.ndarray:
            """Pure JAX function for knapsack fitness.
            
            Args:
                bits_batch: Shape (batch_size, n_items) binary array
                
            Returns:
                Fitness values of shape (batch_size,)
            """
            # Calculate weights and values for all solutions
            weights_batch = bits_batch * self.config.weights[None, :]  # (batch_size, n_items)
            values_batch = bits_batch * self.config.values[None, :]    # (batch_size, n_items)
            
            total_weights = jnp.sum(weights_batch, axis=1)  # (batch_size,)
            total_values = jnp.sum(values_batch, axis=1)    # (batch_size,)
            
            # Apply penalty for exceeding capacity
            over_capacity = jnp.maximum(0, total_weights - self.config.capacity)
            penalties = over_capacity * self.config.penalty_factor
            
            return total_values - penalties
            
        return _knapsack_fitness
        
    @staticmethod
    def create_random_problem(key: jnp.ndarray, n_items: int, 
                            capacity_ratio: float = 0.5) -> 'KnapsackConfig':
        """Create a random knapsack problem instance.
        
        Args:
            key: JAX random key
            n_items: Number of items
            capacity_ratio: Capacity as fraction of total weight
            
        Returns:
            KnapsackConfig for the random problem
        """
        key1, key2, key3 = jr.split(key, 3)
        
        # Random weights and values
        weights = jr.uniform(key1, (n_items,), minval=1.0, maxval=20.0)
        values = jr.uniform(key2, (n_items,), minval=1.0, maxval=50.0)
        
        # Set capacity as fraction of total weight
        total_weight = jnp.sum(weights)
        capacity = capacity_ratio * total_weight
        
        return KnapsackConfig(
            weights=weights,
            values=values,
            capacity=capacity
        )
