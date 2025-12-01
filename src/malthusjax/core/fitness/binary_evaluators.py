"""Fitness evaluators for binary genomes.

This module provides fitness functions specifically designed for binary genomes,
including classic problems like BinarySum (OneMax) and Knapsack optimization.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from typing import Any

from malthusjax.core.genome.binary_genome import BinaryGenome
from .base import BaseEvaluator, BaseEvaluatorConfig


@struct.dataclass
class BinarySumConfig(BaseEvaluatorConfig):
    """Configuration for BinarySum (OneMax) fitness evaluator.
    
    The OneMax problem is to maximize the number of 1s in a binary string.
    This is a classic benchmark problem in evolutionary computation.
    """
    pass
    

@struct.dataclass
class BinarySumEvaluator(BaseEvaluator[BinaryGenome, BinarySumConfig, Any]):
    """BinarySum (OneMax) fitness evaluator.
    
    Evaluates binary genomes by counting the number of 1s (or 0s).
    This is a simple but important benchmark problem for testing
    evolutionary algorithms on binary representations.
    """
    
    config: BinarySumConfig
    data: Any = struct.field(pytree_node=False, default=None)
        
    def evaluate(self, genome: BinaryGenome) -> float:
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


@struct.dataclass 
class KnapsackConfig(BaseEvaluatorConfig):
    """Configuration for Knapsack problem fitness evaluator.
    
    The 0/1 Knapsack problem: given items with weights and values,
    select a subset that maximizes value while staying within weight capacity.
    """
    weights: jnp.ndarray  # Item weights, shape (n_items,)
    values: jnp.ndarray   # Item values, shape (n_items,)
    capacity: float       # Maximum weight capacity
    penalty_factor: float = 1000.0  # Penalty for exceeding capacity

@struct.dataclass
class KnapsackEvaluator(BaseEvaluator[BinaryGenome, KnapsackConfig, Any]):
    """Knapsack problem fitness evaluator.
    
    Evaluates binary genomes where each bit indicates whether
    to include the corresponding item in the knapsack.
    Maximizes value while penalizing weight constraint violations.
    """
    
    config: KnapsackConfig
    data: Any = struct.field(pytree_node=False, default=None)
        
    def evaluate(self, genome: BinaryGenome) -> float:
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
