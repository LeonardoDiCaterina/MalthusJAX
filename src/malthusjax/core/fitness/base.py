"""
Modern fitness evaluator abstractions with JAX-native design.

Provides BaseEvaluator for generic fitness evaluation and specialized
evaluators for different problem types with automatic vectorization.
"""

from typing import TypeVar, Generic, Tuple
from flax import struct  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import chex  # type: ignore

from malthusjax.core.base import BaseGenome, BasePopulation


# Type variables for generic evaluator components
G = TypeVar("G", bound="BaseGenome")
C = TypeVar("C", bound="BaseEvaluatorConfig")  # Config type
D = TypeVar("D")  # Data type


@struct.dataclass
class BaseEvaluatorConfig:
    """
    Base configuration for fitness evaluators.
    
    Ensures all evaluators have a consistent interface for optimization direction.
    """
    maximize: bool = struct.field(pytree_node=False)


@struct.dataclass
class BaseEvaluator(Generic[G, C, D]):
    """
    Abstract base class for fitness evaluators.
    
    Provides automatic vectorization over populations and clean
    separation between single-genome evaluation and batch operations.
    """
    config: C
    data: D

    def evaluate(self, genome: G) -> chex.Array:
        """
        Evaluate a single genome using baked-in data.
        
        Args:
            genome: Single genome to evaluate
            
        Returns:
            Fitness score(s) - can be scalar or array for multi-objective
        """
        raise NotImplementedError

    def evaluate_population(self, population: BasePopulation[G]) -> BasePopulation[G]:
        """
        Evaluate entire population with automatic vectorization.
        
        The population.genes is a batched genome structure (e.g., RealGenome
        with values of shape (pop_size, length)). JAX's vmap automatically
        maps over the first axis of PyTree nodes to extract individual genomes.
        
        Args:
            population: Population with batched genes
            
        Returns:
            Population with updated fitness values of shape (pop_size,)
        """
        # vmap over the batched genome structure
        # JAX automatically handles the PyTree structure of population.genes
        # mapping over axis 0 to extract individual genomes
        fitness_scores = jax.vmap(self.evaluate)(population.genes)
        return population.replace(fitness=fitness_scores)


# Type alias for regression data
RegressionData = Tuple[chex.Array, chex.Array]  # (X, y)