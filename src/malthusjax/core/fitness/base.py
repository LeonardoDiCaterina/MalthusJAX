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
        
        Args:
            population: Population to evaluate
            
        Returns:
            Population with updated fitness values
        """
        # Vectorize over genes (axis 0), keep self constant (including data)
        fitness_scores = jax.vmap(self.evaluate, in_axes=(0,))(population.genes)
        return population.replace(fitness=fitness_scores)


# Type alias for regression data
RegressionData = Tuple[chex.Array, chex.Array]  # (X, y)