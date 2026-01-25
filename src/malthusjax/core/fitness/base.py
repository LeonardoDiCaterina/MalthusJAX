from __future__ import annotations
from typing import Any, Generic, Tuple, TypeVar, cast

import chex
import jax
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation

G = TypeVar("G", bound="BaseGenome")
C = TypeVar("C", bound="BaseEvaluatorConfig")  # Config type
D = TypeVar("D")  # Data type (e.g., training data, environment params)


@struct.dataclass
class BaseEvaluatorConfig:
    """
    Base configuration for fitness evaluators.
    
    Attributes:
        maximize: If True, higher fitness values are better. 
                 Crucial for sorting and selection logic.
    """
    maximize: bool = struct.field(pytree_node=False) # type: ignore[no-untyped-call]


@struct.dataclass
class BaseEvaluator(Generic[G, C, D]):
    """
    Abstract base class for JAX-native fitness evaluation.
    
    This architecture separates the logic for evaluating a single individual 
    from the mechanics of batch processing. It relies on JAX's 'vmap' 
    to transform the single-individual 'evaluate' method into a high-performance 
    parallel evaluator for an entire population.
    """
    config: C
    data: D

    def evaluate(self, genome: G) -> chex.Numeric:
        """
        Calculates the fitness score for a single individual.
        
        Args:
            genome: A single instance of G (e.g., RealGenome with 1D values).
            
        Returns:
            A scalar fitness score or an array for multi-objective problems.
        """
        raise NotImplementedError

    def evaluate_population(self, population: BasePopulation[G]) -> BasePopulation[G]:
        """
        Evaluates an entire population using automatic vectorization.
        
        This method leverages the Struct-of-Arrays (SoA) nature of the population.
        The 'genes' PyTree is unrolled along the leading dimension, and each 
        unrolled genome is passed to the 'evaluate' method in parallel.
        
        Args:
            population: A population instance containing batched genes.
            
        Returns:
            A new population instance with updated 'fitness' values.
        """
        # We vmap over the genes. In our architecture, population.genes 
        # is a 'lifted' Genome where leaf arrays have a leading dim of pop_size.
        fitness_scores = jax.vmap(self.evaluate)(population.genes)
        
        # Cast population to Any to access .replace, then back to BasePopulation[G]
        return cast(BasePopulation[G], cast(Any, population).replace(fitness=fitness_scores))


# Type-safe alias for regression data (Features, Targets)
RegressionData = Tuple[chex.Array, chex.Array]