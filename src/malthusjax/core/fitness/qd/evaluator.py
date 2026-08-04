"""Quality-Diversity evaluator interface."""

from typing import Tuple, Any
import chex
import jax
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluator, G, C, D
from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.qd.population import QDPopulation

@struct.dataclass
class BaseQDEvaluator(BaseEvaluator[G, C, D]):
    """JAX-native Quality-Diversity fitness evaluation interface.

    Extends the standard evaluator to compute behavioral descriptors alongside
    fitness. Returns a specialized `QDPopulation` which safely stores the
    descriptors in its `.info` dictionary, making it a drop-in replacement
    for standard scalar evaluators.
    """

    def evaluate(self, genome: G) -> chex.Numeric:
        """Fallback standard evaluation that simply discards descriptors.
        
        This satisfies the Liskov Substitution Principle if this evaluator
        is used in a standard GeneticEngine.
        """
        fitness, _ = self.evaluate_qd(genome)
        return fitness

    def evaluate_qd(self, genome: G) -> Tuple[chex.Numeric, chex.Array]:
        """Compute fitness and behavioral descriptor for a single genome.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def evaluate_population(self, population: BasePopulation[G]) -> QDPopulation[G]:
        """Vectorized population evaluation via :func:`jax.vmap`.
        
        Evaluates both fitness and descriptors, returning a QDPopulation.
        """
        fitness_scores, descriptors = jax.vmap(self.evaluate_qd)(population.genes)
        
        # Merge info safely, inserting descriptors
        new_info = dict(population.info) if population.info else {}
        new_info["descriptors"] = descriptors
        
        return QDPopulation(
            genes=population.genes,
            fitness=fitness_scores,
            config=population.config,
            info=new_info
        )


@struct.dataclass
class SimpleQDEvaluator(BaseQDEvaluator[G, Any, Any]):
    """Standard wrapper for quality-diversity objective functions.
    
    Expects an objective function that takes a single array of genes 
    and returns a tuple of (fitness, descriptors) or (fitness, descriptors, extra).
    """
    objective_function: Any = struct.field(pytree_node=False)

    def evaluate_qd(self, genome: G) -> Tuple[chex.Numeric, chex.Array]:
        # Unwrap genome to underlying values if applicable
        gene_values = getattr(genome, "values", genome)
        result = self.objective_function(gene_values)
        
        fitness = result[0]
        descriptors = result[1]

        # Evosax / Composer convention is lower-is-better by default, 
        # but QD natively maximizes. If maximize=False, we negate the fitness.
        if self.config is not None and not self.config.maximize:
            fitness = -1.0 * fitness
            
        return fitness, descriptors
