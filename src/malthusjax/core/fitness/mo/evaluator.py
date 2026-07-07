"""Multi-Objective evaluator interface."""

from typing import cast
import chex
import jax
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluator, G, C, D
from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.mo.population import MOPopulation

@struct.dataclass
class BaseMOEvaluator(BaseEvaluator[G, C, D]):
    """JAX-native Multi-Objective fitness evaluation interface.

    Extends the standard evaluator to compute multiple fitness objectives.
    Returns a specialized `MOPopulation` which automatically computes Pareto 
    ranks and crowding distances upon initialization.
    """

    def evaluate(self, genome: G) -> chex.Array:
        """Compute multi-objective fitness vector for a single genome.
        
        Returns:
            A chex.Array of shape (num_objectives,) containing the fitnesses.
        """
        raise NotImplementedError

    def evaluate_population(self, population: BasePopulation[G]) -> MOPopulation:
        """Vectorized population evaluation via :func:`jax.vmap`.
        
        Evaluates fitness objectives and upgrades the standard population into
        a smart `MOPopulation` containing non-dominated sorting metrics.
        """
        fitness_scores = jax.vmap(self.evaluate)(population.genes)
        
        base_pop = cast(BasePopulation[G], population.replace(fitness=fitness_scores))
        
        # MOPopulation handles its own pareto fronts and crowding distance logic natively
        return MOPopulation.from_evaluated(base_pop, maximize=self.config.maximize)
