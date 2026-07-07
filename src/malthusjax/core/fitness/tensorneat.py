import jax
import jax.numpy as jnp
from typing import Any, Callable
from flax import struct

from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
from malthusjax.core.genome.tensorneat_genome import TensorNeatPopulation

@struct.dataclass
class TensorNeatQDEvaluator(BaseQDEvaluator):
    """
    Evaluates a population of TensorNEAT genomes in a QD setting.
    It expects an objective_function that computes fitness and descriptors
    given the batched nodes and conns.
    
    Signature for objective_function:
    (nodes: chex.Array, conns: chex.Array) -> Tuple[fitnesses, descriptors]
    """
    objective_function: Callable = struct.field(pytree_node=False)

    def evaluate_population(self, population: TensorNeatPopulation) -> TensorNeatPopulation:
        """
        Executes the objective function on the batched TensorNEAT graphs.
        """
        nodes, conns = population.genes.values
        
        # We expect the objective function to be already vmapped or natively handle the batch dimension
        fitnesses, descriptors = self.objective_function(nodes, conns)
        
        # Update the population info with the calculated descriptors
        info = dict(population.info)
        info["descriptors"] = descriptors
        
        return population.spawn_offspring(
            new_genes=population.genes,
            fitness=fitnesses,
            info=info
        )
