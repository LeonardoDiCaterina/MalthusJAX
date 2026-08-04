from typing import Any, Callable, cast

from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
from malthusjax.core.genome.tensorneat_genome import TensorNeatGenome, TensorNeatPopulation


@struct.dataclass
class TensorNeatQDEvaluator(BaseQDEvaluator[TensorNeatGenome, Any, Any]):
    """
    Evaluates a population of TensorNEAT genomes in a QD setting.
    It expects an objective_function that computes fitness and descriptors
    given the batched nodes and conns.

    Signature for objective_function:
    (nodes: chex.Array, conns: chex.Array) -> Tuple[fitnesses, descriptors]
    """

    objective_function: Callable[..., Any] = struct.field(pytree_node=False)

    def evaluate_population(self, population: BasePopulation[TensorNeatGenome]) -> TensorNeatPopulation:
        """
        Executes the objective function on the batched TensorNEAT graphs.
        """
        tn_pop = cast(TensorNeatPopulation, population)
        genes = tn_pop.genes
        nodes, conns = genes.values[0], genes.values[1]

        # We expect the objective function to be already vmapped or natively handle the batch dimension
        fitnesses, descriptors = self.objective_function(nodes, conns)

        # Update the population info with the calculated descriptors
        info = dict(tn_pop.info)
        info["descriptors"] = descriptors

        res = tn_pop.spawn_offspring(new_genes=tn_pop.genes, fitness=fitnesses, info=info)
        return cast(TensorNeatPopulation, res)
