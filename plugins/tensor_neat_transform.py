from typing import Any
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.composable.base import BaseTransform

_field: Any = struct.field

@struct.dataclass
class TensorNeatTransform(BaseTransform[Any]):
    """A custom genotype-to-phenotype transform."""
    
    algorithm: Any = _field(pytree_node=False)

    def transform(self, genome: Any, state: Any = None) -> Any:
        return self.algorithm.transform(state, (genome.values[0], genome.values[1]))

    def transform_population(self, genes: Any, state: Any = None) -> Any:
        # The genes is a tuple of (nodes, conns) in TensorNeatPopulation
        nodes, conns = genes.values[0], genes.values[1]
        
        transformed_pop = jax.vmap(
            lambda s, n, c: self.algorithm.transform(s, (n, c)), in_axes=(None, 0, 0)
        )(state, nodes, conns)
        return transformed_pop
