from typing import Any
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.composable.base import BaseTransform

@struct.dataclass
class CustomTransform(BaseTransform[Any]):
    """A custom genotype-to-phenotype transform."""
    
    def transform(self, genome: Any, state: Any = None) -> Any:
        # TODO: Implement your transformation logic here
        return genome

    # Optional: Override transform_population if the transform 
    # must be applied at the population level (e.g. TensorNEAT).
    # def transform_population(self, genes: Any, state: Any = None) -> Any:
    #     return super().transform_population(genes, state)
