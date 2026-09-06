from typing import Any
import chex
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.composable.base import BaseOptimizationEnvironment

@struct.dataclass
class KnapsackEnv(BaseOptimizationEnvironment):
    """A custom environment."""

    # Add custom static fields (e.g. data for optimization instance)
    # my_data: Any = struct.field(pytree_node=False, default=None)

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        # TODO: Implement the evaluation of a solution for OptimizationTask.
        return jnp.sum(solution)

