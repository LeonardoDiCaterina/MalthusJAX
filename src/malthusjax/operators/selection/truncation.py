"""
Truncation Selection Operator.
Deterministic selection of the top-k individuals.
"""
from flax import struct
import jax
import jax.numpy as jnp
import chex
from ..base import BaseSelection

@struct.dataclass
class Truncation(BaseSelection):
    """
    Selects the best 'num_selections' individuals purely by rank.
    Extremely fast on GPUs using jax.lax.top_k.
    """
    num_selections: int = struct.field(pytree_node=False)

    def __call__(self, key: chex.PRNGKey, fitness: chex.Array) -> chex.Array:
        """
        Returns indices of the top 'num_selections' individuals.
        """
        # jax.lax.top_k finds the k largest entries (Maximization)
        # Returns (values, indices)
        _, indices = jax.lax.top_k(fitness, self.num_selections)
        
        return indices