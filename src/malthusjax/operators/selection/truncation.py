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

    # --- Identity Card / Kernel Interface ---
    def num_keys(self, params, input_shape) -> int:
        """Deterministic selection: no RNG keys required."""
        return 0

    def get_output_shape(self, params, input_shape):
        """Return the output shape when selecting elites from a population.

        `input_shape` is expected to be `(pop_size, genome_length)`.
        The output shape is `(num_selections, genome_length)`.
        """
        pop_shape = tuple(input_shape)
        if len(pop_shape) < 2:
            raise ValueError("Expected population array with shape (pop_size, genome_length)")
        genome_length = pop_shape[1]
        return (self.num_selections, genome_length)

    def apply_kernel(self, keys, data, params=None):
        """Pure kernel that selects the top `num_selections` individuals.

        Args:
            keys: unused (deterministic operator)
            data: tuple `(population, fitness)` where
                  `population` has shape `(pop_size, genome_length)` and
                  `fitness` has shape `(pop_size,)`.
            params: optional params (ignored)

        Returns:
            selected population array with shape `(num_selections, genome_length)`
        """
        population, fitness = data
        population = jnp.asarray(population)
        fitness = jnp.asarray(fitness)

        # Get top-k indices (maximization)
        _, indices = jax.lax.top_k(fitness, self.num_selections)

        # Gather selected individuals; use jnp.take for gather over axis 0
        selected = jnp.take(population, indices, axis=0)

        return selected