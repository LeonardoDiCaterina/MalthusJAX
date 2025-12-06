"""
Elite Pool Selection (Evosax Style).
Selects 'num_selections' parents by uniformly sampling from the top 'elite_k' individuals.
"""
from flax import struct
import jax
import jax.numpy as jnp
import chex
from ..base import BaseSelection

@struct.dataclass
class ElitePoolSelection(BaseSelection):
    """
    1. Identifies the top 'elite_k' individuals.
    2. Randomly samples 'num_selections' indices from that elite pool.
    
    This mimics evosax's 'SimpleGA' selection logic exactly.
    """
    num_selections: int = struct.field(pytree_node=False)
    elite_k: int = struct.field(pytree_node=False) # e.g., 1000 for 1M pop (0.1%)

    def __call__(self, key: chex.PRNGKey, fitness: chex.Array) -> chex.Array:
        # 1. Find indices of the top K best individuals
        # (Assuming maximization)
        _, best_k_indices = jax.lax.top_k(fitness, self.elite_k)
        
        # 2. Randomly sample from these indices to fill the parent buffer
        # This allows the best individuals to be picked multiple times
        sample_key, _ = jax.random.split(key)
        
        # We want to pick 'num_selections' indices from 'best_k_indices'
        # random.choice behaves like: pool[randint(0, len(pool))]
        random_selections = jax.random.randint(
            sample_key, 
            shape=(self.num_selections,), 
            minval=0, 
            maxval=self.elite_k
        )
        
        # Map back to original population indices
        selected_indices = best_k_indices[random_selections]
        
        return selected_indices

    # --- Identity Card / Kernel Interface ---
    def num_keys(self, input_shape) -> int:
        """ElitePoolSelection requires RNG to sample from the elite pool.

        Signature matches `BaseSelection.num_keys(input_shape)`.
        """
        return 1

    def get_output_shape(self, input_shape):
        """Return shape of the selected population rows.

        `input_shape` is expected to be the population shape `(pop_size, genome_length)`.
        The kernel returns rows with shape `(num_selections, genome_length)`.
        """
        pop_shape = tuple(input_shape)
        if len(pop_shape) < 2:
            raise ValueError("Expected population shape (pop_size, genome_length)")
        genome_length = pop_shape[1]
        return (self.num_selections, genome_length)

    def apply_kernel(self, keys, data, params=None):
        """Kernel: select `num_selections` parents by sampling from top-`elite_k`.

        Args:
            keys: PRNGKey used for sampling (single key expected)
            data: tuple `(population, fitness)`
            params: optional

        Returns:
            selected population rows with shape `(num_selections, genome_length)`
        """
        population, fitness = data
        population = jnp.asarray(population)
        fitness = jnp.asarray(fitness)

        # Identify elite pool indices
        _, best_k_indices = jax.lax.top_k(fitness, self.elite_k)

        # Sample with replacement from elite pool using provided key
        sample_indices = jax.random.randint(
            keys,
            shape=(self.num_selections,),
            minval=0,
            maxval=self.elite_k,
        )

        selected_indices = best_k_indices[sample_indices]

        # Gather rows from population
        selected = jnp.take(population, selected_indices, axis=0)

        return selected