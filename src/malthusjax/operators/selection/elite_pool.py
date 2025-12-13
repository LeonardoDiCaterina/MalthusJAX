"""
Selection Operators.
Refactored for the 'Consumer Paradigm': Pure index generation.
"""
from flax import struct
import jax
import jax.numpy as jnp
import chex
from malthusjax.operators.base import BaseSelection

from typing import TypeVar

C = TypeVar("C")  # Config Type

@struct.dataclass
class ElitePoolSelection(BaseSelection):
    """
    Elite Pool Selection (Evosax Style).
    1. Identifies the top 'elite_k' individuals.
    2. Uniformly samples 'num_selections' indices from that pool.
    
    Efficient for large populations as it avoids full sorting.
    """
    elite_k: int = struct.field(pytree_node=False, default=10)

    def num_keys(self, input_shape: tuple) -> int:
        """
        We need 1 key to perform the random sampling from the elite pool.
        """
        return 1

    def _select(self, keys: chex.Array, fitness: chex.Array, config: C = None) -> chex.Array:
        """
        Selects parents.
        
        Args:
            keys: A single key (shape (1,) or scalar) derived from the ResourceMap.
            fitness: The population fitness array.
            
        Returns:
            selected_indices: Shape (num_selections,)
        """
        rng = keys[0] if keys.ndim > 1 else keys
        
        # 1. Find indices of the top K best individuals
        # Note: We assume Higher Fitness = Better
        _, best_k_indices = jax.lax.top_k(fitness, self.elite_k)
        
        # 2. Randomly sample from these elite indices
        # We sample with replacement so elites can be picked multiple times
        random_selections = jax.random.randint(
            rng, 
            shape=(self.num_selections,), 
            minval=0, 
            maxval=self.elite_k
        )
        
        # 3. Map back to original population indices
        selected_indices = best_k_indices[random_selections]
        
        return selected_indices
