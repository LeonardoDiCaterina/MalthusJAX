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