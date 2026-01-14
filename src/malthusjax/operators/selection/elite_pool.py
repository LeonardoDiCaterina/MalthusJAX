"""
Selection Operators.
Refactored for the 'Consumer Paradigm': Pure index generation.
"""
from flax import struct
import jax
import jax.lax
import jax.random
import chex
from malthusjax.operators.base import BaseSelection

from typing import TypeVar

C = TypeVar("C")  # Config Type

@struct.dataclass
class ElitePoolSelection(BaseSelection):
    """
    Elite Pool Selection (High Performance).
    Uses jax.lax.top_k for O(N log K) efficiency instead of O(N log N) sorting.
    """
    elite_k: int = struct.field(pytree_node=False, default=10)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(self, keys: chex.Array, fitness: chex.Array, config: C = None) -> chex.Array:
        """
        Selects parents from the top 'elite_k' best individuals.
        """
        rng = keys[0]
        
        # 1. Find indices of the top K best individuals (Efficiency Win)
        # top_k returns values and indices. We only need indices.
        # Note: top_k sorts largest to smallest, which is perfect for maximization.
        _, best_k_indices = jax.lax.top_k(fitness, self.elite_k)
        
        # 2. Randomly sample from these elite indices
        # We sample with replacement so the best elites can be picked multiple times.
        random_selections = jax.random.randint(
            rng, 
            shape=(self.num_selections,), 
            minval=0, 
            maxval=self.elite_k
        )
        
        # 3. Gather final parent indices
        selected_indices = best_k_indices[random_selections]
        
        return selected_indices