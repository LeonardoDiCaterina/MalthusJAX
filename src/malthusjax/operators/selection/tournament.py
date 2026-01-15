"""
Tournament Selection implementation using the new paradigm.

Implements tournament selection with @struct.dataclass for JIT compilation
and automatic vectorization support.
"""

import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct
import chex
from malthusjax.operators.base import BaseSelection


@struct.dataclass
class TournamentSelection(BaseSelection):
    """
    Tournament Selection.
    Selects the best individual from random subsets of the population.
    Standard for Genetic Algorithms.
    """
    tournament_size: int = struct.field(pytree_node=False, default=3)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """We need 1 key to generate the random tournament batches."""
        return 1

    def _select(self, keys: chex.Array, fitness: chex.Array, config=None) -> chex.Array:
        """
        Selects parents via tournament selection.
        
        Args:
            keys: Random keys array (shape may be (1, 2) or (2,))
            fitness: Fitness values array (pop_size,)
            config: Unused, for interface compatibility
            
        Returns:
            Selected indices array (num_selections,)
        """
        # Normalize keys - handle (1, 2) shape vs (2,) shape
        rng = keys[0] if keys.ndim > 1 else keys
        pop_size = fitness.shape[0]
        
        # 1. Generate Tournament Candidates
        # Shape: (num_selections, tournament_size)
        candidates = jax.random.randint(
            rng, 
            shape=(self.num_selections, self.tournament_size), 
            minval=0, 
            maxval=pop_size
        )
        
        # 2. Retrieve Fitness of Candidates
        # Shape: (num_selections, tournament_size)
        candidate_fitness = jnp.take(fitness, candidates, axis=0)
        
        # 3. Find Winner of each Tournament (Argmax along axis 1)
        winner_local_indices = jnp.argmax(candidate_fitness, axis=1)
        
        # 4. Map back to global population indices
        selected_indices = jnp.take_along_axis(
            candidates, 
            winner_local_indices[:, None], 
            axis=1
        ).squeeze()
        
        return selected_indices