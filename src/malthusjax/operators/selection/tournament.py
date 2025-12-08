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

    def num_keys(self, input_shape: tuple) -> int:
        """
        We need 1 key to generate the random tournament batches.
        """
        return 1

    def __call__(self, keys: chex.Array, fitness: chex.Array) -> chex.Array:
        
        pop_size = fitness.shape[0]
        
        # 1. Generate Tournament Candidates
        # Shape: (num_selections, tournament_size)
        # We pick random indices from the population
        candidates = jax.random.randint(
            keys, 
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
        # We need to select the specific candidate index that won
        # Advanced Indexing: candidates[row, winner_col]
        
        # Create row indices: [0, 1, 2, ... N-1]
        row_indices = jnp.arange(self.num_selections)
        
        selected_indices = candidates[row_indices, winner_local_indices]
        
        return selected_indices