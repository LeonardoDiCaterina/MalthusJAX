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
    Tournament Selection using the new paradigm.
    
    Selects individuals by running tournaments of specified size.
    Higher fitness individuals have better chance of winning.
    """
    # --- STATIC PARAMS (affect compilation) ---
    tournament_size: int = struct.field(pytree_node=False, default=4)
    
    def __call__(self, key: chex.PRNGKey, fitness: chex.Array) -> chex.Array:
        """
        Run tournament selection.
        
        Args:
            key: PRNG Key
            fitness: Fitness array (pop_size,)
            
        Returns:
            Selected indices (num_selections,)
        """
        return _tournament_selection(key, fitness, self.num_selections, self.tournament_size)


# --- Pure JAX Function ---

def _tournament_selection(
    key: chex.PRNGKey,
    fitness_values: chex.Array,
    number_of_choices: int,
    tournament_size: int
) -> chex.Array:
    """
    Pure JAX function for tournament selection.
    
    Args:
        key: PRNGKey
        fitness_values: 1D fitness array
        number_of_choices: How many individuals to select
        tournament_size: Size of each tournament
        
    Returns:
        1D array of selected indices
    """
    pop_size = fitness_values.shape[0]
    keys = jar.split(key, number_of_choices)
    
    def single_tournament(k):
        # Pick random contestants for this tournament
        contestants = jar.randint(k, (tournament_size,), 0, pop_size)
        # Get their fitness values
        contestant_fitness = fitness_values[contestants]
        # Find winner (highest fitness)
        winner_idx = jnp.argmax(contestant_fitness)
        return contestants[winner_idx]
    
    # Run all tournaments in parallel
    selected_indices = jax.vmap(single_tournament)(keys)
    return selected_indices


__all__ = ["TournamentSelection"]