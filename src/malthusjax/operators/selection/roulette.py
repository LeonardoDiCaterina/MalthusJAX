"""
Roulette Wheel Selection implementation for MalthusJAX.
"""
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
import chex # type: ignore
from flax import struct # type: ignore
from malthusjax.operators.base import BaseSelection


@struct.dataclass
class RouletteWheelSelection(BaseSelection):
    """
    Selects individuals using roulette wheel (fitness-proportionate) selection.
    Assumes fitness values are non-negative.
    
    Uses the new paradigm with @struct.dataclass for immutable, JIT-friendly operations.
    """

    def __call__(self, key: chex.PRNGKey, fitness: chex.Array) -> chex.Array:
        """
        Perform roulette wheel selection.
        
        Args:
            key: PRNG Key
            fitness: Fitness array (pop_size,)
            
        Returns:
            Selected indices (num_selections,)
        """
        return _roulette_selection(key, fitness, self.num_selections)

# --- Pure JAX Function ---

def _roulette_selection(
    key: chex.PRNGKey,
    fitness_values: chex.Array,
    number_of_choices: int
) -> chex.Array:
    """
    Pure JAX function for roulette wheel selection.
    
    Args:
        key: PRNGKey
        fitness_values: 1D array of non-negative fitnesses.
        number_of_choices: Static int. Total individuals to select.
        
    Returns:
        1D array of indices for the selected individuals.
    """
    # 1. Calculate probabilities
    fitness_sum = jnp.sum(fitness_values)
    # Add epsilon to avoid division by zero if all fitnesses are 0
    probabilities = (fitness_values + 1e-6) / (fitness_sum + 1e-6)
    
    # 2. Choose indices based on probabilities
    selected_indices = jar.choice(
        key,
        jnp.arange(fitness_values.shape[0]),
        shape=(number_of_choices,),
        p=probabilities
    )
    return selected_indices