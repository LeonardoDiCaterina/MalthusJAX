"""
Roulette Wheel Selection (Fitness-Proportionate).
Refactored for the 'Consumer Paradigm' and Robust Numerical Stability.
"""
from flax import struct
import jax
import jax.numpy as jnp
import chex
from malthusjax.operators.base import BaseSelection


@struct.dataclass
class RouletteWheelSelection(BaseSelection):
    """
    Selects individuals proportional to their fitness.
    
    Robustness Strategy:
    Uses 'Softmax' scaling to handle negative fitness values and minimization.
    Probability = exp(fitness / temperature) / sum(exp(fitness / temperature))
    
    This works for ANY fitness range (positive/negative) and avoids division by zero.
    """
    temperature: float = 1.0  # Controls selection pressure (Lower = greedier)

    def num_keys(self, input_shape: tuple) -> int:
        """Need 1 key for the stochastic choice."""
        return 1

    def __call__(self, keys: chex.PRNGKey, fitness: chex.Array) -> chex.Array:
        """
        Perform robust roulette wheel selection.
        
        Args:
            key: Single PRNG Key.
            fitness: Fitness array. Assumes 'Higher is Better'.
                     (Engine handles minimization by flipping sign before passing here if needed,
                      BUT if raw fitness is passed, softmax handles negatives fine).
        """
        
        # Robust Probability Calculation: Softmax
        # 1. Scale by temperature
        logits = fitness / self.temperature
        
        # 2. Subtract max for numerical stability (prevents overflow in exp)
        logits = logits - jnp.max(logits)
        
        # 3. Calculate probabilities
        # exp_vals = jnp.exp(logits)
        # probs = exp_vals / jnp.sum(exp_vals)
        # However, jax.random.categorical accepts logits directly! 
        # This is more numerically stable than manual probability calculation.

        # jax.random.categorical takes log-probabilities (logits)
        selected_indices = jax.random.categorical(
            keys, 
            logits, 
            shape=(self.num_selections,)
        )
        
        return selected_indices