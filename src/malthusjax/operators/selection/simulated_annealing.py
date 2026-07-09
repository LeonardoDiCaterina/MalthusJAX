from typing import Any
import chex
import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class SimulatedAnnealingSelector:
    """Thermodynamic selection via Simulated Annealing.
    
    Operates as a 1+1 Evolutionary Strategy evaluated in parallel across the batch.
    Compares parents directly to their offspring.
    """
    initial_temperature: float = 1.0
    cooling_rate: float = 0.99
    
    def _select_one(self, key: chex.PRNGKey, parent_fit: float, offspring_fit: float, T: float) -> jnp.bool_:
        """Tier 1: Scalar logic for a single Markov Chain transition.
        
        Returns True if the offspring is accepted.
        """
        delta_e = offspring_fit - parent_fit
        
        # Optimal Log-Space Metropolis-Hastings Acceptance:
        # If delta_e < 0 (better), -delta_e/T > 0. log_dice is always < 0, so ALWAYS True.
        # If delta_e > 0 (worse), we check if log_dice < -delta_e/T (exact Boltzmann prob).
        log_dice = jnp.log(jax.random.uniform(key))
        
        return log_dice < -delta_e / (T + 1e-8)
        
    def __call__(
        self, keys: chex.PRNGKey, parent_pop: Any, offspring_pop: Any, generation: int
    ) -> jnp.ndarray:
        """Tier 3: Explicit vmap over the population.
        
        Returns a boolean mask of shape (pop_size,) where True means the offspring
        was accepted and replaces the parent.
        """
        T = self.initial_temperature * (self.cooling_rate ** generation)
        
        # Split keys for the batch
        pop_size = parent_pop.fitness.shape[0]
        batch_keys = jax.random.split(keys, pop_size)
        
        # Explicitly vmap the scalar function across the batch
        vmap_select = jax.vmap(self._select_one, in_axes=(0, 0, 0, None))
        accept_mask = vmap_select(batch_keys, parent_pop.fitness, offspring_pop.fitness, T)
        
        return accept_mask
