"""
ABLATION: Selection Operators with Zero Key Allocation Cost.
These operators bypass the resource allocator and generate keys internally using jax.random.fold_in.
Used to quantify the overhead of the static resource allocation framework.
"""
from typing import Tuple, TypeVar
from flax import struct
import jax
import jax.numpy as jnp
import chex
from malthusjax.operators.base import BaseSelection

C = TypeVar("C")  # Config Type


@struct.dataclass
class AblationElitePoolSelection(BaseSelection):
    """
    ABLATION: Elite Pool Selection with internal key generation.
    num_keys() returns 1 to engage allocator but generate keys internally.
    Keys are generated internally using jax.random.fold_in.
    """
    elite_k: int = struct.field(pytree_node=False, default=10)
    seed: int = struct.field(pytree_node=False, default=42)

    def num_keys(self, input_shape: tuple = None) -> int:
        """
        ABLATION: Return 1 to engage allocator but generate keys internally.
        """
        return 1

    def _select(self, keys: chex.Array, fitness: chex.Array, config: C = None) -> chex.Array:
        """
        Selects parents (unchanged from standard version).
        
        Args:
            keys: A single key (shape (1,) or scalar) derived from the ResourceMap.
            fitness: The population fitness array.
            
        Returns:
            selected_indices: Shape (num_selections,)
        """
        rng = keys[0] if keys.ndim > 1 else keys
        
        # 1. Find indices of the top K best individuals
        _, best_k_indices = jax.lax.top_k(fitness, self.elite_k)
        
        # 2. Randomly sample from these elite indices
        random_selections = jax.random.randint(
            rng, 
            shape=(self.num_selections,), 
            minval=0, 
            maxval=self.elite_k
        )
        
        # 3. Map back to original population indices
        selected_indices = best_k_indices[random_selections]
        
        return selected_indices

    def __call__(self, keys: chex.Array, population, config: C = None) -> chex.Array:
        """
        ABLATION: Override to generate keys internally instead of using pre-allocated keys.
        Keys depend on input keys to prevent JAX constant-folding (runtime-dynamic).
        """
        # Generate a fresh key by folding single allocated key
        base_key = jax.random.PRNGKey(self.seed)
        base_key = jax.random.fold_in(base_key, keys.reshape(-1)[0])
        
        # Call _select with generated key
        return self._select(base_key, population.fitness, config)


__all__ = ["AblationElitePoolSelection"]
