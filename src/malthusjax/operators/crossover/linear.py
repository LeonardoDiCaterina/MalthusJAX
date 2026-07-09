"""Linear Genome crossover operator."""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.linear_genome import LinearGenome
from malthusjax.operators.base import BaseCrossover


@struct.dataclass
class LinearUniformCrossover(BaseCrossover[LinearGenome, Any]):
    """Row-level uniform crossover for LinearGenome and derived classes.
    
    Swaps entire rows (both ops and args simultaneously) between two parents
    based on a boolean mask. This guarantees structural validity since both 
    parents share the same topological rules.
    """

    crossover_rate: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _generate_noise(
        self, keys: chex.PRNGKey, config: Any, generation: int = 0
    ) -> Any:
        """Just pass the key through; shapes are dynamic in Tier 1."""
        return keys[0]

    def _recombine_one(
        self, p1: LinearGenome, p2: LinearGenome, noise_data: chex.PRNGKey, config: Any, **kwargs: Any
    ) -> LinearGenome:
        """Applies uniform row swapping between parents."""
        L = p1.ops.shape[0]
        
        # True means inherit from p2, False means inherit from p1
        swap_mask = jax.random.bernoulli(noise_data, p=self.crossover_rate, shape=(L,))
        
        # Swap ops
        new_ops = jnp.where(swap_mask, p2.ops, p1.ops)
        
        # Swap args (expand mask to match args shape (L, max_arity))
        swap_mask_expanded = jnp.expand_dims(swap_mask, axis=-1)
        new_args = jnp.where(swap_mask_expanded, p2.args, p1.args)
        
        # Return same genome class to support BasePrefixAwareGenome
        return type(p1)(ops=new_ops, args=new_args)
