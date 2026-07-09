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
        return p1.replace(ops=new_ops, args=new_args)

@struct.dataclass
class Linear1PointCrossover(BaseCrossover[LinearGenome, Any]):
    """1-Point crossover for Linear genomes.
    
    A cut point 'c' is determined. Offspring receives rows 0...c from parent 1
    and rows c+1...L-1 from parent 2.
    """
    
    # If None, cut point is chosen uniformly at random.
    # If a float between [0, 1], it acts as a deterministic fractional cut point
    # (e.g. 0.5 means cut exactly in half).
    cut_fraction: float | None = None
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1
        
    def _generate_noise(
        self, keys: chex.PRNGKey, config: Any, generation: int = 0
    ) -> Any:
        return keys[0]
        
    def _recombine_one(
        self, p1: LinearGenome, p2: LinearGenome, noise_data: chex.PRNGKey, config: Any, **kwargs: Any
    ) -> LinearGenome:
        L = p1.ops.shape[0]
        
        if self.cut_fraction is None:
            # Random cut point
            c = jax.random.randint(noise_data, (), 0, L)
        else:
            # Deterministic/Scheduled cut point
            c = jnp.clip(jnp.round(self.cut_fraction * L).astype(jnp.int32), 0, L - 1)
            
        indices = jnp.arange(L)
        # mask is True for indices <= c (inherit from p1), False for indices > c (inherit from p2)
        mask = indices <= c
        
        new_ops = jnp.where(mask, p1.ops, p2.ops)
        
        mask_expanded = jnp.expand_dims(mask, axis=-1)
        new_args = jnp.where(mask_expanded, p1.args, p2.args)
        
        return p1.replace(ops=new_ops, args=new_args)

@struct.dataclass
class LinearAncestorMaskCrossover(BaseCrossover[LinearGenome, Any]):
    """Ancestor-Mask (Motif) Crossover.
    
    Selects a random active node in Parent 1, queries its exact topological
    ancestor graph, and splices ONLY that mathematical motif into Parent 2.
    """
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1
        
    def _generate_noise(
        self, keys: chex.PRNGKey, config: Any, generation: int = 0
    ) -> Any:
        return keys[0]
        
    def _recombine_one(
        self, p1: LinearGenome, p2: LinearGenome, noise_data: chex.PRNGKey, config: Any, **kwargs: Any
    ) -> LinearGenome:
        L = p1.ops.shape[0]
        
        # 1. Get ancestor masks for p1 (L, L)
        # ancestors[i, j] is True if j is an ancestor of i
        # Note: this requires p1 to be a BasePrefixAwareGenome
        if not hasattr(p1, "get_ancestor_sets"):
            # Fallback to uniform if not prefix aware
            mask = jax.random.bernoulli(noise_data, p=0.5, shape=(L,))
        else:
            ancestors = p1.get_ancestor_sets(config)
            
            # 2. Pick a random node v to act as the motif root
            v = jax.random.randint(noise_data, (), 0, L)
            
            # 3. The mask is the ancestors of v, plus v itself
            mask = ancestors[v].at[v].set(True)
            
        # True means inherit from p1 (the motif), False means inherit from p2 (the base)
        new_ops = jnp.where(mask, p1.ops, p2.ops)
        
        mask_expanded = jnp.expand_dims(mask, axis=-1)
        new_args = jnp.where(mask_expanded, p1.args, p2.args)
        
        return p1.replace(ops=new_ops, args=new_args)
