"""
Permutation mutation operators.
Refactored for Zero-Branching to improve XLA fusion.
"""
from typing import Callable, TypeVar, Generic
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct
import chex
from malthusjax.operators.base import BaseMutation

G = TypeVar("G")
C = TypeVar("C")
P = TypeVar("P")

@struct.dataclass
class SwapMutation(BaseMutation[G, C, P]):
    """
    Swap Mutation.
    Exchanges two random genes.
    """
    mutation_rate: float = 0.1

    def _mutate_one(self, key: chex.PRNGKey, genome: G, config: C) -> G:
        k_mask, k_p1, k_p2 = jar.split(key, 3)
        
        # 1. Calculate the Mutation (Always)
        size = genome.genome.shape[-1]
        idx1 = jar.randint(k_p1, (), 0, size)
        idx2 = jar.randint(k_p2, (), 0, size)
        
        val1 = genome.genome[idx1]
        val2 = genome.genome[idx2]
        
        # Perform the swap on a copy
        mutated_data = genome.genome.at[idx1].set(val2).at[idx2].set(val1)
        
        # 2. Decide if we keep it (Masking)
        should_mutate = jar.bernoulli(k_mask, p=self.mutation_rate)
        
        # Select between Original and Swapped
        # This allows XLA to fuse the operation without divergent control flow
        final_data = jnp.where(should_mutate, mutated_data, genome.genome)
        
        return genome.replace(genome=final_data)

@struct.dataclass
class ScrambleMutation(BaseMutation[G, C, P]):
    """
    Scramble Mutation.
    Shuffles the entire genome (or a subsequence).
    """
    mutation_rate: float = 0.1

    def _mutate_one(self, key: chex.PRNGKey, genome: G, config: C) -> G:
        k_mask, k_perm = jar.split(key)
        
        # 1. Calculate Scramble (Always)
        # Note: This scrambles the WHOLE genome. 
        # For sub-sequence scramble, you would need start/end indices.
        indices = jar.permutation(k_perm, jnp.arange(genome.genome.shape[-1]))
        scrambled_data = genome.genome[indices]
        
        # 2. Masking
        should_mutate = jar.bernoulli(k_mask, p=self.mutation_rate)
        final_data = jnp.where(should_mutate, scrambled_data, genome.genome)
        
        return genome.replace(genome=final_data)

__all__ = ["SwapMutation", "ScrambleMutation"]