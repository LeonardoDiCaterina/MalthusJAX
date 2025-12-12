"""
Binary mutation operators using the new paradigm.

This module provides mutation operators for BinaryGenome using the new 
@struct.dataclass factory pattern for JIT compilation and vectorization.
"""

from typing import TypeVar
import jax
import jax.numpy as jnp
import jax.random
from flax import struct
import chex
from malthusjax.operators.base import BaseMutation
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation

@struct.dataclass
class BitFlipMutation(BaseMutation[BinaryGenome, BinaryGenomeConfig, BinaryPopulation]):
    """
    Bit flip mutation.
    Flips each bit with probability mutation_rate.
    Requires 1 key (broadcasted/used for the Bernoulli mask).
    """
    mutation_rate: float = 0.1
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _mutate_one(self, keys: chex.Array, genome: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        """
        Atomic bit flip logic.
        keys shape: (1, 2)
        """
        rng = keys[0]

        # 1. Generate boolean mask
        mask = jax.random.bernoulli(rng, p=self.mutation_rate, shape=genome.bits.shape)

        # 2. XOR Logic (Flipping)
        # Handle both boolean and integer storage of bits
        if jnp.issubdtype(genome.bits.dtype, jnp.bool_):
            mutated = jnp.logical_xor(genome.bits, mask)
        else:
            # Convert to bool for XOR, then cast back
            mutated_bool = jnp.logical_xor(genome.bits.astype(bool), mask)
            mutated = mutated_bool.astype(genome.bits.dtype)

        return genome.replace(bits=mutated)


@struct.dataclass
class ScrambleMutation(BaseMutation[BinaryGenome, BinaryGenomeConfig, BinaryPopulation]):
    """
    Scramble Mutation.
    Scrambles the entire genome with probability `mutation_rate`.
    Requires 2 keys: [0] for decision, [1] for permutation.
    """
    mutation_rate: float = 0.1
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2
    
    def _mutate_one(self, keys: chex.Array, genome: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        # 1. Direct Unpack
        k_do = keys[0]
        k_perm = keys[1]
        
        # 2. Define the Scramble Operation
        def scramble_fn(g_bits):
            indices = jax.random.permutation(k_perm, jnp.arange(g_bits.shape[-1]))
            return g_bits[indices]
        
        # 3. Apply Condition
        # We use lax.cond because permutation is expensive; strict evaluation (where) is wasteful here.
        should_scramble = jax.random.bernoulli(k_do, p=self.mutation_rate)
        
        mutated_bits = jax.lax.cond(
            should_scramble,
            scramble_fn,        # True branch
            lambda x: x,        # False branch (Identity)
            genome.bits         # Operand
        )
        
        return genome.replace(bits=mutated_bits)


@struct.dataclass
class SwapMutation(BaseMutation[BinaryGenome, BinaryGenomeConfig, BinaryPopulation]):
    """
    Swap Mutation.
    Swaps TWO random bits in the genome.
    Requires 3 keys: [0] for decision, [1] Pos A, [2] Pos B.
    """
    mutation_rate: float = 0.1
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 3
    
    def _mutate_one(self, keys: chex.Array, genome: "BinaryGenome", config: "BinaryGenomeConfig") -> "BinaryGenome":
        # 1. Direct Unpack
        k_do = keys[0]
        k_pos1 = keys[1]
        k_pos2 = keys[2]
        
        # 2. Define Swap Operation
        def swap_fn(g_bits):
            size = g_bits.shape[-1]
            idx1 = jax.random.randint(k_pos1, (), 0, size)
            idx2 = jax.random.randint(k_pos2, (), 0, size)
            
            val1 = g_bits[idx1]
            val2 = g_bits[idx2]
            
            # Use .at[].set() for functional updates
            return g_bits.at[idx1].set(val2).at[idx2].set(val1)
        
        # 3. Apply Condition
        should_swap = jax.random.bernoulli(k_do, p=self.mutation_rate)
        
        mutated_bits = jax.lax.cond(
            should_swap,
            swap_fn,
            lambda x: x,
            genome.bits
        )
        
        return genome.replace(bits=mutated_bits)

__all__ = ["BitFlipMutation", "ScrambleMutation", "SwapMutation"]