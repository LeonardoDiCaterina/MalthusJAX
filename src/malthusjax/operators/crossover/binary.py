"""
Binary Crossover Operators.
Optimized for batch-first paradigm.
"""

from typing import TypeVar
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct
import chex
from malthusjax.operators.base import BaseCrossover
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation

@struct.dataclass
class UniformCrossover(BaseCrossover[BinaryGenome, BinaryGenomeConfig, BinaryPopulation]):
    """
    Uniform Crossover.
    Produces offspring where each bit comes from parent1 or parent2
    based on crossover probability (coin flip per bit).
    """
    crossover_rate: float = 0.5
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _cross_one(self, keys: chex.Array, p1: BinaryGenome, p2: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        """
        Atomic uniform crossover.
        keys shape: (1, 2)
        """
        rng = keys[0]
        
        # 1. Generate Mask (1 = Take from P1, 0 = Take from P2)
        # Using bernoulli: True(1) with probability p
        mask = jar.bernoulli(rng, p=self.crossover_rate, shape=p1.bits.shape)
        
        # 2. Select Bits
        # If mask is True, take p1, else take p2
        offspring_bits = jnp.where(mask, p1.bits, p2.bits)
        
        # 3. Safe Return
        return p1.replace(bits=offspring_bits)


@struct.dataclass
class SinglePointCrossover(BaseCrossover[BinaryGenome, BinaryGenomeConfig, BinaryPopulation]):
    """
    Single-Point Crossover.
    Swaps segments at a random crossover point.
    """
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1
    
    def _cross_one(self, keys: chex.Array, p1: BinaryGenome, p2: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        """
        Atomic single-point crossover.
        keys shape: (1, 2)
        """
        rng = keys[0]
        length = p1.bits.shape[0]
        
        # 1. Pick Crossover Point (1 to length-1)
        # We avoid 0 and length to ensure actual crossover happens
        crossover_point = jar.randint(rng, shape=(), minval=1, maxval=length)
        
        # 2. Create Mask
        # [0, 1, 2, ...] < point
        indices = jnp.arange(length)
        mask = indices < crossover_point
        
        # 3. Select Bits
        # First part from p1, second part from p2
        offspring_bits = jnp.where(mask, p1.bits, p2.bits)
        
        return p1.replace(bits=offspring_bits)

__all__ = ["UniformCrossover", "SinglePointCrossover"]