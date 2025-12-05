"""
Real-valued Crossover Operators.
Refactored for Zero-Branching (Masking) to maximize GPU throughput.
"""
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jar
import chex
from malthusjax.operators.base import BaseCrossover
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig

@struct.dataclass
class UniformCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    Uniform Crossover (Standard for Neuroevolution/ES).
    Mixes genes from both parents based on a per-gene probability.
    """
    crossover_rate: float = 0.5  # Probability of taking gene from Parent 2
    
    def _cross_one(self, key: chex.PRNGKey, p1: RealGenome, p2: RealGenome, config: RealGenomeConfig) -> RealGenome:
        # 1. Generate Mixing Mask (No branching)
        # mask=1 means "Swap" (Take P2), mask=0 means "Keep" (Take P1)
        mask = jar.bernoulli(key, p=self.crossover_rate, shape=p1.values.shape)
        
        # 2. Select values
        new_values = jnp.where(mask, p2.values, p1.values)
        
        return RealGenome(values=new_values)

@struct.dataclass
class BlendCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    Blend Crossover (BLX-α).
    """
    crossover_rate: float = 0.9 # Probability of applying operator
    alpha: float = 0.5
    
    def _cross_one(self, key: chex.PRNGKey, p1: RealGenome, p2: RealGenome, config: RealGenomeConfig) -> RealGenome:
        k_do, k_val = jar.split(key)
        
        # 1. Calculate Offspring (Always calculate to avoid warp divergence)
        diff = jnp.abs(p1.values - p2.values)
        cmin = jnp.minimum(p1.values, p2.values) - self.alpha * diff
        cmax = jnp.maximum(p1.values, p2.values) + self.alpha * diff
        
        random_vals = jar.uniform(k_val, shape=p1.values.shape)
        offspring_values = cmin + random_vals * (cmax - cmin)
        
        # Clip
        if hasattr(config, 'min_value') and hasattr(config, 'max_value'):
             offspring_values = jnp.clip(offspring_values, config.min_value, config.max_value)
             
        # 2. Apply Crossover Rate via Masking
        # If the operator fails the prob check, we just return Parent 1
        should_cross = jar.bernoulli(k_do, p=self.crossover_rate)
        
        # Broadcast the scalar decision to the whole array
        final_values = jnp.where(should_cross, offspring_values, p1.values)
        
        return RealGenome(values=final_values)

@struct.dataclass
class SimulatedBinaryCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    Simulated Binary Crossover (SBX).
    """
    crossover_rate: float = 0.9
    eta: float = 20.0
    
    def _cross_one(self, key: chex.PRNGKey, p1: RealGenome, p2: RealGenome, config: RealGenomeConfig) -> RealGenome:
        k_do, k_beta, k_swap = jar.split(key, 3)
        
        # 1. Calculate Beta (Spread Factor)
        u = jar.uniform(k_beta, shape=p1.values.shape)
        beta = jnp.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (self.eta + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1.0))
        )
        
        # 2. Generate Candidate Children
        c1 = 0.5 * ((1.0 + beta) * p1.values + (1.0 - beta) * p2.values)
        c2 = 0.5 * ((1.0 - beta) * p1.values + (1.0 + beta) * p2.values)
        
        # 3. Random Swap to maintain symmetry
        # (Standard SBX detail: sometimes return child 2 logic)
        swap_mask = jar.bernoulli(k_swap, p=0.5, shape=p1.values.shape)
        child_vals = jnp.where(swap_mask, c2, c1)
        
        # Clip
        if hasattr(config, 'min_value') and hasattr(config, 'max_value'):
             child_vals = jnp.clip(child_vals, config.min_value, config.max_value)

        # 4. Apply Rate Mask
        should_cross = jar.bernoulli(k_do, p=self.crossover_rate)
        final_values = jnp.where(should_cross, child_vals, p1.values)
        
        return RealGenome(values=final_values)

__all__ = ["UniformCrossover", "BlendCrossover", "SimulatedBinaryCrossover"]