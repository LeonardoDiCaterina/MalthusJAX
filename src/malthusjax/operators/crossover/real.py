"""
Real-valued Crossover Operators
"""
from flax import struct
import jax.numpy as jnp
import jax.random as jar
import chex
from malthusjax.operators.base import BaseCrossover
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig

@struct.dataclass
class UniformCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    Uniform Crossover.
    Mixes genes from both parents based on a per-gene probability.
    """
    crossover_rate: float = 0.5  # Probability of taking gene from Parent 2
    
    def _cross_one(self, key: chex.PRNGKey, p1: RealGenome, p2: RealGenome, config: RealGenomeConfig) -> RealGenome:
        """
        Creates ONE child from two parents using ONE key.
        """
        # 1. Generate Mixing Mask (1 = Swap/Take P2, 0 = Keep/Take P1)
        mask = jar.bernoulli(key, p=self.crossover_rate, shape=p1.values.shape)
        
        # 2. Select values
        new_values = jnp.where(mask, p2.values, p1.values)
        
        return RealGenome(values=new_values)


@struct.dataclass
class BlendCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    Blend Crossover (BLX-α).
    Creates offspring in a range [min - alpha*diff, max + alpha*diff].
    """
    crossover_rate: float = 0.9 # Probability of applying operator
    alpha: float = 0.5
    
    def _cross_one(self, key: chex.PRNGKey, p1: RealGenome, p2: RealGenome, config: RealGenomeConfig) -> RealGenome:
        k_do, k_val = jar.split(key)
        
        # 1. Calculate Potential Offspring
        # Logic: always compute to avoid warp divergence, then mask result
        diff = jnp.abs(p1.values - p2.values)
        cmin = jnp.minimum(p1.values, p2.values) - self.alpha * diff
        cmax = jnp.maximum(p1.values, p2.values) + self.alpha * diff
        
        random_vals = jar.uniform(k_val, shape=p1.values.shape)
        offspring_values = cmin + random_vals * (cmax - cmin)
        
        # Clip to bounds
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
    Mimics the self-adaptive properties of binary crossover in continuous domains.
    """
    crossover_rate: float = 0.9
    eta: float = 20.0
    
    def _cross_one(self, key: chex.PRNGKey, p1: RealGenome, p2: RealGenome, config: RealGenomeConfig) -> RealGenome:
        k_do, k_beta, k_swap = jar.split(key, 3)
        
        # 1. Calculate Beta (Spread Factor)
        u = jar.uniform(k_beta, shape=p1.values.shape)
        
        # SBX Formula for Beta
        beta = jnp.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (self.eta + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1.0))
        )
        
        # 2. Generate Two Candidate Children from the SBX formula
        # We only need one for this function call, but SBX is symmetric.
        # We calculate C1 and C2, then randomly pick one to preserve symmetry 
        # even when generating single children.
        c1 = 0.5 * ((1.0 + beta) * p1.values + (1.0 - beta) * p2.values)
        c2 = 0.5 * ((1.0 - beta) * p1.values + (1.0 + beta) * p2.values)
        
        # 3. Randomly pick C1 or C2 (Symmetry preservation)
        swap_mask = jar.bernoulli(k_swap, p=0.5, shape=p1.values.shape)
        child_vals = jnp.where(swap_mask, c2, c1)
        
        # Clip
        if hasattr(config, 'min_value') and hasattr(config, 'max_value'):
             child_vals = jnp.clip(child_vals, config.min_value, config.max_value)

        # 4. Apply Rate Mask (Op success vs failure)
        should_cross = jar.bernoulli(k_do, p=self.crossover_rate)
        final_values = jnp.where(should_cross, child_vals, p1.values)
        
        return RealGenome(values=final_values)
    
__all__ = ['UniformCrossover' , 'SimulatedBinaryCrossover' , 'BlendCrossover']