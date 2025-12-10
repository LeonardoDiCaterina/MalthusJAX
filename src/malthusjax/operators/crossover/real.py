"""
Real-valued Crossover Operators.
Refactored to be purely atomic consumers.
Optimized to consume pre-allocated keys directly, avoiding internal splitting.
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
    Requires 1 key for the mixing mask.
    """
    crossover_rate: float = 0.5
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _cross_one(self, keys: chex.PRNGKey, p1: RealGenome, p2: RealGenome, config: RealGenomeConfig) -> RealGenome:
        """
        Atomic logic.
        Args:
            keys: A slice of keys. Shape (1, 2) or (2,) depending on the base slice.
        """
        # Handle the slice safe-guard (take the first key if we got a slice)
        rng = keys[0] if keys.ndim > 1 else keys
        print("UniformCrossover _cross_one keys shape:", keys.shape)
        print("UniformCrossover p1.genes.size:", p1.genes.size)
        # 1. Generate Mixing Mask (1 = Swap/Take P2, 0 = Keep/Take P1) RealPopulation
        mask = jar.bernoulli(rng, p=self.crossover_rate, shape=(p1.genes.size,))
        
        # 2. Select values
        new_values = jnp.where(mask, p2.genes.values, p1.genes.values)
        
        return RealGenome(values=new_values)


@struct.dataclass
class BlendCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    Blend Crossover (BLX-α).
    Requires 2 keys: [0] for decision mask, [1] for random value generation.
    """
    crossover_rate: float = 0.9 
    alpha: float = 0.5
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _cross_one(self, keys: chex.PRNGKey, p1: RealGenome, p2: RealGenome, config: RealGenomeConfig) -> RealGenome:
        # DIRECT UNPACKING - No split needed
        k_do = keys[0]
        k_val = keys[1]
        
        # 1. Calculate Potential Offspring
        diff = jnp.abs(p1.values - p2.values)
        cmin = jnp.minimum(p1.values, p2.values) - self.alpha * diff
        cmax = jnp.maximum(p1.values, p2.values) + self.alpha * diff
        
        random_vals = jar.uniform(k_val, shape=p1.values.shape)
        offspring_values = cmin + random_vals * (cmax - cmin)
        
        # Clip to bounds
        if hasattr(config, 'min_value') and hasattr(config, 'max_value'):
             offspring_values = jnp.clip(offspring_values, config.min_value, config.max_value)
             
        # 2. Apply Crossover Rate via Masking
        should_cross = jar.bernoulli(k_do, p=self.crossover_rate)
        
        final_values = jnp.where(should_cross, offspring_values, p1.values)
        
        return RealGenome(values=final_values)


@struct.dataclass
class SimulatedBinaryCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    Simulated Binary Crossover (SBX).
    Requires 3 keys: [0] for op mask, [1] for beta calc, [2] for symmetry swap.
    """
    crossover_rate: float = 0.9
    eta: float = 20.0
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 3

    def _cross_one(self, keys: chex.PRNGKey, p1: RealGenome, p2: RealGenome, config: RealGenomeConfig) -> RealGenome:
        # DIRECT UNPACKING - No split needed
        k_do = keys[0]
        k_beta = keys[1]
        k_swap = keys[2]
        
        # 1. Calculate Beta (Spread Factor)
        u = jar.uniform(k_beta, shape=p1.values.shape)
        
        beta = jnp.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (self.eta + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1.0))
        )
        
        # 2. Generate Two Candidate Children
        c1 = 0.5 * ((1.0 + beta) * p1.values + (1.0 - beta) * p2.values)
        c2 = 0.5 * ((1.0 - beta) * p1.values + (1.0 + beta) * p2.values)
        
        # 3. Randomly pick C1 or C2
        swap_mask = jar.bernoulli(k_swap, p=0.5, shape=p1.values.shape)
        child_vals = jnp.where(swap_mask, c2, c1)
        
        # Clip
        if hasattr(config, 'min_value') and hasattr(config, 'max_value'):
             child_vals = jnp.clip(child_vals, config.min_value, config.max_value)

        # 4. Apply Rate Mask
        should_cross = jar.bernoulli(k_do, p=self.crossover_rate)
        final_values = jnp.where(should_cross, child_vals, p1.values)
        
        return RealGenome(values=final_values)
    
__all__ = ['UniformCrossover' , 'SimulatedBinaryCrossover' , 'BlendCrossover']