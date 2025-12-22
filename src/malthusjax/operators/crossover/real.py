"""
Real-valued Crossover Operators.
Refactored to be purely atomic consumers.
Optimized to consume pre-allocated keys directly, avoiding internal splitting.
"""
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jar
import chex
from malthusjax.operators.base import BaseCrossover
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation

@struct.dataclass
class UniformCrossover(BaseCrossover[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Uniform Crossover.
    Mixes genes from both parents based on a per-gene probability.
    """
    crossover_rate: float = 0.5
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _cross_one(self, keys: chex.PRNGKey, p1: "RealGenome", p2: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        """
        Atomic logic.
        Args:
            keys: Shape (1, 2) because num_keys_per_atomic_operation=1
        """
        # 1. Extract the single key from the block
        rng = keys[0]
        
        # 2. Generate Mixing Mask 
        # Use p1.values.shape to ensure support for multi-dimensional genomes (images/matrices)
        mask = jar.bernoulli(rng, p=self.crossover_rate, shape=p1.values.shape)
        
        # 3. Select values (Vectorized selection)
        new_values = jnp.where(mask, p2.values, p1.values)
        
        # 4. Return new Genome
        # using .replace() is safer than constructor if Genome has other static fields
        return p1.replace(values=new_values)


@struct.dataclass
class BlendCrossover(BaseCrossover["RealGenome", "RealGenomeConfig", "RealPopulation"]):
    """
    Blend Crossover (BLX-alpha).
    - Keys needed: 2 (One for chance to cross, one for generating values).
    - Logic: Creates a box around parents, expands by alpha, samples uniformly.
    """
    crossover_rate: float = 0.9 
    alpha: float = 0.5
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _cross_one(self, keys: chex.Array, p1: "RealGenome", p2: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        """
        Atomic logic.
        keys shape: (2, 2)
        """
        # 1. Unpack Keys
        k_do = keys[0]
        k_val = keys[1]
        
        # 2. Calculate BLX Interval
        # diff = |p1 - p2|
        diff = jnp.abs(p1.values - p2.values)
        
        alpha = jnp.array(self.alpha, dtype=p1.values.dtype)
        
        # Interval: [min - alpha*diff, max + alpha*diff]
        cmin = jnp.minimum(p1.values, p2.values) - (alpha * diff)
        cmax = jnp.maximum(p1.values, p2.values) + (alpha * diff)
        
        # 3. Generate Candidate Values
        # Note: jar -> jax.random
        random_vals = jax.random.uniform(k_val, shape=p1.values.shape)
        offspring_values = cmin + random_vals * (cmax - cmin)
        
        # 4. Clip to Config Bounds (Trace-time check)
        # hasattr is resolved during tracing, so this doesn't slow down the GPU kernel
        if hasattr(config, 'min_value') and hasattr(config, 'max_value'):
             offspring_values = jnp.clip(offspring_values, config.min_value, config.max_value)
             
        # 5. Apply Crossover Probability (Individual Level)
        # Returns shape (), so it broadcasts: either we take the new offspring OR we keep p1 entirely.
        should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)
        
        final_values = jnp.where(should_cross, offspring_values, p1.values)
        
        # 6. Return (Safe Replace)
        return p1.replace(values=final_values)

@struct.dataclass
class SimulatedBinaryCrossover(BaseCrossover["RealGenome", "RealGenomeConfig", "RealPopulation"]):
    """
    Simulated Binary Crossover (SBX).
    Simulates the behavior of single-point crossover for real-valued genomes.
    
    Keys Breakdown:
    - [0] k_do:   Decides if crossover happens at all (rate check).
    - [1] k_beta: Generates the 'u' random values for the spread factor.
    - [2] k_swap: Decides which of the 2 theoretical children to return.
    """
    crossover_rate: float = 0.9
    eta: float = 20.0
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 3

    def _cross_one(self, keys: chex.Array, p1: "RealGenome", p2: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        """
        Atomic SBX Logic.
        keys shape: (3, 2)
        """
        # 1. Unpack Keys
        k_do = keys[0]
        k_beta = keys[1]
        k_swap = keys[2]
        
        # 2. Calculate Beta (Spread Factor)
        # u is uniform(0, 1)
        u = jax.random.uniform(k_beta, shape=p1.values.shape, dtype=p1.values.dtype)
        
        # SBX Formula:
        # if u <= 0.5: beta = (2u)^(1 / (eta + 1))
        # else:        beta = (1 / (2(1-u)))^(1 / (eta + 1))
        
        exponent = jnp.array(1.0 / (self.eta + 1.0), dtype=p1.values.dtype)
        beta = jnp.where(
            u <= 0.5,
            (2.0 * u) ** exponent,
            (1.0 / (2.0 * (1.0 - u))) ** exponent
        )
        
        # 3. Generate Two Candidate Children
        # c1 contracts/expands based on beta
        # c2 does the symmetric opposite
        c1 = 0.5 * ((1.0 + beta) * p1.values + (1.0 - beta) * p2.values)
        c2 = 0.5 * ((1.0 - beta) * p1.values + (1.0 + beta) * p2.values)
        
        # 4. Symmetrization (Crucial for 1-Child Architecture)
        # SBX naturally produces 2 children. We randomly pick one to maintain 
        # diversity and prevent bias towards the "left" parent's trajectory.
        swap_mask = jax.random.bernoulli(k_swap, p=0.5, shape=p1.values.shape)
        child_vals = jnp.where(swap_mask, c2, c1)
        
        # 5. Bound Constraints (Clip)
        if hasattr(config, 'min_value') and hasattr(config, 'max_value'):
             child_vals = jnp.clip(child_vals, config.min_value, config.max_value)

        # 6. Apply Crossover Rate
        # If no crossover, we return Parent 1 (p1)
        should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)
        final_values = jnp.where(should_cross, child_vals, p1.values)
        
        # 7. Safe Return
        return p1.replace(values=final_values)
    
    
    


@struct.dataclass
class BinomialCrossover(BaseCrossover[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    DE Binomial Crossover.
    Requires 1 key per operation: [0] for the Crossover Mask.
    """
    crossover_rate: float = 0.9

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _cross_one(
        self, 
        keys: chex.Array, 
        target: RealGenome, 
        mutant: RealGenome, 
        config: RealGenomeConfig
    ) -> RealGenome:
        """
        Atomic Logic.
        """
        # DIRECT UNPACKING
        k_mask = keys[0]
        
        # 1. Generate Crossover Mask (Bernoulli)
        # shape matches genome.values
        # True = Take Mutant, False = Take Target
        # Note: In DE, we usually ensure at least one gene is swapped (Glossed over here for speed)
        cross_mask = jax.random.bernoulli(k_mask, p=self.crossover_rate, shape=target.values.shape)
        
        # 2. Select (Fused Select)
        trial_values = jnp.where(cross_mask, mutant.values, target.values)
        
        # 3. Clip (Optional but recommended for DE)
        min_val, max_val = config.bounds
        trial_values = jnp.clip(trial_values, min_val, max_val)
        
        return target.replace(values=trial_values)
    
    def __call__(self, all_keys: chex.Array, target_batch: RealGenome, mutant_batch: RealGenome, config: RealGenomeConfig) -> RealGenome:
        leaves = jax.tree_util.tree_leaves(target_batch)
        pop_size = leaves[0].shape[0]
        
        keys_reshaped = all_keys.reshape(pop_size, 1, self.num_keys_per_atomic_operation, 2)
        
        def _process_one(k_block, t, m):
            return jax.vmap(lambda k: self._cross_one(k, t, m, config))(k_block)

        nested_genes = jax.vmap(_process_one)(keys_reshaped, target_batch, mutant_batch)
        
        new_genes = jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), 
            nested_genes
        )
        
        return new_genes


    
__all__ = ['UniformCrossover' , 'SimulatedBinaryCrossover' , 'BlendCrossover', 'BinomialCrossover']