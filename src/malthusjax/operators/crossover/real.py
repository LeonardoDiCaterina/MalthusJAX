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

    # --- Identity Card / Kernel Interface ---
    def num_keys(self, params: "UniformCrossover", input_shape) -> int:
        """Return number of PRNG keys required by the kernel.

        We use a single key to generate the full bernoulli mask in one call,
        so return 1 per batched invocation.
        """
        return 1

    def get_output_shape(self, params: "UniformCrossover", input_shape):
        """Return the offspring array shape matching parent shapes.

        Expect `input_shape` to be the shape of a parent array (batch, length)
        or (length,) for single pair. We return the same shape as the parents.
        """
        return tuple(input_shape)

    def apply_kernel(self, keys: chex.PRNGKey, data, params) -> jnp.ndarray:
        """Pure kernel for uniform crossover.

        `data` can be either a tuple/list `(p1_array, p2_array)` where each is
        shape `(batch, length)` (or `(length,)` treated as single batch), or a
        stacked array of shape `(2, batch, length)` or `(batch, 2, length)`.

        The kernel generates a bernoulli mask with the same shape as parents
        and returns offspring = where(mask, p2, p1).
        """
        # Normalize data into (batch, length) arrays p1, p2
        if isinstance(data, (tuple, list)):
            p1, p2 = data
        else:
            arr = jnp.asarray(data)
            if arr.ndim == 3 and arr.shape[0] == 2:
                # shape (2, batch, length)
                p1, p2 = arr[0], arr[1]
            elif arr.ndim == 3 and arr.shape[1] == 2:
                # shape (batch, 2, length)
                p1, p2 = arr[:, 0, :], arr[:, 1, :]
            else:
                raise ValueError("Unsupported data layout for UniformCrossover.apply_kernel")

        # Ensure arrays are jax arrays
        p1 = jnp.asarray(p1)
        p2 = jnp.asarray(p2)

        # If inputs are single-parent vectors (length,), promote to batch dim
        promoted = False
        if p1.ndim == 1:
            p1 = p1[None, ...]
            p2 = p2[None, ...]
            promoted = True

        # Generate mask: same shape as p1 (batch, length)
        # Handle batched keys: keys shape is (batch, 2) in FAST_LANE mode
        if keys.ndim == 2:  # Batched keys (batch, 2)
            # Use vmap to apply bernoulli to each key independently
            def make_mask_single(key):
                return jar.bernoulli(key, p=self.crossover_rate, shape=p1.shape[1:])
            mask = jax.vmap(make_mask_single)(keys)
        else:  # Single key (2,)
            mask = jar.bernoulli(keys, p=self.crossover_rate, shape=p1.shape)

        offspring = jnp.where(mask, p2, p1)

        if promoted:
            return offspring[0]
        return offspring

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