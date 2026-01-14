"""
Real-valued Mutation Operators.
Optimized for H100:
1. Uses Masked Arithmetic (genome + noise * mask) instead of Branching (jnp.where).
2. Explicit casting to ensure correct dtypes (e.g., BF16) during random number generation and arithmetic.
"""
from typing import Tuple
from flax import struct
import jax.numpy as jnp
import jax.random
import chex
from malthusjax.operators.base import BaseMutation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation

@struct.dataclass
class GaussianMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Gaussian (Normal) Mutation.
    Optimized: Uses FMA (Fused Multiply-Add) via masked arithmetic.
    """
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    clip: bool = struct.field(pytree_node=False, default=True)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _mutate_one(self, keys: chex.Array, genome: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        k_mask = keys[0]
        k_noise = keys[1]
        
        # 1. Capture Dtype (Critical for BF16)
        dtype = genome.values.dtype
        
        # 2. Vaccinate Constants
        strength = jnp.array(self.mutation_strength, dtype=dtype)
        
        # 3. Generate Mask & Noise
        # We assume standard Bernoulli mask (True/False)
        mask_bool = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        # Cast Mask to Float (0.0 or 1.0) for arithmetic
        mask_val = mask_bool.astype(dtype)
        
        # Explicit dtype for noise generation
        noise = jax.random.normal(k_noise, shape=genome.values.shape, dtype=dtype)
        
        # 4. Apply Masked Arithmetic (The "FMA" Optimization)
        # Replacing: jnp.where(mask, genome + noise * strength, genome)
        # With:      genome + (noise * strength * mask)
        # XLA should fuse (noise * strength * mask) into a single multiplier, then add.
        delta = noise * strength * mask_val
        mutated_values = genome.values + delta
        
        # 5. Clip
        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=mutated_values)


@struct.dataclass
class BallMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Ball (Uniform) Mutation.
    Optimized with Masked Arithmetic.
    """
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    clip: bool = struct.field(pytree_node=False, default=True)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _mutate_one(self, keys: chex.Array, genome: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        k_mask = keys[0]
        k_noise = keys[1]
        
        # 1. Capture Dtype
        dtype = genome.values.dtype
        strength = jnp.array(self.mutation_strength, dtype=dtype)
        
        # 2. Generate Mask & Cast
        mask_bool = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        mask_val = mask_bool.astype(dtype)
        
        # 3. Generate Uniform Noise (Explicit Dtype)
        noise = jax.random.uniform(
            k_noise, 
            shape=genome.values.shape,
            minval=-strength,
            maxval=strength,
            dtype=dtype
        )
        
        # 4. Masked Arithmetic
        # Value adds 0.0 where mask is 0.0
        mutated_values = genome.values + (noise * mask_val)
        
        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=mutated_values)


@struct.dataclass
class PolynomialMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Polynomial Mutation.
    Optimized with Masked Arithmetic.
    """
    mutation_rate: float = 0.1
    eta: float = 20.0
    clip: bool = struct.field(pytree_node=False, default=True)
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _mutate_one(self, keys: chex.Array, genome: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        k_mask = keys[0]
        k_val = keys[1]
        
        # 1. Capture Dtype & Vaccinate
        dtype = genome.values.dtype
        eta = jnp.array(self.eta, dtype=dtype)
        one = jnp.array(1.0, dtype=dtype)
        half = jnp.array(0.5, dtype=dtype)
        two = jnp.array(2.0, dtype=dtype)
        
        # 2. Mask & U
        mask_bool = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        mask_val = mask_bool.astype(dtype)
        
        u = jax.random.uniform(k_val, shape=genome.values.shape, dtype=dtype)
        
        # 3. Calculate Delta (Standard Logic)
        exponent = one / (eta + one)
        
        delta_q = jnp.where(
            u <= half,
            jnp.power(two * u, exponent) - one,
            one - jnp.power(two * (one - u), exponent)
        )
        
        # 4. Scale Delta
        min_val, max_val = config.bounds
        bound_range = max_val - min_val 
        
        # 5. Masked Arithmetic
        # Only add delta where mask is 1.0
        delta = delta_q * bound_range * mask_val
        mutated_values = genome.values + delta
        
        if self.clip:
            mutated_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=mutated_values) 

@struct.dataclass
class DitherMutation(BaseMutation[Tuple[RealGenome, RealGenome, RealGenome], RealGenomeConfig, RealPopulation]):
    """
    DE Dither Mutation.
    v = r1 + F' * (r2 - r3)
    Type-Vaccinated.
    """
    low: float = 0.2
    high: float = 0.8
    clip: bool = struct.field(pytree_node=False, default=True)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _mutate_one(
        self, 
        keys: chex.Array, 
        genome_triplet: Tuple[RealGenome, RealGenome, RealGenome], 
        config: RealGenomeConfig
    ) -> RealGenome:
        
        k_scale = keys[0]
        r1, r2, r3 = genome_triplet
        
        # 1. Capture Dtype
        dtype = r1.values.dtype
        
        # 2. Sample Scalar F (Explicit Dtype)
        F = jax.random.uniform(
            k_scale, 
            shape=(), 
            minval=self.low, 
            maxval=self.high, 
            dtype=dtype
        )
        
        # 3. Vector Math
        diff = r2.values - r3.values
        mutant_values = r1.values + (diff * F)
        
        if self.clip:
            min_val, max_val = config.bounds
            mutant_values = jnp.clip(mutant_values, min_val, max_val)
        
        return r1.replace(values=mutant_values)
    
    def __call__(self, all_keys: chex.Array, triplets: Tuple[RealGenome, RealGenome, RealGenome], config: RealGenomeConfig) -> RealGenome:
        """
        Overridden __call__ that works on raw Genomes/Tuples.
        """
        leaves = jax.tree_util.tree_leaves(triplets)
        pop_size = leaves[0].shape[0]

        keys_reshaped = all_keys.reshape(
            pop_size, 
            1, 
            self.num_keys_per_atomic_operation, 
            2
        )

        def _process_one(k_block, triplet_slice):
            return jax.vmap(
                lambda k: self._mutate_one(k, triplet_slice, config)
            )(k_block)

        nested_genes = jax.vmap(_process_one)(keys_reshaped, triplets)
        
        new_genes = jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), 
            nested_genes
        )
        
        return new_genes    
    
__all__ = ["GaussianMutation", "BallMutation", "PolynomialMutation", "DifferentialMutation", "DitherMutation"]