"""
Real-valued Mutation Operators.
Optimized to consume pre-allocated keys directly, avoiding internal splitting.
"""
from typing import Tuple, TypeVar
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
    Requires 2 keys per mutation: [0] for Mask, [1] for Noise Value.
    """
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    clip: bool = struct.field(pytree_node=False, default=True)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _mutate_one(self, keys: chex.Array, genome: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        """
        Atomic logic.
        """
        # DIRECT UNPACKING - No expensive split!
        k_mask = keys[0]
        k_noise = keys[1]
        
        # 1. Generate Mutation Mask (Which genes change?)
        mutation_mask = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        strength = jnp.array(self.mutation_strength, dtype=genome.values.dtype)
        # 2. Generate Gaussian Noise
        noise = jax.random.normal(k_noise, shape=genome.values.shape, dtype=genome.values.dtype) * strength
        
        # 3. Apply
        mutated_values = jnp.where(mutation_mask, genome.values + noise, genome.values)
        
        # 4. Clip (Static branch)
        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=mutated_values)


@struct.dataclass
class BallMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Ball (Uniform) Mutation.
    Requires 2 keys: [0] for Mask, [1] for Uniform Noise.
    """
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    clip: bool = struct.field(pytree_node=False, default=True)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _mutate_one(self, keys: chex.Array, genome: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        # DIRECT UNPACKING
        k_mask = keys[0]
        k_noise = keys[1]
        strength = jnp.array(self.mutation_strength, dtype=genome.values.dtype)
        # 1. Generate Mutation Mask
        mutation_mask = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        
        # 2. Generate Uniform Noise
        noise = jax.random.uniform(
            k_noise, 
            shape=genome.values.shape,
            minval=-strength,
            maxval=strength
        )
        
        # 3. Apply
        mutated_values = jnp.where(mutation_mask, genome.values + noise, genome.values)
        
        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=mutated_values)


@struct.dataclass
class PolynomialMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Polynomial Mutation.
    Requires 2 keys: [0] for Mask, [1] for random 'u' value.
    """
    mutation_rate: float = 0.1
    eta: float = 20.0
    clip: bool = struct.field(pytree_node=False, default=True)
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _mutate_one(self, keys: chex.Array, genome: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        # DIRECT UNPACKING
        k_mask = keys[0]
        k_val = keys[1]
        
        eta = jnp.array(self.eta, dtype=genome.values.dtype)
        # 1. Generate Mutation Mask
        mutation_mask = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        
        # 2. Generate Random values
        u = jax.random.uniform(k_val, shape=genome.values.shape, dtype=genome.values.dtype)
        
        # 3. Calculate Delta (Standard NSGA-II Logic)
        delta_q = jnp.where(
            u <= 0.5,
            jnp.power(2.0 * u, 1.0 / (self.eta + 1.0)) - 1.0,
            1.0 - jnp.power(2.0 * (1.0 - u), 1.0 / (self.eta + 1.0))
        )
        
        # Scale delta by the search space range
        min_val, max_val = config.bounds
        bound_range = max_val - min_val
        delta = delta_q * bound_range 
        
        # 4. Apply
        mutated_values = jnp.where(mutation_mask, genome.values + delta, genome.values)
        
        if self.clip:
            mutated_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=mutated_values) 


@struct.dataclass
class DifferentialMutation(BaseMutation[Tuple[RealGenome, RealGenome, RealGenome], RealGenomeConfig, RealPopulation]):
    """
    DE/rand/1 Mutation.
    v = r1 + F * (r2 - r3)
    
    Consumes 0 keys (Deterministic given the triplet).
    """
    f_scale: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 0

    def _mutate_one(
        self, 
        keys: chex.Array, 
        genome_triplet: Tuple[RealGenome, RealGenome, RealGenome], 
        config: RealGenomeConfig
    ) -> RealGenome:
        """
        Atomic Logic. 
        Input 'genome_triplet' is (r1, r2, r3).
        """
        # 0. Unpack Triplet
        # Note: The Engine's gather phase constructs this tuple
        r1, r2, r3 = genome_triplet
        
        # 1. Vector Math (Fused FMA)
        # v = r1 + F * (r2 - r3)
        diff = r2.values - r3.values
        mutant_values = r1.values + (diff * self.f_scale)
        
        # 2. Return new Genome
        # We use r1 as the shell to preserve metadata if any
        return r1.replace(values=mutant_values)
    
    
@struct.dataclass
class DitherMutation(BaseMutation[Tuple[RealGenome, RealGenome, RealGenome], RealGenomeConfig, RealPopulation]):
    """
    DE Dither Mutation.
    v = r1 + F' * (r2 - r3)
    where F' is sampled uniformly from [low, high] for each mutant.
    
    Consumes 1 key (Scalar randomness).
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
        """
        Atomic Logic.
        """
        # 0. Unpack
        k_scale = keys[0]
        r1, r2, r3 = genome_triplet
        
        # 1. Sample Scalar F
        # We sample a single scalar for the entire genome vector.
        # Shape: () or (1,) depending on implementation preference.
        # Note: We use the dtype of the genome values to avoid casting overhead.
        F = jax.random.uniform(
            k_scale, 
            shape=(), 
            minval=self.low, 
            maxval=self.high, 
            dtype=r1.values.dtype
        )
        
        # 2. Vector Math
        # XLA will fuse the broadcast of F into the subtraction/addition
        diff = r2.values - r3.values
        mutant_values = r1.values + (diff * F)
        
        # 3. Clip (Optional)
        # Often useful in DE to keep vectors valid
        if self.clip:
            min_val, max_val = config.bounds
            mutant_values = jnp.clip(mutant_values, min_val, max_val)
        
        return r1.replace(values=mutant_values)
    
    def __call__(self, all_keys: chex.Array, triplets: Tuple[RealGenome, RealGenome, RealGenome], config: RealGenomeConfig) -> RealGenome:
        """
        Overridden __call__ that works on raw Genomes/Tuples.
        """
        # 1. Infer Batch Size from the first element of the tuple
        leaves = jax.tree_util.tree_leaves(triplets)
        pop_size = leaves[0].shape[0]

        # 2. Reshape Keys (Standard Logic)
        # Shape: (Pop, 1, KeysPerOp, 2) -> We assume num_offspring=1 for DE
        keys_reshaped = all_keys.reshape(
            pop_size, 
            1, 
            self.num_keys_per_atomic_operation, 
            2
        )

        # 3. Vectorize directly over the tuple
        # vmap over axis 0 of keys and axis 0 of the triplet
        def _process_one(k_block, triplet_slice):
            # Inner vmap over offspring (which is just 1)
            return jax.vmap(
                lambda k: self._mutate_one(k, triplet_slice, config)
            )(k_block)

        nested_genes = jax.vmap(_process_one)(keys_reshaped, triplets)
        
        # 4. Flatten and RETURN RAW GENOME (No spawn_offspring!)
        # Shape comes out as (Pop, 1, Dim...), we reshape to (Pop, Dim...)
        new_genes = jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), 
            nested_genes
        )
        
        return new_genes    
    
__all__ = ["GaussianMutation", "BallMutation", "PolynomialMutation", "DifferentialMutation", "DitherMutation"]