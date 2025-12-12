"""
Real-valued Mutation Operators.
Optimized to consume pre-allocated keys directly, avoiding internal splitting.
"""
from typing import TypeVar
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
        
        # 2. Generate Gaussian Noise
        noise = jax.random.normal(k_noise, shape=genome.values.shape) * self.mutation_strength
        
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
        
        # 1. Generate Mutation Mask
        mutation_mask = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        
        # 2. Generate Uniform Noise
        noise = jax.random.uniform(
            k_noise, 
            shape=genome.values.shape,
            minval=-self.mutation_strength,
            maxval=self.mutation_strength
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
        
        # 1. Generate Mutation Mask
        mutation_mask = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        
        # 2. Generate Random values
        u = jax.random.uniform(k_val, shape=genome.values.shape)
        
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

__all__ = ["GaussianMutation", "BallMutation", "PolynomialMutation"]