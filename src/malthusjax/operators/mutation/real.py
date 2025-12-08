"""
Real-valued Mutation Operators.
Refactored for the 'Consumer Paradigm': Pure single-genome logic.
Batching and key management are handled by the BaseMutation class.
"""
import jax.numpy as jnp
import jax.random as jar
import chex
from flax import struct
from malthusjax.operators.base import BaseMutation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig


@struct.dataclass
class GaussianMutation(BaseMutation[RealGenome, RealGenomeConfig]):
    """
    Gaussian (Normal) Mutation.
    Adds random noise from a normal distribution to genes.
    """
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    
    def _mutate_one(self, key: chex.PRNGKey, genome: RealGenome, config: RealGenomeConfig) -> RealGenome:
        """
        Mutates ONE genome using ONE key.
        """
        # Split the assigned key locally for mask and noise generation
        k_mask, k_noise = jar.split(key)
        
        # 1. Generate Mutation Mask (Which genes change?)
        mutation_mask = jar.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        
        # 2. Generate Gaussian Noise
        noise = jar.normal(k_noise, shape=genome.values.shape) * self.mutation_strength
        
        # 3. Apply noise only where mask is True
        mutated_values = jnp.where(mutation_mask, genome.values + noise, genome.values)
        
        # 4. Clip to bounds
        min_val, max_val = config.bounds
        clipped_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=clipped_values)


@struct.dataclass
class BallMutation(BaseMutation[RealGenome, RealGenomeConfig]):
    """
    Ball (Uniform) Mutation.
    Adds random noise from a uniform distribution to genes.
    """
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    
    def _mutate_one(self, key: chex.PRNGKey, genome: RealGenome, config: RealGenomeConfig) -> RealGenome:
        """
        Mutates ONE genome using ONE key.
        """
        k_mask, k_noise = jar.split(key)
        
        # 1. Generate Mutation Mask
        mutation_mask = jar.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        
        # 2. Generate Uniform Noise [-strength, +strength]
        noise = jar.uniform(
            k_noise, 
            shape=genome.values.shape,
            minval=-self.mutation_strength,
            maxval=self.mutation_strength
        )
        
        # 3. Apply noise and Clip
        mutated_values = jnp.where(mutation_mask, genome.values + noise, genome.values)
        min_val, max_val = config.bounds
        clipped_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=clipped_values)


@struct.dataclass
class PolynomialMutation(BaseMutation[RealGenome, RealGenomeConfig]):
    """
    Polynomial Mutation.
    Uses polynomial distribution to generate mutations (common in NSGA-II).
    """
    mutation_rate: float = 0.1
    eta: float = 20.0  # Distribution index parameter
    
    def _mutate_one(self, key: chex.PRNGKey, genome: RealGenome, config: RealGenomeConfig) -> RealGenome:
        """
        Mutates ONE genome using ONE key.
        """
        k_mask, k_val = jar.split(key)
        
        # 1. Generate Mutation Mask
        mutation_mask = jar.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        
        # 2. Generate Random values for polynomial calculation
        u = jar.uniform(k_val, shape=genome.values.shape)
        
        # 3. Calculate polynomial mutation delta
        # Equation from Deb & Agrawal (1995)
        delta_1 = jnp.where(
            u <= 0.5,
            jnp.power(2.0 * u, 1.0 / (self.eta + 1.0)) - 1.0,
            1.0 - jnp.power(2.0 * (1.0 - u), 1.0 / (self.eta + 1.0))
        )
        
        # 4. Scale by bounds
        min_val, max_val = config.bounds
        bound_range = max_val - min_val
        # Note: The scaling factor implies the max mutation is proportional to domain size
        delta = delta_1 * bound_range 
        
        # 5. Apply and Clip
        mutated_values = jnp.where(mutation_mask, genome.values + delta, genome.values)
        clipped_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=clipped_values)

__all__ = ["GaussianMutation", "BallMutation", "PolynomialMutation"]