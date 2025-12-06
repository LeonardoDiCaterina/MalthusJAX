"""
Real-valued mutation operators using the new paradigm.

This module provides mutation operators for RealGenome using the new 
@struct.dataclass factory pattern for JIT compilation and vectorization.
"""

from typing import Callable
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct
import chex
from malthusjax.operators.base import BaseMutation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig


@struct.dataclass
class GaussianMutation(BaseMutation[RealGenome, RealGenomeConfig]):
    """
    Gaussian (normal) mutation operator for real genomes using the new paradigm.
    
    Adds random noise from a normal distribution to each gene.
    Most commonly used mutation operator for real-valued optimization.
    """
    # --- DYNAMIC PARAMS (Runtime configurable) ---
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    
    def _mutate_one(self, key: chex.PRNGKey, genome: RealGenome, config: RealGenomeConfig) -> RealGenome:
        """Apply Gaussian mutation to a single genome."""
        k1, k2 = jar.split(key)
        
        # Generate mutation mask
        mutation_mask = jar.bernoulli(k1, self.mutation_rate, genome.values.shape)
        
        # Generate Gaussian noise
        noise = jar.normal(k2, genome.values.shape) * self.mutation_strength
        
        # Apply noise only where mask is True
        mutated_values = jnp.where(mutation_mask, genome.values + noise, genome.values)
        
        # Clip to bounds
        min_val, max_val = config.bounds
        clipped_values = jnp.clip(mutated_values, min_val, max_val)
        
        # Create new genome using replace
        return genome.replace(values=clipped_values)
    
    # --- Identity Card / Kernel Interface ---
    def num_keys(self, params: "GaussianMutation", input_shape) -> int:
        """Return number of PRNG keys required by the kernel.

        We use a single key to generate the full Gaussian tensor via
        `jax.random.normal(key, shape=...)`, so return 1 per batch.
        """
        # Standardize on 1 key for the whole tensor generation
        return 1

    def get_output_shape(self, params: "GaussianMutation", input_shape):
        """Return the shape of the kernel output for given input shape.

        The kernel operates on raw genome value arrays, so the output
        shape matches the input `data.shape`.
        """
        return tuple(input_shape)

    def apply_kernel(self, keys: chex.PRNGKey, data: jnp.ndarray, params) -> jnp.ndarray:
        """Pure, stateless kernel for Gaussian mutation.

        Contract:
        - `keys`: a single PRNG key (or pytree) pre-allocated by the Engine
        - `data`: jax array with shape (length,) or (batch, length)
        - `params`: object exposing `mutation_strength` (float)

        The kernel does not call `jax.random.split` and returns a new
        jax array with the same shape and dtype as `data`.
        Operation: x' = x + sigma * N(0,1)
        """
        # Resolve sigma from params (prefer explicit param, fallback to self)
        try:
            sigma = params.mutation_strength
        except Exception:
            sigma = self.mutation_strength

        # Ensure dtype consistency
        dtype = getattr(data, "dtype", jnp.float32)
        # Generate full noise tensor in one call
        noise = jar.normal(keys, shape=data.shape, dtype=dtype) * jnp.asarray(sigma, dtype=dtype)

        return data + noise
        

@struct.dataclass
class BallMutation(BaseMutation[RealGenome, RealGenomeConfig]):
    """
    Ball (uniform) mutation operator for real genomes using the new paradigm.
    
    Adds random noise from a uniform distribution to each gene.
    """
    # --- DYNAMIC PARAMS ---
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    
    def _mutate_one(self, key: chex.PRNGKey, genome: RealGenome, config: RealGenomeConfig) -> RealGenome:
        """Apply ball mutation to a single genome."""
        k1, k2 = jar.split(key)
        
        # Generate mutation mask
        mutation_mask = jar.bernoulli(k1, self.mutation_rate, genome.values.shape)
        
        # Generate uniform noise
        noise = jar.uniform(
            k2, 
            genome.values.shape,
            minval=-self.mutation_strength,
            maxval=self.mutation_strength
        )
        
        # Apply noise only where mask is True
        mutated_values = jnp.where(mutation_mask, genome.values + noise, genome.values)
        
        # Clip to bounds
        min_val, max_val = config.bounds
        clipped_values = jnp.clip(mutated_values, min_val, max_val)
        
        # Create new genome using replace
        return genome.replace(values=clipped_values)


@struct.dataclass
class PolynomialMutation(BaseMutation[RealGenome, RealGenomeConfig]):
    """
    Polynomial mutation operator for real genomes using the new paradigm.
    
    Uses polynomial distribution to generate mutations, commonly used
    in evolutionary algorithms like NSGA-II.
    """
    # --- DYNAMIC PARAMS ---
    mutation_rate: float = 0.1
    eta: float = 20.0  # Distribution index parameter
    
    def _mutate_one(self, key: chex.PRNGKey, genome: RealGenome, config: RealGenomeConfig) -> RealGenome:
        """Apply polynomial mutation to a single genome."""
        k1, k2 = jar.split(key)
        
        # Generate mutation mask
        mutation_mask = jar.bernoulli(k1, self.mutation_rate, genome.values.shape)
        
        # Generate uniform random values
        u = jar.uniform(k2, genome.values.shape)
        
        # Calculate polynomial mutation delta
        delta_1 = jnp.where(
            u <= 0.5,
            jnp.power(2.0 * u, 1.0 / (self.eta + 1.0)) - 1.0,
            1.0 - jnp.power(2.0 * (1.0 - u), 1.0 / (self.eta + 1.0))
        )
        
        # Scale by bounds
        min_val, max_val = config.bounds
        bound_range = max_val - min_val
        delta = delta_1 * bound_range * 0.1  # Scale factor
        
        # Apply mutation and clamp to bounds
        mutated_values = jnp.where(mutation_mask, genome.values + delta, genome.values)
        clipped_values = jnp.clip(mutated_values, min_val, max_val)
        
        # Create new genome using replace
        return genome.replace(values=clipped_values)


__all__ = [
    "GaussianMutation",
    "BallMutation", 
    "PolynomialMutation"
]