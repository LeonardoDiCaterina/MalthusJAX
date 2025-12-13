"""
Categorical mutation operators using the new paradigm.

This module provides mutation operators for CategoricalGenome using the new 
@struct.dataclass factory pattern for JIT compilation and vectorization.
"""

from typing import Callable
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct
import chex
from malthusjax.operators.base import BaseMutation
from malthusjax.core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig, CategoricalPopulation
import functools

from malthusjax.operators.mutation.permutation import SwapMutation, ScrambleMutation


@struct.dataclass 
class CategoricalFlipMutation(BaseMutation[CategoricalGenome, CategoricalGenomeConfig, CategoricalPopulation]):
    """
    Categorical flip mutation using the new paradigm.
    
    Replaces a gene with a new, uniformly random category.
    Supports automatic vectorization for multiple offspring.
    """
    # --- DYNAMIC PARAMS ---
    mutation_rate: float = 0.1
    
    def _mutate_one(self, key: chex.PRNGKey, genome: CategoricalGenome, config: CategoricalGenomeConfig) -> CategoricalGenome:
        """Apply categorical flip mutation to a single genome."""
        k1, k2 = jar.split(key)
        
        # Generate mutation mask
        mutation_mask = jar.bernoulli(k1, self.mutation_rate, genome.categories.shape)
        
        # Generate new random categories
        new_categories = jar.randint(
            k2, 
            genome.categories.shape, 
            0, 
            config.num_categories
        )
        
        # Apply mutation where mask is True
        mutated_values = jnp.where(mutation_mask, new_categories, genome.categories)
        
        # Create new genome using replace
        return genome.replace(categories=mutated_values)


@struct.dataclass
class RandomCategoryMutation(BaseMutation[CategoricalGenome, CategoricalGenomeConfig, CategoricalPopulation]):
    """
    Advanced categorical mutation that ensures new categories are different from current ones.
    Uses the new paradigm with automatic vectorization.
    """
    # --- DYNAMIC PARAMS ---
    mutation_rate: float = 0.1
    
    def _mutate_one(self, key: chex.PRNGKey, genome: CategoricalGenome, config: CategoricalGenomeConfig) -> CategoricalGenome:
        """Apply random category mutation to a single genome, ensuring categories change."""
        k1, k2 = jar.split(key)
        
        # Generate mutation mask
        mutation_mask = jar.bernoulli(k1, self.mutation_rate, genome.categories.shape)
        
        # Generate new categories (different from current)
        new_categories = jar.randint(
            k2, 
            genome.categories.shape, 
            0, 
            config.num_categories - 1
        )
        
        # Ensure new category is different by adjusting if >= current
        new_categories = jnp.where(
            new_categories >= genome.categories,
            (new_categories + 1) % config.num_categories,
            new_categories
        )
        
        # Apply mutation where mask is True
        mutated_values = jnp.where(mutation_mask, new_categories, genome.categories)
        
        # Create new genome using replace
        return genome.replace(categories=mutated_values)


# --- Pure JAX Functions ---

# --- Pure JAX Function ---

@jax.jit
def _categorical_flip_mutation(
    key: jax.Array,
    genome: jax.Array,
    mutation_rate: float,
    num_categories: int
) -> jax.Array:
    """Replaces genes with a new random category based on mutation_rate."""
    
    key_mask, key_vals = jar.split(key)
    
    # 1. Create a mask to decide *which* genes to mutate
    mutation_mask = jar.bernoulli(key_mask, p=mutation_rate, shape=genome.shape)
    
    # 2. Create an array of *new random values* for all positions
    new_values = jar.randint(
        key_vals,
        shape=genome.shape,
        minval=0,
        maxval=num_categories
    )
    
    # 3. Apply new values only where the mask is True
    #    Also, ensure the new value is different from the old one (optional, but good)
    #    A simpler jnp.where is more standard:
    mutated_genome = jnp.where(mutation_mask, new_values, genome)
    
    return mutated_genome.astype(genome.dtype)


@jax.jit
def _random_category_mutation(
    key: jax.Array,
    genome: jax.Array,
    mutation_rate: float,
    num_categories: int
) -> jax.Array:
    """Mutates genes to different random categories."""
    
    key_mask, key_vals = jar.split(key)
    
    # Create mutation mask
    mutation_mask = jar.bernoulli(key_mask, p=mutation_rate, shape=genome.shape)
    
    # Generate new categories (different from current)
    current_categories = genome
    new_categories = jar.randint(
        key_vals, 
        shape=genome.shape,
        minval=0, 
        maxval=num_categories - 1
    )
    
    # Ensure new category is different by adding 1 and wrapping
    new_categories = jnp.where(
        new_categories >= current_categories,
        (new_categories + 1) % num_categories,
        new_categories
    )
    
    # Apply mutation
    mutated_genome = jnp.where(mutation_mask, new_categories, genome)
    return mutated_genome.astype(genome.dtype)


# --- Re-export permutation ops for convenience ---
# These operators work perfectly on categorical genomes too.
__all__ = [
    "CategoricalFlipMutation",
    "RandomCategoryMutation",
    "SwapMutation",
    "ScrambleMutation"
]