"""
Abstract base classes for genetic operators in MalthusJAX Level 2.

This module defines the operator abstractions following the new paradigm:
- @struct.dataclass for immutable, JIT-compatible operators
- Factory pattern with static/dynamic parameters
- Pure JAX functions for maximum performance
- Generic type support for all genome types
"""

from typing import TypeVar, Generic
from flax import struct
import chex
import jax

# Generic Types
G = TypeVar("G", bound="BaseGenome")
C = TypeVar("C")  # Config Type

# ==========================================
# 1. ABSTRACT BASE: MUTATION
# ==========================================
@struct.dataclass
class BaseMutation(Generic[G, C]):
    """
    Abstract Mutation Operator using the new paradigm.
    
    Design Philosophy:
    - Static parameters (num_offspring) are pytree_node=False
    - Dynamic parameters (mutation_rate) are regular fields
    - Factory pattern: __call__ delegates to JIT-compilable _mutate_one
    - Automatic vectorization for multiple offspring
    """
    # --- STATIC PARAMS (Re-compile if changed) ---
    num_offspring: int = struct.field(pytree_node=False, default=1)

    def __call__(self, key: chex.PRNGKey, genome: G, config: C) -> G:
        """
        Applies mutation to produce 'num_offspring' children.
        Output Shape: (Num_Offspring, Genome_Size...)
        """
        # Split keys for the static number of children
        keys = jax.random.split(key, self.num_offspring)
        
        # Vectorize the single mutation logic
        return jax.vmap(
            lambda k, g, c: self._mutate_one(k, g, c), 
            in_axes=(0, None, None)
        )(keys, genome, config)

    def _mutate_one(self, key: chex.PRNGKey, genome: G, config: C) -> G:
        """Abstract: Logic to produce EXACTLY ONE mutant."""
        raise NotImplementedError("Subclasses must implement _mutate_one")


# ==========================================
# 2. ABSTRACT BASE: CROSSOVER
# ==========================================
@struct.dataclass
class BaseCrossover(Generic[G, C]):
    """
    Abstract Crossover Operator using the new paradigm.
    
    Design Philosophy:
    - Static parameters control output shape and compilation
    - Dynamic parameters allow runtime tuning without recompilation
    - Pure JAX functions for maximum performance
    """
    # --- STATIC PARAMS (Re-compile if changed) ---
    num_offspring: int = struct.field(pytree_node=False, default=1)

    def __call__(self, key: chex.PRNGKey, p1: G, p2: G, config: C) -> G:
        """
        Combines two parents to produce 'num_offspring' children.
        Output Shape: (Num_Offspring, Genome_Size...)
        """
        keys = jax.random.split(key, self.num_offspring)
        
        # Vectorize the single crossover logic
        return jax.vmap(
            lambda k, a, b, c: self._cross_one(k, a, b, c),
            in_axes=(0, None, None, None)
        )(keys, p1, p2, config)

    def _cross_one(self, key: chex.PRNGKey, p1: G, p2: G, config: C) -> G:
        """Abstract: Logic to produce EXACTLY ONE child."""
        raise NotImplementedError("Subclasses must implement _cross_one")


# ==========================================
# 3. ABSTRACT BASE: SELECTION
# ==========================================
@struct.dataclass
class BaseSelection:
    """
    Abstract Selection Operator using the new paradigm.
    
    Design Philosophy:
    - Operates purely on fitness arrays, genome-agnostic
    - Returns indices for population gathering
    """
    # --- STATIC PARAMS (Re-compile if changed) ---
    num_selections: int = struct.field(pytree_node=False)

    def __call__(self, key: chex.PRNGKey, fitness: chex.Array) -> chex.Array:
        """
        Select individuals based on fitness.
        
        Args:
            key: PRNG Key
            fitness: Fitness array (pop_size,)
            
        Returns:
            Selected indices (num_selections,)
        """
        raise NotImplementedError("Subclasses must implement __call__")