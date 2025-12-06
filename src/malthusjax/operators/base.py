"""
Abstract base classes for genetic operators in MalthusJAX Level 2.

This module defines the operator abstractions following the new paradigm:
- @struct.dataclass for immutable, JIT-compatible operators
- Factory pattern with static/dynamic parameters
- Pure JAX functions for maximum performance
- Generic type support for all genome types
"""

from typing import Generic, TypeVar

import chex
import jax
from flax import struct

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

    # --- IDENTITY CARD METHODS (Kernel Interface) ---
    def num_keys(self, config: C, input_shape: tuple) -> int:
        """
        Predict exact RNG requirements for kernel execution.
        
        Default implementation: returns num_offspring (one key per child).
        Override if your operator needs different RNG allocation.
        
        Args:
            config: Genome configuration
            input_shape: Shape of input genome (e.g., (genome_length,))
            
        Returns:
            Number of random keys needed
        """
        return self.num_offspring

    def get_output_shape(self, config: C, input_shape: tuple) -> tuple:
        """
        Compute exact output shape for memory pre-allocation.
        
        Default implementation: (num_offspring, *input_shape)
        
        Args:
            config: Genome configuration
            input_shape: Shape of input genome
            
        Returns:
            Output shape tuple
        """
        return (self.num_offspring, *input_shape)

    def apply_kernel(self, keys: chex.Array, genome: G, config: C) -> G:
        """
        Fused kernel implementation (no RNG splitting allowed).
        
        Default implementation: delegates to legacy __call__ by splitting.
        Operators should override this for fast-lane execution.
        
        Args:
            keys: Pre-allocated random keys (num_offspring,) or (num_keys,)
            genome: Input genome
            config: Genome configuration
            
        Returns:
            Mutated genome with shape (num_offspring, ...)
        """
        # Legacy fallback: use first key and delegate to __call__
        # This maintains backward compatibility
        return self.__call__(keys[0] if keys.ndim > 0 else keys, genome, config)


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

    # --- IDENTITY CARD METHODS (Kernel Interface) ---
    def num_keys(self, config: C, input_shape: tuple) -> int:
        """
        Predict exact RNG requirements for kernel execution.
        
        Default implementation: returns num_offspring (one key per child).
        
        Args:
            config: Genome configuration
            input_shape: Shape of input genome
            
        Returns:
            Number of random keys needed
        """
        return self.num_offspring

    def get_output_shape(self, config: C, input_shape: tuple) -> tuple:
        """
        Compute exact output shape for memory pre-allocation.
        
        Default implementation: (num_offspring, *input_shape)
        
        Args:
            config: Genome configuration
            input_shape: Shape of input genome
            
        Returns:
            Output shape tuple
        """
        return (self.num_offspring, *input_shape)

    def apply_kernel(self, keys: chex.Array, p1: G, p2: G, config: C) -> G:
        """
        Fused kernel implementation (no RNG splitting allowed).
        
        Default implementation: delegates to legacy __call__ by splitting.
        Operators should override this for fast-lane execution.
        
        Args:
            keys: Pre-allocated random keys (num_offspring,) or (num_keys,)
            p1: First parent genome
            p2: Second parent genome
            config: Genome configuration
            
        Returns:
            Offspring with shape (num_offspring, ...)
        """
        # Legacy fallback: use first key and delegate to __call__
        return self.__call__(keys[0] if keys.ndim > 0 else keys, p1, p2, config)


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

    # --- IDENTITY CARD METHODS (Kernel Interface) ---
    def num_keys(self, input_shape: tuple) -> int:
        """
        Predict exact RNG requirements for selection kernel.
        
        Default implementation: returns 1 (one key for selection process).
        Override if your selection needs multiple RNG operations.
        
        Args:
            input_shape: Shape of fitness array (pop_size,)
            
        Returns:
            Number of random keys needed
        """
        return 1

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute exact output shape (selected indices).
        
        Args:
            input_shape: Shape of fitness array
            
        Returns:
            Output shape tuple (num_selections,)
        """
        return (self.num_selections,)

    def apply_kernel(self, keys: chex.Array, fitness: chex.Array) -> chex.Array:
        """
        Fused kernel implementation for selection (no RNG splitting).
        
        Default implementation: delegates to legacy __call__.
        Operators should override this for fast-lane execution.
        
        Args:
            keys: Pre-allocated random keys
            fitness: Fitness array (pop_size,)
            
        Returns:
            Selected indices (num_selections,)
        """
        # Legacy fallback: use first key and delegate to __call__
        return self.__call__(keys[0] if keys.ndim > 0 else keys, fitness)
