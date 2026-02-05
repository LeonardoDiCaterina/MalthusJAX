"""
Ablation study decorators for testing key budgeting performance impact.

These decorators convert existing operators to use single-key allocation
instead of the ResourceMapper budgeting system, allowing performance comparison
between pre-allocated keys vs. on-demand key splitting.
"""

from functools import wraps
from typing import Any, Callable, TypeVar, Tuple, cast

import chex
import jax
import jax.numpy as jnp

from malthusjax.operators.base import BaseMutation, BaseCrossover

# Type variables for the decorators
TMutation = TypeVar('TMutation', bound=BaseMutation)
TCrossover = TypeVar('TCrossover', bound=BaseCrossover)


def ablation_single_key_mutation(cls: TMutation) -> TMutation:
    """
    Decorator to convert mutation operators to single-key ablation mode.
    
    Replaces:
    - num_keys() to return 1 (bypass ResourceMapper budgeting)
    - __call__() to split single key internally on-demand
    
    This tests whether key pre-allocation provides performance benefits
    vs. dynamic splitting overhead.
    
    Usage:
        @ablation_single_key_mutation
        class BitFlipMutation_ablation(BitFlipMutation):
            pass
    """
    # Store original methods
    original_num_keys = cls.num_keys
    original_call = cls.__call__
    
    def new_num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """Override: Always return 1 key for ablation study."""
        return 1
    
    def new_call(self, all_keys: chex.Array, population, config, **kwargs):
        """
        Override: Split single key internally, preserving original logic.
        
        This implements the same vectorization strategy but with on-demand
        key splitting instead of pre-allocated keys from ResourceMapper.
        """
        # Extract single key from input array, handling potential batch dimensions
        single_key = jnp.asarray(all_keys).reshape(-1)[:2]  # Ensure shape (2,)
        
        # Calculate total keys needed using original budgeting logic
        keys_shape = (
            self.input_length,
            self.num_offspring, 
            self.num_keys_per_atomic_operation,
            2
        )
        total_needed = (self.input_length * self.num_offspring * 
                       self.num_keys_per_atomic_operation)
        
        # Split and reshape keys for original vectorization pattern
        keys_reshaped = jax.random.split(single_key, total_needed).reshape(keys_shape)
        
        # Use original __call__ with dynamically split keys
        return original_call(self, keys_reshaped, population, config, **kwargs)
    
    # Patch the class methods
    cls.num_keys = new_num_keys
    cls.__call__ = new_call
    
    return cls


def ablation_single_key_crossover(cls: TCrossover) -> TCrossover:
    """
    Decorator to convert crossover operators to single-key ablation mode.
    
    Replaces:
    - num_keys() to return 1 (bypass ResourceMapper budgeting)
    - __call__() to split single key internally on-demand
    
    This tests crossover key budgeting performance vs. dynamic allocation.
    
    Usage:
        @ablation_single_key_crossover  
        class UniformCrossover_ablation(UniformCrossover):
            pass
    """
    # Store original methods
    original_num_keys = cls.num_keys
    original_call = cls.__call__
    
    def new_num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """Override: Always return 1 key for ablation study."""
        return 1
    
    def new_call(self, all_keys: chex.Array, p1_pop, p2_pop, config, **kwargs):
        """
        Override: Split single key internally for crossover operations.
        
        Maintains the same nested vmap structure but uses dynamic key
        splitting instead of ResourceMapper pre-allocation.
        """
        # Extract single key from input array, handling potential batch dimensions
        single_key = jnp.asarray(all_keys).reshape(-1)[:2]  # Ensure shape (2,)
        
        # Calculate total keys needed using original budgeting logic  
        keys_shape = (
            self.input_length,
            self.num_offspring,
            self.num_keys_per_atomic_operation, 
            2
        )
        total_needed = (self.input_length * self.num_offspring * 
                       self.num_keys_per_atomic_operation)
        
        # Split and reshape keys for original vectorization pattern
        keys_reshaped = jax.random.split(single_key, total_needed).reshape(keys_shape)
        
        # Use original __call__ with dynamically split keys
        return original_call(self, keys_reshaped, p1_pop, p2_pop, config, **kwargs)
    
    # Patch the class methods
    cls.num_keys = new_num_keys  
    cls.__call__ = new_call
    
    return cls