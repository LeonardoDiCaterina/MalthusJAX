"""
Ablation study decorators for testing key budgeting performance impact.

These decorators convert existing operators to use single-key allocation
instead of the ResourceMapper budgeting system, allowing performance comparison
between pre-allocated keys vs. on-demand key splitting.
"""

from typing import Any, Tuple, TypeVar, cast

import chex
import jax
import jax.numpy as jnp

from malthusjax.operators.base import BaseCrossover, BaseMutation

# Type variables for the decorators (parameterize generics to satisfy mypy)
TMutation = TypeVar("TMutation", bound=BaseMutation[Any, Any, Any])
TCrossover = TypeVar("TCrossover", bound=BaseCrossover[Any, Any, Any])


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
    original_call = cls.__call__

    def new_num_keys(self: TMutation, input_shape: Tuple[int, ...]) -> int:
        """Override: Always return 1 key for ablation study."""
        return 1

    def new_call(
        self: TMutation,
        all_keys: chex.Array,
        population: Any,
        config: Any,
        **kwargs: Any,
    ) -> Any:
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
            2,
        )
        total_needed = self.input_length * self.num_offspring * self.num_keys_per_atomic_operation

        # Split and reshape keys for original vectorization pattern
        split_keys = cast(Any, jax.random.split)(single_key, num=int(total_needed))
        keys_reshaped = split_keys.reshape(keys_shape)

        # Use original __call__ with dynamically split keys
        return cast(Any, original_call)(self, keys_reshaped, population, config, **kwargs)

    # Patch the class methods safely (cast to Any to avoid mypy method-assign complaints)
    cast(Any, cls).num_keys = new_num_keys
    cast(Any, cls).__call__ = new_call

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
    original_call = cls.__call__

    def new_num_keys(self: TCrossover, input_shape: Tuple[int, ...]) -> int:
        """Override: Always return 1 key for ablation study."""
        return 1

    def new_call(
        self: TCrossover,
        all_keys: chex.Array,
        p1_pop: Any,
        p2_pop: Any,
        config: Any,
        **kwargs: Any,
    ) -> Any:
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
            2,
        )
        total_needed = self.input_length * self.num_offspring * self.num_keys_per_atomic_operation

        # Split and reshape keys for original vectorization pattern
        split_keys = cast(Any, jax.random.split)(single_key, num=int(total_needed))
        keys_reshaped = split_keys.reshape(keys_shape)

        # Use original __call__ with dynamically split keys
        return cast(Any, original_call)(self, keys_reshaped, p1_pop, p2_pop, config, **kwargs)

    # Patch the class methods safely (cast to Any to avoid mypy method-assign complaints)
    cast(Any, cls).num_keys = new_num_keys
    cast(Any, cls).__call__ = new_call

    return cls
