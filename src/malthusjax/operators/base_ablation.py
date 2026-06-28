"""Ablation decorators: Single-key mode for benchmarking key splitting overhead.

Converts operators to consume a single PRNG key and split internally,
allowing performance comparison vs ResourceMapper pre-allocation strategy.
Useful for measuring dynamic key splitting cost in evolutionary loops.
"""

from typing import Any, Tuple, TypeVar, cast

import chex
import jax
import jax.numpy as jnp

from malthusjax.operators.base import BaseCrossover, BaseMutation

TMutation = TypeVar("TMutation", bound=BaseMutation[Any, Any])
TCrossover = TypeVar("TCrossover", bound=BaseCrossover[Any, Any])


def ablation_single_key_mutation(cls: TMutation) -> TMutation:
    """Convert mutation operator to single-key ablation mode for benchmarking.

    Replaces num_keys() to return 1 and __call__() to split internally.
    This benchmarks the cost of dynamic key splitting vs ResourceMapper
    pre-allocation during JIT compilation and execution.

    Usage::

        @ablation_single_key_mutation
        class BitFlipMutation_ablation(BitFlipMutation):
            pass
    """
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
        """Override: Split a single key internally and delegate to original.

        The incoming *all_keys* tensor is reduced to a single PRNG key which
        is then split the requisite number of times. This facilitates
        benchmarking the cost of dynamic splitting versus using a pre‑allocated
        key bundle.
        """
        single_key = jnp.asarray(all_keys).reshape(-1)[:2]

        keys_shape = (
            self.input_length,
            self.num_offspring,
            self.num_keys_per_atomic_operation,
            2,
        )
        total_needed = self.input_length * self.num_offspring * self.num_keys_per_atomic_operation

        split_keys = jax.random.split(single_key, num=int(total_needed))
        keys_reshaped = split_keys.reshape(keys_shape)

        return cast(Any, original_call)(self, keys_reshaped, population, config, **kwargs)

    # Patch the class methods safely (cast to Any to avoid mypy method-assign complaints)
    cast(Any, cls).num_keys = new_num_keys
    cast(Any, cls).__call__ = new_call

    return cls


def ablation_single_key_crossover(cls: TCrossover) -> TCrossover:
    """Convert crossover operator to single-key ablation mode for benchmarking.

    Replaces num_keys() to return 1 and __call__() to split internally.
    Benchmarks dynamic key splitting cost for crossover vs ResourceMapper
    pre-allocation strategy.

    Usage::

        @ablation_single_key_crossover
        class UniformCrossover_ablation(UniformCrossover):
            pass
    """
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
        """Override: Split single key internally for crossover.

        Reduces the provided key to a lone PRNG key, splits it locally and then
        invokes the original crossover logic. This decoration is intended for
        performance profiling.
        """
        single_key = jnp.asarray(all_keys).reshape(-1)[:2]

        keys_shape = (
            self.input_length,
            self.num_offspring,
            self.num_keys_per_atomic_operation,
            2,
        )
        total_needed = self.input_length * self.num_offspring * self.num_keys_per_atomic_operation

        split_keys = jax.random.split(single_key, num=int(total_needed))
        keys_reshaped = split_keys.reshape(keys_shape)

        return cast(Any, original_call)(self, keys_reshaped, p1_pop, p2_pop, config, **kwargs)

    cast(Any, cls).num_keys = new_num_keys
    cast(Any, cls).__call__ = new_call

    return cls
