"""
Binary mutation operators using the new paradigm.

This module provides mutation operators for BinaryGenome using the new
@struct.dataclass factory pattern for JIT compilation and vectorization.
"""

from typing import Any, Tuple, cast

import chex
import jax
import jax.numpy as jnp
import jax.random
from flax import struct

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.operators.base import BaseMutation


@struct.dataclass
class BitFlipMutation(BaseMutation[BinaryGenome, BinaryGenomeConfig, BinaryPopulation]):
    """
    Bit Flip Mutation (3-Tier Paradigm).
    Tier 2: Bernoulli flip mask (N,) where N is bit-string length.
    Tier 1: Bitwise XOR with dtype-conditional logic (bool vs numeric).
    Shape contract: (N,) genome + (N,) flip_mask → (N,) mutated bit-string.
    Key budget: 1 pre-allocated subkey (Bernoulli mask generation).
    """

    mutation_rate: float = 0.1

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _generate_noise(
        self, keys: chex.Array, config: BinaryGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """Tier 2: Generate the flip mask."""
        return jax.random.bernoulli(keys[0], p=self.mutation_rate, shape=config.shape)

    def _mutate_one(
        self,
        genome: BinaryGenome,
        noise_data: chex.Array,
        config: BinaryGenomeConfig,
        **_kwargs: Any,
    ) -> BinaryGenome:
        """Tier 1: Bitwise XOR. Handles bool and numeric (uint8, int32) dtypes."""
        mask = noise_data
        if jnp.issubdtype(genome.values.dtype, jnp.bool_):
            mutated = jnp.logical_xor(genome.values, mask)
        else:
            mutated_bool = jnp.logical_xor(genome.values.astype(bool), mask)
            mutated = mutated_bool.astype(genome.values.dtype)
        return cast(BinaryGenome, cast(Any, genome).replace(values=mutated))


@struct.dataclass
class ScrambleMutation(BaseMutation[BinaryGenome, BinaryGenomeConfig, BinaryPopulation]):
    """
    Scramble Mutation (3-Tier Paradigm).
    Tier 2: Bernoulli decision + permutation indices (N,) reordering.
    Tier 1: Apply permutation conditionally via jax.lax.select (branchless XLA).
    Shape contract: (N,) genome → (N,) permuted_or_original_genome.
    Key budget: 2 pre-allocated subkeys (decision mask, permutation generation).
    Jax.lax.select ensures XLA traces without control flow (scalars broadcast).
    """

    mutation_rate: float = 0.1

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _generate_noise(
        self, keys: chex.Array, config: BinaryGenomeConfig, generation: int = 0
    ) -> Tuple[chex.Array, chex.Array]:
        """Tier 2: Decision mask and permutation indices."""
        should_mutate = jax.random.bernoulli(keys[0], p=self.mutation_rate)
        indices = jax.random.permutation(keys[1], jnp.arange(config.shape[-1]))
        return should_mutate, indices

    def _mutate_one(
        self,
        genome: BinaryGenome,
        noise_data: Tuple[chex.Array, chex.Array],
        config: BinaryGenomeConfig,
        **_kwargs: Any,
    ) -> BinaryGenome:
        """Tier 1: Branchless conditional permutation via jax.lax.select."""
        should_mutate, indices = noise_data
        scrambled = genome.values[indices]
        new_values = jax.lax.select(
            jnp.broadcast_to(should_mutate, scrambled.shape), scrambled, genome.values
        )
        return cast(BinaryGenome, cast(Any, genome).replace(values=new_values))


@struct.dataclass
class SwapMutation(BaseMutation[BinaryGenome, BinaryGenomeConfig, BinaryPopulation]):
    """
    Swap Mutation (3-Tier Paradigm).
    Tier 2: Bernoulli decision + two random bit positions (idx1, idx2).
    Tier 1: Conditional swap via jax.array.at[].set() chaining (immutable arrays).
    Shape contract: (N,) genome → (N,) swapped_or_original_genome.
    Key budget: 3 pre-allocated subkeys (decision, idx1 randint, idx2 randint).
    Functional swap via .at[] preserves JAX immutability for JIT compilation.
    """

    mutation_rate: float = 0.1

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 3

    def _generate_noise(
        self, keys: chex.Array, config: BinaryGenomeConfig, generation: int = 0
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Tier 2: Decision mask and swap positions."""
        should_mutate = jax.random.bernoulli(keys[0], p=self.mutation_rate)
        idx1 = jax.random.randint(keys[1], (), 0, config.shape[-1])
        idx2 = jax.random.randint(keys[2], (), 0, config.shape[-1])
        return should_mutate, idx1, idx2

    def _mutate_one(
        self,
        genome: BinaryGenome,
        noise_data: Tuple[chex.Array, chex.Array, chex.Array],
        config: BinaryGenomeConfig,
        **_kwargs: Any,
    ) -> BinaryGenome:
        """Tier 1: Immutable functional swap via .at[] chaining."""
        should_mutate, idx1, idx2 = noise_data
        v1, v2 = genome.values[idx1], genome.values[idx2]
        swapped = genome.values.at[idx1].set(v2).at[idx2].set(v1)
        new_values = jax.lax.select(should_mutate, swapped, genome.values)
        return cast(BinaryGenome, cast(Any, genome).replace(values=new_values))


__all__ = ["BitFlipMutation", "ScrambleMutation", "SwapMutation"]
