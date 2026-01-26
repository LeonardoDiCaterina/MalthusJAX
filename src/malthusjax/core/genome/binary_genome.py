from __future__ import annotations

from typing import Any, ClassVar, Type, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation


@struct.dataclass
class BinaryGenomeConfig:
    """
    Configuration for combinatorial binary optimization.

    Attributes:
        length: Total number of bits (dimensions) in the genome.
        p: Probability of a bit being 1 during random initialization.
        dtype: Numerical type for bits, usually jnp.int32 or jnp.int8.
    """

    length: int
    p: float = 0.5
    dtype: jnp.dtype[Any] = struct.field(
        pytree_node=False,
        default=jnp.int32,  # type: ignore[no-untyped-call]
    )


@struct.dataclass
class BinaryGenome(BaseGenome):
    """
    A discrete, binary bit-string genome representation.

    Represents candidate solutions as sequences of 0s and 1s. This is the
    standard representation for combinatorial problems like the Knapsack
    Problem, Feature Selection, and Boolean Satisfiability (SAT).

    In a BinaryPopulation, the 'bits' array is promoted to shape (N, length).
    """

    bits: chex.Array  # Shape: (length,) for individuals, (N, length) for populations

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: BinaryGenomeConfig) -> BinaryGenome:
        """
        Samples a bit-string from a Bernoulli distribution.
        """
        bits = jax.random.bernoulli(key, config.p, (config.length,)).astype(config.dtype)
        return cls(bits=bits)

    def autocorrect(self, config: BinaryGenomeConfig) -> BinaryGenome:
        """
        Ensures bits remain discrete (0 or 1). While binary operators
        usually preserve this, this method provides a safety clip.
        """
        corrected_bits = jnp.clip(self.bits, 0, 1).astype(config.dtype)
        return cast(BinaryGenome, cast(Any, self).replace(bits=corrected_bits))

    def distance(self, other: BaseGenome, metric: str = "hamming") -> chex.Numeric:
        """
        Calculates distance between bit-strings.

        Args:
            other: Another genome (cast to BinaryGenome internally).
            metric: Supports 'hamming' (count of different bits) and 'euclidean'.
        """
        other_bin = cast(BinaryGenome, other)

        if metric == "hamming":
            # XOR equivalent logic for JAX arrays
            return jnp.sum(self.bits != other_bin.bits)
        elif metric == "euclidean":
            # Distance in the embedding space
            return jnp.sqrt(jnp.sum(jnp.square(self.bits - other_bin.bits)))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @property
    def size(self) -> int:
        """The number of bits in the genome."""
        return int(self.bits.shape[-1])

    @property
    def shape(self) -> tuple[int, ...]:
        """The logical shape of the bit-string."""
        return cast(tuple[int, ...], self.bits.shape)

    def to_int(self, msb_first: bool = True) -> chex.Numeric:
        """
        Calculates the decimal integer value of the bit-string.

        By default, this treats ``bits[0]`` as the most significant bit (MSB),
        using a descending power sequence. For example, for a 4-bit genome
        ``bits = [b0, b1, b2, b3]``, the value is::

            value = b0 * 2**3 + b1 * 2**2 + b2 * 2**1 + b3 * 2**0

        If ``msb_first`` is set to ``False``, the legacy behavior is used,
        where ``bits[0]`` is treated as the least significant bit (LSB),
        i.e. using an ascending power sequence::

            value = b0 * 2**0 + b1 * 2**1 + ... + b{n-1} * 2**(n-1)

        Note: Large bit-strings may exceed standard integer precision.
        Returns a JAX array (scalar) to remain compatible with JIT.
        """
        if msb_first:
            # Treat bits[0] as the most significant bit (MSB)
            powers = 2 ** jnp.arange(self.size - 1, -1, -1)
        else:
            # Legacy behavior: treat bits[0] as the least significant bit (LSB)
            powers = 2 ** jnp.arange(self.size)
        return jnp.sum(self.bits * powers)

    def count_ones(self) -> chex.Numeric:
        """Computes the 'Hamming Weight' (number of set bits) of the genome."""
        return jnp.sum(self.bits)

    def flip_bit(self, index: int) -> BinaryGenome:
        """
        Returns a new genome with the bit at 'index' toggled.
        Compatible with JAX's functional 'at' syntax.
        """
        new_bits = self.bits.at[index].set(1 - self.bits[index])
        return cast(BinaryGenome, cast(Any, self).replace(bits=new_bits))

    def __repr__(self) -> str:
        try:
            # We slice to avoid huge strings in debug logs
            sample = self.bits[:10]
            bits_str = "".join(str(int(b)) for b in sample)
            if self.size > 10:
                bits_str += "..."
            return f"<BinaryGenome({bits_str}, len={self.size})>"
        except Exception:
            return f"<BinaryGenome(traced, len={self.size})>"


@struct.dataclass
class BinaryPopulation(BasePopulation[BinaryGenome]):
    """
    A specialized container for a population of BinaryGenomes.
    """

    genes: BinaryGenome
    fitness: chex.Array
    config: BinaryGenomeConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    GENOME_CLS: ClassVar[Type[BinaryGenome]] = BinaryGenome

    @classmethod
    def init_random(
        cls, key: chex.PRNGKey, config: BinaryGenomeConfig, size: int
    ) -> BinaryPopulation:
        """
        Orchestrates parallel Bernoulli sampling for the initial population.
        """
        batched_genes = BinaryGenome.create_population(key, config, size)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)
