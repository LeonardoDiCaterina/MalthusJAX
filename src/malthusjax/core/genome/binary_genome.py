from __future__ import annotations

from typing import Any, ClassVar, Tuple, Type, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation

_field: Any = struct.field


@struct.dataclass
class BinaryGenomeConfig:
    """Configuration for binary string genomes.

    Attributes:
        shape: Logical shape of bit-string; defaults to (1,) to ensure non-scalar.
        length: Legacy alias for shape=(length,); overrides shape if set.
        p: Bernoulli probability parameter for bit initialization p(b=1).
        dtype: JAX dtype for bit values; typically jnp.int32 or jnp.int8.
    """

    # Prefer an explicit 1-D shape default to avoid accidental scalar genomes.
    shape: Tuple[int, ...] = _field(pytree_node=False, default_factory=lambda: (1,))

    # Backwards-compatibility: accept a legacy `length` keyword
    length: int | None = _field(pytree_node=False, default=None)

    p: float = 0.5
    dtype: jnp.dtype[Any] = _field(
        pytree_node=False,
        default=jnp.int32,
    )

    @property
    def resolved_shape(self) -> Tuple[int, ...]:
        """Return the effective shape, honoring legacy `length` if present."""
        if self.length is not None:
            return (self.length,)
        return self.shape

    def init_population(
        self, key: chex.PRNGKey, size: int
    ) -> BasePopulation[BinaryGenome]:
        """Create a random population from this config (protocol method for JR-2)."""
        return BinaryPopulation.init_random(key, self, size)


@struct.dataclass
class BinaryGenome(BaseGenome):
    """
    A discrete, binary bit-string genome representation.

    Represents candidate solutions as sequences of 0s and 1s. This is the
    standard representation for combinatorial problems like the Knapsack
    Problem, Feature Selection, and Boolean Satisfiability (SAT).

    In a BinaryPopulation, the 'bits' array is promoted to shape (N, length).
    """

    values: chex.Array  # Shape: (length,) for individuals, (N, length) for populations
    # Enable Pythonic indexing/iteration by default for convenience
    subscriptable: bool = _field(pytree_node=False, default=True)

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: BinaryGenomeConfig) -> BinaryGenome:
        """Initialize bit-string via Bernoulli sampling at scale config.p.

        Args:
            key: PRNGKey for reproducibility.
            config: BinaryGenomeConfig with shape and p parameters.

        Returns:
            BinaryGenome with values shape matching config.resolved_shape,
            dtype matching config.dtype, sampled from Bernoulli(p).
        """
        values = jax.random.bernoulli(key, config.p, config.resolved_shape).astype(config.dtype)
        return cls(values=values)

    def autocorrect(self, config: BinaryGenomeConfig) -> BinaryGenome:
        """Enforce bit domain [0, 1] via clipping and dtype conversion.

        Called post-mutation/crossover to guarantee discrete binary values
        in case floating-point operations introduced out-of-bounds values.
        """
        corrected_values = jnp.clip(self.values, 0, 1).astype(config.dtype)
        return cast(BinaryGenome, cast(Any, self).replace(values=corrected_values))

    def distance(self, other: BaseGenome, metric: str = "hamming") -> chex.Numeric:
        """Compute distance between binary genomes.

        Args:
            other: Another genome; cast to BinaryGenome internally.
            metric: 'hamming' (sum of XOR) or 'euclidean' (L2 norm).

        Returns:
            Scalar distance value (JAX array, JIT-compatible).
        """
        other_bin = cast(BinaryGenome, other)

        if metric == "hamming":
            # XOR equivalent logic for JAX arrays
            return jnp.sum(self.values != other_bin.values)
        elif metric == "euclidean":
            # Distance in the embedding space
            return jnp.sqrt(jnp.sum(jnp.square(self.values - other_bin.values)))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @property
    def size(self) -> int:
        """The number of bits in the genome."""
        return int(self.values.shape[-1])

    @property
    def shape(self) -> tuple[int, ...]:
        """The logical shape of the bit-string."""
        return cast(tuple[int, ...], self.values.shape)

    @classmethod
    def from_tensor(cls, arr: chex.Array, config: Any = None) -> "BinaryGenome":
        """Construct a batched BinaryGenome from a raw array.

        Keeps implementation minimal and JIT-safe: simply wraps the provided
        array in the `BinaryGenome` dataclass.
        """
        return cls(values=arr)

    def to_int(self, msb_first: bool = True) -> chex.Numeric:
        """Convert bit-string to integer via positional weighting.

        Args:
            msb_first: If True (default), treat values[0] as most significant bit (MSB).
                If False, treat values[0] as least significant bit (LSB).

        Returns:
            Scalar JAX array (integer type) representing the bit-string value.
            Note: May exceed standard int precision for long strings; use
            JAX arrays to maintain XLA compatibility.
        """
        if msb_first:
            # Treat values[0] as the most significant bit (MSB)
            powers = 2 ** jnp.arange(self.size - 1, -1, -1)
        else:
            # Legacy behavior: treat values[0] as the least significant bit (LSB)
            powers = 2 ** jnp.arange(self.size)
        return jnp.sum(self.values * powers)

    def count_ones(self) -> chex.Numeric:
        """Sum of bits; Hamming weight. Scalar JAX array."""
        return jnp.sum(self.values)

    def flip_bit(self, index: int) -> BinaryGenome:
        """Toggle bit at index via JAX .at[index].set() functional update."""
        new_values = self.values.at[index].set(1 - self.values[index])
        return cast(BinaryGenome, cast(Any, self).replace(values=new_values))

    def __repr__(self) -> str:
        try:
            # We slice to avoid huge strings in debug logs
            sample = self.values[:10]
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
