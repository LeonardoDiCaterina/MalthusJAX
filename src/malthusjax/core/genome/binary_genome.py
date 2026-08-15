"""Binary genome and population types.

Defines the representation for bit‑string genomes along with helper methods
for conversion, distance computation and simple bit tricks. Includes the
corresponding population container and random initialization logic.
"""

from __future__ import annotations

from typing import Any, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation

_field: Any = struct.field


@struct.dataclass
class BinaryGenomeConfig:
    """Static configuration for discrete binary string genomes.

    This class defines the genome length and initialization properties for all
    binary genomes created with this configuration. All individuals share the same
    shape and bit encoding.

    Parameters
    ----------
    shape : tuple[int, ...]
        Logical shape of the bit-string (default: (1,) to prevent scalar genomes).
        - shape=(10,) → 10-bit string (length 10)
        - shape=(64,) → 64-bit string (for binary optimization)
        - shape=(20, 16) → 320-bit matrix (if shaped genomes needed)
        Default: (1,) (single bit)

        **Legacy Alias**: Passing `length=N` automatically sets `shape=(N,)`.

    length : int | None
        Backward-compatibility alias for genome_length.
        If provided, overrides `shape` with `shape=(length,)`.
        **Deprecated**: Use `shape` directly instead.
        Default: None

    p : float
        Bernoulli probability parameter for bit initialization.
        - p=0.5: Unbiased initialization (default, typical for most problems)
        - p<0.5: Bias toward 0-bits
        - p>0.5: Bias toward 1-bits
        Valid range: [0.0, 1.0]
        Default: 0.5

    dtype : jnp.dtype
        JAX data type for bit values.
        - jnp.int32 (default): Standard integer type, GPU-efficient
        - jnp.int8: More memory-efficient for very long genomes
        - jnp.int64: Rarely needed (overkill for binary)
        Default: jnp.int32

    Notes
    -----
    **Binary Genome Representation**:

    Binary genomes use **{0, 1} values internally** (not floating-point).
    Each bit is stored as an integer (typically jnp.int32 or jnp.int8).

    **Typical Uses**:
    - Combinatorial optimization (Knapsack, Traveling Salesman)
    - Feature selection (bit = feature on/off)
    - Boolean satisfiability (SAT) problems
    - Constrained selection problems

    **Initialization**:
    Each bit is sampled from Bernoulli(p), so approximately p fraction of bits
    are 1 and (1-p) fraction are 0 in initial population.
    - p=0.5 → ~50% 1-bits, ~50% 0-bits (unbiased)
    - p=0.1 → ~10% 1-bits, ~90% 0-bits (sparse)
    - p=0.9 → ~90% 1-bits, ~10% 0-bits (dense)

    **Genome Length via shape**:
    The effective genome_length is the product of all dimensions, e.g.:
    - shape=(64,) → length = 64 bits
    - shape=(8, 8) → length = 64 bits
    - shape=(16, 16, 4) → length = 1024 bits

    The :attr:`resolved_shape` method handles backward-compatibility
    by honoring the legacy `length` parameter if provided.
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

    def init_population(self, key: chex.PRNGKey, size: int) -> BasePopulation[BinaryGenome]:
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
        """Initialize bit-string via Bernoulli sampling at probability *p*.

        The returned genome has dtype and shape determined by the provided
        configuration.
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

        The other genome is cast to :class:`BinaryGenome`. Available metrics
        are Hamming (bitwise mismatch count) or Euclidean norm.
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
        """Convert the bit-string to an integer using positional weights.

        The *msb_first* flag controls bit significance ordering. The result is
        a scalar JAX array which may exceed native integer precision for long
        genomes.
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
    Parallel population container specialized for BinaryGenome bitstrings.

    As of v2.0, this class does not override core population mechanics or
    intercept `.values` properties. It serves strictly as a strongly-typed
    alias/subclass of `BasePopulation[BinaryGenome]` to provide convenient
    initialization helpers and IDE completion.
    """

    genes: BinaryGenome
    fitness: chex.Array
    config: BinaryGenomeConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

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
