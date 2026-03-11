"""Categorical genome representation for discrete search spaces.

Provides integer‑indexed sequence genomes along with permutation helpers
and distance metrics. A matching population class offers batched
operations and initialization.
"""

from __future__ import annotations

from typing import Any, ClassVar, Tuple, Type, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation, DistanceMetric


@struct.dataclass
class CategoricalGenomeConfig:
    """Configuration for discrete categorical genomes.

    Attributes:
        num_categories: Cardinality of the discrete alphabet.
        shape: Logical shape of the categorical sequence.
        dtype: JAX dtype for category indices; typically jnp.int32.
    """

    num_categories: int
    shape: Tuple[int, ...] = struct.field(pytree_node=False, default_factory=lambda: ())  # type: ignore[no-untyped-call]
    dtype: jnp.dtype[Any] = struct.field(
        pytree_node=False,
        default=jnp.int32,  # type: ignore[no-untyped-call]
    )


    def init_population(
        self, key: chex.PRNGKey, size: int
    ) -> BasePopulation[CategoricalGenome]:
        """Create a random population from this config (protocol method for JR-2)."""
        return CategoricalPopulation.init_random(key, self, size)


@struct.dataclass
class CategoricalGenome(BaseGenome):
    """Categorical sequence genome for discrete optimization.

    Values are integer indices in [0, num_categories). Suitable for
    permutation-based and combinatorial problems.
    """

    values: chex.Array  # Shape: (length,) for individuals, (N, length) for populations

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: CategoricalGenomeConfig) -> CategoricalGenome:
        """Create random categorical genome using discrete uniform sampling."""
        values = jax.random.randint(key, config.shape, 0, config.num_categories).astype(
            config.dtype
        )
        return cls(values=values)

    def autocorrect(self, config: CategoricalGenomeConfig) -> CategoricalGenome:
        """Ensure all categories are within [0, num_categories-1]."""
        corrected_values = jnp.clip(self.values, 0, config.num_categories - 1)
        return cast(CategoricalGenome, cast(Any, self).replace(values=corrected_values))

    def distance(self, other: BaseGenome, metric: str = DistanceMetric.HAMMING) -> chex.Numeric:
        """Compute distance between categorical genomes.

        Args:
            other: Another genome; cast to CategoricalGenome internally.
            metric: 'hamming' (mismatch count), 'euclidean' (L2), or 'manhattan' (L1).

        Returns:
            Scalar distance value (JAX array, JIT-compatible).
        """
        other_cat = cast(CategoricalGenome, other)

        if metric == DistanceMetric.HAMMING:
            return jnp.sum(self.values != other_cat.values)
        elif metric == DistanceMetric.EUCLIDEAN:
            return jnp.sqrt(jnp.sum(jnp.square(self.values - other_cat.values)))
        elif metric == DistanceMetric.MANHATTAN:
            return jnp.sum(jnp.abs(self.values - other_cat.values))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @property
    def size(self) -> int:
        """Return number of categorical positions."""
        return int(self.values.shape[-1])

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of the genome array."""
        return cast(tuple[int, ...], self.values.shape)

    @classmethod
    def from_tensor(cls, arr: chex.Array, config: Any = None) -> "CategoricalGenome":
        """Construct a batched CategoricalGenome from a raw array.

        Implementation is intentionally trivial and JIT-safe.
        """
        return cls(values=arr)

    def is_permutation(self) -> chex.Numeric:
        """Check if all categorical values are unique (valid permutation).

        Uses jnp.unique(size=self.size, fill_value=-1) to pad uniques to
        expected length, then verifies absence of sentinel (-1). Returns
        JAX scalar boolean (JIT-safe; avoids Python control flow).
        """
        unique_vals = jnp.unique(self.values, size=self.size, fill_value=-1)
        return jnp.all(unique_vals != -1)

    def to_permutation(self, config: CategoricalGenomeConfig) -> CategoricalGenome:
        """Generate permutation via argsort; deterministic and JIT-safe."""
        # This is a standard JAX trick to generate a permutation from any vector
        permutation = jnp.argsort(self.values).astype(config.dtype)
        return cast(CategoricalGenome, cast(Any, self).replace(values=permutation))

    def swap_positions(self, pos1: int, pos2: int) -> CategoricalGenome:
        """Exchange values at two indices via functional .at updates."""
        val1 = self.values[pos1]
        val2 = self.values[pos2]
        new_values = self.values.at[pos1].set(val2)
        new_values = new_values.at[pos2].set(val1)
        return cast(CategoricalGenome, cast(Any, self).replace(values=new_values))

    def count_category(self, category: int) -> chex.Numeric:
        """Return count of a specific category value. Scalar JAX array."""
        return jnp.sum(self.values == category)

    def __repr__(self) -> str:
        try:
            sample = self.values[:8]
            cats_str = ", ".join(str(int(c)) for c in sample)
            if self.size > 8:
                cats_str += f", ..., {int(self.values[-1])}"
            return f"<CategoricalGenome([{cats_str}], len={self.size})>"
        except Exception:
            return f"<CategoricalGenome(traced, len={self.size})>"


@struct.dataclass
class CategoricalPopulation(BasePopulation[CategoricalGenome]):
    """Population container for CategoricalGenome objects."""

    genes: CategoricalGenome
    fitness: chex.Array
    config: CategoricalGenomeConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    GENOME_CLS: ClassVar[Type[CategoricalGenome]] = CategoricalGenome

    @classmethod
    def init_random(
        cls, key: chex.PRNGKey, config: CategoricalGenomeConfig, size: int
    ) -> CategoricalPopulation:
        """Create random population of categorical genomes."""
        batched_genes = CategoricalGenome.create_population(key, config, size)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)
