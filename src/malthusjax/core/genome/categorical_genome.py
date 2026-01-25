from __future__ import annotations
from typing import Any, ClassVar, Type, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation, DistanceMetric

@struct.dataclass
class CategoricalGenomeConfig:
    """
    Configuration for categorical genomes.
    
    Attributes:
        length: Number of positions in the sequence.
        num_categories: Number of possible discrete values at each position.
        dtype: Data type for the categories, typically jnp.int32.
    """
    length: int
    num_categories: int
    dtype: jnp.dtype[Any] = struct.field(
        pytree_node=False, default=jnp.int32 # type: ignore[no-untyped-call]
    )

@struct.dataclass
class CategoricalGenome(BaseGenome):
    """
    Categorical genome for discrete choice optimization.
    
    Represents solutions as sequences of discrete labels (integers), 
    ideal for problems like TSP, job scheduling, or resource allocation.
    """
    categories: chex.Array  # Shape: (length,) for individuals, (N, length) for populations

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: CategoricalGenomeConfig) -> CategoricalGenome:
        """Create random categorical genome using discrete uniform sampling."""
        categories = jax.random.randint(
            key, (config.length,), 0, config.num_categories, dtype=config.dtype
        )
        return cls(categories=categories)

    def autocorrect(self, config: CategoricalGenomeConfig) -> CategoricalGenome:
        """Ensure all categories are within [0, num_categories-1]."""
        corrected_categories = jnp.clip(self.categories, 0, config.num_categories - 1)
        return cast(CategoricalGenome, cast(Any, self).replace(categories=corrected_categories))

    def distance(self, other: BaseGenome, metric: str = DistanceMetric.HAMMING) -> chex.Numeric:
        """
        Compute distance between categorical genomes.
        
        Note: Hamming distance is the standard for categorical data as it 
        represents the number of mismatched positions.
        """
        other_cat = cast(CategoricalGenome, other)
        
        if metric == DistanceMetric.HAMMING:
            return jnp.sum(self.categories != other_cat.categories)
        elif metric == DistanceMetric.EUCLIDEAN:
            return jnp.sqrt(jnp.sum(jnp.square(self.categories - other_cat.categories)))
        elif metric == DistanceMetric.MANHATTAN:
            return jnp.sum(jnp.abs(self.categories - other_cat.categories))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @property
    def size(self) -> int:
        """Return number of categorical positions."""
        return int(self.categories.shape[-1])

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of the genome array."""
        return cast(tuple[int, ...], self.categories.shape)

    def is_permutation(self) -> chex.Numeric:
        """Check if this genome represents a valid permutation (all unique values)."""
        unique_vals = jnp.unique(self.categories, size=self.size, fill_value=-1)
        # Check if number of unique values matches the length
        # Note: We keep this as a JAX boolean/scalar for JIT compatibility
        return jnp.all(unique_vals != -1)

    def to_permutation(self, config: CategoricalGenomeConfig) -> CategoricalGenome:
        """Convert to a valid permutation via argsort."""
        # This is a standard JAX trick to generate a permutation from any vector
        permutation = jnp.argsort(self.categories).astype(config.dtype)
        return cast(CategoricalGenome, cast(Any, self).replace(categories=permutation))

    def swap_positions(self, pos1: int, pos2: int) -> CategoricalGenome:
        """Swap categories at two positions using JAX .at index updates."""
        val1 = self.categories[pos1]
        val2 = self.categories[pos2]
        new_categories = self.categories.at[pos1].set(val2)
        new_categories = new_categories.at[pos2].set(val1)
        return cast(CategoricalGenome, cast(Any, self).replace(categories=new_categories))

    def count_category(self, category: int) -> chex.Numeric:
        """Count occurrences of a specific category."""
        return jnp.sum(self.categories == category)

    def __repr__(self) -> str:
        try:
            sample = self.categories[:8]
            cats_str = ", ".join(str(int(c)) for c in sample)
            if self.size > 8:
                cats_str += f", ..., {int(self.categories[-1])}"
            return f"<CategoricalGenome([{cats_str}], len={self.size})>"
        except Exception:
            return f"<CategoricalGenome(traced, len={self.size})>"

@struct.dataclass
class CategoricalPopulation(BasePopulation[CategoricalGenome]):
    """Population container for CategoricalGenome objects."""
    genes: CategoricalGenome
    fitness: chex.Array
    config: CategoricalGenomeConfig = struct.field(pytree_node=False) # type: ignore[no-untyped-call]

    GENOME_CLS: ClassVar[Type[CategoricalGenome]] = CategoricalGenome

    @classmethod
    def init_random(
        cls, key: chex.PRNGKey, config: CategoricalGenomeConfig, size: int
    ) -> CategoricalPopulation:
        """Create random population of categorical genomes."""
        batched_genes = CategoricalGenome.create_population(key, config, size)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)