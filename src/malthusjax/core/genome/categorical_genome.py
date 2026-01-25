"""
Categorical genome implementation for discrete choice problems.

Provides CategoricalGenome and CategoricalPopulation for discrete choice
representations commonly used in combinatorial optimization like TSP, scheduling.
"""

from typing import ClassVar, Type

import chex  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from flax import struct  # type: ignore

from malthusjax.core.base import BaseGenome, BasePopulation


@struct.dataclass
class CategoricalGenomeConfig:
    """Configuration for Categorical genomes."""
    length: int           # Number of positions in the sequence
    num_categories: int   # Number of possible values at each position


def validate_categorical_config(config: CategoricalGenomeConfig) -> None:
    """Validate categorical genome configuration parameters."""
    if config.length <= 0:
        raise ValueError(f"Length must be positive, got {config.length}")
    if config.num_categories <= 1:
        raise ValueError(f"Number of categories must be greater than 1, got {config.num_categories}")


@struct.dataclass
class CategoricalGenome(BaseGenome):
    """
    Categorical genome for discrete choice optimization.
    
    Represents solutions as sequences of discrete categories, useful for
    problems like TSP, job scheduling, resource allocation, etc.
    Each position can take values from 0 to num_categories-1.
    """
    categories: chex.Array  # Shape (length,) - Integer array

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: CategoricalGenomeConfig) -> "CategoricalGenome":
        """Create random categorical genome."""
        categories = jax.random.randint(
            key, (config.length,), 0, config.num_categories
        )
        return cls(categories=categories)

    def autocorrect(self, config: CategoricalGenomeConfig) -> "CategoricalGenome":
        """Ensure all categories are within valid range."""
        corrected_categories = jnp.clip(self.categories, 0, config.num_categories - 1)
        return self.replace(categories=corrected_categories)

    def distance(self, other: "CategoricalGenome", metric: str = "hamming") -> float:
        """Compute distance between categorical genomes."""
        if metric == "hamming":
            return jnp.sum(self.categories != other.categories).astype(jnp.float32)
        elif metric == "euclidean":
            # Treat categories as points in space
            return jnp.sqrt(jnp.sum((self.categories - other.categories) ** 2)).astype(jnp.float32)
        elif metric == "manhattan":
            return jnp.sum(jnp.abs(self.categories - other.categories)).astype(jnp.float32)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @property
    def size(self) -> int:
        """Return number of categorical positions."""
        return self.categories.shape[-1]

    @property
    def shape(self) -> tuple:
        """Return shape of genome as (length,)."""
        return self.categories.shape

    def is_permutation(self) -> bool:
        """Check if this genome represents a valid permutation (all unique values)."""
        unique_vals = jnp.unique(self.categories)
        return len(unique_vals) == len(self.categories)

    def to_permutation(self, config: CategoricalGenomeConfig) -> "CategoricalGenome":
        """
        Convert to a valid permutation by ensuring all values 0..length-1 appear exactly once.
        
        This is useful for TSP-like problems where each city must be visited exactly once.
        """
        if config.num_categories != config.length:
            raise ValueError("Can only convert to permutation when num_categories == length")

        # Sort the categories to get a permutation
        sorted_indices = jnp.argsort(self.categories)
        permutation = jnp.arange(config.length)[sorted_indices]
        return self.replace(categories=permutation)

    def swap_positions(self, pos1: int, pos2: int) -> "CategoricalGenome":
        """Swap categories at two positions."""
        new_categories = self.categories.at[pos1].set(self.categories[pos2])
        new_categories = new_categories.at[pos2].set(self.categories[pos1])
        return self.replace(categories=new_categories)

    def count_category(self, category: int) -> int:
        """Count occurrences of a specific category."""
        return int(jnp.sum(self.categories == category))

    def get_category_distribution(self, config: CategoricalGenomeConfig) -> chex.Array:
        """Get distribution of categories (histogram)."""
        # Count occurrences of each category
        counts = jnp.zeros(config.num_categories)
        for i in range(config.num_categories):
            counts = counts.at[i].set(jnp.sum(self.categories == i))
        return counts

    def __repr__(self) -> str:
        if self.size <= 10:
            cats_str = ", ".join(str(int(c)) for c in self.categories)
        else:
            cats_str = ", ".join(str(int(c)) for c in self.categories[:8])
            cats_str += f", ..., {int(self.categories[-1])}"
        return f"<CategoricalGenome([{cats_str}], len={self.size})>"


@struct.dataclass
class CategoricalPopulation(BasePopulation[CategoricalGenome]):
    """Population container for CategoricalGenome objects."""
    genes: CategoricalGenome
    fitness: chex.Array
    config: CategoricalGenomeConfig

    GENOME_CLS: ClassVar[Type[CategoricalGenome]] = CategoricalGenome

    @classmethod
    def init_random(cls, key: chex.PRNGKey, config: CategoricalGenomeConfig, size: int) -> "CategoricalPopulation":
        """Create random population of categorical genomes."""
        batched_genes = CategoricalGenome.create_population(key, config, size)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)
