"""
Core abstract base classes for MalthusJAX.

This module defines the fundamental abstractions for genomes and populations
with JAX-native design, automatic vectorization, and clean functional interfaces.
"""

from abc import abstractmethod
from typing import Any, Type, TypeVar, Generic, ClassVar, Union, Iterator, Optional
from flax import struct  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import chex  # type: ignore


# Type variables for generic components
G = TypeVar("G", bound="BaseGenome")


class DistanceMetric:
    """Standard distance metrics for genome comparison."""
    HAMMING = "hamming"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


@struct.dataclass
class BaseGenome:
    """
    Abstract base class for a single individual/genome.
    
    Uses Flax struct.dataclass for JAX compatibility and immutability.
    All genomes are JAX arrays underneath for efficient vectorization.
    """
    
    # --- Abstract Interface ---
    @classmethod
    @abstractmethod
    def random_init(cls: Type[G], key: chex.PRNGKey, config: Any) -> G:
        """Create ONE random genome with given configuration."""
        raise NotImplementedError

    @abstractmethod
    def distance(self, other: "BaseGenome", metric: str) -> float:
        """Compute distance between this genome and another."""
        raise NotImplementedError

    @abstractmethod
    def autocorrect(self, config: Any) -> "BaseGenome":
        """Fix any invalid states in this genome."""
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the size/length of this genome."""
        raise NotImplementedError

    # --- Shared Logic (Vectorization) ---
    @classmethod
    def create_population(cls: Type[G], key: chex.PRNGKey, config: Any, pop_size: int) -> G:
        """
        Vectorize random_init to create a raw batch of genomes.
        Returns batched genomes with shape (Pop, ...) for all genome arrays.
        
        Note: This returns raw genome batches, not a Population object.
        Use Population.init_random() to get a proper Population container.
        """
        keys = jax.random.split(key, pop_size)
        return jax.vmap(cls.random_init, in_axes=(0, None))(keys, config)


@struct.dataclass
class BasePopulation(Generic[G]):
    """
    Abstract population container with list-like behavior and automatic vectorization.
    
    Provides clean indexing, slicing, and batch operations over genome collections.
    All operations are JAX-compiled for performance.
    """
    genes: G  # Batch of genomes
    fitness: chex.Array  # Shape (pop_size,) or (pop_size, num_objectives)
    config: Any = struct.field(pytree_node=False)  # Genome configuration (static)
    
    GENOME_CLS: ClassVar[Type[G]] = None  # Override in concrete classes

    # --- Factory Pattern (Critical for Engine) ---
    @classmethod
    @abstractmethod
    def init_random(cls, key: chex.PRNGKey, config: Any, size: int) -> "BasePopulation[G]":
        """Create a new population with random genomes and initialized fitness."""
        raise NotImplementedError

    # --- List-like Interface ---
    def __len__(self) -> int:
        """Return population size."""
        return int(self.fitness.shape[0])

    def __getitem__(self, key: Union[int, slice, chex.Array]) -> Union[G, "BasePopulation[G]"]:
        """
        Index into population. Supports integer, slice, and array indexing.
        
        Args:
            key: Index (int returns single genome, others return sub-population)
            
        Returns:
            Single genome (int key) or BasePopulation (other keys)
        """
        # Slice the genes Pytree automatically
        sliced_genes = jax.tree_util.tree_map(lambda x: x[key], self.genes)
        
        if isinstance(key, int):
            # Return single genome
            return sliced_genes
        else:
            # Return sliced population
            return self.replace(genes=sliced_genes, fitness=self.fitness[key])

    def __iter__(self) -> Iterator[G]:
        """Iterate over individual genomes."""
        for i in range(len(self)):
            yield self[i]

    # --- Automated Vectorized Operations ---
    def autocorrect(self, config: Any) -> "BasePopulation[G]":
        """Apply autocorrect to entire population."""
        new_genes = jax.vmap(lambda g: g.autocorrect(config))(self.genes)
        return self.replace(genes=new_genes)

    def distance_matrix(self, metric: str = "hamming") -> chex.Array:
        """
        Compute pairwise distance matrix between all genomes.
        
        Returns:
            Array of shape (pop_size, pop_size) with distances
        """
        pair_fn = lambda g1, g2: g1.distance(g2, metric)
        return jax.vmap(jax.vmap(pair_fn, in_axes=(None, 0)), in_axes=(0, None))(
            self.genes, self.genes
        )