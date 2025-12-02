"""
Core abstract base classes for MalthusJAX.
"""

from abc import abstractmethod
from typing import Any, Type, TypeVar, Generic, ClassVar, Union, Iterator, Optional
from flax import struct  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import chex  # type: ignore


G = TypeVar("G", bound="BaseGenome")


class DistanceMetric:
    HAMMING = "hamming"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


@struct.dataclass
class BaseGenome:
    """Abstract base class for a single individual/genome."""
    
    @classmethod
    @abstractmethod
    def random_init(cls: Type[G], key: chex.PRNGKey, config: Any) -> G:
        raise NotImplementedError

    @abstractmethod
    def distance(self, other: "BaseGenome", metric: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def autocorrect(self, config: Any) -> "BaseGenome":
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    @classmethod
    def create_population(cls: Type[G], key: chex.PRNGKey, config: Any, pop_size: int) -> G:
        keys = jax.random.split(key, pop_size)
        return jax.vmap(cls.random_init, in_axes=(0, None))(keys, config)


@struct.dataclass
class BasePopulation(Generic[G]):
    """Abstract population container."""
    genes: G 
    fitness: chex.Array
    config: Any = struct.field(pytree_node=False)
    
    GENOME_CLS: ClassVar[Type[G]] = None

    @classmethod
    @abstractmethod
    def init_random(cls, key: chex.PRNGKey, config: Any, size: int) -> "BasePopulation[G]":
        raise NotImplementedError

    def __len__(self) -> int:
        return int(self.fitness.shape[0])

    def __getitem__(self, key: Union[int, slice, chex.Array]) -> Union[G, "BasePopulation[G]"]:
        sliced_genes = jax.tree_util.tree_map(lambda x: x[key], self.genes)
        if isinstance(key, int):
            return sliced_genes
        else:
            return self.replace(genes=sliced_genes, fitness=self.fitness[key])

    def __iter__(self) -> Iterator[G]:
        for i in range(len(self)):
            yield self[i]

    def autocorrect(self, config: Any) -> "BasePopulation[G]":
        new_genes = jax.vmap(lambda g: g.autocorrect(config))(self.genes)
        return self.replace(genes=new_genes)

    def distance_matrix(self, metric: str = "hamming") -> chex.Array:
        """
        Compute pairwise distance matrix using explicit tree reconstruction.
        This handles cases where vmap strips the class wrapper.
        """
        # 1. Get the structure (treedef) and leaves
        leaves, treedef = jax.tree_util.tree_flatten(self.genes)
        
        # 2. Define distance on LEAVES, reconstructing objects inside
        def dist_on_leaves(leaves_a, leaves_b):
            # Reconstruct the Genome Objects
            g_a = jax.tree_util.tree_unflatten(treedef, leaves_a)
            g_b = jax.tree_util.tree_unflatten(treedef, leaves_b)
            return g_a.distance(g_b, metric)

        # 3. Vectorize
        # vmap over rows (a), then cols (b).
        # Inputs will be tuples of leaves.
        # We need to handle the tuple structure manually for vmap if multiple leaves exist.
        
        # Helper to wrap the vmap logic
        # We act on the leaves directly
        vmapped_dist = jax.vmap(
            jax.vmap(dist_on_leaves, in_axes=(None, 0)), 
            in_axes=(0, None)
        )
        
        # JAX vmap expects *args. If leaves is a list [arr1, arr2], we unpack it?
        # Actually, if we pass the whole 'leaves' list as a single arg, vmap treats it as a list of arrays
        # and tries to map over the list structure which is wrong.
        
        # ROBUST STRATEGY: Use a lambda that takes the original object structure
        # but force reconstruction if it fails? No.
        
        # Let's fallback to the simplest robust implementation: 
        # Pass the PyTree (self.genes) and let JAX handle it, but use the class method
        # if the instance method fails? No.
        
        # Implementation using implicit reconstruction:
        def safe_dist(g1, g2):
            return g1.distance(g2, metric)
            
        # We assume self.genes is a valid PyTree node.
        # If it wasn't, init would fail.
        return jax.vmap(jax.vmap(safe_dist, in_axes=(None, 0)), in_axes=(0, None))(
            self.genes, self.genes
        )