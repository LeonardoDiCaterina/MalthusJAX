from abc import abstractmethod
from typing import Any, Type, TypeVar, Generic, ClassVar, Union, Iterator
from flax import struct
import jax
import jax.numpy as jnp
import chex

# Ensure G matches the bound
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
        """Initialize a single genome."""
        raise NotImplementedError

    @abstractmethod
    def distance(self, other: "BaseGenome", metric: str) -> float:
        """Compute distance to another genome instance."""
        raise NotImplementedError

    @abstractmethod
    def autocorrect(self, config: Any) -> "BaseGenome":
        """Fix constraints for a single genome."""
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def shape(self) -> tuple:
        raise NotImplementedError

    @classmethod
    def create_population(cls: Type[G], key: chex.PRNGKey, config: Any, pop_size: int) -> G:
        """Factory: Creates a batch of genomes (Struct of Arrays)."""
        keys = jax.random.split(key, pop_size)
        return jax.vmap(cls.random_init, in_axes=(0, None))(keys, config)


@struct.dataclass
class BasePopulation(Generic[G]):
    """Abstract population container."""
    genes: G 
    fitness: chex.Array
    config: Any = struct.field(pytree_node=False)
    
    # Generic type placeholder, mostly for static analysis
    GENOME_CLS: ClassVar[Type[G]] = Any 

    def spawn_offspring(self, new_genes: G) -> "BasePopulation":
        """
        Creates a new population instance with new genes, resetting fitness.
        """
        # 1. Infer new population size safely (No assumptions about .values)
        leaves = jax.tree_util.tree_leaves(new_genes)
        if not leaves:
            raise ValueError("New genes struct is empty.")
        n_offspring = leaves[0].shape[0]

        # 2. "Allocate" Space (Zero-Cost Broadcast)
        empty_fitness = jnp.broadcast_to(jnp.nan, (n_offspring,))

        # 3. Return new instance
        return self.replace(
            genes=new_genes,
            fitness=empty_fitness
        )
        
    @classmethod
    @abstractmethod
    def init_random(cls, key: chex.PRNGKey, config: Any, size: int) -> "BasePopulation[G]":
        raise NotImplementedError

    def __len__(self) -> int:
        return int(self.fitness.shape[0])

    def __getitem__(self, key: Union[int, slice, chex.Array]) -> Union[G, "BasePopulation[G]"]:
        """
        Slicing returns a smaller Population (wrapped).
        Indexing returns a single Genome (unwrapped).
        """
        # Slice the genes struct
        sliced_genes = jax.tree_util.tree_map(lambda x: x[key], self.genes)
        
        if isinstance(key, int):
            # Return unwrapped Genome
            return sliced_genes
        else:
            # Return wrapped Population with sliced fitness
            return self.replace(genes=sliced_genes, fitness=self.fitness[key])

    def __iter__(self) -> Iterator[G]:
        """
        WARNING: Iteration in Python is slow. Use vmap/scan for logic.
        This is primarily for debugging or printing.
        """
        for i in range(len(self)):
            yield self[i]

    def autocorrect(self, config: Any) -> "BasePopulation[G]":
        # vmap automatically handles the struct unboxing/reboxing
        new_genes = jax.vmap(lambda g: g.autocorrect(config))(self.genes)
        return self.replace(genes=new_genes)

    def distance_matrix(self, metric: str = "hamming") -> chex.Array:
        """
        Compute pairwise distance matrix.
        Returns matrix of shape (N, N).
        """
        # vmap logic:
        # Outer vmap: splits 'self.genes' (A) into rows -> g1
        # Inner vmap: splits 'self.genes' (B) into cols -> g2
        # Because BaseGenome is a struct, 'g1' and 'g2' inside the loop 
        # are valid Genome instances representing a single row.
        
        def dist_fn(g1, g2):
            return g1.distance(g2, metric)
            
        return jax.vmap(
            jax.vmap(dist_fn, in_axes=(None, 0)), 
            in_axes=(0, None)
        )(self.genes, self.genes)