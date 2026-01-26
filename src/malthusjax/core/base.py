from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar, Generic, Iterator, Type, TypeVar, Union, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

# Covariant TypeVar for the Genome implementation.
# This ensures that a RealPopulation is recognized as a valid BasePopulation[RealGenome]
G = TypeVar("G", bound="BaseGenome")


class DistanceMetric:
    """Standard metrics supported by most genomes."""

    HAMMING: str = "hamming"
    EUCLIDEAN: str = "euclidean"
    MANHATTAN: str = "manhattan"


@struct.dataclass
class BaseGenome:
    """
    Abstract blueprint for a single candidate solution (individual).

    In the MalthusJAX framework, a Genome is a PyTree container. While this
    base class defines logic for a single individual, implementations are
    designed to be 'lifted' via jax.vmap.

    When 'lifted' into a Population, each field in the Genome (e.g., 'values')
    is transformed from a scalar/vector into a batched array where the leading
    dimension represents the population size.
    """

    @classmethod
    @abstractmethod
    def random_init(cls: Type[G], key: chex.PRNGKey, config: Any) -> G:
        """
        Initialize a single genome instance with random values.

        Args:
            key: A JAX PRNG key for reproducibility.
            config: A configuration object specific to the genome implementation.

        Returns:
            An instance of the specific Genome subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def distance(self, other: BaseGenome, metric: str) -> chex.Numeric:
        """
        Calculate the phenotypic or genotypic distance to another genome.

        Note: The 'other' argument uses 'BaseGenome' to ensure compatibility
        with the Liskov Substitution Principle under strict type checking.
        Implementations should cast 'other' to their specific type.
        """
        raise NotImplementedError

    @abstractmethod
    def autocorrect(self, config: Any) -> BaseGenome:
        """
        Enforce domain-specific constraints on the genome (e.g., clipping bounds).

        This method is typically called after mutation or crossover to ensure
        the genome remains in a valid state.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        """The total number of parameters/elements within the genome."""
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The logical shape of the genome's primary data payload."""
        raise NotImplementedError

    @classmethod
    def create_population(cls: Type[G], key: chex.PRNGKey, config: Any, pop_size: int) -> G:
        """
        Factory method to create a batch of genomes using the SoA pattern.

        Utilizes jax.vmap to transform the 'random_init' logic of a single
        genome into a parallelized initialization of an entire population.
        """
        keys = jax.random.split(key, pop_size)
        return jax.vmap(cls.random_init, in_axes=(0, None))(keys, config)


@struct.dataclass
class BasePopulation(Generic[G]):
    """
    A unified container for a collection of candidate solutions.

    This class implements the Struct-of-Arrays (SoA) pattern. The 'genes'
    attribute holds a Genome instance where every leaf array has an added
    leading dimension of size N (population size).

    Attributes:
        genes: The batched genome data (SoA).
        fitness: A (N,) array representing the objective value for each individual.
        config: Static configuration shared by all individuals in the population.
    """

    genes: G
    fitness: chex.Array
    config: Any = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    # Reference to the Genome class for factory patterns and static analysis.
    GENOME_CLS: ClassVar[Type[Any]] = cast(Type[Any], Any)

    def spawn_offspring(self, new_genes: G) -> BasePopulation[G]:
        """
        Creates a new population container using a new batch of genes.

        Fitness is automatically reset to NaN to signify that the new
        individuals have not yet been evaluated.
        """
        leaves = jax.tree_util.tree_leaves(new_genes)
        if not leaves:
            raise ValueError("Gene structure contains no arrays.")

        n_offspring = leaves[0].shape[0]
        empty_fitness = jnp.broadcast_to(jnp.nan, (n_offspring,))

        return cast(
            BasePopulation[G], cast(Any, self).replace(genes=new_genes, fitness=empty_fitness)
        )

    def __len__(self) -> int:
        """Returns the number of individuals currently in the population."""
        return int(self.fitness.shape[0])

    def __getitem__(self, key: Union[int, slice, chex.Array]) -> Union[G, BasePopulation[G]]:
        """
        Provides intuitive slicing and indexing for the population.

        - If 'key' is an integer: Returns a single, unwrapped Genome (single individual).
        - If 'key' is a slice/mask: Returns a new Population instance containing
          only the selected individuals (sub-population).
        """
        sliced_genes = jax.tree_util.tree_map(lambda x: x[key], self.genes)

        if isinstance(key, int):
            return cast(G, sliced_genes)

        return cast(
            BasePopulation[G],
            cast(Any, self).replace(genes=sliced_genes, fitness=self.fitness[key]),
        )

    def __iter__(self) -> Iterator[G]:
        """
        Iterates over the population, yielding individual Genome instances.

        Note: Iteration is a Python-side operation and is slow. For heavy
        computation, prefer jax.vmap or jax.lax.scan over the population.
        """
        for i in range(len(self)):
            yield cast(G, self[i])

    def autocorrect(self, config: Any) -> BasePopulation[G]:
        """
        Applies the genome-level autocorrect logic to every individual
        in the population in parallel using jax.vmap.
        """
        new_genes = jax.vmap(lambda g: g.autocorrect(config))(self.genes)
        return cast(BasePopulation[G], cast(Any, self).replace(genes=new_genes))

    def distance_matrix(self, metric: str = DistanceMetric.EUCLIDEAN) -> chex.Array:
        """
        Computes pairwise distances between all individuals in the population.

        Args:
            metric: The distance metric to use. This is forwarded to the
                underlying genome-level ``distance`` implementation.

        Returns:
            A (N, N) array where entry (i, j) is the distance between the
            i-th and j-th individuals in the population.
        """

        def _pairwise_distance(g1: G, g2: G) -> chex.Array:
            # Delegate to the genome-level distance implementation.
            return g1.distance(g2, metric=metric)

        # Outer vmap iterates over the first individual, inner vmap over the
        # second individual, producing an (N, N) distance matrix.
        return jax.vmap(
            lambda g1: jax.vmap(lambda g2: _pairwise_distance(g1, g2))(self.genes)
        )(self.genes)
