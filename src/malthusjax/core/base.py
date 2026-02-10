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

_field: Any = struct.field


class DistanceMetric:
    """Standard metrics supported by most genomes."""

    HAMMING: str = "hamming"
    EUCLIDEAN: str = "euclidean"
    MANHATTAN: str = "manhattan"


@struct.dataclass
class BaseGenome:
    """Immutable genome representation optimized for JAX PyTree lifting.

    Single-genome methods compose with jax.vmap to implement population-level
    operations via the Struct-of-Arrays (SoA) pattern: each leaf array gains
    a leading batch dimension (N,). The `subscriptable` flag enables optional
    Pythonic indexing/iteration, trading PyTree traceability for convenience.
    """

    def __len__(self) -> int:
        """Return number of elements in the primary values array."""
        try:
            return int(cast(Any, self).values.shape[0])
        except Exception as e:
            raise TypeError("len() is not supported for this genome (missing 'values').") from e

    def __getitem__(self, key: Union[int, slice, chex.Array]) -> Any:
        """Index into the genome's primary values payload if enabled.

        Subclasses enable this behavior by declaring a `subscriptable` field
        if they want Pythonic indexing/iteration semantics.
        """
        if not getattr(self, "subscriptable", False):
            msg = (
                f"{self.__class__.__name__} object is not subscriptable; "
                "set subscriptable=True to enable indexing."
            )
            raise TypeError(msg)
        try:
            return cast(Any, self).values[key]
        except AttributeError as e:
            raise TypeError("Genome does not expose 'values' for indexing.") from e

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the genome's primary values payload if enabled."""
        if not getattr(self, "subscriptable", False):
            msg = (
                f"{self.__class__.__name__} object is not iterable; "
                "set subscriptable=True to enable iteration."
            )
            raise TypeError(msg)
        return iter(cast(Any, self).values)

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
    @abstractmethod
    def from_tensor(cls: Type[G], arr: chex.Array, config: Any = None) -> G:
        """Construct a batched Genome instance from a raw array.

        `arr` is expected to be a batched array with leading dim equal to the
        number of individuals (or offspring). This method must be JIT-
        friendly and avoid Python-side checks on traced arrays.
        """
        raise NotImplementedError

    @classmethod
    def create_population(cls: Type[G], key: chex.PRNGKey, config: Any, pop_size: int) -> G:
        """Vectorized population initialization via jax.vmap.

        Transforms single-genome random_init(key, config) into batched
        initialization by vmapping over split random keys (in_axes=(0, None)).
        Returns SoA-lifted genome where each array leaf has shape (pop_size, ...).
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
    config: Any = _field(pytree_node=False)
    GENOME_CLS: ClassVar[Type[Any]] = cast(Type[Any], Any)

    @classmethod
    def from_array(cls, arr: chex.Array, config: Any, axis: int = 0) -> BasePopulation[G]:
        """Construct a population from a raw JAX array.

        Interprets ``axis`` as the population (batch) dimension. The array
        is rearranged so that the population dimension becomes the leading
        axis, then each slice along that axis is treated as one genome.

        For example, given ``arr.shape == (x, y, z)`` and ``axis=1``, the
        resulting population has ``y`` individuals each with genome shape
        ``(x, z)``.

        Args:
            arr: Raw array containing all individuals' data.
            config: Genome-level configuration (forwarded to
                ``GENOME_CLS.from_tensor``).
            axis: Which dimension of *arr* corresponds to the population.
                Defaults to ``0`` (leading dimension).

        Returns:
            A new population with fitness initialized to ``-inf``.
        """
        arr_batched = jnp.moveaxis(arr, axis, 0)
        pop_size = arr_batched.shape[0]
        genes = cls.GENOME_CLS.from_tensor(arr_batched, config)
        fitness = jnp.full((pop_size,), -jnp.inf)
        return cls(genes=genes, fitness=fitness, config=config)

    @property
    def values(self) -> Any:
        """Proxies to the genome's values (batched)."""
        return cast(Any, self.genes).values

    def spawn_offspring(self, new_genes: G) -> BasePopulation[G]:
        """Create offspring population with evaluation-pending fitness state.

        Resets fitness to NaN to signal pending evaluation, allowing engines
        to detect unevaluated individuals via jnp.isnan() without conditional
        branching (XLA-compatible).
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

        def _autocorrect_genome(g: G) -> G:
            return cast(G, g.autocorrect(config))

        new_genes = jax.vmap(_autocorrect_genome)(self.genes)
        return cast(BasePopulation[G], cast(Any, self).replace(genes=new_genes))

    def distance_matrix(self, metric: str = DistanceMetric.EUCLIDEAN) -> chex.Array:
        """Compute pairwise distances between all population members.

        Nested vmap: outer fixes individual i, inner iterates over all j,
        generating (N, N) distance matrix without Python loops (JIT-safe).

        Args:
            metric: Distance metric forwarded to genome-level distance().

        Returns:
            (N, N) array where (i, j) = distance(genome[i], genome[j]).
        """

        def _pairwise_distance(g1: G, g2: G) -> chex.Array:
            """Delegate to the genome-level distance implementation."""
            return g1.distance(g2, metric=metric)

        def _vmap_second(g1: G) -> chex.Array:
            def _distance_wrapper(g2: G) -> chex.Array:
                return _pairwise_distance(g1, g2)

            return jax.vmap(_distance_wrapper)(self.genes)

        # Outer vmap iterates over the first individual, inner vmap over the
        # second individual, producing an (N, N) distance matrix.
        return jax.vmap(_vmap_second)(self.genes)
