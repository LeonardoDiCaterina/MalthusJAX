"""Core genome and population abstractions.

This module defines the base classes used throughout the framework. It
provides immutable, JAX‑friendly Genome and Population types along with
utility functions for random initialization, distance metrics and structural
helpers. All types are designed to be compatible with JAX's PyTree and
jit/vmap patterns.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace
from typing import Any, Generic, Iterator, Optional, Type, TypeVar, Union, cast

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
    operations via the Struct-of-Arrays (SoA) pattern: each leaf array in the
    PyTree gains a leading batch dimension (N,).

    Type System Note:
    Tier-3 vectorization leverages `jax.tree_util` maps, treating `BaseGenome`
    subclasses as opaque PyTrees. There is no explicit requirement for a single
    primary data array. However, as a unified framework practice, implementations
    lacking domain-specific structural requirements should conventionally store
    their primary payload in a `.values` attribute.

    The `subscriptable` flag enables optional Pythonic indexing/iteration over
    the conventional `.values` attribute, trading pure PyTree traceability for
    convenience in single-genome extraction.
    """

    def __len__(self) -> int:
        """Return number of elements in the primary values array."""
        try:
            return int(getattr(self, "values").shape[0])
        except Exception as e:
            raise TypeError("len() is not supported for this genome (missing 'values').") from e

    def __getitem__(self, key: Union[int, slice, chex.Array]) -> Any:
        """Index into the genome, returning a genome type when possible.

        Behavior depends on whether this is a batched genome or a single genome:

        **Batched genome indexing** (all genome subtypes):
            - Integer key: Extract individual genome and reconstruct via tree_map
                pop.genes[0] → returns RealGenome / CategoricalGenome / etc.
            - Slice/array key: Extract sub-population of genomes with same type
                pop.genes[10:20] → returns batched genome with 10 individuals

        **Single genome value indexing** (subscriptable=True only):
            - scalar/array return from genome.values array directly
                The genome must declare subscriptable=True to enable this.

        Implementation leverages JAX PyTree structure: tree_map automatically
        reconstructs the correct subclass type (RealGenome, CategoricalGenome, etc.)
        based on the frozen dataclass wrapper, with no explicit dispatch needed.
        """
        # Determine if this is a batched genome by checking array leaf dimensions
        leaves = jax.tree_util.tree_leaves(self)
        is_batched = leaves and hasattr(leaves[0], "shape") and len(leaves[0].shape) > 1

        if is_batched:
            # Batched genome: use tree_map to extract and reconstruct individual/sub-genome
            # This preserves the type:
            # RealGenome → RealGenome, CategoricalGenome → CategoricalGenome, etc.
            return jax.tree_util.tree_map(lambda x: x[key], self)

        # Single genome value indexing (subscriptable mode only)
        if not getattr(self, "subscriptable", False):
            msg = (
                f"{self.__class__.__name__} object is not subscriptable; "
                "set subscriptable=True to enable value-level indexing."
            )
            raise TypeError(msg)
        try:
            return getattr(self, "values")[key]
        except AttributeError as e:
            raise TypeError("Genome does not expose 'values' for indexing.") from e

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the genome, yielding genomes or values depending on type.

        **Batched genome iteration**:
            Yields individual reconstructed genomes for each population member.
            Example: iterating batched RealGenome yields RealGenome individuals.

        **Single genome iteration** (subscriptable=True only):
            Yields values from the genome's values array.
            Example: iterating subscriptable RealGenome yields scalars.
        """
        # Determine if this is a batched genome
        leaves = jax.tree_util.tree_leaves(self)
        is_batched = leaves and hasattr(leaves[0], "shape") and len(leaves[0].shape) > 1

        if is_batched:
            # Batched genome: iterate and yield individual genomes
            n = int(leaves[0].shape[0])
            for i in range(n):
                yield self[i]  # Uses __getitem__ to extract and reconstruct
        else:
            # Single genome value iteration (subscriptable mode only)
            if not getattr(self, "subscriptable", False):
                msg = (
                    f"{self.__class__.__name__} object is not iterable; "
                    "set subscriptable=True to enable value-level iteration, "
                    "or iterate a batched genome to yield individual genomes."
                )
                raise TypeError(msg)
            for val in getattr(self, "values"):
                yield val

    @classmethod
    @abstractmethod
    def random_init(cls: Type[G], key: chex.PRNGKey, config: Any) -> G:
        """
        Initialize a single genome instance with random values.

        Implementations should draw from the appropriate distribution using
        *key* and respect any bounds encoded in *config*. The result is a
        concrete `BaseGenome` subclass instance ready for evaluation.
        """
        raise NotImplementedError

    def clone_buffers(self: G) -> G:
        """Deep-copies all JAX/NumPy array leaves in the PyTree.

        Guarantees buffer isolation for safe JAX buffer donation (`donate_argnums`)
        across multi-seed, multi-pipeline, or host-device transfers.
        """
        return cast(
            G,
            jax.tree_util.tree_map(
                lambda x: jnp.array(x, copy=True) if hasattr(x, "shape") else x,
                self,
            ),
        )

    def copy(self: G) -> G:
        """Alias for :meth:`clone_buffers` for backwards compatibility."""
        return self.clone_buffers()

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

        The supplied array should already include the population dimension as
        its leading axis. This factory is intentionally minimal to allow
        JIT tracing without Python conditionals.
        """
        raise NotImplementedError

    @classmethod
    def create_population(cls: Type[G], key: chex.PRNGKey, config: Any, pop_size: int) -> G:
        """Vectorized population initialization via :func:`jax.vmap`.

        Splits *key* into *pop_size* subkeys and calls ``random_init`` on each.
        The resulting batched genome has leading dimension equal to the
        population size.
        """
        keys = jax.random.split(key, pop_size)
        return jax.vmap(cls.random_init, in_axes=(0, None))(keys, config)  # type: ignore[no-any-return]


@struct.dataclass
class BasePopulation(Generic[G]):
    """
    A unified container for a collection of candidate solutions.

    This class implements the Struct-of-Arrays (SoA) pattern. The 'genes'
    attribute holds a single `BaseGenome` instance where every internal leaf
    array has an added leading dimension of size N (population size).

    Type System Note:
    As of v2.0, the population parameter `P` has been removed. Population types
    are strictly inferred as `BasePopulation[G]` using the genome type `G`.
    Additionally, the explicit `GENOME_CLS` property has been removed. The
    population is strictly agnostic of its internal structure and relies
    purely on PyTree manipulations.

    Attributes:
        genes: The batched genome data (SoA).
        fitness: A (N,) array representing the objective value for each individual.
        config: Static configuration shared by all individuals in the population.
    """

    genes: G
    fitness: chex.Array
    config: Any = _field(pytree_node=False, default=None)
    info: dict[str, Any] = _field(default_factory=dict)

    def clone_buffers(self: BasePopulation[G]) -> BasePopulation[G]:
        """Deep-copies the population PyTree and all leaf arrays.

        Guarantees buffer isolation for safe JAX buffer donation (`donate_argnums`)
        across multi-seed or multi-pipeline evaluations.
        """
        return cast(
            BasePopulation[G],
            jax.tree_util.tree_map(
                lambda x: jnp.array(x, copy=True) if hasattr(x, "shape") else x,
                self,
            ),
        )

    def copy(self: BasePopulation[G]) -> BasePopulation[G]:
        """Alias for :meth:`clone_buffers` for backwards compatibility."""
        return self.clone_buffers()

    @classmethod
    def from_array(
        cls, arr: chex.Array, config: Any, genome_cls: Type[G], axis: int = 0
    ) -> BasePopulation[G]:
        """Build a population by interpreting one axis of *arr* as individuals.

        The method moves *axis* to the front and delegates to
        ``genome_cls.from_tensor`` for per‑individual wrapping. Output
        fitness values are initialized to ``-inf`` as a sentinel.
        """
        arr_batched = jnp.moveaxis(arr, axis, 0)
        pop_size = arr_batched.shape[0]
        genes = genome_cls.from_tensor(arr_batched, config)
        fitness = jnp.full((pop_size,), -jnp.inf)
        return cls(genes=genes, fitness=fitness, config=config, info={})

    def spawn_offspring(
        self,
        new_genes: G,
        fitness: Optional[chex.Array] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> BasePopulation[G]:
        """Create offspring population, optionally with pre-set fitness.

        Passing ``fitness=None`` triggers allocation of a NaN vector of the
        appropriate length; supplying an array avoids the allocation cost
        when the values are immediately overwritten.
        """
        if fitness is None:
            leaves = jax.tree_util.tree_leaves(new_genes)
            if not leaves:
                raise ValueError("Gene structure contains no arrays.")
            n_offspring = leaves[0].shape[0]
            fitness = jnp.broadcast_to(jnp.nan, (n_offspring,))

        if info is None:
            info = {}

        return replace(self, genes=new_genes, fitness=fitness, info=info)

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

        # Safely slice info dict arrays while ignoring non-arrays (like strings/metadata)
        sliced_info = jax.tree_util.tree_map(
            lambda x: x[key] if hasattr(x, "shape") else x, self.info
        )

        return replace(self, genes=sliced_genes, fitness=self.fitness[key], info=sliced_info)

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
        return replace(self, genes=new_genes)

    def distance_matrix(self, metric: str = DistanceMetric.EUCLIDEAN) -> chex.Array:
        """Compute pairwise distances between all population members.

        The implementation nests two ``jax.vmap`` calls to avoid Python loops
        and remain fully JIT-compatible. Metric is forwarded to each genome's
        ``distance`` method; results form an (N, N) matrix.
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
