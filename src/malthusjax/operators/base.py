import dataclasses
import warnings
from abc import abstractmethod
from typing import Any, Generic, Optional, Tuple, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.core.random import is_new_style_key

G = TypeVar("G", bound=BaseGenome)  # Genome Data
C = TypeVar("C")  # Config Data

# Backward compatibility: P is kept as a module-level name so that existing
# code using ``from malthusjax.operators.base import P`` continues to work,
# but it is no longer used in operator class definitions.
P = TypeVar("P", bound=BasePopulation[Any])

_field: Any = struct.field


@struct.dataclass
class BaseMutation(Generic[G, C]):
    """Vectorized mutation operator with pre-allocated key budgeting.

    Tier 3 of the 3-tier architecture: separates RNG (_generate_noise, Tier 2)
    from pure arithmetic (_mutate_one, Tier 1). Fuses both into single vmap
    kernel via _mutate_fused for XLA optimization (RNG + arithmetic in one pass).

    Vmap structure:
    - Outer vmap: Iterate over input_length individuals (population dimension).
    - Inner vmap: Iterate over num_offspring mutants per individual (offspring dimension).
    - Key budgeting: __call__ expects keys shape (input_length, num_offspring, atomic_keys, 2).

    Shape contracts:
    - Input: population.genes with leaf shapes (N, d, ...)
    - Output: population.genes with leaf shapes (N*K, d, ...) where K=num_offspring

    Type system note (v2 array-family refactor):
    The genome parameter `G` represents the structural PyTree type (e.g. `RealGenome`).
    The population parameter `P` has been eliminated; population types are now strictly
    inferred as `BasePopulation[G]`. This forces Tier-3 infrastructure to treat `G` as an
    opaque PyTree without assuming a contiguous `.values` array.

    .. deprecated:: 2.0
        The third type parameter ``P`` (population type) has been removed.
        Use ``BaseMutation[G, C]`` instead of ``BaseMutation[G, C, P]``.
        Old signatures continue to work but emit a deprecation warning.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Detect old-style 3-arg Generic usage in class bases.
        for base in getattr(cls, "__orig_bases__", ()):
            args = getattr(base, "__args__", None)
            if args is not None and len(args) == 3:
                origin = getattr(base, "__origin__", None)
                if origin is BaseMutation:
                    warnings.warn(
                        f"{cls.__qualname__} uses BaseMutation[G, C, P] which is "
                        "deprecated. Use BaseMutation[G, C] instead — the population "
                        "type P is now inferred from G.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    break

    num_offspring: int = _field(pytree_node=False, default=1)
    input_length: int = _field(pytree_node=False, default=-1)
    typed_keys: bool = _field(pytree_node=False, default=False)
    max_generations: int = _field(pytree_node=False, default=1)

    @property
    @abstractmethod
    def num_keys_per_atomic_operation(self) -> int:
        """Keys required per (individual, offspring) pair. Budgeted by ResourceMapper."""
        raise NotImplementedError

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """Total keys needed: input_length * num_offspring * atomic_keys.

        Returns shape for ResourceMapper pre-allocation:
        (input_length, num_offspring, num_keys_per_atomic_operation, [2]).
        """
        return input_shape[0] * self.num_offspring * self.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> "BaseMutation[G, C]":
        """Lock population size for static key budgeting."""
        return dataclasses.replace(self, input_length=length)

    def set_typed_keys(self, typed: bool) -> "BaseMutation[G, C]":
        """
        Set key format based on PRNG impl.
        True = new-style typed keys,
        False = legacy uint32[2].
        """
        return dataclasses.replace(self, typed_keys=typed)

    def set_max_generations(self, n: int) -> "BaseMutation[G, C]":
        """Set total generation count for operator-level scheduling."""
        return dataclasses.replace(self, max_generations=n)

    @abstractmethod
    def _mutate_one(self, genome: G, noise_data: Any, config: C, **kwargs: Any) -> G:
        """Tier 1 — Pure mutation arithmetic: genome + noise → mutated genome.

        This method operates on a single `G` structural instance. It should manipulate
        the PyTree directly (e.g., via `jax.tree_util.tree_map`) rather than assuming
        a rigid `.values` attribute. The Tier-3 `__call__` orchestrates the vmap to
        map this operation across population batches without manual array slicing.
        """
        raise NotImplementedError

    def _apply_noise(self, values: chex.Array, noise_data: Any, config: C) -> chex.Array:
        """Tier 1 (array-native) — Pure mutation arithmetic on raw arrays.

        Array-native variant of ``_mutate_one`` that operates directly on flat
        ``jax.Array`` values, returning a flat ``jax.Array`` without constructing
        a PyTree wrapper.  Used by ``VectorizedEngine`` to avoid tuple overhead
        in the ``lax.scan`` carry.

        Concrete subclasses should override this with their specific arithmetic
        (typically ``values + noise_data`` with optional clipping).  The default
        raises ``NotImplementedError`` to signal that the operator has not yet
        opted into array-native support.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _apply_noise. "
            "Override this method for VectorizedEngine support."
        )

    @abstractmethod
    def _generate_noise(self, keys: chex.PRNGKey, config: C, generation: int = 0) -> Any:
        """Tier 2 — Noise generation: keys → noise PyTree."""
        raise NotImplementedError

    def _mutate_fused(self, keys: chex.Array, genome: G, config: C, generation: int = 0) -> G:
        """Fused RNG + arithmetic for single (individual, offspring) pair.

        Combines Tier 1 and Tier 2 into a single traceable unit for XLA
        kernel fusion: RNG (bernoulli, normal, etc.) and arithmetic (add,
        multiply, clip) fused into one kernel without intermediate materialization.

        Input key shape: (num_keys_per_atomic_operation, 2)
        """
        noise = self._generate_noise(keys, config, generation)
        return self._mutate_one(genome, noise, config)

    def __call__(
        self, all_keys: chex.Array, population: BasePopulation[G], config: C, generation: int = 0
    ) -> BasePopulation[G]:
        """Tier 3 — Population-level mutation using JAX vmaps.

        This method orchestrates either a flat or nested vmap over the provided
        pre‑allocated keys to mutate an entire population. When
        ``num_offspring == 1`` it mirrors evosax's simple structure; when
        >1 it reshapes and flattens the result so that offspring are stacked
        behind their parents.
        """
        n_keys = self.num_keys_per_atomic_operation

        if self.num_offspring == 1:
            """
            Fast path: flat single vmap — same structure as evosax mutation.
            Eliminates inner vmap, (N,1,d)→(N,d) reshape, and tree_map traversal
            """
            if self.typed_keys:
                keys_flat = all_keys.reshape(self.input_length, n_keys)
            else:
                keys_flat = all_keys.reshape(self.input_length, n_keys, 2)

            def _mutate_flat(k: chex.Array, g: G) -> G:
                return self._mutate_fused(k, g, config, generation)

            new_genes = jax.vmap(_mutate_flat, in_axes=(0, 0))(keys_flat, population.genes)
            return population.spawn_offspring(new_genes)

        """
        General path: nested vmap for num_offspring > 1.
        typed_keys=True (new-style): keys are 1D array of typed scalars → 3D reshape.
        typed_keys=False (legacy): keys are (N,2) uint32 arrays → 4D reshape.
        """
        if self.typed_keys:
            keys_reshaped = all_keys.reshape(self.input_length, self.num_offspring, n_keys)
        else:
            keys_reshaped = all_keys.reshape(self.input_length, self.num_offspring, n_keys, 2)

        def _mutate_single(keys_block: chex.Array, genome: G) -> G:
            return self._mutate_fused(keys_block, genome, config, generation)

        def _process_population(k_block: chex.Array, g: G) -> G:
            return jax.vmap(_mutate_single, in_axes=(0, None))(k_block, g)  # type: ignore[no-any-return]

        vmap_process = jax.vmap(_process_population, in_axes=(0, 0))
        nested_offspring = vmap_process(keys_reshaped, population.genes)

        def flatten_fn(x: chex.Array) -> chex.Array:
            """flatten (individuals, offspring, ...d) → (individuals * offspring, ...d)"""
            return x.reshape((-1,) + x.shape[2:])

        new_genes_flat = cast(G, jax.tree_util.tree_map(flatten_fn, nested_offspring))
        return population.spawn_offspring(new_genes_flat)


@struct.dataclass
class BaseCrossover(Generic[G, C]):
    """Vectorized crossover operator with pre-allocated key budgeting.

    Tier 3 of the 3-tier architecture: separates RNG (_generate_noise, Tier 2)
    from pure recombination (_recombine_one, Tier 1). Fuses both via _cross_fused
    for XLA optimization.

    Vmap structure:
    - Outer vmap: Iterate over input_length parent pairs.
    - Inner vmap: Iterate over num_offspring per pair.
    - Key budgeting: __call__ expects keys shape (input_length, num_offspring, atomic_keys, 2).

    Shape contracts:
    - Input: p1_pop, p2_pop with genes leaf shape (N, d, ...).
    - Output: genes shape (N*K, d, ...) where K=num_offspring.
    - Axis ordering: pair-major (direct reshape, no transpose).

    Type system note (v2 array-family refactor):
    The genome parameter `G` represents the structural PyTree type (e.g. `RealGenome`).
    The population parameter `P` has been eliminated; population types are now strictly
    inferred as `BasePopulation[G]`. This forces Tier-3 infrastructure to treat `G` as an
    opaque PyTree without assuming a contiguous `.values` array.

    .. deprecated:: 2.0
        The third type parameter ``P`` (population type) has been removed.
        Use ``BaseCrossover[G, C]`` instead of ``BaseCrossover[G, C, P]``.
        Old signatures continue to work but emit a deprecation warning.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            args = getattr(base, "__args__", None)
            if args is not None and len(args) == 3:
                origin = getattr(base, "__origin__", None)
                if origin is BaseCrossover:
                    warnings.warn(
                        f"{cls.__qualname__} uses BaseCrossover[G, C, P] which is "
                        "deprecated. Use BaseCrossover[G, C] instead — the population "
                        "type P is now inferred from G.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    break

    num_offspring: int = _field(pytree_node=False, default=1)
    input_length: int = _field(pytree_node=False, default=-1)
    typed_keys: bool = _field(pytree_node=False, default=False)
    max_generations: int = _field(pytree_node=False, default=1)

    @property
    @abstractmethod
    def num_keys_per_atomic_operation(self) -> int:
        """Keys required per (pair, offspring) combo. Budgeted by ResourceMapper."""
        raise NotImplementedError

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """Total keys needed: num_pairs * num_offspring * atomic_keys.

        Returns shape for ResourceMapper:
        (input_length, num_offspring, num_keys_per_atomic_operation, [2]).
        """
        num_pairs = input_shape[0]
        return num_pairs * self.num_offspring * self.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> "BaseCrossover[G, C]":
        """Lock pair count for static key budgeting."""
        return dataclasses.replace(self, input_length=length)

    def set_typed_keys(self, typed: bool) -> "BaseCrossover[G, C]":
        """
        Set key format based on PRNG impl.
        True = new-style typed keys,
        False = legacy uint32[2].
        """
        return dataclasses.replace(self, typed_keys=typed)

    def set_max_generations(self, n: int) -> "BaseCrossover[G, C]":
        """Set total generation count for operator-level scheduling."""
        return dataclasses.replace(self, max_generations=n)

    @abstractmethod
    def _generate_noise(self, keys: chex.PRNGKey, config: C, generation: int = 0) -> Any:
        """Tier 2 — Recombination mask/index generation: keys → noise PyTree."""
        raise NotImplementedError

    @abstractmethod
    def _recombine_one(self, p1: G, p2: G, noise_data: Any, config: C, **kwargs: Any) -> G:
        """Tier 1 — Pure recombination: p1 + p2 + noise → offspring genome.

        This method operates on single `G` structural instances representing parents.
        It should manipulate the PyTrees directly rather than assuming a rigid `.values`
        attribute. The Tier-3 `__call__` orchestrates the vmap to map this operation
        across population batches.

        Returns a single genome `G`, not a tuple. The base class handles replication
        via `num_offspring`.
        """
        raise NotImplementedError

    def _apply_mask(
        self, p1_values: chex.Array, p2_values: chex.Array, noise_data: Any, config: C
    ) -> chex.Array:
        """Tier 1 (array-native) — Pure recombination arithmetic on raw arrays.

        Array-native variant of ``_recombine_one`` that operates directly on flat
        ``jax.Array`` parent values, returning a flat ``jax.Array`` without
        constructing a PyTree wrapper.  Used by ``VectorizedEngine`` to avoid
        tuple overhead in the ``lax.scan`` carry.

        Concrete subclasses should override this with their specific arithmetic
        (e.g., ``jnp.where(mask, p2_values, p1_values)``).  The default raises
        ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _apply_mask. "
            "Override this method for VectorizedEngine support."
        )

    def _cross_fused(self, keys: chex.Array, p1: G, p2: G, config: C, generation: int = 0) -> G:
        """Fused RNG + recombination for single (pair, offspring) combination.

        Input key shape: (num_keys_per_atomic_operation, 2).
        Combines Tier 1 and 2 for XLA kernel fusion.
        """
        noise = self._generate_noise(keys, config, generation)
        return self._recombine_one(p1, p2, noise, config)

    def cross_single_pair(self, key: chex.Array, p1: G, p2: G, config: C, generation: int = 0) -> G:
        """Perform crossover on a single parent pair outside of __call__.

        This helper is handy for on‑the‑fly debugging or visualization when
        parents are not drawn from a batched population. It internally splits
        the provided *key* into the required format and then applies the
        fused crossover logic.
        """
        keys = jax.random.split(key, self.num_offspring * self.num_keys_per_atomic_operation)
        typed = self.typed_keys or is_new_style_key(key)
        if typed:
            keys_reshaped = keys.reshape(self.num_offspring, self.num_keys_per_atomic_operation)
        else:
            keys_reshaped = keys.reshape(self.num_offspring, self.num_keys_per_atomic_operation, 2)

        def _cross_one(k: chex.Array) -> G:
            return self._cross_fused(k, p1, p2, config, generation)

        offspring = jax.vmap(_cross_one)(keys_reshaped)
        return cast(G, offspring)

    def __call__(
        self,
        all_keys: chex.Array,
        p1_pop: BasePopulation[G],
        p2_pop: BasePopulation[G],
        config: C,
        generation: int = 0,
    ) -> BasePopulation[G]:
        """Tier 3 — Population-level crossover executed via JAX vmaps.

        Handles both the single‑offspring fast path and the general nested vmap
        case, reshaping keys appropriately and flattening the resulting offspring
        into a single batch.
        for num_offspring == 1, performs a simple vmap over pairs with no reshape.
        for num_offspring > 1, performs a nested vmap and then flattens
        """
        n_keys = self.num_keys_per_atomic_operation

        if self.num_offspring == 1:
            if self.typed_keys:
                keys_flat = all_keys.reshape(self.input_length, n_keys)
            else:
                keys_flat = all_keys.reshape(self.input_length, n_keys, 2)

            def _cross_flat(k: chex.Array, p1: G, p2: G) -> G:
                return self._cross_fused(k, p1, p2, config, generation)

            new_genes = jax.vmap(_cross_flat, in_axes=(0, 0, 0))(
                keys_flat, p1_pop.genes, p2_pop.genes
            )
            return p1_pop.spawn_offspring(new_genes)

        if self.typed_keys:
            keys_reshaped = all_keys.reshape(self.input_length, self.num_offspring, n_keys)
        else:
            keys_reshaped = all_keys.reshape(self.input_length, self.num_offspring, n_keys, 2)

        def _process_pairs(k_block: chex.Array, parent1: G, parent2: G) -> Any:
            def _inner_cross(k: chex.Array) -> G:
                return self._cross_fused(k, parent1, parent2, config, generation)

            return jax.vmap(_inner_cross, in_axes=0)(k_block)

        vmap_pairs = jax.vmap(_process_pairs, in_axes=(0, 0, 0))
        nested_offspring = vmap_pairs(keys_reshaped, p1_pop.genes, p2_pop.genes)

        def flatten_fn(x: chex.Array) -> chex.Array:
            # collapse the pair/offspring axes into one batch dimension,
            # producing shape (pairs*offspring, …).  The ordering of elements
            # is preserved, so downstream stages (mutation, evaluation, etc.)
            # see a simple flat population and XLA won’t incur any copies
            return x.reshape((-1,) + x.shape[2:])

        new_genes_flat = cast(G, jax.tree_util.tree_map(flatten_fn, nested_offspring))
        return p1_pop.spawn_offspring(new_genes_flat)


@struct.dataclass
class BaseSelection(Generic[P, C]):
    """Stateless selection operator for fitness-based index sampling.

    Selects indices from a population based on fitness values. Optional PRNG key
    enables stochastic selection (tournament, rank-based); deterministic selectors
    (best, truncation) ignore key.

    Shape contracts:
    - Input fitness: (pop_size,) array.
    - Output parent indices: (num_selections,) integer array.
    - Output elite indices: (n_elites,) integer array (for preservation).

    The ``n_elites`` field controls how many elite indices the operator
    returns alongside parents.  It is set once during engine init via
    ``set_n_elites()`` and is static (pytree_node=False).
    """

    num_selections: int = _field(pytree_node=False)
    input_length: int = _field(pytree_node=False, default=-1)
    typed_keys: bool = _field(pytree_node=False, default=False)
    n_elites: int = _field(pytree_node=False, default=0)

    def set_input_length(self, length: int) -> "BaseSelection[P, C]":
        """Lock population size for static budgeting."""
        return dataclasses.replace(self, input_length=length)

    def set_typed_keys(self, typed: bool) -> "BaseSelection[P, C]":
        """
        Set key format based on PRNG impl.
        True = new-style typed keys,
        False = legacy uint32[2].
        """
        return dataclasses.replace(self, typed_keys=typed)

    def set_n_elites(self, n: int) -> "BaseSelection[P, C]":
        """Set elite count for preservation (called once at engine init)."""
        return dataclasses.replace(self, n_elites=n)

    @property
    @abstractmethod
    def num_keys_per_atomic_operation(self) -> int:
        """Keys required per selection (0 for deterministic, ≥1 for stochastic)."""
        raise NotImplementedError

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """Total keys needed for one selection pass (typically ≤ atomic_keys)."""
        return self.num_keys_per_atomic_operation

    @abstractmethod
    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        """Select parent indices from a fitness vector.

        Concrete subclasses implement the selection logic, using *keys* when a
        stochastic mechanism is required. The returned array has length
        ``num_selections``.
        """
        raise NotImplementedError

    def get_elite_indices(self, fitness: chex.Array) -> chex.Array:
        """Return indices of the top ``n_elites`` individuals.

        Default implementation: O(N) ``jnp.argpartition``.  Subclasses
        (e.g. ``ElitePoolSelection``) may override ``__call__`` to fuse
        this with parent selection in a single pass.
        """
        if self.n_elites == 0:
            return jnp.zeros(0, dtype=jnp.int32)
        pop_size = fitness.shape[0]
        if self.n_elites >= pop_size:
            return jnp.arange(pop_size, dtype=jnp.int32)
        # Project convention: lower fitness is better (minimization).
        # Select the indices of the smallest `n_elites` values.
        return jnp.argpartition(fitness, self.n_elites)[: self.n_elites]

    def __call__(
        self, keys: chex.Array, population: P, config: Optional[C] = None, **kwargs: Any
    ) -> Tuple[chex.Array, chex.Array]:
        """Run a selection pass returning parents and elites.

        This wrapper extracts fitness from the provided *population* (or uses
        the array directly), then calls ``_select`` followed by
        ``get_elite_indices`` to produce both parent and elite index arrays.
        """
        fitness = getattr(population, "fitness", population)
        parent_idx = self._select(keys, fitness, config, **kwargs)
        elite_idx = self.get_elite_indices(fitness)
        return parent_idx, elite_idx
