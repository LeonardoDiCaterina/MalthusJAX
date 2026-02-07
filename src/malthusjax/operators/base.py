from abc import abstractmethod
from typing import Any, Generic, Optional, Tuple, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation

# 1. Strict TypeVars
G = TypeVar("G")  # Genome Data
C = TypeVar("C")  # Config Data
P = TypeVar("P", bound=BasePopulation[Any])

_field: Any = struct.field


@struct.dataclass
class BaseMutation(Generic[G, C, P]):
    """
    Tier 3 — Vectorized Mutation Wrapper.
    Implements the 3-tier paradigm: Separation of RNG logic from arithmetic kernels.
    """

    num_offspring: int = _field(pytree_node=False, default=1)
    input_length: int = _field(pytree_node=False, default=-1)

    @property
    @abstractmethod
    def num_keys_per_atomic_operation(self) -> int:
        """Requirement for ResourceMapper budgeting."""
        raise NotImplementedError

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """Calculates total key budget for the population and offspring."""
        return input_shape[0] * self.num_offspring * self.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> "BaseMutation[G, C, P]":
        """
        Critical for ResourceMapper: Sets the problem dimension for budget pre-calculation.
        Uses Flax .replace to maintain PyTree immutability.
        """
        return cast("BaseMutation[G, C, P]", cast(Any, self).replace(input_length=length))

    @abstractmethod
    def _mutate_one(self, genome: G, noise_data: Any, config: C, **kwargs: Any) -> G:
        """
        Tier 1 — Arithmetic Kernel (Pure & Promotion-Free).
        Uses masked arithmetic: genome.replace(values=genome.values + noise).
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_noise(self, keys: chex.PRNGKey, config: C) -> Any:
        """
        Tier 2 — Noise Generation (RNG).
        Consumes pre-allocated keys to produce a PyTree of noise data.
        """
        raise NotImplementedError

    def _mutate_fused(self, keys: chex.Array, genome: G, config: C, **kwargs: Any) -> G:
        """
        Fused RNG + Arithmetic pass for optimal XLA kernel fusion.
        Combines Tier 1 & 2 into single traceable unit.
        """
        noise = self._generate_noise(keys, config)
        return self._mutate_one(genome, noise, config, **kwargs)

    def __call__(self, all_keys: chex.Array, population: P, config: C, **kwargs: Any) -> P:
        """
        Tier 3 — Fused Bulk Mutation (Mode E).
        Single-pass fusion: RNG → Arithmetic in one kernel for optimal XLA scheduling
        """
        # 1. Reshape keys based on static ResourceMapper allocation
        # shape: (input_length, offspring, atomic_keys, 2)
        keys_reshaped = all_keys.reshape(
            self.input_length, self.num_offspring, self.num_keys_per_atomic_operation, 2
        )

        # 2. FUSED PASS: Generate noise AND apply mutation in single vmap
        # This allows XLA to fuse: bernoulli → normal → multiply → add → clip
        # into a single kernel without intermediate tensor materialization.
        def _mutate_single(keys_block: chex.Array, genome: G) -> G:
            return self._mutate_fused(keys_block, genome, config, **kwargs)

        def _process_population(k_block: chex.Array, g: G) -> G:
            # Inner vmap over offspring axis
            return jax.vmap(_mutate_single, in_axes=(0, None))(k_block, g)

        # Outer vmap over population
        vmap_process = jax.vmap(_process_population, in_axes=(0, 0))
        nested_offspring = vmap_process(keys_reshaped, population.genes)

        # 3. Flatten and Return
        def flatten_fn(x: chex.Array) -> chex.Array:
            return x.reshape((-1,) + x.shape[2:])

        new_genes = jax.tree_util.tree_map(flatten_fn, nested_offspring)
        return cast(P, population.spawn_offspring(cast(G, new_genes)))


@struct.dataclass
class BaseCrossover(Generic[G, C, P]):
    """
    Tier 3 — Vectorized Crossover Wrapper.
    Implements Fused Mode E: RNG and Recombination logic are fused into a single kernel
    """

    num_offspring: int = _field(pytree_node=False, default=1)
    input_length: int = _field(pytree_node=False, default=-1)

    @property
    @abstractmethod
    def num_keys_per_atomic_operation(self) -> int:
        """Budget requirement for ResourceMapper."""
        raise NotImplementedError

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """
        Calculates total key budget for crossover stage.

        By default keys are budgeted per-pair *and* per-offspring so that
        the base class can vectorize across the offspring axis (two vmaps)
        This mirrors the mutation key budgeting shape: (input_length, num_offspring, atomic_keys, 2)
        """
        num_pairs = input_shape[0]
        return num_pairs * self.num_offspring * self.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> "BaseCrossover[G, C, P]":
        """Locks the number of pairs for static key budgeting."""
        return cast("BaseCrossover[G, C, P]", cast(Any, self).replace(input_length=length))

    @abstractmethod
    def _generate_noise(self, keys: chex.PRNGKey, config: C) -> Any:
        """
        Tier 2 — Recombination Mask/Index Generation.
        Consumes keys to produce crossover points or masks
        """
        raise NotImplementedError

    @abstractmethod
    def _recombine_one(self, p1: G, p2: G, noise_data: Any, config: C, **kwargs: Any) -> G:
        """
        Tier 1 — Recombination Kernel (Pure).
        Deterministic logic: p1, p2 + noise_data -> single offspring genome.

        Note: Returns a single genome G, not a tuple. The base class handles
        offspring repetition via num_offspring parameter.
        """
        raise NotImplementedError

    def _cross_fused(self, keys: chex.Array, p1: G, p2: G, config: C, **kwargs: Any) -> G:
        """
        Fused RNG + Recombination pass for a single offspring.
        """
        noise = self._generate_noise(keys, config)
        return self._recombine_one(p1, p2, noise, config, **kwargs)

    def cross_single_pair(self, key: chex.Array, p1: G, p2: G, config: C, **kwargs: Any) -> G:
        """
        Crossover for a single pair of genomes.

        Args:
            key: Single PRNG key (shape (2,))
            p1, p2: Individual genomes
            config: Genome configuration

        Returns:
            Batched genome with shape (num_offspring, ...)
        """
        # Split key for all offspring
        keys = jax.random.split(key, self.num_offspring * self.num_keys_per_atomic_operation)
        keys_reshaped = keys.reshape(self.num_offspring, self.num_keys_per_atomic_operation, 2)

        # vmap over offspring dimension (typed helper to satisfy mypy)
        def _cross_one_return_values(k: chex.Array) -> chex.Array:
            # _cross_fused returns a Genome object which may have a `.values` attribute
            # cast to Any for attribute access in typed contexts
            return cast(Any, self._cross_fused(k, p1, p2, config, **kwargs)).values

        offspring_values = jax.vmap(_cross_one_return_values)(keys_reshaped)

        # Use cast to Any to call .replace which is a protocol provided by Flax dataclasses
        return cast(G, cast(Any, p1).replace(values=offspring_values))

    def __call__(self, all_keys: chex.Array, p1_pop: P, p2_pop: P, config: C, **kwargs: Any) -> P:
        """
        Tier 3 — Fused Bulk Crossover (Population-level).
        Handles population-level crossover with pre-allocated keys.

        For single-pair crossover, use `cross_single_pair()` instead.
        """
        # 1. Reshape keys based on static input_length and num_offspring
        # shape: (input_length, num_offspring, atomic_keys, 2)
        keys_reshaped = all_keys.reshape(
            self.input_length, self.num_offspring, self.num_keys_per_atomic_operation, 2
        )

        # 2. FUSED PASS: We use two vmaps —
        #    outer: iterate over parent pairs (input_length)
        #    inner: iterate over offspring per pair (num_offspring)
        # The inner vmap calls _cross_fused for each offspring using its dedicated keys slice.
        def _process_pairs(k_block: chex.Array, parent1: G, parent2: G) -> Any:
            # Inner vmap: vectorize over offspring
            def _inner_cross(k: chex.Array) -> G:
                return self._cross_fused(k, parent1, parent2, config, **kwargs)

            return jax.vmap(_inner_cross, in_axes=0)(k_block)

        # Outer vmap: vectorize over pairs
        vmap_pairs = jax.vmap(_process_pairs, in_axes=(0, 0, 0))
        nested_offspring = vmap_pairs(keys_reshaped, p1_pop.genes, p2_pop.genes)

        # 3. Reorder & Flatten to offspring-major ordering:
        #    (input_length, num_offspring, ...) -> (num_offspring, input_length, ...) -> (-1, ...)
        def flatten_fn(x: chex.Array) -> chex.Array:
            transposed = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
            return transposed.reshape((-1,) + transposed.shape[2:])

        new_genes = jax.tree_util.tree_map(flatten_fn, nested_offspring)
        return cast(P, p1_pop.spawn_offspring(cast(G, new_genes)))


@struct.dataclass
class BaseSelection(Generic[P, C]):
    num_selections: int = _field(pytree_node=False)
    input_length: int = _field(pytree_node=False, default=-1)

    def set_input_length(self, length: int) -> "BaseSelection[P, C]":
        return cast("BaseSelection[P, C]", cast(Any, self).replace(input_length=length))

    @property
    @abstractmethod
    def num_keys_per_atomic_operation(self) -> int:
        """Number of PRNG keys required per atomic selection operation."""
        raise NotImplementedError

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        return self.num_keys_per_atomic_operation

    @abstractmethod
    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        """Select indices from the population based on fitness values.

        Args:
            keys: PRNG key(s) for randomization
            fitness: Array of fitness values, shape (pop_size,)
            config: Optional configuration object
            **kwargs: Additional operator-specific arguments

        Returns:
            Array of selected indices, shape (num_selections,)
        """
        raise NotImplementedError

    def __call__(
        self, keys: chex.Array, population: P, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        # Accept either a Population object (with `.fitness`) or a raw fitness array
        fitness = getattr(population, "fitness", population)
        return self._select(keys, fitness, config, **kwargs)
