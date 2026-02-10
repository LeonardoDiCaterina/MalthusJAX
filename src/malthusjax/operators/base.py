from abc import abstractmethod
from typing import Any, Generic, Optional, Tuple, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.random import is_new_style_key

# 1. Strict TypeVars
G = TypeVar("G")  # Genome Data
C = TypeVar("C")  # Config Data
P = TypeVar("P", bound=BasePopulation[Any])

_field: Any = struct.field


@struct.dataclass
class BaseMutation(Generic[G, C, P]):
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
    """

    num_offspring: int = _field(pytree_node=False, default=1)
    input_length: int = _field(pytree_node=False, default=-1)
    typed_keys: bool = _field(pytree_node=False, default=False)

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

    def set_input_length(self, length: int) -> "BaseMutation[G, C, P]":
        """Lock population size for static key budgeting."""
        return cast("BaseMutation[G, C, P]", cast(Any, self).replace(input_length=length))

    def set_typed_keys(self, typed: bool) -> "BaseMutation[G, C, P]":
        """
        Set key format based on PRNG impl.
        True = new-style typed keys,
        False = legacy uint32[2].
        """
        return cast("BaseMutation[G, C, P]", cast(Any, self).replace(typed_keys=typed))

    @abstractmethod
    def _mutate_one(self, genome: G, noise_data: Any, config: C, **kwargs: Any) -> G:
        """Tier 1 — Pure mutation arithmetic: genome + noise → mutated genome."""
        raise NotImplementedError

    @abstractmethod
    def _generate_noise(self, keys: chex.PRNGKey, config: C) -> Any:
        """Tier 2 — Noise generation: keys → noise PyTree."""
        raise NotImplementedError

    def _mutate_fused(self, keys: chex.Array, genome: G, config: C, **kwargs: Any) -> G:
        """Fused RNG + arithmetic for single (individual, offspring) pair.

        Combines Tier 1 and Tier 2 into a single traceable unit for XLA
        kernel fusion: RNG (bernoulli, normal, etc.) and arithmetic (add,
        multiply, clip) fused into one kernel without intermediate materialization.

        Input key shape: (num_keys_per_atomic_operation, 2)
        """
        noise = self._generate_noise(keys, config)
        return self._mutate_one(genome, noise, config, **kwargs)

    def __call__(self, all_keys: chex.Array, population: P, config: C, **kwargs: Any) -> P:
        """Tier 3 — Population-level mutation via nested vmap.

        Reshapes pre-allocated keys to (input_length, num_offspring, atomic_keys, 2),
        applies _mutate_fused via two vmaps, then flattens output to (N*K, ...).

        Args:
            all_keys: Pre-allocated keys, shape product = num_keys() result.
            population: Input population with genes shape (N, d, ...).
            config: Genome configuration.

        Returns:
            New population with genes shape (N*K, d, ...) where K=num_offspring.
        """
        # Key reshape is determined by PRNG implementation (set at engine init).
        # typed_keys=True (new-style): keys are 1D array of typed scalars → 3D reshape.
        # typed_keys=False (legacy): keys are (N,2) uint32 arrays → 4D reshape.
        if self.typed_keys:
            keys_reshaped = all_keys.reshape(
                self.input_length, self.num_offspring, self.num_keys_per_atomic_operation
            )
        else:
            keys_reshaped = all_keys.reshape(
                self.input_length, self.num_offspring, self.num_keys_per_atomic_operation, 2
            )

        def _mutate_single(keys_block: chex.Array, genome: G) -> G:
            return self._mutate_fused(keys_block, genome, config, **kwargs)

        def _process_population(k_block: chex.Array, g: G) -> G:
            return jax.vmap(_mutate_single, in_axes=(0, None))(k_block, g)

        vmap_process = jax.vmap(_process_population, in_axes=(0, 0))
        nested_offspring = vmap_process(keys_reshaped, population.genes)

        def flatten_fn(x: chex.Array) -> chex.Array:
            return x.reshape((-1,) + x.shape[2:])

        new_genes = jax.tree_util.tree_map(flatten_fn, nested_offspring)
        return cast(P, population.spawn_offspring(cast(G, new_genes)))


@struct.dataclass
class BaseCrossover(Generic[G, C, P]):
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
    - Axis ordering: output is offspring-major after transpose.
    """

    num_offspring: int = _field(pytree_node=False, default=1)
    input_length: int = _field(pytree_node=False, default=-1)
    typed_keys: bool = _field(pytree_node=False, default=False)

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

    def set_input_length(self, length: int) -> "BaseCrossover[G, C, P]":
        """Lock pair count for static key budgeting."""
        return cast("BaseCrossover[G, C, P]", cast(Any, self).replace(input_length=length))

    def set_typed_keys(self, typed: bool) -> "BaseCrossover[G, C, P]":
        """
        Set key format based on PRNG impl.
        True = new-style typed keys,
        False = legacy uint32[2].
        """
        return cast("BaseCrossover[G, C, P]", cast(Any, self).replace(typed_keys=typed))

    @abstractmethod
    def _generate_noise(self, keys: chex.PRNGKey, config: C) -> Any:
        """Tier 2 — Recombination mask/index generation: keys → noise PyTree."""
        raise NotImplementedError

    @abstractmethod
    def _recombine_one(self, p1: G, p2: G, noise_data: Any, config: C, **kwargs: Any) -> G:
        """Tier 1 — Pure recombination: p1 + p2 + noise → offspring genome.

        Returns single genome G, not tuple. Base class handles replication via num_offspring.
        """
        raise NotImplementedError

    def _cross_fused(self, keys: chex.Array, p1: G, p2: G, config: C, **kwargs: Any) -> G:
        """Fused RNG + recombination for single (pair, offspring) combination.

        Input key shape: (num_keys_per_atomic_operation, 2).
        Combines Tier 1 and 2 for XLA kernel fusion.
        """
        noise = self._generate_noise(keys, config)
        return self._recombine_one(p1, p2, noise, config, **kwargs)

    def cross_single_pair(self, key: chex.Array, p1: G, p2: G, config: C, **kwargs: Any) -> G:
        """Crossover for a single pair (not from population-level __call__).

        Useful for interactive/debug crossover where pair comes from arbitrary
        genomes (not pre-split population). Use __call__ for population-level.

        Args:
            key: Single PRNG key, shape (2,).
            p1, p2: Individual parent genomes.
            config: Genome configuration.

        Returns:
            Batched offspring genome, shape (num_offspring, ...).
        """
        keys = jax.random.split(key, self.num_offspring * self.num_keys_per_atomic_operation)
        # Determine key format: use operator flag if set, otherwise auto-detect from key.
        typed = self.typed_keys or is_new_style_key(key)
        if typed:
            keys_reshaped = keys.reshape(self.num_offspring, self.num_keys_per_atomic_operation)
        else:
            keys_reshaped = keys.reshape(self.num_offspring, self.num_keys_per_atomic_operation, 2)

        def _cross_one_return_values(k: chex.Array) -> chex.Array:
            return cast(Any, self._cross_fused(k, p1, p2, config, **kwargs)).values

        offspring_values = jax.vmap(_cross_one_return_values)(keys_reshaped)
        return cast(G, cast(Any, p1).replace(values=offspring_values))

    def __call__(self, all_keys: chex.Array, p1_pop: P, p2_pop: P, config: C, **kwargs: Any) -> P:
        """Tier 3 — Population-level crossover via nested vmap.

        Reshapes pre-allocated keys to (input_length, num_offspring, atomic_keys, 2),
        applies _cross_fused via two vmaps over pairs and offspring, then transposes
        and flattens output from (pairs, offspring, ...) to (offspring*pairs, ...).

        Args:
            all_keys: Pre-allocated keys, shape product = num_keys() result.
            p1_pop, p2_pop: Parent populations with genes shape (N, d, ...).
            config: Genome configuration.

        Returns:
            Offspring population with genes shape (N*K, d, ...) where K=num_offspring.
            Axis ordering: offspring-major (transpose reorders from pair-major).
        """
        # Key reshape is determined by PRNG implementation (set at engine init).
        if self.typed_keys:
            keys_reshaped = all_keys.reshape(
                self.input_length, self.num_offspring, self.num_keys_per_atomic_operation
            )
        else:
            keys_reshaped = all_keys.reshape(
                self.input_length, self.num_offspring, self.num_keys_per_atomic_operation, 2
            )

        def _process_pairs(k_block: chex.Array, parent1: G, parent2: G) -> Any:
            def _inner_cross(k: chex.Array) -> G:
                return self._cross_fused(k, parent1, parent2, config, **kwargs)

            return jax.vmap(_inner_cross, in_axes=0)(k_block)

        vmap_pairs = jax.vmap(_process_pairs, in_axes=(0, 0, 0))
        nested_offspring = vmap_pairs(keys_reshaped, p1_pop.genes, p2_pop.genes)

        def flatten_fn(x: chex.Array) -> chex.Array:
            transposed = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
            return transposed.reshape((-1,) + transposed.shape[2:])

        new_genes = jax.tree_util.tree_map(flatten_fn, nested_offspring)
        return cast(P, p1_pop.spawn_offspring(cast(G, new_genes)))


@struct.dataclass
class BaseSelection(Generic[P, C]):
    """Stateless selection operator for fitness-based index sampling.

    Selects indices from a population based on fitness values. Optional PRNG key
    enables stochastic selection (tournament, rank-based); deterministic selectors
    (best, truncation) ignore key.

    Shape contracts:
    - Input fitness: (pop_size,) array.
    - Output indices: (num_selections,) integer array.
    """

    num_selections: int = _field(pytree_node=False)
    input_length: int = _field(pytree_node=False, default=-1)
    typed_keys: bool = _field(pytree_node=False, default=False)

    def set_input_length(self, length: int) -> "BaseSelection[P, C]":
        """Lock population size for static budgeting."""
        return cast("BaseSelection[P, C]", cast(Any, self).replace(input_length=length))

    def set_typed_keys(self, typed: bool) -> "BaseSelection[P, C]":
        """
        Set key format based on PRNG impl.
        True = new-style typed keys,
        False = legacy uint32[2].
        """
        return cast("BaseSelection[P, C]", cast(Any, self).replace(typed_keys=typed))

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
        """Select indices from population based on fitness.

        Args:
            keys: PRNG key(s) for stochastic selection; ignored if deterministic.
            fitness: Fitness array, shape (pop_size,).
            config: Optional configuration.
            **kwargs: Operator-specific arguments.

        Returns:
            Selected indices, shape (num_selections,).
        """
        raise NotImplementedError

    def __call__(
        self, keys: chex.Array, population: P, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        """Select from population (accepts either Population or fitness array).

        Args:
            keys: PRNG key(s) for selection.
            population: Population object (with .fitness) or fitness array directly.
            config: Optional configuration.

        Returns:
            Selected indices, shape (num_selections,).
        """
        fitness = getattr(population, "fitness", population)
        return self._select(keys, fitness, config, **kwargs)
