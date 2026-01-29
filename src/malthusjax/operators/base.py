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


# ==========================================
# 1. MUTATION
# ==========================================
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

        nested_offspring = jax.vmap(
            lambda k, g: jax.vmap(_mutate_single, in_axes=(0, None))(k, g), in_axes=(0, 0)
        )(keys_reshaped, population.genes)

        # 3. Flatten and Return
        def flatten_fn(x: chex.Array) -> chex.Array:
            return x.reshape((-1,) + x.shape[2:])

        new_genes = jax.tree_util.tree_map(flatten_fn, nested_offspring)
        return cast(P, population.spawn_offspring(cast(G, new_genes)))


# ==========================================
# 2. CROSSOVER
# ==========================================
@struct.dataclass
class BaseCrossover(Generic[G, C, P]):
    """
    Tier 3 — Vectorized Crossover Wrapper.
    Implements Fused Mode E: RNG and Recombination logic are fused into a single kernel
    """

    num_offspring: int = _field(pytree_node=False, default=2)
    input_length: int = _field(pytree_node=False, default=-1)

    # ==========================================
    # RESOURCE MAPPER INTERFACE
    # ==========================================

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

    # ==========================================
    # THE THREE TIERS
    # ==========================================

    @abstractmethod
    def _generate_noise(self, keys: chex.PRNGKey, config: C) -> Any:
        """
        Tier 2 — Recombination Mask/Index Generation.
        Consumes keys to produce crossover points or masks
        """
        raise NotImplementedError

    @abstractmethod
    def _recombine_one(
        self, p1: G, p2: G, noise_data: Any, config: C, **kwargs: Any
    ) -> Tuple[G, ...]:
        """
        Tier 1 — Recombination Kernel (Pure).
        Deterministic logic: p1, p2 + noise_data -> offspring_tuple.
        """
        raise NotImplementedError

    def _cross_fused(self, keys: chex.Array, p1: G, p2: G, config: C, **kwargs: Any) -> G:
        """
        Fused RNG + Recombination pass for a single offspring.

        The BaseCrossover now drives the `num_offspring` repetition. This helper
        consumes the atomic keys for a single offspring and returns a single genome.
        Operators that previously returned a tuple of offspring should be adapted
        to return a single offspring; the repetition is handled at the base level.
        """
        noise = self._generate_noise(keys, config)
        out = self._recombine_one(p1, p2, noise, config, **kwargs)

        # Backward compatibility: if operator returned a tuple, pick the first element
        # and return it. We prefer operators to return a single genome when possible.
        if isinstance(out, tuple):
            return out[0]
        return out

    def __call__(self, all_keys: chex.Array, p1_pop: P, p2_pop: P, config: C, **kwargs: Any) -> P:
        """
        Tier 3 — Fused Bulk Crossover.
        Correctly flattens and concatenates multiple offspring into a single population.
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
        nested_offspring = jax.vmap(
            lambda k_block, g1, g2: jax.vmap(
                lambda k, g1_inner, g2_inner: self._cross_fused(
                    k, g1_inner, g2_inner, config, **kwargs
                ),
                in_axes=(0, None, None),
            )(k_block, g1, g2),
            in_axes=(0, 0, 0),
        )(keys_reshaped, p1_pop.genes, p2_pop.genes)

        # 3. Collapse & Flatten
        # nested_offspring shape currently: (input_length, num_offspring, ...)
        # We need to reorder to (num_offspring, input_length, ...)
        # then flatten to (num_offspring * input_length, ...)
        def merge_and_flatten_block(x: chex.Array) -> chex.Array:
            # x: (input_length, num_offspring, ...)
            transposed = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
            return transposed.reshape((-1,) + transposed.shape[2:])

        new_genes = jax.tree_util.tree_map(merge_and_flatten_block, nested_offspring)

        return cast(P, p1_pop.spawn_offspring(cast(G, new_genes)))


# ==========================================
# 3. SELECTION
# ==========================================
@struct.dataclass
class BaseSelection(Generic[P, C]):
    num_selections: int = _field(pytree_node=False)
    input_length: int = _field(pytree_node=False, default=-1)

    def set_input_length(self, length: int) -> "BaseSelection[P, C]":
        return cast("BaseSelection[P, C]", cast(Any, self).replace(input_length=length))

    @property
    def num_keys_per_atomic_operation(self) -> int:
        raise NotImplementedError

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        return self.num_keys_per_atomic_operation

    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        raise NotImplementedError

    def __call__(
        self, keys: chex.Array, population: P, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        # Accept either a Population object (with `.fitness`) or a raw fitness array
        if hasattr(population, "fitness"):
            fitness = population.fitness
        else:
            fitness = population
        return self._select(keys, fitness, config, **kwargs)
