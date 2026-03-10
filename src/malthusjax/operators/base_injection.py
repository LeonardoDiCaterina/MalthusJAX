from abc import abstractmethod
from typing import Any, Generic, Tuple, cast

import chex
import jax
from flax import struct

from malthusjax.operators.base import BaseMutation, C, G, P, _field


@struct.dataclass
class BaseMutation_injection(BaseMutation[G, C, P]):
    """Vectorized mutation with external noise injection (single-key mode).

    Differs from BaseMutation: consumes single PRNG key, expects _generate_noise
    to return fully materialized noise tensor shaped (input_length, num_offspring, ...).
    This trades off memory for explicit noise control and determinism.

    Design trade-offs:
    - RNG: Single key splits internally in _generate_noise (user responsibility).
    - Memory: Full noise materialization increases buffer size but enables replay.
    - XLA: reshape/transpose are metadata-only unless downstream requires copy.

    Architecture: Tier 1 (_mutate_one, pure) → Tier 2 (_generate_noise, RNG) →
    Tier 3 (__call__, vmap nesting with single-pass arithmetic).
    """

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Injection mode: single key budgeted, split internally."""
        return 0

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """Returns 1 key; _generate_noise responsible for internal splitting."""
        return 1

    @abstractmethod
    def _mutate_one(self, genome: G, noise_data: Any, config: C, **kwargs: Any) -> G:
        """Tier 1 — Pure mutation kernel: genome + noise → mutated genome."""
        raise NotImplementedError

    @abstractmethod
    def _generate_noise(self, keys: chex.PRNGKey, config: C, generation: int = 0) -> Any:
        """Tier 2 — Noise generation from single PRNG key.

        Contract:
        - Input: Single PRNG key, shape (2,).
        - Output: PyTree of arrays with leading shape (input_length, num_offspring, ...).
        - Responsibility: Caller (implementation) must split key internally to
          produce per-sample randomness via jax.random.split or equivalent.

        Trade-offs:
        - Full materialization: Allocates buffers for entire noise array (memory cost).
        - Determinism: Exact same noise produced for identical key (replay capability).
        - XLA: Noise generation happens outside vmap; large tensors may not fuse
          with arithmetic. Use when noise complexity justifies explicit control.
        """
        raise NotImplementedError

    def _mutate_fused(self, keys: chex.Array, genome: G, config: C, **kwargs: Any) -> G:
        """Tier 2 implementation unsupported in injection mode; use _generate_noise."""
        raise NotImplementedError("Injection mode: override _generate_noise instead")

    def __call__(self, all_keys: chex.Array, population: P, config: C, generation: int = 0) -> P:
        """Tier 3 — Vectorized bulk mutation via vmap.

        Input: Single key (flattened to shape (2,)).
        Output: Population with shape (input_length * num_offspring, ...)

        For num_offspring=1 uses a single flat vmap (no reshape, no tree_map).
        For num_offspring>1 uses nested vmap with a pair-major flatten at the end.
        """
        flat_keys = all_keys.reshape((-1, all_keys.shape[-1]))
        if flat_keys.shape[0] == 0:
            raise ValueError("No RNG keys provided to BaseMutation_injection")
        single_key = flat_keys[0]

        noise = self._generate_noise(single_key, config, generation)  # leading dim: (N*K, ...)

        if self.num_offspring == 1:
            # Fast path: noise is already (N, ...) — flat vmap, no reshape needed.
            def _mutate_flat(n: chex.Array, g: G) -> G:
                return self._mutate_one(g, n, config)

            new_genes = jax.vmap(_mutate_flat, in_axes=(0, 0))(noise, population.genes)
            return cast(P, population.spawn_offspring(cast(G, new_genes)))

        # General path: reshape flat noise to (N, K, ...) then nested vmap.
        def reshape_noise(x: chex.Array) -> chex.Array:
            return x.reshape((self.input_length, self.num_offspring) + x.shape[1:])

        noise_nk = jax.tree_util.tree_map(reshape_noise, noise)

        def _process_noise_block(noise_block: chex.Array, g: G) -> G:
            def _inner(n: chex.Array) -> G:
                return self._mutate_one(g, n, config)

            return jax.vmap(_inner, in_axes=0)(noise_block)

        nested_offspring = jax.vmap(_process_noise_block, in_axes=(0, 0))(
            noise_nk, population.genes
        )

        def flatten_fn(x: chex.Array) -> chex.Array:
            return x.reshape((-1,) + x.shape[2:])

        new_genes = jax.tree_util.tree_map(flatten_fn, nested_offspring)
        return cast(P, population.spawn_offspring(cast(G, new_genes)))


@struct.dataclass
class BaseCrossover_injection(Generic[G, C, P]):
    """Vectorized crossover with external noise injection (single-key mode).

    Mirrors mutation injection: consumes single key, expects _generate_noise
    to return masks/indices with shape (input_length, num_offspring, ...).
    Trades memory for explicit control and determinism.

    Design trade-offs:
    - RNG: Single key split internally in _generate_noise.
    - Memory: Full mask materialization for reproducible recombination.
    - XLA: transpose (axis swap) typically metadata-only if downstream
      layout matches; may trigger copy if not. reshape usually no-cost.

    Architecture: Tier 1 (_recombine_one, pure) → Tier 2 (_generate_noise, RNG) →
    Tier 3 (__call__, triple nested vmap: pairs, offspring, elements).
    """

    num_offspring: int = _field(pytree_node=False, default=1)
    input_length: int = _field(pytree_node=False, default=-1)
    typed_keys: bool = _field(pytree_node=False, default=False)
    max_generations: int = _field(pytree_node=False, default=1)

    @property
    @abstractmethod
    def num_keys_per_atomic_operation(self) -> int:
        """Budget requirement (injection uses 1 key total)."""
        return 0

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """Returns 1 key; _generate_noise handles internal splitting."""
        return 1

    def set_typed_keys(self, typed: bool) -> "BaseCrossover_injection[G, C, P]":
        """Set the PRNG key format flag (new-style typed vs legacy uint32)."""
        return cast("BaseCrossover_injection[G, C, P]", cast(Any, self).replace(typed_keys=typed))

    def set_input_length(self, length: int) -> "BaseCrossover_injection[G, C, P]":
        """Locks pair count for static budgeting."""
        return cast(
            "BaseCrossover_injection[G, C, P]", cast(Any, self).replace(input_length=length)
        )

    def set_max_generations(self, n: int) -> "BaseCrossover_injection[G, C, P]":
        """Set total generation count for operator-level scheduling."""
        return cast("BaseCrossover_injection[G, C, P]", cast(Any, self).replace(max_generations=n))

    @abstractmethod
    def _generate_noise(self, keys: chex.PRNGKey, config: C, generation: int = 0) -> Any:
        """Tier 2 — Mask/index generation from single PRNG key.

        Contract:
        - Input: Single PRNG key, shape (2,).
        - Output: PyTree with leading shape (input_length, num_offspring, ...).
        - Responsibility: Split key internally to produce per-pair/per-offspring masks.

        Example: For uniform crossover, _generate_noise(key, config) might return
        Bernoulli mask array shaped (input_length, num_offspring, genome_length).
        """
        raise NotImplementedError

    @abstractmethod
    def _recombine_one(self, p1: G, p2: G, noise_data: Any, config: C, **kwargs: Any) -> G:
        """Tier 1 — Pure recombination kernel: p1 + p2 + noise → offspring."""
        raise NotImplementedError

    def _cross_fused(self, keys: chex.Array, p1: G, p2: G, config: C, **kwargs: Any) -> G:
        """Tier 2 unsupported; override _generate_noise instead."""
        raise NotImplementedError("Injection mode: override _generate_noise instead")

    def __call__(
        self, all_keys: chex.Array, p1_pop: P, p2_pop: P, config: C, generation: int = 0
    ) -> P:
        """Tier 3 — Vectorized bulk crossover via vmap.

        Input: Single key (flattened to shape (2,)).
        Output: Population of offspring shape (input_length * num_offspring, ...)

        For num_offspring=1 uses a single flat vmap (no reshape, no tree_map).
        For num_offspring>1 uses nested vmap with a pair-major flatten at the end
        — no transpose needed (FB-1).

        Note: _recombine_one must return a single genome G, not a tuple.
        """
        flat_keys = all_keys.reshape((-1, all_keys.shape[-1]))
        if flat_keys.shape[0] == 0:
            raise ValueError("No RNG keys provided to BaseCrossover_injection")
        single_key = flat_keys[0]

        noise = self._generate_noise(single_key, config, generation)

        if self.num_offspring == 1:

            def _cross_flat(n: chex.Array, p1: G, p2: G) -> G:
                return self._recombine_one(p1, p2, n, config)

            new_genes = jax.vmap(_cross_flat, in_axes=(0, 0, 0))(noise, p1_pop.genes, p2_pop.genes)
            return cast(P, p1_pop.spawn_offspring(cast(G, new_genes)))

        def reshape_noise(x: chex.Array) -> chex.Array:
            return x.reshape((self.input_length, self.num_offspring) + x.shape[1:])

        noise_nk = jax.tree_util.tree_map(reshape_noise, noise)

        def _per_pair_block(noise_block: chex.Array, p1: G, p2: G) -> Any:
            def _per_offspring(n: chex.Array) -> G:
                return self._recombine_one(p1, p2, n, config)

            return jax.vmap(_per_offspring, in_axes=0)(noise_block)

        nested_offspring = jax.vmap(_per_pair_block, in_axes=(0, 0, 0))(
            noise_nk, p1_pop.genes, p2_pop.genes
        )

        def flatten_fn(x: chex.Array) -> chex.Array:
            return x.reshape((-1,) + x.shape[2:])

        new_genes = jax.tree_util.tree_map(flatten_fn, nested_offspring)
        return cast(P, p1_pop.spawn_offspring(cast(G, new_genes)))


__all__ = [
    "BaseMutation_injection",
    "BaseCrossover_injection",
]
