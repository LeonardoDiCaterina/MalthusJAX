from abc import abstractmethod
from typing import Any, Generic, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.base import BaseCrossover, BaseMutation, C, G, P, _field


# ==========================================
# 1. MUTATION
# ==========================================
@struct.dataclass
class BaseMutation_injection(BaseMutation[G, C, P]):
    """
    Tier 3 — Vectorized Mutation Wrapper (INJECTION MODE).

    This variant differs from the fused `BaseMutation` in that it consumes a
    single PRNG key and expects `_generate_noise` to return a fully realized
    noise tensor shaped like `(input_length, num_offspring, ...)` (or a matching
    PyTree of arrays). The key design points and trade-offs are:

    - RNG API: callers provide a single key (`all_keys`) which is flattened and
      the first key is used as the seed for `_generate_noise`. The noise
      generator is responsible for splitting that key internally (e.g., via
      `jax.random.split`) to produce any subkeys needed for per-sample noise.

    - Memory / Performance trade-offs: generating the full noise tensor up
      front materializes that tensor in memory which can increase memory
      pressure for large `input_length` or `num_offspring`. However this
      approach makes noise deterministic and easier to record or replay.

    - XLA behavior ("no-code" ops): many of the shape manipulations used in
      the tensor pipeline (e.g., `reshape`, `transpose`) are typically
      implemented as metadata-only transformations in XLA and can be thought of
      as "no-code" or very cheap operations. These ops will not allocate new
      buffers or run expensive kernels unless downstream operations require a
      physical layout change or copy. Conversely, generating large noise arrays
      or performing elementwise arithmetic (add/multiply/bernoulli) will create
      real compute kernels that XLA must schedule and may fuse together.

    - Best practice: implementations of `_generate_noise` should be explicit
      about shapes produced and should internally split the provided key.

    See inline comments for more details on specific reshape/transpose
    locations in the code.
    """

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Injection mode uses 1 key total, split internally."""
        return 0

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """Calculates total key budget for the population and offspring."""
        return 1

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

        Contract:
        - `keys` is a single PRNG key (one array of shape `(2,)` typically).
        - Implementations MUST split this key internally (e.g., `jax.random.split`)
          to obtain subkeys for per-sample randomness.
        - The returned value should be a PyTree of arrays where each array has
          leading axes `(input_length, num_offspring, ...)` so it can be easily
          vmapped over.

        Notes on performance and XLA:
        - Producing the full noise PyTree up front will materialize buffers and
          thus may increase memory pressure. This approach is ideal when you
          want deterministic, recordable noise or correlated noise across
          individuals, but may be less memory-efficient than per-atomic fused
          RNG calls.
        """
        raise NotImplementedError

    def _mutate_fused(self, keys: chex.Array, genome: G, config: C, **kwargs: Any) -> G:
        msg = "Not implemented for injection mutation as the noise is generated externally"
        raise Exception(msg)

    def __call__(self, all_keys: chex.Array, population: P, config: C, **kwargs: Any) -> P:
        """
        Tier 3 — Fused Bulk Mutation (Mode E).
        Single-pass fusion: RNG → Arithmetic in one kernel for optimal XLA scheduling.
        """
        # Expect `all_keys` to be an array of PRNG keys with trailing dimension
        # equal to the key width (commonly 2). In injection mode we only need a
        # single key: the `_generate_noise` method is responsible for splitting
        # it further as required. We flatten the key array here and sanity-check
        # that at least one key is present to surface a clear error otherwise.
        flat_keys = all_keys.reshape((-1, all_keys.shape[-1]))
        if flat_keys.shape[0] == 0:
            raise ValueError("No RNG keys provided to BaseMutation_injection")
        single_key = flat_keys[0]

        # returns (input_length, num_offspring, ...)
        noise = self._generate_noise(single_key, config)

        def reshape_noise(x: chex.Array) -> chex.Array:
            return x.reshape((self.input_length, self.num_offspring) + x.shape[1:])

        noise = jax.tree_util.tree_map(reshape_noise, noise)

        # For each (input index) we have a block of per-offspring noise.
        # We need to call the atomic `_mutate_one(genome, noise, config)` for
        # each noise vector. Use an inner vmap that calls `_mutate_one` with
        # the correct argument order (genome first, then noise) so the
        # implementation remains consistent with the abstract contract.
        def _mutate_one_inner(n: chex.Array, g: G) -> G:
            return self._mutate_one(g, n, config)

        def _process_noise_block(noise_block: chex.Array, g: G) -> G:
            # Inner vmap over offspring axis; define a typed inner helper to avoid
            # passing an untyped lambda to jax.vmap
            def _inner(n: chex.Array) -> G:
                return _mutate_one_inner(n, g)

            return jax.vmap(_inner, in_axes=0)(noise_block)

        # Outer vmap over input indices
        nested_offspring = jax.vmap(_process_noise_block, in_axes=(0, 0))(noise, population.genes)

        # Merge the first two axes (input_length * num_offspring) into a single
        # batch axis. `reshape` is commonly a metadata-only transform in XLA
        # and will not allocate a new buffer unless required by later ops.
        def flatten_fn(x: chex.Array) -> chex.Array:
            return x.reshape((-1,) + x.shape[2:])

        new_genes = jax.tree_util.tree_map(flatten_fn, nested_offspring)
        # If `new_genes` is a raw array (no `.values`), wrap it back into the
        # population's genome class so downstream code has a consistent
        # `Population.genes` object with `.values` attribute.
        if not hasattr(new_genes, "values"):
            new_genes = population.GENOME_CLS.from_tensor(new_genes, population.config)
        return cast(P, population.spawn_offspring(cast(G, new_genes)))


# ==========================================
# 2. CROSSOVER
# ==========================================
@struct.dataclass
class BaseCrossover_injection(Generic[G, C, P]):
    """
    Tier 3 — Vectorized Crossover Wrapper (INJECTION MODE).

    This crossover variant mirrors the mutation injection approach: it consumes a
    single PRNG key and expects `_generate_noise` to produce masks/indices/etc
    with leading axes `(input_length, num_offspring, ...)` that the vmaps will
    iterate over.

    Key points:
    - The single provided key should be split internally by `_generate_noise`.
    - Reshaping/transpose operations used to reorder offspring vs pair axes are
      typically metadata-only ("no-code") for XLA. This makes them cheap in
      terms of compute but they may trigger copies if layout changes are
      required.
    - As with mutation, producing full noise tensors up front improves
      reproducibility and supports correlated noise patterns but increases
      memory usage.
    """

    num_offspring: int = _field(pytree_node=False, default=1)
    input_length: int = _field(pytree_node=False, default=-1)

    @property
    @abstractmethod
    def num_keys_per_atomic_operation(self) -> int:
        """Budget requirement for ResourceMapper."""
        return 0

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """
        Calculates total key budget for crossover stage.

        By default keys are budgeted per-pair *and* per-offspring so that
        the base class can vectorize across the offspring axis (two vmaps).
        This mirrors the mutation key budgeting shape:
        (input_length, num_offspring, atomic_keys, 2).
        """
        return 1

    def set_input_length(self, length: int) -> "BaseCrossover[G, C, P]":
        """Locks the number of pairs for static key budgeting."""
        return cast("BaseCrossover[G, C, P]", cast(Any, self).replace(input_length=length))

    @abstractmethod
    def _generate_noise(self, keys: chex.PRNGKey, config: C) -> Any:
        """
        Tier 2 — Recombination Mask/Index Generation.
        Consumes a single key to produce crossover points or masks.
        Make sure that the noise data shape is (input_length, num_offspring, ...).
        """
        raise NotImplementedError

    @abstractmethod
    def _recombine_one(self, p1: G, p2: G, noise_data: Any, config: C, **kwargs: Any) -> G:
        """
        Tier 1 — Recombination Kernel (Pure).
        Deterministic logic: p1, p2 + noise_data -> offspring_tuple.
        """
        raise NotImplementedError

    def _cross_fused(self, keys: chex.Array, p1: G, p2: G, config: C, **kwargs: Any) -> G:
        msg = "Not implemented for injection crossover as the noise is generated externally"
        raise Exception(msg)

    def __call__(self, all_keys: chex.Array, p1_pop: P, p2_pop: P, config: C, **kwargs: Any) -> P:
        """
        Tier 3 — Fused Bulk Crossover.
        Correctly flattens and concatenates multiple offspring into a single population.
        """
        # Flatten incoming keys and grab the first key. `_generate_noise` is
        # expected to split the key internally to produce per-pair/per-offspring
        # randomness if needed.
        flat_keys = all_keys.reshape((-1, all_keys.shape[-1]))
        if flat_keys.shape[0] == 0:
            raise ValueError("No RNG keys provided to BaseCrossover_injection")
        single_key = flat_keys[0]
        # returns (input_length, num_offspring, ...)
        noise = self._generate_noise(single_key, config)

        def reshape_noise(x: chex.Array) -> chex.Array:
            return x.reshape((self.input_length, self.num_offspring) + x.shape[1:])

        noise = jax.tree_util.tree_map(reshape_noise, noise)

        # Invoke `_recombine_one(p1, p2, noise, config)` for each per-pair
        # per-offspring noise item. The inner vmap iterates over the offspring
        # axis of the noise block and calls the pure recombination kernel.
        def _per_pair_block(noise_block: chex.Array, p1: G, p2: G) -> Any:
            def _per_offspring(n: chex.Array) -> G:
                out = self._recombine_one(p1, p2, n, config, **kwargs)
                # Backward compatibility: if operator returned a tuple, pick the first
                # element (most operators return a single offspring as a scalar
                # genome). This keeps the downstream shapes consistent.
                if isinstance(out, tuple):
                    return cast(G, out[0])
                return out

            # vmap over offspring axis
            return jax.vmap(_per_offspring, in_axes=0)(noise_block)

        nested_offspring = jax.vmap(_per_pair_block, in_axes=(0, 0, 0))(
            noise, p1_pop.genes, p2_pop.genes
        )

        # Reorder axes from (input_length, num_offspring, ...) to
        # (num_offspring, input_length, ...) and then flatten. The `transpose`
        # operation is often metadata-only in XLA (layout transform) and
        # `reshape` is usually a no-copy metadata op too. However, if later
        # kernels require a different memory layout this may cause a physical
        # copy and should be considered when benchmarking large populations.
        def merge_and_flatten_block(x: chex.Array) -> chex.Array:
            transposed = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
            return transposed.reshape((-1,) + transposed.shape[2:])

        new_genes = jax.tree_util.tree_map(merge_and_flatten_block, nested_offspring)

        return cast(P, p1_pop.spawn_offspring(cast(G, new_genes)))


__all__ = [
    "BaseMutation_injection",
    "BaseCrossover_injection",
]
