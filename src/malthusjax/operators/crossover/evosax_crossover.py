from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from evosax.algorithms.population_based.simple_ga import crossover as evosax_crossover
from flax import struct

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.base import BaseCrossover, _field


@struct.dataclass
class EvosaxUniformCrossoverWrapper(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    Evosax Compatibility Wrapper — Single-Key Mode.
    Consumes single key, splits internally in _cross_fused to generate per-pair masks.
    Alternative to standard pre-allocated key budgeting; enables direct evosax integration.
    Design trade-off: Dynamic key splitting vs. static shape stability.

    Use for: Benchmarking evosax compatibility; ablation studies; comparative evolution.
    Shape contract: Parent (d,) X Parent (d,) -> Offspring (d,)
    Key budget: 1 key (split dynamically, not pre-allocated)
    With ``injection_mode=True`` (default), the engine passes a single key and
    this operator splits internally for maximum performance.
    """

    num_offspring: int = _field(pytree_node=False, default=1)
    crossover_rate: float = 0.5
    injection_mode: bool = _field(pytree_node=False, default=True)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def num_keys(self, input_shape: tuple[int, ...]) -> int:
        """Return key budget.
        With ``injection_mode=True``, always returns 1 (single key, split internally).
        """
        if self.injection_mode:
            return 1
        return input_shape[0] * self.num_offspring * self.num_keys_per_atomic_operation

    def _generate_noise(
        self, keys: chex.Array, config: RealGenomeConfig, generation: int = 0
    ) -> Any:
        """Unused — _cross_fused overrides the full Tier-1/2 pipeline."""
        raise NotImplementedError("EvosaxUniformCrossoverWrapper does not use _generate_noise")

    def _recombine_one(  # type: ignore [override]
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: Any,
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> None:
        msg = "EvosaxUniformCrossoverWrapper does not use _recombine_one"
        raise NotImplementedError(msg)

    def _cross_fused(
        self,
        keys: chex.Array,
        p1: RealGenome,
        p2: RealGenome,
        config: RealGenomeConfig,
        generation: int = 0,
    ) -> RealGenome:
        """
        Atomic Crossover Kernel (Single-Key Wrapper Pattern).
        Extracts a PRNG key from `keys` and calls ``evosax.crossover`` on the
        raw value arrays.  Supports both legacy (uint32[2]) keys and new-style
        typed keys.

        ``keys`` shape depends on ``self.typed_keys``:

        * ``typed_keys=False`` (legacy): ``(..., atomic_keys, 2)``
        * ``typed_keys=True`` (new-style):  ``(..., atomic_keys)``

        In either case we flatten away the outer dimensions and select the
        first atomic key.  The resulting ``prng_key`` is either a length-2
        array or a scalar, both of which are valid to pass to JAX random
        primitives and hence to evosax.

        Returns
        -------
        RealGenome
            Offspring genome with shape ``(d,)``.
        """
        if self.typed_keys:
            prng_key = keys.reshape(-1)[0]
        else:
            prng_key = keys.reshape((-1, keys.shape[-1]))[0]

        child_vals = evosax_crossover(prng_key, p1.values, p2.values, self.crossover_rate)

        return RealGenome.from_tensor(child_vals, config)

    def __call__(
        self,
        all_keys: chex.Array,
        p1_pop: RealPopulation,
        p2_pop: RealPopulation,
        config: RealGenomeConfig,
        generation: int = 0,
    ) -> RealPopulation:
        """Population-level crossover with injection_mode support.

        When ``injection_mode=True``, consumes single key and splits internally
        for maximum performance (single vmap instead of nested vmaps).

        Args:
            all_keys: Pre-allocated keys. Shape depends on injection_mode:
                - True: single key (scalar or shape (2,) for legacy)
                - False: standard pre-allocated shape
            p1_pop, p2_pop: Parent populations.
            config: Genome configuration.

        Returns:
            Offspring population.
        """
        if all_keys.size == 0:
            raise ValueError("No PRNG keys provided to EvosaxUniformCrossoverWrapper")

        if not self.injection_mode:
            return super().__call__(all_keys, p1_pop, p2_pop, config, generation=generation)

        num_pairs = p1_pop.genes.values.shape[0]

        # Support two injection-mode calling conventions:
        # 1) Single-origin key (legacy): ResourceMapper gives a single key
        #    slice (shape (1,) or (1,2)); we split it internally.
        # 2) Pre-split keys: caller supplies per-pair (or per-pair-per-offspring)
        #    subkeys. Detect by checking the leading axis length of `all_keys`.
        if all_keys.shape[0] == 1:
            key = all_keys[0]
            keys = jax.random.split(key, num_pairs * self.num_offspring)
        else:
            # Caller provided subkeys directly. Flatten to a 2D list of
            # subkeys with length `num_pairs * num_offspring` for downstream
            # vmap usage. Preserve typed/untyped shapes.
            if self.typed_keys:
                keys = all_keys.reshape(-1)
            else:
                keys = all_keys.reshape(num_pairs * self.num_offspring, all_keys.shape[-1])

        def _cross_one(k: chex.Array, p1_vals: chex.Array, p2_vals: chex.Array) -> chex.Array:
            return evosax_crossover(k, p1_vals, p2_vals, self.crossover_rate)

        if self.num_offspring == 1:
            offspring_vals = jax.vmap(_cross_one)(keys, p1_pop.genes.values, p2_pop.genes.values)
        else:
            if self.typed_keys:
                keys_reshaped = keys.reshape(num_pairs, self.num_offspring)
            else:
                keys_reshaped = keys.reshape(num_pairs, self.num_offspring, keys.shape[-1])
            p1_vals_rep = jnp.repeat(p1_pop.genes.values[:, None, :], self.num_offspring, axis=1)
            p2_vals_rep = jnp.repeat(p2_pop.genes.values[:, None, :], self.num_offspring, axis=1)

            def _cross_pair(k_block: chex.Array, p1: chex.Array, p2: chex.Array) -> chex.Array:
                return jax.vmap(_cross_one)(k_block, p1, p2)

            offspring_vals = jax.vmap(_cross_pair)(keys_reshaped, p1_vals_rep, p2_vals_rep)
            offspring_vals = offspring_vals.reshape(-1, offspring_vals.shape[-1])

        new_genes = RealGenome(values=offspring_vals)
        return cast(RealPopulation, p1_pop.spawn_offspring(new_genes))
