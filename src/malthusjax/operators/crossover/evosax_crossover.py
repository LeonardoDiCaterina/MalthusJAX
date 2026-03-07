from typing import Any

import chex
from evosax.algorithms.population_based.simple_ga import crossover as evosax_crossover
from flax import struct

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.base import BaseCrossover, _field


@struct.dataclass
class EvosaxUniformCrossoverWrapper(BaseCrossover[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Evosax Compatibility Wrapper — Single-Key Mode.
    Consumes single key, splits internally in _cross_fused to generate per-pair masks.
    Alternative to standard pre-allocated key budgeting; enables direct evosax integration.
    Design trade-off: Dynamic key splitting vs. static shape stability.

    Use for: Benchmarking evosax compatibility; ablation studies; comparative evolution.
    Shape contract: Parent (d,) × Parent (d,) → Offspring (d,)
    Key budget: 1 key (split dynamically, not pre-allocated)
    """

    num_offspring: int = _field(pytree_node=False, default=1)
    crossover_rate: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

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
        **kwargs: Any,
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
            # keys: (..., atomic_keys) -> flatten to (atomic_keys,)
            prng_key = keys.reshape(-1)[0]
        else:
            # keys: (..., atomic_keys, 2) -> flatten to (atomic_keys, 2)
            prng_key = keys.reshape((-1, keys.shape[-1]))[0]

        # evosax.crossover: (key, p1_values, p2_values, rate) -> (d,) offspring
        child_vals = evosax_crossover(prng_key, p1.values, p2.values, self.crossover_rate)

        return RealGenome.from_tensor(child_vals, config)
