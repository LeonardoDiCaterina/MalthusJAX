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
        Extracts first PRNG key from `keys` (shape (..., atomic_keys=1, 2));
        calls evosax.crossover on value arrays; wraps result into RealGenome.
        Decouples XLA boundary from evosax kernel (operates on pure arrays).

        Returns: Offspring RealGenome with (d,) values
        """
        # Reshape to extract single key: keys is (..., 1, 2) → flatten to (2,)
        prng_key = keys.reshape((-1, keys.shape[-1]))[0]

        # evosax.crossover: (key, p1_values, p2_values, rate) → (d,) offspring
        child_vals = evosax_crossover(prng_key, p1.values, p2.values, self.crossover_rate)

        return RealGenome.from_tensor(child_vals, config)
