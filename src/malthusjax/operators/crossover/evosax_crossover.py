from typing import Any, Tuple

import chex
from evosax.algorithms.population_based.simple_ga import crossover as evosax_crossover
from flax import struct

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.base import BaseCrossover, _field


@struct.dataclass
class EvosaxUniformCrossoverWrapper(BaseCrossover[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Simple Evosax-style uniform crossover wrapper that consumes a single PRNG
    key and internally splits it to generate per-(pair, offspring) masks.

    This class intentionally exposes `num_keys(...) -> 1` so the operator
    allocation remains minimal and the single key is split internally.
    """

    num_offspring: int = _field(pytree_node=False, default=1)
    crossover_rate: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: Any,
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> Tuple[RealGenome, ...]:
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
        Atomic fused pass for a single offspring. Extracts a single PRNG
        key from `keys`, calls `evosax_crossover` on the two parental value
        arrays, and wraps the returned child array into a `RealGenome` via
        `from_tensor`.
        """
        # keys is expected to be shaped (atomic_keys, 2) where atomic_keys == 1
        prng_key = keys.reshape((-1, keys.shape[-1]))[0]

        # evosax_crossover expects (key, parent1_values, parent2_values, crossover_rate)
        child_vals = evosax_crossover(prng_key, p1.values, p2.values, self.crossover_rate)

        return RealGenome.from_tensor(child_vals, config)
