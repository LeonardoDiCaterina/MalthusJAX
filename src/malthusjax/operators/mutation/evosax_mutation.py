from typing import Any

import chex
import jax.numpy as jnp
from evosax.algorithms.population_based.simple_ga import mutation as evosax_mutation
from flax import struct

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.base import BaseMutation
from malthusjax.operators.mutation.real import (
    GaussianMutation_injection as InjectionGaussianMutation,
)

__all__ = ["EvosaxGaussianWrapper", "InjectionGaussianMutation"]


@struct.dataclass
class EvosaxGaussianWrapper(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Ablation Baseline: Lean Evosax Wrapper.
    """

    mutation_strength: float = 0.1

    @property
    def num_keys_per_atomic_operation(self) -> int:
        # This wrapper consumes 1 atomic key for each fused operation. We
        # expose 1 here so that the BaseMutation reshape logic produces a
        # non-zero atomic key axis. The single key will be flattened and the
        # first 2-int PRNG pair extracted before calling Evosax.
        return 1

    def _mutate_fused(
        self, keys: chex.Array, genome: RealGenome, config: RealGenomeConfig, **kwargs: Any
    ) -> RealGenome:
        """
        Atomic fused mutation for a single offspring. Extracts the atomic
        PRNG key, calls Evosax mutation on the genome values and returns a
        JIT-friendly `RealGenome` via `from_tensor`.
        """
        # keys: (atomic_keys, 2) where atomic_keys == self.num_keys_per_atomic_operation
        prng_key = keys.reshape((-1, keys.shape[-1]))[0]

        mutated_values = evosax_mutation(
            prng_key,
            genome.values,
            jnp.array(self.mutation_strength, dtype=config.dtype),
        )

        return RealGenome.from_tensor(mutated_values, config)

