from typing import Any, Tuple, cast

import chex
import jax
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

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        # Evosax wrapper requests a single global key for the operation
        return 1

    def _mutate_fused(
        self, keys: chex.Array, genome: RealGenome, config: RealGenomeConfig, **kwargs: Any
    ) -> RealGenome:
        """
        Keys will be a small array shaped like (atomic_keys, 2) — take the
        first PRNGKey and forward it to the Evosax mutation function. Wrap
        the returned array into a RealGenome instance to satisfy the base
        contract.
        """
        # Ensure we extract a proper PRNG key of shape (2,)
        prng_key = keys.reshape((-1, keys.shape[-1]))[0]

        # The Evosax mutation accepts positional arguments: (key, x, std)
        mutated_values = evosax_mutation(
            prng_key,
            genome.values,
            jnp.array(self.mutation_strength, dtype=config.dtype),
        )

        return cast(RealGenome, cast(Any, genome).replace(values=mutated_values))

    def __call__(
        self,
        all_keys: chex.Array,
        population: RealPopulation,
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealPopulation:
        """
        Override the bulk caller for Evosax wrapper to support a single global
        RNG key which will be applied to each genome. This avoids forcing the
        ResourceMapper to allocate per-sample keys when the underlying Evosax
        mutation expects a single key and mutates each sample deterministically
        from that key.
        """
        # Extract flat key
        flat = all_keys.reshape((-1, all_keys.shape[-1]))
        if flat.shape[0] == 0:
            raise ValueError("No RNG key provided to EvosaxGaussianWrapper")
        key = flat[0]

        # Vectorize the fused mutation across genomes using the single key.
        def mutate_one(genome: RealGenome) -> jnp.ndarray:
            return cast(
                jnp.ndarray, self._mutate_fused(keys=key, genome=genome, config=config).values
            )

        mutated_vals = jax.vmap(mutate_one)(population.genes)

        new_genes = cast(Any, population.genes).replace(values=mutated_vals)
        return cast(RealPopulation, population.spawn_offspring(new_genes))
