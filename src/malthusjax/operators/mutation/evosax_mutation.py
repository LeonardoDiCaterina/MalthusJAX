from typing import Any

import chex
import jax
import jax.numpy as jnp
from evosax.algorithms.population_based.simple_ga import mutation as evosax_mutation
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.operators.base import BaseMutation, _field
from malthusjax.operators.mutation.real import (
    GaussianMutation_injection as InjectionGaussianMutation,
)

__all__ = ["EvosaxGaussianWrapper", "InjectionGaussianMutation", "BatchedEvosaxGaussianWrapper"]


@struct.dataclass
class EvosaxGaussianWrapper(BaseMutation[RealGenome, RealGenomeConfig]):
    """
    Evosax Gaussian Mutation Wrapper (Integration Adapter).
    Single-key injection-mode integration of evosax.mutation Gaussian operator.
    Key strategy: Accepts single PRNG key, splits per individual (pop_size subkeys),
    applies evosax mutation to each genome independently via jax.vmap.
    Default num_keys_per_atomic_operation=1 (reported budget for static framework).
    Trade-off: Dynamic key splitting (memory efficient) vs static XLA shape stability.
    Purpose: Benchmarking evosax compatibility, ablation baseline for framework
    comparison, demonstrating alternative single-key injection interface pattern.
    """

    mutation_strength: float = 0.1
    injection_mode: bool = _field(pytree_node=False, default=True)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """
        This wrapper consumes 1 atomic key for each fused operation. We
         expose 1 here so that the BaseMutation reshape logic produces a
         non-zero atomic key axis. The single key will be flattened and the
         first 2-int PRNG pair extracted before calling Evosax.
        """
        return 1

    def _mutate_fused(
        self, keys: chex.Array, genome: RealGenome, config: RealGenomeConfig, generation: int = 0
    ) -> RealGenome:
        """
        Atomic fused mutation for a single offspring. Extracts the atomic
        PRNG key, calls Evosax mutation on the genome values and returns a
        JIT-friendly `RealGenome` via `from_tensor`.
        """
        if self.typed_keys:
            prng_key = keys.reshape(-1)[0]
        else:
            prng_key = keys.reshape((-1, keys.shape[-1]))[0]

        mutated_values = evosax_mutation(
            key=prng_key,
            solution=genome.values,
            std=self.mutation_strength,
        )

        return RealGenome.from_tensor(mutated_values, config)

    def num_keys(self, input_shape: tuple[int, ...]) -> int:
        """Return key budget.

        With ``injection_mode=True`` (default), always returns 1 so the
        ResourceMapper allocates a single key that this wrapper splits
        dynamically.  With ``injection_mode=False``, returns the standard
        per-individual budget delegated to the base-class fused path.
        """
        if self.injection_mode:
            return 1
        return int(input_shape[0] * self.num_offspring * self.num_keys_per_atomic_operation)

    def _generate_noise(
        self, keys: chex.Array, config: RealGenomeConfig, generation: int = 0
    ) -> Any:
        """Unused — _mutate_fused overrides the full Tier-1/2 pipeline."""
        raise NotImplementedError("EvosaxGaussianWrapper does not use _generate_noise")

    def _mutate_one(
        self, genome: RealGenome, noise_data: Any, config: RealGenomeConfig, **kwargs: Any
    ) -> RealGenome:
        """Unused — _mutate_fused overrides the full Tier-1/2 pipeline."""
        raise NotImplementedError("EvosaxGaussianWrapper does not use _mutate_one")

    def __call__(
        self,
        all_keys: chex.Array,
        population: BasePopulation[RealGenome],
        config: RealGenomeConfig,
        generation: int = 0,
    ) -> BasePopulation[RealGenome]:
        """Population-level mutation with injection_mode support.

        When ``injection_mode=True``, consumes a single pre-allocated key
        (ResourceMapper slice shape ``(1, 2)`` legacy / ``(1,)`` typed),
        extracts it, and splits into ``N * num_offspring`` subkeys for a
        single flat vmap.  Correct output shape ``(N * num_offspring, D)``
        is always produced regardless of ``num_offspring``.

        When ``injection_mode=False``, delegates to the base-class fused
        path which uses the full pre-allocated key budget.
        """
        if all_keys.size == 0:
            raise ValueError("No PRNG keys provided to EvosaxGaussianWrapper")

        if not self.injection_mode:
            return super().__call__(all_keys, population, config, generation=generation)

        # all_keys may be either a single origin key slice (shape (1,) or
        # (1,2)) which we must split internally, or it may already contain
        # per-individual/per-offspring subkeys provided by the caller.
        if all_keys.shape[0] == 1:
            key = all_keys[0]  # (1,2)→(2,) for legacy; (1,)→scalar for typed
            n = population.genes.values.shape[0]
            total_offspring = n * self.num_offspring
            subkeys = jax.random.split(key, total_offspring)
        else:
            # Caller provided subkeys directly. Flatten to a list of length
            # `n * num_offspring`. Preserve typed/untyped formats.
            n = population.genes.values.shape[0]
            if self.typed_keys:
                subkeys = all_keys.reshape(-1)
            else:
                subkeys = all_keys.reshape(n * self.num_offspring, all_keys.shape[-1])

        def _call_evosax(k: Any, sol: Any) -> Any:
            # evosax library is untyped; keep helper untyped to match
            return evosax_mutation(key=k, solution=sol, std=self.mutation_strength)

        if self.num_offspring == 1:
            mutated_vals = jax.vmap(_call_evosax)(subkeys, population.genes.values)
        else:
            # Tile each genome num_offspring times so vmap covers all (individual, offspring)
            repeated_vals = jnp.repeat(population.genes.values, self.num_offspring, axis=0)
            mutated_vals = jax.vmap(_call_evosax)(subkeys, repeated_vals)

        new_genes = RealGenome(values=mutated_vals)
        return population.spawn_offspring(new_genes)

@struct.dataclass
class BatchedEvosaxGaussianWrapper(EvosaxGaussianWrapper):
    """Batched EvoSAX Gaussian Wrapper (Monolithic Execution).
    
    This operator completely bypasses the JAX vmap structure.
    It expects to be called on a batched RealPopulation, where `genes.values`
    is a `(pop_size, d)` array. It generates a single monolithic noise tensor
    using the underlying `evosax_mutation` function which naturally supports
    any array shape.
    """
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def num_keys(self, input_shape: tuple[int, ...]) -> int:
        return 1

    def __call__(
        self, all_keys: chex.Array, population: Any, config: RealGenomeConfig, generation: int = 0
    ) -> Any:
        # Extract the single monolithic key
        k = all_keys[0] if len(all_keys.shape) == 2 else all_keys[0][0]
        
        genes = population.genes.values
        
        if self.num_offspring > 1:
            genes = jnp.repeat(genes, self.num_offspring, axis=0)
            
        # evosax_mutation automatically scales the noise shape to match the solution shape!
        mutated_vals = evosax_mutation(
            key=k,
            solution=genes,
            std=self.mutation_strength,
        )
        
        return population.spawn_offspring(RealGenome(values=mutated_vals))
