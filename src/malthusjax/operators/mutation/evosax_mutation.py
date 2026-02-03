from typing import Any

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
            key=prng_key,
            solution=genome.values,
            std=self.mutation_strength,
        )

        return RealGenome.from_tensor(mutated_values, config)

    def num_keys(self, input_shape: tuple[int, ...]) -> int:
        """Return key budget.

        - If `input_length` has been set (via `set_input_length`), behave like a
          standard fused operator and return per-pair budgeting.
        - Otherwise the wrapper uses a single global key (injection-style) and
          reports a budget of 1.
        """
        if self.input_length > 0:
            # behave like BaseMutation.num_keys when input_length is explicitly set
            return int(self.input_length * self.num_offspring * self.num_keys_per_atomic_operation)
        return 1

    def __call__(
        self, all_keys: chex.Array, population: RealPopulation, config: RealGenomeConfig, **kwargs: Any
    ) -> RealPopulation:
        """
        Injection-style call: accepts a single PRNG key and applies Evosax mutation
        per individual by splitting the single key into `pop_size` subkeys. If more
        than one key is provided, fall back to the fused BaseMutation implementation.
        """
        if all_keys.size == 0:
            raise ValueError("No PRNG keys provided to EvosaxGaussianWrapper")

        # Single-key optimization: split into per-individual subkeys
        if all_keys.shape[0] == 1:
            key = all_keys[0]
            n = len(population)
            subkeys = jax.random.split(key, n)

            # Vectorized Evosax mutation over all individuals
            mutated_vals = jax.vmap(
                lambda k, sol: evosax_mutation(key=k, solution=sol, std=self.mutation_strength)
            )(subkeys, population.genes.values)

            new_genes = RealGenome(values=mutated_vals)
            return population.replace(genes=new_genes, fitness=jnp.full((n,), jnp.nan))

        # Otherwise, use default fused behavior from BaseMutation
        return super().__call__(all_keys, population, config, **kwargs)

