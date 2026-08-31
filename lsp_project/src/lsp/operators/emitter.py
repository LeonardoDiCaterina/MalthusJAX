"""LSP MO Emitter — bridges LSP crossover + mutation into MOEngine's BaseEmitter.

Implements a simple genetic emitter (crossover then mutation) compatible with
MOEngine's ask/tell interface.  Uses HomologousPrefixCrossover and
AnnealedTopologicalMutation to produce offspring from the current Pareto population.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from lsp.genome.linear import PrefixGenomeConfig
from lsp.operators.crossover import HomologousPrefixCrossover
from lsp.operators.mutation import AnnealedTopologicalMutation

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.linear_genome import LinearGenome
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState


class LSPMOEmitter(BaseEmitter):
    """Genetic emitter for LSP LinearGenome / MOEngine.

    Produces ``_batch_size`` offspring per generation by:
        1. Randomly sampling parent pairs from the current population.
        2. Applying ``HomologousPrefixCrossover`` on ``variation_percentage``
           of the batch.
        3. Applying ``AnnealedTopologicalMutation`` on ALL offspring.

    Args:
        mutation:            AnnealedTopologicalMutation instance.
        crossover:           HomologousPrefixCrossover instance.
        variation_percentage: Fraction of offspring created via crossover
                              (remainder use parent directly, then mutate).
        genome_config:       PrefixGenomeConfig for the LinearGenome.
        _batch_size:         Number of offspring to emit per step.
    """

    def __init__(
        self,
        mutation: AnnealedTopologicalMutation,
        crossover: HomologousPrefixCrossover,
        variation_percentage: float,
        genome_config: PrefixGenomeConfig,
        _batch_size: int,
    ) -> None:
        self.mutation = mutation
        self.crossover = crossover
        self.variation_percentage = variation_percentage
        self.genome_config = genome_config
        self._batch_size = _batch_size

    # -----------------------------------------------------------------------
    # BaseEmitter protocol
    # -----------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        # 2 keys per offspring: 1 for crossover mask + keys for mutation
        return 2 + self.mutation.num_keys_per_atomic_operation

    def num_keys(self) -> int:
        # 2 for parent sampling + per-offspring keys
        return 2 + self.batch_size * self.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> "LSPMOEmitter":
        return LSPMOEmitter(
            mutation=self.mutation,
            crossover=self.crossover,
            variation_percentage=self.variation_percentage,
            genome_config=self.genome_config,
            _batch_size=length,
        )

    def init(
        self,
        key: chex.Array,
        initial_population: BasePopulation[Any],
        params: Any = None,
    ) -> Optional[EmitterState]:
        return None  # Stateless emitter

    def ask(
        self,
        state: Optional[EmitterState],
        repertoire: Any,  # MOPopulation
        keys: chex.Array,
        generation: int = 0,
        params: Any = None,
    ) -> Tuple[BasePopulation[Any], Optional[EmitterState]]:
        """Generate offspring by crossover + mutation from repertoire population."""
        pop = repertoire  # MOPopulation
        B = self._batch_size
        N = pop.fitness.shape[0]
        config = self.genome_config

        # --- 1. Sample parent indices ---
        k_p1, k_p2, *k_offspring = jax.random.split(keys[0], 2 + B)
        idx1 = jax.random.randint(k_p1, (B,), 0, N)
        idx2 = jax.random.randint(k_p2, (B,), 0, N)

        # --- 2. Gather parent genomes (SoA indexing) ---
        def _gather(genes: LinearGenome, idx: chex.Array) -> LinearGenome:
            return jax.tree_util.tree_map(lambda x: x[idx], genes)

        parents1 = _gather(pop.genes, idx1)
        parents2 = _gather(pop.genes, idx2)

        # --- 3. Crossover on variation_percentage of batch ---
        n_cross = int(B * self.variation_percentage)
        B - n_cross

        k_cross_keys = jax.random.split(k_offspring[0], n_cross)

        def _cross_one(k: chex.Array, p1: LinearGenome, p2: LinearGenome) -> LinearGenome:
            noise = self.crossover._generate_noise(k.reshape(1, -1)[:1], config, generation)
            return self.crossover._recombine_one(p1, p2, noise, config)

        crossed = jax.vmap(_cross_one)(
            k_cross_keys,
            jax.tree_util.tree_map(lambda x: x[:n_cross], parents1),
            jax.tree_util.tree_map(lambda x: x[:n_cross], parents2),
        )
        mut_only = jax.tree_util.tree_map(lambda x: x[n_cross:], parents1)

        # Concatenate crossed + mutation-only into one batch
        combined = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=0), crossed, mut_only
        )

        # --- 4. Mutate all offspring ---
        n_mut_keys = self.mutation.num_keys_per_atomic_operation
        k_mut = jax.random.split(k_offspring[1], B * n_mut_keys)

        def _mutate_one(k_block: chex.Array, genome: LinearGenome) -> LinearGenome:
            noise = self.mutation._generate_noise(k_block, config, generation)
            return self.mutation._mutate_one(genome, noise, config)

        k_mut_reshaped = k_mut.reshape(B, n_mut_keys, *k_mut.shape[1:])
        offspring_genes = jax.vmap(_mutate_one)(k_mut_reshaped, combined)

        # --- 5. Wrap in MOPopulation ---
        n_objectives = pop.fitness.shape[1] if pop.fitness.ndim > 1 else 1
        offspring_fitness = jnp.zeros((B, n_objectives))
        offspring_pop = pop.replace(  # type: ignore[attr-defined]
            genes=offspring_genes,
            fitness=offspring_fitness,
        )
        return offspring_pop, None


__all__ = ["LSPMOEmitter"]
