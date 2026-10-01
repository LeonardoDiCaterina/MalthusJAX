"""Cartesian Genetic Programming Quality-Diversity Emitter.

Selects elite parent graphs from a MAP-Elites repertoire and generates
mutated offspring using Cartesian micro-mutations.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, cast

import chex
from flax import struct
import jax

from malthus_gp.genomes.cartesian_genome import CartesianGenomeConfig
from malthus_gp.operators.cartesian_micro_mutation import CartesianMicroMutation
from malthusjax.composer import register_emitter
from malthusjax.core.base import BasePopulation
from malthusjax.core.logger import get_logger
from malthusjax.operators.base import BaseMutation
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState

logger = get_logger("malthus_gp.operators.cartesian_emitter")

_field: Any = struct.field


@struct.dataclass
class CartesianEmitterState(EmitterState):  # type: ignore[misc]
    """Internal state for CartesianEmitter."""

    pass


@register_emitter("cartesian_emitter", override=True)
@struct.dataclass
class CartesianEmitter(BaseEmitter):  # type: ignore[misc]
    """Quality-Diversity emitter for Cartesian Genetic Programming.

    Samples elite parents from the MAP-Elites repertoire and applies point mutations
    across opcodes, connection arguments, and outputs.
    """

    mutation: BaseMutation[Any, Any] = _field(
        pytree_node=False, default_factory=CartesianMicroMutation
    )
    genome_config: CartesianGenomeConfig = _field(
        pytree_node=False, default_factory=CartesianGenomeConfig
    )
    _batch_size: int = _field(pytree_node=False, default=20)

    @property
    def batch_size(self) -> int:
        """The number of genomes generated per generation cycle."""
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """PRNG keys required per offspring mutation."""
        return getattr(self.mutation, "num_keys_per_atomic_operation", 6)

    def set_input_length(self, length: int) -> "CartesianEmitter":
        """Locks the batch size for static key budgeting."""
        logger.debug("Configuring CartesianEmitter batch size: %d", length)
        return cast("CartesianEmitter", self.replace(_batch_size=length))

    def init(
        self, key: chex.Array, initial_population: BasePopulation[Any], params: Any = None
    ) -> Optional[EmitterState]:
        """Initializes internal emitter state."""
        return CartesianEmitterState()

    def num_keys_for_sampling(self) -> int:
        """Keys needed for Tier 2 sampling from the repertoire."""
        return 1

    def num_keys(self) -> int:
        """Total keys needed: 1 key for repertoire sampling + batch_size * atomic mutation keys."""
        return self.num_keys_for_sampling() + (self.batch_size * self.num_keys_per_atomic_operation)

    def ask(
        self,
        state: Optional[EmitterState],
        repertoire: Any,
        keys: chex.Array,
        generation: int = 0,
        params: Any = None,
    ) -> Tuple[BasePopulation[Any], Optional[EmitterState]]:
        """Samples parents from the repertoire and generates a batch of mutated offspring."""
        k_sample = keys[0]
        k_mut = keys[1:]

        # 1. Sample parents from repertoire
        if hasattr(repertoire, "select"):
            selection = repertoire.select(k_sample, self.batch_size)
            parent_genotypes = selection.genotypes if hasattr(selection, "genotypes") else selection
        elif hasattr(repertoire, "sample"):
            sample_res = repertoire.sample(k_sample, self.batch_size)
            parent_genotypes = sample_res[0] if isinstance(sample_res, tuple) else sample_res
        else:
            raise AttributeError("Repertoire must provide .select() or .sample() method.")

        # 2. Wrap parent genotypes in a population container
        parent_pop = self.genome_config.init_population(k_sample, self.batch_size)
        parent_pop = parent_pop.replace(genes=parent_genotypes)

        # 3. Shape PRNG keys for vectorized mutation
        atomic_keys = self.num_keys_per_atomic_operation
        if getattr(k_mut, "ndim", 1) > 1 and k_mut.shape[-1] == 2:
            keys_per_genome = k_mut.reshape(self.batch_size, atomic_keys, 2)
        else:
            keys_per_genome = k_mut.reshape(self.batch_size, atomic_keys)

        # 4. Apply Cartesian mutation directly across batch
        if hasattr(self.mutation, "_mutate_fused"):

            def _mutate_one(k: chex.Array, g: Any) -> Any:
                return self.mutation._mutate_fused(k, g, self.genome_config, generation)

            new_genes = jax.vmap(_mutate_one, in_axes=(0, 0))(keys_per_genome, parent_pop.genes)
            offspring_pop = parent_pop.spawn_offspring(new_genes)
        else:
            offspring_pop = self.mutation(
                k_mut, parent_pop, config=self.genome_config, generation=generation
            )

        return offspring_pop, state

    def tell(
        self,
        state: Optional[EmitterState],
        repertoire: Any,
        population: BasePopulation[Any],
        fitnesses: chex.Array,
        descriptors: chex.Array,
        key: chex.Array,
    ) -> Optional[EmitterState]:
        """Stateless emitter pass-through."""
        return state


__all__ = ["CartesianEmitter", "CartesianEmitterState"]
