"""Quality-Diversity Emitter for NEAT neural networks.

Selects elite parent networks from a MAP-Elites repertoire and emits mutated
topologies using vectorized structural and synaptic weight perturbations.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthus_neat.core.state import State
from malthus_neat.genomes.neat_genome import NeatGenome, NeatGenomeConfig, NeatPopulation
from malthusjax.composer import register_emitter
from malthusjax.core.base import BasePopulation
from malthusjax.core.logger import get_logger
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState

logger = get_logger("malthus_neat.operators.neat_emitter")

_field: Any = struct.field


@struct.dataclass
class NeatEmitterState(EmitterState):  # type: ignore[misc]
    """Internal state for NeatEmitter."""

    pass


@register_emitter("neat_emitter", override=True)
@struct.dataclass
class NeatEmitter(BaseEmitter):  # type: ignore[misc]
    """Quality-Diversity emitter for NEAT network topologies."""

    genome_config: NeatGenomeConfig = _field(pytree_node=False, default_factory=NeatGenomeConfig)
    _batch_size: int = _field(pytree_node=False, default=20)
    mutation: Any = _field(pytree_node=False, default=None)
    _tn_genome: Any = _field(pytree_node=False, default=None)
    _state: Any = _field(pytree_node=False, default=None)

    def __post_init__(self) -> None:
        if self._tn_genome is None:
            object.__setattr__(self, "_tn_genome", self.genome_config.to_tensorneat_genome())
        if self._state is None:
            st = State().register(randkey=jax.random.PRNGKey(0))
            st = self._tn_genome.setup(st)
            object.__setattr__(self, "_state", st)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def set_input_length(self, length: int) -> "NeatEmitter":
        return cast("NeatEmitter", self.replace(_batch_size=length))

    def init(
        self, key: chex.Array, initial_population: BasePopulation[Any], params: Any = None
    ) -> Optional[EmitterState]:
        return NeatEmitterState()

    def num_keys_for_sampling(self) -> int:
        return 1

    def num_keys(self) -> int:
        return self.num_keys_for_sampling() + self.batch_size

    def ask(
        self,
        state: Optional[EmitterState],
        repertoire: Any,
        keys: chex.Array,
        generation: int = 0,
        params: Any = None,
    ) -> Tuple[BasePopulation[Any], Optional[EmitterState]]:
        """Samples elite networks from repertoire and emits mutated offspring."""
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

        if isinstance(parent_genotypes, NeatGenome):
            p_nodes, p_conns = parent_genotypes.nodes, parent_genotypes.conns
        elif isinstance(parent_genotypes, tuple):
            p_nodes, p_conns = parent_genotypes
        else:
            p_nodes, p_conns = parent_genotypes.nodes, parent_genotypes.conns

        # 2. Shape mutation keys and innovation IDs
        if getattr(k_mut, "ndim", 1) > 1 and k_mut.shape[-1] == 2:
            flat_mut_keys = k_mut.reshape(self.batch_size, 2)
        else:
            flat_mut_keys = k_mut.reshape(self.batch_size, -1) if k_mut.ndim > 1 else k_mut

        next_node_base = generation * self.batch_size + 200
        new_node_keys = jnp.arange(self.batch_size) + next_node_base
        new_conn_keys = jnp.arange(self.batch_size * 3).reshape(self.batch_size, 3) + (
            next_node_base * 3
        )

        def _mutate_one(
            k: chex.Array,
            n: chex.Array,
            c: chex.Array,
            node_key: chex.Array,
            conn_keys: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            res: Tuple[chex.Array, chex.Array] = self._tn_genome.execute_mutation(
                self._state, k, n, c, node_key, conn_keys
            )
            return res

        mut_nodes, mut_conns = jax.vmap(_mutate_one, in_axes=(0, 0, 0, 0, 0))(
            flat_mut_keys, p_nodes, p_conns, new_node_keys, new_conn_keys
        )

        offspring_genes = NeatGenome(nodes=mut_nodes, conns=mut_conns)
        offspring_pop = NeatPopulation(
            genes=offspring_genes,
            fitness=jnp.zeros(self.batch_size, dtype=jnp.float32),
            config=self.genome_config,
            info={},
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
        return state


__all__ = ["NeatEmitter", "NeatEmitterState"]
