"""JAX-native NEAT mutation operator.

Applies synaptic weight perturbation, node insertion (edge split), and connection
addition across tensorized NEAT genomes with full BaseMutation compliance.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthus_neat.core.state import State
from malthus_neat.genomes.neat_genome import NeatGenome, NeatGenomeConfig
from malthusjax.composer import register_mutation
from malthusjax.core.base import BasePopulation
from malthusjax.core.logger import get_logger
from malthusjax.operators.base import BaseMutation

logger = get_logger("malthus_neat.operators.neat_mutation")

_field: Any = struct.field


@register_mutation("neat_mutation", override=True)
@struct.dataclass
class NeatMutation(BaseMutation[NeatGenome, NeatGenomeConfig]):  # type: ignore[misc]
    """NEAT Mutation operator for topological and synaptic weight mutations."""

    genome_config: NeatGenomeConfig = _field(pytree_node=False, default_factory=NeatGenomeConfig)
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
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        n_individuals = input_shape[0] if len(input_shape) > 0 else 1
        return int(n_individuals * self.num_offspring * self.num_keys_per_atomic_operation)

    def _mutate_one(
        self,
        genome: NeatGenome,
        noise_data: Any,
        config: NeatGenomeConfig,
        **kwargs: Any,
    ) -> NeatGenome:
        return genome

    def __call__(
        self,
        keys: chex.Array,
        population: BasePopulation[NeatGenome],
        config: Optional[NeatGenomeConfig] = None,
        generation: int = 0,
    ) -> BasePopulation[NeatGenome]:
        """Applies vectorized NEAT mutation across the population."""
        pop_size = len(population)
        if getattr(keys, "ndim", 1) > 1 and keys.shape[-1] == 2:
            flat_keys = keys.reshape(pop_size, 2)
        else:
            flat_keys = keys.reshape(pop_size, -1) if keys.ndim > 1 else keys

        # Batch innovation keys based on generation counter
        next_node_base = generation * pop_size + 100
        new_node_keys = jnp.arange(pop_size) + next_node_base
        new_conn_keys = jnp.arange(pop_size * 3).reshape(pop_size, 3) + (next_node_base * 3)

        def _mutate_one_internal(
            k: chex.Array,
            nodes: chex.Array,
            conns: chex.Array,
            node_key: chex.Array,
            conn_keys: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            res: Tuple[chex.Array, chex.Array] = self._tn_genome.execute_mutation(
                self._state, k, nodes, conns, node_key, conn_keys
            )
            return res

        mut_nodes, mut_conns = jax.vmap(_mutate_one_internal, in_axes=(0, 0, 0, 0, 0))(
            flat_keys,
            population.genes.nodes,
            population.genes.conns,
            new_node_keys,
            new_conn_keys,
        )

        new_genes = NeatGenome(nodes=mut_nodes, conns=mut_conns)
        return population.spawn_offspring(new_genes)


__all__ = ["NeatMutation"]
