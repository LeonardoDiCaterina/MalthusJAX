"""JAX-native NEAT crossover operator.

Recombines two parent neural network topologies by aligning historical innovations
with full BaseCrossover compliance and JIT compatibility.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import chex
import jax
from flax import struct

from malthus_neat.core.state import State
from malthus_neat.genomes.neat_genome import NeatGenome, NeatGenomeConfig
from malthusjax.composer import register_crossover
from malthusjax.core.base import BasePopulation
from malthusjax.core.logger import get_logger
from malthusjax.operators.base import BaseCrossover

logger = get_logger("malthus_neat.operators.neat_crossover")

_field: Any = struct.field


@register_crossover("neat_crossover", override=True)
@struct.dataclass
class NeatCrossover(BaseCrossover[NeatGenome, NeatGenomeConfig]):  # type: ignore[misc]
    """NEAT Crossover operator aligning matching genes by historical innovation."""

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

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        return input_shape[0] if len(input_shape) > 0 else 1

    def _recombine_one(
        self,
        rng: chex.Array,
        parent1: NeatGenome,
        parent2: NeatGenome,
        config: NeatGenomeConfig,
        **kwargs: Any,
    ) -> Tuple[NeatGenome, NeatGenome]:
        return parent1, parent2

    def __call__(
        self,
        keys: chex.Array,
        pop1: BasePopulation[NeatGenome],
        pop2: BasePopulation[NeatGenome],
        config: Optional[NeatGenomeConfig] = None,
        generation: int = 0,
    ) -> BasePopulation[NeatGenome]:
        """Applies vectorized crossover between two parent populations."""
        pop_size = len(pop1)
        if getattr(keys, "ndim", 1) > 1 and keys.shape[-1] == 2:
            flat_keys = keys.reshape(pop_size, 2)
        else:
            flat_keys = keys.reshape(pop_size, -1) if keys.ndim > 1 else keys

        def _cross_one(
            k: chex.Array,
            n1: chex.Array,
            c1: chex.Array,
            n2: chex.Array,
            c2: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            res: Tuple[chex.Array, chex.Array] = self._tn_genome.execute_crossover(
                self._state, k, n1, c1, n2, c2
            )
            return res

        cross_nodes, cross_conns = jax.vmap(_cross_one, in_axes=(0, 0, 0, 0, 0))(
            flat_keys,
            pop1.genes.nodes,
            pop1.genes.conns,
            pop2.genes.nodes,
            pop2.genes.conns,
        )

        new_genes = NeatGenome(nodes=cross_nodes, conns=cross_conns)
        return pop1.spawn_offspring(new_genes)


__all__ = ["NeatCrossover"]
