"""Identity crossover operator for Cartesian Genetic Programming.

Since standard Cartesian GP uses a (1 + λ)-ES strategy driven purely by point
mutations, NoOpCrossover satisfies the GeneticEngine interface without
modifying the parent.
"""

from __future__ import annotations

from typing import Any

import chex
import jax.numpy as jnp
from flax import struct

from malthus_gp.genomes.cartesian_genome import CartesianGenome, CartesianGenomeConfig
from malthusjax.composer import register_crossover
from malthusjax.core.logger import get_logger
from malthusjax.operators.base import BaseCrossover

logger = get_logger("malthus_gp.operators.no_op_crossover")

_field: Any = struct.field


@register_crossover(
    name="cartesian_noop",
    override=True,
    compatible_genomes=["cartesian_genome", "cartesian"],
)
@struct.dataclass
class NoOpCrossover(BaseCrossover[CartesianGenome, CartesianGenomeConfig]):  # type: ignore[misc]
    """Identity crossover — always returns parent 1 unchanged.

    CGP relies on point mutations with a (1 + λ) Evolutionary Strategy.
    This operator satisfies the GeneticEngine pipeline requirements while
    passing the parent through.
    """

    crossover_rate: float = _field(pytree_node=False, default=1.0)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _generate_noise(
        self,
        keys: chex.Array,
        config: CartesianGenomeConfig,
        generation: int = 0,
    ) -> chex.Array:
        """No noise required — return a dummy scalar."""
        return jnp.zeros((), dtype=jnp.int32)

    def _recombine_one(
        self,
        p1: CartesianGenome,
        p2: CartesianGenome,
        noise_data: Any,
        config: CartesianGenomeConfig,
        **_kwargs: Any,
    ) -> CartesianGenome:
        """Return parent 1 unchanged."""
        return p1
