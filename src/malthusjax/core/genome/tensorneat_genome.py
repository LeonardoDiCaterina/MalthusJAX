from typing import Any, Tuple, Type

import chex
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome
from malthusjax.core.genome.qd.population import QDPopulation


@struct.dataclass
class TensorNeatGenomeConfig:
    max_nodes: int = 50
    max_conns: int = 100
    # Additional TensorNEAT specific config can be stored here


@struct.dataclass
class TensorNeatGenome(BaseGenome):
    """
    MalthusJAX Wrapper for TensorNEAT's dynamic graph representation.
    The underlying value is a tuple of two matrices: (nodes, conns).

    Because XLA requires static sizes, these matrices are padded with NaNs
    and I_INF up to max_nodes and max_conns.
    """

    values: Tuple[chex.Array, chex.Array]  # (nodes, conns)

    @classmethod
    def random_init(
        cls: Type["TensorNeatGenome"], key: chex.PRNGKey, config: Any
    ) -> "TensorNeatGenome":
        """
        In TensorNEAT, initialization also requires the global `State`.
        This method is provided to fulfill the BaseGenome signature but in practice,
        the Emitter or specialized factory will handle the exact TensorNEAT initialization.
        """
        raise NotImplementedError(
            "TensorNeat genomes should be initialized using TensorNeat QDEmitter or dedicated factory."
        )

    def distance(self, other: BaseGenome, metric: str) -> chex.Numeric:
        """Structural distance or parameter distance."""
        # For QD, typically distance is computed on behavioral descriptors, not genotypes.
        return jnp.array(0.0)

    def autocorrect(self, config: Any) -> BaseGenome:
        return self

    @property
    def size(self) -> int:
        return int(self.values[0].size + self.values[1].size)

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    @classmethod
    def from_tensor(
        cls: Type["TensorNeatGenome"], arr: Any, config: Any = None
    ) -> "TensorNeatGenome":
        # Usually 'arr' here is already the tuple (nodes, conns) due to JAX tree mappings
        return cls(values=arr)


@struct.dataclass
class TensorNeatPopulation(QDPopulation[TensorNeatGenome]):
    """
    Population class containing batched (nodes, conns) matrices and QD descriptors.
    """

    pass
