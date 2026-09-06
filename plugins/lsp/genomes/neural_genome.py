from typing import Any, Tuple, Type

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome
from malthusjax.composer import register_genome

@register_genome("neural_genome", override=True)
@struct.dataclass
class NeuralGenome(BaseGenome):
    """A custom genome configuration."""

    values: chex.Array
    
    @classmethod
    def random_init(cls: Type["NeuralGenome"], key: chex.PRNGKey, config: Any) -> "NeuralGenome":
        # TODO: Initialize random genome values
        shape = (10,)
        return cls(values=jax.random.normal(key, shape))

    def distance(self, other: BaseGenome, metric: str) -> chex.Numeric:
        # TODO: Implement distance metric
        return jnp.sum(jnp.abs(self.values - other.values))

    def autocorrect(self, config: Any) -> "NeuralGenome":
        # TODO: Enforce constraints
        return self

    @property
    def size(self) -> int:
        return self.values.size

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @classmethod
    def from_tensor(cls: Type["NeuralGenome"], arr: chex.Array, config: Any = None) -> "NeuralGenome":
        return cls(values=arr)
