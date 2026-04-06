from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.base import BaseSelection, C, P

_field = struct.field

@struct.dataclass
class SimplifiedElitePoolSelection(BaseSelection[P, C]):
    """
    A stripped-down selection operator for benchmarking parity with Evosax.
    This skips `jnp.argpartition` entirely.
    """
    elite_k: int = _field(pytree_node=False, default=1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        
        pop_size = fitness.shape[0]
        pool_k = min(self.elite_k, pop_size)
        
        # 1. SKIP ARGPARTITION! Just pretend the first 'pool_k' items are the elites
        # (This mimics Evosax's mathematical cost without requiring a global sort)
        
        # 2. Select parents uniformly from this pool
        # jax.random.randint is massively faster than jax.random.choice with probabilities
        if self.typed_keys:
            rng = keys if keys.ndim == 0 else keys[0]
        else:
            rng = keys if keys.ndim <= 1 else keys[0]
            
        parent_idx = jax.random.randint(rng, (self.num_selections,), minval=0, maxval=pool_k)
        
        return parent_idx

    def get_elite_indices(self, fitness: chex.Array) -> chex.Array:
        # Skip sorting here too
        if self.n_elites == 0:
            return jnp.zeros(0, dtype=jnp.int32)
        return jnp.arange(self.n_elites, dtype=jnp.int32)

    def __call__(
        self,
        keys: chex.Array,
        population: P,
        config: Optional[C] = None,
        **kwargs: Any,
    ) -> Tuple[chex.Array, chex.Array]:
        """Override __call__ to mimic ElitePoolSelection's fused signature."""
        fitness = jnp.asarray(getattr(population, "fitness", population))
        parent_idx = self._select(keys, fitness, config, **kwargs)
        elite_idx = self.get_elite_indices(fitness)
        return parent_idx, elite_idx