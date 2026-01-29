"""
Selection Operators.
Refactored for the 'Consumer Paradigm': Pure index generation.
"""

from typing import Any, Optional

import chex
import jax
import jax.lax
import jax.random
from flax import struct

from malthusjax.operators.base import BaseSelection, C, P

_field: Any = struct.field


@struct.dataclass
class ElitePoolSelection(BaseSelection[P, C]):
    """
    Elite Pool Selection (High Performance).
    Uses jax.lax.top_k for O(N log K) efficiency.
    """

    elite_k: int = _field(pytree_node=False, default=10)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        """
        Selects parents from the top 'elite_k' best individuals.
        """
        rng = keys[0] if keys.ndim > 1 else keys

        # 1. Efficiently find top K (Maximization)
        _, best_k_indices = jax.lax.top_k(fitness, self.elite_k)

        # 2. Sample from the pool with replacement
        random_selections = jax.random.randint(
            rng, shape=(self.num_selections,), minval=0, maxval=self.elite_k
        )

        # 3. Map to global indices
        return best_k_indices[random_selections]
