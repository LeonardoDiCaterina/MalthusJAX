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
    Strategy: Restrict selection to top elite_k individuals, sample uniformly from pool.
    Shape contract: fitness (pop_size,) → selected_indices (num_selections,).
    Key budget: 1 pre-allocated subkey (randint for pool sampling).
    Performance: O(N log K) via jax.lax.top_k for elite filtering; O(1) per sample.
    Trade-off: High exploitation (best genes preserved) vs low diversity (limited gene pool).
    Use when: Final refinement phase or small elite trusted pool (K << N).
    """

    elite_k: int = _field(pytree_node=False, default=10)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        """
        Selects num_selections parents uniformly from top elite_k individuals.
        Returns: (num_selections,) indices into [0, pop_size).
        """
        # Key extraction driven by PRNG impl (typed_keys set at engine init).
        # typed_keys=True: single typed key is scalar (ndim=0), batch is 1D.
        # typed_keys=False (legacy): single key is (2,) ndim=1, batch is (N,2) ndim=2.
        if self.typed_keys:
            rng = keys if keys.ndim == 0 else keys[0]
        else:
            rng = keys if keys.ndim <= 1 else keys[0]
        _, best_k_indices = jax.lax.top_k(fitness, self.elite_k)
        random_selections = jax.random.randint(
            rng, shape=(self.num_selections,), minval=0, maxval=self.elite_k
        )
        return best_k_indices[random_selections]
