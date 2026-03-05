"""
Selection Operators.
Refactored for the 'Consumer Paradigm': Pure index generation.
"""

from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
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
    Performance: O(N) via jnp.argpartition for elite filtering; O(1) per sample.
    Trade-off: High exploitation (best genes preserved) vs low diversity (limited gene pool).
    Use when: Final refinement phase or small elite trusted pool (K << N).

    When ``n_elites > 0`` (set by the engine via ``set_n_elites``), ``__call__``
    fuses parent pool construction and elite preservation into a single
    ``argpartition`` pass, eliminating the redundant O(N) scan the engine would
    otherwise perform.
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

        This is the primitive parent-only path.  Prefer ``__call__`` which
        fuses parent selection with elite extraction.
        """
        if self.typed_keys:
            rng = keys if keys.ndim == 0 else keys[0]
        else:
            rng = keys if keys.ndim <= 1 else keys[0]

        pop_size = fitness.shape[0]
        pool_k = min(self.elite_k, pop_size)

        if pool_k >= pop_size:
            # Pool covers entire population — argpartition unnecessary
            best_k_indices = jnp.arange(pop_size)
        else:
            best_k_indices = jnp.argpartition(-fitness, pool_k)[:pool_k]

        random_selections = jax.random.randint(
            rng, shape=(self.num_selections,), minval=0, maxval=pool_k
        )
        return best_k_indices[random_selections]

    def __call__(
        self,
        keys: chex.Array,
        population: P,
        config: Optional[C] = None,
        **kwargs: Any,
    ) -> Tuple[chex.Array, chex.Array]:
        """Select parents from elite pool AND extract preservation elites.

        Fuses both operations into a single ``argpartition(-fitness, k)``
        where ``k = max(elite_k, n_elites)``.  When the two sizes differ a
        lightweight O(k log k) ``argsort`` within the top-k identifies the
        correct subsets — negligible cost since k << N.

        Returns:
            ``(parent_indices, elite_indices)`` tuple.
        """
        fitness = getattr(population, "fitness", population)

        # Key extraction (same logic as _select)
        if self.typed_keys:
            rng = keys if keys.ndim == 0 else keys[0]
        else:
            rng = keys if keys.ndim <= 1 else keys[0]

        pop_size = fitness.shape[0]
        k = min(max(self.elite_k, self.n_elites), pop_size)

        if k >= pop_size:
            # Pool covers entire population — argpartition unnecessary
            top_k_idx = jnp.arange(pop_size)
        else:
            # Single O(N) partial sort covers both parent pool and preservation
            top_k_idx = jnp.argpartition(-fitness, k)[:k]

        pool_k = min(self.elite_k, pop_size)

        if self.n_elites == 0:
            # No preservation — pool is exactly top_elite_k (fast path)
            pool = top_k_idx[:pool_k]
            elite_idx = jnp.zeros(0, dtype=jnp.int32)
        elif self.elite_k == self.n_elites:
            # Both want the same set — no secondary sort
            pool = top_k_idx[:pool_k]
            elite_idx = top_k_idx[:self.n_elites]
        else:
            # Need ranking within top-k to split correctly.
            # O(k log k) on ≤max(elite_k, n_elites) elements — negligible.
            sorted_within = jnp.argsort(-fitness[top_k_idx])
            sorted_top_k = top_k_idx[sorted_within]
            pool = sorted_top_k[:pool_k]
            elite_idx = sorted_top_k[:self.n_elites]

        random_selections = jax.random.randint(
            rng, shape=(self.num_selections,), minval=0, maxval=pool_k
        )
        parent_idx = pool[random_selections]

        return parent_idx, elite_idx
