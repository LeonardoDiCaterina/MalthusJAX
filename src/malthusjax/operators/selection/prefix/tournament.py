"""Tournament Selection over a Prefix-Aware Flat Pool.

Applies genome-local rank deflation to penalize autocorrelation,
flattens the (pop_size, L) matrix into a single candidate pool,
runs a standard k-tournament, and returns (parent_idx, prefix_idx) pairs.
"""

from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.selection.prefix.base import BasePrefixSelection, C


@struct.dataclass
class PrefixTournamentConfig:
    """Configuration for Prefix Tournament Selection.

    Attributes:
        alpha: Deflation factor in [0, 1].
            - alpha = 1.0: No deflation (full flat pool).
            - alpha = 0.0: Strict deduplication (only rank 0 survives).
    """
    alpha: float = struct.field(pytree_node=False, default=0.0)  # type: ignore[no-untyped-call]
    maximize: bool = struct.field(pytree_node=False, default=False)  # type: ignore[no-untyped-call]


@struct.dataclass
class PrefixTournamentSelection(BasePrefixSelection[C]):
    """Tournament Selection for the Flat Pool Engine.

    Before flattening the ``(pop_size, L)`` fitness matrix, this operator applies
    genome-local rank deflation to prevent adjacent highly-correlated prefixes
    from flooding the tournament.

    For minimization:  f'_{i,l} = f_{i,l} / (alpha^{rank} + 1e-8)
    For maximization: f'_{i,l} = f_{i,l} * alpha^{rank}
    """

    tournament_size: int = struct.field(pytree_node=False, default=3)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _apply_rank_deflation(
        self, prefix_fitness: chex.Array, alpha: float, maximize: bool
    ) -> chex.Array:
        """Applies deflation across the L dimension of the fitness matrix."""
        # Handle the edge case of exact 0.0 or 1.0 to avoid numerical instability
        if alpha >= 1.0:
            return prefix_fitness
            
        pop_size, L = prefix_fitness.shape
        
        # We need the rank of each prefix within its genome.
        # jnp.argsort gives the index of the n-th smallest element.
        # jnp.argsort(jnp.argsort(x)) gives the rank of each element!
        
        if maximize:
            # Rank 0 should be the largest value
            sort_order = jnp.argsort(-prefix_fitness, axis=1)
        else:
            # Rank 0 should be the smallest value
            sort_order = jnp.argsort(prefix_fitness, axis=1)
            
        ranks = jnp.argsort(sort_order, axis=1)  # Shape (pop_size, L)

        # Normal deflation
        safe_alpha = jnp.maximum(alpha, 1e-6)
        decay = safe_alpha ** ranks
        
        if maximize:
            return prefix_fitness * decay
        else:
            return prefix_fitness / (decay + 1e-8)

    def _select_prefix(
        self, keys: chex.Array, prefix_fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        """Runs the flat-pool tournament."""
        
        # 1. Deflation Configuration
        # Fall back to 0.0 deflation if no config is provided
        alpha = getattr(config, "alpha", 0.0) if config is not None else 0.0
        maximize = getattr(config, "maximize", False) if config is not None else False
        
        # 2. Apply Genome-Local Rank Deflation
        deflated_fitness = self._apply_rank_deflation(prefix_fitness, alpha, maximize)
        
        # 3. Flatten the matrix to (pop_size * L)
        pop_size, L = deflated_fitness.shape
        total_candidates = pop_size * L
        flat_fitness = deflated_fitness.reshape(-1)
        
        # 4. Standard Tournament Sampling
        if self.typed_keys:
            rng = keys if keys.ndim == 0 else keys[0]
        else:
            rng = keys if keys.ndim <= 1 else keys[0]
            
        # Sample tournament batches: shape (num_selections, tournament_size)
        candidates = jax.random.randint(
            rng, shape=(self.num_selections, self.tournament_size), minval=0, maxval=total_candidates
        )
        
        candidate_fitness = jnp.take(flat_fitness, candidates, axis=0)
        
        # 5. Determine Winners
        if maximize:
            winner_local_indices = jnp.argmax(candidate_fitness, axis=1)
        else:
            winner_local_indices = jnp.argmin(candidate_fitness, axis=1)
            
        # Extract the winning flat indices: shape (num_selections,)
        winning_flat_idx = jnp.take_along_axis(candidates, winner_local_indices[:, None], axis=1).reshape(-1)
        
        # 6. Unravel Flat Indices back to (parent_idx, prefix_idx) pairs
        parent_idx, prefix_idx = jnp.divmod(winning_flat_idx, L)
        
        return jnp.stack([parent_idx, prefix_idx], axis=-1)
