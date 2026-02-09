from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.base import BaseSelection, C, P

_field: Any = struct.field


@struct.dataclass
class TournamentSelection(BaseSelection[P, C]):
    """
    Tournament Selection (Balanced Exploitation & Exploration).
    Strategy: Sample tournament_size random individuals, select winner (highest fitness).
    Shape contract: fitness (pop_size,) → selected_indices (num_selections,).
    Key budget: 1 pre-allocated subkey (randint for candidate generation).
    Tournament dynamics: Larger tournament_size → stronger selection pressure (less diversity).
    Performance: O(num_selections * tournament_size) for all selections.
    Trade-off: Tournament selection offers middle ground between Elite (high exploitation)
    and Roulette (fitness-weighted). Recommended: tournament_size ∈ [2, 7] for balance.
    Use when: Need controlled selection pressure with maintained diversity.
    """

    tournament_size: int = _field(pytree_node=False, default=3)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        """
        Selects num_selections parents via competitive tournaments.
        Returns: (num_selections,) indices into [0, pop_size).
        """
        # Key extraction driven by PRNG impl (typed_keys set at engine init).
        # typed_keys=True: single typed key is scalar (ndim=0), batch is 1D.
        # typed_keys=False (legacy): single key is (2,) ndim=1, batch is (N,2) ndim=2.
        if self.typed_keys:
            rng = keys if keys.ndim == 0 else keys[0]
        else:
            rng = keys if keys.ndim <= 1 else keys[0]
        pop_size = fitness.shape[0]
        candidates = jax.random.randint(
            rng, shape=(self.num_selections, self.tournament_size), minval=0, maxval=pop_size
        )
        candidate_fitness = jnp.take(fitness, candidates, axis=0)
        winner_local_indices = jnp.argmax(candidate_fitness, axis=1)
        return jnp.take_along_axis(candidates, winner_local_indices[:, None], axis=1).reshape(-1)
