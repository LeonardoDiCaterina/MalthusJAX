from typing import Any, Optional, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.base import BaseSelection, C, P


@struct.dataclass
class TournamentSelection(BaseSelection[P, C]):
    tournament_size: int = cast(Any, struct.field)(pytree_node=False, default=3)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
        ) -> chex.Array:
        # Normalize keys - handles both (2,) and (1, 2) shapes
        rng = keys[0] if keys.ndim > 1 else keys
        pop_size = fitness.shape[0]

        # 1. Generate Tournament Candidates
        candidates = jax.random.randint(
            rng, shape=(self.num_selections, self.tournament_size), minval=0, maxval=pop_size
        )

        # 2. Retrieve Fitness of Candidates
        candidate_fitness = jnp.take(fitness, candidates, axis=0)

        # 3. Find Winner of each Tournament
        winner_local_indices = jnp.argmax(candidate_fitness, axis=1)

        # 4. Map back to global population indices
        return jnp.take_along_axis(
            candidates, winner_local_indices[:, None], axis=1
        ).reshape(-1)
