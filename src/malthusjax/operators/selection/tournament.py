from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.base import BaseSelection, C, P

_field: Any = struct.field


@struct.dataclass
class TournamentSelection(BaseSelection[P, C]):
    """Tournament Selection (Balanced Exploitation & Exploration).

    In tournament selection, the algorithm repeatedly:
    1. Randomly sample ``tournament_size`` individuals from the population
    2. Select the individual with the highest fitness
    3. Repeat ``num_selections`` times (with replacement)

    This provides a good middle ground between pure elite selection (high exploitation)
    and fitness-proportional selection (fitness-weighted exploration).

    **String Specification Format**::

        "tournament:num_selections=INT,tournament_size=INT"

    Examples::

        "tournament"  # Use defaults (num_selections=4,
                     #                tournament_size=3)
        "tournament:num_selections=25"  # Custom num_selections, default tournament_size
        "tournament:num_selections=50,tournament_size=5"  # Both custom
        "tournament:tournament_size=7"  # Custom tournament size, default num_selections

    Parameters
    ----------
    tournament_size : int, optional
        Number of candidates sampled per tournament round.
        Valid range: 2 to population_size.
        - tournament_size=2: Mild selection pressure, high diversity
        - tournament_size=3-5: Balanced (recommended for most problems)
        - tournament_size=7+: Strong selection pressure, lower diversity
        Default: 3 (recommended for general-purpose problems).

    num_selections : int
        Number of individuals to select (typically equals population size).
        Set by the engine during :meth:`set_input_length`.

    Notes
    -----
    **Fitness Requirements**: Tournament selection supports all fitness ranges
    (positive, negative, zero, or mixed). The algorithm only compares relative
    fitness values, making it robust to any fitness function.

    **Selection Pressure**: The ``tournament_size`` parameter directly controls
    selection pressure.
    - Larger tournaments → stronger preference for high-fitness individuals
    - Smaller tournaments → more uniform selection (higher diversity)

    **Recommended Default**: ``tournament_size=3`` offers a good balance for
    most optimization problems. Increase to 5-7 for harder landscapes or when
    you want faster convergence. Use 2 for maintaining diversity.

    **Computational Complexity**: O(num_selections × tournament_size) for one call.
    Very efficient on GPUs due to vectorization of tournament sampling.

    **Trade-offs**:
    - Better diversity than elite selection alone
    - Simpler than fitness-proportional selection (no averaging)
    - Works with any fitness function (unlike roulette selection)
    - Consistent selection pressure across generations
    """

    tournament_size: int = _field(pytree_node=False, default=3)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _select(
        self, keys: chex.Array, fitness: chex.Array, config: Optional[C] = None, **kwargs: Any
    ) -> chex.Array:
        """Select parents through competitive tournaments.

        Each of the ``num_selections`` tournaments samples ``tournament_size``
        candidates and picks the fittest (lowest fitness in minimization convention).
        The returned array contains the winning indices.
        """
        if self.typed_keys:
            rng = keys if keys.ndim == 0 else keys[0]
        else:
            rng = keys if keys.ndim <= 1 else keys[0]
        pop_size = fitness.shape[0]
        candidates = jax.random.randint(
            rng, shape=(self.num_selections, self.tournament_size), minval=0, maxval=pop_size
        )
        candidate_fitness = jnp.take(fitness, candidates, axis=0)
        winner_local_indices = jnp.argmin(candidate_fitness, axis=1)
        return jnp.take_along_axis(candidates, winner_local_indices[:, None], axis=1).reshape(-1)
