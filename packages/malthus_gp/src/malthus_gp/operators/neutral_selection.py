"""Neutral Drift Selection for Cartesian Genetic Programming.

Standard (1 + λ)-ES in Cartesian GP requires that when an offspring has fitness
equal to or better than the parent, the offspring MUST replace the parent.
This neutral drift allows the algorithm to traverse neutral genotype networks
without stalling on local optima (Miller & Thomson, 2000).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, cast

import chex
import jax.numpy as jnp
from flax import struct

from malthusjax.composer import register_selection
from malthusjax.core.logger import get_logger
from malthusjax.operators.selection.elite_pool import ElitePoolSelection

logger = get_logger("malthus_gp.operators.neutral_selection")

_field: Any = struct.field


@register_selection("cgp_neutral", override=True)
@struct.dataclass
class NeutralSelection(ElitePoolSelection):  # type: ignore[misc]
    """(1 + λ)-ES Selection with Neutral Drift tie-breaking.

    Standard argsort preserves the parent (index 0) on equal fitness ties.
    NeutralSelection adjusts fitness values by a tiny epsilon (1e-5) proportional to
    individual index so that newer offspring aggressively win ties against the parent.
    """

    elite_k: int = _field(pytree_node=False, default=1)
    maximize: bool = _field(pytree_node=False, default=False)

    def __call__(
        self,
        keys: chex.Array,
        population: Any,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[chex.Array, chex.Array]:
        raw_fitness = getattr(population, "fitness", population)
        fitness = jnp.asarray(raw_fitness)

        pop_size = fitness.shape[0]
        tie_breaker = jnp.arange(pop_size, dtype=fitness.dtype) * 1e-5

        # In minimization, lower is better, so subtract tie_breaker to favor higher index
        # In maximization, higher is better, so add tie_breaker
        modified_fitness = jnp.where(
            self.maximize,
            fitness + tie_breaker,
            fitness - tie_breaker,
        )

        pop_container = population
        if hasattr(population, "replace"):
            pop_container = population.replace(fitness=modified_fitness)
        else:
            pop_container = modified_fitness

        result = super().__call__(keys, pop_container, config, **kwargs)
        return cast(Tuple[chex.Array, chex.Array], result)
