"""Quality-Diversity specific population and genome structures."""

from __future__ import annotations

from typing import Any, TypeVar

import chex
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation

G = TypeVar("G", bound=BaseGenome)


def has_descriptors(pop: Any) -> bool:
    """Return True if the population carries valid, non-None QD descriptors in .info."""
    if not hasattr(pop, "info") or not isinstance(pop.info, dict):
        return False
    return "descriptors" in pop.info and pop.info["descriptors"] is not None


@struct.dataclass
class QDPopulation(BasePopulation[G]):
    """A specialized population class for Quality-Diversity algorithms.

    Provides strongly-typed access to Quality-Diversity metrics (like behavioral
    descriptors) that are stored in the underlying `info` dictionary.
    """

    @property
    def descriptors(self) -> chex.Array:
        """Access the behavioral descriptors for the population."""
        if (
            not hasattr(self, "info")
            or not isinstance(self.info, dict)
            or self.info.get("descriptors") is None
        ):
            raise KeyError(
                "Descriptors not found in population info. Ensure a BaseQDEvaluator "
                "or ProjectedQDEvaluator was used to evaluate this population."
            )
        return self.info["descriptors"]

    @classmethod
    def from_population(
        cls, pop: BasePopulation[G], descriptors: chex.Array | None = None
    ) -> QDPopulation[G]:
        """Upgrade any BasePopulation to a typed QDPopulation, attaching descriptors if provided."""
        info = dict(pop.info) if hasattr(pop, "info") and isinstance(pop.info, dict) else {}
        if descriptors is not None:
            info["descriptors"] = descriptors
        if info.get("descriptors") is None:
            raise ValueError(
                "Cannot construct QDPopulation without valid descriptors. "
                "Provide descriptors or ensure pop.info['descriptors'] is populated."
            )
        return cls(
            genes=pop.genes,
            fitness=pop.fitness,
            config=getattr(pop, "config", None),
            info=info,
        )

    def get_qd_score(self, f_opt: chex.Numeric = 0.0) -> chex.Numeric:
        """Compute the sum of fitnesses of all valid solutions (example metric)."""
        # A simple QD score formulation: sum of fitnesses for valid individuals.
        valid_mask = jnp.isfinite(self.fitness)
        return jnp.sum(jnp.where(valid_mask, self.fitness, 0.0))
