"""Quality-Diversity specific population and genome structures."""

from typing import TypeVar
import chex
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation

G = TypeVar("G")

@struct.dataclass
class QDPopulation(BasePopulation[G]):
    """A specialized population class for Quality-Diversity algorithms.
    
    Provides strongly-typed access to Quality-Diversity metrics (like behavioral
    descriptors) that are stored in the underlying `info` dictionary.
    """
    
    @property
    def descriptors(self) -> chex.Array:
        """Access the behavioral descriptors for the population."""
        if "descriptors" not in self.info:
            raise KeyError(
                "Descriptors not found in population info. Ensure a BaseQDEvaluator "
                "was used to evaluate this population."
            )
        return self.info["descriptors"]
        
    def get_qd_score(self, f_opt: chex.Numeric = 0.0) -> chex.Numeric:
        """Compute the sum of fitnesses of all valid solutions (example metric)."""
        # A simple QD score formulation: sum of fitnesses for valid individuals.
        valid_mask = jnp.isfinite(self.fitness)
        return jnp.sum(jnp.where(valid_mask, self.fitness, 0.0))
