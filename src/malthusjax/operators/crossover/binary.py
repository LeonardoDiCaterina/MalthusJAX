"""
Binary Crossover Operators.
Optimized for batch-first paradigm.
"""

from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from malthusjax.operators.base import BaseCrossover


@struct.dataclass
class UniformCrossover(BaseCrossover[BinaryGenome, BinaryGenomeConfig, BinaryPopulation]):
    """
    Uniform Crossover (Fused 3-Tier Paradigm).
    Each bit is independently sourced from Parent 1 or Parent 2.
    """

    crossover_rate: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 1 key for the Bernoulli mixing mask."""
        return 1

    def _generate_noise(self, keys: chex.PRNGKey, config: BinaryGenomeConfig) -> chex.Array:
        """Tier 2: Generate per-bit mixing mask."""
        return jax.random.bernoulli(keys[0], p=self.crossover_rate, shape=config.shape)

    def _recombine_one(
        self,
        p1: BinaryGenome,
        p2: BinaryGenome,
        noise_data: chex.Array,
        config: BinaryGenomeConfig,
        **_kwargs: Any,
    ) -> BinaryGenome:
        """Tier 1: Bitwise selection using genome genes."""
        mask = noise_data
        # Convention: mask=True selects from p2, False selects from p1
        offspring_genes = jnp.where(mask, p2.values, p1.values)
        return cast(BinaryGenome, cast(Any, p1).replace(values=offspring_genes))


@struct.dataclass
class SinglePointCrossover(BaseCrossover[BinaryGenome, BinaryGenomeConfig, BinaryPopulation]):
    """
    Single-Point Crossover (Fused 3-Tier Paradigm).
    Swaps segments at a random crossover point.
    """

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 1 key to determine the crossover point."""
        return 1

    def _generate_noise(self, keys: chex.PRNGKey, config: BinaryGenomeConfig) -> chex.Array:
        """Tier 2: Generate segment mask based on random point."""
        length = config.shape[0]
        # Avoid 0 and length to ensure meaningful recombination
        crossover_point = jax.random.randint(keys[0], shape=(), minval=1, maxval=length)
        indices = jnp.arange(length)
        return jnp.where(indices < crossover_point, True, False)

    def _recombine_one(
        self,
        p1: BinaryGenome,
        p2: BinaryGenome,
        noise_data: chex.Array,
        config: BinaryGenomeConfig,
        **_kwargs: Any,
    ) -> BinaryGenome:
        """Tier 1: Segment-wise selection."""
        mask = noise_data
        # Convention: mask=True selects from p2, False selects from p1
        offspring_genes = jnp.where(mask, p2.values, p1.values)
        return cast(BinaryGenome, cast(Any, p1).replace(values=offspring_genes))


__all__ = ["UniformCrossover", "SinglePointCrossover"]
