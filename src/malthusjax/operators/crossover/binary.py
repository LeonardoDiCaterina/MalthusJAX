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
class UniformCrossover(BaseCrossover[BinaryGenome, BinaryGenomeConfig]):
    """
    Uniform Crossover (Fused 3-Tier Paradigm).
    Per-bit independent selection from parents via Bernoulli mask. XLA fuses mask generation
    (Tier 2) with selection kernel (Tier 1) into single compiled operation.

    Shape contract: Parent (N,) X Parent (N,) -> Offspring (N,)
    Key budget: 1 pre-allocated subkey (from ResourceMapper) per pair.
    """

    crossover_rate: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Bernoulli mask generation requires 1 PRNG subkey."""
        return 1

    def _generate_noise(
        self, keys: chex.PRNGKey, config: BinaryGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """Tier 2 — Bernoulli Mask. Returns (N,) boolean array for per-bit selection."""
        return jax.random.bernoulli(keys[0], p=self.crossover_rate, shape=config.shape)

    def _recombine_one(
        self,
        p1: BinaryGenome,
        p2: BinaryGenome,
        noise_data: chex.Array,
        config: BinaryGenomeConfig,
        **_kwargs: Any,
    ) -> BinaryGenome:
        """
        Tier 1 — XLA-Fused Recombination Kernel.
        Per-bit selection (True=p2, False=p1) fused with mask generation for single kernel launch.

        Returns: Offspring BinaryGenome with (N,) bits
        """
        mask = noise_data
        offspring_genes = jnp.where(mask, p2.values, p1.values)
        return cast(BinaryGenome, cast(Any, p1).replace(values=offspring_genes))


@struct.dataclass
class SinglePointCrossover(BaseCrossover[BinaryGenome, BinaryGenomeConfig]):
    """
    Single-Point Crossover (Fused 3-Tier Paradigm).
    Selects a random crossover point [1, N-1); swaps segments. Avoids boundary points (0, N)
    to ensure meaningful recombination (both parents contribute genes).

    Shape contract: Parent (N,) X Parent (N,) -> Offspring (N,)
    Key budget: 1 pre-allocated subkey (from ResourceMapper) per pair.
    """

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Randint sampling for crossover point requires 1 PRNG subkey."""
        return 1

    def _generate_noise(
        self, keys: chex.PRNGKey, config: BinaryGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """
        Tier 2 — Segment Mask.
        Generates (N,) boolean mask based on random crossover point [1, N-1).
        True indicates second segment (from parent 2).
        """
        length = config.shape[0]
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
        """
        Tier 1 — XLA-Fused Segment Crossover Kernel.
        Segment-wise selection (True=p2, False=p1) fused with mask generation.

        Returns: Offspring BinaryGenome with (N,) bits
        """
        mask = noise_data
        offspring_genes = jnp.where(mask, p2.values, p1.values)
        return cast(BinaryGenome, cast(Any, p1).replace(values=offspring_genes))


__all__ = ["UniformCrossover", "SinglePointCrossover"]
