"""Prefix-aware population container for Multi-Expression Programming.

Uses struct fields (the MOPopulation pattern) rather than the ``info``
dict to store prefix-level fitness and statistics.  This gives type
safety, proper pytree tracing, and automatic slicing/merging.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional, Type, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.genome.prefix.genome import BasePrefixAwareGenome, PrefixGenomeConfig


@struct.dataclass
class PrefixPopulation(BasePopulation[BasePrefixAwareGenome]):
    """Population container with per-prefix fitness as a proper struct field.

    Follows the same pattern as :class:`MOPopulation`: extension-specific
    data lives in typed struct fields, not in a loose ``info`` dict.

    Attributes:
        prefix_fitness: Per-prefix fitness matrix of shape ``(pop_size, L)``.
            Each entry ``[i, l]`` is the fitness of individual *i* evaluated
            at prefix (readout row) *l*.  ``None`` before evaluation.
        winning_prefix_idx: Index of the best-performing prefix for each
            individual, shape ``(pop_size,)``.  ``None`` before evaluation.
    """

    GENOME_CLS: ClassVar[Type[BasePrefixAwareGenome]] = BasePrefixAwareGenome

    prefix_fitness: Optional[chex.Array] = struct.field(default=None)
    winning_prefix_idx: Optional[chex.Array] = struct.field(default=None)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def init_random(
        cls,
        key: chex.PRNGKey,
        config: PrefixGenomeConfig,
        size: int,
    ) -> PrefixPopulation:
        """Initialise a random population of prefix-aware genomes."""
        batched_genes = BasePrefixAwareGenome.create_population(key, config, size)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(
            genes=batched_genes,
            fitness=initial_fitness,
            config=config,
            prefix_fitness=None,
            winning_prefix_idx=None,
        )

    # ------------------------------------------------------------------
    # Provenance statistics (vmapped over the population)
    # ------------------------------------------------------------------

    def get_population_provenance(self) -> chex.Array:
        """Compute the operand-provenance mask for every individual.

        Returns:
            Boolean array of shape ``(pop_size, L, max_arity)`` where
            ``True`` = raw input reference, ``False`` = previous-row
            reference.
        """

        def _single_provenance(genome: BasePrefixAwareGenome) -> chex.Array:
            return genome.get_operand_provenance(self.config)

        return jax.vmap(_single_provenance)(self.genes)

    def get_population_effective_p_input(self) -> chex.Array:
        """Per-individual effective p_input: shape ``(pop_size,)``."""

        def _single(genome: BasePrefixAwareGenome) -> chex.Numeric:
            return genome.get_effective_p_input(self.config)

        return jax.vmap(_single)(self.genes)

    @property
    def mean_effective_p_input(self) -> chex.Numeric:
        """Population-wide mean effective p_input (scalar)."""
        return jnp.mean(self.get_population_effective_p_input())
