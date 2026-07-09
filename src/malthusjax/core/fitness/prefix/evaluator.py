"""Base prefix-aware evaluator for Multi-Expression Programming.

Extends :class:`BaseEvaluator` following the same Liskov-compatible
pattern as :class:`BaseQDEvaluator`: the standard ``evaluate()`` method
delegates to the new ``evaluate_all_prefixes()`` and reduces to a scalar,
while ``evaluate_population()`` returns a :class:`PrefixPopulation`
carrying the full ``(pop_size, L)`` fitness matrix.
"""

from __future__ import annotations

from typing import Any, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.genome.prefix.genome import BasePrefixAwareGenome
from malthusjax.core.genome.prefix.population import PrefixPopulation

C = TypeVar("C", bound="BaseEvaluatorConfig")
D = TypeVar("D")


@struct.dataclass
class BasePrefixEvaluator(BaseEvaluator[BasePrefixAwareGenome, C, D]):
    """Evaluator that returns per-prefix fitness: ``(L,)`` instead of scalar.

    Concrete subclasses must implement :meth:`evaluate_all_prefixes`.
    The scalar :meth:`evaluate` method is provided for Liskov
    compatibility — it takes the best (minimum) prefix fitness.

    Type Parameters:
        C: Config type (pytree_node=False).
        D: Data type (e.g., regression data).
    """

    # ------------------------------------------------------------------
    # Core interface: subclasses implement this
    # ------------------------------------------------------------------

    def evaluate_all_prefixes(
        self, genome: BasePrefixAwareGenome
    ) -> chex.Array:
        """Compute fitness for every prefix of *genome*.

        Args:
            genome: An unbatched :class:`BasePrefixAwareGenome`.

        Returns:
            A 1-D array of shape ``(L,)`` where entry *l* is the
            fitness of the program defined by rows ``0 … l``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate_all_prefixes"
        )

    # ------------------------------------------------------------------
    # Liskov-compatible scalar evaluate
    # ------------------------------------------------------------------

    def evaluate(self, genome: BasePrefixAwareGenome) -> chex.Numeric:
        """Scalar fitness: best prefix wins (symbiotic selection).

        Uses the evaluator's ``config.maximize`` flag to pick the best
        prefix (``jnp.max`` when maximising, ``jnp.min`` when minimising).
        """
        all_prefix_fitness = self.evaluate_all_prefixes(genome)
        if self.config.maximize:
            return jnp.max(all_prefix_fitness)
        return jnp.min(all_prefix_fitness)

    # ------------------------------------------------------------------
    # Population-level evaluation → PrefixPopulation
    # ------------------------------------------------------------------

    def evaluate_population(
        self, population: BasePopulation[BasePrefixAwareGenome]
    ) -> PrefixPopulation:
        """Vectorised evaluation returning a :class:`PrefixPopulation`.

        The returned population carries:

        * ``fitness``: scalar best-prefix fitness per individual ``(pop_size,)``.
        * ``prefix_fitness``: full ``(pop_size, L)`` matrix.
        * ``winning_prefix_idx``: index of the winning prefix ``(pop_size,)``.
        """
        prefix_fitness = jax.vmap(self.evaluate_all_prefixes)(population.genes)

        if self.config.maximize:
            scalar_fitness = jnp.max(prefix_fitness, axis=-1)
            winning_idx = jnp.argmax(prefix_fitness, axis=-1)
        else:
            scalar_fitness = jnp.min(prefix_fitness, axis=-1)
            winning_idx = jnp.argmin(prefix_fitness, axis=-1)

        return PrefixPopulation(
            genes=population.genes,
            fitness=scalar_fitness,
            config=population.config,
            prefix_fitness=prefix_fitness,
            winning_prefix_idx=winning_idx,
        )
