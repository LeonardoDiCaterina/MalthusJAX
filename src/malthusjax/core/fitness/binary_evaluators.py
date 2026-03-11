"""Binary genome fitness evaluators.

Contains classic combinatorial objectives like OneMax (binary sum) and
0/1 knapsack. Each evaluator adheres to the
:class:`BaseEvaluator` interface and supports optional maximization.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from malthusjax.core.genome.binary_genome import BinaryGenome

from .base import BaseEvaluator, BaseEvaluatorConfig


@struct.dataclass
class BinarySumConfig(BaseEvaluatorConfig):
    """Configuration for BinarySum (OneMax) fitness evaluator."""

    pass


@struct.dataclass
class BinarySumEvaluator(BaseEvaluator[BinaryGenome, BinarySumConfig, Any]):
    """OneMax fitness evaluator: count set bits (binary sum).

    Returns count of 1s for maximize=True, count of 0s for maximize=False.
    """

    config: BinarySumConfig
    data: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    def evaluate(self, genome: BinaryGenome) -> chex.Numeric:
        """Evaluate a single binary genome by counting set values."""
        ones_count = jnp.sum(genome.values)
        zeros_count = genome.size - ones_count
        return jax.lax.select(self.config.maximize, ones_count, zeros_count)


@struct.dataclass
class KnapsackConfig(BaseEvaluatorConfig):
    """Configuration for 0/1 Knapsack problem fitness evaluation.

    Attributes:
        weights: Item weights, shape (n_items,).
        values: Item values, shape (n_items,).
        capacity: Maximum weight capacity (scalar).
        penalty_factor: Linear constraint penalty coefficient (default 1000.0).
    """

    weights: chex.Array  # Item weights, shape (n_items,)
    values: chex.Array  # Item values, shape (n_items,)
    capacity: chex.Numeric  # Maximum weight capacity
    penalty_factor: float = 1000.0  # Penalty for exceeding capacity


@struct.dataclass
class KnapsackEvaluator(BaseEvaluator[BinaryGenome, KnapsackConfig, Any]):
    """Knapsack problem fitness evaluator with linear constraint penalties.

    Computes total value minus linear penalty for weight constraint violation.
    Penalty = excess_weight * penalty_factor (jax.lax.select, XLA-safe).
    """

    config: KnapsackConfig
    data: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    def evaluate(self, genome: BinaryGenome) -> chex.Numeric:
        """Evaluate a binary genome representing item selection.

        Args:
            genome: BinaryGenome with values shape (n_items,).

        Returns:
            Total value minus penalty for capacity violation (scalar).
        """
        total_weight = jnp.sum(genome.values * self.config.weights)
        total_value = jnp.sum(genome.values * self.config.values)

        # Penalize infeasibility via JAX arithmetic (no Python control flow)
        excess_weight = jnp.maximum(0.0, total_weight - self.config.capacity)
        penalty = excess_weight * self.config.penalty_factor

        return total_value - penalty

    @staticmethod
    def create_random_problem(
        key: chex.PRNGKey, n_items: int, capacity_ratio: float = 0.5, maximize: bool = True
    ) -> KnapsackConfig:
        """Factory: create random 0/1 knapsack instance (static configuration).

        Args:
            key: JAX PRNG key for random weights and values.
            n_items: Number of items.
            capacity_ratio: Fraction of total weight for capacity (default 0.5).
            maximize: Optimization direction (default True).

        Returns:
            KnapsackConfig with randomly sampled weights, values, and capacity.
        """
        key1, key2 = jr.split(key, 2)

        weights = jr.uniform(key1, (n_items,), minval=1.0, maxval=20.0)
        values = jr.uniform(key2, (n_items,), minval=1.0, maxval=50.0)

        total_weight = jnp.sum(weights)
        capacity = capacity_ratio * total_weight

        return KnapsackConfig(maximize=maximize, weights=weights, values=values, capacity=capacity)
