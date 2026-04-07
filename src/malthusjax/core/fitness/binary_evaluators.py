"""Binary genome fitness evaluators.

Contains classic combinatorial objectives like OneMax (binary sum) and
0/1 knapsack. Each evaluator adheres to the
:class:`BaseEvaluator` interface and supports optional maximization.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from malthusjax.core.genome.binary_genome import BinaryGenome

from .base import BaseEvaluator, BaseEvaluatorConfig


class KnapsackData(NamedTuple):
    """Container for knapsack problem data (weights and values)."""

    weights: chex.Array  # Item weights, shape (n_items,)
    values: chex.Array  # Item values, shape (n_items,)


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
        n_items: Number of items in the knapsack problem.
        weights: Item weights array, shape (n_items,).
        values: Item values array, shape (n_items,).
        capacity: Maximum weight capacity (scalar).
        penalty_factor: Linear constraint penalty coefficient (default 1000.0).
    """

    n_items: int = struct.field(default=50)  # type: ignore[no-untyped-call]
    weights: chex.Array = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]
    values: chex.Array = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]
    capacity: chex.Numeric = struct.field(default=100.0)  # type: ignore[no-untyped-call]
    penalty_factor: float = 1000.0


@struct.dataclass
class KnapsackEvaluator(BaseEvaluator[BinaryGenome, KnapsackConfig, Optional[KnapsackData]]):
    """Knapsack problem fitness evaluator with linear constraint penalties.

    Computes total value minus linear penalty for weight constraint violation.
    Penalty = excess_weight * penalty_factor (jax.lax.select, XLA-safe).

    Data is stored as KnapsackData containing weights and values arrays.
    Use create_synthetic() to generate random problems or create_from_data()
    to load pre-computed weights/values (matching TSP interface).
    """

    config: KnapsackConfig
    data: Optional[KnapsackData] = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    def evaluate(self, genome: BinaryGenome) -> chex.Numeric:
        """Evaluate a binary genome representing item selection.

        The returned value equals total item value minus a linear penalty for
        any capacity violation. This method is JAX‑safe and avoids Python
        control flow.
        """
        weights = self.data.weights if self.data is not None else self.config.weights
        values = self.data.values if self.data is not None else self.config.values

        total_weight = jnp.sum(genome.values * weights)
        total_value = jnp.sum(genome.values * values)

        # Penalize infeasibility via JAX arithmetic (no Python control flow)
        excess_weight = jnp.maximum(0.0, total_weight - self.config.capacity)
        penalty = excess_weight * self.config.penalty_factor

        return total_value - penalty

    @classmethod
    def create_random_problem(
        cls,
        rng_key: chex.Array,
        n_items: int = 50,
        capacity_ratio: float = 0.5,
        maximize: bool = True,
    ) -> KnapsackConfig:
        """Create a random knapsack configuration with synthetic weights and values."""
        key1, key2 = jr.split(rng_key)

        weights = jr.uniform(key1, (n_items,), minval=1.0, maxval=20.0)
        values = jr.uniform(key2, (n_items,), minval=1.0, maxval=50.0)

        total_weight = jnp.sum(weights)
        capacity = capacity_ratio * total_weight

        return KnapsackConfig(
            n_items=n_items,
            weights=weights,
            values=values,
            capacity=capacity,
            maximize=maximize,
        )

    @classmethod
    def create_synthetic(
        cls,
        n_items: int = 50,
        capacity_ratio: float = 0.5,
        seed: int = 42,
        maximize: bool = True,
    ) -> "KnapsackEvaluator":
        """Factory: create random 0/1 knapsack instance with synthetic data.

        Matches TSP interface: generates random weights and values procedurally.
        """
        rng_key = jr.PRNGKey(seed)
        config = cls.create_random_problem(
            rng_key,
            n_items=n_items,
            capacity_ratio=capacity_ratio,
            maximize=maximize,
        )
        data = KnapsackData(weights=config.weights, values=config.values)

        return cls(config=config, data=data)

    @classmethod
    def create_from_data(
        cls,
        weights: chex.Array,
        values: chex.Array,
        capacity: chex.Numeric,
        penalty_factor: float = 1000.0,
        maximize: bool = True,
    ) -> "KnapsackEvaluator":
        """Factory: create knapsack evaluator from pre-loaded weights and values.

        Matches TSP interface: accepts pre-computed problem data.

        Args:
            weights: Item weights array, shape (n_items,)
            values: Item values array, shape (n_items,)
            capacity: Maximum weight capacity
            penalty_factor: Linear constraint penalty coefficient
            maximize: Whether to maximize (default: True for compatibility)

        Returns:
            Initialized KnapsackEvaluator with loaded data.
        """
        n_items = weights.shape[0]
        config = KnapsackConfig(
            n_items=n_items,
            capacity=capacity,
            penalty_factor=penalty_factor,
            maximize=maximize,
        )
        data = KnapsackData(weights=weights, values=values)

        return cls(config=config, data=data)
