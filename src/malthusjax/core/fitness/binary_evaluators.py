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
    """BinarySum (OneMax) fitness evaluator."""

    config: BinarySumConfig
    data: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    def evaluate(self, genome: BinaryGenome) -> chex.Numeric:
        """Evaluate a single binary genome by counting set bits."""
        ones_count = jnp.sum(genome.bits)
        zeros_count = genome.size - ones_count
        return jax.lax.select(self.config.maximize, ones_count, zeros_count)


@struct.dataclass
class KnapsackConfig(BaseEvaluatorConfig):
    """Configuration for 0/1 Knapsack problem fitness evaluator."""
    weights: chex.Array   # Item weights, shape (n_items,)
    values: chex.Array    # Item values, shape (n_items,)
    capacity: chex.Numeric # Maximum weight capacity
    penalty_factor: float = 1000.0  # Penalty for exceeding capacity


@struct.dataclass
class KnapsackEvaluator(BaseEvaluator[BinaryGenome, KnapsackConfig, Any]):
    """Knapsack problem fitness evaluator with linear constraint penalties."""

    config: KnapsackConfig
    data: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    def evaluate(self, genome: BinaryGenome) -> chex.Numeric:
        """Evaluate a binary genome representing item selection."""
        # Calculate total weight and value using vector dot products
        total_weight = jnp.sum(genome.bits * self.config.weights)
        total_value = jnp.sum(genome.bits * self.config.values)

        # Apply linear penalty for exceeding capacity
        excess_weight = jnp.maximum(0.0, total_weight - self.config.capacity)
        penalty = excess_weight * self.config.penalty_factor

        # Return total value minus penalty
        # (In maximization, penalized infeasible solutions score lower)
        return total_value - penalty

    @staticmethod
    def create_random_problem(key: chex.PRNGKey, n_items: int,
                            capacity_ratio: float = 0.5, maximize: bool = True) -> KnapsackConfig:
        """Create a random knapsack problem instance."""
        key1, key2 = jr.split(key, 2)

        # Random weights and values
        weights = jr.uniform(key1, (n_items,), minval=1.0, maxval=20.0)
        values = jr.uniform(key2, (n_items,), minval=1.0, maxval=50.0)

        # capacity as fraction of total weight
        total_weight = jnp.sum(weights)
        capacity = capacity_ratio * total_weight

        return KnapsackConfig(
            maximize=maximize,
            weights=weights,
            values=values,
            capacity=capacity
        )