"""Correlation and Diversity Evaluator for Automated Feature Engineering.

Evaluates candidate multi-output Cartesian GP genomes by:
1. Maximizing absolute Pearson correlation between each engineered feature and the target y.
2. Penalizing pairwise cross-correlation (redundancy) among the engineered features.
"""

from __future__ import annotations

from typing import Any

import chex
import jax.numpy as jnp
from flax import struct

from malthus_gp.genomes.cartesian_genome import CartesianGenome, CartesianGenomeConfig
from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig, RegressionData
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.evaluators.correlation_evaluator")

_field: Any = struct.field


@struct.dataclass
class CorrelationEvaluatorConfig(BaseEvaluatorConfig):  # type: ignore[misc]
    """Configuration for Correlation Evaluator."""

    genome_config: CartesianGenomeConfig = _field(
        pytree_node=False, default_factory=CartesianGenomeConfig
    )
    redundancy_weight: float = _field(pytree_node=False, default=0.5)
    maximize: bool = _field(pytree_node=False, default=False)


@struct.dataclass
class CorrelationEvaluator(
    BaseEvaluator[CartesianGenome, CorrelationEvaluatorConfig, RegressionData]  # type: ignore[misc]
):
    """Evaluates multi-output Cartesian genomes for feature engineering."""

    interpreter: CartesianInterpreter = _field(default_factory=CartesianInterpreter)

    def predict(self, genome: CartesianGenome, X: chex.Array) -> chex.Array:
        """Computes outputs for a full data matrix."""
        return self.interpreter.apply(genome, X)

    def evaluate(self, genome: CartesianGenome) -> chex.Numeric:
        """Computes correlation fitness with redundancy penalty."""
        X, y = self.data
        preds = self.predict(genome, X)  # shape (N, K)

        N = preds.shape[0]
        K = preds.shape[1] if preds.ndim > 1 else 1

        preds_2d = preds.reshape(N, K)
        y_flat = y.flatten()

        # Center and standardize predictions
        preds_mean = jnp.mean(preds_2d, axis=0, keepdims=True)
        preds_std = jnp.std(preds_2d, axis=0, keepdims=True) + 1e-6
        preds_norm = (preds_2d - preds_mean) / preds_std

        # Center and standardize target
        y_mean = jnp.mean(y_flat)
        y_std = jnp.std(y_flat) + 1e-6
        y_norm = (y_flat - y_mean) / y_std

        # Pearson correlation vector with target y (shape: (K,))
        corr_with_y = jnp.matmul(preds_norm.T, y_norm) / float(N)
        mean_abs_corr = jnp.mean(jnp.abs(corr_with_y))

        # Pairwise feature cross-correlation matrix (shape: (K, K))
        cross_corr = jnp.matmul(preds_norm.T, preds_norm) / float(N)
        abs_cross_corr = jnp.abs(cross_corr)

        # Off-diagonal redundancy penalty
        if K > 1:
            off_diag_sum = jnp.sum(abs_cross_corr) - jnp.trace(abs_cross_corr)
            redundancy_penalty = off_diag_sum / float(K * (K - 1))
        else:
            redundancy_penalty = jnp.array(0.0)

        # Score to maximize: correlation minus redundancy penalty
        score = mean_abs_corr - self.config.redundancy_weight * redundancy_penalty

        # Minimize loss: -score
        loss = -score
        loss = jnp.nan_to_num(loss, nan=1e5, posinf=1e5, neginf=-1e5)

        return jnp.where(self.config.maximize, score, loss)
