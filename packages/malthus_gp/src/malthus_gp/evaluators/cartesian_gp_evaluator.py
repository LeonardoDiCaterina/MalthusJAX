"""Cartesian GP Evaluator.

Evaluates Cartesian genomes against target data arrays (X, y) for regression
and classification problems.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthus_gp.genomes.cartesian_genome import CartesianGenome, CartesianGenomeConfig
from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig, RegressionData
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.evaluators.cartesian_gp_evaluator")

_field: Any = struct.field


@struct.dataclass
class CartesianGPEvaluatorConfig(BaseEvaluatorConfig):  # type: ignore[misc]
    """Configuration for Cartesian GP Evaluator."""

    genome_config: CartesianGenomeConfig = _field(
        pytree_node=False, default_factory=CartesianGenomeConfig
    )
    loss_function: str = _field(pytree_node=False, default="mse")
    maximize: bool = _field(pytree_node=False, default=False)


@struct.dataclass
class CartesianGPEvaluator(
    BaseEvaluator[CartesianGenome, CartesianGPEvaluatorConfig, RegressionData]  # type: ignore[misc]
):
    """Evaluator for Cartesian GP genomes over supervised datasets.

    Supports MSE, MAE, Binary Cross Entropy (BCE), and Categorical Cross Entropy (CCE).
    """

    interpreter: CartesianInterpreter = _field(default_factory=CartesianInterpreter)

    def predict(self, genome: CartesianGenome, X: chex.Array) -> chex.Array:
        """Computes outputs for a full data matrix."""
        return self.interpreter.apply(genome, X)

    def evaluate(self, genome: CartesianGenome) -> chex.Numeric:
        """Computes loss over the entire dataset."""
        X, y = self.data
        preds = self.predict(genome, X)

        loss_type = self.config.loss_function.lower()

        if loss_type == "mse":
            y_bcast = jnp.reshape(y, preds.shape)
            loss = jnp.mean(jnp.square(preds - y_bcast))
        elif loss_type == "mae":
            y_bcast = jnp.reshape(y, preds.shape)
            loss = jnp.mean(jnp.abs(preds - y_bcast))
        elif loss_type == "bce":
            probs = jax.nn.sigmoid(preds)
            y_bcast = jnp.reshape(y, probs.shape)
            loss = -jnp.mean(
                y_bcast * jnp.log(probs + 1e-7) + (1.0 - y_bcast) * jnp.log(1.0 - probs + 1e-7)
            )
        elif loss_type == "cce":
            y_int = y.squeeze().astype(jnp.int32)
            y_onehot = jax.nn.one_hot(y_int, preds.shape[-1])
            loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(preds) * y_onehot, axis=-1))
        else:
            y_bcast = jnp.reshape(y, preds.shape)
            loss = jnp.mean(jnp.square(preds - y_bcast))

        # If maximizing, negate loss so higher is better
        return jnp.where(self.config.maximize, -loss, loss)
