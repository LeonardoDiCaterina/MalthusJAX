"""Supervised loss evaluator for NeatGenome.

Computes MSE, MAE, Binary Cross-Entropy, or Categorical Cross-Entropy across
supervised tabular or sequence data splits (X, y) in pure XLA kernels.
"""

from __future__ import annotations

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthus_neat.genomes.neat_genome import NeatGenome, NeatGenomeConfig
from malthus_neat.interpreters.neat_interpreter import NeatInterpreter
from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_neat.evaluators.neat_evaluator")

_field: Any = struct.field
RegressionData = Tuple[chex.Array, chex.Array]


@struct.dataclass
class NeatEvaluatorConfig(BaseEvaluatorConfig):  # type: ignore[misc]
    """Configuration for NEAT supervised evaluator."""

    genome_config: NeatGenomeConfig = _field(pytree_node=False, default_factory=NeatGenomeConfig)
    loss_function: str = _field(pytree_node=False, default="mse")
    maximize: bool = _field(pytree_node=False, default=False)


@struct.dataclass
class NeatEvaluator(
    BaseEvaluator[NeatGenome, NeatEvaluatorConfig, RegressionData]  # type: ignore[misc]
):
    """Evaluates supervised regression or classification loss on (X, y) data."""

    interpreter: NeatInterpreter = _field(default_factory=NeatInterpreter)

    def predict(self, genome: NeatGenome, X: chex.Array) -> chex.Array:
        """Evaluates model predictions on input features X."""
        return self.interpreter.evaluate_batch(genome, X)

    def evaluate(self, genome: NeatGenome) -> chex.Numeric:
        """Computes scalar loss over the dataset."""
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

        # Replace NaN with high loss
        loss = jnp.where(jnp.isnan(loss), 1e6, loss)
        return jnp.where(self.config.maximize, -loss, loss)

    def evaluate_population(
        self, population: BasePopulation[NeatGenome]
    ) -> BasePopulation[NeatGenome]:
        """Evaluates an entire population in a single fused vmap pass."""
        X, y = self.data
        pop_preds = self.interpreter.evaluate_population(population.genes, X)

        loss_type = self.config.loss_function.lower()

        def _calc_loss(preds: chex.Array) -> chex.Numeric:
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

            loss = jnp.where(jnp.isnan(loss), 1e6, loss)
            return jnp.where(self.config.maximize, -loss, loss)

        losses = jax.vmap(_calc_loss)(pop_preds)
        return population.replace(fitness=losses)


__all__ = ["NeatEvaluator", "NeatEvaluatorConfig"]
