"""Cartesian Genetic Programming Quality-Diversity Evaluator.

Evaluates supervised loss alongside behavioral descriptors (e.g. active graph complexity
and semantic prediction variance) to drive MAP-Elites repertoire exploration.
"""

from __future__ import annotations

from typing import Any, Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp

from malthus_gp.genomes.cartesian_genome import CartesianGenome, CartesianGenomeConfig
from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter
from malthusjax.core.fitness.base import BaseEvaluatorConfig, RegressionData
from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.evaluators.cartesian_qd_evaluator")

_field: Any = struct.field


@struct.dataclass
class CartesianQDEvalConfig(BaseEvaluatorConfig):  # type: ignore[misc]
    """Configuration for Cartesian GP Quality-Diversity Evaluator."""

    genome_config: CartesianGenomeConfig = _field(
        pytree_node=False, default_factory=CartesianGenomeConfig
    )
    loss_function: str = _field(pytree_node=False, default="mse")
    maximize: bool = _field(pytree_node=False, default=False)
    descriptor_type: str = _field(pytree_node=False, default="complexity_variance")
    num_descriptors: int = _field(pytree_node=False, default=2)


@struct.dataclass
class CartesianQDEvaluator(
    BaseQDEvaluator[CartesianGenome, CartesianQDEvalConfig, RegressionData]  # type: ignore[misc]
):
    """Quality-Diversity Evaluator for Cartesian GP.

    Computes:
    1. Fitness: Supervised loss (MSE, MAE, BCE, CCE) between predictions and targets.
    2. Behavioral Descriptors (BD):
       - BD 0: Active graph complexity (fraction of total nodes participating in outputs).
       - BD 1: Prediction variance (tanh of output standard deviation across inputs).
    """

    interpreter: CartesianInterpreter = _field(default_factory=CartesianInterpreter)

    def predict(self, genome: CartesianGenome, X: chex.Array) -> chex.Array:
        """Computes predictions for a dataset matrix X."""
        return self.interpreter.apply(genome, X)

    def evaluate_qd(self, genome: CartesianGenome) -> Tuple[chex.Numeric, chex.Array]:
        """Evaluates both scalar fitness and multi-dimensional behavioral descriptors."""
        X, y = self.data
        preds = self.predict(genome, X)

        # 1. Supervised Loss Computation
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
            raise ValueError(f"Unsupported loss function: {loss_type}")

        # 2. Behavioral Descriptors
        # BD 0: Active Graph Complexity in [0, 1]
        active_mask = genome.get_active_mask(self.interpreter.num_inputs)
        active_nodes = jnp.sum(active_mask[self.interpreter.num_inputs :])
        total_nodes = jnp.maximum(1, genome.ops.shape[0])
        bd0 = jnp.clip(active_nodes / total_nodes, 0.0, 1.0)

        # BD 1: Semantic Prediction Spread in [0, 1]
        pred_std = jnp.std(preds)
        bd1 = jnp.clip(jnp.tanh(pred_std), 0.0, 1.0)

        descriptors = jnp.array([bd0, bd1], dtype=jnp.float32)
        return loss, descriptors


__all__ = ["CartesianQDEvaluator", "CartesianQDEvalConfig"]
