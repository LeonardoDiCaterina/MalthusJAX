"""Quality-Diversity evaluator for NEAT neural networks.

Evaluates supervised loss and computes 2D behavioral descriptors:
- BD 0: Connection Density (ratio of active synapses to max_conns in [0, 1]).
- BD 1: Prediction Spread (tanh of prediction standard deviation in [0, 1]).
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
from malthusjax.core.fitness.base import BaseEvaluatorConfig
from malthusjax.core.fitness.qd.evaluator import BaseQDEvaluator
from malthusjax.core.genome.qd.population import QDPopulation
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_neat.evaluators.neat_qd_evaluator")

_field: Any = struct.field
RegressionData = Tuple[chex.Array, chex.Array]


@struct.dataclass
class NeatQDEvalConfig(BaseEvaluatorConfig):  # type: ignore[misc]
    """Configuration for NEAT Quality-Diversity evaluator."""

    genome_config: NeatGenomeConfig = _field(pytree_node=False, default_factory=NeatGenomeConfig)
    loss_function: str = _field(pytree_node=False, default="mse")
    maximize: bool = _field(pytree_node=False, default=False)
    descriptor_type: str = _field(pytree_node=False, default="density_variance")
    num_descriptors: int = _field(pytree_node=False, default=2)


@struct.dataclass
class NeatQDEvaluator(
    BaseQDEvaluator[NeatGenome, NeatQDEvalConfig, RegressionData]  # type: ignore[misc]
):
    """Quality-Diversity Evaluator for NEAT networks."""

    interpreter: NeatInterpreter = _field(default_factory=NeatInterpreter)

    def predict(self, genome: NeatGenome, X: chex.Array) -> chex.Array:
        return self.interpreter.evaluate_batch(genome, X)

    def evaluate_qd(self, genome: NeatGenome) -> Tuple[chex.Numeric, chex.Array]:
        """Evaluates loss and 2D behavioral descriptors for a single genome."""
        X, y = self.data
        preds = self.predict(genome, X)

        # 1. Loss calculation
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

        loss = jnp.where(jnp.isnan(loss), 1e6, loss)

        # 2. Behavioral Descriptors
        # BD 0: Connection Density in [0, 1]
        active_conns = jnp.sum(~jnp.isnan(genome.conns[:, 0]))
        max_conns = genome.conns.shape[0]
        bd0 = jnp.clip(active_conns / jnp.maximum(max_conns, 1), 0.0, 1.0)

        # BD 1: Prediction Output Variance in [0, 1]
        pred_std = jnp.std(preds)
        bd1 = jnp.clip(jnp.tanh(jnp.nan_to_num(pred_std, nan=0.0)), 0.0, 1.0)

        descriptors = jnp.array([bd0, bd1], dtype=jnp.float32)
        score = jnp.where(self.config.maximize, -loss, loss)
        return score, descriptors

    def evaluate_population(
        self, population: BasePopulation[NeatGenome]
    ) -> QDPopulation[NeatGenome]:
        """Evaluates population loss and descriptors in a vectorized batch."""
        X, y = self.data
        pop_preds = self.interpreter.evaluate_population(population.genes, X)

        loss_type = self.config.loss_function.lower()

        def _calc_metrics(preds: chex.Array, conns: chex.Array) -> Tuple[chex.Numeric, chex.Array]:
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

            # BD 0
            act_conns = jnp.sum(~jnp.isnan(conns[:, 0]))
            bd0 = jnp.clip(act_conns / jnp.maximum(conns.shape[0], 1), 0.0, 1.0)

            # BD 1
            pred_std = jnp.std(preds)
            bd1 = jnp.clip(jnp.tanh(jnp.nan_to_num(pred_std, nan=0.0)), 0.0, 1.0)

            descriptors = jnp.array([bd0, bd1], dtype=jnp.float32)
            score = jnp.where(self.config.maximize, -loss, loss)
            return score, descriptors

        scores, descriptors = jax.vmap(_calc_metrics)(pop_preds, population.genes.conns)

        new_info = dict(population.info) if population.info else {}
        new_info["descriptors"] = descriptors

        return QDPopulation(
            genes=population.genes,
            fitness=scores,
            config=population.config,
            info=new_info,
        )


__all__ = ["NeatQDEvaluator", "NeatQDEvalConfig"]
