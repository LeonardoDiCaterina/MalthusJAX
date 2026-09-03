"""Cartesian GP Evaluator.

Executes a Cartesian GP genome on input data. Unlike MEP (which uses symbiotic
selection to find the best node), CGP evaluates the entire graph and its fitness
is determined explicitly by the program's output nodes.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluatorConfig, RegressionData, StochasticEvaluator
from malthusjax.core.fitness.linear_gp_evaluator import TENSORGP_FUNCTIONS
from malthusjax.core.genome.cartesian_genome import CartesianGenome, CartesianGenomeConfig


@struct.dataclass
class CartesianGPEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Cartesian GP Evaluator."""

    genome_config: CartesianGenomeConfig = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]


@struct.dataclass
class CartesianGPEvaluator(
    StochasticEvaluator[CartesianGenome, CartesianGPEvaluatorConfig, RegressionData]
):
    """Evaluator for Cartesian GP genomes.

    Minimization convention: returns Mean Squared Error (MSE). Lower is better.
    """

    def predict_one(self, genome: CartesianGenome, x_input: chex.Array) -> chex.Array:
        """Execute the genome on a single input vector to produce output values."""
        N = x_input.shape[-1]
        L = genome.ops.shape[0]
        num_outs = genome.out_nodes.shape[0]

        total_mem = N + L
        memory = jnp.zeros(total_mem).at[:N].set(x_input)

        def step(current_mem: Any, inputs: Any) -> Any:
            mem, write_idx = current_mem
            op_code, arg_indices = inputs

            args_val = jnp.take(mem, arg_indices)

            arg0 = args_val[0] if args_val.shape[0] > 0 else 0.0
            arg1 = args_val[1] if args_val.shape[0] > 1 else 0.0
            arg2 = args_val[2] if args_val.shape[0] > 2 else 0.0

            result = jax.lax.switch(op_code, TENSORGP_FUNCTIONS, arg0, arg1, arg2)

            result = jnp.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

            new_mem = mem.at[write_idx].set(result)

            return (new_mem, write_idx + 1), result

        init_state = (memory, N)
        (final_mem, _), _ = jax.lax.scan(step, init_state, (genome.ops, genome.args))

        outputs = jnp.take(final_mem, genome.out_nodes)

        return outputs.reshape(num_outs)

    def evaluate(self, genome: CartesianGenome, rng: chex.PRNGKey | None = None) -> chex.Numeric:
        """Evaluate the genome over all data points."""
        X, y = self.data

        if self.config.batch_size is not None and rng is not None:
            indices = jax.random.choice(rng, X.shape[0], shape=(self.config.batch_size,), replace=False)
            X = X[indices]
            y = y[indices]

        all_preds = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)

        if self.config.loss_function == "cce":
            y_onehot = jax.nn.one_hot(y.squeeze(), all_preds.shape[-1])
            loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(all_preds) * y_onehot, axis=-1))
            return loss
        elif self.config.loss_function == "bce":
            probs = jax.nn.sigmoid(all_preds)
            y_bcast = y.reshape(probs.shape)
            loss = -jnp.mean(y_bcast * jnp.log(probs + 1e-7) + (1 - y_bcast) * jnp.log(1 - probs + 1e-7))
            return loss
        else: # mse
            if y.ndim == 1:
                y = y[:, None]
            squared_errors = jnp.square(all_preds - y)
            return jnp.mean(squared_errors)
