"""Evaluator for Differentiable MEP (dMEP).

Evaluates the linear genome by treating it as an ANN, where each node
computes an activation function over the weighted sum of its inputs,
plus a bias. Returns all intermediate states (shape: (L,)), so that
the fitness function can pick the minimum (and naturally route gradients
through the argmin!).
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct
from lsp.genome.neural_cartesian import ACTIVATIONS_LIST
from lsp.genome.neural_linear import NeuralPrefixGenome

from malthusjax.core.fitness.base import BaseEvaluatorConfig, StochasticEvaluator


@struct.dataclass
class NeuralPrefixEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Neural Prefix Evaluator."""

    num_inputs: int = struct.field(pytree_node=False, default=10)
    length: int = struct.field(pytree_node=False, default=100)
    # Note: batch_size, loss_function are inherited from BaseEvaluatorConfig


@struct.dataclass
class NeuralPrefixEvaluator(StochasticEvaluator[NeuralPrefixGenome, NeuralPrefixEvaluatorConfig, Any]):
    """Evaluates Neural Prefix Genomes (dMEP)."""

    def predict_one(self, genome: NeuralPrefixGenome, x_input: chex.Array) -> chex.Array:
        """Forward pass for a single input pattern.

        Evaluates the sequence of instructions, computing a weighted sum and activation.
        Returns the output values for ALL instructions (shape: (L,)).
        """
        N = self.config.num_inputs
        L = self.config.length

        # memory size = N (inputs) + L (intermediate instruction results)
        total_mem = N + L
        memory = jnp.zeros(total_mem)
        memory = memory.at[:N].set(x_input)

        def step(current_mem: chex.Array, idx: int) -> tuple[chex.Array, chex.Array]:
            node_idx = idx - N

            # Extract discrete topology
            op_idx = genome.ops[node_idx]
            arg_indices = genome.args[node_idx]

            # Extract continuous parameters
            w = genome.weights[node_idx]
            b = genome.biases[node_idx]

            # Gather input values from memory based on discrete args
            gathered_inputs = current_mem[arg_indices]

            # 1. Compute Linear Combination: z = sum(w * x) + b
            weighted_sum = jnp.sum(w * gathered_inputs) + b

            # 2. Compute Activation Function: a = act(z)
            out_val = jax.lax.switch(op_idx, ACTIVATIONS_LIST, weighted_sum)

            # Update memory
            new_mem = current_mem.at[idx].set(out_val)

            return new_mem, out_val

        # Execute scan from N to N+L
        final_mem, instruction_outputs = jax.lax.scan(
            step, memory, jnp.arange(N, N + L)
        )

        return instruction_outputs

    def evaluate(self, genome: NeuralPrefixGenome, rng: chex.PRNGKey | None = None) -> chex.Array:
        """Evaluates fitness with batching and dynamic output routing."""
        X, y = self.data

        if self.config.batch_size is not None and rng is not None:
            indices = jax.random.choice(rng, X.shape[0], shape=(self.config.batch_size,), replace=False)
            X = X[indices]
            y = y[indices]

        # all_preds shape: (Batch, L)
        all_preds = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)

        # Route predictions based on strategy
        if getattr(genome, "readout_weights", None) is not None:
            # all_preds is (Batch, L), readout_weights is (L, K), readout_biases is (K,)
            # logits shape: (Batch, K)
            logits = jnp.dot(all_preds, genome.readout_weights) + genome.readout_biases

            if self.config.loss_function == "cce":
                # y should be (Batch, K) one-hot or (Batch,) integers
                # For simplicity, assuming y is (Batch,) integers
                y_onehot = jax.nn.one_hot(y.squeeze(), logits.shape[-1])
                loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * y_onehot, axis=-1))
                return loss
            elif self.config.loss_function == "bce":
                probs = jax.nn.sigmoid(logits)
                y_bcast = y.reshape(probs.shape)
                loss = -jnp.mean(y_bcast * jnp.log(probs + 1e-7) + (1 - y_bcast) * jnp.log(1 - probs + 1e-7))
                return loss
            else: # mse
                y_bcast = y.reshape(logits.shape)
                return jnp.mean(jnp.square(logits - y_bcast))

        elif getattr(genome, "out_nodes", None) is not None: # Explicit out_nodes
            # out_nodes is (K,), all_preds is (Batch, L)
            # logits shape: (Batch, K)
            logits = all_preds[:, genome.out_nodes]

            if self.config.loss_function == "cce":
                y_onehot = jax.nn.one_hot(y.squeeze(), logits.shape[-1])
                loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * y_onehot, axis=-1))
                return loss
            elif self.config.loss_function == "bce":
                probs = jax.nn.sigmoid(logits)
                y_bcast = y.reshape(probs.shape)
                loss = -jnp.mean(y_bcast * jnp.log(probs + 1e-7) + (1 - y_bcast) * jnp.log(1 - probs + 1e-7))
                return loss
            else: # mse
                y_bcast = y.reshape(logits.shape)
                return jnp.mean(jnp.square(logits - y_bcast))

        else: # Dynamic routing (Standard MEP)
            # Find the instruction that minimizes loss across the batch
            if self.config.loss_function == "mse":
                y_bcast = y.reshape(-1, 1)
                squared_errors = jnp.square(all_preds - y_bcast)
                loss_per_tree = jnp.mean(squared_errors, axis=0)
                return jnp.min(loss_per_tree)
            elif self.config.loss_function == "bce":
                probs = jax.nn.sigmoid(all_preds)
                y_bcast = y.reshape(-1, 1)
                bce = -(y_bcast * jnp.log(probs + 1e-7) + (1 - y_bcast) * jnp.log(1 - probs + 1e-7))
                loss_per_tree = jnp.mean(bce, axis=0)
                return jnp.min(loss_per_tree)
            else:
                raise ValueError("Dynamic routing does not support CCE directly.")
