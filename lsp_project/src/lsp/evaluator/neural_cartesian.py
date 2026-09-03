"""Neural Cartesian GP Evaluator.

Evaluates a NeuralCartesianGenome over a dataset, computing the forward pass
using the continuous weights, biases, and dynamic activation functions.
"""


import chex
import jax
import jax.numpy as jnp
from flax import struct
from lsp.genome.neural_cartesian import (
    ACTIVATIONS_LIST,
    NeuralCartesianGenome,
    NeuralCartesianGenomeConfig,
)

from malthusjax.core.fitness.base import BaseEvaluatorConfig, StochasticEvaluator


@struct.dataclass
class NeuralCartesianEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for NeuralCartesianEvaluator."""
    genome_config: NeuralCartesianGenomeConfig = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]


@struct.dataclass
class NeuralCartesianEvaluator(StochasticEvaluator[NeuralCartesianGenome, NeuralCartesianEvaluatorConfig, tuple[chex.Array, chex.Array]]):
    """Evaluates a NeuralCartesianGenome on a dataset."""

    def evaluate(
        self, genome: NeuralCartesianGenome, rng: chex.PRNGKey | None = None
    ) -> chex.Array:
        """Evaluate the neural genome over the dataset."""
        X, y = self.data
        N = X.shape[-1]
        num_nodes = genome.ops.shape[0]

        # Vectorize the forward pass over the batch
        def predict_single(x: chex.Array) -> chex.Array:
            memory = jnp.zeros((N + num_nodes,))
            memory = memory.at[:N].set(x)

            def eval_node(mem: chex.Array, idx: int) -> tuple[chex.Array, None]:
                node_idx = idx - N

                node_args = genome.args[node_idx]
                gathered_inputs = mem[node_args]

                w = genome.weights[node_idx]
                b = genome.biases[node_idx]

                weighted_sum = jnp.sum(w * gathered_inputs) + b

                op_idx = genome.ops[node_idx]
                out_val = jax.lax.switch(op_idx, ACTIVATIONS_LIST, weighted_sum)

                mem = mem.at[idx].set(out_val)
                return mem, None

            final_mem, _ = jax.lax.scan(
                eval_node,
                memory,
                jnp.arange(N, N + num_nodes),
            )

            return final_mem[genome.out_nodes]

        X, y = self.data
        if self.config.batch_size is not None and rng is not None:
            indices = jax.random.choice(rng, X.shape[0], shape=(self.config.batch_size,), replace=False)
            X = X[indices]
            y = y[indices]

        # Vectorize across the batch
        logits = jax.vmap(predict_single)(X)

        # Dynamic Loss
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

__all__ = ["NeuralCartesianEvaluatorConfig", "NeuralCartesianEvaluator"]
