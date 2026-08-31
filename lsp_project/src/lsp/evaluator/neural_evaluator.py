"""LSP Neural Evaluator — forward pass for NeuralGenome.

Implements the CGPANN node equation as a single jax.lax.scan over the linear
chromosome:

    N_i = activation( sum_k( weights[i,k] * memory[args[i,k]] ) )

Each node produces a vector of shape (output_dim,).  Input slots are
initialised by broadcasting each scalar observation feature to output_dim.
"""

from __future__ import annotations

from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
from flax import struct
from lsp.genome.neural import ACTIVATIONS, NeuralGenome

from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def neural_forward_one(
    genome: NeuralGenome,
    x_input: chex.Array,
    *,
    num_inputs: int,
    output_dim: int,
    activation_fn: Callable[[chex.Array], chex.Array],
) -> chex.Array:
    """Single-pass linear-GP forward for NeuralGenome.

    Executes the chromosome top-down via ``jax.lax.scan``, maintaining a
    memory buffer of shape ``(num_inputs + L, output_dim)``.

    Input slots (indices 0..num_inputs-1) are initialised by broadcasting
    each scalar feature *x_input[j]* to a vector of shape ``(output_dim,)``:

        memory[j] = x_input[j] * ones(output_dim)

    Each intermediate row i computes:

        memory[num_inputs + i] = activation_fn(
            sum_k  weights[i, k] * memory[args[i, k]]
        )

    Args:
        genome:        NeuralGenome instance (single, unbatched).
        x_input:       Observation vector of shape ``(num_inputs,)``.
        num_inputs:    Number of observation features.
        output_dim:    Dimensionality of each node's output vector.
        activation_fn: Element-wise activation (e.g. ``jnp.tanh``).

    Returns:
        Array of shape ``(L, output_dim)`` — one output vector per row.
    """
    L: int = genome.ops.shape[0]
    total_mem = num_inputs + L

    # --- Initialise memory ---
    # Shape: (total_mem, output_dim)
    # Input slots: broadcast scalar x_j → (output_dim,)
    input_mem = jnp.outer(x_input, jnp.ones(output_dim))  # (num_inputs, output_dim)
    memory = jnp.zeros((total_mem, output_dim)).at[:num_inputs].set(input_mem)

    def step(
        carry: tuple[chex.Array, int], row_data: tuple[chex.Array, chex.Array, chex.Array]
    ) -> tuple[tuple[chex.Array, int], chex.Array]:
        mem, write_idx = carry
        args_row, weights_row, _op = row_data  # (max_arity,), (max_arity,), scalar

        # Gather argument vectors: shape (max_arity, output_dim)
        arg_vecs = jnp.take(mem, args_row, axis=0)

        # Weighted sum: sum_k w[k] * arg_vecs[k]  → (output_dim,)
        weighted_sum = jnp.einsum("k,kd->d", weights_row, arg_vecs)

        # Apply activation
        out = activation_fn(weighted_sum)
        out = jnp.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        new_mem = mem.at[write_idx].set(out)
        return (new_mem, write_idx + 1), out

    init_carry = (memory, num_inputs)
    xs = (genome.args, genome.weights, genome.ops)
    _, all_outputs = jax.lax.scan(step, init_carry, xs)

    return all_outputs  # (L, output_dim)


# ---------------------------------------------------------------------------
# NeuralEvaluator (deterministic — evaluates on a fixed dataset)
# ---------------------------------------------------------------------------


@struct.dataclass
class NeuralEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for NeuralEvaluator."""

    activation: str = struct.field(pytree_node=False, default="tanh")
    num_inputs: int = struct.field(pytree_node=False, default=4)
    output_dim: int = struct.field(pytree_node=False, default=2)


@struct.dataclass
class NeuralEvaluator(BaseEvaluator[NeuralGenome, NeuralEvaluatorConfig, Any]):
    """Evaluates NeuralGenome on a fixed regression / classification dataset.

    Fitness = negative MSE of the last row's output vs targets
    (or override ``evaluate`` for custom objectives).
    """

    def _activation(self) -> Callable[[chex.Array], chex.Array]:
        return ACTIVATIONS[self.config.activation]

    def evaluate(self, genome: NeuralGenome) -> chex.Numeric:
        """Default: negative MSE on self.data = (X, Y)."""
        X, Y = self.data  # (B, num_inputs), (B, output_dim)
        act = self._activation()

        def predict_one(x: chex.Array) -> chex.Array:
            outs = neural_forward_one(
                genome,
                x,
                num_inputs=self.config.num_inputs,
                output_dim=self.config.output_dim,
                activation_fn=act,
            )
            return outs[-1]  # last row: (output_dim,)

        preds = jax.vmap(predict_one)(X)  # (B, output_dim)
        mse = jnp.mean(jnp.square(preds - Y))
        return mse  # minimise


__all__ = ["neural_forward_one", "NeuralEvaluator", "NeuralEvaluatorConfig"]
