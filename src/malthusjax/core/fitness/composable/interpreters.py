"""Concrete Interpreter implementations for native MalthusJAX genomes.

Provides:
- ``IdentityInterpreter``: genome IS the solution (OptimizationTask).
- ``MLPInterpreter``: flat RealGenome → MLP weights → apply(inputs) (SL/RL).
"""

from __future__ import annotations

import math
from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.real_genome import RealGenome
from malthusjax.core.genome.linear_genome import LinearGenome
from malthusjax.core.fitness.composable.base import BaseInterpreter


# =============================================================================
# IdentityInterpreter
# =============================================================================

@struct.dataclass
class IdentityInterpreter(BaseInterpreter[RealGenome]):
    """Passes genome values directly as the solution.

    Used for ``OptimizationTask`` environments (BBOB, TSP, Knapsack, etc.)
    where the genome encodes the candidate solution directly — no neural
    network inference or program execution is involved.

    The ``inputs`` argument is ignored; this interpreter is not meaningful
    for RL or Supervised tasks which require a predictor/policy.

    ``num_params`` returns -1 because the genome length is determined by
    the problem instance (e.g., number of cities for TSP), not by this
    interpreter's architecture.
    """

    @property
    def num_params(self) -> int:
        """Returns -1: genome length is problem-determined, not architecture-determined."""
        return -1

    def apply(self, genome: RealGenome, inputs: chex.Array | None = None) -> chex.Array:
        """Return the raw genome values as the solution vector.

        Args:
            genome: A single (unbatched) RealGenome.
            inputs: Ignored. Provided only for interface compatibility.

        Returns:
            ``genome.values`` — the raw solution vector.
        """
        return genome.values


# =============================================================================
# MLPInterpreter
# =============================================================================

@struct.dataclass
class MLPInterpreter(BaseInterpreter[RealGenome]):
    """Decodes a flat RealGenome into MLP weights and applies them to inputs.

    The MLP architecture is fully specified at construction time (stateless).
    All dimensions are compiled-in as ``pytree_node=False`` fields to ensure
    static shapes for JAX tracing.

    Self-consistency constraint::

        interpreter = MLPInterpreter(input_dim=4, output_dim=2, hidden=(64, 64))
        # interpreter.num_params == 4*64 + 64 + 64*2 + 2 == 450
        # genome_spec must have length == 450

    The Composer validates this at build time and raises ``ConfigurationError``
    loudly on mismatch — before any JAX tracing begins.

    Args:
        input_dim: Number of input features / observation dimensions.
        output_dim: Number of outputs (action dims for RL, predictions for SL).
        hidden: Tuple of hidden layer sizes, e.g. ``(64, 64)``.
        activation: Activation function name. Supported: ``"tanh"``, ``"relu"``,
            ``"sigmoid"``. Default: ``"tanh"``.
    """

    input_dim: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    output_dim: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    hidden: tuple = struct.field(pytree_node=False, default=())  # type: ignore[no-untyped-call]
    activation: str = struct.field(pytree_node=False, default="tanh")  # type: ignore[no-untyped-call]

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """Full layer size sequence: (input_dim, *hidden, output_dim)."""
        return (self.input_dim,) + tuple(self.hidden) + (self.output_dim,)

    @property
    def num_params(self) -> int:
        """Total number of parameters (weights + biases) across all layers.

        This is the required genome length. The Composer validates that
        ``genome_spec.length == interpreter.num_params`` at build time.
        """
        sizes = self.layer_sizes
        return sum(
            sizes[i] * sizes[i + 1] + sizes[i + 1]
            for i in range(len(sizes) - 1)
        )

    def _get_activation_fn(self):
        """Return the JAX activation function for the configured name."""
        if self.activation == "tanh":
            return jnp.tanh
        elif self.activation == "relu":
            return jax.nn.relu
        elif self.activation == "sigmoid":
            return jax.nn.sigmoid
        else:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                "Supported: 'tanh', 'relu', 'sigmoid'."
            )

    def _unflatten(self, flat_params: chex.Array) -> list[tuple[chex.Array, chex.Array]]:
        """Split a flat parameter vector into (W, b) pairs per layer.
        
        Note: Extracts bias before weights to match Flax's alphabetical
        tree_flatten order ('bias', 'kernel'). This ensures mathematical
        parity with legacy Flax-based evaluators.

        Args:
            flat_params: 1D array of length ``num_params``.

        Returns:
            List of (W, b) tuples, one per layer transition.
        """
        sizes = self.layer_sizes
        params = []
        idx = 0
        for i in range(len(sizes) - 1):
            w_size = sizes[i] * sizes[i + 1]
            b_size = sizes[i + 1]
            b = flat_params[idx : idx + b_size]
            W = flat_params[idx + b_size : idx + b_size + w_size].reshape(sizes[i], sizes[i + 1])
            params.append((W, b))
            idx += w_size + b_size
        return params

    def apply(self, genome: RealGenome, inputs: chex.Array) -> chex.Array:
        """Apply the MLP encoded in the genome to the given inputs.

        Unfolds the genome's flat parameter vector into weight matrices and
        bias vectors, then performs a forward pass through the MLP.

        Args:
            genome: A single (unbatched) RealGenome of length ``num_params``.
            inputs: Input array of shape ``(input_dim,)``.

        Returns:
            Output array of shape ``(output_dim,)``.
        """
        params = self._unflatten(genome.values)
        act_fn = self._get_activation_fn()
        x = inputs
        for i, (W, b) in enumerate(params):
            x = x @ W + b
            # Apply activation to all layers except the last
            if i < len(params) - 1:
                x = act_fn(x)
        return x


# =============================================================================
# LinearGPInterpreter
# =============================================================================

@struct.dataclass
class LinearGPInterpreter(BaseInterpreter[LinearGenome]):
    """Interpreter for Linear Genetic Programming genomes.
    
    Translates a sequence of operation codes and argument indices into
    intermediate outputs (symbiotic selection).
    """

    num_inputs: int = struct.field(pytree_node=False, default=10)  # type: ignore[no-untyped-call]
    length: int = struct.field(pytree_node=False, default=100)  # type: ignore[no-untyped-call]

    @property
    def num_params(self) -> int:
        return -1

    def apply(self, genome: LinearGenome, inputs: chex.Array | None = None) -> chex.Array:
        from malthusjax.core.fitness.linear_gp_evaluator import TENSORGP_FUNCTIONS
        
        total_mem = self.num_inputs + self.length
        # Note: inputs might be batched (or singular from vmap). Assumes inputs is 1D inside apply.
        memory = jnp.zeros(total_mem).at[: self.num_inputs].set(inputs)

        def step(current_mem: Any, step_inputs: Any) -> Any:
            mem, write_idx = current_mem
            op_code, arg_indices = step_inputs

            args_val = jnp.take(mem, arg_indices)
            result = jax.lax.switch(
                op_code, TENSORGP_FUNCTIONS, args_val[0], args_val[1], args_val[2]
            )
            result = jnp.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            new_mem = mem.at[write_idx].set(result)

            return (new_mem, write_idx + 1), result

        init_state = (memory, self.num_inputs)
        _, instruction_outputs = jax.lax.scan(step, init_state, (genome.ops, genome.args))
        return instruction_outputs
