"""Differentiable Cartesian Genetic Programming Interpreter.

Executes weighted Cartesian DAGs (dCGP / dCGPANN) and provides exact analytical
derivatives (gradients, Jacobians, Hessians) with respect to input variables using
JAX automatic differentiation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthus_gp.core.functions import ACTIVATION_FUNCTIONS, ACTIVATION_NAMES
from malthus_gp.genomes.differentiable_cartesian_genome import DifferentiableCartesianGenome
from malthusjax.core.fitness.composable.base import BaseInterpreter
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.interpreters.differentiable_cartesian_interpreter")

_field: Any = struct.field


@struct.dataclass
class DifferentiableCartesianInterpreter(BaseInterpreter[DifferentiableCartesianGenome]):  # type: ignore[misc]
    """Interpreter for Differentiable Cartesian Genomes.

    Node equation:
        N_i = Activation_i( sum_k( w_{i,k} * N_{args[i,k]} ) + b_i )
    """

    num_inputs: int = _field(pytree_node=False, default=1)
    num_outputs: int = _field(pytree_node=False, default=1)
    activations: Tuple[Callable[[jax.Array], jax.Array], ...] = _field(
        pytree_node=False, default=ACTIVATION_FUNCTIONS
    )
    activation_names: Tuple[str, ...] = _field(pytree_node=False, default=ACTIVATION_NAMES)

    @property
    def num_params(self) -> int:
        return -1

    def evaluate_single(
        self, genome: DifferentiableCartesianGenome, x_single: chex.Array
    ) -> chex.Array:
        """Forward pass for a single 1D input vector."""
        n_in = self.num_inputs
        n_nodes = genome.ops.shape[0]
        total_mem = n_in + n_nodes

        memory = jnp.zeros(total_mem, dtype=jnp.float32)
        x_flat = jnp.atleast_1d(x_single)
        memory = memory.at[:n_in].set(x_flat[:n_in])

        def step(
            current_mem: chex.Array,
            node_data: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
        ) -> Tuple[chex.Array, None]:
            op_code, arg_indices, w, b, write_idx = node_data

            # Gather inputs
            inputs_gathered = current_mem[arg_indices]

            # Weighted sum plus bias
            weighted_sum = jnp.sum(w * inputs_gathered) + b

            # Activation
            val = jax.lax.switch(op_code, self.activations, weighted_sum)
            val = jnp.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6)

            new_mem = current_mem.at[write_idx].set(val)
            return new_mem, None

        write_indices = n_in + jnp.arange(n_nodes)
        node_inputs = (genome.ops, genome.args, genome.weights, genome.biases, write_indices)

        final_mem, _ = jax.lax.scan(step, memory, node_inputs)

        outputs = jnp.take(final_mem, genome.out_nodes)
        return cast(chex.Array, outputs.reshape(self.num_outputs))

    def apply(self, genome: DifferentiableCartesianGenome, inputs: chex.Array) -> chex.Array:
        """Executes the genome over single or batched inputs."""
        inputs_arr = jnp.asarray(inputs, dtype=jnp.float32)
        if inputs_arr.ndim <= 1:
            return self.evaluate_single(genome, inputs_arr)
        else:
            return jax.vmap(self.evaluate_single, in_axes=(None, 0))(genome, inputs_arr)

    def grad_x(
        self, genome: DifferentiableCartesianGenome, x: chex.Array, output_idx: int = 0
    ) -> chex.Array:
        """Computes analytical gradient dS/dx for a single input vector x."""

        def scalar_fn(inp: chex.Array) -> chex.Array:
            return self.evaluate_single(genome, inp)[output_idx]

        return cast(chex.Array, jax.grad(scalar_fn)(x))

    def jacobian_x(self, genome: DifferentiableCartesianGenome, x: chex.Array) -> chex.Array:
        """Computes full analytical Jacobian matrix (num_outputs, num_inputs) at x."""

        def multi_fn(inp: chex.Array) -> chex.Array:
            return self.evaluate_single(genome, inp)

        return cast(chex.Array, jax.jacfwd(multi_fn)(x))

    def hessian_x(
        self, genome: DifferentiableCartesianGenome, x: chex.Array, output_idx: int = 0
    ) -> chex.Array:
        """Computes analytical Hessian matrix d^2S/dx^2 at x."""

        def scalar_fn(inp: chex.Array) -> chex.Array:
            return self.evaluate_single(genome, inp)[output_idx]

        return cast(chex.Array, jax.hessian(scalar_fn)(x))

    def to_formula(
        self,
        genome: DifferentiableCartesianGenome,
        feature_names: Optional[Sequence[str]] = None,
        output_idx: int = 0,
    ) -> str:
        """Extracts the symbolic formula with embedded weights and biases."""
        n_in = self.num_inputs
        if feature_names is None:
            names = [f"x{i}" for i in range(n_in)]
        else:
            names = list(feature_names)[:n_in]

        ops = [int(op) for op in genome.ops]
        args = [[int(a) for a in arg_row] for arg_row in genome.args]
        weights = [[float(w) for w in w_row] for w_row in genome.weights]
        biases = [float(b) for b in genome.biases]
        out_node = int(genome.out_nodes[output_idx])

        memo: dict[int, str] = {i: names[i] for i in range(n_in)}

        def resolve(idx: int) -> str:
            if idx in memo:
                return memo[idx]
            node_rel = idx - n_in
            op_code = ops[node_rel]
            arg_idxs = args[node_rel]
            w_vals = weights[node_rel]
            b_val = biases[node_rel]

            terms = []
            for arg_i, w_i in zip(arg_idxs, w_vals):
                s_i = resolve(arg_i)
                terms.append(f"({w_i:.3f} * {s_i})")

            inner_expr = " + ".join(terms)
            if abs(b_val) > 1e-4:
                inner_expr += f" + {b_val:.3f}" if b_val > 0 else f" - {abs(b_val):.3f}"

            op_name = self.activation_names[op_code % len(self.activation_names)]
            if op_name == "linear":
                expr = f"({inner_expr})"
            else:
                expr = f"{op_name}({inner_expr})"

            memo[idx] = expr
            return expr

        return resolve(out_node)
