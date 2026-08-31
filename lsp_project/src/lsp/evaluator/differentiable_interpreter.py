"""Differentiable Linear GP Interpreter."""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from lsp.evaluator.operator_set import DEFAULT_DIFFERENTIABLE_OPS, OperatorSet

from malthusjax.core.genome.linear_genome import LinearGenome


def predict_one_jacobian(
    genome: LinearGenome,
    x_input: chex.Array,
    *,
    num_inputs: int,
    length: int,
    operator_set: OperatorSet = DEFAULT_DIFFERENTIABLE_OPS,
) -> tuple[chex.Array, chex.Array]:
    """Execute one genome with forward-mode AD, returning values and Jacobians.

    Returns:
        values: 1-D array of shape (L,) containing instruction outputs.
        jacobians: 2-D array of shape (L, N) containing Jacobians wrt inputs.
    """

    # Check if genome has constants and append them to x_input
    if hasattr(genome, "constants"):
        x_input = jnp.concatenate([x_input, genome.constants])
        actual_inputs = num_inputs + genome.constants.shape[0]
    else:
        actual_inputs = num_inputs

    total_mem = actual_inputs + length

    # Initialize values and gradients
    # memory_val shape: (total_mem,)
    # memory_grad shape: (total_mem, actual_inputs)
    memory_val = jnp.zeros(total_mem).at[:actual_inputs].set(x_input)

    # The gradient of the initial inputs w.r.t themselves is the identity matrix
    init_grads = jnp.eye(actual_inputs)
    memory_grad = jnp.zeros((total_mem, actual_inputs)).at[:actual_inputs, :].set(init_grads)

    def step(current_mem: Any, inputs: Any) -> Any:
        mem_v, mem_g, write_idx = current_mem
        op_code, arg_indices = inputs

        # Fetch values and gradients for the 3 arguments
        args_val = jnp.take(mem_v, arg_indices)  # (3,)
        args_grad = jnp.take(mem_g, arg_indices, axis=0)  # (3, actual_inputs)

        v1, v2, v3 = args_val[0], args_val[1], args_val[2]
        g1, g2, g3 = args_grad[0], args_grad[1], args_grad[2]

        # Forward pass
        result_v = jax.lax.switch(op_code, operator_set.forward_fns, v1, v2, v3)
        result_v = jnp.nan_to_num(result_v, nan=0.0, posinf=1e6, neginf=-1e6)

        # Compute partial derivatives
        dx1 = jax.lax.switch(op_code, operator_set.dx1_fns, v1, v2, v3)
        dx2 = jax.lax.switch(op_code, operator_set.dx2_fns, v1, v2, v3)
        dx3 = jax.lax.switch(op_code, operator_set.dx3_fns, v1, v2, v3)

        # Chain rule: dz/dx = (dz/dv1 * dv1/dx) + (dz/dv2 * dv2/dx) + (dz/dv3 * dv3/dx)
        # Note: g1, g2, g3 are vectors of shape (actual_inputs,)
        result_g = dx1 * g1 + dx2 * g2 + dx3 * g3
        result_g = jnp.nan_to_num(result_g, nan=0.0, posinf=1e6, neginf=-1e6)

        new_mem_v = mem_v.at[write_idx].set(result_v)
        new_mem_g = mem_g.at[write_idx, :].set(result_g)

        return (new_mem_v, new_mem_g, write_idx + 1), (result_v, result_g)

    init_state = (memory_val, memory_grad, actual_inputs)
    _, (instruction_outputs, instruction_grads) = jax.lax.scan(
        step, init_state, (genome.ops, genome.args)
    )

    return instruction_outputs, instruction_grads
