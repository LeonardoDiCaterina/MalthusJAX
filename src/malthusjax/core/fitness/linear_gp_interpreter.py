"""Standalone Linear GP interpreter.

Provides the core instruction-execution loop as a pure function so that
multiple evaluator classes (``LinearGPEvaluator``, ``LinearGPPrefixEvaluator``,
etc.) can share the same interpreter without inheritance coupling.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp

from malthusjax.core.fitness.linear_gp_evaluator import TENSORGP_FUNCTIONS
from malthusjax.core.genome.linear_genome import LinearGenome


def predict_one(
    genome: LinearGenome,
    x_input: chex.Array,
    *,
    num_inputs: int,
    length: int,
) -> chex.Array:
    """Execute one genome on one input vector, returning all intermediate results.

    This is the core LGP interpreter loop.  It initialises a flat memory
    buffer, writes the external inputs into the first ``num_inputs`` slots,
    then executes every instruction via ``jax.lax.scan``, accumulating
    results.

    Args:
        genome: An unbatched ``LinearGenome`` instance.
        x_input: A 1-D input vector of shape ``(num_inputs,)``.
        num_inputs: Number of external input slots (``N``).
        length: Number of instructions (``L``).

    Returns:
        A 1-D array of shape ``(L,)`` containing the output of every
        instruction in execution order.
    """
    total_mem = num_inputs + length
    memory = jnp.zeros(total_mem).at[:num_inputs].set(x_input)

    def step(current_mem: Any, inputs: Any) -> Any:
        mem, write_idx = current_mem
        op_code, arg_indices = inputs

        args_val = jnp.take(mem, arg_indices)
        result = jax.lax.switch(
            op_code, TENSORGP_FUNCTIONS, args_val[0], args_val[1], args_val[2]
        )
        result = jnp.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        new_mem = mem.at[write_idx].set(result)

        return (new_mem, write_idx + 1), result

    init_state = (memory, num_inputs)
    _, instruction_outputs = jax.lax.scan(step, init_state, (genome.ops, genome.args))
    return instruction_outputs
