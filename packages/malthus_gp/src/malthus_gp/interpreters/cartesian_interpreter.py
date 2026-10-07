"""Cartesian Genetic Programming interpreter.

Evaluates feed-forward DAG programs over inputs using jax.lax.scan and jax.lax.switch.
Supports symbolic formula extraction for the active phenotype graph.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthus_gp.core.functions import ARITHMETIC_4_FUNCTIONS, ARITHMETIC_4_NAMES
from malthus_gp.genomes.cartesian_genome import CartesianGenome
from malthusjax.core.fitness.composable.base import BaseInterpreter
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.interpreters.cartesian_interpreter")

_field: Any = struct.field


@struct.dataclass
class CartesianInterpreter(BaseInterpreter[CartesianGenome]):  # type: ignore[misc]
    """Decodes and executes a CartesianGenome on input arrays.

    Uses jax.lax.scan to execute nodes sequentially in feed-forward order,
    storing intermediate activations in a pre-allocated memory tape.
    """

    num_inputs: int = _field(pytree_node=False, default=1)
    num_outputs: int = _field(pytree_node=False, default=1)
    functions: Tuple[Callable[[jax.Array, jax.Array], jax.Array], ...] = _field(
        pytree_node=False, default=ARITHMETIC_4_FUNCTIONS
    )
    function_names: Tuple[str, ...] = _field(pytree_node=False, default=ARITHMETIC_4_NAMES)

    @property
    def num_params(self) -> int:
        return -1

    def evaluate_single(self, genome: CartesianGenome, x_single: chex.Array) -> chex.Array:
        """Executes the genome on a single 1D input sample."""
        n_in = self.num_inputs
        n_nodes = genome.ops.shape[0]
        total_mem = n_in + n_nodes

        # Initialize memory with inputs
        memory = jnp.zeros(total_mem, dtype=jnp.float32)
        x_flat = jnp.atleast_1d(x_single)
        memory = memory.at[:n_in].set(x_flat[:n_in])

        # Step function for jax.lax.scan
        def step(
            current_mem: chex.Array, node_data: Tuple[chex.Array, chex.Array, chex.Array]
        ) -> Tuple[chex.Array, None]:
            op_code, arg_indices, write_idx = node_data

            # Gather inputs for this node
            arg0 = current_mem[arg_indices[0]]
            arg1 = current_mem[arg_indices[1]]

            # Execute primitive operation
            val = jax.lax.switch(op_code, self.functions, arg0, arg1)
            val = jnp.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6)

            # Store in tape
            new_mem = current_mem.at[write_idx].set(val)
            return new_mem, None

        write_indices = n_in + jnp.arange(n_nodes)
        node_inputs = (genome.ops, genome.args, write_indices)

        final_mem, _ = jax.lax.scan(step, memory, node_inputs)

        # Gather outputs
        outputs = jnp.take(final_mem, genome.out_nodes)
        return cast(chex.Array, outputs.reshape(self.num_outputs))

    def apply(self, genome: CartesianGenome, inputs: chex.Array) -> chex.Array:
        """Executes the genome over single or batched inputs."""
        inputs_arr = jnp.asarray(inputs, dtype=jnp.float32)
        if inputs_arr.ndim <= 1:
            return self.evaluate_single(genome, inputs_arr)
        else:
            return jax.vmap(self.evaluate_single, in_axes=(None, 0))(genome, inputs_arr)

    def to_formula(
        self,
        genome: CartesianGenome,
        feature_names: Optional[Sequence[str]] = None,
        output_idx: int = 0,
    ) -> str:
        """Extracts the symbolic mathematical expression for an active output.

        Traverses active connections backwards to produce an interpretable string.
        """
        n_in = self.num_inputs
        if feature_names is None:
            names = [f"x{i}" for i in range(n_in)]
        else:
            names = list(feature_names)[:n_in]

        ops = [int(op) for op in genome.ops]
        args = [[int(a) for a in arg_row] for arg_row in genome.args]
        out_node = int(genome.out_nodes[output_idx])

        # Memoized recursive string reconstruction
        memo: dict[int, str] = {i: names[i] for i in range(n_in)}

        def resolve(idx: int) -> str:
            if idx in memo:
                return memo[idx]
            node_rel = idx - n_in
            op_code = ops[node_rel]
            arg0_idx = args[node_rel][0]
            arg1_idx = args[node_rel][1]

            s0 = resolve(arg0_idx)
            s1 = resolve(arg1_idx)

            op_name = self.function_names[op_code % len(self.function_names)]
            if op_name in ("+", "-", "*", "/"):
                expr = f"({s0} {op_name} {s1})"
            else:
                expr = f"{op_name}({s0})"

            memo[idx] = expr
            return expr

        return resolve(out_node)
