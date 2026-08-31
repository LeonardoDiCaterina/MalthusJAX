"""Cartesian Genetic Programming genome and population definitions.

Encodes programs as a 2-D grid of computational nodes (num_rows x num_cols).
Primary inputs are indexed [0, num_inputs), nodes are indexed column-by-column
starting at num_inputs.

Two variants are supported:
  - 1-D (num_rows=1, levels_back=num_cols): fully-connected linear DAG,
    equivalent to Linear GP with neutrality.
  - 2-D (num_rows>1): standard Cartesian GP grid with optional levels_back < num_cols.

The genome is stored as:
  ops       : (num_rows * num_cols,)       int32 - opcode per node
  args      : (num_rows * num_cols, max_arity)  int32 - input indices per node
  out_nodes : (num_outputs,)               int32 - indices of output nodes/inputs
"""

from __future__ import annotations

from typing import Any, ClassVar, List, Optional, Tuple, Type, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation, DistanceMetric


@struct.dataclass
class CartesianGenomeConfig:
    """Configuration for a Cartesian GP genome.

    Attributes:
        num_rows:    Number of node rows (n_r). Use 1 for the 1-D variant.
        num_cols:    Number of node columns (n_c).
        levels_back: Maximum number of preceding columns a node may reference.
                     Set to num_cols (default) for full connectivity.
        num_inputs:  Number of primary external inputs (N).
        num_outputs: Number of program output values.
        num_ops:     Size of the function set (opcode alphabet).
        max_arity:   Maximum arity of any function in the set.
    """

    num_rows: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    num_cols: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    num_inputs: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    num_outputs: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    num_ops: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    max_arity: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    levels_back: int = struct.field(pytree_node=False, default=-1)  # type: ignore[no-untyped-call]
    # -1 means "full connectivity" (levels_back == num_cols)

    @property
    def num_nodes(self) -> int:
        return self.num_rows * self.num_cols

    @property
    def effective_levels_back(self) -> int:
        return self.num_cols if self.levels_back == -1 else self.levels_back

    @property
    def dtype(self) -> Any:
        return jnp.int32

    def init_population(self, key: chex.PRNGKey, size: int) -> "BasePopulation":
        keys = jax.random.split(key, size)
        genomes = jax.vmap(CartesianGenome.random_init, in_axes=(0, None))(keys, self)
        # Use CartesianPopulation instead of BasePopulation so type checking works out
        return CartesianPopulation(genes=genomes, fitness=jnp.zeros(size), config=self, info={})

    def _col_connection_bounds(self) -> Tuple[chex.Array, chex.Array]:
        """Precompute (lo, hi-exclusive) valid input ranges for each node.

        For a node at column c (0-indexed), valid connections are:
          - primary inputs: [0, num_inputs)
          - nodes from column max(0, c - l) to c-1

        To keep this XLA-friendly we compute a lo/hi for the *combined* range
        [0, num_inputs + c * num_rows], treating primary inputs as column -inf.

        Returns:
            lo: shape (num_nodes,) - always 0 (primary inputs always valid).
            hi: shape (num_nodes,) - exclusive upper bound per node.
        """
        N = self.num_inputs
        nr = self.num_rows
        node_idx = jnp.arange(self.num_nodes)  # 0 .. num_nodes-1
        col = node_idx // nr  # column of each node

        # The earliest column reachable is max(0, c - l).
        # Nodes from [earliest_col, col) are valid sources.
        # Absolute node index of those nodes: N + earliest_col * nr  ..  N + col * nr - 1
        # Primary inputs [0, N) are ALWAYS valid so lo = 0.
        hi = N + col * nr  # exclusive upper bound
        lo = jnp.zeros_like(hi)  # primary inputs always OK
        return lo, hi


@struct.dataclass
class CartesianGenome(BaseGenome):
    """Cartesian Genetic Programming genome.

    Fields:
        ops:       (num_rows * num_cols,)           int32 - opcode per node.
        args:      (num_rows * num_cols, max_arity) int32 - input indices.
        out_nodes: (num_outputs,)                   int32 - output node indices.
    """

    ops: chex.Array  # (num_nodes,)
    args: chex.Array  # (num_nodes, max_arity)
    out_nodes: chex.Array  # (num_outputs,)

    @property
    def values(self) -> Tuple[chex.Array, chex.Array, chex.Array]:
        return (self.ops, self.args, self.out_nodes)

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: CartesianGenomeConfig) -> "CartesianGenome":
        """Randomly initialize a CGP genome respecting the levels_back constraint."""
        k_ops, k_args, k_out = jax.random.split(key, 3)

        num_nodes = config.num_nodes
        N = config.num_inputs

        # Opcodes: uniform over [0, num_ops)
        ops = jax.random.randint(k_ops, (num_nodes,), 0, config.num_ops)

        # Connection bounds per node
        _, hi = config._col_connection_bounds()  # (num_nodes,)

        # Sample args globally then clip per-node to [0, hi-1]
        raw_args = jax.random.randint(k_args, (num_nodes, config.max_arity), 0, N + num_nodes)
        args = jnp.clip(raw_args, 0, hi[:, None] - 1)

        # Output genes: point to any node or primary input
        total_nodes_and_inputs = N + num_nodes
        out_nodes = jax.random.randint(k_out, (config.num_outputs,), N, total_nodes_and_inputs)

        return cls(ops=ops, args=args, out_nodes=out_nodes)

    def autocorrect(self, config: CartesianGenomeConfig) -> "CartesianGenome":
        """Clip all genes to valid ranges (XLA-safe, no branches)."""
        valid_ops = jnp.clip(self.ops, 0, config.num_ops - 1)

        _, hi = config._col_connection_bounds()
        max_arg = hi[:, None] - 1
        valid_args = jnp.clip(self.args, 0, max_arg)

        total = config.num_inputs + config.num_nodes
        valid_out = jnp.clip(self.out_nodes, 0, total - 1)

        return cast(
            CartesianGenome,
            cast(Any, self).replace(ops=valid_ops, args=valid_args, out_nodes=valid_out),
        )

    def distance(self, other: BaseGenome, metric: str = DistanceMetric.HAMMING) -> chex.Numeric:
        other_cgp = cast(CartesianGenome, other)
        d_ops = jnp.sum(self.ops != other_cgp.ops)
        d_args = jnp.sum(self.args != other_cgp.args)
        d_out = jnp.sum(self.out_nodes != other_cgp.out_nodes)
        return d_ops + d_args + d_out

    @property
    def size(self) -> int:
        return int(self.ops.shape[-1])

    @property
    def shape(self) -> tuple:
        return cast(tuple, self.ops.shape + self.args.shape[1:])

    @classmethod
    def from_tensor(cls, arr: Any, config: Any = None) -> "CartesianGenome":
        ops, args, out_nodes = arr
        return cls(ops=ops, args=args, out_nodes=out_nodes)

    def render(
        self,
        config: CartesianGenomeConfig,
        op_names: Optional[List[str]] = None,
    ) -> str:
        """Human-readable dump of the CGP program."""
        ops_cpu = np.array(self.ops)
        args_cpu = np.array(self.args)
        out_cpu = np.array(self.out_nodes)
        N = config.num_inputs

        def _label(idx: int) -> str:
            return f"x_{idx}" if idx < N else f"v_{idx - N}"

        lines = [f"{'Node':<6} | {'Expression':<35} | {'raw args'}"]
        lines.append("-" * 60)
        for i in range(config.num_nodes):
            op_idx = int(ops_cpu[i])
            op_str = op_names[op_idx] if op_names and op_idx < len(op_names) else f"OP_{op_idx}"
            arg_labels = [_label(int(a)) for a in args_cpu[i]]
            expr = f"v_{i} = {op_str}({', '.join(arg_labels)})"
            lines.append(f"{i:<6} | {expr:<35} | {args_cpu[i].tolist()}")

        lines.append(f"\nOutputs: {[_label(int(o)) for o in out_cpu]}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<CartesianGenome(nodes={self.size}, "
            f"max_arity={self.args.shape[-1]}, "
            f"outputs={self.out_nodes.shape[0]})>"
        )


@struct.dataclass
class CartesianPopulation(BasePopulation[CartesianGenome]):
    """Population container for batched CartesianGenomes."""

    genes: CartesianGenome
    fitness: chex.Array

    GENOME_CLS: ClassVar[Type[CartesianGenome]] = CartesianGenome

    @classmethod
    def init_random(
        cls,
        key: chex.PRNGKey,
        config: CartesianGenomeConfig,
        size: int,
    ) -> "CartesianPopulation":
        batched_genes = CartesianGenome.create_population(key, config, size)
        return cls(
            genes=batched_genes,
            fitness=jnp.full((size,), jnp.inf),
            config=config,
        )
