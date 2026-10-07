"""Cartesian Genetic Programming genome and population definitions.

Encodes programs as a 2-D grid of computational nodes (num_rows x num_cols).
Primary inputs are indexed [0, num_inputs), nodes are indexed column-by-column
starting at num_inputs.

Two variants are supported:
  - 1-D (num_rows=1, levels_back=-1): fully-connected linear DAG,
    equivalent to Linear GP with neutrality.
  - 2-D (num_rows>1): standard Cartesian GP grid with optional levels_back < num_cols.

The genome is stored as:
  ops       : (num_rows * num_cols,)           int32 - opcode per node
  args      : (num_rows * num_cols, max_arity) int32 - input indices per node
  out_nodes : (num_outputs,)                   int32 - indices of output nodes/inputs
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional, Tuple, Type, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.composer import register_genome
from malthusjax.core.base import BaseGenome, BasePopulation, DistanceMetric
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.genomes.cartesian_genome")

_field: Any = struct.field


@register_genome("cartesian_genome", override=True)
@struct.dataclass
class CartesianGenomeConfig:
    """Configuration for a Cartesian GP genome.

    Attributes:
        num_rows:    Number of node rows (n_r). Use 1 for the 1-D variant.
        num_cols:    Number of node columns (n_c).
        num_inputs:  Number of primary external inputs (N).
        num_outputs: Number of program output values.
        num_ops:     Size of the function set (opcode alphabet).
        max_arity:   Maximum arity of any function in the set.
        levels_back: Maximum number of preceding columns a node may reference.
                     Set to -1 (default) for full feed-forward connectivity.
    """

    num_rows: int = _field(pytree_node=False, default=1)
    num_cols: int = _field(pytree_node=False, default=40)
    num_inputs: int = _field(pytree_node=False, default=1)
    num_outputs: int = _field(pytree_node=False, default=1)
    num_ops: int = _field(pytree_node=False, default=4)
    max_arity: int = _field(pytree_node=False, default=2)
    levels_back: int = _field(pytree_node=False, default=-1)

    @property
    def num_nodes(self) -> int:
        return self.num_rows * self.num_cols

    @property
    def effective_levels_back(self) -> int:
        return self.num_cols if self.levels_back == -1 else self.levels_back

    @property
    def total_genes(self) -> int:
        """Total number of mutable alleles in the chromosome."""
        return self.num_nodes + (self.num_nodes * self.max_arity) + self.num_outputs

    @property
    def dtype(self) -> Any:
        return jnp.int32

    def init_population(self, key: chex.PRNGKey, size: int) -> "CartesianPopulation":
        keys = jax.random.split(key, size)
        genomes = jax.vmap(CartesianGenome.random_init, in_axes=(0, None))(keys, self)
        return CartesianPopulation(
            genes=genomes,
            fitness=jnp.zeros(size),
            config=self,
            info={},
        )

    def _col_connection_bounds(self) -> Tuple[chex.Array, chex.Array]:
        """Precomputes valid (lo, hi-exclusive) input ranges for each node.

        For a node at column c (0-indexed), valid connections are:
          - primary inputs: [0, num_inputs)
          - nodes from column max(0, c - l) to c - 1

        Returns:
            lo: shape (num_nodes,) - inclusive lower bound per node.
            hi: shape (num_nodes,) - exclusive upper bound per node.
        """
        N = self.num_inputs
        nr = self.num_rows
        node_idx = jnp.arange(self.num_nodes)
        col = node_idx // nr

        hi = N + col * nr

        l_back = self.effective_levels_back
        lo = (col >= l_back) * (N + (col - l_back) * nr)

        return lo, hi


@struct.dataclass
class CartesianGenome(BaseGenome):  # type: ignore[misc]
    """Cartesian Genetic Programming genome.

    Fields:
        ops:       (num_nodes,)           int32 - opcode per node.
        args:      (num_nodes, max_arity) int32 - input indices.
        out_nodes: (num_outputs,)         int32 - output node indices.
    """

    ops: chex.Array
    args: chex.Array
    out_nodes: chex.Array

    @property
    def values(self) -> Tuple[chex.Array, chex.Array, chex.Array]:
        return (self.ops, self.args, self.out_nodes)

    @classmethod
    def random_init(
        cls: Type["CartesianGenome"], key: chex.PRNGKey, config: CartesianGenomeConfig
    ) -> "CartesianGenome":
        k_ops, k_args, k_out = jax.random.split(key, 3)

        num_nodes = config.num_nodes
        ops = jax.random.randint(k_ops, (num_nodes,), 0, config.num_ops, dtype=jnp.int32)

        lo, hi = config._col_connection_bounds()

        raw = jax.random.uniform(k_args, (num_nodes, config.max_arity))
        span = jnp.maximum(1, hi - lo)[:, None]
        args = (lo[:, None] + jnp.floor(raw * span)).astype(jnp.int32)
        args = jnp.clip(args, lo[:, None], jnp.maximum(lo[:, None], hi[:, None] - 1))

        total_available = config.num_inputs + num_nodes
        out_nodes = jax.random.randint(
            k_out, (config.num_outputs,), 0, total_available, dtype=jnp.int32
        )

        return cls(ops=ops, args=args, out_nodes=out_nodes)

    def autocorrect(self, config: Optional[CartesianGenomeConfig] = None) -> "CartesianGenome":
        """Clamps opcodes, arguments, and output pointers to valid ranges."""
        if config is None:
            return self

        lo, hi = config._col_connection_bounds()

        valid_hi = jnp.maximum(lo[:, None], hi[:, None] - 1)
        clamped_args = jnp.clip(self.args, lo[:, None], valid_hi)
        clamped_ops = jnp.clip(self.ops, 0, config.num_ops - 1)

        total_available = config.num_inputs + config.num_nodes
        clamped_out = jnp.clip(self.out_nodes, 0, total_available - 1)

        return cast(
            "CartesianGenome",
            self.replace(ops=clamped_ops, args=clamped_args, out_nodes=clamped_out),
        )

    def distance(
        self,
        other: BaseGenome,
        metric: DistanceMetric = "hamming",
        config: Optional[CartesianGenomeConfig] = None,
    ) -> chex.Numeric:
        """Hamming distance between two Cartesian genomes."""
        other_cgp = cast(CartesianGenome, other)
        diff_ops = jnp.sum(self.ops != other_cgp.ops)
        diff_args = jnp.sum(self.args != other_cgp.args)
        diff_out = jnp.sum(self.out_nodes != other_cgp.out_nodes)
        total_diff = diff_ops + diff_args + diff_out

        total_genes = self.ops.size + self.args.size + self.out_nodes.size
        return total_diff.astype(jnp.float32) / float(total_genes)

    @property
    def size(self) -> int:
        return int(self.ops.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.ops.shape)

    @classmethod
    def from_tensor(
        cls: Type["CartesianGenome"], arr: Any, config: Any = None
    ) -> "CartesianGenome":
        if isinstance(arr, (tuple, list)):
            return cls(ops=arr[0], args=arr[1], out_nodes=arr[2])
        raise ValueError("CartesianGenome requires (ops, args, out_nodes) tuple")

    @classmethod
    def create_population(
        cls: Type["CartesianGenome"],
        key: chex.PRNGKey,
        config: CartesianGenomeConfig,
        size: int,
    ) -> "CartesianGenome":
        keys = jax.random.split(key, size)
        return jax.vmap(cls.random_init, in_axes=(0, None))(keys, config)

    def get_active_mask(self, num_inputs: int) -> chex.Array:
        """Computes a boolean mask of active nodes contributing to the outputs.

        Returns:
            mask: shape (num_inputs + num_nodes,) bool array.
        """
        total = num_inputs + self.ops.shape[0]
        init_mask = jnp.zeros(total, dtype=jnp.bool_)
        init_mask = init_mask.at[self.out_nodes].set(True)

        def step(mask: chex.Array, idx: chex.Array) -> Tuple[chex.Array, None]:
            node_idx = num_inputs + idx
            is_active = mask[node_idx]
            arg_indices = self.args[idx]
            new_mask = jnp.where(is_active, mask.at[arg_indices].set(True), mask)
            return new_mask, None

        # Scan backwards from highest node down to lowest node
        node_indices = jnp.arange(self.ops.shape[0] - 1, -1, -1)
        final_mask, _ = jax.lax.scan(step, init_mask, node_indices)
        return cast(chex.Array, final_mask)


@struct.dataclass
class CartesianPopulation(BasePopulation[CartesianGenome]):  # type: ignore[misc]
    """Population container for batched CartesianGenomes."""

    genes: CartesianGenome
    GENOME_CLS: ClassVar[Type[CartesianGenome]] = CartesianGenome

    @classmethod
    def init(
        cls: Type["CartesianPopulation"],
        key: chex.PRNGKey,
        config: CartesianGenomeConfig,
        size: int,
    ) -> "CartesianPopulation":
        batched_genes = CartesianGenome.create_population(key, config, size)
        return cls(
            genes=batched_genes,
            fitness=jnp.zeros(size, dtype=jnp.float32),
            config=config,
            info={},
        )
