"""Linear genetic programming genome and population definitions.

Encodes programs as sequences of opcode/argument pairs with DAG validity.
Includes utilities for random initialization, autocorrection, rendering and
structural distance metrics, plus the corresponding batched container.
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
class LinearGenomeConfig:
    """Configuration for Linear Genetic Programming (LGP) genomes.

    Defines program structure: L instructions, N external inputs, num_ops
    distinct opcodes, max_arity arguments per operation.

    Attributes:
        length: Total instruction count (L).
        num_inputs: Number of external inputs (N).
        num_ops: Size of opcode alphabet.
        max_arity: Maximum operation arity.
    """

    length: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    num_inputs: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    num_ops: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]
    max_arity: int = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]


@struct.dataclass
class LinearGenome(BaseGenome):
    """Linear Genetic Programming genome with topological DAG constraint.

    Encodes a program as a sequence of (op, args) pairs. Maintains DAG
    validity: instruction i references only external inputs (indices 0:N)
    or prior instructions (indices N:N+i). This prevents cycles and enables
    single-pass evaluation without control-flow analysis during tracing.
    """

    ops: chex.Array  # Shape (L,) - Integer operation codes
    args: chex.Array  # Shape (L, max_arity) - Integer argument indices

    @property
    def values(self) -> Tuple[chex.Array, chex.Array]:
        """Alias for compatibility with BaseGenome interface."""
        return (self.ops, self.args)

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: LinearGenomeConfig) -> LinearGenome:
        """Initialize LGP genome with topological DAG validity.

        Opcodes are drawn uniformly while argument indices are sampled such
        that each instruction can only reference earlier instructions or
        external inputs, ensuring acyclic topology.
        """
        k_ops, k_args = jax.random.split(key)

        ops = jax.random.randint(k_ops, (config.length,), 0, config.num_ops)

        row_limits = jnp.arange(config.num_inputs, config.num_inputs + config.length)

        def gen_row(rk: chex.PRNGKey, climit: chex.Numeric) -> chex.Array:
            return jax.random.randint(rk, (config.max_arity,), 0, climit)

        row_keys = jax.random.split(k_args, config.length)
        args = jax.vmap(gen_row)(row_keys, row_limits)

        return cls(ops=ops, args=args)

    def autocorrect(self, config: LinearGenomeConfig) -> LinearGenome:
        """Restore topological DAG validity via per-instruction index clipping.

        Clips opcodes to [0, num_ops) and argument indices to per-instruction
        limits [0, N+i). Ensures correctness post-mutation without conditional
        branching (XLA-safe).
        """
        valid_ops = jnp.clip(self.ops, 0, config.num_ops - 1)

        # Re-calculate legal index limits for each row
        row_limits = jnp.arange(config.num_inputs, config.num_inputs + config.length)
        max_indices = row_limits[:, None] - 1
        valid_args = jnp.clip(self.args, 0, max_indices)

        return cast(LinearGenome, cast(Any, self).replace(ops=valid_ops, args=valid_args))

    def distance(self, other: BaseGenome, metric: str = DistanceMetric.HAMMING) -> chex.Numeric:
        """
        Computes the structural Hamming distance between two programs.
        Sums mismatches in both operation codes and argument indices.
        """
        other_lin = cast(LinearGenome, other)

        d_ops = jnp.sum(self.ops != other_lin.ops)
        d_args = jnp.sum(self.args != other_lin.args)

        if metric == DistanceMetric.HAMMING:
            return d_ops + d_args
        elif metric == DistanceMetric.EUCLIDEAN:
            # Treating indices as spatial coordinates (less common for LGP)
            return jnp.sqrt(
                jnp.sum(jnp.square(self.ops - other_lin.ops))
                + jnp.sum(jnp.square(self.args - other_lin.args))
            )
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @property
    def size(self) -> int:
        """Returns the total number of instructions (L)."""
        return int(self.ops.shape[-1])

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the logical dimensions of the genome data."""
        # Combining instruction count and arity
        return cast(tuple[int, ...], self.ops.shape + self.args.shape[1:])

    @classmethod
    def from_tensor(cls, arr: tuple[chex.Array, chex.Array], config: Any = None) -> "LinearGenome":
        """Construct LinearGenome from an (ops, args) tensor pair.

        The input tuple may be batched along a leading population dimension.
        The optional *config* argument is ignored and exists purely for
        interface consistency.
        """
        ops, args = arr
        return cls(ops=ops, args=args)

    def render(self, config: LinearGenomeConfig, op_names: Optional[List[str]] = None) -> str:
        """Format the genome as human‑readable assembly text.

        Uses *config* to label inputs and temporaries. If *op_names* is
        provided, those symbols will be used instead of generic ``OP_N``
        placeholders.
        """
        ops_cpu = np.array(self.ops)
        args_cpu = np.array(self.args)
        lines = [f"{'Row':<4} | {'Expression':<30} | {'Raw'}"]
        lines.append("-" * 50)

        for i in range(config.length):
            op_idx = int(ops_cpu[i])
            op_str = op_names[op_idx] if op_names and op_idx < len(op_names) else f"OP_{op_idx}"

            decoded_args = []
            for arg_idx in args_cpu[i]:
                if arg_idx < config.num_inputs:
                    decoded_args.append(f"x_{arg_idx}")
                else:
                    decoded_args.append(f"v_{arg_idx - config.num_inputs}")

            expr = f"v_{i} = {op_str}({', '.join(decoded_args)})"
            lines.append(f"{i:<4} | {expr:<30} | {args_cpu[i]}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Compact representation: instruction count and arity."""
        return f"<LinearGenome(L={self.size}, max_arity={self.args.shape[-1]})>"


@struct.dataclass
class LinearPopulation(BasePopulation[LinearGenome]):
    """Population container for batch-optimized LinearGenomes."""

    genes: LinearGenome
    fitness: chex.Array
    config: LinearGenomeConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    GENOME_CLS: ClassVar[Type[LinearGenome]] = LinearGenome

    @classmethod
    def init_random(
        cls, key: chex.PRNGKey, config: LinearGenomeConfig, size: int
    ) -> LinearPopulation:
        """Parallelized initialization of LGP programs."""
        batched_genes = LinearGenome.create_population(key, config, size)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)
