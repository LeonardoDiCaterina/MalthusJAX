"""
Linear Genetic Programming genome implementation.

Provides LinearGenome and LinearPopulation with topological DAG structure,
autocorrection, and efficient batch operations for symbolic regression.
"""

from typing import ClassVar, List, Optional, Type

import chex  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np
from flax import struct  # type: ignore

from malthusjax.core.base import BaseGenome, BasePopulation


@struct.dataclass
class LinearGenomeConfig:
    """Configuration for Linear GP genomes."""
    length: int  = struct.field(pytree_node=False)      # L: Number of instructions
    num_inputs: int  = struct.field(pytree_node=False)  # N: Number of input features
    num_ops: int   = struct.field(pytree_node=False)    # Number of available functions
    max_arity: int = struct.field(pytree_node=False)


@struct.dataclass
class LinearGenome(BaseGenome):
    """
    Linear Genetic Programming genome.
    
    Represents a program as a sequence of instructions, each with an opcode
    and arguments. Maintains topological DAG structure where instruction i
    can only reference inputs 0..N-1 or previous instructions 0..i-1.
    """
    ops: chex.Array   # Shape (L,) - Operation codes
    args: chex.Array  # Shape (L, max_arity) - Argument indices

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: LinearGenomeConfig) -> "LinearGenome":
        """Create random linear genome with valid topological structure."""
        k_ops, k_args = jax.random.split(key)

        # 1. Random operation codes
        ops = jax.random.randint(k_ops, (config.length,), 0, config.num_ops)

        # 2. Topological arguments - row i can only see 0..N+i-1
        row_limits = jnp.arange(config.num_inputs, config.num_inputs + config.length)

        def gen_row(rk, climit):
            return jax.random.randint(rk, (config.max_arity,), 0, climit)

        row_keys = jax.random.split(k_args, config.length)
        args = jax.vmap(gen_row)(row_keys, row_limits)

        return cls(ops=ops, args=args)

    def autocorrect(self, config: LinearGenomeConfig) -> "LinearGenome":
        """Fix invalid references to ensure topological DAG structure."""
        valid_ops = jnp.clip(self.ops, 0, config.num_ops - 1)

        # Re-calculate limits for each instruction
        row_limits = jnp.arange(config.num_inputs, config.num_inputs + config.length)
        max_indices = row_limits[:, None] - 1  # Max valid index per row
        valid_args = jnp.clip(self.args, 0, max_indices)

        return self.replace(ops=valid_ops, args=valid_args)

    def distance(self, other: "LinearGenome", metric: str = "hamming") -> float:
        """Compute Hamming distance between genomes."""
        d_ops = jnp.sum(self.ops != other.ops)
        d_args = jnp.sum(self.args != other.args)
        return (d_ops + d_args).astype(jnp.float32)

    @property
    def size(self) -> int:
        """Return number of instructions in genome."""
        return self.ops.shape[-1]

    @property
    def shape(self) -> tuple:
        """Return shape of genome as (length,)."""
        return self.ops.shape + self.args.shape[1:]

    def render(self, config: LinearGenomeConfig, op_names: Optional[List[str]] = None) -> str:
        """
        Generate human-readable representation of the program.
        
        Args:
            config: Genome configuration
            op_names: Optional list of operation names for display
            
        Returns:
            Multi-line string representation
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
        return f"<LinearGenome(L={self.ops.shape})>"


@struct.dataclass
class LinearPopulation(BasePopulation[LinearGenome]):
    """
    Population container for LinearGenome objects.
    
    Provides efficient batch operations and list-like interface
    for collections of linear genomes.
    """
    genes: LinearGenome
    fitness: chex.Array
    config: LinearGenomeConfig = struct.field(pytree_node=False)

    GENOME_CLS: ClassVar[Type[LinearGenome]] = LinearGenome

    @classmethod
    def init_random(cls, key: chex.PRNGKey, config: LinearGenomeConfig, size: int) -> "LinearPopulation":
        """
        Create random population of linear genomes.
        
        Args:
            key: JAX random key
            config: Genome configuration
            size: Population size
            
        Returns:
            New LinearPopulation with random genomes
        """
        # Use genome factory to create batch
        batched_genes = LinearGenome.create_population(key, config, size)

        # Initialize fitness to negative infinity
        initial_fitness = jnp.full((size,), -jnp.inf)

        return cls(genes=batched_genes, fitness=initial_fitness, config=config)
