"""
Linear genome mutation operators.

Implements mutation operators tailored for linear genomes
with topological constraints and automatic repair mechanisms.
"""

from flax import struct  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import chex  # type: ignore

from malthusjax.operators.base import BaseMutation
from malthusjax.core.genome.linear import LinearGenome, LinearGenomeConfig, LinearPopulation


@struct.dataclass
class LinearMutation(BaseMutation[LinearGenome, LinearGenomeConfig, LinearPopulation]):
    """
    Linear GP mutation operator.
    
    Applies probabilistic mutations to operation codes and arguments
    with automatic topological repair mechanisms.
    """
    # Dynamic parameters (can be changed without recompilation)
    op_rate: float = 0.1    # Probability to mutate operation code
    arg_rate: float = 0.1   # Probability to mutate argument

    def _mutate_one(self, key: chex.PRNGKey, genome: LinearGenome, config: LinearGenomeConfig) -> LinearGenome:
        """Apply mutation to a single genome."""
        k_op, k_arg, k_noise = jax.random.split(key, 3)

        # 1. Generate boolean masks for mutations
        mask_ops = jax.random.bernoulli(k_op, self.op_rate, genome.ops.shape)
        mask_args = jax.random.bernoulli(k_arg, self.arg_rate, genome.args.shape)

        # 2. Generate random replacement values
        noise_ops = jax.random.randint(k_noise, genome.ops.shape, 0, config.num_ops)
        
        # For arguments, allow references to full memory space (autocorrect will fix invalid refs)
        max_mem = config.num_inputs + config.length
        k_noise_args = jax.random.split(k_noise)[0]
        noise_args = jax.random.randint(k_noise_args, genome.args.shape, 0, max_mem)

        # 3. Apply mutations where masks are True
        new_ops = jnp.where(mask_ops, noise_ops, genome.ops)
        new_args = jnp.where(mask_args, noise_args, genome.args)

        # 4. Create new genome and repair any topological violations
        return genome.replace(ops=new_ops, args=new_args).autocorrect(config)


@struct.dataclass
class LinearPointMutation(BaseMutation[LinearGenome, LinearGenomeConfig, LinearPopulation]):
    """
    Point mutation for linear genomes - mutates exactly one position.
    
    Useful for fine-grained search and local optimization.
    """
    
    def _mutate_one(self, key: chex.PRNGKey, genome: LinearGenome, config: LinearGenomeConfig) -> LinearGenome:
        """Apply single-point mutation."""
        k_choice, k_op, k_arg = jax.random.split(key, 3)
        
        # Choose random instruction to mutate
        inst_idx = jax.random.randint(k_choice, (), 0, config.length)
        
        # Choose whether to mutate op or arg (50/50 chance)
        mutate_op = jax.random.bernoulli(k_choice, 0.5)
        
        def mutate_ops():
            new_op = jax.random.randint(k_op, (), 0, config.num_ops)
            return genome.ops.at[inst_idx].set(new_op), genome.args
            
        def mutate_args():
            # Choose random argument position
            arg_idx = jax.random.randint(k_arg, (), 0, config.max_arity)
            # Generate valid argument value for this instruction
            max_ref = config.num_inputs + inst_idx
            new_arg = jax.random.randint(k_arg, (), 0, max_ref)
            return genome.ops, genome.args.at[inst_idx, arg_idx].set(new_arg)
        
        new_ops, new_args = jax.lax.cond(mutate_op, mutate_ops, mutate_args)
        
        return genome.replace(ops=new_ops, args=new_args)