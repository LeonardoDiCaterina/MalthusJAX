"""Linear Genome mutation operator with two-stage Bernoulli + decay sampling."""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.linear_genome import LinearGenome
from malthusjax.operators.base import BaseMutation, _field


def _uniform_decay(row: int, param: float, max_len: int) -> chex.Array:
    """Uniform over [0, row-1]."""
    return jnp.where(jnp.arange(max_len) < row, 1.0, 0.0)


def _geometric_decay(row: int, param: float, max_len: int) -> chex.Array:
    """Exponential recency bias (param=beta)."""
    j = jnp.arange(max_len)
    dist = jnp.where(j < row, row - j, 0)
    weights = jnp.where(j < row, param ** (dist - 1), 0.0)
    return weights


def _linear_decay(row: int, param: float, max_len: int) -> chex.Array:
    """Linear recency bias (recent is heavier)."""
    j = jnp.arange(max_len)
    weights = jnp.where(j < row, j + 1.0, 0.0)
    return weights


def _window_decay(row: int, param: float, max_len: int) -> chex.Array:
    """Uniform over last `param` rows."""
    window_size = int(param)
    j = jnp.arange(max_len)
    weights = jnp.where((j < row) & (j >= row - window_size), 1.0, 0.0)
    return weights


DECAY_FUNCTIONS = {
    "uniform": _uniform_decay,
    "geometric": _geometric_decay,
    "linear": _linear_decay,
    "window": _window_decay,
}


@struct.dataclass
class LinearMutation(BaseMutation[LinearGenome, Any]):
    """Mutation operator for LinearGenome and derived classes."""

    mutation_rate: float = 0.1
    p_internal: float = 0.5
    decay_name: str = _field(pytree_node=False, default="uniform")
    decay_param: float = 0.9

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _generate_noise(
        self, keys: chex.PRNGKey, config: Any, generation: int = 0
    ) -> Any:
        return (keys[0], keys[1])

    def _mutate_one(
        self, genome: LinearGenome, noise_data: Any, config: Any, **kwargs: Any
    ) -> LinearGenome:
        k0, k1 = noise_data
        
        L = genome.ops.shape[0]
        max_arity = genome.args.shape[1]
        
        N = getattr(config, "num_inputs", 0) + getattr(config, "num_constants", 0)
        num_ops = config.num_ops
        
        k_ops_mask, k_ops_val, k_args_mask, k_source = jax.random.split(k0, 4)
        k_input_idx, k_internal_idx = jax.random.split(k1, 2)
        
        ops_mutate_mask = jax.random.bernoulli(k_ops_mask, self.mutation_rate, shape=(L,))
        new_ops = jax.random.randint(k_ops_val, shape=(L,), minval=0, maxval=num_ops)
        mutated_ops = jnp.where(ops_mutate_mask, new_ops, genome.ops)
        
        args_mutate_mask = jax.random.bernoulli(k_args_mask, self.mutation_rate, shape=(L, max_arity))
        
        source = jax.random.bernoulli(k_source, self.p_internal, shape=(L, max_arity))
        row_indices = jnp.arange(L)[:, None]
        source = source * (row_indices > 0)
        
        new_input_args = jax.random.randint(k_input_idx, shape=(L, max_arity), minval=0, maxval=N)
        
        decay_fn = DECAY_FUNCTIONS.get(self.decay_name, _uniform_decay)
        
        def _sample_internal_row(i: int, k_row: chex.PRNGKey) -> chex.Array:
            weights = decay_fn(i, self.decay_param, L)
            safe_weights = jnp.where(weights.sum() > 0, weights, 1.0)
            probs = safe_weights / safe_weights.sum()
            return jax.random.choice(k_row, L, shape=(max_arity,), p=probs)
            
        k_internal_rows = jax.random.split(k_internal_idx, L)
        new_internal_indices = jax.vmap(_sample_internal_row)(jnp.arange(L), k_internal_rows)
        new_internal_args = N + new_internal_indices
        
        new_args = jnp.where(source == 1, new_internal_args, new_input_args)
        mutated_args = jnp.where(args_mutate_mask, new_args, genome.args)
        
        return genome.replace(ops=mutated_ops, args=mutated_args)
