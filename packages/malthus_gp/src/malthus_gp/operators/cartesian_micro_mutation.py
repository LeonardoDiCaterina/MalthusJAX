"""Cartesian Genetic Programming micro-mutation operator.

Applies independent uniform point mutations across opcodes, connection arguments,
and program outputs as described by Miller & Thomson (2000).
"""

from __future__ import annotations

from typing import Any, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthus_gp.genomes.cartesian_genome import CartesianGenome, CartesianGenomeConfig
from malthusjax.composer import register_mutation
from malthusjax.core.logger import get_logger
from malthusjax.operators.base import BaseMutation

logger = get_logger("malthus_gp.operators.cartesian_micro")

_field: Any = struct.field


@register_mutation(
    name="cartesian_micro",
    override=True,
    compatible_genomes=["cartesian_genome", "cartesian"],
)
@struct.dataclass
class CartesianMicroMutation(BaseMutation[CartesianGenome, CartesianGenomeConfig]):  # type: ignore[misc]
    """Point mutation for Cartesian GP genomes.

    Each gene (op, each arg slot, each out_node) is independently replaced
    with a fresh uniformly-sampled valid allele with probability ``mutation_rate``.

    Args:
        mutation_rate: Per-gene mutation probability. Miller typically uses
                       1% to 5% of total genes, or ~3 mutated genes per chromosome.
    """

    mutation_rate: float = _field(pytree_node=False, default=0.05)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 6

    def _generate_noise(
        self,
        keys: chex.Array,
        config: CartesianGenomeConfig,
        generation: int = 0,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        num_nodes = config.num_nodes
        p = self.mutation_rate

        # 1. Mutation masks
        mask_ops = jax.random.bernoulli(keys[0], p=p, shape=(num_nodes,))
        mask_args = jax.random.bernoulli(keys[1], p=p, shape=(num_nodes, config.max_arity))
        mask_out = jax.random.bernoulli(keys[2], p=p, shape=(config.num_outputs,))

        # 2. Replacement opcodes
        new_ops = jax.random.randint(keys[3], (num_nodes,), 0, config.num_ops, dtype=jnp.int32)

        # 3. Replacement arguments bounded by levels_back
        lo, hi = config._col_connection_bounds()
        raw = jax.random.uniform(keys[4], (num_nodes, config.max_arity))
        span = jnp.maximum(1, hi - lo)[:, None]
        new_args = (lo[:, None] + jnp.floor(raw * span)).astype(jnp.int32)
        new_args = jnp.clip(new_args, lo[:, None], jnp.maximum(lo[:, None], hi[:, None] - 1))

        # 4. Replacement output pointers
        total_available = config.num_inputs + num_nodes
        new_out = jax.random.randint(
            keys[5], (config.num_outputs,), 0, total_available, dtype=jnp.int32
        )

        return mask_ops, mask_args, mask_out, new_ops, new_args, new_out

    def _mutate_one(
        self,
        genome: CartesianGenome,
        noise_data: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
        config: CartesianGenomeConfig,
        **_kwargs: Any,
    ) -> CartesianGenome:
        mask_ops, mask_args, mask_out, new_ops, new_args, new_out = noise_data

        mutated_ops = jnp.where(mask_ops, new_ops, genome.ops)
        mutated_args = jnp.where(mask_args, new_args, genome.args)
        mutated_out = jnp.where(mask_out, new_out, genome.out_nodes)

        result = genome.replace(ops=mutated_ops, args=mutated_args, out_nodes=mutated_out)
        return cast(CartesianGenome, result.autocorrect(config))
