"""Differentiable Cartesian GP Mutation Operator.

Applies point micro-mutations to discrete topology (opcodes, argument pointers,
and output nodes), while preserving continuous weights and biases for Lamarckian
local gradient optimization, or optionally applying Gaussian weight perturbations.
"""

from __future__ import annotations

from typing import Any, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthus_gp.genomes.differentiable_cartesian_genome import (
    DifferentiableCartesianGenome,
    DifferentiableCartesianGenomeConfig,
)
from malthusjax.composer import register_mutation
from malthusjax.core.logger import get_logger
from malthusjax.operators.base import BaseMutation

logger = get_logger("malthus_gp.operators.differentiable_mutation")

_field: Any = struct.field


@register_mutation(
    name="diff_cartesian_mutation",
    override=True,
    compatible_genomes=["diff_cartesian_genome"],
)
@struct.dataclass
class DifferentiableMicroMutation(
    BaseMutation[DifferentiableCartesianGenome, DifferentiableCartesianGenomeConfig]  # type: ignore[misc]
):
    """Mutation operator for Differentiable Cartesian genomes.

    Parameters
    ----------
    mutation_rate : float, default=0.05
        Per-gene probability of perturbing topological genes (ops, args, outputs).
    weight_noise_std : float, default=0.0
        Standard deviation of Gaussian noise added to weights. If 0.0, weights
        are preserved intact for Lamarckian gradient descent.
    """

    mutation_rate: float = _field(pytree_node=False, default=0.05)
    weight_noise_std: float = _field(pytree_node=False, default=0.0)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        # keys: mask_ops, mask_args, mask_out, new_ops, new_args, new_out, weight_noise, bias_noise
        return 8

    def _generate_noise(
        self,
        keys: chex.Array,
        config: DifferentiableCartesianGenomeConfig,
        generation: int = 0,
    ) -> Tuple[
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ]:
        num_nodes = config.num_nodes
        p = self.mutation_rate

        # 1. Topological masks
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

        # 5. Continuous parameter noise (optional)
        w_noise = jax.random.normal(keys[6], (num_nodes, config.max_arity)) * self.weight_noise_std
        b_noise = jax.random.normal(keys[7], (num_nodes,)) * self.weight_noise_std

        return mask_ops, mask_args, mask_out, new_ops, new_args, new_out, w_noise, b_noise

    def _mutate_one(
        self,
        genome: DifferentiableCartesianGenome,
        noise_data: Tuple[
            chex.Array,
            chex.Array,
            chex.Array,
            chex.Array,
            chex.Array,
            chex.Array,
            chex.Array,
            chex.Array,
        ],
        config: DifferentiableCartesianGenomeConfig,
        **_kwargs: Any,
    ) -> DifferentiableCartesianGenome:
        mask_ops, mask_args, mask_out, new_ops, new_args, new_out, w_noise, b_noise = noise_data

        mutated_ops = jnp.where(mask_ops, new_ops, genome.ops)
        mutated_args = jnp.where(mask_args, new_args, genome.args)
        mutated_out = jnp.where(mask_out, new_out, genome.out_nodes)

        mutated_w = genome.weights + w_noise
        mutated_b = genome.biases + b_noise

        result = genome.replace(
            ops=mutated_ops,
            args=mutated_args,
            out_nodes=mutated_out,
            weights=mutated_w,
            biases=mutated_b,
        )
        return cast(DifferentiableCartesianGenome, result.autocorrect(config))
