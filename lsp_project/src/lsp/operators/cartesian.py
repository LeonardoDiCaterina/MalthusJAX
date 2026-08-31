"""Cartesian GP Operators.

Implements:
  - CartesianMicroMutation: point mutation of ops, args, and out_nodes.
  - NoOpCrossover:          identity crossover (passes parent through unchanged).
"""

from __future__ import annotations

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.cartesian_genome import CartesianGenome, CartesianGenomeConfig
from malthusjax.operators.base import BaseCrossover, BaseMutation

# ---------------------------------------------------------------------------
# NoOpCrossover — identity operator, required by GeneticEngine pipeline
# ---------------------------------------------------------------------------


@struct.dataclass
class NoOpCrossover(BaseCrossover[CartesianGenome, CartesianGenomeConfig]):
    """Identity crossover — always returns parent 1 unchanged.

    CGP relies entirely on point mutation with a 1+λ strategy.
    This operator satisfies the GeneticEngine's crossover slot without
    performing any actual recombination.
    """

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1  # Consume 1 key (discarded) to satisfy ResourceMapper

    def _generate_noise(
        self, keys: chex.Array, config: CartesianGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """No noise needed — return a dummy scalar."""
        return jnp.zeros((), dtype=jnp.int32)

    def _recombine_one(
        self,
        p1: CartesianGenome,
        p2: CartesianGenome,
        noise_data: Any,
        config: CartesianGenomeConfig,
        **_kwargs: Any,
    ) -> CartesianGenome:
        """Return p1 unchanged."""
        return p1


# ---------------------------------------------------------------------------
# CartesianMicroMutation — point mutation
# ---------------------------------------------------------------------------


@struct.dataclass
class CartesianMicroMutation(BaseMutation[CartesianGenome, CartesianGenomeConfig]):
    """Point mutation for Cartesian GP genomes.

    Each gene (op, each arg slot, each out_node) is independently replaced
    with a fresh uniformly-sampled valid allele with probability ``mutation_rate``.

    This matches the standard CGP mutation described in Miller & Thomson (2000).

    Args:
        mutation_rate: Per-gene mutation probability (p_m).
                       Miller typically uses 1% to 10% of total genes.
    """

    mutation_rate: float = struct.field(pytree_node=False, default=0.05)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        # key 0: mutation mask for ops        (num_nodes,)
        # key 1: mutation mask for args       (num_nodes, max_arity)
        # key 2: mutation mask for out_nodes  (num_outputs,)
        # key 3: new opcodes
        # key 4: new args
        # key 5: new out_nodes
        return 6

    def _generate_noise(
        self,
        keys: chex.Array,
        config: CartesianGenomeConfig,
        generation: int = 0,
    ) -> Tuple[
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ]:
        """Tier 2 — sample mutation masks and replacement alleles."""
        num_nodes = config.num_nodes
        p = self.mutation_rate

        # ---- Mutation masks ----
        mask_ops = jax.random.bernoulli(keys[0], p=p, shape=(num_nodes,))
        mask_args = jax.random.bernoulli(keys[1], p=p, shape=(num_nodes, config.max_arity))
        mask_out = jax.random.bernoulli(keys[2], p=p, shape=(config.num_outputs,))

        # ---- New opcodes ----
        new_ops = jax.random.randint(keys[3], (num_nodes,), 0, config.num_ops)

        # ---- New args (respecting levels_back / per-node bounds) ----
        N = config.num_inputs
        _, hi = config._col_connection_bounds()  # (num_nodes,)
        raw_args = jax.random.randint(keys[4], (num_nodes, config.max_arity), 0, N + num_nodes)
        new_args = jnp.clip(raw_args, 0, hi[:, None] - 1)

        # ---- New out_nodes ----
        total = config.num_inputs + num_nodes
        new_out = jax.random.randint(keys[5], (config.num_outputs,), config.num_inputs, total)

        return mask_ops, mask_args, mask_out, new_ops, new_args, new_out

    def _mutate_one(
        self,
        genome: CartesianGenome,
        noise_data: Tuple[
            chex.Array,
            chex.Array,
            chex.Array,
            chex.Array,
            chex.Array,
            chex.Array,
        ],
        config: CartesianGenomeConfig,
        **_kwargs: Any,
    ) -> CartesianGenome:
        """Tier 1 — apply mutation masks."""
        mask_ops, mask_args, mask_out, new_ops, new_args, new_out = noise_data

        mutated_ops = jnp.where(mask_ops, new_ops, genome.ops)
        mutated_args = jnp.where(mask_args, new_args, genome.args)
        mutated_out = jnp.where(mask_out, new_out, genome.out_nodes)

        result = genome.replace(ops=mutated_ops, args=mutated_args, out_nodes=mutated_out)
        return result.autocorrect(config)


__all__ = ["NoOpCrossover", "CartesianMicroMutation"]
