from __future__ import annotations

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig
from malthusjax.operators.base import BaseCrossover


@struct.dataclass
class OrderCrossover(BaseCrossover[CategoricalGenome, CategoricalGenomeConfig]):
    """Copy a segment, then fill the remaining slots in the second parent's order.

    The fill sequence starts after the copied segment and skips categories
    already present in it. The operator preserves the permutation without
    repair or dynamic output shapes.
    """

    crossover_rate: float = 0.8

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 3

    def _generate_noise(
        self, keys: chex.Array, config: CategoricalGenomeConfig, generation: int = 0
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        if not isinstance(config, CategoricalGenomeConfig) or (
            config.is_permutation is False
            or config.shape != (config.num_categories,)
            or config.num_categories < 1
        ):
            raise ValueError("OrderCrossover requires a one-dimensional permutation genome")
        n = config.shape[-1]
        should_cross = jax.random.bernoulli(keys[0], p=self.crossover_rate)
        a = jax.random.randint(keys[1], (), 0, n)
        b = jax.random.randint(keys[2], (), 0, n)
        cut1 = jnp.minimum(a, b)
        cut2 = jnp.maximum(a, b)
        return should_cross, cut1, cut2

    def _recombine_one(
        self,
        p1: CategoricalGenome,
        p2: CategoricalGenome,
        noise_data: Tuple[chex.Array, chex.Array, chex.Array],
        config: CategoricalGenomeConfig,
        **_kwargs: Any,
    ) -> CategoricalGenome:
        should_cross, cut1, cut2 = noise_data
        n = config.shape[-1]
        idx = jnp.arange(n)
        in_slice = (idx >= cut1) & (idx <= cut2)

        copied = jnp.zeros(n, dtype=bool).at[p1.values].set(in_slice)

        # The fill order starts just after the copied segment.
        shift = -(cut2 + 1)
        p2_rot = jnp.roll(p2.values, shift)
        p1_rot = jnp.roll(p1.values, shift)
        in_slice_rot = jnp.roll(in_slice, shift)

        # Skip categories already copied from the first parent.
        is_kept = ~copied[p2_rot]

        # Stable argsort on ~is_kept packs all kept elements to the front in relative order
        order = jnp.argsort(~is_kept, stable=True)
        fill_vals = p2_rot[order]

        # Map each non-slice slot in rotated coordinates to its rank in fill_vals
        slot_rank = jnp.cumsum(~in_slice_rot) - 1
        child_rot = jnp.where(in_slice_rot, p1_rot, fill_vals[slot_rank])

        # Rotate back to original coordinate frame
        child = jnp.roll(child_rot, cut2 + 1)
        new_values = jax.lax.select(jnp.broadcast_to(should_cross, child.shape), child, p1.values)
        return p1.replace(values=new_values)
