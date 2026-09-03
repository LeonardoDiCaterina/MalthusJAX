"""MEP Crossover Operators.

Implements the three recombination variants described in Oltean & Dumitrescu (2002):

    - One-point recombination  (§3.7.1.1)
    - Two-point recombination  (§3.7.1.2)
    - Uniform recombination    (§3.7.1.3)

Plus a HomologousPrefixCrossover that preserves DAG validity after segment
swap — used for NeuralGenome and MO workflows where suffix arg pointers may
reference novel prefixes.

Key JAX insight:
    DAG validity (args[i] < num_inputs + i) is AUTOMATICALLY preserved by
    segment-swap crossover because both parents satisfy the constraint, and
    swapping at the same row positions leaves the index bounds unchanged.
    HomologousPrefixCrossover adds autocorrect() as a safety net for the
    neural genome case where extra fields exist.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.linear_genome import LinearGenome, LinearGenomeConfig
from malthusjax.operators.base import BaseCrossover

# ---------------------------------------------------------------------------
# One-point recombination  (§3.7.1.1)
# ---------------------------------------------------------------------------


from malthusjax.composer.decorators import register_crossover

@register_crossover(name="mep_one_point")
@struct.dataclass
class MEPOnePointCrossover(BaseCrossover[LinearGenome, LinearGenomeConfig]):
    """One-point MEP recombination.

    A single crossover point *pt* ∈ [1, L) is chosen uniformly.  Offspring
    F1 receives rows 0..pt-1 from p1 and rows pt..L-1 from p2; F2 is the
    complement.

    Paper reference: Oltean & Dumitrescu 2002, §3.7.1.1.
    """

    crossover_rate: float = struct.field(pytree_node=False, default=1.0)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _generate_noise(
        self, keys: chex.Array, config: LinearGenomeConfig, generation: int = 0
    ) -> tuple[chex.Array, chex.Array]:
        """Tier 2 — sample crossover point ∈ [1, L) and apply rate."""
        L = config.length
        point = jax.random.randint(keys[0], shape=(), minval=1, maxval=L)
        do_cross = jax.random.bernoulli(keys[1], p=self.crossover_rate)
        return point, do_cross

    def _recombine_one(
        self,
        p1: LinearGenome,
        p2: LinearGenome,
        noise_data: tuple[chex.Array, chex.Array],
        config: LinearGenomeConfig,
        **_kwargs: Any,
    ) -> LinearGenome:
        """Tier 1 — segment swap at crossover point."""
        L = config.length
        point, do_cross = noise_data
        # True = take from p1 (prefix), False = take from p2 (suffix)
        mask = jnp.arange(L) < point  # (L,)

        crossed_ops = jnp.where(mask, p1.ops, p2.ops)
        crossed_args = jnp.where(mask[:, None], p1.args, p2.args)

        offspring_ops = jnp.where(do_cross, crossed_ops, p1.ops)
        offspring_args = jnp.where(do_cross, crossed_args, p1.args)

        return p1.replace(ops=offspring_ops, args=offspring_args)


# ---------------------------------------------------------------------------
# Two-point recombination  (§3.7.1.2)
# ---------------------------------------------------------------------------


@struct.dataclass
class MEPTwoPointCrossover(BaseCrossover[LinearGenome, LinearGenomeConfig]):
    """Two-point MEP recombination.

    Two crossover points lo < hi are drawn; the middle segment [lo, hi) is
    exchanged between parents.

    Paper reference: Oltean & Dumitrescu 2002, §3.7.1.2.
    """

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _generate_noise(
        self, keys: chex.Array, config: LinearGenomeConfig, generation: int = 0
    ) -> tuple[chex.Array, chex.Array]:
        """Tier 2 — sample two ordered points."""
        L = config.length
        pt1 = jax.random.randint(keys[0], shape=(), minval=1, maxval=L)
        pt2 = jax.random.randint(keys[1], shape=(), minval=1, maxval=L)
        lo = jnp.minimum(pt1, pt2)
        hi = jnp.maximum(pt1, pt2)
        # Ensure lo < hi (if equal, shift hi by 1, clamped)
        hi = jnp.where(hi == lo, jnp.minimum(lo + 1, L - 1), hi)
        return lo, hi

    def _recombine_one(
        self,
        p1: LinearGenome,
        p2: LinearGenome,
        noise_data: tuple[chex.Array, chex.Array],
        config: LinearGenomeConfig,
        **_kwargs: Any,
    ) -> LinearGenome:
        """Tier 1 — exchange the middle segment [lo, hi)."""
        lo, hi = noise_data
        L = config.length
        indices = jnp.arange(L)
        in_middle = (indices >= lo) & (indices < hi)  # (L,)

        offspring_ops = jnp.where(in_middle, p2.ops, p1.ops)
        offspring_args = jnp.where(in_middle[:, None], p2.args, p1.args)
        return p1.replace(ops=offspring_ops, args=offspring_args)


# ---------------------------------------------------------------------------
# Uniform recombination  (§3.7.1.3)
# ---------------------------------------------------------------------------


@struct.dataclass
class MEPUniformCrossover(BaseCrossover[LinearGenome, LinearGenomeConfig]):
    """Uniform MEP recombination.

    Each gene is independently taken from one parent or the other via a
    Bernoulli mask.  ``crossover_rate`` controls how often p2 is preferred
    (default 0.5 → unbiased).

    Paper reference: Oltean & Dumitrescu 2002, §3.7.1.3.
    """

    crossover_rate: float = struct.field(pytree_node=False, default=0.5)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _generate_noise(
        self, keys: chex.Array, config: LinearGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """Tier 2 — Bernoulli gene-selection mask of shape (L,)."""
        L = config.length
        return jax.random.bernoulli(keys[0], p=self.crossover_rate, shape=(L,))

    def _recombine_one(
        self,
        p1: LinearGenome,
        p2: LinearGenome,
        noise_data: chex.Array,
        config: LinearGenomeConfig,
        **_kwargs: Any,
    ) -> LinearGenome:
        """Tier 1 — per-gene selection from p1 (mask=False) or p2 (mask=True)."""
        mask = noise_data  # (L,) bool
        offspring_ops = jnp.where(mask, p2.ops, p1.ops)
        offspring_args = jnp.where(mask[:, None], p2.args, p1.args)
        return p1.replace(ops=offspring_ops, args=offspring_args)


# ---------------------------------------------------------------------------
# Homologous Prefix Crossover (NeuralGenome / MO workflows)
# ---------------------------------------------------------------------------


@struct.dataclass
class HomologousPrefixCrossover(BaseCrossover[LinearGenome, LinearGenomeConfig]):
    """Structure-aware one-point crossover with post-swap autocorrect.

    Identical to ``MEPOnePointCrossover`` in mechanism, but calls
    ``offspring.autocorrect(config)`` after the swap.  This is the safe
    choice when the genome has extra continuous fields (e.g. ``NeuralGenome``
    weights) or when downstream code may rely on strictly valid arg pointers
    in the suffix.

    The autocorrect clips any out-of-bounds arg pointers introduced by the
    suffix transplantation — a no-op when pointers are already valid, so
    there is no correctness cost for standard ``LinearGenome`` usage.
    """

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _generate_noise(
        self, keys: chex.Array, config: LinearGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """Tier 2 — sample split point ∈ [1, L)."""
        L = config.length
        return jax.random.randint(keys[0], shape=(), minval=1, maxval=L)

    def _recombine_one(
        self,
        p1: LinearGenome,
        p2: LinearGenome,
        noise_data: chex.Array,
        config: LinearGenomeConfig,
        **_kwargs: Any,
    ) -> LinearGenome:
        """Tier 1 — prefix from p1 / suffix from p2, then autocorrect."""
        L = config.length
        point = noise_data
        mask = jnp.arange(L) < point

        offspring_ops = jnp.where(mask, p1.ops, p2.ops)
        offspring_args = jnp.where(mask[:, None], p1.args, p2.args)

        child = p1.replace(ops=offspring_ops, args=offspring_args)
        return child.autocorrect(config)


__all__ = [
    "MEPOnePointCrossover",
    "MEPTwoPointCrossover",
    "MEPUniformCrossover",
    "HomologousPrefixCrossover",
]
