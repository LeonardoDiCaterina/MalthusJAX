"""
Real-valued Mutation Operators.
Optimized for H100:
1. Uses Masked Arithmetic (genome + noise * mask) instead of Branching (jnp.where).
2. Explicit casting to ensure correct dtypes (e.g., BF16)
    during random number generation and arithmetic operations.
"""

from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random
from flax import struct

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.engine.schedules import ScheduleType, compute_scheduled_strength
from malthusjax.operators.base import BaseMutation, _field
from malthusjax.operators.base_injection import BaseMutation_injection


@struct.dataclass
class GaussianMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Gaussian (Normal) Mutation (3-Tier Paradigm).
    Tier 2: Bernoulli mask + Gaussian noise scaled by mutation_strength.
    Tier 1: Masked addition (genome + noise*strength*mask) via FMA.
    Shape contract: (d,) genome + (d,) noise → (d,) mutated genome.
    Key budget: 2 pre-allocated subkeys (Bernoulli mask, Gaussian noise).
    Optimization: Fused multiply-add avoids intermediate arrays; masked
    arithmetic (multiplication by 0.0/1.0) beats jnp.where for XLA latency.
    """

    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    clip: bool = _field(pytree_node=False, default=False)
    schedule_type: ScheduleType = _field(pytree_node=False, default=ScheduleType.CONSTANT)
    final_strength: float = _field(pytree_node=False, default=0.0)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 2 keys: one for the Bernoulli mask and one for Gaussian noise."""
        return 2

    def _generate_noise(
        self, keys: chex.Array, config: RealGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """
        Tier 2 — Noise Generation.
        Generates mask (Bernoulli) and noise (Gaussian scaled by scheduled strength).
        Returns: (d,) array = noise * strength * mask for Tier 1 arithmetic.
        """
        k_mask, k_noise = keys[0], keys[1]
        dtype = config.dtype

        mask_bool = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=config.shape)
        mask_val = mask_bool.astype(dtype)
        raw_noise = jax.random.normal(k_noise, shape=config.shape, dtype=dtype)
        if self.schedule_type == ScheduleType.CONSTANT:
            strength = jnp.array(self.mutation_strength, dtype=dtype)
        else:
            strength = compute_scheduled_strength(
                self.schedule_type,
                generation,
                self.max_generations,
                self.mutation_strength,
                self.final_strength,
            ).astype(dtype)
        return raw_noise * strength * mask_val

    def _mutate_one(
        self, genome: RealGenome, noise_data: chex.Array, config: RealGenomeConfig, **kwargs: Any
    ) -> RealGenome:
        """
        Tier 1 — Arithmetic Kernel.
        Fused addition: genome + (noise * strength * mask). Clip if enabled.
        """
        mutated_values = genome.values + noise_data
        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)
        return cast(RealGenome, cast(Any, genome).replace(values=mutated_values))


@struct.dataclass
class GaussianMutation_injection(
    BaseMutation_injection[RealGenome, RealGenomeConfig, RealPopulation]
):
    """
    Injection-mode Gaussian Mutation (single-key variant).
    Splits single key to (n*K) subkeys, reshaped (n, K, -1) for vmap.
    Vmap generates all (n*num_offspring) noise arrays in parallel.
    Shape contract: (d,) noise per (pair, offspring) → (n, d) flattened output.
    Trade-off: Full noise materialization vs no re-splitting (reproducibility).
    """

    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    clip: bool = _field(pytree_node=False, default=False)
    schedule_type: ScheduleType = _field(pytree_node=False, default=ScheduleType.CONSTANT)
    final_strength: float = _field(pytree_node=False, default=0.0)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 2 keys:
        - one for the Bernoulli mask
        - one for Gaussian noise.
        """
        return 2

    def _generate_noise(
        self, key: chex.PRNGKey, config: RealGenomeConfig, generation: int = 0
    ) -> chex.Array:
        if self.input_length <= 0 or self.num_offspring <= 0:
            msg = "Set `input_length` and `num_offspring` before calling _generate_noise."
            raise ValueError(msg)
        n = int(self.input_length * self.num_offspring)
        total = n * self.num_keys_per_atomic_operation
        subkeys = jax.random.split(key, total)
        # reshape to (n, atomic_keys, 2)
        subkeys = subkeys.reshape((n, self.num_keys_per_atomic_operation, -1))

        if self.schedule_type == ScheduleType.CONSTANT:
            strength = self.mutation_strength
        else:
            strength = compute_scheduled_strength(
                self.schedule_type,
                generation,
                self.max_generations,
                self.mutation_strength,
                self.final_strength,
            )

        def per_row(k_row: chex.Array) -> chex.Array:
            k_mask, k_noise = k_row[0], k_row[1]
            mask_bool = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=config.shape)
            mask_val = mask_bool.astype(config.dtype)
            raw_noise = jax.random.normal(k_noise, shape=config.shape, dtype=config.dtype)
            return raw_noise * strength * mask_val

        noise = jax.vmap(per_row)(subkeys)
        return noise

    def _mutate_one(
        self, genome: RealGenome, noise_data: chex.Array, config: RealGenomeConfig, **kwargs: Any
    ) -> RealGenome:
        """
        Tier 1 — Arithmetic Kernel (same as fused version).
        """
        mutated_values = genome.values + noise_data

        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)

        return cast(RealGenome, cast(Any, genome).replace(values=mutated_values))


@struct.dataclass
class BallMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Ball Mutation (3-Tier Paradigm).
    Tier 2: Muller's Method—Gaussian direction normalized, scaled by u^(1/d).
    Tier 1: Masked addition (genome + ball_delta * mask).
    Shape contract: (d,) genome + (d,) ball_delta → (d,) mutated genome.
    Key budget: 3 pre-allocated subkeys (mask, direction, magnitude scaling).
    Muller's Method ensures uniform spatial distribution within ball volume
    via power-law radial scaling u^(1/d) to counter dimension bias.
    """

    radius: float = 0.1
    mutation_rate: float = 1.0
    clip: bool = _field(pytree_node=False, default=False)
    schedule_type: ScheduleType = _field(pytree_node=False, default=ScheduleType.CONSTANT)
    final_radius: float = _field(pytree_node=False, default=0.0)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 3 keys: 1 for Bernoulli mask, 1 for Gaussian vector, 1 for Magnitude scaling."""
        return 3

    def _generate_noise(
        self, keys: chex.Array, config: RealGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """
        Tier 2 — Noise Generation (Muller's Method).
        Returns: (d,) array = unit_direction * scheduled_radius * u^(1/d) * mask.
        """
        k_mask, k_vector, k_mag = keys[0], keys[1], keys[2]
        dtype = config.dtype
        shape = config.shape

        mask_val = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=shape).astype(dtype)
        raw_vector = jax.random.normal(k_vector, shape=shape, dtype=dtype)
        norm = jnp.sqrt(jnp.sum(jnp.square(raw_vector))) + 1e-8
        u = jax.random.uniform(k_mag, shape=(), minval=0.0, maxval=1.0, dtype=dtype)
        dimension = jnp.array(jnp.prod(jnp.array(shape)), dtype=dtype)
        if self.schedule_type == ScheduleType.CONSTANT:
            radius = self.radius
        else:
            radius = compute_scheduled_strength(
                self.schedule_type,
                generation,
                self.max_generations,
                self.radius,
                self.final_radius,
            )
        r = radius * jnp.power(u, 1.0 / dimension)
        unit_vector = raw_vector / norm
        ball_delta = unit_vector * r
        return ball_delta * mask_val

    def _mutate_one(
        self, genome: RealGenome, noise_data: chex.Array, config: RealGenomeConfig, **kwargs: Any
    ) -> RealGenome:
        """Tier 1 — Arithmetic Kernel."""
        mutated_values = genome.values + noise_data

        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)

        return cast(RealGenome, cast(Any, genome).replace(values=mutated_values))


@struct.dataclass
class BallMutation_injection(BaseMutation_injection[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Injection-mode Ball Mutation (single-key variant).
    Splits single key to (n*3) subkeys, reshaped (n, 3, -1) for vmap.
    Vmap generates all (n*num_offspring) ball deltas in parallel.
    Shape contract: (d,) ball_delta per (pair, offspring) → (n, d) flattened.
    Trade-off: Full vector/magnitude materialization vs reproducibility.
    """

    radius: float = 0.1
    mutation_rate: float = 1.0
    clip: bool = _field(pytree_node=False, default=False)
    schedule_type: ScheduleType = _field(pytree_node=False, default=ScheduleType.CONSTANT)
    final_radius: float = _field(pytree_node=False, default=0.0)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 3

    def _generate_noise(
        self, key: chex.PRNGKey, config: RealGenomeConfig, generation: int = 0
    ) -> chex.Array:
        if self.input_length <= 0 or self.num_offspring <= 0:
            msg = "Set `input_length` and `num_offspring` before calling _generate_noise."
            raise ValueError(msg)
        n = int(self.input_length * self.num_offspring)
        total = n * self.num_keys_per_atomic_operation
        subkeys = jax.random.split(key, total)
        subkeys = subkeys.reshape((n, self.num_keys_per_atomic_operation, -1))

        if self.schedule_type == ScheduleType.CONSTANT:
            radius = self.radius
        else:
            radius = compute_scheduled_strength(
                self.schedule_type,
                generation,
                self.max_generations,
                self.radius,
                self.final_radius,
            )

        def per_row(k_row: chex.Array) -> chex.Array:
            k_mask, k_vector, k_mag = k_row[0], k_row[1], k_row[2]
            dtype = config.dtype
            mask_val = jax.random.bernoulli(
                k_mask, p=self.mutation_rate, shape=config.shape
            ).astype(dtype)
            raw_vector = jax.random.normal(k_vector, shape=config.shape, dtype=dtype)
            norm = jnp.sqrt(jnp.sum(jnp.square(raw_vector))) + 1e-8
            u = jax.random.uniform(k_mag, shape=(), minval=0.0, maxval=1.0, dtype=dtype)
            dimension = jnp.array(jnp.prod(jnp.array(config.shape)), dtype=dtype)
            r = radius * jnp.power(u, 1.0 / dimension)
            unit_vector = raw_vector / norm
            ball_delta = unit_vector * r
            return ball_delta * mask_val

        return jax.vmap(per_row)(subkeys)

    def _mutate_one(
        self, genome: RealGenome, noise_data: chex.Array, config: RealGenomeConfig, **kwargs: Any
    ) -> RealGenome:
        mutated_values = genome.values + noise_data

        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)

        return cast(RealGenome, cast(Any, genome).replace(values=mutated_values))


@struct.dataclass
class PolynomialMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Polynomial Mutation (3-Tier Paradigm).
    Tier 2: Masks delta via jnp.where(u≤0.5, (2u)^(1/(η+1))-1, 1-(2(1-u))^(1/(η+1))).
    Tier 1: Scaled addition (genome + delta * bound_range * mask).
    Shape contract: (d,) genome + (d,) delta → (d,) mutated genome.
    Key budget: 2 pre-allocated subkeys (mask, uniform u for polynomial spread).
    Delta branching ensures symmetric distribution around parent for both u<0.5
    and u≥0.5 regions, producing offspring concentrated near parent (small η).
    """

    mutation_rate: float = 0.1
    eta: float = 20.0
    clip: bool = _field(pytree_node=False, default=False)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 2 keys: one for the Bernoulli mask and one for the 'u' value."""
        return 2

    def _generate_noise(
        self, keys: chex.Array, config: RealGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """
        Tier 2 — Noise Generation (Polynomial mutation).
        Returns: (d,) array = delta_q * bound_range * mask (scaled by [min,max]).
        """
        k_mask, k_val = keys[0], keys[1]
        dtype = config.dtype
        eta = jnp.array(self.eta, dtype=dtype)
        one = jnp.array(1.0, dtype=dtype)
        half = jnp.array(0.5, dtype=dtype)
        two = jnp.array(2.0, dtype=dtype)
        exponent = one / (eta + one)
        mask_val = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=config.shape).astype(
            dtype
        )
        u = jax.random.uniform(k_val, shape=config.shape, dtype=dtype)
        delta_q = jnp.where(
            u <= half,
            jnp.power(two * u, exponent) - one,
            one - jnp.power(two * (one - u), exponent),
        )
        min_val, max_val = config.bounds
        bound_range = jnp.array(max_val - min_val, dtype=dtype)
        return delta_q * bound_range * mask_val

    def _mutate_one(
        self, genome: RealGenome, noise_data: chex.Array, config: RealGenomeConfig, **kwargs: Any
    ) -> RealGenome:
        """Tier 1 — Arithmetic Kernel."""
        mutated_values = genome.values + noise_data

        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)

        return cast(RealGenome, cast(Any, genome).replace(values=mutated_values))


@struct.dataclass
class PolynomialMutation_injection(
    BaseMutation_injection[RealGenome, RealGenomeConfig, RealPopulation]
):
    """
    Injection-mode Polynomial Mutation (single-key variant).
    Splits single key to (n*2) subkeys, reshaped (n, 2, -1) for vmap.
    Vmap generates all (n*num_offspring) delta_q arrays in parallel.
    Shape contract: (d,) delta per (pair, offspring) → (n, d) flattened.
    Trade-off: Full delta materialization vs reproducibility (no re-splitting).
    """

    mutation_rate: float = 0.1
    eta: float = 20.0
    clip: bool = _field(pytree_node=False, default=False)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _generate_noise(
        self, key: chex.PRNGKey, config: RealGenomeConfig, generation: int = 0
    ) -> chex.Array:
        if self.input_length <= 0 or self.num_offspring <= 0:
            msg = "Set `input_length` and `num_offspring` before calling _generate_noise."
            raise ValueError(msg)
        n = int(self.input_length * self.num_offspring)
        total = n * self.num_keys_per_atomic_operation
        subkeys = jax.random.split(key, total)
        subkeys = subkeys.reshape((n, self.num_keys_per_atomic_operation, -1))

        def per_row(k_row: chex.Array) -> chex.Array:
            k_mask, k_val = k_row[0], k_row[1]
            dtype = config.dtype
            eta = jnp.array(self.eta, dtype=dtype)
            one = jnp.array(1.0, dtype=dtype)
            half = jnp.array(0.5, dtype=dtype)
            two = jnp.array(2.0, dtype=dtype)
            exponent = one / (eta + one)
            mask_val = jax.random.bernoulli(
                k_mask, p=self.mutation_rate, shape=config.shape
            ).astype(dtype)
            u = jax.random.uniform(k_val, shape=config.shape, dtype=dtype)
            delta_q = jnp.where(
                u <= half,
                jnp.power(two * u, exponent) - one,
                one - jnp.power(two * (one - u), exponent),
            )
            min_val, max_val = config.bounds
            bound_range = jnp.array(max_val - min_val, dtype=dtype)
            return delta_q * bound_range * mask_val

        return jax.vmap(per_row)(subkeys)

    def _mutate_one(
        self, genome: RealGenome, noise_data: chex.Array, config: RealGenomeConfig, **kwargs: Any
    ) -> RealGenome:
        mutated_values = genome.values + noise_data

        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)

        return cast(RealGenome, cast(Any, genome).replace(values=mutated_values))


__all__ = [
    "GaussianMutation",
    "BallMutation",
    "PolynomialMutation",
    "GaussianMutation_injection",
    "BallMutation_injection",
    "PolynomialMutation_injection",
]
