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
from malthusjax.operators.base import BaseMutation, _field
from malthusjax.operators.base_injection import BaseMutation_injection


@struct.dataclass
class GaussianMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Gaussian (Normal) Mutation following the 3-Tier Paradigm.
    Optimized: Uses FMA (Fused Multiply-Add)
    via masked arithmetic
    """

    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    clip: bool = _field(pytree_node=False, default=False)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 2 keys: one for the Bernoulli mask and one for Gaussian noise."""
        return 2

    def _generate_noise(self, keys: chex.Array, config: RealGenomeConfig) -> chex.Array:
        """
        Tier 2 — Noise Generation.
        Handles all RNG and masking logic to produce the 'delta' payload.
        """
        k_mask, k_noise = keys[0], keys[1]
        dtype = config.dtype

        # 1. Generate Bernoulli Mask (0.0 or 1.0)
        mask_bool = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=config.shape)
        mask_val = mask_bool.astype(dtype)

        # 2. Generate Gaussian Noise scaled by strength
        raw_noise = jax.random.normal(k_noise, shape=config.shape, dtype=dtype)
        strength = jnp.array(self.mutation_strength, dtype=dtype)

        # 3. Combine into a single delta for Tier 1
        # This is the 'noise_data' passed to the arithmetic kernel.
        return raw_noise * strength * mask_val

    def _mutate_one(
        self, genome: RealGenome, noise_data: chex.Array, config: RealGenomeConfig, **kwargs: Any
    ) -> RealGenome:
        """
        Tier 1 — Arithmetic Kernel.
        Pure, promotion-free arithmetic focusing on FMA optimization.
        """
        # Fused addition: genome + (noise * strength * mask)
        # noise_data already contains (noise * strength * mask) from Tier 2.
        mutated_values = genome.values + noise_data

        # Optional clipping based on static config
        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)

        return cast(RealGenome, cast(Any, genome).replace(values=mutated_values))


@struct.dataclass
class GaussianMutation_injection(
    BaseMutation_injection[RealGenome, RealGenomeConfig, RealPopulation]
):
    """
    Injection-mode Gaussian Mutation.

    Consumes a single PRNG key and splits it internally to produce a flattened
    noise array of shape `(input_length * num_offspring, ...)`. The base
    injection wrapper will reshape this into `(input_length, num_offspring, ...)`
    for the vmapped mutation kernel.
    """

    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    clip: bool = _field(pytree_node=False, default=False)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 2 keys:
        - one for the Bernoulli mask
        - one for Gaussian noise.
        """
        return 2

    def _generate_noise(self, key: chex.PRNGKey, config: RealGenomeConfig) -> chex.Array:
        if self.input_length <= 0 or self.num_offspring <= 0:
            msg = "Set `input_length` and `num_offspring` before calling _generate_noise."
            raise ValueError(msg)
        n = int(self.input_length * self.num_offspring)
        total = n * self.num_keys_per_atomic_operation
        subkeys = jax.random.split(key, total)
        # reshape to (n, atomic_keys, 2)
        subkeys = subkeys.reshape((n, self.num_keys_per_atomic_operation, -1))

        def per_row(k_row: chex.Array) -> chex.Array:
            k_mask, k_noise = k_row[0], k_row[1]
            mask_bool = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=config.shape)
            mask_val = mask_bool.astype(config.dtype)
            raw_noise = jax.random.normal(k_noise, shape=config.shape, dtype=config.dtype)
            return raw_noise * self.mutation_strength * mask_val

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
    Samples a point uniformly within an n-dimensional ball of given radius.
    Optimized: Tier 2 generates the vector; Tier 1 applies it.
    """

    radius: float = 0.1
    mutation_rate: float = 1.0
    clip: bool = _field(pytree_node=False, default=False)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 3 keys: 1 for Bernoulli mask, 1 for Gaussian vector, 1 for Magnitude scaling."""
        return 3

    def _generate_noise(self, keys: chex.Array, config: RealGenomeConfig) -> chex.Array:
        """
        Tier 2 — Noise Generation.
        Implements the 'Muller's Method' for uniform ball sampling.
        """
        k_mask, k_vector, k_mag = keys[0], keys[1], keys[2]
        dtype = config.dtype
        shape = config.shape

        # 1. Bernoulli Mask (0.0 or 1.0)
        mask_val = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=shape).astype(dtype)

        # 2. Muller's Method:
        # a) Sample from a standard Normal distribution
        raw_vector = jax.random.normal(k_vector, shape=shape, dtype=dtype)

        # b) Calculate the norm
        norm = jnp.sqrt(jnp.sum(jnp.square(raw_vector))) + 1e-8

        # c) Sample uniform 'u' for radial scaling (ensures uniform density inside the volume)
        u = jax.random.uniform(k_mag, shape=(), minval=0.0, maxval=1.0, dtype=dtype)

        # d) Combine: (direction) * (volume-scaled magnitude)
        # We use jnp.power(u, 1/N) to ensure uniform spatial distribution
        dimension = jnp.array(jnp.prod(jnp.array(shape)), dtype=dtype)
        r = self.radius * jnp.power(u, 1.0 / dimension)

        unit_vector = raw_vector / norm
        ball_delta = unit_vector * r

        # 3. Apply mask as a multiplier
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
    Injection-mode Ball Mutation.
    """

    radius: float = 0.1
    mutation_rate: float = 1.0
    clip: bool = _field(pytree_node=False, default=False)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 3

    def _generate_noise(self, key: chex.PRNGKey, config: RealGenomeConfig) -> chex.Array:
        if self.input_length <= 0 or self.num_offspring <= 0:
            msg = "Set `input_length` and `num_offspring` before calling _generate_noise."
            raise ValueError(msg)
        n = int(self.input_length * self.num_offspring)
        total = n * self.num_keys_per_atomic_operation
        subkeys = jax.random.split(key, total)
        subkeys = subkeys.reshape((n, self.num_keys_per_atomic_operation, -1))

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
            r = self.radius * jnp.power(u, 1.0 / dimension)
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
    Optimized: Tier 2 handles the complex power-logic; Tier 1 performs FMA.
    """

    mutation_rate: float = 0.1
    eta: float = 20.0
    clip: bool = _field(pytree_node=False, default=False)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 2 keys: one for the Bernoulli mask and one for the 'u' value."""
        return 2

    def _generate_noise(self, keys: chex.Array, config: RealGenomeConfig) -> chex.Array:
        """
        Tier 2 — Noise Generation.
        Implements the standard polynomial mutation logic to produce a delta payload.
        """
        k_mask, k_val = keys[0], keys[1]
        dtype = config.dtype

        # 1. Vaccinate Constants
        eta = jnp.array(self.eta, dtype=dtype)
        one = jnp.array(1.0, dtype=dtype)
        half = jnp.array(0.5, dtype=dtype)
        two = jnp.array(2.0, dtype=dtype)
        exponent = one / (eta + one)

        # 2. Sample Mask & U
        mask_val = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=config.shape).astype(
            dtype
        )
        u = jax.random.uniform(k_val, shape=config.shape, dtype=dtype)

        # 3. Calculate Delta_Q
        delta_q = jnp.where(
            u <= half,
            jnp.power(two * u, exponent) - one,
            one - jnp.power(two * (one - u), exponent),
        )

        # 4. Scale by Bounds
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
    Injection-mode Polynomial Mutation.
    """

    mutation_rate: float = 0.1
    eta: float = 20.0
    clip: bool = _field(pytree_node=False, default=False)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _generate_noise(self, key: chex.PRNGKey, config: RealGenomeConfig) -> chex.Array:
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
