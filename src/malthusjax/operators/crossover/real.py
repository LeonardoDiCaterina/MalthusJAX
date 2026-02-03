"""
Real-valued Crossover Operators.
Refactored to be purely atomic consumers.
Optimized to consume pre-allocated keys directly, avoiding internal splitting.
"""

from typing import Any, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.operators.base import BaseCrossover
from malthusjax.operators.base_injection import BaseCrossover_injection as BaseCrossover_injection


@struct.dataclass
class UniformCrossover(BaseCrossover[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Uniform Crossover (Fused 3-Tier Paradigm).
    Mixes genes from both parents based on a per-gene probability.
    """

    # This operator produces a single offspring per pair (static contract)
    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 1 key to generate the Bernoulli mixing mask."""
        return 1

    def _generate_noise(self, keys: chex.PRNGKey, config: RealGenomeConfig) -> chex.Array:
        """
        Tier 2 — Mask Generation.
        Generates a boolean mask matching the genome shape.
        """
        # keys[0] is the subkey allocated by the ResourceMapper.
        return jax.random.bernoulli(keys[0], p=self.crossover_rate, shape=config.shape)

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: chex.Array,
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:
        """
        Tier 1 — Recombination Kernel (Pure).
        Uses the pre-generated mask to select values from parents.
        """
        mask = noise_data
        # Convention: mask=True selects from p2, False selects from p1
        offspring_values = jnp.where(mask, p2.values, p1.values)
        return cast(RealGenome, cast(Any, p1).replace(values=offspring_values))


@struct.dataclass
class UniformCrossover_injection(
    BaseCrossover_injection[RealGenome, RealGenomeConfig, RealPopulation]
):
    """
    Injection-mode Uniform Crossover.
    Produces pre-generated masks per (pair, offspring) flattened into a single
    axis. The base injection wrapper will reshape to `(input_length, num_offspring, ...)`.
    """

    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _generate_noise(self, key: chex.PRNGKey, config: RealGenomeConfig) -> chex.Array:
        if self.input_length <= 0 or self.num_offspring <= 0:
            raise ValueError(
                "Set `input_length` and `num_offspring` before calling _generate_noise."
            )
        n = int(self.input_length * self.num_offspring)
        subkeys = jax.random.split(key, n)

        def per_row(k: chex.PRNGKey) -> chex.Array:
            return jax.random.bernoulli(k, p=self.crossover_rate, shape=config.shape)

        return jax.vmap(per_row)(subkeys)

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: chex.Array,
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:
        """
        Tier 1 — Recombination Kernel (Pure).
        Implemented for injection: same logic as the fused `UniformCrossover`.
        """
        mask = noise_data
        offspring_values = jnp.where(mask, p2.values, p1.values)
        return cast(RealGenome, cast(Any, p1).replace(values=offspring_values))


@struct.dataclass
class BlendCrossover(BaseCrossover[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Blend Crossover (BLX-alpha) - Fused 3-Tier Paradigm.
    Creates an expanded box around parents and samples uniformly within it.
    """

    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9
    alpha: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 2 keys: one for the crossover decision and one for uniform sampling."""
        return 2

    def _generate_noise(
        self,
        keys: chex.PRNGKey,
        config: RealGenomeConfig,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Tier 2 — Stochastic Payload.
        Generates the raw data needed for the blend arithmetic.
        """
        k_do, k_val = keys[0], keys[1]
        dtype = config.dtype

        # 1. Decision mask (Boolean)
        should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)

        # 2. Uniform random samples (Matching genome shape)
        random_samples = jax.random.uniform(k_val, shape=config.shape, dtype=dtype)

        return should_cross, random_samples

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: Tuple[chex.Array, chex.Array],
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:
        """
        Tier 1 — Blend Kernel (Pure).
        Pure arithmetic using pre-generated stochastic values.
        """
        should_cross, random_vals = noise_data
        dtype = p1.values.dtype

        # 1. Calculate BLX Interval Logic
        diff = jnp.abs(p1.values - p2.values)
        alpha_val = jnp.array(self.alpha, dtype=dtype)

        cmin = jnp.minimum(p1.values, p2.values) - (alpha_val * diff)
        cmax = jnp.maximum(p1.values, p2.values) + (alpha_val * diff)

        # 2. Apply Blend
        offspring_values = cmin + random_vals * (cmax - cmin)

        # 3. Clip to Config Bounds
        min_b, max_b = config.bounds
        offspring_values = jnp.clip(offspring_values, min_b, max_b)

        # 4. Fused Selection
        # XLA fuses this with the arithmetic above.
        final_values = jnp.where(should_cross, offspring_values, p1.values)

        # 5. Return as offspring
        return cast(RealGenome, cast(Any, p1).replace(values=final_values))


@struct.dataclass
class BlendCrossover_injection(
    BaseCrossover_injection[RealGenome, RealGenomeConfig, RealPopulation]
):
    """
    Injection-mode Blend Crossover.
    Returns tuple (should_cross, random_samples) flattened over (pair- offspring).
    """

    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9
    alpha: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def _generate_noise(
        self,
        key: chex.PRNGKey,
        config: RealGenomeConfig,
    ) -> Tuple[chex.Array, chex.Array]:
        if self.input_length <= 0 or self.num_offspring <= 0:
            raise ValueError(
                "Set `input_length` and `num_offspring` before calling _generate_noise."
            )
        n = int(self.input_length * self.num_offspring)
        subkeys = jax.random.split(key, n * 2).reshape((n, 2, -1))

        def per_row(k_row: chex.Array) -> Tuple[chex.Array, chex.Array]:
            k_do, k_val = k_row[0], k_row[1]
            should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)
            random_samples = jax.random.uniform(k_val, shape=config.shape, dtype=config.dtype)
            return should_cross, random_samples

        should_crosss, random_samples = jax.vmap(per_row, in_axes=0)(subkeys)
        # jax.vmap of a tuple returns a tuple of arrays
        return should_crosss, random_samples

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: Tuple[chex.Array, chex.Array],
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:
        """
        Tier 1 — Blend Kernel (Pure).
        Implemented for injection counterparts to reuse fused logic.
        """
        should_cross, random_vals = noise_data
        dtype = p1.values.dtype

        diff = jnp.abs(p1.values - p2.values)
        alpha_val = jnp.array(self.alpha, dtype=dtype)

        cmin = jnp.minimum(p1.values, p2.values) - (alpha_val * diff)
        cmax = jnp.maximum(p1.values, p2.values) + (alpha_val * diff)

        offspring_values = cmin + random_vals * (cmax - cmin)
        min_b, max_b = config.bounds
        offspring_values = jnp.clip(offspring_values, min_b, max_b)
        final_values = jnp.where(should_cross, offspring_values, p1.values)
        return cast(RealGenome, cast(Any, p1).replace(values=final_values))


@struct.dataclass
class SimulatedBinaryCrossover(BaseCrossover[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    Simulated Binary Crossover (SBX) - Fused 3-Tier Paradigm.
    Simulates single-point crossover with a distribution-defined spread factor.
    """

    # SBX produces two children per pair as standard
    num_offspring: int = struct.field(pytree_node=False, default=2)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9
    eta: float = 20.0

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 3 keys: decision to cross, spread factor (u), and child selection."""
        return 3

    def _generate_noise(
        self,
        keys: chex.PRNGKey,
        config: RealGenomeConfig,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Tier 2 — Stochastic Payload.
        Generates the raw spread variables and decision masks.

        Note: Keys are now allocated per-offspring; the swap mask will differ per
        offspring call making it possible to obtain distinct children when the
        base class repeats this kernel across `num_offspring`.
        """
        k_do, k_beta, k_swap = keys[0], keys[1], keys[2]
        dtype = config.dtype

        # 1. Crossover decision (scalar bool — affects whether we cross at all)
        should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)

        # 2. Uniform 'u' for spread factor (beta) calculation
        u = jax.random.uniform(k_beta, shape=config.shape, dtype=dtype)

        # 3. Swap mask to pick between child 1 and child 2 (will vary per-offspring)
        swap_mask = jax.random.bernoulli(k_swap, p=0.5, shape=config.shape)

        return should_cross, u, swap_mask

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: Tuple[chex.Array, chex.Array, chex.Array],
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:
        """
        Tier 1 — SBX Kernel (Pure) that returns a single offspring.

        When the base class runs this kernel `num_offspring` times with different
        per-offspring keys, you will obtain multiple children per pair.
        """
        should_cross, u, swap_mask = noise_data
        dtype = p1.values.dtype

        # 1. Calculate Beta (Spread Factor)
        exponent = jnp.array(1.0 / (self.eta + 1.0), dtype=dtype)
        beta = jnp.where(u <= 0.5, (2.0 * u) ** exponent, (1.0 / (2.0 * (1.0 - u))) ** exponent)

        # 2. Generate Symmetric Candidate Children
        c1 = 0.5 * ((1.0 + beta) * p1.values + (1.0 - beta) * p2.values)
        c2 = 0.5 * ((1.0 - beta) * p1.values + (1.0 + beta) * p2.values)

        # 3. Select Child & Apply Constraints
        child_vals = jnp.where(swap_mask, c2, c1)

        min_b, max_b = config.bounds
        child_vals = jnp.clip(child_vals, min_b, max_b)

        # 4. Conditional Recombination
        final_values = jnp.where(should_cross, child_vals, p1.values)

        return cast(RealGenome, cast(Any, p1).replace(values=final_values))


@struct.dataclass
class SimulatedBinaryCrossover_injection(
    BaseCrossover_injection[RealGenome, RealGenomeConfig, RealPopulation]
):
    """
    Injection-mode Simulated Binary Crossover (SBX).
    Returns triple (should_cross, u, swap_mask) flattened over (pair- offspring).
    """

    num_offspring: int = struct.field(pytree_node=False, default=2)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9
    eta: float = 20.0

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 3

    def _generate_noise(
        self,
        key: chex.PRNGKey,
        config: RealGenomeConfig,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        if self.input_length <= 0 or self.num_offspring <= 0:
            raise ValueError(
                "Set `input_length` and `num_offspring` before calling _generate_noise."
            )
        n = int(self.input_length * self.num_offspring)
        total = n * self.num_keys_per_atomic_operation
        subkeys = jax.random.split(key, total).reshape((n, self.num_keys_per_atomic_operation, -1))

        def per_row(k_row: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
            k_do, k_beta, k_swap = k_row[0], k_row[1], k_row[2]
            should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)
            u = jax.random.uniform(k_beta, shape=config.shape, dtype=config.dtype)
            swap_mask = jax.random.bernoulli(k_swap, p=0.5, shape=config.shape)
            return should_cross, u, swap_mask

        should_crosss, u_arr, swap_mask_arr = jax.vmap(per_row, in_axes=0)(subkeys)
        return should_crosss, u_arr, swap_mask_arr

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: Tuple[chex.Array, chex.Array, chex.Array],
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:
        """
        Tier 1 — SBX Kernel (Pure) for injection variant. Returns a single offspring.
        """
        should_cross, u, swap_mask = noise_data
        dtype = p1.values.dtype

        exponent = jnp.array(1.0 / (self.eta + 1.0), dtype=dtype)
        beta = jnp.where(u <= 0.5, (2.0 * u) ** exponent, (1.0 / (2.0 * (1.0 - u))) ** exponent)

        c1 = 0.5 * ((1.0 + beta) * p1.values + (1.0 - beta) * p2.values)
        c2 = 0.5 * ((1.0 - beta) * p1.values + (1.0 + beta) * p2.values)

        child_vals = jnp.where(swap_mask, c2, c1)
        min_b, max_b = config.bounds
        child_vals = jnp.clip(child_vals, min_b, max_b)
        final_values = jnp.where(should_cross, child_vals, p1.values)

        return cast(RealGenome, cast(Any, p1).replace(values=final_values))


@struct.dataclass
class BinomialCrossover(BaseCrossover[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    DE Binomial Crossover - Fused 3-Tier Paradigm.
    Constructs a trial vector by selecting genes from a mutant and a target.
    """

    # Produces a single trial vector per (target, mutant) pair
    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 1 key for the Bernoulli crossover mask."""
        return 1

    def _generate_noise(self, keys: chex.PRNGKey, config: RealGenomeConfig) -> chex.Array:
        """
        Tier 2 — Mask Generation.
        Generates the boolean mask used to select between target and mutant.
        """
        # keys[0] matches the ResourceMapper allocation for this pair.
        return jax.random.bernoulli(keys[0], p=self.crossover_rate, shape=config.shape)

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: Any,
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:
        """
        Tier 1 — Recombination Kernel (Pure).
        Fused selection and boundary clipping.
        """
        cross_mask = noise_data

        # 1. Select (Fused Selection)
        # True -> Take Mutant, False -> Take Target
        trial_values = jnp.where(cross_mask, p1.values, p2.values)

        # 2. Boundary Constraints
        # Vaccination: config.bounds is already typed from the Config class.
        min_val, max_val = config.bounds
        trial_values = jnp.clip(trial_values, min_val, max_val)

        # 3. Return as single offspring
        return cast(RealGenome, cast(Any, p1).replace(values=trial_values))


@struct.dataclass
class BinomialCrossover_injection(
    BaseCrossover_injection[RealGenome, RealGenomeConfig, RealPopulation]
):
    """
    Injection-mode Binomial Crossover.
    Produces per-(pair, offspring) masks for DE binomial selection.
    """

    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _generate_noise(self, key: chex.PRNGKey, config: RealGenomeConfig) -> chex.Array:
        if self.input_length <= 0 or self.num_offspring <= 0:
            raise ValueError(
                "Set `input_length` and `num_offspring` before calling _generate_noise."
            )
        n = int(self.input_length * self.num_offspring)
        subkeys = jax.random.split(key, n)

        def per_row(k: chex.PRNGKey) -> chex.Array:
            return jax.random.bernoulli(k, p=self.crossover_rate, shape=config.shape)

        return jax.vmap(per_row)(subkeys)

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: chex.Array,
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:
        """
        Tier 1 — Binomial Kernel (Pure) for injection variant.
        """
        cross_mask = noise_data
        trial_values = jnp.where(cross_mask, p2.values, p1.values)
        min_val, max_val = config.bounds
        trial_values = jnp.clip(trial_values, min_val, max_val)
        return cast(RealGenome, cast(Any, p1).replace(values=trial_values))


__all__ = [
    "UniformCrossover",
    "SimulatedBinaryCrossover",
    "BlendCrossover",
    "BinomialCrossover",
    "UniformCrossover_injection",
    "BlendCrossover_injection",
    "BinomialCrossover_injection",
    "SimulatedBinaryCrossover_injection",
]
