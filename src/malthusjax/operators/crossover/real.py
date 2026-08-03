"""
Real-valued Crossover Operators.
Refactored to be purely atomic consumers.
Optimized to consume pre-allocated keys directly, avoiding internal splitting.
"""

from dataclasses import replace
from typing import Any, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.operators.base import BaseCrossover
from malthusjax.operators.base_injection import BaseCrossover_injection as BaseCrossover_injection


@struct.dataclass
class UniformCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    Uniform Crossover (Fused 3-Tier Paradigm).
    Per-gene independent selection from parents via Bernoulli mask. XLA fuses mask generation
    (Tier 2) with selection kernel (Tier 1) into single compiled operation.

    Shape contract: Parent (d,) × Parent (d,) → Offspring (d,)
    Key budget: 1 pre-allocated subkey (from ResourceMapper) per pair.
    """

    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Bernoulli mask generation requires 1 PRNG subkey."""
        return 1

    def _generate_noise(
        self, keys: chex.PRNGKey, config: RealGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """
        Tier 2 — Bernoulli Mask (Binary noise).
        Generates (d,) boolean array via Bernoulli(p=crossover_rate). Pre-allocated key
        ensures deterministic, reproducible masking per pair (key determinism).

        Returns: (d,) boolean array
        """
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
        Tier 1 — XLA-Fused Recombination Kernel.
        Pure arithmetic: select parent values using per-gene mask (True=p2, False=p1).
        XLA fuses this selection with Tier 2 mask generation for single kernel launch.

        Returns: Offspring RealGenome with (d,) values
        """
        mask = noise_data
        offspring_values = jnp.where(mask, p2.values, p1.values)
        return replace(p1, values=offspring_values)


@struct.dataclass
class UniformCrossover_injection(BaseCrossover_injection[RealGenome, RealGenomeConfig]):
    """
    Injection-mode Uniform Crossover.
    Single key splits into (n_pairs * n_offspring) subkeys; jax.vmap(per_row) generates all masks
    in parallel, returning (n_pairs * n_offspring, d) flattened array for base wrapper to unfold.
    Trade-off: Full noise materialization enables reproducibility without key re-splitting.

    Shape contract: Parent (d,) → Noise (d,), vmapped to (K, d) then flattened to (K*d,)
    """

    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Single key for Bernoulli mask generation (split internally)."""
        return 1

    def _generate_noise(
        self, key: chex.PRNGKey, config: RealGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """Generate all (pair, offspring) masks upfront. Shape: (n_pairs * n_offspring, d)."""
        if self.input_length <= 0 or self.num_offspring <= 0:
            raise ValueError(
                "Set `input_length` and `num_offspring` before calling _generate_noise."
            )
        n = int(self.input_length * self.num_offspring)
        subkeys = jax.random.split(key, n)

        def per_row(k: chex.PRNGKey) -> chex.Array:
            return jax.random.bernoulli(k, p=self.crossover_rate, shape=config.shape)

        return jax.vmap(per_row)(subkeys)  # (n, d) boolean masks

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: chex.Array,
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:
        """XLA-fused recombination: select using per-gene pre-generated mask."""
        mask = noise_data
        offspring_values = jnp.where(mask, p2.values, p1.values)
        return replace(p1, values=offspring_values)


@struct.dataclass
class BlendCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """Blend Crossover (BLX-α) — Adaptive Parent-Centered Exploration.

    Blend crossover expands the interval [min(p1,p2), max(p1,p2)] by a factor proportional
    to the parent distance, then uniformly samples offspring within this expanded range.
    This encourages exploration beyond parent bounds while still exploiting the distance
    between them.

    **Algorithm**:

    1. Compute interval [L, U] = [min(p1,p2), max(p1,p2)]
    2. Expand interval by ±α × ``|p1-p2|`` → [L-α×diff, U+α×diff]
    3. Sample offspring uniformly from expanded interval
    4. Apply boundary clipping and optional crossover gating

    **String Specification Format**::

        "blend:alpha=FLOAT[,crossover_rate=FLOAT]"

    Examples::

        "blend"                                     # Defaults (alpha=0.5, crossover_rate=0.9)
        "blend:alpha=0.5"                          # Classic BLX-0.5
        "blend:alpha=0.0"                          # Uniform crossover (no expansion)
        "blend:alpha=1.0"                          # Aggressive expansion (more exploration)
        "blend:alpha=0.25,crossover_rate=0.8"      # Both custom

    Parameters
    ----------
    alpha : float, optional
        Extension factor controlling exploration magnitude.
        Valid range: [0.0, ∞)
        - alpha=0.0: Offspring uniform in [min(p1,p2), max(p1,p2)] (minimal exploration)
        - alpha=0.5: Standard BLX-α (balanced exploration beyond parent range)
        - alpha=1.0: Aggressive expansion (exploration emphasizes space beyond parents)
        - alpha≥2.0: Very large intervals (high exploration, risk of escaping useful region)
        Default: 0.5 (well-established as effective for most problems).

    crossover_rate : float, optional
        Probability that blend crossover is applied (vs returning parent unchanged).
        Valid range: [0.0, 1.0]
        Default: 0.9.

    num_offspring : int, optional
        Number of distinct offspring produced per parent pair.
        Default: 1.

    Notes
    -----
    **When to Use**:  Blend crossover is ideal for continuous optimization problems
    where you want adaptive, distance-aware exploration. It naturally adjusts exploration
    intensity based on parent proximity (nearby parents → small intervals, distant parents
    → large intervals).

    **Boundary Handling**: Offspring are clipped to the genome bounds [min_b, max_b]
    after sampling, which may introduce slight bias toward boundaries for small intervals.
    Larger alpha values mitigate this by expanding beyond parent range first.

    **Computational Complexity**: O(num_offspring × genome_dimension) per parent pair.
    Very efficient; no sorting or complex calculations.

    **Comparison to SBX**:
    - Blend: Simpler, faster, adaptive to parent distance, better for moderate exploration
    - SBX: Distribution-aware (concentrated near parents), recommended for fine-grained control

    **Trade-offs**:
    - Pros: Simple, adaptive to parent distance, less bias than uniform crossover
    - Cons: Less control over distribution shape compared to SBX
    """

    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9
    alpha: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 2 PRNG subkeys: decision (Bernoulli) and sampling (Uniform)."""
        return 2

    def _generate_noise(
        self,
        keys: chex.PRNGKey,
        config: RealGenomeConfig,
        generation: int = 0,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Tier 2 — Stochastic Payload (Heterogeneous).
        Returns tuple: (should_cross: scalar bool, random_samples: (d,) uniform[0,1]).
        Pre-allocated keys ensure deterministic noise per pair.
        """
        k_do, k_val = keys[0], keys[1]
        should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)
        float_dtype = config.dtype if jnp.issubdtype(config.dtype, jnp.floating) else jnp.float32
        random_samples = jax.random.uniform(k_val, shape=config.shape, dtype=float_dtype)
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
        Tier 1 — XLA-Fused Blend Kernel.
        Interval expansion: [cmin, cmax] = [min(p1,p2) - α·diff, max(p1,p2) + α·diff];
        sampling: cmin + u·(cmax-cmin); boundary clipping; conditional selection (should_cross).
        XLA fuses all arithmetic into single kernel.

        Returns: Offspring RealGenome with (d,) clipped values
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

        return replace(p1, values=final_values)


@struct.dataclass
class BlendCrossover_injection(BaseCrossover_injection[RealGenome, RealGenomeConfig]):
    """
    Injection-mode Blend Crossover.
    Single key splits to (n_pairs * n_offspring * 2) subkeys, reshaped (n, 2, -1).
    jax.vmap(per_row, in_axes=0) generates all masks and samples, returning tuple of
    (decision: (n,), samples: (n, d)) arrays for base wrapper to apply per (pair, offspring).

    Shape contract: Noise = (tuple of (n,) scalars, (n, d) uniform)
    Trade-off: Full materialization enables exact reproducibility.
    """

    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9
    alpha: float = 0.5

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """2 keys per (pair, offspring) — split into (n*2) total, reshaped for vmap."""
        return 2

    def _generate_noise(
        self,
        key: chex.PRNGKey,
        config: RealGenomeConfig,
        generation: int = 0,
    ) -> Tuple[chex.Array, chex.Array]:
        """Generate all (pair, offspring) decisions and samples in parallel. Returns tuple."""
        if self.input_length <= 0 or self.num_offspring <= 0:
            raise ValueError(
                "Set `input_length` and `num_offspring` before calling _generate_noise."
            )
        n = int(self.input_length * self.num_offspring)
        subkeys = jax.random.split(key, n * 2).reshape((n, 2, -1))

        def per_row(k_row: chex.Array) -> Tuple[chex.Array, chex.Array]:
            k_do, k_val = k_row[0], k_row[1]
            should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)
            float_dtype = config.dtype if jnp.issubdtype(config.dtype, jnp.floating) else jnp.float32
            random_samples = jax.random.uniform(k_val, shape=config.shape, dtype=float_dtype)
            return should_cross, random_samples

        should_crosss, random_samples = jax.vmap(per_row, in_axes=0)(subkeys)
        return should_crosss, random_samples

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: Tuple[chex.Array, chex.Array],
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:
        """XLA-fused blend kernel matching fused variant logic."""
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
        return replace(p1, values=final_values)


@struct.dataclass
class SimulatedBinaryCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """Simulated Binary Crossover (SBX) — Distribution-Aware Recombination.

    SBX simulates the behavior of single-point binary crossover on real-valued genomes.
    It generates two candidate offspring (c1, c2) via a spread factor β that concentrates
    probability near the parent values but allows distant exploration. The distribution
    index η controls concentration strength.

    **Algorithm**:

    1. For each gene:

       - Draw uniform random u ~ U[0,1]
       - Compute spread factor β based on η and u
       - Generate two candidates: c1 = 0.5×[(1+β)p1 + (1-β)p2], c2 = 0.5×[(1-β)p1 + (1+β)p2]
       - Randomly select c1 or c2

    2. Apply boundary clipping
    3. Conditionally apply (based on crossover_rate)

    **String Specification Format**::

        "simulated_binary:eta=FLOAT[,crossover_rate=FLOAT]"

    Examples::

        "simulated_binary"                          # Defaults (eta=20.0, crossover_rate=0.9)
        "simulated_binary:eta=10"                   # Less concentrated, more exploration
        "simulated_binary:eta=20"                   # Standard (balanced exploitation/exploration)
        "simulated_binary:eta=50"                   # Very concentrated near parents (exploitation)
        "simulated_binary:eta=5,crossover_rate=1.0" # Always apply, explore more

    Parameters
    ----------
    eta : float, optional
        **Distribution index** controlling concentration of offspring around parent values.
        Valid range: (0, ∞). Typical range: 2–30 for continuous optimization.
        - eta=2: Low concentration, offspring far from parents (high exploration)
        - eta=5: Balanced exploration
        - eta=10: Balanced exploration/exploitation (recommended for general problems)
        - eta=20: Strong concentration near parents (moderate exploration, fine-tuning)
        - eta=50+: Very strong concentration (heavy exploitation, late stages)
        Default: 20.0 (most commonly recommended value).

    crossover_rate : float, optional
        Probability that SBX is applied (vs returning parent unchanged).
        Valid range: [0.0, 1.0]
        Default: 0.9.

    num_offspring : int, optional
        Number of distinct offspring produced per parent pair.
        Default: 2 (SBX naturally produces two offspring).

    Notes
    -----
    **MOST RECOMMENDED for Continuous Optimization**: SBX is the industry standard
    crossover operator for real-valued evolutionary algorithms due to its theoretical
    foundation (simulates binary SPC), distribution-aware exploration, and proven performance
    across diverse benchmark problems. If in doubt, use SBX with η=10-20.

    **When to Use**:
    - Continuous optimization problems (real-valued genomes)
    - Need fine control over exploration vs exploitation balance via η
    - Want distribution-aware offspring (concentrated near parents, not uniform)
    - Solving challenging non-convex optimization problems (recommended default)

    **Distribution Index Tuning**:
    - **Early generations** (exploration): Use lower η (5-10) for more distant offspring
    - **Late generations** (exploitation): Use higher η (20-50) for fine-tuning near parents
    - **Adaptive approaches**: Reduce η over generations (rare, requires scheduler)

    **Computational Complexity**: O(num_offspring × genome_dimension) per parent pair.
    Slightly higher than Blend due to spread factor calculation, but still efficient.

    **Comparison to Blend Crossover**:
    - SBX: Distribution-aware (concentrated near parents), fine control via η, recommended
    - Blend: Simpler, adaptive to parent distance, faster but less control

    **Trade-offs**:
    - Pros: Theoretically grounded, distribution-aware, proven track record
    - Cons: Requires tuning η for different problems/phases
    """

    num_offspring: int = struct.field(pytree_node=False, default=2)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9
    eta: float = 20.0

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Requires 3 PRNG subkeys: decision (Bernoulli), spread (Uniform), swap (Bernoulli)."""
        return 3

    def _generate_noise(
        self,
        keys: chex.PRNGKey,
        config: RealGenomeConfig,
        generation: int = 0,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Tier 2 — Stochastic Payload (Heterogeneous).
        Returns: (should_cross: scalar bool, u: (d,) uniform, swap_mask: (d,) bool).
        Per-offspring key allocation ensures swap_mask varies across kernel calls,
        enabling distinct children even with identical parents.
        """
        k_do, k_beta, k_swap = keys[0], keys[1], keys[2]
        should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)
        float_dtype = config.dtype if jnp.issubdtype(config.dtype, jnp.floating) else jnp.float32
        u = jax.random.uniform(k_beta, shape=config.shape, dtype=float_dtype)
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
        Tier 1 — XLA-Fused SBX Kernel (Pure, returns single offspring).
        Per-offspring keys in base class ensure swap_mask differs across num_offspring calls,
        yielding distinct children from identical parents. XLA fuses spread factor calculation,
        child generation, selection, clipping, and conditional application.

        Returns: Offspring RealGenome with (d,) clipped values
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

        return replace(p1, values=final_values)


@struct.dataclass
class SimulatedBinaryCrossover_injection(BaseCrossover_injection[RealGenome, RealGenomeConfig]):
    """
    Injection-mode Simulated Binary Crossover (SBX).
    Single key splits to (n_pairs * n_offspring * 3) subkeys, reshaped (n, 3, -1).
    jax.vmap(per_row, in_axes=0) generates decisions, spreads, and swaps in parallel,
    returning tuple of ((n,) bools, (n, d) uniform, (n, d) bools) for per-pair-offspring.

    Shape contract: Noise = (tuple of (n,) scalars, (n, d) uniform, (n, d) bools)
    Trade-off: Full materialization enables exact reproducibility and distinctness.
    """

    num_offspring: int = struct.field(pytree_node=False, default=2)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9
    eta: float = 20.0

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """3 keys per (pair, offspring) — split into (n*3) total, reshaped for vmap."""
        return 3

    def _generate_noise(
        self,
        key: chex.PRNGKey,
        config: RealGenomeConfig,
        generation: int = 0,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Generate all (pair, offspring) decisions, spreads, and swaps in parallel."""
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
            float_dtype = config.dtype if jnp.issubdtype(config.dtype, jnp.floating) else jnp.float32
            u = jax.random.uniform(k_beta, shape=config.shape, dtype=float_dtype)
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
        """XLA-fused SBX kernel matching fused variant logic."""
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

        return replace(p1, values=final_values)


@struct.dataclass
class BinomialCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    DE Binomial Crossover — Fused 3-Tier Paradigm.
    Per-gene selection between mutant (p1) and target (p2) via Bernoulli mask, followed by
    boundary clipping. Typical use in differential evolution context (mutant ↔ candidate).

    Shape contract: Parent (d,) × Parent (d,) → Offspring (d,)
    Key budget: 1 subkey (Bernoulli mask generation)
    """

    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Bernoulli mask requires 1 PRNG subkey."""
        return 1

    def _generate_noise(
        self, keys: chex.PRNGKey, config: RealGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """Tier 2 — Bernoulli Mask. Returns (d,) boolean array for per-gene selection."""
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
        Tier 1 — XLA-Fused Binomial Kernel.
        Per-gene selection (True=mutant p1, False=target p2) fused with boundary clipping
        into single XLA kernel.

        Returns: Offspring RealGenome with (d,) clipped values
        """
        cross_mask = noise_data
        trial_values = jnp.where(cross_mask, p1.values, p2.values)
        min_val, max_val = config.bounds
        trial_values = jnp.clip(trial_values, min_val, max_val)
        return replace(p1, values=trial_values)


@struct.dataclass
class BinomialCrossover_injection(BaseCrossover_injection[RealGenome, RealGenomeConfig]):
    """
    Injection-mode Binomial Crossover.
    Single key splits into (n_pairs * n_offspring) subkeys; jax.vmap(per_row) generates all
    masks in parallel, returning (n_pairs * n_offspring, d) flattened array for base wrapper
    to apply per (pair, offspring).

    Shape contract: Noise (K, d) flattened for (pair, offspring) application
    Trade-off: Full materialization enables reproducibility without key re-splitting.
    """

    num_offspring: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    crossover_rate: float = 0.9

    @property
    def num_keys_per_atomic_operation(self) -> int:
        """Single key for Bernoulli mask generation (split internally)."""
        return 1

    def _generate_noise(
        self, key: chex.PRNGKey, config: RealGenomeConfig, generation: int = 0
    ) -> chex.Array:
        """Generate all (pair, offspring) masks upfront. Shape: (n_pairs * n_offspring, d)."""
        if self.input_length <= 0 or self.num_offspring <= 0:
            raise ValueError(
                "Set `input_length` and `num_offspring` before calling _generate_noise."
            )
        n = int(self.input_length * self.num_offspring)
        subkeys = jax.random.split(key, n)

        def per_row(k: chex.PRNGKey) -> chex.Array:
            return jax.random.bernoulli(k, p=self.crossover_rate, shape=config.shape)

        return jax.vmap(per_row)(subkeys)  # (n, d) boolean masks

    def _recombine_one(
        self,
        p1: RealGenome,
        p2: RealGenome,
        noise_data: chex.Array,
        config: RealGenomeConfig,
        **kwargs: Any,
    ) -> RealGenome:
        """XLA-fused binomial kernel: per-gene selection fused with clipping."""
        cross_mask = noise_data
        trial_values = jnp.where(cross_mask, p2.values, p1.values)
        min_val, max_val = config.bounds
        trial_values = jnp.clip(trial_values, min_val, max_val)
        return replace(p1, values=trial_values)


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
