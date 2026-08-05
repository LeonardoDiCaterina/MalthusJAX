"""
Real-valued Mutation Operators.
Optimized for H100:

1. Uses Masked Arithmetic (genome + noise * mask) instead of Branching (jnp.where).
2. Explicit casting to ensure correct dtypes (e.g., BF16)
   during random number generation and arithmetic operations.
"""

from dataclasses import replace
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random
from flax import struct

from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.engine.schedules import ScheduleType, compute_scheduled_strength
from malthusjax.operators.base import BaseMutation, _field
from malthusjax.operators.base_injection import BaseMutation_injection


@struct.dataclass
class GaussianMutation(BaseMutation[RealGenome, RealGenomeConfig]):
    """Gaussian (Normal) Mutation — Independent Per-Gene Perturbation.

    Gaussian mutation applies independent additive Gaussian noise to each gene,
    with per-gene probability control. This is the most common mutation operator
    in continuous-domain evolutionary algorithms.

    **Algorithm**:

    1. For each gene:

       - Draw Bernoulli(mutation_rate) → gene i is mutated with probability mutation_rate
       - If mutated, add N(0, mutation_strength) to gene i
       - Optionally clip to bounds

    2. Return mutated genome

    **String Specification Format**::

        "gaussian:mutation_rate=FLOAT[,mutation_strength=FLOAT]"

    Examples::

        "gaussian"                                              # Defaults (rate=0.1, strength=0.1)
        "gaussian:mutation_rate=0.05"                         # Lower rate, less mutations
        "gaussian:mutation_rate=0.2,mutation_strength=0.2"    # Both custom
        "gaussian:mutation_strength=0.05"                     # Milder mutations
        "gaussian:mutation_rate=1.0,mutation_strength=0.01"   # All genes affected, small noise

    Parameters
    ----------
    mutation_rate : float, optional
        **Per-gene mutation probability**: Fraction of genes affected per mutation event.
        Valid range: [0.0, 1.0]

        - mutation_rate=0.0: No mutation (disabled)
        - mutation_rate=0.05: 5% of genes mutated (typical for large genomes, d≥100)
        - mutation_rate=0.1: 10% mutated (common default, works well for d≈10-100)
        - mutation_rate=1.0: All genes mutated every generation (strong perturbation)

        Default: 0.1 (recommended starting point).

        **Typical values by genome dimension**:

        - d=1-10: mutation_rate ≈ 0.15-0.3 (higher, because fewer genes)
        - d=10-100: mutation_rate ≈ 0.05-0.15 (moderate)
        - d=100+: mutation_rate ≈ 0.01-0.05 (lower, to avoid excessive perturbation)

    mutation_strength : float, optional
        **Gaussian noise standard deviation**: Controls magnitude of per-gene perturbations.
        Valid range: (0, ∞), but typically [0.01, 1.0]

        - mutation_strength=0.01: Very small noise (fine-tuning near optima)
        - mutation_strength=0.1: Moderate noise (balanced exploration/exploitation)
        - mutation_strength=1.0: Large noise (aggressive exploration)

        Default: 0.1 (works well when genome is roughly normalized to [-1, 1]).

        **Interaction with genome bounds**: If your genome bounds are [low, high],
        set mutation_strength ≈ 0.05-0.1 × (high - low) for reasonable perturbations.

    clip : bool, optional
        If True, clip mutated values to genome bounds [min_b, max_b].
        If False, allow violations (clipping happens at evaluation boundary).
        Default: False (for compatibility with landscape exploration).

    schedule_type : ScheduleType, optional
        Mutation strength schedule across generations (CONSTANT, LINEAR_DECAY, etc.).
        Default: ScheduleType.CONSTANT.

    final_strength : float, optional
        Target mutation strength at final generation (used if schedule_type != CONSTANT).
        Default: 0.0 (decay to zero).

    Notes
    -----
    **INTERACTION BETWEEN mutation_rate AND mutation_strength**:
    These two parameters control different aspects of mutation:
    - `mutation_rate`: **Which** genes are modified (controls sparsity)
    - `mutation_strength`: **How much** each modified gene changes (controls perturbation scale)

    **Example scenarios**:
    1. **Exploration phase**: Higher mutation_rate (0.2) + higher mutation_strength (0.2)
       → Many genes perturbed by large amounts
    2. **Exploitation/fine-tuning**: Lower mutation_rate (0.05) + lower mutation_strength (0.05)
       → Few genes, small noise
    3. **Balanced default**: mutation_rate=0.1, mutation_strength=0.1
       → ~10% of genes affected, each by ~0.1 std units

    **When to Use**: Gaussian mutation is the recommended operator for continuous
    optimization. It's simple, effective, and works well across nearly all problem types.
    Use it unless you have a specific reason to prefer other mutations (e.g., Ball mutation
    for constrained, high-dimensional problems).

    **Computational Complexity**: O(genome_dimension) per individual.
    Very efficient; no sorting or complex operations.

    **Best Practices**:
    - Start with mutation_rate=1.0/genome_dimension (standard recommendation)
    - Tune mutation_strength based on your problem scale (not genome dimension)
    - Use scheduling to decrease strength over generations for fine-tuning
    - Monitor diversity and adjust rate if offspring differ too little from parents
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
        Tier 2 — Noise generation for Gaussian mutation.
        This produces a noise array shaped like the genome that already includes
        strength scaling and masking, allowing the arithmetic kernel to simply add it.
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
        return replace(genome, values=mutated_values)


@struct.dataclass
class GaussianMutation_injection(BaseMutation_injection[RealGenome, RealGenomeConfig]):
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
            strength = jnp.asarray(self.mutation_strength)
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

        return replace(genome, values=mutated_values)


@struct.dataclass
class BallMutation(BaseMutation[RealGenome, RealGenomeConfig]):
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
        Tier 2 — Noise generation using Muller's method.
        The returned array encodes a uniformly distributed perturbation within
        a hypersphere, already masked by the mutation rate.
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
            radius = jnp.asarray(self.radius)
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

        return replace(genome, values=mutated_values)


@struct.dataclass
class BallMutation_injection(BaseMutation_injection[RealGenome, RealGenomeConfig]):
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
            radius = jnp.asarray(self.radius)
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

        return replace(genome, values=mutated_values)


@struct.dataclass
class PolynomialMutation(BaseMutation[RealGenome, RealGenomeConfig]):
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
        Tier 2 — Noise generation for polynomial mutation.
        Produces an array of deltas scaled by the genome’s bound range and
        masked by the mutation rate, ready for addition in the arithmetic
        kernel.
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

        return replace(genome, values=mutated_values)


@struct.dataclass
class PolynomialMutation_injection(BaseMutation_injection[RealGenome, RealGenomeConfig]):
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

        return replace(genome, values=mutated_values)


__all__ = [
    "GaussianMutation",
    "BallMutation",
    "PolynomialMutation",
    "GaussianMutation_injection",
    "BallMutation_injection",
    "PolynomialMutation_injection",
    "BatchedGaussianMutation",
]

@struct.dataclass
class BatchedGaussianMutation(GaussianMutation):
    """Batched Gaussian Mutation (Monolithic Execution).
    
    This operator completely bypasses the generic JAX vmap structure.
    It expects to be called on a batched RealPopulation, where `genes.values`
    is a `(pop_size, d)` array. It generates a single monolithic noise tensor
    and applies it all at once, matching EvoSAX's raw execution speed.
    """
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def num_keys(self, input_shape: tuple[int, ...]) -> int:
        # We only ever need 2 keys, regardless of population size.
        return self.num_keys_per_atomic_operation

    def __call__(
        self, all_keys: chex.Array, population: Any, config: RealGenomeConfig, generation: int = 0
    ) -> Any:
        k_mask = all_keys[0] if len(all_keys.shape) == 2 else all_keys[0][0]
        k_noise = all_keys[1] if len(all_keys.shape) == 2 else all_keys[0][1]
        
        strength = compute_scheduled_strength(
            self.mutation_strength, self.final_strength, generation, self.max_generations, self.schedule_type
        )
        
        genes = population.genes.values
        mask = jax.random.bernoulli(k_mask, self.mutation_rate, shape=genes.shape)
        noise = jax.random.normal(k_noise, shape=genes.shape) * strength
        
        new_values = genes + mask * noise
        if self.clip:
            min_val, max_val = config.bounds
            new_values = jnp.clip(new_values, min_val, max_val)
            
        return population.replace(genes=RealGenome(values=new_values))
