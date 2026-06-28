"""Real‑valued genome types and utilities.

Implements continuous genome representations along with vector norms,
noise injection and basic algebraic helpers. Also provides the parallel
population container specialized for real vectors.
"""

from __future__ import annotations

from typing import Any, Tuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation
from malthusjax.core.base import DistanceMetric as BaseDistances

_field: Any = struct.field


class RealDistanceMetric(BaseDistances):
    """Metrics specific to real-valued vectors."""

    # could add COSINE = "cosine" here later
    pass


@struct.dataclass
class RealGenomeConfig:
    """Static configuration for continuous real-valued genomes.

    This class defines the search space and numerical properties for all genomes
    created with this configuration. All individuals share the same shape and bounds.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the real-valued vector. Defines the dimensionality and array structure.
        Examples:
        - shape=(10,) → 10-dimensional vector [d0, d1, ..., d9]
        - shape=(5, 3) → 5×3 matrix (15 total parameters)
        - shape=(4, 4, 4) → 4×4×4 tensor (64 total parameters)
        Default: () (empty shape for backward compatibility)

    bounds : tuple[float, float]
        Valid range [lower, upper] applied **uniformly to all genes**.
        - lower: Minimum allowed value (e.g., -10.0)
        - upper: Maximum allowed value (e.g., +10.0)
        - All genes will be initialized uniformly within [lower, upper]
        Default: (-∞, +∞) (unbounded)

        Examples:
        - bounds=(-1.0, 1.0) → normalized range
        - bounds=(0.0, 100.0) → positive-only range
        - bounds=(-5.12, 5.12) → Rastrigin function standard
        - bounds=(-512.0, 512.0) → Schwefel function standard

    dtype : type or jnp.dtype
        JAX floating-point dtype for all gene values.
        - jnp.float32 (default): 32-bit precision, GPU-efficient
        - jnp.float64: 64-bit precision, higher accuracy but slower
        Default: jnp.float32

    Notes
    -----
    ** CRITICAL: Bounds Enforcement**:

    Bounds are **enforced during initialization** (via :meth:`RealGenome.random_init`).
    However, bounds are **NOT automatically enforced after mutation or crossover**.

    If your mutation operators can push values outside [lower, upper], you **MUST**:
    1. Enable clipping in the mutation operator (clip=True for GaussianMutation)
    2. **OR** call :meth:`RealGenome.autocorrect()` after mutation
    3. **OR** accept that individuals may operate outside bounds (and handle in fitness)

    **Example Workflow**:

    Without clipping (allowing out-of-bounds):

    - Configuration: bounds=[-1, 1]
    - After mutation: values might become [-1.5, 1.2] (outside bounds)
    - Evaluator must handle these values (e.g., clip before evaluation, or penalize)

    With clipping enabled:

    - Use GaussianMutation(..., clip=True)
    - OR manually: mutated_genome = mutated_genome.autocorrect(config)
    - Guarantees values stay in [lower, upper]

    **When to Use Unbounded Search**:
    - Research/exploration: Allow temporary excursions outside nominal bounds
    - Penalty-based approaches: Evaluator penalizes out-of-bounds genomes
    - Adaptive problems: Bounds change during evolution (recompute dynamically)

    **When to Enforce Bounds**:
    - Constrained optimization: Hard-constraint problems
    - Physics simulations: Physical laws require values in specific ranges
    - Benchmark problems: Standard bounds are part of problem definition
    """

    shape: Tuple[int, ...] = _field(pytree_node=False, default_factory=lambda: ())
    bounds: Tuple[float, float] = _field(pytree_node=False, default=(-jnp.inf, jnp.inf))
    dtype: type[jnp.floating[Any]] | jnp.dtype[jnp.floating[Any]] = _field(
        pytree_node=False, default=jnp.float32
    )

    def init_population(self, key: chex.PRNGKey, size: int) -> BasePopulation[RealGenome]:
        """Create a random population from this config (protocol method for JR-2)."""
        return RealPopulation.init_random(key, self, size)


@struct.dataclass
class RealGenome(BaseGenome):
    """
    A continuous, real-valued genome representation.

    This genome is represented as a 1D vector of floating-point numbers. It is
    ideal for function optimization, neural network weight evolution, and
    parameter estimation where the search space is a manifold in R^n.

    When part of a RealPopulation, the 'values' array becomes 2D (N, length),
    where N is the population size.
    """

    values: chex.Array
    # Enable indexing & iteration by default for convenience
    subscriptable: bool = _field(pytree_node=False, default=True)

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: RealGenomeConfig) -> RealGenome:
        """
        Samples a genome from a uniform distribution within the configured bounds.
        """
        min_val, max_val = config.bounds
        values = jax.random.uniform(
            key, config.shape, minval=min_val, maxval=max_val, dtype=config.dtype
        )
        return cls(values=values)

    def autocorrect(self, config: RealGenomeConfig) -> RealGenome:
        """
        Clamps the genome values to ensure they remain within the hypercube
        defined by the configuration bounds.
        """
        min_val, max_val = config.bounds
        corrected_values = jnp.clip(self.values, min_val, max_val)
        return cast(RealGenome, cast(Any, self).replace(values=corrected_values))

    def distance(self, other: BaseGenome, metric: str = "euclidean") -> chex.Numeric:
        """
        Computes the distance between this genome and another in continuous space.

        The *other* genome is cast internally to :class:`RealGenome`. Supported
        metrics include 'euclidean' (L2), 'manhattan' (L1), and an approximate
        'hamming' based on a value threshold.
        """
        other_real = cast(RealGenome, other)

        if metric == "euclidean":
            # Standard L2 Norm
            return jnp.sqrt(jnp.sum(jnp.square(self.values - other_real.values)))
        elif metric == "manhattan":
            # Standard L1 Norm
            return jnp.sum(jnp.abs(self.values - other_real.values))
        elif metric == "hamming":
            # Approximated Hamming: treats values as different if they exceed
            # 1% of the observed value range.
            value_range = jnp.max(self.values) - jnp.min(self.values) + 1e-8
            threshold = 0.01 * value_range
            return jnp.sum(jnp.abs(self.values - other_real.values) > threshold)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @property
    def size(self) -> int:
        """Returns the length of the vector."""
        return int(self.values.shape[-1])

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the array shape of the genome values."""
        return cast(tuple[int, ...], self.values.shape)

    def magnitude(self) -> chex.Numeric:
        """Calculates the L2 norm (magnitude) of the genome vector."""
        return jnp.sqrt(jnp.sum(jnp.square(self.values)))

    def normalize(self) -> RealGenome:
        """
        Scales the genome to unit length (norm=1).
        Utilizes jnp.where to prevent division by zero during JIT compilation.
        """
        norm = self.magnitude()
        norm_safe = jnp.maximum(norm, 1e-8)
        normalized_values = jnp.where(norm > 0, self.values / norm_safe, self.values)
        return cast(RealGenome, cast(Any, self).replace(values=normalized_values))

    def add_noise(self, key: chex.PRNGKey, noise_std: float = 0.1) -> RealGenome:
        """
        Applies Gaussian jitter to the genome, often used as a mutation operator.
        """
        noise = jax.random.normal(key, self.values.shape) * noise_std
        noisy_values = self.values + noise
        return cast(RealGenome, cast(Any, self).replace(values=noisy_values))

    @classmethod
    def from_tensor(cls, arr: chex.Array, config: Any = None) -> "RealGenome":
        """Construct a batched RealGenome from a raw array.

        This is intentionally a trivial factory: it wraps the provided array in
        the `RealGenome` dataclass. Keep the implementation pure and avoid
        Python-side validation so it remains JIT-traceable.
        """
        return cls(values=arr)


@struct.dataclass
class RealPopulation(BasePopulation[RealGenome]):
    """
    A specialized container for a population of RealGenomes.

    This container ensures that all internal genes are correctly typed and
    that the population-wide config matches the RealGenome requirements.
    """

    genes: RealGenome
    fitness: chex.Array
    config: RealGenomeConfig = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    # TODO: support multidimensional genome populations by accepting `shape` tuples
    #       instead of a scalar `size` for non-1D genomes.
    @classmethod
    def init_random(cls, key: chex.PRNGKey, config: RealGenomeConfig, size: int) -> RealPopulation:
        """
        Orchestrates the parallel creation of 'size' random real genomes.
        """
        batched_genes = RealGenome.create_population(key, config, size)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)
