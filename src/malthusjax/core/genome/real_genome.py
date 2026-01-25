from __future__ import annotations
from typing import Any, ClassVar, Tuple, Type, cast

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome, BasePopulation

from malthusjax.core.base import DistanceMetric as BaseDistances

class RealDistanceMetric(BaseDistances):
    """Metrics specific to real-valued vectors."""
    # could add COSINE = "cosine" here later
    pass

@struct.dataclass
class RealGenomeConfig:
    """
    Static configuration defining the search space for real-valued genomes.
    
    Attributes:
        length: The dimensionality of the real-valued vector.
        bounds: A tuple (min, max) defining the valid search range for all elements.
        dtype: The numerical precision (e.g., jnp.float32 or jnp.float64).
    """
    length: int
    bounds: Tuple[float, float] = struct.field(
        pytree_node=False, default=(-jnp.inf, jnp.inf)
    ) # type: ignore[no-untyped-call]
    dtype: type[jnp.floating[Any]] | jnp.dtype[jnp.floating[Any]] = struct.field(
        pytree_node=False, default=jnp.float32 # type: ignore[no-untyped-call]
    )

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
    values: chex.Array  # Shape: (length,) for individuals, (N, length) for populations

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: RealGenomeConfig) -> RealGenome:
        """
        Samples a genome from a uniform distribution within the configured bounds.
        """
        min_val, max_val = config.bounds
        values = jax.random.uniform(
            key, (config.length,), minval=min_val, maxval=max_val, dtype=config.dtype
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
        
        Args:
            other: Another genome (cast to RealGenome internally).
            metric: The distance type. Supports 'euclidean' (L2), 'manhattan' (L1), 
                   and a thresholded 'hamming' distance.
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


@struct.dataclass
class RealPopulation(BasePopulation[RealGenome]):
    """
    A specialized container for a population of RealGenomes.
    
    This container ensures that all internal genes are correctly typed and 
    that the population-wide config matches the RealGenome requirements.
    """
    genes: RealGenome
    fitness: chex.Array
    config: RealGenomeConfig = struct.field(pytree_node=False) # type: ignore[no-untyped-call]

    GENOME_CLS: ClassVar[Type[RealGenome]] = RealGenome

    @classmethod
    def init_random(
        cls, key: chex.PRNGKey, config: RealGenomeConfig, size: int
    ) -> RealPopulation:
        """
        Orchestrates the parallel creation of 'size' random real genomes.
        """
        batched_genes = RealGenome.create_population(key, config, size)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)