"""
Real-valued genome implementation for continuous optimization problems.

Provides RealGenome and RealPopulation for floating-point vector representations
commonly used in evolutionary algorithms for function optimization.
"""

from typing import ClassVar, Tuple, Type

import chex  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from flax import struct  # type: ignore

from malthusjax.core.base import BaseGenome, BasePopulation


@struct.dataclass
class RealGenomeConfig:
    """Configuration for Real-valued genomes."""
    length: int           # Number of real values in the vector
    bounds: Tuple[float, float] = struct.field(  # (min, max) bounds for each value
        pytree_node=False, default=(-jnp.inf, jnp.inf)
    )
    dtype: jnp.dtype = struct.field(pytree_node=False, default=jnp.float32)  # Data type of the values

def validate_real_config(config: RealGenomeConfig) -> None:
    """Validate real genome configuration parameters."""
    if config.length <= 0:
        raise ValueError(f"Length must be positive, got {config.length}")
    if config.bounds[0] >= config.bounds[1]:
        raise ValueError(f"Lower bound must be less than upper bound, got {config.bounds}")


@struct.dataclass
class RealGenome(BaseGenome):
    """
    Real-valued genome for continuous optimization.
    
    Represents solutions as vectors of floating-point numbers for problems
    like function optimization, neural network weights, parameter tuning, etc.
    """
    values: chex.Array  # Shape (length,) - Real-valued array

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: RealGenomeConfig) -> "RealGenome":
        """Create random real-valued genome within bounds."""
        min_val, max_val = config.bounds
        values = jax.random.uniform(
            key, (config.length,), minval=min_val, maxval=max_val, dtype=config.dtype
        )
        return cls(values=values)

    def autocorrect(self, config: RealGenomeConfig) -> "RealGenome":
        """Clip values to be within specified bounds."""
        min_val, max_val = config.bounds
        corrected_values = jnp.clip(self.values, min_val, max_val)
        return self.replace(values=corrected_values)

    def distance(self, other: "RealGenome", metric: str = "euclidean") -> float:
        """Compute distance between real genomes."""
        if metric == "euclidean":
            return jnp.sqrt(jnp.sum((self.values - other.values) ** 2)).astype(jnp.float32)
        elif metric == "manhattan":
            return jnp.sum(jnp.abs(self.values - other.values)).astype(jnp.float32)
        elif metric == "hamming":
            # For real values, use a threshold-based approach
            # Use 1% of the value range as threshold
            value_range = jnp.max(self.values) - jnp.min(self.values) + 1e-8
            threshold = 0.01 * value_range
            return jnp.sum(jnp.abs(self.values - other.values) > threshold).astype(jnp.float32)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @property
    def size(self) -> int:
        """Return number of real values."""
        return self.values.shape[-1]

    @property
    def shape(self) -> tuple:
        """Return shape of the genome values."""
        return self.values.shape

    def magnitude(self) -> float:
        """Compute L2 norm (magnitude) of the genome vector."""
        return float(jnp.sqrt(jnp.sum(self.values ** 2)))

    def normalize(self) -> "RealGenome":
        """Normalize the genome to unit length."""
        norm = self.magnitude()
        if norm > 0:
            normalized_values = self.values / norm
        else:
            normalized_values = self.values
        return self.replace(values=normalized_values)

    def add_noise(self, key: chex.PRNGKey, noise_std: float = 0.1) -> "RealGenome":
        """Add Gaussian noise to the genome."""
        noise = jax.random.normal(key, self.values.shape) * noise_std
        noisy_values = self.values + noise
        return self.replace(values=noisy_values)

    def __repr__(self) -> str:
        try:
            # Avoid formatting traced values during JIT compilation
            if self.size <= 5:
                values_str = ", ".join(f"{float(v):.3f}" for v in self.values)
            else:
                values_str = ", ".join(f"{float(v):.3f}" for v in self.values[:3])
                values_str += f", ..., {float(self.values[-1]):.3f}"
            return f"<RealGenome([{values_str}], len={self.size})>"
        except:
            # Fallback for traced values or other issues
            return f"<RealGenome(shape={self.values.shape})>"


@struct.dataclass
class RealPopulation(BasePopulation[RealGenome]):
    """Population container for RealGenome objects."""
    genes: RealGenome
    fitness: chex.Array
    config: RealGenomeConfig = struct.field(pytree_node=False)

    GENOME_CLS: ClassVar[Type[RealGenome]] = RealGenome

    @classmethod
    def init_random(cls, key: chex.PRNGKey, config: RealGenomeConfig, size: int) -> "RealPopulation":
        """Create random population of real genomes."""
        batched_genes = RealGenome.create_population(key, config, size)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)
