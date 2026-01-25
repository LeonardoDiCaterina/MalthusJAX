"""
Binary genome implementation for combinatorial optimization problems.

Provides BinaryGenome and BinaryPopulation for bit string representations
commonly used in genetic algorithms for binary optimization problems.
"""

from typing import ClassVar, Type

import chex  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from flax import struct  # type: ignore

from malthusjax.core.base import BaseGenome, BasePopulation


@struct.dataclass
class BinaryGenomeConfig:
    """Configuration for Binary genomes."""
    length: int           # Number of bits in the binary string
    p: float = 0.5        # Probability of 1 in random initialization
    dtype: jnp.dtype = struct.field(pytree_node=False, default=jnp.int32)  # Data type of the bits


def validate_binary_config(config: BinaryGenomeConfig) -> None:
    """Validate binary genome configuration parameters."""
    if config.length <= 0:
        raise ValueError(f"Length must be positive, got {config.length}")


@struct.dataclass
class BinaryGenome(BaseGenome):
    """
    Binary genome for combinatorial optimization.
    
    Represents solutions as bit strings (0s and 1s) for problems like
    knapsack, subset selection, feature selection, etc.
    """
    bits: chex.Array  # Shape (length,) - Binary array

    @classmethod
    def random_init(cls, key: chex.PRNGKey, config: BinaryGenomeConfig) -> "BinaryGenome":
        """Create random binary genome."""
        bits = jax.random.bernoulli(key, config.p, (config.length,)).astype(config.dtype)
        return cls(bits=bits)

    def autocorrect(self, config: BinaryGenomeConfig) -> "BinaryGenome":
        """Ensure all bits are 0 or 1."""
        corrected_bits = jnp.clip(self.bits, 0, 1).astype(config.dtype)
        return self.replace(bits=corrected_bits)

    def distance(self, other: "BinaryGenome", metric: str = "hamming") -> float:
        """Compute distance between binary genomes."""
        if metric == "hamming":
            return jnp.sum(self.bits != other.bits).astype(jnp.float32)
        elif metric == "euclidean":
            return jnp.sqrt(jnp.sum((self.bits - other.bits) ** 2)).astype(jnp.float32)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @property
    def size(self) -> int:
        """Return number of bits."""
        return self.bits.shape[-1]

    @property
    def shape(self) -> tuple:
        """Return shape of genome as (length,)."""
        return self.bits.shape

    def to_int(self) -> int:
        """Convert binary genome to integer representation."""
        # Convert to numpy for standard Python int
        bits_np = jnp.array(self.bits)
        return int(jnp.sum(bits_np * (2 ** jnp.arange(len(bits_np)))))

    def count_ones(self) -> int:
        """Count number of 1s in the genome."""
        return int(jnp.sum(self.bits))

    def flip_bit(self, index: int) -> "BinaryGenome":
        """Flip a single bit at given index."""
        new_bits = self.bits.at[index].set(1 - self.bits[index])
        return self.replace(bits=new_bits)

    def __repr__(self) -> str:
        bits_str = "".join(str(int(b)) for b in self.bits[:10])
        if self.size > 10:
            bits_str += "..."
        return f"<BinaryGenome({bits_str}, len={self.size})>"


@struct.dataclass
class BinaryPopulation(BasePopulation[BinaryGenome]):
    """Population container for BinaryGenome objects."""
    genes: BinaryGenome
    fitness: chex.Array
    config: BinaryGenomeConfig = struct.field(pytree_node=False)

    GENOME_CLS: ClassVar[Type[BinaryGenome]] = BinaryGenome

    @classmethod
    def init_random(cls, key: chex.PRNGKey, config: BinaryGenomeConfig, size: int) -> "BinaryPopulation":
        """Create random population of binary genomes."""
        batched_genes = BinaryGenome.create_population(key, config, size)
        initial_fitness = jnp.full((size,), -jnp.inf)
        return cls(genes=batched_genes, fitness=initial_fitness, config=config)
