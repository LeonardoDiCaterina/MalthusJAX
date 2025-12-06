from malthusjax.operators.base import BaseCrossover
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from flax import struct
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
import chex # type: ignore

@struct.dataclass
class UniformCrossover(BaseCrossover[BinaryGenome, BinaryGenomeConfig]):
    """
    Uniform Crossover using new batch-first paradigm.
    
    Produces multiple offspring where each bit comes from parent1 or parent2
    based on crossover probability.
    """
    # --- DYNAMIC PARAMS (runtime tunable) ---
    crossover_rate: float = 0.5
    
    def _cross_one(self, key: chex.PRNGKey, p1: BinaryGenome, p2: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        """Create one offspring via uniform crossover."""
        mask = jar.bernoulli(key, p=self.crossover_rate, shape=p1.bits.shape)
        offspring_bits = jnp.where(mask, p1.bits, p2.bits)
        return BinaryGenome(bits=offspring_bits)

@struct.dataclass
class SinglePointCrossover(BaseCrossover[BinaryGenome, BinaryGenomeConfig]):
    """
    Single-point crossover using new batch-first paradigm.
    
    Creates offspring by swapping segments at a random crossover point.
    """
    
    def _cross_one(self, key: chex.PRNGKey, p1: BinaryGenome, p2: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        """Create one offspring via single-point crossover."""
        length = p1.bits.shape[0]
        crossover_point = jar.randint(key, shape=(), minval=1, maxval=length)
        
        # Use masking approach for JAX compatibility
        indices = jnp.arange(length)
        mask = indices < crossover_point
        offspring_bits = jnp.where(mask, p1.bits, p2.bits)
        return BinaryGenome(bits=offspring_bits)

    # --- KERNEL IDENTITY CARD ---
    def num_keys(self, config: BinaryGenomeConfig, input_shape: tuple) -> int:
        """Single-point crossover requires a single PRNG key."""
        return 1

    def get_output_shape(self, config: BinaryGenomeConfig, input_shape: tuple) -> tuple:
        """Output shape matches a single genome (no leading offspring axis)."""
        return input_shape

    def apply_kernel(self, keys: chex.Array, p1: BinaryGenome, p2: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        """Kernel-style single-point crossover using a pre-allocated key.

        Args:
            keys: single PRNG key or array of keys (uses first element)
            p1, p2: parent genomes
            config: genome config

        Returns:
            Offspring BinaryGenome (same shape as parents)
        """
        # Accept both a single key (shape (2,)) and an array of keys (shape (N,2)).
        # Only index into the array when keys.ndim == 2 (multiple keys).
        if hasattr(keys, "ndim") and getattr(keys, "ndim") == 2:
            key = keys[0]
        else:
            key = keys

        length = p1.bits.shape[0]
        # randint uses [minval, maxval) so choose 1..length-1 inclusive split
        cut = jar.randint(key, shape=(), minval=1, maxval=length)
        indices = jnp.arange(length)
        mask = indices < cut
        offspring_bits = jnp.where(mask, p1.bits, p2.bits)
        return BinaryGenome(bits=offspring_bits)

__all__ = ["UniformCrossover", "SinglePointCrossover"]