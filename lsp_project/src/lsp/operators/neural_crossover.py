"""Neural Crossover utilities.

Provides composable crossover operators that simultaneously exchange 
discrete topological features and interpolate continuous neural parameters.
"""

from typing import Any
import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome
from malthusjax.operators.base import BaseCrossover

# ---------------------------------------------------------------------------
# Continuous Blend Crossover
# ---------------------------------------------------------------------------

@struct.dataclass
class ContinuousBlendCrossover(BaseCrossover[BaseGenome, Any]):
    """Standard continuous interpolation (Blend Crossover) for continuous arrays.
    
    Generates a uniform blend factor alpha ~ U(-blend_alpha, 1 + blend_alpha)
    and computes: offspring = alpha * p1 + (1 - alpha) * p2
    """
    blend_alpha: float = struct.field(pytree_node=False, default=0.5)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def _generate_noise(self, keys: chex.Array, config: Any, generation: int = 0) -> chex.Array:
        """Tier 2 — sample interpolation scalar alpha."""
        return jax.random.uniform(
            keys[0], shape=(), minval=-self.blend_alpha, maxval=1.0 + self.blend_alpha
        )

    def _recombine_one(
        self, p1: BaseGenome, p2: BaseGenome, noise_data: chex.Array, config: Any, **_kwargs: Any
    ) -> BaseGenome:
        """Tier 1 — apply linear interpolation to float arrays."""
        alpha = noise_data
        
        def interpolate(x, y):
            if jnp.issubdtype(x.dtype, jnp.floating):
                return alpha * x + (1.0 - alpha) * y
            return x 
            
        return jax.tree_util.tree_map(interpolate, p1, p2)


# ---------------------------------------------------------------------------
# Hybrid Neural Crossover
# ---------------------------------------------------------------------------

@struct.dataclass
class HybridNeuralCrossover(BaseCrossover[BaseGenome, Any]):
    """Composes a discrete topological crossover with a continuous weight crossover.
    
    Applies the topology_crossover to the discrete arrays (ops, args) and the 
    weight_crossover to the continuous arrays (weights, biases), resolving them 
    simultaneously for Differentiable CGP / MEP architectures.
    """
    topology_crossover: Any = struct.field(pytree_node=False, default=None)
    weight_crossover: Any = struct.field(pytree_node=False, default=None)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return (
            self.topology_crossover.num_keys_per_atomic_operation +
            self.weight_crossover.num_keys_per_atomic_operation
        )

    def _generate_noise(self, keys: chex.Array, config: Any, generation: int = 0) -> tuple[Any, Any]:
        """Tier 2 — generate noise for both sub-operators."""
        tk = self.topology_crossover.num_keys_per_atomic_operation
        keys_top = keys[:tk]
        keys_w = keys[tk:]
        
        noise_top = self.topology_crossover._generate_noise(keys_top, config, generation)
        noise_w = self.weight_crossover._generate_noise(keys_w, config, generation)
        
        return (noise_top, noise_w)

    def _recombine_one(
        self, p1: BaseGenome, p2: BaseGenome, noise_data: tuple[Any, Any], config: Any, **_kwargs: Any
    ) -> BaseGenome:
        """Tier 1 — merge topological crossover with continuous weight blend."""
        noise_top, noise_w = noise_data
        
        # 1. Apply topology crossover
        offspring_top = self.topology_crossover._recombine_one(p1, p2, noise_top, config)
        
        # 2. Apply weight crossover
        offspring_w = self.weight_crossover._recombine_one(p1, p2, noise_w, config)
        
        # 3. Merge: Take discrete fields from offspring_top, and continuous fields from offspring_w
        def merge(x_top, x_w):
            if jnp.issubdtype(x_top.dtype, jnp.floating):
                return x_w
            return x_top
            
        merged = jax.tree_util.tree_map(merge, offspring_top, offspring_w)
        return merged
