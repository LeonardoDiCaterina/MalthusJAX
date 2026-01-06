"""
ABLATION: Crossover Operators with Zero Key Allocation Cost.
These operators bypass the resource allocator and generate keys internally using jax.random.fold_in.
Used to quantify the overhead of the static resource allocation framework.
"""
from typing import Tuple
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jar
import chex
from malthusjax.operators.base import BaseCrossover
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


@struct.dataclass
class AblationUniformCrossover(BaseCrossover[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    ABLATION: Uniform Crossover with internal key generation.
    num_keys() returns 0 to signal no resource allocation.
    """
    crossover_rate: float = 0.5
    seed: int = struct.field(pytree_node=False, default=42)
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def num_keys(self, input_shape: Tuple[int, ...] = None) -> int:
        """ABLATION: Return 1 to engage allocator but generate keys internally."""
        return 1

    def _cross_one(self, keys: chex.PRNGKey, p1: "RealGenome", p2: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        """Atomic logic (unchanged from standard version)."""
        rng = keys[0]
        mask = jar.bernoulli(rng, p=self.crossover_rate, shape=p1.values.shape)
        new_values = jnp.where(mask, p2.values, p1.values)
        return p1.replace(values=new_values)

    def __call__(self, all_keys: chex.Array, p1_batch: RealPopulation, p2_batch: RealPopulation, config: RealGenomeConfig) -> RealPopulation:
        """ABLATION: Override to generate keys internally."""
        leaves = jax.tree_util.tree_leaves(p1_batch)
        if not leaves:
            raise ValueError("Empty Parent 1 Batch")
        num_pairs = leaves[0].shape[0]
        
        num_keys_needed = num_pairs * self.num_offspring * self.num_keys_per_atomic_operation
        base_key = jax.random.PRNGKey(self.seed)
        # Fold in single allocated key to make key generation runtime-dynamic
        base_key = jax.random.fold_in(base_key, all_keys.reshape(-1)[0])
        
        all_keys_generated = jax.random.split(base_key, num_keys_needed)
        all_keys_generated = all_keys_generated.reshape(
            num_pairs, 
            self.num_offspring, 
            self.num_keys_per_atomic_operation,
            2
        )
        
        return super().__call__(all_keys_generated, p1_batch, p2_batch, config)


@struct.dataclass
class AblationBlendCrossover(BaseCrossover["RealGenome", "RealGenomeConfig", "RealPopulation"]):
    """ABLATION: Blend Crossover with internal key generation."""
    crossover_rate: float = 0.9 
    alpha: float = 0.5
    seed: int = struct.field(pytree_node=False, default=42)
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def num_keys(self, input_shape: Tuple[int, ...] = None) -> int:
        """ABLATION: Return 1 to engage allocator but generate keys internally."""
        return 1

    def _cross_one(self, keys: chex.Array, p1: "RealGenome", p2: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        """Atomic logic (unchanged from standard version)."""
        k_do = keys[0]
        k_val = keys[1]
        
        diff = jnp.abs(p1.values - p2.values)
        alpha = jnp.array(self.alpha, dtype=p1.values.dtype)
        
        cmin = jnp.minimum(p1.values, p2.values) - (alpha * diff)
        cmax = jnp.maximum(p1.values, p2.values) + (alpha * diff)
        
        random_vals = jax.random.uniform(k_val, shape=p1.values.shape)
        offspring_values = cmin + random_vals * (cmax - cmin)
        
        if hasattr(config, 'min_value') and hasattr(config, 'max_value'):
             offspring_values = jnp.clip(offspring_values, config.min_value, config.max_value)
             
        should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)
        final_values = jnp.where(should_cross, offspring_values, p1.values)
        
        return p1.replace(values=final_values)

    def __call__(self, all_keys: chex.Array, p1_batch: RealPopulation, p2_batch: RealPopulation, config: RealGenomeConfig) -> RealPopulation:
        leaves = jax.tree_util.tree_leaves(p1_batch)
        if not leaves:
            raise ValueError("Empty Parent 1 Batch")
        num_pairs = leaves[0].shape[0]
        
        num_keys_needed = num_pairs * self.num_offspring * self.num_keys_per_atomic_operation
        base_key = jax.random.PRNGKey(self.seed)
        # Fold in single allocated key to make key generation runtime-dynamic
        base_key = jax.random.fold_in(base_key, all_keys.reshape(-1)[0])
        
        all_keys_generated = jax.random.split(base_key, num_keys_needed)
        all_keys_generated = all_keys_generated.reshape(
            num_pairs, 
            self.num_offspring, 
            self.num_keys_per_atomic_operation,
            2
        )
        
        return super().__call__(all_keys_generated, p1_batch, p2_batch, config)


@struct.dataclass
class AblationSimulatedBinaryCrossover(BaseCrossover["RealGenome", "RealGenomeConfig", "RealPopulation"]):
    """ABLATION: Simulated Binary Crossover with internal key generation."""
    crossover_rate: float = 0.9
    eta: float = 20.0
    seed: int = struct.field(pytree_node=False, default=42)
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 3

    def num_keys(self, input_shape: Tuple[int, ...] = None) -> int:
        """ABLATION: Return 1 to engage allocator but generate keys internally."""
        return 1

    def _cross_one(self, keys: chex.Array, p1: "RealGenome", p2: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        """Atomic logic (unchanged from standard version)."""
        k_do = keys[0]
        k_beta = keys[1]
        k_swap = keys[2]
        
        u = jax.random.uniform(k_beta, shape=p1.values.shape, dtype=p1.values.dtype)
        
        exponent = jnp.array(1.0 / (self.eta + 1.0), dtype=p1.values.dtype)
        beta = jnp.where(
            u <= 0.5,
            (2.0 * u) ** exponent,
            (1.0 / (2.0 * (1.0 - u))) ** exponent
        )
        
        c1 = 0.5 * ((1.0 + beta) * p1.values + (1.0 - beta) * p2.values)
        c2 = 0.5 * ((1.0 - beta) * p1.values + (1.0 + beta) * p2.values)
        
        swap_mask = jax.random.bernoulli(k_swap, p=0.5, shape=p1.values.shape)
        child_vals = jnp.where(swap_mask, c2, c1)
        
        if hasattr(config, 'min_value') and hasattr(config, 'max_value'):
             child_vals = jnp.clip(child_vals, config.min_value, config.max_value)

        should_cross = jax.random.bernoulli(k_do, p=self.crossover_rate)
        final_values = jnp.where(should_cross, child_vals, p1.values)
        
        return p1.replace(values=final_values)

    def __call__(self, all_keys: chex.Array, p1_batch: RealPopulation, p2_batch: RealPopulation, config: RealGenomeConfig) -> RealPopulation:
        leaves = jax.tree_util.tree_leaves(p1_batch)
        if not leaves:
            raise ValueError("Empty Parent 1 Batch")
        num_pairs = leaves[0].shape[0]
        
        num_keys_needed = num_pairs * self.num_offspring * self.num_keys_per_atomic_operation
        base_key = jax.random.PRNGKey(self.seed)
        # Fold in single allocated key to make key generation runtime-dynamic
        base_key = jax.random.fold_in(base_key, all_keys.reshape(-1)[0])
        
        all_keys_generated = jax.random.split(base_key, num_keys_needed)
        all_keys_generated = all_keys_generated.reshape(
            num_pairs, 
            self.num_offspring, 
            self.num_keys_per_atomic_operation,
            2
        )
        
        return super().__call__(all_keys_generated, p1_batch, p2_batch, config)


__all__ = ['AblationUniformCrossover', 'AblationSimulatedBinaryCrossover', 'AblationBlendCrossover']
