"""
ABLATION: Mutation Operators with Zero Key Allocation Cost.
These operators bypass the resource allocator and generate keys internally using jax.random.fold_in.
Used to quantify the overhead of the static resource allocation framework.
"""
from typing import Tuple
from flax import struct
import jax
import jax.numpy as jnp
import jax.random
import chex
from malthusjax.operators.base import BaseMutation
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


@struct.dataclass
class AblationGaussianMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """
    ABLATION: Gaussian Mutation with internal key generation.
    num_keys() returns 0 to signal no resource allocation.
    Keys are generated internally using jax.random.fold_in.
    """
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    clip: bool = struct.field(pytree_node=False, default=True)
    seed: int = struct.field(pytree_node=False, default=42)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """
        ABLATION: Return 1 to still go through resource allocator,
        but generate all other keys internally.
        """
        return 1

    def _mutate_one(self, keys: chex.Array, genome: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        """Atomic logic (unchanged from standard version)."""
        k_mask = keys[0]
        k_noise = keys[1]
        
        mutation_mask = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        strength = jnp.array(self.mutation_strength, dtype=genome.values.dtype)
        noise = jax.random.normal(k_noise, shape=genome.values.shape, dtype=genome.values.dtype) * strength
        
        mutated_values = jnp.where(mutation_mask, genome.values + noise, genome.values)
        
        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=mutated_values)

    def __call__(self, all_keys: chex.Array, population: RealPopulation, config: RealGenomeConfig) -> RealPopulation:
        """
        ABLATION: Override to generate keys internally instead of using pre-allocated keys.
        Keys depend on all_keys to prevent JAX constant-folding (runtime-dynamic).
        """
        # 1. Infer population size
        leaves = jax.tree_util.tree_leaves(population)
        if not leaves:
            raise ValueError("Empty Population")
        pop_size = leaves[0].shape[0]
        
        # 2. Generate keys internally with fold_in on all_keys to make it runtime-dynamic
        num_keys_needed = pop_size * self.num_offspring * self.num_keys_per_atomic_operation
        base_key = jax.random.PRNGKey(self.seed)
        # Fold in single allocated key to make key generation runtime-dynamic
        base_key = jax.random.fold_in(base_key, all_keys.reshape(-1)[0])
        
        # Split to generate the required number of keys
        all_keys_generated = jax.random.split(base_key, num_keys_needed)
        
        # Reshape to match expected format (Pop, Offspring, KeysPerOp, 2)
        all_keys_generated = all_keys_generated.reshape(
            pop_size, 
            self.num_offspring, 
            self.num_keys_per_atomic_operation,
            2
        )
        
        # 3. Call parent implementation with generated keys
        return super().__call__(all_keys_generated, population, config)


@struct.dataclass
class AblationBallMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """ABLATION: Ball Mutation with internal key generation."""
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    clip: bool = struct.field(pytree_node=False, default=True)
    seed: int = struct.field(pytree_node=False, default=42)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        return 1

    def _mutate_one(self, keys: chex.Array, genome: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        k_mask = keys[0]
        k_noise = keys[1]
        strength = jnp.array(self.mutation_strength, dtype=genome.values.dtype)
        
        mutation_mask = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        noise = jax.random.uniform(
            k_noise, 
            shape=genome.values.shape,
            minval=-strength,
            maxval=strength
        )
        
        mutated_values = jnp.where(mutation_mask, genome.values + noise, genome.values)
        
        if self.clip:
            min_val, max_val = config.bounds
            mutated_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=mutated_values)

    def __call__(self, all_keys: chex.Array, population: RealPopulation, config: RealGenomeConfig) -> RealPopulation:
        leaves = jax.tree_util.tree_leaves(population)
        if not leaves:
            raise ValueError("Empty Population")
        pop_size = leaves[0].shape[0]
        
        num_keys_needed = pop_size * self.num_offspring * self.num_keys_per_atomic_operation
        base_key = jax.random.PRNGKey(self.seed)
        # Fold in single allocated key to make key generation runtime-dynamic
        base_key = jax.random.fold_in(base_key, all_keys.reshape(-1)[0])
        
        all_keys_generated = jax.random.split(base_key, num_keys_needed)
        all_keys_generated = all_keys_generated.reshape(
            pop_size, 
            self.num_offspring, 
            self.num_keys_per_atomic_operation,
            2
        )
        
        return super().__call__(all_keys_generated, population, config)


@struct.dataclass
class AblationPolynomialMutation(BaseMutation[RealGenome, RealGenomeConfig, RealPopulation]):
    """ABLATION: Polynomial Mutation with internal key generation."""
    mutation_rate: float = 0.1
    eta: float = 20.0
    clip: bool = struct.field(pytree_node=False, default=True)
    seed: int = struct.field(pytree_node=False, default=42)
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 2

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        return 1

    def _mutate_one(self, keys: chex.Array, genome: "RealGenome", config: "RealGenomeConfig") -> "RealGenome":
        k_mask = keys[0]
        k_val = keys[1]
        
        eta = jnp.array(self.eta, dtype=genome.values.dtype)
        mutation_mask = jax.random.bernoulli(k_mask, p=self.mutation_rate, shape=genome.values.shape)
        u = jax.random.uniform(k_val, shape=genome.values.shape, dtype=genome.values.dtype)
        
        delta_q = jnp.where(
            u <= 0.5,
            jnp.power(2.0 * u, 1.0 / (self.eta + 1.0)) - 1.0,
            1.0 - jnp.power(2.0 * (1.0 - u), 1.0 / (self.eta + 1.0))
        )
        
        min_val, max_val = config.bounds
        bound_range = max_val - min_val
        delta = delta_q * bound_range 
        
        mutated_values = jnp.where(mutation_mask, genome.values + delta, genome.values)
        
        if self.clip:
            mutated_values = jnp.clip(mutated_values, min_val, max_val)
        
        return genome.replace(values=mutated_values)

    def __call__(self, all_keys: chex.Array, population: RealPopulation, config: RealGenomeConfig) -> RealPopulation:
        leaves = jax.tree_util.tree_leaves(population)
        if not leaves:
            raise ValueError("Empty Population")
        pop_size = leaves[0].shape[0]
        
        num_keys_needed = pop_size * self.num_offspring * self.num_keys_per_atomic_operation
        base_key = jax.random.PRNGKey(self.seed)
        # Fold in single allocated key to make key generation runtime-dynamic
        base_key = jax.random.fold_in(base_key, all_keys.reshape(-1)[0])
        
        all_keys_generated = jax.random.split(base_key, num_keys_needed)
        all_keys_generated = all_keys_generated.reshape(
            pop_size, 
            self.num_offspring, 
            self.num_keys_per_atomic_operation,
            2
        )
        
        return super().__call__(all_keys_generated, population, config)


__all__ = ["AblationGaussianMutation", "AblationBallMutation", "AblationPolynomialMutation"]
