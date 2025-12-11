from typing import Generic, TypeVar, Any
import jax
import jax.numpy as jnp
import chex
from flax import struct
from ..core.base import BasePopulation

G = TypeVar("G")  # Genome Type
C = TypeVar("C")  # Config Type

# ==========================================
# 1. MUTATION (Reshape-Based Vmap)
# ==========================================
@struct.dataclass
class BaseMutation(Generic[G, C]):
    """
    BaseMutation using the 'Reshape + Nested Vmap' Strategy.
    This avoids dynamic slicing and index math inside the kernel.
    """
    num_offspring: int = struct.field(pytree_node=False, default=1)
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        raise NotImplementedError

    def num_keys(self, config: C, input_shape: tuple) -> int:
        # Total keys = Offspring_Count * Keys_Per_Operation
        return self.num_offspring * self.num_keys_per_atomic_operation

    # --- 1. The Atomic Logic ---
    def _atomic_operation(self, keys: chex.Array, genome: G, config: C) -> G:
        return self._mutate_one(keys, genome, config)

    def _mutate_one(self, key: chex.Array, genome: G, config: C) -> G:
        raise NotImplementedError

    # --- 2. The Scheduler (Nested Vmap) ---
    def _double_vmap(self,
                     # Keys are already reshaped to (Pop, Offspring, Key_Dim)
                     # So we map over the Population Dimension (0)
                     keys_structured: chex.Array, 
                     population: BasePopulation,
                     config: C) -> G:
        
        # Outer Vmap: Iterate over Population (Map keys and genome)
        return jax.vmap(
            lambda p_keys, p_genome: jax.vmap(
                # Inner Vmap: Iterate over Offspring (Map p_keys)
                lambda o_keys: self._atomic_operation(o_keys, p_genome, config)
            )(p_keys) 
        )(keys_structured, population)

    # --- 3. Public Interface ---
    def __call__(self, all_keys: chex.Array, population: Any, config: C) -> G:
        """
        Args:
            all_keys: Flat tensor of keys (Total_Mutations * Keys_Per_Op, 2)
            population: Input population/batch (N_indiv, ...)
        """
        # 1. Setup Shapes
        pop_size = len(population) if hasattr(population, '__len__') else population.values.shape[0]
        
        # 2. THE FIX: Reshape keys to match the hierarchy
        # From: (Pop * Offspring * KeysPerOp, 2)
        # To:   (Pop, Offspring, KeysPerOp, 2) OR (Pop, Offspring, 2) if KeysPerOp=1
        
        # If your atomic op expects a shape of (K, 2), preserve that last dimension
        if self.num_keys_per_atomic_operation > 1:
             keys_reshaped = all_keys.reshape(pop_size, self.num_offspring, self.num_keys_per_atomic_operation, 2)
        else:
             # If it expects a single key (2,), reshape absorbs the extra dim
             keys_reshaped = all_keys.reshape(pop_size, self.num_offspring, 2)

        # 3. Execute (Fast Nested Vmap)
        nested_result = self._double_vmap(
            keys_reshaped,
            population,
            config
        )
        
        # 4. Flatten Results
        return jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), 
            nested_result
        )

# ==========================================
# 2. CROSSOVER (Index-Based Vmap)
# ==========================================
# ==========================================
# 2. CROSSOVER (Reshape-Based Vmap)
# ==========================================
@struct.dataclass
class BaseCrossover(Generic[G, C]):
    num_offspring: int = struct.field(pytree_node=False, default=1)

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return self.num_offspring

    def num_keys(self, config: C, input_shape: tuple) -> int:
        return self.num_offspring

    def _atomic_operation(self, keys: chex.Array, p1: G, p2: G, config: C) -> G:
        return self._cross_one(keys, p1, p2, config)

    def _cross_one(self, key: chex.Array, p1: G, p2: G, config: C) -> G:
        raise NotImplementedError

    def _double_vmap(self,
                     keys_structured: chex.Array, # (Num_Pairs, Offspring, Key_Dim)
                     p1_batch: Any,
                     p2_batch: Any,
                     config: C) -> G:
        
        # Outer Vmap: Iterate over Pairs (Map keys, p1, p2)
        return jax.vmap(
            lambda k_block, parent1, parent2: jax.vmap(
                # Inner Vmap: Iterate over Offspring (Map keys only)
                lambda k: self._atomic_operation(k, parent1, parent2, config)
            )(k_block)
        )(keys_structured, p1_batch, p2_batch)

    def __call__(self, all_keys: chex.Array, p1_batch: Any, p2_batch: Any, config: C) -> G:
        
        num_pairs = len(p1_batch) if hasattr(p1_batch, '__len__') else p1_batch.genes.shape[0]
        
        # Reshape Keys: (Pairs, Offspring, 2)
        # Assuming KeysPerOp = 1 for crossover usually
        keys_reshaped = all_keys.reshape(num_pairs, self.num_offspring, -1)
        
        nested_result = self._double_vmap(
            keys_reshaped,
            p1_batch,
            p2_batch,
            config
        )
        
        return jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), 
            nested_result
        )
# ==========================================
# 3. SELECTION (Consumer)
# ==========================================
@struct.dataclass
class BaseSelection:
    """
    Pure Functional Selection.
    Receives a batch of keys sized exactly for the selection algorithm.
    """
    num_selections: int = struct.field(pytree_node=False)

    def num_keys(self, input_shape: tuple) -> int:
        """
        How many keys total? 
        Default: 1 key per selection needed (e.g. Tournament).
        Override to 1 if you use global sort/roulette.
        """
        return self.num_selections

    def __call__(self, keys: chex.Array, fitness: chex.Array) -> chex.Array:
        """
        Args:
            keys: Random keys. Shape determined by num_keys().
            fitness: Population fitness array.
        Returns:
            Selected indices. Shape: (num_selections,)
        """
        raise NotImplementedError