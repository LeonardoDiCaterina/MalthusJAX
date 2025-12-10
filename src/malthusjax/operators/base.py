from typing import Generic, TypeVar, Any
import jax
import jax.numpy as jnp
import chex
from flax import struct
from ..core.base import BasePopulation

G = TypeVar("G")  # Genome Type
C = TypeVar("C")  # Config Type

# ==========================================
# 1. MUTATION (Index-Based Vmap)
# ==========================================
@struct.dataclass
class BaseMutation(Generic[G, C]):
    """
    BaseMutation using the 'Double Vmap' Indexing Strategy.
    """
    num_offspring: int = struct.field(pytree_node=False, default=1)
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        return self.num_offspring

    def num_keys(self, config: C, input_shape: tuple) -> int:
        return self.num_offspring

    # --- 1. The Atomic Logic (User Implementation) ---
    def _atomic_operation(self, keys: chex.Array, genome: G, config: C) -> G:
        """
        Implementation specific logic (Aliases to _mutate_one for compatibility).
        """
        # We bridge to the existing concrete method name
        return self._mutate_one(keys, genome, config)

    def _mutate_one(self, key: chex.Array, genome: G, config: C) -> G:
        raise NotImplementedError

    # --- 2. The Data Retrieval (Index -> Data) ---
    def _atomic_operation_from_index(self,
                                     keys_index: int,
                                     genome_index: int,
                                     all_keys: chex.Array,
                                     population: BasePopulation, # Or generic G batch
                                     config: C) -> G:
        
        # Dynamic Slice for keys: [start : start + N]
        # keys_index is a scalar tracer here from the inner vmap
        keys_slice = jax.lax.dynamic_slice(
            all_keys, 
            (keys_index, 0), 
            (self.num_keys_per_atomic_operation, all_keys.shape[1])
        )
        
        # Simple Gather for the genome
        # Use tree_map to extract genome at index from PyTree structure
        genome = jax.tree_util.tree_map(lambda x: x[genome_index], population)
        
        # Depending on your specific concrete implementation, 
        # _mutate_one might expect (N, 2) keys or (2,) key.
        # If _mutate_one expects a single key but num_offspring > 1, 
        # you might need another vmap here or pass the slice.
        # For now, we pass the slice (N, 2).
        return self._atomic_operation(keys_slice, genome, config)

    # --- 3. The Scheduler (Double Vmap) ---
    def _double_vmap(self,
                     population_arange: chex.Array,
                     num_offspring_arange: chex.Array,
                     all_keys: chex.Array,
                     population: BasePopulation,
                     config: C) -> G:
        
        # Scale offspring indices to match key consumption
        # e.g., if we need 2 keys per op: [0, 1] -> [0, 2]
        key_start_indices = num_offspring_arange * self.num_keys_per_atomic_operation
        
        # Outer Vmap: Iterate over Population
        return jax.vmap(
            lambda g_idx: jax.vmap(
                lambda k_idx: self._atomic_operation_from_index(
                    k_idx,
                    g_idx,
                    all_keys,
                    population,
                    config
                )
            )(key_start_indices) # Inner Vmap: Iterate over Offspring count
        )(population_arange)

    # --- 4. Public Interface ---
    def __call__(self, all_keys: chex.Array, population: Any, config: C) -> G:
        """
        Args:
            all_keys: Flat tensor of keys (total_mutations, 2)
            population: Input population/batch (N_indiv, ...)
        Returns:
            Nested batch of mutants (N_indiv, num_offspring, ...)
        """
        # 1. Setup Ranges
        # If population is a PyTree, getting size is slightly different, 
        # but usually shape[0] of a leaf works.
        # Here assuming population has __len__ or .shape
        print("BaseMutation __call__ all_keys shape:", all_keys.shape)
        print("BaseMutation type(population):", type(population))
        print("BaseMutation population:", population)
        
        pop_size = len(population) if hasattr(population, '__len__') else population.values.shape[0]
        
        population_arange = jnp.arange(pop_size)
        num_offspring_arange = jnp.arange(self.num_offspring)
        
        # 2. Execute Double Vmap
        nested_result = self._double_vmap(
            population_arange,
            num_offspring_arange,
            all_keys,
            population,
            config
        )
        
        # 3. Flatten (Optional, but usually required by Engine)
        # Result is currently (N_pop, N_offspring, ...). 
        # We flatten to (N_pop * N_offspring, ...)
        return jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), 
            nested_result
        )


# ==========================================
# 2. CROSSOVER (Index-Based Vmap)
# ==========================================
@struct.dataclass
class BaseCrossover(Generic[G, C]):
    """
    BaseCrossover using the 'Double Vmap' Indexing Strategy.
    """
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

    def _atomic_operation_from_index(self,
                                     keys_index: int,
                                     pair_index: int,
                                     all_keys: chex.Array,
                                     p1_batch: Any, 
                                     p2_batch: Any,
                                     config: C) -> G:
        
        # Dynamic Slice Keys
        keys_slice = jax.lax.dynamic_slice(
            all_keys, 
            (keys_index, 0), 
            (self.num_keys_per_atomic_operation, all_keys.shape[1])
        )
        
        # Gather Parents using tree_map for PyTree compatibility
        p1 = jax.tree_util.tree_map(lambda x: x[pair_index], p1_batch)
        p2 = jax.tree_util.tree_map(lambda x: x[pair_index], p2_batch)
        
        return self._atomic_operation(keys_slice, p1, p2, config)

    def _double_vmap(self,
                     pair_arange: chex.Array,
                     offspring_arange: chex.Array,
                     all_keys: chex.Array,
                     p1_batch: Any,
                     p2_batch: Any,
                     config: C) -> G:
        
        key_start_indices = offspring_arange * self.num_keys_per_atomic_operation
        
        return jax.vmap(
            lambda p_idx: jax.vmap(
                lambda k_idx: self._atomic_operation_from_index(
                    k_idx,
                    p_idx,
                    all_keys,
                    p1_batch,
                    p2_batch,
                    config
                )
            )(key_start_indices)
        )(pair_arange)

    def __call__(self, all_keys: chex.Array, p1_batch: Any, p2_batch: Any, config: C) -> G:
        """
        Args:
            all_keys: Flat keys (num_pairs * num_offspring, 2)
            p1_batch: (num_pairs, ...)
            p2_batch: (num_pairs, ...)
        """
        # 1. Setup Ranges
        # Assume p1_batch is indexable
        num_pairs = len(p1_batch) if hasattr(p1_batch, '__len__') else p1_batch.genes.shape[0]
        
        pair_arange = jnp.arange(num_pairs)
        offspring_arange = jnp.arange(self.num_offspring)
        
        # 2. Double Vmap
        nested_result = self._double_vmap(
            pair_arange,
            offspring_arange,
            all_keys,
            p1_batch,
            p2_batch,
            config
        )
        
        # 3. Flatten (num_pairs * num_offspring, ...)
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