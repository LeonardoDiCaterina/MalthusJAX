from typing import Generic, Tuple, TypeVar, Any
import jax
import jax.numpy as jnp
import chex
from flax import struct
from ..core.base import BasePopulation

G = TypeVar("G")  # Genome Type
C = TypeVar("C")  # Config Type
P = TypeVar("P")  # Population Type
# ==========================================
# 1. MUTATION (Reshape-Based Vmap)
# ==========================================
@struct.dataclass
class BaseMutation(Generic[G, C, P]):
    num_offspring: int = struct.field(pytree_node=False, default=1)
    input_length: int = struct.field(pytree_node=False, default=-1)
    
    
    @property
    def num_keys_per_atomic_operation(self) -> int:
        """
        CONTRACT: How many keys are needed to mutate ONE genome into ONE offspring?
        Must be overridden by subclasses if > 0.
        """
        raise NotImplementedError

    def num_keys(self, input_shape: Tuple[int, ...]) -> int:
        """
        CONTRACT: Calculates total keys needed for the batch.
        Used by the Resource Allocator.
        """
        # input_shape[0] is the population size.
        # NOTE: We use the local variable directly. In Flax, we cannot write 
        # self.input_length = input_shape[0] because the instance is frozen.
        pop_size = input_shape[0]
        
        return pop_size * self.num_offspring * self.num_keys_per_atomic_operation

    def set_input_length(self, length: int) -> "BaseMutation":
        """
        Returns a NEW instance with the updated input_length.
        Flax structs are immutable, so we use .replace().
        """
        return self.replace(input_length=length)

    # --- Atomic Logic (The "Kernel") ---
    def _mutate_one(self, key: chex.Array, genome: G, config: C) -> G:
        """
        Logic to produce ONE offspring from ONE parent using ONE key block.
        key shape: (num_keys_per_atomic_operation, 2)
        """
        raise NotImplementedError

    # --- 4. Execution (The Pipeline) ---
    def __call__(self, all_keys: chex.Array, population: P, config: C) -> P:
        # 1. Validation
        leaves = jax.tree_util.tree_leaves(population)
        if not leaves: raise ValueError("Empty Population")
        pop_size = leaves[0].shape[0]

        # 2. Reshape Keys (Metadata operation - Free)
        # Shape: (Pop, Offspring, KeysPerOp, 2)
        keys_reshaped = all_keys.reshape(
            pop_size, 
            self.num_offspring, 
            self.num_keys_per_atomic_operation, 
            2
        )
        
        # 3. Unwrap (Type P -> Type G)
        # We extract the inner data to keep the kernel pure
        genes: G = population.genes if hasattr(population, 'genes') else population

        # 4. Transform (Vectorized on GPU)
        def _process_population(p_keys, p_genome):
            # Inner vmap: Vectorize over Offspring
            return jax.vmap(
                lambda o_keys: self._mutate_one(o_keys, p_genome, config)
            )(p_keys)

        # Outer vmap: Vectorize over Population
        nested_genes = jax.vmap(_process_population)(keys_reshaped, genes)
        
        # 5. Flatten (Pop, Offspring, ...) -> (Pop * Offspring, ...)
        new_genes: G = jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), 
            nested_genes
        )

        # 6. Rewrap (Type G -> Type P)
        # We reconstruct the Population wrapper.
        # Note: This assumes P has a constructor that accepts 'genes'.
        # Since 'replace' would keep old (wrong-sized) fitness fields, we usually
        # prefer creating a fresh instance to ensure metadata is reset.
        return population.spawn_offspring(new_genes)
        
# ==========================================
# 2. CROSSOVER (Reshape-Based Vmap)
# ==========================================
@struct.dataclass
class BaseCrossover(Generic[G, C, P]):
    """
    BaseCrossover with strict Resource Management & Type Safety.
    Operates on PAIRS of parents to produce batches of offspring.
    """
    num_offspring: int = struct.field(pytree_node=False, default=1)
    input_length: int = struct.field(pytree_node=False, default=-1)

    # --- 1. Immutability Helper ---
    def set_input_length(self, length: int) -> "BaseCrossover[G, C, P]":
        """
        Returns a NEW instance with the updated input_length (number of PAIRS).
        """
        return self.replace(input_length=length)

    # --- 2. Resource Contract ---
    @property
    def num_keys_per_atomic_operation(self) -> int:
        """
        CONTRACT: How many keys are needed to cross TWO parents into ONE offspring?
        Defaults to 1 (usually just for mask generation). Override if > 1.
        """
        return 1

    def num_keys(self, input_shape: Tuple[int, ...] = None) -> int:
        """
        CONTRACT: Calculates total keys needed for the batch.
        input_shape[0] should be the number of PAIRS.
        """
        if input_shape is not None:
            num_pairs = input_shape[0]
        elif self.input_length != -1:
            num_pairs = self.input_length
        else:
            raise ValueError("Input length (num_pairs) unknown. Call set_input_length() or pass shape.")

        return num_pairs * self.num_offspring * self.num_keys_per_atomic_operation

    # --- 3. Atomic Logic (The "Kernel") ---
    def _cross_one(self, key_block: chex.Array, p1: G, p2: G, config: C) -> G:
        """
        Logic to produce ONE offspring from TWO parents using ONE key block.
        key_block shape: (num_keys_per_atomic_operation, 2)
        """
        raise NotImplementedError

    # --- 4. Execution (The Pipeline) ---
    def __call__(self, all_keys: chex.Array, p1_batch: P, p2_batch: P, config: C) -> P:
        """
        Args:
            all_keys: Flat key array.
            p1_batch: First parent population (N pairs).
            p2_batch: Second parent population (N pairs).
        Returns:
            New Population of size N * num_offspring.
        """
        # 1. Validation & Shape Inference
        leaves = jax.tree_util.tree_leaves(p1_batch)
        if not leaves: raise ValueError("Empty Parent 1 Batch")
        num_pairs = leaves[0].shape[0]
        
        # 2. Reshape Keys (Metadata operation - Free)
        # Shape: (NumPairs, Offspring, KeysPerOp, 2)
        keys_reshaped = all_keys.reshape(
            num_pairs, 
            self.num_offspring, 
            self.num_keys_per_atomic_operation, 
            2
        )
        
        # 3. Unwrap (Type P -> Type G)
        # Extract genes to keep the kernel pure.
        genes1: G = p1_batch.genes if hasattr(p1_batch, 'genes') else p1_batch
        genes2: G = p2_batch.genes if hasattr(p2_batch, 'genes') else p2_batch

        # 4. Transform (Vectorized on GPU)
        def _process_pairs(k_block_pairs, parent1, parent2):
            # Inner vmap: Vectorize over Offspring
            # k_block_pairs shape: (Offspring, KeysPerOp, 2)
            return jax.vmap(
                lambda o_keys: self._cross_one(o_keys, parent1, parent2, config)
            )(k_block_pairs)

        # Outer vmap: Vectorize over Pairs
        # Maps over (Keys, P1, P2) -> Returns (NumPairs, Offspring, GenomeShape)
        nested_genes = jax.vmap(_process_pairs)(keys_reshaped, genes1, genes2)
        
        # 5. Flatten (NumPairs, Offspring, ...) -> (NumPairs * Offspring, ...)
        new_genes: G = jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), 
            nested_genes
        )

        # 6. Rewrap (Type G -> Type P)
        # We use p1_batch as the template for the new population to preserve config.
        # This relies on p1_batch.spawn_offspring implementation.
        return p1_batch.spawn_offspring(new_genes)
    
# ==========================================
# 3. SELECTION (Consumer)
# ==========================================
@struct.dataclass
class BaseSelection:
    """
    BaseSelection with strict Resource Management
    Selects indices from a population based on fitness
    """
    num_selections: int = struct.field(pytree_node=False)
    input_length: int = struct.field(pytree_node=False)
    
    # --- 1. Immutability Helper ---
    def set_input_length(self, length: int) -> "BaseSelection":
        """
        Returns a NEW instance with updated input_length (Population Size).
        """
        return self.replace(input_length=length)
    
    # --- 2. Resource Contract ---
    @property
    def keys_per_selection(self) -> int:
        """
        CONTRACT: How many keys are needed for ONE selection event?
        Example: Tournament might need 1 key. Roulette might need 1 global key (handle in num_keys).
        Default is 1 per selection (Atomic strategy).
        """
        return 1
    
    def num_keys(self, input_shape: Tuple[int, ...] = None) -> int:
        """
        CONTRACT: Calculates total keys needed.
        """
        # Note: Selection keys usually depend on 'num_selections', not input_shape.
        # But we keep signature consistent.
        # Logic: Total Keys = (Number of Selections to make) * (Keys per selection)
        return self.num_selections * self.keys_per_selection

    def __call__(self, keys: chex.Array, fitness: chex.Array) -> chex.Array:
        """
        Args:
            keys: Random keys. Shape determined by num_keys().
            fitness: Population fitness array.
        Returns:
            Selected indices. Shape: (num_selections,)
        """
        raise NotImplementedError