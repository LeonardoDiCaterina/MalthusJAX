"""
Abstract base classes for genetic operators in MalthusJAX Level 2.

This module defines the operator abstractions following the new paradigm:
- @struct.dataclass for immutable, JIT-compatible operators
- Factory pattern with static/dynamic parameters
- Pure JAX functions for maximum performance
- Generic type support for all genome types
"""

from typing import Generic, TypeVar, Tuple
import jax
import chex
from flax import struct

G = TypeVar("G")  # Genome Type
C = TypeVar("C")  # Config Type

# ==========================================
# 1. MUTATION (Consumer)
# ==========================================
@struct.dataclass
class BaseMutation(Generic[G, C]):
    """
    Pure Functional Mutation. 
    Expects `keys` to be a vector of shape (num_keys,).
    """
    num_offspring: int = struct.field(pytree_node=False, default=1)

    def num_keys(self, config: C, input_shape: tuple) -> int:
        """How many keys do you need per parent?"""
        return self.num_offspring

    def __call__(self, keys: chex.Array, genome: G, config: C) -> G:
        """
        Produces multiple mutants from a single genome.
        
        Args:
            keys: Random keys. Shape: (num_offspring, 2)
            genome: Input genome (SINGLE individual, NO batch dimension)
        Returns:
            Mutated genomes. Shape: (num_offspring, ...genome_shape)
            
        Note:
            Each key in the batch produces one mutant from the SAME input genome.
            The engine vmaps this method over individuals to process populations.
        """
        # Vmap only over keys to produce num_offspring mutants from the same genome
        return jax.vmap(
            lambda k: self._mutate_one(k, genome, config),
            in_axes=(0,)
        )(keys)

    def _mutate_one(self, key: chex.PRNGKey, genome: G, config: C) -> G:
        raise NotImplementedError

# ==========================================
# 2. CROSSOVER (Consumer)
# ==========================================
@struct.dataclass
class BaseCrossover(Generic[G, C]):
    """
    Pure Functional Crossover.
    Expects `keys` to be a vector of shape (num_keys,).
    """
    num_offspring: int = struct.field(pytree_node=False, default=1)

    def num_keys(self, config: C, input_shape: tuple) -> int:
        """How many keys do you need per pair?"""
        return self.num_offspring

    def __call__(self, keys: chex.Array, p1: G, p2: G, config: C) -> G:
        """
        Produces multiple offspring from a single parent pair.
        
        Args:
            keys: Random keys. Shape: (num_offspring, 2)
            p1, p2: Parent genomes (SINGLE individuals, NO batch dimension)
        Returns:
            Offspring genomes. Shape: (num_offspring, ...genome_shape)
            
        Note:
            Each key in the batch produces one offspring from the SAME parent pair.
            The engine vmaps this method over parent pairs to process populations.
        """
        # Vmap only over keys to produce num_offspring children from the same parent pair
        return jax.vmap(
            lambda k: self._cross_one(k, p1, p2, config),
            in_axes=(0,)
        )(keys)

    def _cross_one(self, key: chex.PRNGKey, p1: G, p2: G, config: C) -> G:
        raise NotImplementedError

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