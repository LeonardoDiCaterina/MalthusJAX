"""Multi-Objective Population with self-sorting semantics."""

from __future__ import annotations

from typing import Any, Optional, Dict, cast
import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation

from malthusjax.core.genome.mo.sorting import (
    compute_dominance_matrix,
    compute_pareto_ranks,
    compute_crowding_distance,
)


@struct.dataclass
class MOPopulation(BasePopulation):
    """A self-sorting population container for Multi-Objective evolution.
    
    This class perfectly mirrors the $\mu+\lambda$ survival and NSGA-II 
    tournament selection paradigms inherently as pure functions.
    """
    
    # We guarantee these are populated for MOPopulation
    pareto_rank: chex.Array = struct.field(default=None)
    crowding_distance: chex.Array = struct.field(default=None)
    maximize: bool = struct.field(pytree_node=False, default=False)
    
    @classmethod
    def from_evaluated(cls, base_pop: BasePopulation, maximize: bool = False) -> 'MOPopulation':
        """Upgrades a standard evaluated population to an MOPopulation by computing fronts."""
        dom_matrix = compute_dominance_matrix(base_pop.fitness, maximize)
        ranks = compute_pareto_ranks(dom_matrix)
        crowding = compute_crowding_distance(base_pop.fitness, ranks)
        
        return cls(
            genes=base_pop.genes,
            fitness=base_pop.fitness,
            config=base_pop.config,
            info=base_pop.info,
            pareto_rank=ranks,
            crowding_distance=crowding,
            maximize=maximize
        )
        
    def merge(self, offspring_pop: BasePopulation) -> 'MOPopulation':
        """Concatenates this population with offspring and recalculates fronts.
        
        This forms the merged pool for $\mu+\lambda$ survival.
        """
        # Merge genes
        merged_genes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), 
            self.genes, 
            offspring_pop.genes
        )
        # Merge fitness
        merged_fitness = jnp.concatenate([self.fitness, offspring_pop.fitness], axis=0)
        
        # Merge info dict if it exists (handling edge cases)
        merged_info = None
        if self.info is not None and offspring_pop.info is not None:
            merged_info = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                self.info,
                offspring_pop.info
            )
            
        # Re-evaluate ranks and crowding on the combined pool
        dom_matrix = compute_dominance_matrix(merged_fitness, self.maximize)
        ranks = compute_pareto_ranks(dom_matrix)
        crowding = compute_crowding_distance(merged_fitness, ranks)
        
        return self.replace(
            genes=merged_genes,
            fitness=merged_fitness,
            info=merged_info,
            pareto_rank=ranks,
            crowding_distance=crowding
        )

    def truncate(self, pop_size: int) -> 'MOPopulation':
        """Truncates the population back down to pop_size based on NSGA-II elitism.
        
        Sorts by rank ascending, then crowding descending.
        """
        # Sort keys: ranks first, then -crowding (lexsort processes last-to-first)
        sort_keys = (-self.crowding_distance, self.pareto_rank)
        idx = jnp.lexsort(sort_keys)
        
        survivor_idx = idx[:pop_size]
        
        # Slicer helper
        def slice_tree(tree):
            if tree is None:
                return None
            return jax.tree_util.tree_map(lambda x: x[survivor_idx], tree)
            
        return self.replace(
            genes=slice_tree(self.genes),
            fitness=self.fitness[survivor_idx],
            info=slice_tree(self.info),
            pareto_rank=self.pareto_rank[survivor_idx],
            crowding_distance=self.crowding_distance[survivor_idx]
        )

    def select(self, key: chex.Array, batch_size: int) -> 'MOPopulation':
        """Performs NSGA-II binary tournament selection to generate a batch of parents.
        
        Returns a new MOPopulation object representing the selected parents.
        """
        pop_size = self.fitness.shape[0]
        
        idx1_key, idx2_key = jax.random.split(key)
        idx1 = jax.random.randint(idx1_key, (batch_size,), 0, pop_size)
        idx2 = jax.random.randint(idx2_key, (batch_size,), 0, pop_size)
        
        rank1 = self.pareto_rank[idx1]
        rank2 = self.pareto_rank[idx2]
        crowding1 = self.crowding_distance[idx1]
        crowding2 = self.crowding_distance[idx2]
        
        # Winner is idx1 if (rank1 < rank2) OR (rank1 == rank2 AND crowding1 > crowding2)
        idx1_wins = (rank1 < rank2) | ((rank1 == rank2) & (crowding1 > crowding2))
        
        winner_idx = jnp.where(idx1_wins, idx1, idx2)
        
        # Slicer helper
        def slice_tree(tree):
            if tree is None:
                return None
            return jax.tree_util.tree_map(lambda x: x[winner_idx], tree)
            
        return self.replace(
            genes=slice_tree(self.genes),
            fitness=self.fitness[winner_idx],
            info=slice_tree(self.info),
            pareto_rank=self.pareto_rank[winner_idx],
            crowding_distance=self.crowding_distance[winner_idx]
        )
