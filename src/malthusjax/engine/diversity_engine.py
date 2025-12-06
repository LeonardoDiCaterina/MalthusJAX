"""
Diversity-Aware Genetic Engine Extension.

This module demonstrates how to extend the GeneticEngine by overriding
the selection mechanism to incorporate distance matrix information,
promoting population diversity alongside fitness optimization.
"""
import jax
import jax.numpy as jnp
import jax.random as jar
import chex
from flax import struct
from typing import Any

from .genetic_engine import GeneticEngine, GeneticEngineParams
from .base import AbstractEvolutionState
from ..core.base import BasePopulation


@struct.dataclass
class DiversityAwareEngine(GeneticEngine):
    """
    Extended Genetic Engine that uses distance-based diversity metrics during selection.
    
    This engine overrides the parent selection mechanism to balance fitness and diversity.
    Instead of purely fitness-based tournament selection, it:
    1. Computes the population distance matrix
    2. Selects parents that are both fit AND diverse
    3. Uses a weighted combination of fitness and crowding distance
    
    Key Parameters:
        diversity_weight: float in [0, 1] controlling fitness vs diversity trade-off
            - 0.0: Pure fitness-based selection (standard GA)
            - 1.0: Pure diversity-based selection
            - 0.3: Balanced approach (recommended default)
    """
    
    diversity_weight: float = struct.field(default=0.3, pytree_node=False)
    distance_metric: str = struct.field(default="hamming", pytree_node=False)
    
    def _compute_crowding_scores(self, population: BasePopulation) -> chex.Array:
        """
        Compute crowding distance scores for each individual.
        
        Higher scores indicate more isolated (diverse) individuals.
        """
        # Compute distance matrix: (pop_size, pop_size)
        dist_matrix = population.distance_matrix(metric=self.distance_metric)
        
        # Crowding score = average distance to all other individuals
        # Set diagonal to 0 to exclude self-distance
        dist_matrix = dist_matrix.at[jnp.diag_indices_from(dist_matrix)].set(0.0)
        
        # Sum distances to get crowding measure (higher = more isolated)
        crowding_scores = jnp.sum(dist_matrix, axis=1)
        
        return crowding_scores
    
    def _compute_diversity_fitness(
        self, 
        fitness: chex.Array, 
        crowding: chex.Array
    ) -> chex.Array:
        """
        Combine fitness and diversity into a single selection criterion.
        """
        # Respect optimization direction: convert fitness so that higher is better
        opt_sign = jnp.where(self.evaluator.config.maximize, 1.0, -1.0)
        adj_fitness = fitness * opt_sign

        # Normalize both to [0, 1] range for fair weighting
        fitness_min, fitness_max = jnp.min(adj_fitness), jnp.max(adj_fitness)
        crowding_min, crowding_max = jnp.min(crowding), jnp.max(crowding)

        fitness_norm = (adj_fitness - fitness_min) / (fitness_max - fitness_min + 1e-8)
        crowding_norm = (crowding - crowding_min) / (crowding_max - crowding_min + 1e-8)
        
        # Weighted combination
        combined = (1 - self.diversity_weight) * fitness_norm + self.diversity_weight * crowding_norm
        
        return combined
    
    def _select_parents(self, key: chex.PRNGKey, state: AbstractEvolutionState, params: GeneticEngineParams) -> BasePopulation:
        """
        Override parent selection to incorporate diversity.
        
        New Signature: (key, state, params)
        """
        population = state.population
        
        # 1. Compute diversity metrics
        crowding_scores = self._compute_crowding_scores(population)
        
        # 2. Create diversity-aware fitness
        diversity_fitness = self._compute_diversity_fitness(
            population.fitness, 
            crowding_scores
        )
        
        # 3. Use the base selection operator with diversity-aware fitness
        # Note: We pass diversity_fitness instead of raw fitness
        indices = self.selection(key, diversity_fitness)
        
        # CRITICAL FIX: Flatten indices to avoid (N, 1, 1) dimension errors in downstream merge
        indices = indices.flatten()
        
        # Return selected parents with ORIGINAL fitness values
        # (fitness is used for actual evaluation, diversity only for selection)
        return population[indices]
    
    def _select_elites(self, key: chex.PRNGKey, state: AbstractEvolutionState, params: GeneticEngineParams) -> chex.ArrayTree:
        """
        Override elite selection to balance fitness and diversity.
        
        New Signature: (key, state, params)
        Strategy:
        - Top 70% by fitness (exploitation)
        - Top 30% by diversity (exploration)
        """
        n_elites = params.elitism
        population = state.population
        
        if n_elites == 0:
            # Return empty structure matching genome shape
            return jax.tree_util.tree_map(lambda x: x[:0], population.genes)
        
        # Split elites between fitness and diversity
        n_fitness_elites = int(n_elites * 0.7)
        n_diversity_elites = n_elites - n_fitness_elites
        
        # 1. Select top fitness elites (respect optimization direction)
        opt_sign = jnp.where(self.evaluator.config.maximize, 1.0, -1.0)
        _, fitness_elite_indices = jax.lax.top_k(population.fitness * opt_sign, n_fitness_elites)
        
        # 2. Select top diversity elites
        crowding_scores = self._compute_crowding_scores(population)
        _, diversity_elite_indices = jax.lax.top_k(crowding_scores, n_diversity_elites)
        
        # 3. Combine indices
        elite_indices = jnp.concatenate([fitness_elite_indices, diversity_elite_indices])
        
        # Return elite genes (chex.ArrayTree)
        return population[elite_indices].genes