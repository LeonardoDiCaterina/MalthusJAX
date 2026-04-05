"""TSP (Traveling Salesman Problem) fitness evaluator."""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome


@struct.dataclass
class TSPConfig(BaseEvaluatorConfig):
    """TSP configuration."""

    num_cities: int = struct.field(default=50)  # type: ignore[no-untyped-call]
    maximize: bool = struct.field(default=False)  # type: ignore[no-untyped-call] # TSP is naturally minimization


@struct.dataclass
class TSPEvaluator(BaseEvaluator[RealGenome, TSPConfig, Any]):
    """TSP evaluator using random keys (argsort) on RealGenome."""

    config: TSPConfig
    data: chex.Array = struct.field(pytree_node=False)  # type: ignore[no-untyped-call]

    @classmethod
    def create_synthetic(cls, num_cities: int = 50, seed: int = 42) -> "TSPEvaluator":
        """Create a synthetic TSP instance with random Euclidean points."""
        config = TSPConfig(num_cities=num_cities)
        
        # Consistent synthetic generation
        key = jax.random.PRNGKey(seed)
        coords = jax.random.uniform(key, (num_cities, 2))
        
        # Compute distance matrix
        diff = coords[:, jnp.newaxis, :] - coords[jnp.newaxis, :, :]
        distance_matrix = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
        
        return cls(config=config, data=distance_matrix)

    @classmethod
    def create_from_data(cls, config: Any, distance_matrix: chex.Array) -> "TSPEvaluator":
        """Create evaluator from loaded distance matrix."""
        # if distance_matrix is passed directly
        if not isinstance(config, TSPConfig):
            # If a dictionary (e.g. from registry) was passed
            num_cities = distance_matrix.shape[0]
            config = TSPConfig(
                num_cities=num_cities, 
                maximize=config.get("maximize", False) if isinstance(config, dict) else False
            )
        
        return cls(config=config, data=distance_matrix)

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate a genome's fitness on TSP.
        
        Uses Random Key encoding: the argsort of the real-valued array
        gives the permutation of cities.
        """
        # Decode real array to permutation
        tour = jnp.argsort(genome.values)
        
        # Compute total distance
        # [city1, city2, ..., cityN, city1]
        tour_shifted = jnp.roll(tour, shift=-1)
        distances = self.data[tour, tour_shifted]
        total_distance = jnp.sum(distances)

        if self.config.maximize:
            return -total_distance
            
        return total_distance
