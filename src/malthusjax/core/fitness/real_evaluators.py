"""Fitness evaluators for real-valued genomes.

This module provides classic continuous optimization benchmark functions
including Griewank, Sphere, and constrained Box optimization problems.
"""

from typing import Any, Tuple

import jax.numpy as jnp
import jax.random as jr
from flax import struct

from ..genome.real_genome import RealGenome
from .base import BaseEvaluator, BaseEvaluatorConfig


@struct.dataclass
class SphereConfig(BaseEvaluatorConfig):
    """Configuration for Sphere function optimization.
    
    The Sphere function: f(x) = sum(x_i^2)
    Global minimum: f(0, 0, ..., 0) = 0
    """
    pass


@struct.dataclass
class SphereEvaluator(BaseEvaluator[RealGenome, SphereConfig, Any]):
    """Sphere function fitness evaluator.
    
    The Sphere function is one of the simplest continuous optimization
    benchmark functions. It has a single global minimum at the origin.
    f(x) = sum(x_i^2)
    """

    config: SphereConfig
    data: Any = struct.field(pytree_node=False, default=None)

    def evaluate(self, genome: RealGenome) -> float:
        """Evaluate a single real genome on Sphere function.
        
        Args:
            genome: RealGenome to evaluate
            
        Returns:
            Fitness value (sum of squares)
        """
        sphere_value = jnp.sum(genome.values ** 2).astype(jnp.float32)
        if self.config.maximize:
            return sphere_value
        else:
            return -sphere_value


@struct.dataclass
class GriewankConfig(BaseEvaluatorConfig):
    """Configuration for Griewank function optimization.
    
    The Griewank function: f(x) = 1 + (1/4000)*sum(x_i^2) - prod(cos(x_i/sqrt(i)))
    Global minimum: f(0, 0, ..., 0) = 0
    Typically evaluated on domain [-600, 600]^n
    """
    pass


@struct.dataclass
class GriewankEvaluator(BaseEvaluator[RealGenome, GriewankConfig, Any]):
    """Griewank function fitness evaluator.
    
    The Griewank function is a multimodal benchmark with many local optima.
    It combines a quadratic trend with cosine modulation.
    f(x) = 1 + (1/4000)*sum(x_i^2) - prod(cos(x_i/sqrt(i+1)))
    """

    config: GriewankConfig
    data: Any = struct.field(pytree_node=False, default=None)

    def evaluate(self, genome: RealGenome) -> float:
        """Evaluate a single real genome on Griewank function.
        
        Args:
            genome: RealGenome to evaluate
            
        Returns:
            Fitness value (Griewank function value)
        """
        x = genome.values

        # Quadratic term
        quad_term = jnp.sum(x ** 2) / 4000.0

        # Cosine product term
        indices = jnp.arange(1, len(x) + 1, dtype=jnp.float32)
        cos_term = jnp.prod(jnp.cos(x / jnp.sqrt(indices)))

        griewank_value = (1.0 + quad_term - cos_term).astype(jnp.float32)

        if self.config.maximize:
            return griewank_value  # Negative for minimization
        else:
            return -griewank_value


@struct.dataclass
class BoxConfig(BaseEvaluatorConfig):
    """Configuration for Box-constrained optimization.
    
    Constrained optimization problem where the goal is to stay within
    specified bounds while optimizing an objective function.
    """
    target_point: jnp.ndarray  # Target point to reach
    box_bounds: Tuple[jnp.ndarray, jnp.ndarray]  # (lower_bounds, upper_bounds)
    penalty_factor: float = 1000.0  # Penalty for violating constraints
    objective_type: str = struct.field(
        pytree_node=False, default="distance"
    )  # Static configuration


@struct.dataclass
class BoxEvaluator(BaseEvaluator[RealGenome, BoxConfig, Any]):
    """Box-constrained optimization fitness evaluator.
    
    Evaluates real genomes on constrained optimization problems.
    The goal is to minimize distance to a target point while staying
    within specified box constraints.
    """

    config: BoxConfig
    data: Any = struct.field(pytree_node=False, default=None)

    def evaluate(self, genome: RealGenome) -> float:
        """Evaluate a single real genome on box-constrained problem.
        
        Args:
            genome: RealGenome to evaluate
            
        Returns:
            Fitness value (negative distance with constraint penalties)
        """
        x = genome.values
        lower, upper = self.config.box_bounds

        # Calculate objective function
        if self.config.objective_type == "distance":
            objective = jnp.sqrt(jnp.sum((x - self.config.target_point) ** 2))
        elif self.config.objective_type == "sphere":
            centered = x - self.config.target_point
            objective = jnp.sum(centered ** 2)
        else:
            raise ValueError(f"Unknown objective type: {self.config.objective_type}")

        # Calculate constraint violations
        lower_violations = jnp.maximum(0, lower - x)
        upper_violations = jnp.maximum(0, x - upper)
        total_violation = jnp.sum(lower_violations) + jnp.sum(upper_violations)

        # Apply penalty
        penalty = total_violation * self.config.penalty_factor

        # Return negative (for maximization, since we want to minimize distance)
        return (-objective - penalty).astype(jnp.float32)

    @staticmethod
    def create_random_problem(key: jnp.ndarray, dimensions: int,
                            box_size: float = 10.0, maximize: bool = False) -> BoxConfig:
        """Create a random box-constrained optimization problem.
        
        Args:
            key: JAX random key
            dimensions: Problem dimensions
            box_size: Size of the constraint box
            
        Returns:
            BoxConfig for the random problem
        """
        key1, key2 = jr.split(key, 2)

        # Random target point
        target = jr.uniform(key1, (dimensions,), minval=-box_size/2, maxval=box_size/2)

        # Box bounds centered around target
        margin = box_size / 4
        lower = target - margin
        upper = target + margin

        return BoxConfig(
            target_point=target,
            box_bounds=(lower, upper),
            objective_type="distance",
            maximize=maximize
        )
