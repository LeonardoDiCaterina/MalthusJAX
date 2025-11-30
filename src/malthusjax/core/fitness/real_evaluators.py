"""Fitness evaluators for real-valued genomes.

This module provides classic continuous optimization benchmark functions
including Griewank, Sphere, and constrained Box optimization problems.
"""

from typing import Optional, Tuple, List
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from ..base import BaseGenome
from ..genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


@struct.dataclass
class SphereConfig:
    """Configuration for Sphere function optimization.
    
    The Sphere function: f(x) = sum(x_i^2)
    Global minimum: f(0, 0, ..., 0) = 0
    """
    maximize: bool = struct.field(pytree_node=False, default=True)  # Static configuration
    

@struct.dataclass
class SphereEvaluator:
    """Sphere function fitness evaluator.
    
    The Sphere function is one of the simplest continuous optimization
    benchmark functions. It has a single global minimum at the origin.
    f(x) = sum(x_i^2)
    """
    
    config: SphereConfig
        
    def evaluate_single(self, genome: RealGenome) -> float:
        """Evaluate a single real genome on Sphere function.
        
        Args:
            genome: RealGenome to evaluate
            
        Returns:
            Fitness value (sum of squares)
        """
        sphere_value = float(jnp.sum(genome.values ** 2))
        if self.config.maximize:
            return sphere_value 
        else:
            return -sphere_value
            
    def evaluate_batch(self, population: RealPopulation) -> jnp.ndarray:
        """Evaluate a population of real genomes."""
        fitness_fn = self.get_tensor_fitness_function()
        fitness_values = fitness_fn(population.genes.values)
        return fitness_values
            
    def get_tensor_fitness_function(self):
        """Get pure JAX function for batch evaluation."""
        
        def _sphere_fitness(values_batch: jnp.ndarray) -> jnp.ndarray:
            """Pure JAX function for Sphere fitness.
            
            Args:
                values_batch: Shape (batch_size, dimensions) real array
                
            Returns:
                Fitness values of shape (batch_size,)
            """
            sphere_values = jnp.sum(values_batch ** 2, axis=1)
            if self.config.maximize:
                return sphere_values  # Negative for minimization
            else:
                return -sphere_values
                
        return _sphere_fitness


@struct.dataclass
class GriewankConfig:
    """Configuration for Griewank function optimization.
    
    The Griewank function: f(x) = 1 + (1/4000)*sum(x_i^2) - prod(cos(x_i/sqrt(i)))
    Global minimum: f(0, 0, ..., 0) = 0
    Typically evaluated on domain [-600, 600]^n
    """
    maximize: bool = struct.field(pytree_node=False, default=True)  # Static configuration
    

@struct.dataclass
class GriewankEvaluator:
    """Griewank function fitness evaluator.
    
    The Griewank function is a multimodal benchmark with many local optima.
    It combines a quadratic trend with cosine modulation.
    f(x) = 1 + (1/4000)*sum(x_i^2) - prod(cos(x_i/sqrt(i+1)))
    """
    
    config: GriewankConfig
        
    def evaluate_single(self, genome: RealGenome) -> float:
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
        
        griewank_value = float(1.0 + quad_term - cos_term)
        
        if self.config.maximize:
            return griewank_value  # Negative for minimization
        else:
            return -griewank_value
            
    def evaluate_batch(self, population: RealPopulation) -> jnp.ndarray:
        """Evaluate a population of real genomes."""
        fitness_fn = self.get_tensor_fitness_function()
        fitness_values = fitness_fn(population.genes.values)
        return fitness_values
            
    def get_tensor_fitness_function(self):
        """Get pure JAX function for batch evaluation."""
        
        def _griewank_fitness(values_batch: jnp.ndarray) -> jnp.ndarray:
            """Pure JAX function for Griewank fitness.
            
            Args:
                values_batch: Shape (batch_size, dimensions) real array
                
            Returns:
                Fitness values of shape (batch_size,)
            """
            batch_size, dimensions = values_batch.shape
            
            # Quadratic terms
            quad_terms = jnp.sum(values_batch ** 2, axis=1) / 4000.0
            
            # Cosine product terms
            indices = jnp.arange(1, dimensions + 1, dtype=jnp.float32)
            cos_terms = jnp.prod(jnp.cos(values_batch / jnp.sqrt(indices)[None, :]), axis=1)
            
            griewank_values = 1.0 + quad_terms - cos_terms
            
            if self.config.maximize:
                return griewank_values  # Negative for minimization
            else:
                return -griewank_values
                
        return _griewank_fitness


@struct.dataclass
class BoxConfig:
    """Configuration for Box-constrained optimization.
    
    Constrained optimization problem where the goal is to stay within
    specified bounds while optimizing an objective function.
    """
    target_point: jnp.ndarray  # Target point to reach
    box_bounds: Tuple[jnp.ndarray, jnp.ndarray]  # (lower_bounds, upper_bounds)
    penalty_factor: float = 1000.0  # Penalty for violating constraints
    objective_type: str = struct.field(pytree_node=False, default="distance")  # Static configuration
            

@struct.dataclass
class BoxEvaluator:
    """Box-constrained optimization fitness evaluator.
    
    Evaluates real genomes on constrained optimization problems.
    The goal is to minimize distance to a target point while staying
    within specified box constraints.
    """
    
    config: BoxConfig
        
    def evaluate_single(self, genome: RealGenome) -> float:
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
        return float(-objective - penalty)
        
    def evaluate_batch(self, population: RealPopulation) -> jnp.ndarray:
        """Evaluate a population of real genomes."""
        fitness_fn = self.get_tensor_fitness_function()
        fitness_values = fitness_fn(population.genes.values)
        return fitness_values
        
    def get_tensor_fitness_function(self):
        """Get pure JAX function for batch evaluation."""
        
        def _box_fitness(values_batch: jnp.ndarray) -> jnp.ndarray:
            """Pure JAX function for box-constrained fitness.
            
            Args:
                values_batch: Shape (batch_size, dimensions) real array
                
            Returns:
                Fitness values of shape (batch_size,)
            """
            lower, upper = self.config.box_bounds
            target = self.config.target_point
            
            # Calculate objective function
            if self.config.objective_type == "distance":
                diff = values_batch - target[None, :]
                objectives = jnp.sqrt(jnp.sum(diff ** 2, axis=1))
            elif self.config.objective_type == "sphere":
                diff = values_batch - target[None, :]
                objectives = jnp.sum(diff ** 2, axis=1)
            else:
                raise ValueError(f"Unknown objective type: {self.config.objective_type}")
                
            # Calculate constraint violations
            lower_violations = jnp.maximum(0, lower[None, :] - values_batch)
            upper_violations = jnp.maximum(0, values_batch - upper[None, :])
            total_violations = jnp.sum(lower_violations, axis=1) + jnp.sum(upper_violations, axis=1)
            
            # Apply penalties
            penalties = total_violations * self.config.penalty_factor
            
            # Return negative (for maximization)
            return -(objectives + penalties)
            
        return _box_fitness
        
    @staticmethod
    def create_random_problem(key: jnp.ndarray, dimensions: int,
                            box_size: float = 10.0) -> 'BoxConfig':
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
            objective_type="distance"
        )
