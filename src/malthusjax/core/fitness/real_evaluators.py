from __future__ import annotations
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from ..genome.real_genome import RealGenome
from .base import BaseEvaluator, BaseEvaluatorConfig


@struct.dataclass
class SphereConfig(BaseEvaluatorConfig):
    """Configuration for Sphere function optimization."""
    pass


@struct.dataclass
class SphereEvaluator(BaseEvaluator[RealGenome, SphereConfig, Any]):
    """Sphere function fitness evaluator."""

    config: SphereConfig
    data: Any = struct.field(pytree_node=False, default=None) # type: ignore[no-untyped-call]

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate a single real genome on Sphere function."""
        sphere_value = jnp.sum(jnp.square(genome.values))
        return jax.lax.select(self.config.maximize, sphere_value, -sphere_value)


@struct.dataclass
class GriewankConfig(BaseEvaluatorConfig):
    """Configuration for Griewank function optimization."""
    pass


@struct.dataclass
class GriewankEvaluator(BaseEvaluator[RealGenome, GriewankConfig, Any]):
    """Griewank function fitness evaluator."""

    config: GriewankConfig
    data: Any = struct.field(pytree_node=False, default=None) # type: ignore[no-untyped-call]

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate a single real genome on Griewank function."""
        x = genome.values
        quad_term = jnp.sum(jnp.square(x)) / 4000.0
        indices = jnp.arange(1, x.shape[0] + 1, dtype=jnp.float32)
        cos_term = jnp.prod(jnp.cos(x / jnp.sqrt(indices)))

        griewank_value = 1.0 + quad_term - cos_term
        return jax.lax.select(self.config.maximize, griewank_value, -griewank_value)


@struct.dataclass
class BoxConfig(BaseEvaluatorConfig):
    """Configuration for Box-constrained optimization."""
    target_point: chex.Array
    box_bounds: Tuple[chex.Array, chex.Array]
    penalty_factor: float = 1000.0
    objective_type: str = struct.field(
        pytree_node=False, default="distance"
    ) # type: ignore[no-untyped-call]


@struct.dataclass
class BoxEvaluator(BaseEvaluator[RealGenome, BoxConfig, Any]):
    """Box-constrained optimization fitness evaluator."""

    config: BoxConfig
    data: Any = struct.field(pytree_node=False, default=None) # type: ignore[no-untyped-call]

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate a single real genome on box-constrained problem."""
        x = genome.values
        lower, upper = self.config.box_bounds

        if self.config.objective_type == "distance":
            objective = jnp.sqrt(jnp.sum(jnp.square(x - self.config.target_point)))
        elif self.config.objective_type == "sphere":
            centered = x - self.config.target_point
            objective = jnp.sum(jnp.square(centered))
        else:
            raise ValueError(f"Unknown objective type: {self.config.objective_type}")

        # Constraint violations
        lower_violations = jnp.maximum(0, lower - x)
        upper_violations = jnp.maximum(0, x - upper)
        total_violation = jnp.sum(lower_violations) + jnp.sum(upper_violations)

        penalty = total_violation * self.config.penalty_factor
        return -objective - penalty

    @staticmethod
    def create_random_problem(key: chex.PRNGKey, dimensions: int,
                            box_size: float = 10.0, maximize: bool = False) -> BoxConfig:
        """Create a random box-constrained optimization problem."""
        key1, key2 = jr.split(key, 2)
        target = jr.uniform(key1, (dimensions,), minval=-box_size/2, maxval=box_size/2)

        margin = box_size / 4
        lower = target - margin
        upper = target + margin

        return BoxConfig(
            target_point=target,
            box_bounds=(lower, upper),
            objective_type="distance",
            maximize=maximize
        )