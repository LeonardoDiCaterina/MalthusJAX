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
    """Configuration for Sphere function optimization (f(x) = sum(x^2))."""

    pass


@struct.dataclass
class SphereEvaluator(BaseEvaluator[RealGenome, SphereConfig, Any]):
    """Sphere function (sum of squares) fitness evaluator.

    Minimization is standard; config.maximize controls sign convention.
    """

    config: SphereConfig
    data: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate sphere function on real genome.

        Args:
            genome: RealGenome with values shape (d,).

        Returns:
            Scalar fitness. Negative if maximize=False (minimization convention).
        """
        sphere_value = jnp.sum(jnp.square(genome.values))
        return jax.lax.select(self.config.maximize, sphere_value, -sphere_value)


@struct.dataclass
class GriewankConfig(BaseEvaluatorConfig):
    """Configuration for Griewank function optimization (multimodal benchmark)."""

    pass


@struct.dataclass
class GriewankEvaluator(BaseEvaluator[RealGenome, GriewankConfig, Any]):
    """Griewank function fitness evaluator (multimodal, many local optima).

    Combines quadratic and cosine terms for high-dimensional complexity.
    """

    config: GriewankConfig
    data: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate Griewank function on real genome.

        Args:
            genome: RealGenome with values shape (d,).

        Returns:
            Scalar fitness. Negative if maximize=False (minimization convention).
        """
        x = genome.values
        quad_term = jnp.sum(jnp.square(x)) / 4000.0
        indices = jnp.arange(1, x.shape[0] + 1, dtype=jnp.float32)
        cos_term = jnp.prod(jnp.cos(x / jnp.sqrt(indices)))

        griewank_value = 1.0 + quad_term - cos_term
        return jax.lax.select(self.config.maximize, griewank_value, -griewank_value)


@struct.dataclass
class BoxConfig(BaseEvaluatorConfig):
    """Configuration for box-constrained optimization.

    Attributes:
        target_point: Target location in solution space, shape (d,).
        box_bounds: Tuple (lower, upper) constraint bounds, each shape (d,).
        penalty_factor: Linear constraint violation penalty coefficient.
        objective_type: 'distance' (L2) or 'sphere' (sum of squares).
    """

    target_point: chex.Array
    box_bounds: Tuple[chex.Array, chex.Array]
    penalty_factor: float = 1000.0
    objective_type: str = struct.field(pytree_node=False, default="distance")  # type: ignore[no-untyped-call]


@struct.dataclass
class BoxEvaluator(BaseEvaluator[RealGenome, BoxConfig, Any]):
    """Box-constrained optimization with linear penalty for infeasibility.

    Evaluates objective (distance or sphere) and adds linear penalty for
    constraint violations. Penalty uses jax.lax.select and jnp.maximum for
    XLA compatibility (no Python control flow during tracing).
    """

    config: BoxConfig
    data: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        """Evaluate box-constrained problem on real genome.

        Args:
            genome: RealGenome with values shape (d,).

        Returns:
            Scalar fitness: negative objective minus penalty. Returns -objective
            to convert minimization to the XLA-safe evaluation convention.

        Note:
            Objective type selection via string (if/elif) traces both branches
            in JAX; for high cardinality, use enum-based dispatch instead.
        """
        x = genome.values
        lower, upper = self.config.box_bounds

        if self.config.objective_type == "distance":
            objective = jnp.sqrt(jnp.sum(jnp.square(x - self.config.target_point)))
        elif self.config.objective_type == "sphere":
            centered = x - self.config.target_point
            objective = jnp.sum(jnp.square(centered))
        else:
            raise ValueError(f"Unknown objective type: {self.config.objective_type}")

        # Constraint violations: sum of excess magnitudes (XLA-safe)
        lower_violations = jnp.maximum(0, lower - x)
        upper_violations = jnp.maximum(0, x - upper)
        total_violation = jnp.sum(lower_violations) + jnp.sum(upper_violations)

        penalty = total_violation * self.config.penalty_factor
        return -objective - penalty

    @staticmethod
    def create_random_problem(
        key: chex.PRNGKey, dimensions: int, box_size: float = 10.0, maximize: bool = False
    ) -> BoxConfig:
        """Factory: create random box-constrained optimization instance (static config).

        Args:
            key: JAX PRNG key for random target and bounds.
            dimensions: Problem dimensionality.
            box_size: Size of bounding box around target.
            maximize: Optimization direction (default False = minimization).

        Returns:
            BoxConfig with random target and symmetric box constraints.
        """
        key1, key2 = jr.split(key, 2)
        target = jr.uniform(key1, (dimensions,), minval=-box_size / 2, maxval=box_size / 2)

        margin = box_size / 4
        lower = target - margin
        upper = target + margin

        return BoxConfig(
            target_point=target,
            box_bounds=(lower, upper),
            objective_type="distance",
            maximize=maximize,
        )
