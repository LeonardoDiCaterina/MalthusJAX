"""Ordinary Differential Equation (ODE) Residual Evaluator for dCGP.

Evaluates candidate symbolic programs by computing their exact analytical
derivatives via JAX AutoDiff and penalizing the differential equation residual
and initial/boundary condition errors (Izzo et al., 2016).
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthus_gp.genomes.differentiable_cartesian_genome import DifferentiableCartesianGenome
from malthus_gp.interpreters.differentiable_cartesian_interpreter import (
    DifferentiableCartesianInterpreter,
)
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_gp.evaluators.ode_evaluator")

_field: Any = struct.field


@struct.dataclass
class ODEResidualEvaluatorConfig(BaseEvaluatorConfig):  # type: ignore[misc]
    """Configuration for ODE residual evaluation.

    Attributes:
        bc_weight: Penalty weight for boundary/initial condition violations.
        maximize: Optimization direction (default False for minimization).
    """

    bc_weight: float = _field(pytree_node=False, default=10.0)
    maximize: bool = _field(pytree_node=False, default=False)


@struct.dataclass
class ODEResidualEvaluator(
    BaseEvaluator[DifferentiableCartesianGenome, ODEResidualEvaluatorConfig, chex.Array]  # type: ignore[misc]
):
    """Evaluates candidate programs against an ODE: dy/dt = f(t, y) with y(t0) = y0.

    Attributes:
        data: Grid of evaluation points T in the time domain, shape (n_points,).
        interpreter: DifferentiableCartesianInterpreter for forward & AutoDiff passes.
        ode_fn: Target differential equation f(t, y) where dy/dt = f(t, y).
        initial_condition: Tuple of (t0, y0) representing boundary value y(t0) = y0.
    """

    interpreter: DifferentiableCartesianInterpreter = _field(
        default_factory=DifferentiableCartesianInterpreter
    )
    ode_fn: Callable[[chex.Array, chex.Array], chex.Array] = _field(
        pytree_node=False, default=lambda t, y: -y
    )
    initial_condition: Tuple[float, float] = _field(pytree_node=False, default=(0.0, 1.0))

    def evaluate(self, genome: DifferentiableCartesianGenome) -> chex.Numeric:
        """Computes mean squared ODE residual and boundary condition penalty."""
        t_domain = self.data
        t0, y0 = self.initial_condition

        # Scalar forward function y(t)
        def predict_scalar(t_val: chex.Array) -> chex.Array:
            return self.interpreter.evaluate_single(genome, t_val)[0]

        # Exact analytical derivative dy/dt via JAX AutoDiff
        grad_predict = jax.grad(predict_scalar)

        def compute_residual_at_t(t_val: chex.Array) -> chex.Array:
            y_pred = predict_scalar(t_val)
            dy_dt = grad_predict(t_val)
            # ODE residual: (dy/dt - f(t, y))^2
            f_target = self.ode_fn(t_val, y_pred)
            res = jnp.square(dy_dt - f_target)
            return jnp.nan_to_num(res, nan=1e6, posinf=1e6, neginf=1e6)

        # Vectorize residual calculation over time domain T
        residuals = jax.vmap(compute_residual_at_t)(t_domain)
        mean_residual = jnp.mean(residuals)

        # Boundary condition penalty: (y(t0) - y0)^2
        y_at_t0 = predict_scalar(jnp.asarray(t0, dtype=jnp.float32))
        bc_error = jnp.square(y_at_t0 - y0)

        total_loss = mean_residual + (self.config.bc_weight * bc_error)

        return jnp.where(self.config.maximize, -total_loss, total_loss)
