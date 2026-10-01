"""Least-squares residual evaluator for accelerated curve fitting."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome

_field: Any = struct.field


@struct.dataclass
class LeastSquaresConfig(BaseEvaluatorConfig):
    """Configuration for non-linear curve fitting."""

    num_params: int = _field(pytree_node=False, default=1)
    is_pure_jax: bool = _field(pytree_node=False, default=True)


@struct.dataclass
class LeastSquaresEvaluator(
    BaseEvaluator[
        RealGenome,
        LeastSquaresConfig,
        Tuple[
            Callable[..., Any], chex.Array, chex.Array, chex.Array | None, chex.Array, chex.Array
        ],
    ]
):
    """Evaluates weighted sum of squared residuals for curve fitting: sum(((f(x, p) - y) / sigma)^2).

    Data format: (model_func, xdata, ydata, sigma, lower_bounds, upper_bounds)
    """

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        model_func, xdata, ydata, sigma, lower, upper = self.data

        # 1. Denormalize parameters from [0, 1] to physical bounds
        p_norm = jnp.clip(genome.values, 0.0, 1.0)
        p_phys = lower + p_norm * (upper - lower)

        # 2. Evaluate model function
        if self.config.is_pure_jax:
            y_pred = model_func(xdata, *p_phys)
            residuals = y_pred - ydata
            if sigma is not None:
                residuals = residuals / sigma
            cost = jnp.sum(jnp.square(residuals))
        else:

            def _py_cost(p_arr: np.ndarray) -> np.ndarray:
                p_list = list(p_arr)
                yp = model_func(np.asarray(xdata), *p_list)
                res = np.asarray(yp) - np.asarray(ydata)
                if sigma is not None:
                    res = res / np.asarray(sigma)
                return np.sum(np.square(res), dtype=np.float32)

            cost = jax.pure_callback(
                _py_cost,
                jax.ShapeDtypeStruct((), jnp.float32),
                p_phys,
                vmap_method="sequential",
            )

        cost = jnp.where(jnp.isnan(cost), 1e12, cost)
        return cost


def make_least_squares_evaluator(
    model_func: Callable[..., Any],
    xdata: Any,
    ydata: Any,
    sigma: Any = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> LeastSquaresEvaluator:
    """Constructs a LeastSquaresEvaluator inspecting JAX compatibility."""
    num_params = len(lower) if lower is not None else 1
    x_arr = jnp.asarray(xdata, dtype=jnp.float32)
    y_arr = jnp.asarray(ydata, dtype=jnp.float32)
    sig_arr = jnp.asarray(sigma, dtype=jnp.float32) if sigma is not None else None
    lower_arr = jnp.asarray(lower if lower is not None else np.zeros(num_params), dtype=jnp.float32)
    upper_arr = jnp.asarray(upper if upper is not None else np.ones(num_params), dtype=jnp.float32)

    test_p = lower_arr + 0.5 * (upper_arr - lower_arr)
    is_pure_jax = True
    try:
        jax.make_jaxpr(lambda p: model_func(x_arr, *p))(test_p)
    except Exception:
        is_pure_jax = False

    config = LeastSquaresConfig(num_params=num_params, is_pure_jax=is_pure_jax, maximize=False)
    data = (model_func, x_arr, y_arr, sig_arr, lower_arr, upper_arr)
    return LeastSquaresEvaluator(config=config, data=data)


__all__ = ["LeastSquaresEvaluator", "LeastSquaresConfig", "make_least_squares_evaluator"]
