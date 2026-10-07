"""JAX-native SciPy objective evaluator with automatic vectorization and NumPy fallback."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.genome.real_genome import RealGenome
from malthusjax.core.logger import get_logger

logger = get_logger("malthus_optimize.evaluators.scipy_evaluator")

_field: Any = struct.field


@struct.dataclass
class ScipyEvalConfig(BaseEvaluatorConfig):
    """Configuration for SciPy objective evaluation."""

    num_dims: int = _field(pytree_node=False, default=1)
    is_pure_jax: bool = _field(pytree_node=False, default=True)


@struct.dataclass
class ScipyObjectiveEvaluator(
    BaseEvaluator[
        RealGenome,
        ScipyEvalConfig,
        Tuple[Callable[..., Any], Tuple[Any, ...], chex.Array, chex.Array],
    ]
):
    """Evaluates an arbitrary user objective function within MalthusJAX.

    Data format: (func, args, lower_bounds, upper_bounds)
    """

    def evaluate(self, genome: RealGenome) -> chex.Numeric:
        func, args, lower, upper = self.data

        # 1. Denormalize genome parameters from [0, 1] to [lower, upper]
        x_norm = jnp.clip(genome.values, 0.0, 1.0)
        x_phys = lower + x_norm * (upper - lower)

        # 2. Evaluate objective
        if self.config.is_pure_jax:
            raw_val = func(x_phys, *args)
            val = jnp.asarray(raw_val, dtype=jnp.float32).squeeze()
        else:
            # Safe pure_callback for arbitrary Python / NumPy functions
            def _py_eval(x_arr: np.ndarray) -> np.ndarray:
                res = func(np.asarray(x_arr), *args)
                return np.asarray(res, dtype=np.float32)

            val = jax.pure_callback(
                _py_eval,
                jax.ShapeDtypeStruct((), jnp.float32),
                x_phys,
                vmap_method="sequential",
            )

        val = jnp.where(jnp.isnan(val), 1e9, val)
        return jnp.where(self.config.maximize, -val, val)


def make_scipy_evaluator(
    func: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    maximize: bool = False,
) -> ScipyObjectiveEvaluator:
    """Constructs a ScipyObjectiveEvaluator inspecting whether func is JAX-traceable."""
    num_dims = len(lower) if lower is not None else 1
    lower_arr = jnp.asarray(lower if lower is not None else np.zeros(num_dims), dtype=jnp.float32)
    upper_arr = jnp.asarray(upper if upper is not None else np.ones(num_dims), dtype=jnp.float32)

    # Test if func can be traced natively by JAX
    test_x = lower_arr + 0.5 * (upper_arr - lower_arr)
    is_pure_jax = True
    try:
        jax.make_jaxpr(lambda x: func(x, *args))(test_x)
    except Exception as exc:
        logger.debug(
            "Objective function uses non-JAX operations (%s); using pure_callback fallback.", exc
        )
        is_pure_jax = False

    config = ScipyEvalConfig(
        num_dims=num_dims,
        is_pure_jax=is_pure_jax,
        maximize=maximize,
    )
    data = (func, args, lower_arr, upper_arr)
    return ScipyObjectiveEvaluator(config=config, data=data)


__all__ = ["ScipyObjectiveEvaluator", "ScipyEvalConfig", "make_scipy_evaluator"]
