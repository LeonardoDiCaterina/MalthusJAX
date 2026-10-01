"""Fitness evaluators for SciPy objectives and curve fitting."""

from malthus_optimize.evaluators.least_squares_evaluator import (
    LeastSquaresConfig,
    LeastSquaresEvaluator,
    make_least_squares_evaluator,
)
from malthus_optimize.evaluators.scipy_evaluator import (
    ScipyEvalConfig,
    ScipyObjectiveEvaluator,
    make_scipy_evaluator,
)

__all__ = [
    "ScipyObjectiveEvaluator",
    "ScipyEvalConfig",
    "make_scipy_evaluator",
    "LeastSquaresEvaluator",
    "LeastSquaresConfig",
    "make_least_squares_evaluator",
]
