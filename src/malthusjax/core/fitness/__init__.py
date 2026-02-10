"""
Fitness module for MalthusJAX.

This module provides NEW paradigm fitness evaluators using @struct.dataclass
for efficient batch evaluation using JAX JIT compilation.
"""

# NEW architecture evaluators
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator

from .base import BaseEvaluator, RegressionData
from .binary_evaluators import (
    BinarySumConfig,
    BinarySumEvaluator,
    KnapsackConfig,
    KnapsackEvaluator,
)
from .linear_gp_evaluator import (
    TENSORGP_FUNCTIONS,
    TENSORGP_NAMES,
    LinearGPEvaluator,
    LinearGPEvaluatorConfig,
)
from .real_evaluators import (
    BoxConfig,
    BoxEvaluator,
    GriewankConfig,
    GriewankEvaluator,
    SphereConfig,
    SphereEvaluator,
)

__all__ = [
    # Base classes
    "BaseEvaluator",
    "RegressionData",
    # NEW evaluators
    "LinearGPEvaluator",
    "LinearGPEvaluatorConfig",
    "TENSORGP_FUNCTIONS",
    "TENSORGP_NAMES",
    # Binary evaluators
    "BinarySumEvaluator",
    "BinarySumConfig",
    "KnapsackEvaluator",
    "KnapsackConfig",
    # Real evaluators
    "SphereEvaluator",
    "SphereConfig",
    "GriewankEvaluator",
    "GriewankConfig",
    "BoxEvaluator",
    "BoxConfig",
]

# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------

from typing import Any as _Any
from typing import Callable as _Callable


def _make_bbob_factory(fn_name: str, *, maximize: bool = True) -> _Callable[..., "BBOBEvaluator"]:
    """Return a factory function for a specific BBOB function preset."""

    def _factory(**kwargs: _Any) -> "BBOBEvaluator":
        from .bbob_evaluator import BBOBConfig, BBOBEvaluator

        config = BBOBConfig(
            fn_name=fn_name,
            num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)),
            maximize=kwargs.get("maximize", maximize),
            seed=kwargs.get("seed", 42),
        )
        return BBOBEvaluator.create(config)

    return _factory


def _create_bbob_evaluator(**kwargs: _Any) -> "BBOBEvaluator":
    """General BBOB factory accepting fn_name as a kwarg."""
    from .bbob_evaluator import BBOBConfig, BBOBEvaluator

    config = BBOBConfig(
        fn_name=kwargs.get("fn_name", "sphere"),
        num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)),
        maximize=kwargs.get("maximize", True),
        seed=kwargs.get("seed", 42),
    )
    return BBOBEvaluator.create(config)


def _create_knapsack_evaluator(**kwargs: _Any) -> "KnapsackEvaluator":
    kwargs.setdefault("maximize", True)
    config = KnapsackConfig(**kwargs)
    return KnapsackEvaluator(config)


def _create_binary_sum_evaluator(**kwargs: _Any) -> "BinarySumEvaluator":
    kwargs.setdefault("maximize", True)
    config = BinarySumConfig(**kwargs)
    return BinarySumEvaluator(config)


def _create_griewank_evaluator(**kwargs: _Any) -> "GriewankEvaluator":
    kwargs.setdefault("maximize", True)
    config = GriewankConfig(**kwargs)
    return GriewankEvaluator(config)


def _register_fitness() -> None:
    """Register fitness evaluators with the global catalog registry."""
    from malthusjax.composer._registry import register_table

    register_table(
        [
            # BBOB presets
            ("sphere", _make_bbob_factory("sphere", maximize=True), {}),
            ("rastrigin", _make_bbob_factory("rastrigin", maximize=True), {}),
            ("sphere_minimize", _make_bbob_factory("sphere", maximize=False), {}),
            ("sphere_maximize", _make_bbob_factory("sphere", maximize=True), {}),
            # General BBOB
            ("bbob", _create_bbob_evaluator, {}),
            # Classic evaluators
            ("griewank", _create_griewank_evaluator, {}),
            ("binary_sum", _create_binary_sum_evaluator, {}),
            ("knapsack", _create_knapsack_evaluator, {}),
        ],
        override=True,
    )


_register_fitness()
