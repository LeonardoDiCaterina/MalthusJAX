"""
Fitness module for MalthusJAX.

This module provides fitness evaluators using @struct.dataclass
for efficient batch evaluation using JAX JIT compilation.
"""

# Evaluators
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
from .tsp_evaluator import TSPConfig, TSPEvaluator

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
    # Combinatorial evaluators
    "TSPEvaluator",
    "TSPConfig",
]

# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------

from typing import Any as _Any
from typing import Callable as _Callable


def _make_bbob_factory(fn_name: str, *, maximize: bool = False) -> _Callable[..., "BBOBEvaluator"]:
    """Return a factory function for a specific BBOB function preset (minimization by default)."""

    def _factory(**kwargs: _Any) -> "BBOBEvaluator":
        from .bbob_evaluator import BBOBConfig, BBOBEvaluator

        _resolved_data = kwargs.pop("_resolved_data", None)

        config = BBOBConfig(
            fn_name=fn_name,
            num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)),
            maximize=kwargs.get("maximize", maximize),
            seed=kwargs.get("seed", 42),
        )
        return BBOBEvaluator.create(config)

    return _factory


def _create_bbob_evaluator(**kwargs: _Any) -> "BBOBEvaluator":
    """General BBOB factory accepting fn_name as a kwarg (minimization by default)."""
    from .bbob_evaluator import BBOBConfig, BBOBEvaluator

    _resolved_data = kwargs.pop("_resolved_data", None)

    config = BBOBConfig(
        fn_name=kwargs.get("fn_name", "sphere"),
        num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)),
        maximize=kwargs.get("maximize", False),
        seed=kwargs.get("seed", 42),
    )
    return BBOBEvaluator.create(config)


def _create_knapsack_evaluator(**kwargs: _Any) -> "KnapsackEvaluator":
    _resolved_data = kwargs.pop("_resolved_data", None)
    kwargs.setdefault("maximize", False)
    config = KnapsackConfig(**kwargs)
    return KnapsackEvaluator(config)


def _create_binary_sum_evaluator(**kwargs: _Any) -> "BinarySumEvaluator":
    _resolved_data = kwargs.pop("_resolved_data", None)
    kwargs.setdefault("maximize", False)
    config = BinarySumConfig(**kwargs)
    return BinarySumEvaluator(config)


def _create_griewank_evaluator(**kwargs: _Any) -> "GriewankEvaluator":
    _resolved_data = kwargs.pop("_resolved_data", None)
    kwargs.setdefault("maximize", False)
    config = GriewankConfig(**kwargs)
    return GriewankEvaluator(config)


def _create_tsp_evaluator(**kwargs: _Any) -> "TSPEvaluator":
    from .tsp_evaluator import TSPEvaluator

    _resolved_data = kwargs.pop("_resolved_data", None)

    if _resolved_data is not None:
        # If it's a dict holding data source specs (synthetic)
        if isinstance(_resolved_data, dict) and _resolved_data.get("source") == "synthetic":
            num_cities = _resolved_data.get("num_cities", kwargs.get("num_cities", 50))
            seed = _resolved_data.get("random_seed", kwargs.get("seed", 42))
            return TSPEvaluator.create_synthetic(num_cities=num_cities, seed=seed)

        # If it's an array (loaded from file)
        distance_matrix = _resolved_data
        if hasattr(distance_matrix, "distance_matrix"):
            distance_matrix = distance_matrix.distance_matrix

        return TSPEvaluator.create_from_data(kwargs, distance_matrix)

    num_cities = kwargs.get("num_cities", 50)
    seed = kwargs.get("seed", 42)
    return TSPEvaluator.create_synthetic(num_cities=num_cities, seed=seed)


def _register_fitness() -> None:
    """Register fitness evaluators with the global catalog registry."""
    from malthusjax.composer._registry import register_table

    register_table(
        [
            # BBOB presets - Standard functions
            ("sphere", _make_bbob_factory("sphere", maximize=False), {}),
            ("sphere_minimize", _make_bbob_factory("sphere", maximize=False), {}),
            ("sphere_maximize", _make_bbob_factory("sphere", maximize=True), {}),
            ("rastrigin", _make_bbob_factory("rastrigin", maximize=False), {}),
            ("griewank_rosenbrock", _make_bbob_factory("griewank_rosenbrock", maximize=False), {}),
            ("rosenbrock", _make_bbob_factory("rosenbrock", maximize=False), {}),
            ("ellipsoidal_rotated", _make_bbob_factory("ellipsoidal_rotated", maximize=False), {}),
            # General BBOB for custom functions
            ("bbob", _create_bbob_evaluator, {}),
            # Classic evaluators
            ("griewank", _create_griewank_evaluator, {}),
            ("binary_sum", _create_binary_sum_evaluator, {}),
            ("knapsack", _create_knapsack_evaluator, {}),
            ("tsp", _create_tsp_evaluator, {}),
        ],
        override=True,
    )


_register_fitness()

