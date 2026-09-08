"""
Fitness module for MalthusJAX.

This module provides fitness evaluators using @struct.dataclass
for efficient batch evaluation using JAX JIT compilation.
"""

# Evaluators
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.fitness.binary_evaluators import (
    BinarySumEvaluator,
    KnapsackEvaluator,
)
from malthusjax.core.fitness.composable.base import (
    BaseInterpreter,
    BaseOutputMode,
    BaseTransform,
    IdentityTransform,
    MOOutput,
    QDOutput,
    ScalarOutput,
)
from malthusjax.core.fitness.composable.environments import (
    BBOBEnv,
    BinarySumEnv,
    BoxEnv,
    BraxEnv,
    CustomDatasetEnv,
    GriewankEnv,
    GymnaxEnv,
    JumanjiEnv,
    KnapsackEnv,
    SklearnEnv,
    SphereEnv,
    TSPEnv,
)
from malthusjax.core.fitness.composable.evaluators import (
    BaseComposableEvaluator,
    OptimizationEvaluator,
    RLEvaluator,
    SupervisedEvaluator,
    TensorNeatEvaluator,
)
from malthusjax.core.fitness.composable.interpreters import (
    IdentityInterpreter,
    LinearGPInterpreter,
    MLPInterpreter,
)
from malthusjax.core.fitness.linear_gp_evaluator import (
    LinearGPEvaluator,
)

__all__ = [
    "BaseEvaluator",
    "BaseEvaluatorConfig",
    "KnapsackEvaluator",
    "BinarySumEvaluator",
    "LinearGPEvaluator",
    "BaseComposableEvaluator",
    "OptimizationEvaluator",
    "SupervisedEvaluator",
    "RLEvaluator",
    "TensorNeatEvaluator",
    "BaseTransform",
    "IdentityTransform",
    "BaseInterpreter",
    "IdentityInterpreter",
    "MLPInterpreter",
    "LinearGPInterpreter",
    "BaseOutputMode",
    "ScalarOutput",
    "QDOutput",
    "MOOutput",
    "SklearnEnv",
    "CustomDatasetEnv",
    "BBOBEnv",
    "SphereEnv",
    "GriewankEnv",
    "BoxEnv",
    "TSPEnv",
    "BinarySumEnv",
    "KnapsackEnv",
    "GymnaxEnv",
    "JumanjiEnv",
    "BraxEnv",
]

# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------

from typing import Any as _Any
from typing import Callable as _Callable


def _make_bbob_factory(fn_name: str, *, maximize: bool = False) -> _Callable[..., "OptimizationEvaluator"]:
    def _factory(**kwargs: _Any) -> "OptimizationEvaluator":
        _resolved_data = kwargs.pop("_resolved_data", None)
        return OptimizationEvaluator(
            env=BBOBEnv.create(fn_name=fn_name, num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)), seed=kwargs.get("seed", 42)),
            transform=IdentityTransform(),
            interpreter=IdentityInterpreter(),
            output=ScalarOutput(maximize=kwargs.get("maximize", maximize))
        )
    return _factory


def _create_bbob_evaluator(**kwargs: _Any) -> "OptimizationEvaluator":
    _resolved_data = kwargs.pop("_resolved_data", None)
    return OptimizationEvaluator(
        env=BBOBEnv.create(fn_name=kwargs.get("fn_name", "sphere"), num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)), seed=kwargs.get("seed", 42)),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=kwargs.get("maximize", False))
    )





def _create_knapsack_evaluator(**kwargs: _Any) -> "OptimizationEvaluator":
    _resolved_data = kwargs.pop("_resolved_data", None)
    kwargs.pop("data_id", None)
    maximize = kwargs.get("maximize", False)

    if _resolved_data is not None:
        if isinstance(_resolved_data, dict) and _resolved_data.get("source") == "synthetic":
            # For simplicity, we just use defaults if synthetic
            pass

    return OptimizationEvaluator(
        env=KnapsackEnv(weights=kwargs.get("weights"), values=kwargs.get("values"), capacity=kwargs.get("capacity", 100.0)),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=maximize)
    )

def _create_binary_sum_evaluator(**kwargs: _Any) -> "OptimizationEvaluator":
    _resolved_data = kwargs.pop("_resolved_data", None)
    maximize = kwargs.get("maximize", False)
    return OptimizationEvaluator(
        env=BinarySumEnv(),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=maximize)
    )

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
            ("binary_sum", _create_binary_sum_evaluator, {}),
            ("knapsack", _create_knapsack_evaluator, {}),
        ],
        override=True,
    )


_register_fitness()
