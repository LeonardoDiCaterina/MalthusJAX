"""
Fitness module for MalthusJAX.

This module provides fitness evaluators using @struct.dataclass
for efficient batch evaluation using JAX JIT compilation.
"""

from typing import Any

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


def _make_bbob_factory(
    fn_name: str, *, maximize: bool = False
) -> _Callable[..., "OptimizationEvaluator"]:
    def _factory(**kwargs: _Any) -> "OptimizationEvaluator":
        _resolved_data = kwargs.pop("_resolved_data", None)
        return OptimizationEvaluator(
            env=BBOBEnv.create(
                fn_name=fn_name,
                num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)),
                seed=kwargs.get("seed", 42),
            ),
            transform=IdentityTransform(),
            interpreter=IdentityInterpreter(),
            output=ScalarOutput(maximize=kwargs.get("maximize", maximize)),
        )

    return _factory


def _create_bbob_evaluator(**kwargs: _Any) -> "OptimizationEvaluator":
    _resolved_data = kwargs.pop("_resolved_data", None)
    return OptimizationEvaluator(
        env=BBOBEnv.create(
            fn_name=kwargs.get("fn_name", "sphere"),
            num_dims=kwargs.get("dim", kwargs.get("num_dims", 10)),
            seed=kwargs.get("seed", 42),
        ),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=kwargs.get("maximize", False)),
    )


def _create_knapsack_evaluator(**kwargs: _Any) -> "OptimizationEvaluator":
    import jax
    import jax.numpy as jnp

    _resolved_data = kwargs.pop("_resolved_data", None)
    kwargs.pop("data_id", None)
    maximize = kwargs.get("maximize", False)

    weights = kwargs.get("weights")
    values = kwargs.get("values")
    capacity = kwargs.get("capacity", 100.0)

    if (weights is None or values is None) and _resolved_data is not None:
        if isinstance(_resolved_data, dict):
            if "weights" in _resolved_data and "values" in _resolved_data:
                weights = jnp.array(_resolved_data["weights"], dtype=jnp.float32)
                values = jnp.array(_resolved_data["values"], dtype=jnp.float32)
                if "capacity" in _resolved_data:
                    capacity = float(_resolved_data["capacity"])
            elif _resolved_data.get("source") == "synthetic":
                num_items = _resolved_data.get("num_items", 50)
                capacity = float(_resolved_data.get("capacity", capacity))
                w_min, w_max = _resolved_data.get("weight_range", [10, 50])
                v_min, v_max = _resolved_data.get("value_range", [30, 150])
                seed = _resolved_data.get("random_seed", 42)
                k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
                weights = jax.random.uniform(k1, (num_items,), minval=w_min, maxval=w_max)
                values = jax.random.uniform(k2, (num_items,), minval=v_min, maxval=v_max)

    if weights is None or values is None:
        num_items = kwargs.get("num_items", 50)
        seed = kwargs.get("seed", 42)
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
        weights = jax.random.uniform(k1, (num_items,), minval=10.0, maxval=50.0)
        values = jax.random.uniform(k2, (num_items,), minval=30.0, maxval=150.0)

    return OptimizationEvaluator(
        env=KnapsackEnv(
            weights=weights,
            values=values,
            capacity=capacity,
        ),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=maximize),
    )


def _create_binary_sum_evaluator(**kwargs: _Any) -> "OptimizationEvaluator":
    _resolved_data = kwargs.pop("_resolved_data", None)
    maximize = kwargs.get("maximize", False)
    return OptimizationEvaluator(
        env=BinarySumEnv(),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=maximize),
    )


def _create_tsp_evaluator(**kwargs: _Any) -> "OptimizationEvaluator":
    import jax
    import jax.numpy as jnp

    _resolved_data = kwargs.pop("_resolved_data", None)
    kwargs.pop("data_id", None)
    maximize = kwargs.get("maximize", False)

    distance_matrix = kwargs.get("distance_matrix")
    if distance_matrix is None and _resolved_data is not None:
        if isinstance(_resolved_data, dict):
            if "distance_matrix" in _resolved_data:
                distance_matrix = jnp.array(_resolved_data["distance_matrix"])
            elif _resolved_data.get("source") == "synthetic":
                num_cities = _resolved_data.get("num_cities", 52)
                seed = _resolved_data.get("random_seed", 42)
                key = jax.random.PRNGKey(seed)
                coords = jax.random.uniform(key, (num_cities, 2))
                diff = coords[:, jnp.newaxis, :] - coords[jnp.newaxis, :, :]
                distance_matrix = jnp.sqrt(jnp.sum(diff**2, axis=-1))

    if distance_matrix is None:
        num_cities = kwargs.get("num_cities", 52)
        seed = kwargs.get("seed", 42)
        key = jax.random.PRNGKey(seed)
        coords = jax.random.uniform(key, (num_cities, 2))
        diff = coords[:, jnp.newaxis, :] - coords[jnp.newaxis, :, :]
        distance_matrix = jnp.sqrt(jnp.sum(diff**2, axis=-1))

    return OptimizationEvaluator(
        env=TSPEnv(distance_matrix=distance_matrix),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=maximize),
    )


def register_fitness_catalog(register_table_fn: Any) -> None:
    """Register fitness evaluators with the global catalog registry."""
    register_table_fn(
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
            ("tsp", _create_tsp_evaluator, {}),
        ],
        override=True,
    )
