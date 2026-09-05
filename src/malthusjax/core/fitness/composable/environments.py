"""Concrete Environment implementations for composable evaluators.

Provides:
- ``SklearnEnv``: supervised learning dataset from scikit-learn.
- ``BBOBEnv``: BBOB analytic black-box benchmark (OptimizationTask).
- ``CustomDatasetEnv``: user-provided (X, y) arrays.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.composable.base import (
    BaseSupervisedEnvironment,
    BaseOptimizationEnvironment,
)


# =============================================================================
# Supervised Environments
# =============================================================================

def _load_sklearn_data(dataset_name: str, **kwargs) -> tuple[chex.Array, chex.Array]:
    """Load and JAX-ify a scikit-learn dataset.

    Args:
        dataset_name: One of ``"california_housing"``, ``"diabetes"``,
            ``"breast_cancer"``, ``"make_regression"``, ``"make_classification"``.
        **kwargs: Passed through to the scikit-learn loader/generator.

    Returns:
        ``(X, y)`` as JAX float32 arrays.
    """
    from sklearn.datasets import (
        fetch_california_housing,
        load_diabetes,
        load_breast_cancer,
        make_regression,
        make_classification,
    )

    seed = int(kwargs.pop("seed", 42))

    if dataset_name == "california_housing":
        data = fetch_california_housing()
        X, y = data.data, data.target
    elif dataset_name == "diabetes":
        data = load_diabetes()
        X, y = data.data, data.target
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif dataset_name == "make_regression":
        X, y = make_regression(random_state=seed, **kwargs)
    elif dataset_name == "make_classification":
        X, y = make_classification(random_state=seed, **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: california_housing, "
            "diabetes, breast_cancer, make_regression, make_classification."
        )

    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)


@struct.dataclass
class SklearnEnv(BaseSupervisedEnvironment):
    """Supervised learning environment backed by a scikit-learn dataset.

    Loads the dataset at construction time and caches it as non-pytree
    ``data = (X, y)`` so it remains static across ``jax.vmap`` lifting.

    Compatible with any ``SupervisedEvaluator`` and any Interpreter that
    produces predictions over tabular data (MLPInterpreter, LinearGPInterpreter,
    CartesianGPInterpreter, TensorNEATInterpreter, etc.).

    Args:
        dataset: Dataset name string (e.g. ``"breast_cancer"``).
        **kwargs: Additional arguments forwarded to the sklearn loader.

    Usage::

        env = SklearnEnv.create(dataset="breast_cancer")
        # env.X  →  (569, 30)
        # env.y  →  (569,)
    """

    @classmethod
    def create(cls, dataset: str, **kwargs) -> "SklearnEnv":
        """Factory: load the dataset and return a ready-to-use ``SklearnEnv``.

        Args:
            dataset: Scikit-learn dataset name.
            **kwargs: Forwarded to the sklearn loader.

        Returns:
            Initialized ``SklearnEnv`` with ``data = (X, y)``.
        """
        X, y = _load_sklearn_data(dataset, **kwargs)
        return cls(data=(X, y))


@struct.dataclass
class CustomDatasetEnv(BaseSupervisedEnvironment):
    """Supervised learning environment backed by user-provided (X, y) arrays.

    Use this when your dataset is already loaded as JAX arrays and you don't
    want to depend on scikit-learn.

    Usage::

        env = CustomDatasetEnv(data=(my_X, my_y))
    """
    pass


# =============================================================================
# Optimization Environments
# =============================================================================

@struct.dataclass
class BBOBEnv(BaseOptimizationEnvironment):
    """BBOB benchmark wrapped as an OptimizationTask environment.

    Wraps the evosax ``BBOBProblem`` API behind the unified
    ``BaseOptimizationEnvironment`` interface. The genome is treated as a
    direct solution vector — no Interpreter transformation is applied (use
    ``IdentityInterpreter``).

    Args:
        fn_name: BBOB function identifier (e.g. ``"sphere"``, ``"rastrigin"``).
        num_dims: Problem dimensionality.
        seed: Instance seed (controls rotation matrices and shifts).

    Usage::

        env = BBOBEnv.create(fn_name="sphere", num_dims=10)
        evaluator = OptimizationEvaluator(
            env=env,
            interpreter=IdentityInterpreter(),
            output=ScalarOutput(),
        )
    """

    fn_name: str = struct.field(pytree_node=False, default="sphere")  # type: ignore[no-untyped-call]
    num_dims: int = struct.field(pytree_node=False, default=10)  # type: ignore[no-untyped-call]
    seed: int = struct.field(pytree_node=False, default=1)  # type: ignore[no-untyped-call]
    _problem: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]
    _state: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    @classmethod
    def create(
        cls,
        fn_name: str = "sphere",
        num_dims: int = 10,
        seed: int = 1,
    ) -> "BBOBEnv":
        """Factory: initialize the evosax BBOBProblem and return a ready environment.

        Args:
            fn_name: BBOB function name (e.g. ``"sphere"``, ``"rastrigin"``).
            num_dims: Problem dimensionality.
            seed: Instance seed.

        Returns:
            Initialized ``BBOBEnv``.
        """
        from evosax.problems import BBOBProblem

        problem = BBOBProblem(fn_name=fn_name, num_dims=num_dims, seed=seed)
        rng = jax.random.PRNGKey(seed)
        state = problem.init(rng)
        return cls(
            fn_name=fn_name,
            num_dims=num_dims,
            seed=seed,
            _problem=problem,
            _state=state,
        )

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        """Evaluate a candidate solution vector on the BBOB function.

        Args:
            solution: 1D solution vector of length ``num_dims``.

        Returns:
            Scalar objective.
        """
        rng = jax.random.PRNGKey(0)
        fitness_scores, _, _ = self._problem.eval(rng, solution[None, :], self._state)
        return fitness_scores[0]

    @property
    def f_opt(self) -> chex.Numeric:
        """Known global optimum value for this BBOB instance."""
        return self._problem.f_opt

    @property
    def x_opt(self) -> chex.Array:
        """Known global optimum location for this BBOB instance."""
        return self._problem.x_opt

@struct.dataclass
class SphereEnv(BaseOptimizationEnvironment):
    """Sphere function (sum of squares) environment."""

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        """Evaluate sphere function on real vector.
        Returns the sum-of-squares value (minimization convention).
        """
        return jnp.sum(jnp.square(solution))


@struct.dataclass
class GriewankEnv(BaseOptimizationEnvironment):
    """Griewank function environment (multimodal, many local optima)."""

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        """Evaluate Griewank function on real genome.
        Returns value directly (minimization convention).
        """
        quad_term = jnp.sum(jnp.square(solution)) / 4000.0
        indices = jnp.arange(1, solution.shape[0] + 1, dtype=jnp.float32)
        cos_term = jnp.prod(jnp.cos(solution / jnp.sqrt(indices)))

        return 1.0 + quad_term - cos_term


@struct.dataclass
class BoxEnv(BaseOptimizationEnvironment):
    """Box-constrained optimization environment with linear penalty for infeasibility."""

    target_point: chex.Array = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]
    box_bounds: tuple[chex.Array, chex.Array] = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]
    penalty_factor: float = struct.field(pytree_node=False, default=1000.0)  # type: ignore[no-untyped-call]
    objective_type: str = struct.field(pytree_node=False, default="distance")  # type: ignore[no-untyped-call]

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        """Evaluate box-constrained problem on real vector.
        Computes the objective and adds a linear penalty for violations.
        """
        x = solution
        lower, upper = self.box_bounds

        if self.objective_type == "distance":
            objective = jnp.sqrt(jnp.sum(jnp.square(x - self.target_point)))
        elif self.objective_type == "sphere":
            centered = x - self.target_point
            objective = jnp.sum(jnp.square(centered))
        else:
            # We must use safe jax constructs, so we'll just return a large value if misconfigured
            objective = jnp.array(1e9, dtype=jnp.float32)

        # Constraint violations: sum of excess magnitudes (XLA-safe)
        lower_violations = jnp.maximum(0, lower - x)
        upper_violations = jnp.maximum(0, x - upper)
        total_violation = jnp.sum(lower_violations) + jnp.sum(upper_violations)

        penalty = total_violation * self.penalty_factor
        return objective + penalty


@struct.dataclass
class TSPEnv(BaseOptimizationEnvironment):
    """TSP (Traveling Salesman Problem) environment.
    
    Uses Random Key encoding: the argsort of the real-valued solution array
    gives the permutation of cities.
    """

    distance_matrix: chex.Array = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        """Evaluate a solution's fitness on TSP.
        
        Args:
            solution: Continuous real array which is argsorted to form a tour.
        Returns:
            Scalar objective (minimization convention: total distance).
        """
        # Decode real array to permutation
        tour = jnp.argsort(solution)

        # Compute total distance: [city1, city2, ..., cityN, city1]
        tour_shifted = jnp.roll(tour, shift=-1)
        distances = self.distance_matrix[tour, tour_shifted]
        return jnp.sum(distances)
