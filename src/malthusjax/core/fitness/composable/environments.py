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
        maximize: If True, fitness is negated so higher is better.

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
    maximize: bool = struct.field(pytree_node=False, default=False)  # type: ignore[no-untyped-call]
    _problem: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]
    _state: Any = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]

    @classmethod
    def create(
        cls,
        fn_name: str = "sphere",
        num_dims: int = 10,
        seed: int = 1,
        maximize: bool = False,
    ) -> "BBOBEnv":
        """Factory: initialize the evosax BBOBProblem and return a ready environment.

        Args:
            fn_name: BBOB function name (e.g. ``"sphere"``, ``"rastrigin"``).
            num_dims: Problem dimensionality.
            seed: Instance seed.
            maximize: Optimization direction.

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
            maximize=maximize,
            _problem=problem,
            _state=state,
        )

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        """Evaluate a candidate solution vector on the BBOB function.

        Args:
            solution: 1D solution vector of length ``num_dims``.

        Returns:
            Scalar objective (minimization by default; negated if ``maximize=True``).
        """
        rng = jax.random.PRNGKey(0)
        fitness_scores, _, _ = self._problem.eval(rng, solution[None, :], self._state)
        result = fitness_scores[0]
        return -result if self.maximize else result

    @property
    def f_opt(self) -> chex.Numeric:
        """Known global optimum value for this BBOB instance."""
        return self._problem.f_opt

    @property
    def x_opt(self) -> chex.Array:
        """Known global optimum location for this BBOB instance."""
        return self._problem.x_opt
