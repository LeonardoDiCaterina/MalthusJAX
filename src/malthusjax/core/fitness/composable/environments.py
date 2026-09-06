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
    BaseRLEnvironment,
)

try:
    from tensorneat.problem import BaseProblem as TNBaseProblem
except ImportError:
    TNBaseProblem = Any



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
class BinarySumEnv(BaseOptimizationEnvironment):
    """Binary sum environment (OneMax problem)."""

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        """Evaluate a binary solution by counting ones.
        Optimization convention requires minimizing, so we return the number of zeros.
        """
        ones_count = jnp.sum(solution)
        zeros_count = solution.size - ones_count
        return zeros_count

@struct.dataclass
class KnapsackEnv(BaseOptimizationEnvironment):
    """0/1 Knapsack environment with linear constraint penalty."""

    weights: chex.Array = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]
    values: chex.Array = struct.field(pytree_node=False, default=None)  # type: ignore[no-untyped-call]
    capacity: float = struct.field(pytree_node=False, default=100.0)  # type: ignore[no-untyped-call]
    penalty_factor: float = struct.field(pytree_node=False, default=1000.0)  # type: ignore[no-untyped-call]

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        """Evaluate a binary solution for the Knapsack problem."""
        total_weight = jnp.sum(solution * self.weights)
        total_value = jnp.sum(solution * self.values)

        excess_weight = jnp.maximum(0.0, total_weight - self.capacity)
        penalty = excess_weight * self.penalty_factor

        value = total_value - penalty
        # Minimization convention
        return -value

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


# =============================================================================
# Reinforcement Learning Environments
# =============================================================================

@struct.dataclass
class GymnaxEnv(BaseRLEnvironment):
    """Gymnax reinforcement learning environment adapter."""

    env_name: str = struct.field(pytree_node=False)
    env: Any = struct.field(pytree_node=False)
    env_params: Any = struct.field(pytree_node=False) # gymnax EnvParams is mostly static/dataclass

    @classmethod
    def create(cls, env_name: str, **kwargs) -> GymnaxEnv:
        """Create a Gymnax environment instance."""
        import gymnax
        env, env_params = gymnax.make(env_name, **kwargs)
        return cls(env_name=env_name, env=env, env_params=env_params)

    @property
    def max_steps(self) -> int:
        """Maximum steps per episode."""
        return self.env_params.max_steps_in_episode

    def reset(self, key: chex.PRNGKey) -> tuple[chex.Array, Any]:
        """Reset the environment and return initial (obs, state)."""
        obs, state = self.env.reset(key, self.env_params)
        return obs, state

    def step(self, state: Any, action: chex.Array, key: chex.PRNGKey) -> tuple[chex.Array, Any, chex.Numeric, chex.Array, Any]:
        """Take one step: (obs, state, reward, done, info)."""
        next_obs, next_state, reward, done, info = self.env.step(key, state, action, self.env_params)
        return next_obs, next_state, reward, done, info

    def preprocess_obs(self, obs: chex.Array) -> chex.Array:
        """Flatten/normalize raw environment observation to a 1D array."""
        return obs.flatten()

    def postprocess_action(self, logits: chex.Array, raw_obs: Any = None) -> chex.Array:
        """Convert interpreter output logits to environment-compatible action."""
        # Check if environment is discrete or continuous
        if hasattr(self.env.action_space(self.env_params), "n"):
            # Discrete space -> Argmax
            return jnp.argmax(logits)
        else:
            # Continuous space -> Direct output (or tanh applied by interpreter)
            return logits

    @property
    def obs_dim(self) -> int:
        """Flattened observation dimensionality."""
        import numpy as np
        shape = self.env.observation_space(self.env_params).shape
        return int(np.prod(shape))

    @property
    def action_dim(self) -> int:
        """Action dimensionality (num logits required)."""
        space = self.env.action_space(self.env_params)
        if hasattr(space, "n"):
            return space.n
        else:
            import numpy as np
            return int(np.prod(space.shape))

@struct.dataclass
class JumanjiEnv(BaseRLEnvironment):
    """Jumanji reinforcement learning environment adapter."""

    env_name: str = struct.field(pytree_node=False)
    env: Any = struct.field(pytree_node=False)

    @classmethod
    def create(cls, env_name: str, **kwargs) -> JumanjiEnv:
        import jumanji
        env = jumanji.make(env_name, **kwargs)
        return cls(env_name=env_name, env=env)

    @property
    def max_steps(self) -> int:
        return self.env.time_limit

    def reset(self, key: chex.PRNGKey) -> tuple[Any, Any]:
        state, timestep = self.env.reset(key)
        return timestep, state

    def step(self, state: Any, action: chex.Array, key: chex.PRNGKey) -> tuple[Any, Any, chex.Numeric, chex.Array, Any]:
        next_state, next_timestep = self.env.step(state, action)
        return next_timestep, next_state, next_timestep.reward, next_timestep.last(), {}

    def preprocess_obs(self, obs: Any) -> chex.Array:
        # 'obs' is actually a Jumanji TimeStep object, so its .observation is the true observation struct
        true_obs = getattr(obs, "observation", obs)
        
        # Heuristic to extract the main feature array from the observation struct
        if hasattr(true_obs, "grid"):
            return jnp.reshape(true_obs.grid, -1)
        elif hasattr(true_obs, "feature"):
            return jnp.reshape(true_obs.feature, -1)
        elif hasattr(true_obs, "observation"):
            return jnp.reshape(true_obs.observation, -1)
        return jnp.reshape(true_obs, -1)

    def postprocess_action(self, logits: chex.Array, raw_obs: Any = None) -> chex.Array:
        true_obs = getattr(raw_obs, "observation", raw_obs)
        mask = getattr(true_obs, "action_mask", None)
        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)
        return jnp.argmax(logits)

    @property
    def obs_dim(self) -> int:
        import numpy as np
        spec = self.env.observation_spec() if callable(self.env.observation_spec) else self.env.observation_spec
        if hasattr(spec, "observation"):
            return int(np.prod(spec.observation.shape))
        elif hasattr(spec, "grid"):
            return int(np.prod(spec.grid.shape))
        elif hasattr(spec, "feature"):
            return int(np.prod(spec.feature.shape))
        return int(np.prod(spec.shape))

    @property
    def action_dim(self) -> int:
        spec = self.env.action_spec() if callable(self.env.action_spec) else self.env.action_spec
        # Jumanji Discrete spaces usually expose num_values
        if hasattr(spec, "num_values"):
            # MultiDiscrete is handled by prod(num_values) 
            import numpy as np
            return int(np.prod(spec.num_values))
        return spec.num_values # scalar discrete


@struct.dataclass
class BraxEnv(BaseRLEnvironment):
    """Brax reinforcement learning environment adapter."""

    env_name: str = struct.field(pytree_node=False)
    env: Any = struct.field(pytree_node=False)

    @classmethod
    def create(cls, env_name: str, **kwargs) -> BraxEnv:
        from brax import envs
        env = envs.create(env_name=env_name, **kwargs)
        return cls(env_name=env_name, env=env)

    def reset(self, key: chex.PRNGKey) -> tuple[chex.Array, Any]:
        state = self.env.reset(key)
        return state.obs, state

    def step(self, state: Any, action: chex.Array, key: chex.PRNGKey) -> tuple[chex.Array, Any, chex.Numeric, chex.Array, Any]:
        next_state = self.env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done, {}

    def preprocess_obs(self, obs: chex.Array) -> chex.Array:
        return obs.flatten()

    def postprocess_action(self, logits: chex.Array, raw_obs: Any = None) -> chex.Array:
        # Brax typically uses continuous actions in [-1, 1]
        return jnp.tanh(logits)

    @property
    def obs_dim(self) -> int:
        return self.env.observation_size

    @property
    def action_dim(self) -> int:
        return self.env.action_size


# =============================================================================
# TensorNEAT Proxy Environment
# =============================================================================

@struct.dataclass
class TensorNEATProblemWrapper(BaseOptimizationEnvironment):
    """A wrapper for TensorNEAT internal problems to be used directly by TensorNeatEvaluator.
    
    TensorNEAT natively handles RL and optimization rollouts internally via `.evaluate()`.
    This wrapper strictly adheres to their paper-grade benchmarking by passing the raw problem through
    so MalthusJAX can leverage it directly without breaking open the episode loop.
    """
    
    problem: TNBaseProblem = struct.field(pytree_node=False)
    
    @property
    def num_dims(self) -> int:
        return 0

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        # Evaluation is handled directly by TensorNeatEvaluator bypassing this fallback.
        raise NotImplementedError("TensorNEAT environments must be evaluated via TensorNeatEvaluator")


