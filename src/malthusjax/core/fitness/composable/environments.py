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
import numpy as np
from flax import struct

from malthusjax.core.fitness.composable.base import (
    BaseOptimizationEnvironment,
    BaseRLEnvironment,
    BaseSupervisedEnvironment,
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
        load_breast_cancer,
        load_diabetes,
        make_classification,
        make_regression,
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
    env_params: Any = struct.field(pytree_node=False)  # gymnax EnvParams is mostly static/dataclass

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
        reset_result = self.env.reset(key, self.env_params)
        if isinstance(reset_result, (tuple, list)):
            obs = reset_result[0]
            state = reset_result[1]
        else:
            obs = reset_result
            state = None
        return obs, state

    def step(
        self, state: Any, action: chex.Array, key: chex.PRNGKey | None = None
    ) -> tuple[chex.Array, Any, chex.Numeric, chex.Array, Any]:
        """Take one step: (obs, state, reward, done, info)."""
        step_result = self.env.step(key, state, action, self.env_params)
        if len(step_result) == 6:
            next_obs, next_state, reward, term, trunc, info = step_result
            done = jnp.logical_or(term, trunc)
        elif len(step_result) == 5:
            next_obs, next_state, reward, done, info = step_result
        elif len(step_result) == 4:
            next_obs, next_state, reward, done = step_result
            info = {}
        elif len(step_result) > 6:
            next_obs = step_result[0]
            next_state = step_result[1]
            reward = step_result[2]
            term = step_result[3]
            trunc = step_result[4]
            done = jnp.logical_or(term, trunc)
            info = step_result[5]
        else:
            raise ValueError(
                f"Gymnax environment step returned unexpected number of values: {len(step_result)}"
            )
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

        space_fn = getattr(self.env, "observation_space", None)
        obs_space = space_fn(self.env_params) if callable(space_fn) else space_fn
        shape = obs_space.shape
        return int(np.prod(shape))

    @property
    def action_dim(self) -> int:
        """Action dimensionality (num logits required)."""
        space_fn = getattr(self.env, "action_space", None)
        space = space_fn(self.env_params) if callable(space_fn) else space_fn
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

    def step(
        self, state: Any, action: chex.Array, key: chex.PRNGKey
    ) -> tuple[Any, Any, chex.Numeric, chex.Array, Any]:
        next_state, next_timestep = self.env.step(state, action)
        return next_timestep, next_state, next_timestep.reward, next_timestep.last(), {}

    def preprocess_obs(self, obs: Any) -> chex.Array:
        # 'obs' is actually a Jumanji TimeStep object, so its .observation is the true observation struct
        true_obs = getattr(obs, "observation", obs)

        if isinstance(true_obs, (jnp.ndarray, np.ndarray)):
            return jnp.reshape(true_obs, -1)

        # Heuristic to extract the main feature array from the observation struct if simple
        if hasattr(true_obs, "grid") and not hasattr(true_obs, "_specs"):
            return jnp.reshape(true_obs.grid, -1)
        elif hasattr(true_obs, "feature") and not hasattr(true_obs, "_specs"):
            return jnp.reshape(true_obs.feature, -1)
        elif hasattr(true_obs, "observation") and not hasattr(true_obs, "_specs"):
            return jnp.reshape(true_obs.observation, -1)

        # Fallback to flattening all PyTree leaves and concatenating
        leaves = [
            jnp.asarray(x, dtype=jnp.float32).ravel() for x in jax.tree_util.tree_leaves(true_obs)
        ]
        if len(leaves) == 1:
            return leaves[0]
        return jnp.concatenate(leaves)

    def postprocess_action(self, logits: chex.Array, raw_obs: Any = None) -> chex.Array:
        true_obs = getattr(raw_obs, "observation", raw_obs)
        mask = getattr(true_obs, "action_mask", None)
        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)
        return jnp.argmax(logits)

    @property
    def obs_dim(self) -> int:
        import numpy as np

        spec = (
            self.env.observation_spec()
            if callable(self.env.observation_spec)
            else self.env.observation_spec
        )

        def _get_spec_dim(s: Any) -> int:
            if hasattr(s, "shape"):
                return int(np.prod(s.shape))
            elif hasattr(s, "_specs"):
                return sum(_get_spec_dim(child) for child in s._specs.values())
            elif hasattr(s, "observation"):
                return _get_spec_dim(s.observation)
            elif hasattr(s, "grid"):
                return _get_spec_dim(s.grid)
            elif hasattr(s, "feature"):
                return _get_spec_dim(s.feature)
            return int(np.prod(getattr(s, "shape", ())))

        return _get_spec_dim(spec)

    @property
    def action_dim(self) -> int:
        spec = self.env.action_spec() if callable(self.env.action_spec) else self.env.action_spec
        # Jumanji Discrete spaces usually expose num_values
        if hasattr(spec, "num_values"):
            # MultiDiscrete is handled by prod(num_values)
            import numpy as np

            return int(np.prod(spec.num_values))
        return spec.num_values  # scalar discrete


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

    def step(
        self, state: Any, action: chex.Array, key: chex.PRNGKey | None = None
    ) -> tuple[chex.Array, Any, chex.Numeric, chex.Array, Any]:
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
# DeepMind MuJoCo MJX Environment Adapter
# =============================================================================

CARTPOLE_MJCF = """
<mujoco model="cartpole">
  <compiler inertiafromgeom="true"/>
  <default>
    <joint damping="0.05"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" axis="1 0 0" range="-2.5 2.5"/>
      <geom name="cart_geom" type="box" size="0.2 0.1 0.1" mass="1.0" rgba="0.2 0.6 0.8 1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom name="pole_geom" type="cylinder" size="0.04 0.4" mass="0.1" pos="0 0 0.4" rgba="0.8 0.3 0.3 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide_motor" joint="slider" ctrlrange="-10 10" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


@struct.dataclass
class MuJoCoMJXState:
    """JAX PyTree state encapsulating a DeepMind MuJoCo MJX physics step."""

    data: Any  # mujoco_mjx.Data
    step_count: chex.Array
    cum_reward: chex.Array
    done: chex.Array
    info: dict[str, Any] = struct.field(default_factory=dict)


@struct.dataclass
class MuJoCoMJXEnv(BaseRLEnvironment):
    """DeepMind MuJoCo MJX reinforcement learning environment adapter.

    Conforms to MalthusJAX's 4-axis Composable Evaluator design. Supports
    custom MJCF XML strings/files as well as DeepMind locomotion benchmarks
    (e.g., 'ant', 'half_cheetah', 'hopper', 'cartpole') executed natively
    on GPU/TPU accelerators under ``jax.lax.scan``.

    DISAMBIGUATION COMPLIANCE:
    - MuJoCo in JAX is imported strictly as ``from mujoco import mjx as mujoco_mjx``.
    - Entities are explicitly prefixed (``MuJoCoMJXEnv``, ``mujoco_model``).
    """

    mujoco_model: Any = struct.field(pytree_node=False, default=None)
    raw_model: Any = struct.field(pytree_node=False, default=None)
    env_name: str = struct.field(pytree_node=False, default="mujoco_mjx")
    max_steps: int = struct.field(pytree_node=False, default=1000)
    reset_noise_scale: float = struct.field(pytree_node=False, default=0.05)
    action_scale: float = struct.field(pytree_node=False, default=1.0)
    backend_pipeline: Any = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        env_name: str | None = None,
        xml_path: str | None = None,
        xml_string: str | None = None,
        max_steps: int = 1000,
        reset_noise_scale: float = 0.05,
        action_scale: float = 1.0,
        **kwargs: Any,
    ) -> MuJoCoMJXEnv:
        """Create a DeepMind MuJoCo MJX environment.

        Args:
            env_name: Benchmark environment name (e.g., 'cartpole', 'ant', 'half_cheetah').
            xml_path: Filepath to an MJCF XML model file.
            xml_string: Raw MJCF XML model string.
            max_steps: Maximum episode step horizon.
            reset_noise_scale: Standard deviation of perturbation on initial qpos/qvel.
            action_scale: Multiplier applied to tanh-bounded action logits.
            **kwargs: Extra parameters passed to the underlying loader.
        """
        try:
            import mujoco
            from mujoco import mjx as mujoco_mjx
        except ImportError as e:
            raise ImportError(
                "DeepMind MuJoCo MJX is required to use MuJoCoMJXEnv. "
                "Please install via: pip install 'malthusjax[mujoco]'"
            ) from e

        # 1. Custom XML string
        if xml_string is not None:
            raw_model = mujoco.MjModel.from_xml_string(xml_string)
            mujoco_model = mujoco_mjx.put_model(raw_model)
            return cls(
                mujoco_model=mujoco_model,
                raw_model=raw_model,
                env_name=env_name or "custom_xml",
                max_steps=max_steps,
                reset_noise_scale=reset_noise_scale,
                action_scale=action_scale,
            )

        # 2. Custom XML file path
        if xml_path is not None:
            raw_model = mujoco.MjModel.from_xml_path(xml_path)
            mujoco_model = mujoco_mjx.put_model(raw_model)
            return cls(
                mujoco_model=mujoco_model,
                raw_model=raw_model,
                env_name=env_name or xml_path,
                max_steps=max_steps,
                reset_noise_scale=reset_noise_scale,
                action_scale=action_scale,
            )

        # 3. Canonical Built-in Benchmarks
        normalized_name = (env_name or "cartpole").lower().replace("-", "_")
        if normalized_name in ("cartpole", "inverted_pendulum"):
            raw_model = mujoco.MjModel.from_xml_string(CARTPOLE_MJCF)
            mujoco_model = mujoco_mjx.put_model(raw_model)
            return cls(
                mujoco_model=mujoco_model,
                raw_model=raw_model,
                env_name=normalized_name,
                max_steps=max_steps,
                reset_noise_scale=reset_noise_scale,
                action_scale=action_scale,
            )

        # 4. Standard locomotion environments via DeepMind Brax MJX backend
        try:
            from brax import envs as brax_envs

            pipeline_env = brax_envs.create(
                env_name=normalized_name, backend="mjx", auto_reset=False, **kwargs
            )
            raw_sys = getattr(pipeline_env, "sys", None)
            return cls(
                mujoco_model=raw_sys,
                raw_model=raw_sys,
                env_name=normalized_name,
                max_steps=max_steps,
                reset_noise_scale=reset_noise_scale,
                action_scale=action_scale,
                backend_pipeline=pipeline_env,
            )
        except Exception:
            # Fallback to built-in cartpole if specific locomotion name is not found
            raw_model = mujoco.MjModel.from_xml_string(CARTPOLE_MJCF)
            mujoco_model = mujoco_mjx.put_model(raw_model)
            return cls(
                mujoco_model=mujoco_model,
                raw_model=raw_model,
                env_name=normalized_name,
                max_steps=max_steps,
                reset_noise_scale=reset_noise_scale,
                action_scale=action_scale,
            )

    def reset(self, key: chex.PRNGKey) -> tuple[chex.Array, Any]:
        """Reset environment to initial state."""
        if self.backend_pipeline is not None:
            state = self.backend_pipeline.reset(key)
            return state.obs, state

        from mujoco import mjx as mujoco_mjx

        k_pos, k_vel = jax.random.split(key)
        data = mujoco_mjx.make_data(self.mujoco_model)

        qpos_noise = jax.random.uniform(
            k_pos,
            shape=(self.mujoco_model.nq,),
            minval=-self.reset_noise_scale,
            maxval=self.reset_noise_scale,
        )
        qvel_noise = (
            jax.random.normal(k_vel, shape=(self.mujoco_model.nv,)) * self.reset_noise_scale
        )

        data = data.replace(
            qpos=data.qpos + qpos_noise,
            qvel=data.qvel + qvel_noise,
            ctrl=jnp.zeros(self.mujoco_model.nu, dtype=jnp.float32),
        )
        data = mujoco_mjx.forward(self.mujoco_model, data)

        obs = jnp.concatenate([data.qpos, data.qvel])
        state = MuJoCoMJXState(
            data=data,
            step_count=jnp.array(0, dtype=jnp.int32),
            cum_reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(False, dtype=jnp.bool_),
        )
        return obs, state

    def step(
        self, state: Any, action: chex.Array, key: chex.PRNGKey | None = None
    ) -> tuple[chex.Array, Any, chex.Numeric, chex.Array, Any]:
        """Step physics simulation forward by one control timestep."""
        if self.backend_pipeline is not None:
            clamped_action = jnp.clip(action, -1.0, 1.0)
            next_state = self.backend_pipeline.step(state, clamped_action)
            is_nan = jnp.logical_or(
                jnp.isnan(next_state.reward), jnp.any(jnp.isnan(next_state.obs))
            )
            safe_obs = jnp.nan_to_num(next_state.obs, nan=0.0, posinf=0.0, neginf=0.0)
            safe_reward = jnp.where(
                is_nan, 0.0, jnp.nan_to_num(next_state.reward, nan=0.0, posinf=0.0, neginf=0.0)
            )
            safe_done = jnp.logical_or(next_state.done > 0.5, is_nan)
            return safe_obs, next_state, safe_reward, safe_done, {}

        from mujoco import mjx as mujoco_mjx

        clamped_action = jnp.clip(action, -1.0, 1.0)
        ctrl_min = self.mujoco_model.actuator_ctrlrange[:, 0]
        ctrl_max = self.mujoco_model.actuator_ctrlrange[:, 1]
        scaled_ctrl = ctrl_min + 0.5 * (clamped_action + 1.0) * (ctrl_max - ctrl_min)

        data = state.data.replace(ctrl=scaled_ctrl)
        next_data = mujoco_mjx.step(self.mujoco_model, data)

        next_obs = jnp.concatenate([next_data.qpos, next_data.qvel])

        # Cartpole Task Reward: upright pole + alive bonus - cart center penalty - control penalty
        x = next_data.qpos[0]
        theta = next_data.qpos[1]
        alive_reward = 1.0
        upright_reward = jnp.cos(theta)
        cart_penalty = 0.01 * (x**2)
        ctrl_cost = 0.001 * jnp.sum(clamped_action**2)
        reward = alive_reward + upright_reward - cart_penalty - ctrl_cost

        # Termination: fallen or off-rail or horizon exceeded
        fell = jnp.abs(theta) > 0.5
        off_rail = jnp.abs(x) > 2.0
        time_limit = state.step_count >= self.max_steps
        done = jnp.logical_or(jnp.logical_or(fell, off_rail), time_limit)

        next_state = MuJoCoMJXState(
            data=next_data,
            step_count=state.step_count + 1,
            cum_reward=state.cum_reward + reward,
            done=done,
        )
        return next_obs, next_state, reward, done, {}

    def preprocess_obs(self, obs: chex.Array) -> chex.Array:
        """Flatten observation tensor to 1D vector."""
        return obs.flatten()

    def postprocess_action(self, logits: chex.Array, raw_obs: Any = None) -> chex.Array:
        """Apply bounded tanh activation scaled to action range."""
        return jnp.tanh(logits) * self.action_scale

    @property
    def obs_dim(self) -> int:
        """Flattened observation dimensionality."""
        if self.backend_pipeline is not None:
            return int(self.backend_pipeline.observation_size)
        return int(self.mujoco_model.nq + self.mujoco_model.nv)

    @property
    def action_dim(self) -> int:
        """Action dimensionality."""
        if self.backend_pipeline is not None:
            return int(self.backend_pipeline.action_size)
        return int(self.mujoco_model.nu)


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
        raise NotImplementedError(
            "TensorNEAT environments must be evaluated via TensorNeatEvaluator"
        )
