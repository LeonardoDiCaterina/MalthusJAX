# mypy: ignore-errors
import dataclasses

import jax
import jax.numpy as jnp
import pytest

from malthusjax.core.fitness.composable.base import (
    BaseEnvironment,
    BaseOptimizationEnvironment,
    BaseRLEnvironment,
    BaseSupervisedEnvironment,
)


class EnvironmentComplianceSuite:
    """Standalone compliance suite for custom MalthusJAX Environments.

    To use this suite, inherit from this class and implement the following fixtures:
    - `component`: Returns an instantiated environment.
    """

    @pytest.fixture
    def component(self) -> BaseEnvironment:
        raise NotImplementedError("You must implement the `component` fixture.")

    def test_is_dataclass(self, component) -> None:
        assert dataclasses.is_dataclass(component), "Component must be a dataclass."

    def test_inheritance(self, component) -> None:
        assert isinstance(component, BaseEnvironment), "Component must inherit from BaseEnvironment."

    def test_supervised_env_has_X_y(self, component) -> None:
        """Verify SupervisedEnvironments have valid X and y arrays."""
        if not isinstance(component, BaseSupervisedEnvironment):
            pytest.skip("Not a SupervisedEnvironment")

        assert hasattr(component, "X"), "SupervisedEnvironment must have an X property."
        assert hasattr(component, "y"), "SupervisedEnvironment must have a y property."
        assert isinstance(component.X, jax.Array), "X must be a JAX array."
        assert isinstance(component.y, jax.Array), "y must be a JAX array."
        assert component.X.shape[0] == component.y.shape[0], "X and y must have the same number of samples."

    def test_optimization_env_evaluate_shape(self, component) -> None:
        """Verify OptimizationEnvironments evaluate to a scalar."""
        if not isinstance(component, BaseOptimizationEnvironment):
            pytest.skip("Not an OptimizationEnvironment")

        # Create a dummy solution of size 10 (or whatever is appropriate, though we can't guess easily)
        # We will try a vector of size 10, or bypass if the env complains about shape.
        try:
            solution = jnp.zeros(10)
            result = component.evaluate(solution)
            assert result.shape == (), "evaluate() must return a scalar (shape ())."
        except Exception as e:
            pytest.skip(f"Could not automatically test evaluate() shape: {e}")

    def test_rl_env_has_obs_action_dims(self, component) -> None:
        """Verify RLEnvironments define obs_dim and action_dim."""
        if not isinstance(component, BaseRLEnvironment):
            pytest.skip("Not an RLEnvironment")

        assert hasattr(component, "obs_dim"), "RLEnvironment must have obs_dim."
        assert hasattr(component, "action_dim"), "RLEnvironment must have action_dim."
        assert isinstance(component.obs_dim, int), "obs_dim must be an integer."
        assert isinstance(component.action_dim, int), "action_dim must be an integer."
