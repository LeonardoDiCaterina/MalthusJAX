# mypy: ignore-errors
import dataclasses

import jax
from typing import Any
import pytest

from malthusjax.engine.base import AbstractEngine, AbstractEvolutionState, AbstractGenerationOutput


class EngineComplianceSuite:
    """Standalone compliance suite for custom MalthusJAX Engines.

    To use this suite, inherit from this class and implement the following fixtures:
    - `component`: Returns an instantiated engine.
    """

    @pytest.fixture
    def component(self) -> AbstractEngine:
        raise NotImplementedError("You must implement the `component` fixture.")

    def test_is_dataclass(self, component) -> None:
        assert dataclasses.is_dataclass(component), "Engine must be a dataclass."

    def test_inheritance(self, component) -> None:
        assert isinstance(component, AbstractEngine), "Engine must inherit from AbstractEngine."

    def test_init_state_jit(self, component) -> None:
        """Verify init_state can be JIT compiled and returns the correct state type."""
        jitted_init = jax.jit(component.init_state)
        rng = jax.random.PRNGKey(0)

        try:
            state = jitted_init(rng)
        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(f"init_state JIT compilation failed due to a tracer leak.\nDetails: {e}")

        assert isinstance(state, AbstractEvolutionState), (
            f"init_state must return a subclass of AbstractEvolutionState, got {type(state)}"
        )

    def test_step_jit(self, component) -> None:
        """Verify step can be JIT compiled and returns the correct tuple."""
        # We don't JIT init_state here just to get a clean state, though it should be safe
        rng = jax.random.PRNGKey(0)
        state = component.init_state(rng)

        jitted_step = jax.jit(component.step)
        try:
            next_state, output = jitted_step(state)
        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(f"step JIT compilation failed due to a tracer leak.\nDetails: {e}")

        assert isinstance(next_state, AbstractEvolutionState), "step must return AbstractEvolutionState"
        assert isinstance(output, AbstractGenerationOutput), "step must return AbstractGenerationOutput"
