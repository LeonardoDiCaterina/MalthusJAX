# mypy: ignore-errors
import dataclasses
from typing import Any

import jax
import pytest

from malthusjax.core.fitness.composable.base import BaseInterpreter


class InterpreterComplianceSuite:
    """Standalone compliance suite for custom MalthusJAX Interpreters.

    To use this suite, inherit from this class and implement the following fixtures:
    - `component`: Returns an instantiated interpreter.
    - `mock_genome`: Returns a single unbatched genome (matching the interpreter's expected genome type).
    - `mock_inputs`: Returns mock inputs (like an observation or features), or None if the interpreter doesn't use them.
    """

    @pytest.fixture
    def component(self) -> BaseInterpreter[Any]:
        raise NotImplementedError("You must implement the `component` fixture.")

    @pytest.fixture
    def mock_genome(self) -> Any:
        raise NotImplementedError("You must implement the `mock_genome` fixture.")
        
    @pytest.fixture
    def mock_inputs(self) -> Any:
        raise NotImplementedError("You must implement the `mock_inputs` fixture.")

    def test_is_dataclass(self, component) -> None:
        assert dataclasses.is_dataclass(component), "Component must be a dataclass."

    def test_inheritance(self, component) -> None:
        assert isinstance(component, BaseInterpreter), "Component must inherit from BaseInterpreter."
        
    def test_num_params_type(self, component) -> None:
        assert isinstance(component.num_params, int), "num_params must return an integer."

    def test_apply_is_jittable(self, component, mock_genome, mock_inputs) -> None:
        """Verify that apply executes purely inside jax.jit."""
        jitted_apply = jax.jit(component.apply)
        try:
            output = jitted_apply(mock_genome, mock_inputs)
            assert output is not None
        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(f"JIT compilation failed due to a tracer leak.\nDetails: {e}")
            
    def test_apply_is_vmappable(self, component, mock_genome, mock_inputs) -> None:
        """Verify that apply can be vmapped over inputs."""
        if mock_inputs is None:
            pytest.skip("Interpreter does not take inputs, skipping vmap test.")
        
        # Create a batch of 5 identical inputs
        batched_inputs = jax.tree_map(lambda x: jax.numpy.stack([x]*5), mock_inputs)
        
        vmapped_apply = jax.vmap(component.apply, in_axes=(None, 0))
        try:
            output = vmapped_apply(mock_genome, batched_inputs)
            assert output is not None
            # Check if output has a batch dimension of size 5
            if hasattr(output, "shape"):
                assert output.shape[0] == 5
        except Exception as e:
            pytest.fail(f"vmap compilation/execution failed.\nDetails: {e}")

    def test_stateless(self, component, mock_genome, mock_inputs) -> None:
        """Verify that multiple calls with the same inputs return the exact same outputs."""
        output1 = component.apply(mock_genome, mock_inputs)
        output2 = component.apply(mock_genome, mock_inputs)
        
        # Deep compare outputs
        jax.tree_util.tree_map(
            lambda x, y: jax.numpy.testing.assert_allclose(x, y),
            output1, output2
        )
