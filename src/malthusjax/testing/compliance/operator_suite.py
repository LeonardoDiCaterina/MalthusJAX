# mypy: ignore-errors
import dataclasses
from typing import Any

import jax
import pytest

from malthusjax.core.base import BasePopulation
from malthusjax.operators.base import BaseCrossover, BaseMutation, BaseSelection


class MutationComplianceSuite:
    """Standalone compliance suite for custom MalthusJAX Mutation operators.

    To use this suite, inherit from this class and implement the following fixtures:
    - `component`: Returns an instantiated mutation operator.
    - `mock_population`: Returns a `BasePopulation[Any]` instance to mutate.
    """

    @pytest.fixture
    def component(self) -> BaseMutation[Any, Any]:
        raise NotImplementedError("You must implement the `component` fixture.")

    @pytest.fixture
    def mock_population(self) -> BasePopulation[Any]:
        raise NotImplementedError("You must implement the `mock_population` fixture.")

    def test_is_dataclass(self, component) -> None:
        """Verify the operator is an immutable flax.struct.dataclass."""
        assert dataclasses.is_dataclass(component), (
            f"{component.__class__.__name__} is not a dataclass. "
            "Did you forget to inherit from BaseMutation[Any, Any] and apply @flax.struct.dataclass?"
        )

    def test_has_registry_metadata(self, component) -> None:
        """Verify the operator was correctly registered."""
        assert hasattr(component, "_malthusjax_metadata"), (
            f"{component.__class__.__name__} is missing registry metadata. "
            "Did you forget to decorate the class with @register_mutation?"
        )

    def test_inheritance(self, component) -> None:
        """Verify the component inherits from the correct base class."""
        assert isinstance(component, BaseMutation), (
            f"{component.__class__.__name__} must inherit from BaseMutation."
        )

    def test_jit_compilation(self, component, mock_population) -> None:
        """Verify the mutation operator executes purely inside jax.jit without leaking tracers."""
        input_shape = jax.tree_util.tree_leaves(mock_population.genes)[0].shape

        # MalthusJAX keys calculation
        total_keys = component.num_keys(input_shape)
        keys = jax.random.split(jax.random.PRNGKey(0), total_keys)

        # Ensure JIT compiles cleanly
        jitted_call = jax.jit(component.__call__)
        try:
            mutated_pop = jitted_call(keys, mock_population, config=None, generation=0)
            assert mutated_pop is not None
        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(f"JIT compilation failed due to a tracer leak (e.g., using Python 'if' on dynamic JAX arrays).\nDetails: {e}")

    def test_output_shape_contract(self, component, mock_population) -> None:
        """Verify the mutation produces the correct number of offspring."""
        input_shape = jax.tree_util.tree_leaves(mock_population.genes)[0].shape
        keys = jax.random.split(jax.random.PRNGKey(0), component.num_keys(input_shape))

        mutated_pop = component(keys, mock_population, config=None, generation=0)

        expected_size = input_shape[0] * component.num_offspring
        actual_size = len(mutated_pop)

        assert actual_size == expected_size, (
            f"Shape contract violation: Expected {expected_size} individuals after mutation, "
            f"but got {actual_size}. Check your num_offspring implementation."
        )


class CrossoverComplianceSuite:
    """Standalone compliance suite for custom MalthusJAX Crossover operators."""

    @pytest.fixture
    def component(self) -> BaseCrossover[Any, Any]:
        raise NotImplementedError("You must implement the `component` fixture.")

    @pytest.fixture
    def mock_pop1(self) -> BasePopulation[Any]:
        raise NotImplementedError("You must implement the `mock_pop1` fixture.")

    @pytest.fixture
    def mock_pop2(self) -> BasePopulation[Any]:
        raise NotImplementedError("You must implement the `mock_pop2` fixture.")

    def test_is_dataclass(self, component) -> None:
        assert dataclasses.is_dataclass(component), "Component must be a dataclass."

    def test_has_registry_metadata(self, component) -> None:
        assert hasattr(component, "_malthusjax_metadata"), "Missing @register_crossover decorator."

    def test_inheritance(self, component) -> None:
        assert isinstance(component, BaseCrossover[Any, Any]), "Component must inherit from BaseCrossover[Any, Any]."

    def test_jit_compilation(self, component, mock_pop1, mock_pop2) -> None:
        input_shape = jax.tree_util.tree_leaves(mock_pop1.genes)[0].shape
        keys = jax.random.split(jax.random.PRNGKey(0), component.num_keys(input_shape))

        jitted_call = jax.jit(component.__call__)
        try:
            crossed_pop = jitted_call(keys, mock_pop1, mock_pop2, config=None, generation=0)
            assert crossed_pop is not None
        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(f"JIT compilation failed due to a tracer leak.\nDetails: {e}")

    def test_output_shape_contract(self, component, mock_pop1, mock_pop2) -> None:
        input_shape = jax.tree_util.tree_leaves(mock_pop1.genes)[0].shape
        keys = jax.random.split(jax.random.PRNGKey(0), component.num_keys(input_shape))

        crossed_pop = component(keys, mock_pop1, mock_pop2, config=None, generation=0)

        expected_size = input_shape[0] * component.num_offspring
        actual_size = len(crossed_pop)

        assert actual_size == expected_size, (
            f"Shape contract violation: Expected {expected_size} individuals, got {actual_size}."
        )


class SelectionComplianceSuite:
    """Standalone compliance suite for custom MalthusJAX Selection operators."""

    @pytest.fixture
    def component(self) -> BaseSelection[Any, Any]:
        raise NotImplementedError("You must implement the `component` fixture.")

    @pytest.fixture
    def mock_fitness(self) -> jax.Array:
        raise NotImplementedError("You must implement the `mock_fitness` fixture.")

    def test_is_dataclass(self, component) -> None:
        assert dataclasses.is_dataclass(component), "Component must be a dataclass."

    def test_has_registry_metadata(self, component) -> None:
        assert hasattr(component, "_malthusjax_metadata"), "Missing @register_selection decorator."

    def test_inheritance(self, component) -> None:
        assert isinstance(component, BaseSelection[Any, Any]), "Component must inherit from BaseSelection[Any, Any]."

    def test_jit_compilation(self, component, mock_fitness) -> None:
        keys = jax.random.split(jax.random.PRNGKey(0), component.num_keys(mock_fitness.shape))

        jitted_call = jax.jit(component.__call__)
        try:
            parents, elites = jitted_call(keys, mock_fitness, config=None)
            assert parents is not None
        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(f"JIT compilation failed due to a tracer leak.\nDetails: {e}")

    def test_output_shape_contract(self, component, mock_fitness) -> None:
        keys = jax.random.split(jax.random.PRNGKey(0), component.num_keys(mock_fitness.shape))
        parents, elites = component(keys, mock_fitness, config=None)

        assert parents.shape == (component.num_selections,), (
            f"Shape contract violation: Expected parent shape ({component.num_selections},), "
            f"got {parents.shape}."
        )
        assert elites.shape == (component.n_elites,), (
            f"Shape contract violation: Expected elite shape ({component.n_elites},), "
            f"got {elites.shape}."
        )
