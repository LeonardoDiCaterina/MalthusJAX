# mypy: ignore-errors
import dataclasses
from typing import Any

import jax
import pytest

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.base import BaseEvaluator, StochasticEvaluator


class EvaluatorComplianceSuite:
    """Standalone compliance suite for custom MalthusJAX Evaluators.

    To use this suite, inherit from this class and implement the following fixtures:
    - `component`: Returns an instantiated evaluator.
    - `mock_population`: Returns a `BasePopulation[Any]` instance to evaluate.
    """

    @pytest.fixture
    def component(self) -> BaseEvaluator:
        raise NotImplementedError("You must implement the `component` fixture.")

    @pytest.fixture
    def mock_population(self) -> BasePopulation:
        raise NotImplementedError("You must implement the `mock_population` fixture.")

    def test_is_dataclass(self, component) -> None:
        assert dataclasses.is_dataclass(component), "Component must be a dataclass."

    def test_has_registry_metadata(self, component) -> None:
        assert hasattr(component, "_malthusjax_metadata"), "Missing @register_fitness decorator."

    def test_inheritance(self, component) -> None:
        assert isinstance(component, BaseEvaluator[Any]), "Component must inherit from BaseEvaluator[Any]."

    def test_jit_compilation(self, component, mock_population) -> None:
        """Verify the evaluator executes purely inside jax.jit."""
        rng = jax.random.PRNGKey(0)

        if isinstance(component, StochasticEvaluator):
            jitted_call = jax.jit(component.evaluate_population)
            try:
                evaluated_pop = jitted_call(mock_population, rng)
                assert evaluated_pop is not None
            except jax.errors.ConcretizationTypeError as e:
                pytest.fail(f"JIT compilation failed due to a tracer leak.\nDetails: {e}")
        else:
            jitted_call = jax.jit(component.evaluate_population)
            try:
                evaluated_pop = jitted_call(mock_population)
                assert evaluated_pop is not None
            except jax.errors.ConcretizationTypeError as e:
                pytest.fail(f"JIT compilation failed due to a tracer leak.\nDetails: {e}")

    def test_fitness_mutation(self, component, mock_population) -> None:
        """Verify that evaluate_population correctly updates the fitness field."""
        rng = jax.random.PRNGKey(0)

        if isinstance(component, StochasticEvaluator):
            evaluated_pop = component.evaluate_population(mock_population, rng)
        else:
            evaluated_pop = component.evaluate_population(mock_population)

        assert evaluated_pop.fitness.shape == mock_population.fitness.shape, (
            "Evaluated population fitness shape does not match original."
        )


class ComposableEvaluatorComplianceSuite:
    """Standalone compliance suite for composable MalthusJAX Evaluators (OptimizationEvaluator, SupervisedEvaluator, RLEvaluator).

    To use this suite, inherit from this class and implement the following fixtures:
    - `component`: Returns an instantiated evaluator (e.g., SupervisedEvaluator).
    - `mock_population`: Returns a `BasePopulation[Any]` instance to evaluate.
    """

    @pytest.fixture
    def component(self) -> Any:
        raise NotImplementedError("You must implement the `component` fixture.")

    @pytest.fixture
    def mock_population(self) -> BasePopulation:
        raise NotImplementedError("You must implement the `mock_population` fixture.")

    def test_is_dataclass(self, component) -> None:
        assert dataclasses.is_dataclass(component), "Component must be a dataclass."

    def test_evaluate_single(self, component, mock_population) -> None:
        """Verify that evaluate() works on a single genome and returns a scalar."""
        genome = jax.tree_map(lambda x: x[0], mock_population.genes)
        fitness = component.evaluate(genome)
        assert fitness.shape == (), "evaluate() must return a scalar fitness value."

    def test_evaluate_population_updates_fitness(self, component, mock_population) -> None:
        """Verify that evaluate_population correctly updates the fitness field."""
        evaluated_pop = component.evaluate_population(mock_population)
        assert evaluated_pop.fitness.shape == mock_population.fitness.shape, (
            "Evaluated population fitness shape does not match original."
        )

    def test_jittable(self, component, mock_population) -> None:
        """Verify that evaluate_population is jittable."""
        jitted_call = jax.jit(component.evaluate_population)
        try:
            evaluated_pop = jitted_call(mock_population)
            assert evaluated_pop is not None
        except jax.errors.ConcretizationTypeError as e:
            pytest.fail(f"JIT compilation failed due to a tracer leak.\nDetails: {e}")

