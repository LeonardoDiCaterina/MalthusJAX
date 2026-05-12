from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pytest
from flax import struct

from malthusjax.engine.base import (
    AbstractEngine,
    AbstractEngineParams,
    AbstractEvolutionState,
    AbstractGenerationOutput,
    validate_engine_params,
)

# --- Mocks for Concrete Implementation ---


@struct.dataclass
class MockParams(AbstractEngineParams):
    pass


@struct.dataclass
class MockState(AbstractEvolutionState):
    pass


@struct.dataclass
class MockOutput(AbstractGenerationOutput):
    """Mock output with required fields from AbstractGenerationOutput."""

    pass


@struct.dataclass
class ConcreteEngine(AbstractEngine):
    """Minimal concrete implementation for testing AbstractEngine logic."""

    def init_state(self, rng_key: chex.Array) -> AbstractEvolutionState:
        return MockState(
            population=None,  # type: ignore
            best_genome=None,  # type: ignore
            generation=0,
            best_fitness=jnp.array(-1.0),
            rng_key=rng_key,
        )

    def step(self, state: AbstractEvolutionState) -> Tuple[AbstractEvolutionState, MockOutput]:
        # Simple increment logic
        new_gen = state.generation + 1
        new_state = state.replace(generation=new_gen)
        output = MockOutput(
            best_fitness=jnp.array(0.0), mean_fitness=jnp.array(0.0), generation=jnp.array(new_gen)
        )
        return new_state, output

    def ask(self, state: AbstractEvolutionState):
        return self, state.population

    def tell(self, state: AbstractEvolutionState, population):
        return state.replace(population=population)


@struct.dataclass
class EngineWithoutAskTell(AbstractEngine):
    def init_state(self, rng_key: chex.Array) -> AbstractEvolutionState:
        return MockState(
            population=None,  # type: ignore
            best_genome=None,  # type: ignore
            generation=0,
            best_fitness=jnp.array(-1.0),
            rng_key=rng_key,
        )

    def step(self, state: AbstractEvolutionState) -> Tuple[AbstractEvolutionState, MockOutput]:
        return state, MockOutput(
            best_fitness=jnp.array(0.0),
            mean_fitness=jnp.array(0.0),
            generation=jnp.array(state.generation),
        )


# --- Tests ---


def test_param_validation():
    """Test the validation logic for engine parameters."""
    # Valid params
    p = MockParams(pop_size=10, num_generations=5, elitism=2)
    validate_engine_params(p)

    # Invalid pop_size
    with pytest.raises(ValueError):
        validate_engine_params(MockParams(pop_size=0, num_generations=5, elitism=0))

    # Invalid elitism (>= pop_size)
    with pytest.raises(ValueError):
        validate_engine_params(MockParams(pop_size=10, num_generations=5, elitism=10))

    # Negative generations
    with pytest.raises(ValueError):
        validate_engine_params(MockParams(pop_size=10, num_generations=-1, elitism=2))


def test_engine_compilation_and_run():
    """Test that the abstract engine correctly compiles and runs the loop."""
    params = MockParams(pop_size=10, num_generations=5, elitism=1)
    engine = ConcreteEngine(engine_params=params)
    key = jax.random.PRNGKey(0)

    state = engine.init_state(key)

    # Run in compiled mode
    final_state, history, _ = engine.run(state, compile=True)

    # Check if loop ran for 5 generations
    assert final_state.generation == 5
    # Check if history captured 5 steps
    assert history.generation.shape[0] == 5


def test_ask_with_key_delegates_to_ask():
    params = MockParams(pop_size=4, num_generations=1, elitism=0)
    engine = ConcreteEngine(engine_params=params)
    state = engine.init_state(jax.random.PRNGKey(0))

    returned_engine, population = engine.ask_with_key(state, jax.random.PRNGKey(1))

    assert returned_engine is engine
    assert population is state.population


def test_tell_with_key_delegates_to_tell():
    params = MockParams(pop_size=4, num_generations=1, elitism=0)
    engine = ConcreteEngine(engine_params=params)
    state = engine.init_state(jax.random.PRNGKey(0))

    updated = engine.tell_with_key(state, state.population, jax.random.PRNGKey(2))
    assert updated.population is state.population


def test_key_aware_methods_raise_without_ask_tell():
    params = MockParams(pop_size=4, num_generations=1, elitism=0)
    engine = EngineWithoutAskTell(engine_params=params)
    state = engine.init_state(jax.random.PRNGKey(0))

    with pytest.raises(NotImplementedError, match="ask_with_key"):
        _ = engine.ask_with_key(state, jax.random.PRNGKey(1))

    with pytest.raises(NotImplementedError, match="tell_with_key"):
        _ = engine.tell_with_key(state, state.population, jax.random.PRNGKey(2))
