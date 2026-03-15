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
