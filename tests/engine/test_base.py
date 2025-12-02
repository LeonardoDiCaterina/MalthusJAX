import pytest
import jax
import jax.numpy as jnp
from flax import struct
import chex
from typing import Tuple, Any

from malthusjax.engine.base import (
    AbstractEngine, AbstractEvolutionState, AbstractEngineParams, 
    AbstractGenerationOutput, validate_engine_params
)
from malthusjax.core.base import BaseGenome, BasePopulation

# --- Mocks for Concrete Implementation ---

@struct.dataclass
class MockParams(AbstractEngineParams):
    pass

@struct.dataclass
class MockState(AbstractEvolutionState):
    pass

@struct.dataclass
class MockOutput(AbstractGenerationOutput):
    pass

class ConcreteEngine(AbstractEngine):
    """Minimal concrete implementation for testing AbstractEngine logic."""
    
    def init_state(self, rng_key: chex.Array, params: AbstractEngineParams) -> AbstractEvolutionState:
        return MockState(
            population=None, # type: ignore
            fitness_values=jnp.zeros(10),
            best_genome=None, # type: ignore
            best_fitness=-1.0,
            generation=0,
            rng_key=rng_key,
            stagnation_counter=0
        )

    def step(self, key: chex.Array, state: AbstractEvolutionState, params: AbstractEngineParams) -> Tuple[chex.Array, AbstractEvolutionState, MockOutput]:
        # Simple increment logic
        new_gen = state.generation + 1
        new_state = state.replace(generation=new_gen)
        output = MockOutput(best_fitness=0.0, mean_fitness=0.0, generation=new_gen)
        return key, new_state, output

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
    engine = ConcreteEngine()
    params = MockParams(pop_size=10, num_generations=5, elitism=1)
    key = jax.random.PRNGKey(0)
    
    state = engine.init_state(key, params)
    
    # Run in compiled mode
    final_state, history, _ = engine.run(state, params, compile=True)
    
    # Check if loop ran for 5 generations
    assert final_state.generation == 5
    # Check if history captured 5 steps
    # Note: history structure depends on how scan returns it, usually (num_gens, ...)
    assert history.generation.shape[0] == 5