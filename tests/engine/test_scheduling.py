"""
Tests for mutation strength scheduling in GeneticEngine.
"""
import pytest
import jax
from malthusjax.engine.schedules import ScheduleType, compute_scheduled_strength
from malthusjax.engine.genetic_fastengine import GeneticEngineParams

def test_constant_returns_initial():
    for gen in [0, 5, 10]:
        result = compute_scheduled_strength(
            ScheduleType.CONSTANT, gen, 10, initial_strength=0.5
        )
        assert abs(float(result) - 0.5) < 1e-5

def test_linear_decay_endpoints():
    s0 = compute_scheduled_strength(ScheduleType.LINEAR_DECAY, 0, 100, initial_strength=1.0, final_strength=0.1)
    assert abs(float(s0) - 1.0) < 1e-5
    
    s100 = compute_scheduled_strength(ScheduleType.LINEAR_DECAY, 100, 100, initial_strength=1.0, final_strength=0.1)
    assert abs(float(s100) - 0.1) < 1e-5

def test_linear_decay_midpoint():
    s50 = compute_scheduled_strength(ScheduleType.LINEAR_DECAY, 50, 100, initial_strength=1.0, final_strength=0.0)
    assert abs(float(s50) - 0.5) < 1e-5

def test_cosine_anneal_endpoints():
    s0 = compute_scheduled_strength(ScheduleType.COSINE_ANNEAL, 0, 100, initial_strength=1.0, final_strength=0.0)
    assert abs(float(s0) - 1.0) < 1e-4
    
    s100 = compute_scheduled_strength(ScheduleType.COSINE_ANNEAL, 100, 100, initial_strength=1.0, final_strength=0.0)
    assert abs(float(s100) - 0.0) < 1e-4

def test_exponential_decay_decreases():
    s0 = compute_scheduled_strength(ScheduleType.EXPONENTIAL_DECAY, 0, 100, initial_strength=1.0)
    s50 = compute_scheduled_strength(ScheduleType.EXPONENTIAL_DECAY, 50, 100, initial_strength=1.0)
    assert float(s0) > float(s50)

def test_jit_safe():
    @jax.jit
    def _compute(gen):
        return compute_scheduled_strength(ScheduleType.LINEAR_DECAY, gen, 100, initial_strength=1.0, final_strength=0.0)
    result = _compute(50)
    assert abs(float(result) - 0.5) < 1e-5

def test_schedule_type_is_int_enum():
    assert isinstance(ScheduleType.CONSTANT.value, int)
    assert ScheduleType.CONSTANT == 0
    assert ScheduleType.LINEAR_DECAY == 1

def test_engine_runs_with_linear_decay(make_engine, prng_key):
    engine = make_engine(schedule_type=ScheduleType.LINEAR_DECAY, pop_size=30, num_generations=10)
    state = engine.init_state(prng_key)
    for _ in range(3):
        state, _ = engine.step(state)
    assert state.generation == 3

def test_engine_runs_with_cosine_anneal(make_engine, prng_key):
    engine = make_engine(schedule_type=ScheduleType.COSINE_ANNEAL, pop_size=30, num_generations=10)
    state = engine.init_state(prng_key)
    for _ in range(3):
        state, _ = engine.step(state)
    assert state.generation == 3

@pytest.mark.parametrize("schedule_type", list(ScheduleType))
def test_different_schedules_complete(make_engine, prng_key, schedule_type):
    engine = make_engine(schedule_type=schedule_type, pop_size=25, genome_shape=(2,), num_generations=10)
    state = engine.init_state(prng_key)
    for _ in range(3):
        state, _ = engine.step(state)
    assert state.generation == 3
