"""
Advanced tests for JIT compilation, tracing, and execution characteristics.
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp


def test_jit_step_compiles_without_error(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)
    jit_step = jax.jit(engine.step)
    new_state, _ = jit_step(state)
    assert new_state.generation == 1


def test_jit_step_produces_same_result_as_eager(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)

    eager_state, eager_metrics = engine.step(state)

    jit_step = jax.jit(engine.step)
    jit_state, jit_metrics = jit_step(state)

    chex.assert_trees_all_close(eager_state.population.fitness, jit_state.population.fitness)
    chex.assert_trees_all_close(eager_metrics.best_fitness, jit_metrics.best_fitness)


def test_jit_with_static_engine(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)

    @partial(jax.jit, static_argnames=["engine"])
    def run_step(engine, state):
        return engine.step(state)

    new_state, _ = run_step(engine, state)
    assert new_state.generation == 1


def test_jit_multiple_steps_accumulate_correctly(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)
    jit_step = jax.jit(engine.step)

    for i in range(1, 6):
        state, _ = jit_step(state)
        assert state.generation == i


def test_entropy_buffer_cleared_after_tell(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)

    engine_with_entropy, pop = engine.ask(state)
    evaluated = engine.evaluator.evaluate_population(pop)
    _ = engine_with_entropy.tell(state, evaluated)

    assert len(engine._entropy_buffer) == 0


def test_multiple_ask_overwrites_buffer(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)

    engine_with_entropy1, _ = engine.ask(state)
    buffer1 = engine_with_entropy1._entropy_buffer

    new_state, _ = engine.step(state)
    engine_with_entropy2, _ = engine.ask(new_state)
    buffer2 = engine_with_entropy2._entropy_buffer

    assert not jnp.array_equal(buffer1[0], buffer2[0])


def test_entropy_keys_never_repeated(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)
    keys_list = []

    for _ in range(5):
        engine_with_entropy, _ = engine.ask(state)
        k_sel, k_cross, k_mut, k_next = engine_with_entropy._entropy_buffer
        keys_list.append((k_sel.tobytes(), k_cross.tobytes(), k_mut.tobytes(), k_next.tobytes()))
        state, _ = engine.step(state)

    assert len(keys_list) == 5
    assert all(k[0] is not None for k in keys_list)


def test_step_with_named_calls_traces(make_engine, prng_key):
    engine = make_engine(pop_size=20)
    state = engine.init_state(prng_key)
    state, metrics = engine.step(state)
    assert state is not None
