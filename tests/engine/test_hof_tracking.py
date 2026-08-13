"""
Tests for TrackBest modes (FB-4 HOF redesign).
"""

import chex
import jax.numpy as jnp
import pytest

from malthusjax.engine.genetic_fastengine import GeneticEngineParams
from malthusjax.engine.schedules import TrackBest


def test_track_best_none(make_engine, prng_key):
    engine = make_engine(track_best=TrackBest.NONE, pop_size=20, num_generations=10)
    state = engine.init_state(prng_key)
    final, history, _ = engine.run(state, compile=False)

    assert final.generation == 10
    chex.assert_shape(history.best_fitness, (10,))

    expected_best = jnp.max(final.population.fitness)
    assert abs(float(final.best_fitness) - float(expected_best)) < 30.0

    chex.assert_tree_all_finite(history.best_fitness)

    monotonic = jnp.maximum.accumulate(history.best_fitness)
    diffs = jnp.diff(monotonic)
    assert jnp.all(diffs >= -1e-7)


def test_track_best_light(make_engine, prng_key):
    engine = make_engine(track_best=TrackBest.LIGHT, pop_size=20, num_generations=10)
    state = engine.init_state(prng_key)
    final, history, _ = engine.run(state, compile=False)

    assert final.generation == 10

    diffs = jnp.diff(history.best_fitness)
    assert jnp.all(diffs <= 1e-7)  # minimization, so non-increasing

    chex.assert_shape(final.best_genome.values, (10,))


def test_light_mode_step_does_not_track_genome(make_engine, prng_key):
    engine = make_engine(track_best=TrackBest.LIGHT, pop_size=20, num_generations=10)
    state = engine.init_state(prng_key)

    new_state, _ = engine.step(state)
    chex.assert_trees_all_close(state.best_genome.values, new_state.best_genome.values)


def test_track_best_full(make_engine, prng_key):
    engine = make_engine(track_best=TrackBest.FULL, pop_size=20, num_generations=10)
    state = engine.init_state(prng_key)
    final, history, _ = engine.run(state, compile=False)

    assert final.generation == 10
    diffs = jnp.diff(history.best_fitness)
    assert jnp.all(diffs <= 1e-7)

    assert float(final.best_fitness) >= float(state.best_fitness) - 100.0
    chex.assert_shape(final.best_genome.values, (10,))


def test_stagnation_from_history(make_engine, prng_key):
    engine = make_engine(track_best=TrackBest.LIGHT, pop_size=20, num_generations=10)
    state = engine.init_state(prng_key)
    _, history, _ = engine.run(state, compile=False)

    monotonic = jnp.maximum.accumulate(history.best_fitness)
    diffs = jnp.diff(monotonic)
    stagnation = (diffs == 0.0).astype(jnp.int32)
    chex.assert_shape(stagnation, (9,))


def test_default_track_best():
    params = GeneticEngineParams(pop_size=10, num_generations=5, elitism=0)
    assert params.track_best == TrackBest.LIGHT


@pytest.mark.parametrize("track_best", [TrackBest.NONE, TrackBest.LIGHT, TrackBest.FULL])
def test_track_best_jit(make_engine, prng_key, track_best):
    engine = make_engine(track_best=track_best, pop_size=20, num_generations=10)
    state = engine.init_state(prng_key)
    final, history, _ = engine.run(state, compile=True)
    assert final.generation == 10
    chex.assert_tree_all_finite(history.best_fitness)
