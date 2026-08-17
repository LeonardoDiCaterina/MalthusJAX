"""
Tests for Ask/Tell (async) interface of GeneticEngine.

Tests focus on the async evolution pattern where ask() allocates entropy
and tell() consumes it, allowing external evaluation between phases.
"""

import chex
import jax
import jax.numpy as jnp
import pytest


def test_ask_returns_engine_and_population(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)

    engine_with_entropy, population = engine.ask(state)
    assert engine_with_entropy._entropy_buffer is not None
    assert len(engine_with_entropy._entropy_buffer) == 5
    assert population is not None
    chex.assert_shape(population.fitness, (30,))


def test_ask_entropy_buffer_contains_keys(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)
    engine_with_entropy, _ = engine.ask(state)

    k_sel, k_cross, k_mut, k_eval, k_next = engine_with_entropy._entropy_buffer
    for key in [k_sel, k_cross, k_mut, k_eval]:
        assert key.shape[-1] == 2
        assert len(key.shape) >= 1
    assert k_next.shape == (2,)


def test_tell_requires_ask_to_be_called_first(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)

    with pytest.raises(RuntimeError, match="tell.. called before ask.."):
        engine.tell(state, state.population)


def test_tell_updates_state_correctly(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)

    engine_with_entropy, population = engine.ask(state)
    new_fitness = population.fitness * 0.9
    evaluated_pop = population.replace(fitness=new_fitness)

    new_state = engine_with_entropy.tell(state, evaluated_pop)
    assert new_state.generation == state.generation + 1
    chex.assert_shape(new_state.population.fitness, (30,))


@pytest.mark.integration
def test_ask_tell_loop_produces_evolution(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)
    initial_gen = state.generation

    for _ in range(3):
        engine_with_entropy, population = engine.ask(state)
        evaluated_pop = engine.evaluator.evaluate_population(population)
        state = engine_with_entropy.tell(state, evaluated_pop)

    assert state.generation == initial_gen + 3
    chex.assert_tree_all_finite(state.best_fitness)


def test_tell_updates_best_genome_on_improvement(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)

    new_fitness = state.population.fitness - 5.0
    better_pop = state.population.replace(fitness=new_fitness)

    engine_with_entropy, _ = engine.ask(state)
    updated_state = engine_with_entropy.tell(state, better_pop)

    best_idx = jnp.argmin(better_pop.fitness)
    expected_genome = better_pop[best_idx].genes

    chex.assert_trees_all_close(updated_state.best_genome, expected_genome)


def test_multiple_ask_tell_cycles_maintain_consistency(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)

    for i in range(5):
        chex.assert_shape(state.population.fitness, (30,))
        engine_with_entropy, pop = engine.ask(state)
        chex.assert_shape(pop.fitness, (30,))

        evaluated = engine.evaluator.evaluate_population(pop)
        state = engine_with_entropy.tell(state, evaluated)
        assert state.generation == i + 1


def test_ask_entropy_buffer_survives_engine_copy(make_engine, prng_key):
    engine = make_engine(pop_size=30)
    state = engine.init_state(prng_key)
    engine_with_entropy, _ = engine.ask(state)

    assert engine_with_entropy._entropy_buffer is not None
    same_engine = engine_with_entropy
    assert len(same_engine._entropy_buffer) == 4


def test_ask_tell_produces_same_progression_with_same_seed(make_engine):
    engine = make_engine(pop_size=25, genome_shape=(2,), num_generations=5)

    # Sync evolution
    state_sync = engine.init_state(jax.random.PRNGKey(100))
    best_sync = []
    for _ in range(3):
        state_sync, metrics = engine.step(state_sync)
        best_sync.append(float(metrics.best_fitness))

    # Async evolution
    state_async = engine.init_state(jax.random.PRNGKey(100))
    best_async = []
    for _ in range(3):
        engine_with_entropy, pop = engine.ask(state_async)
        evaluated = engine.evaluator.evaluate_population(pop)
        state_async = engine_with_entropy.tell(state_async, evaluated)
        best_async.append(float(state_async.best_fitness))

    assert len(best_sync) == 3
    assert len(best_async) == 3


def test_ask_tell_equivalence(make_engine, prng_key):
    engine = make_engine()
    state_0 = engine.init_state(prng_key)

    state_step, _ = engine.step(state_0)

    engine_with_entropy, _ = engine.ask(state_0)
    state_tell = engine_with_entropy.tell(state_0, state_0.population)

    genes_step = state_step.population.genes.values
    genes_tell = state_tell.population.genes.values
    chex.assert_trees_all_close(genes_step, genes_tell)

    chex.assert_trees_all_close(state_step.rng_key, state_tell.rng_key)
