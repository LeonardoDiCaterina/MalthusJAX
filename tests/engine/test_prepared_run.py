"""Compilation belongs to preparation, including the first measured seed."""

import time
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import malthusjax.composer.adapters.map_elites_adapter as map_adapter
import malthusjax.engine.base as base
from malthusjax.core.logger import StepLoggingConfig
from tests.timing_helpers import TimingEngine


def test_equivalent_kernel_requests_share_one_function():
    params = base.AbstractEngineParams(pop_size=4, num_generations=3)
    default = base._get_evolution_kernel(params)
    explicit = base._get_evolution_kernel(
        params, return_history=True, step_logging=None, unroll_num=1, compile_jit=True
    )
    positional = base._get_evolution_kernel(params, True, 1, True, None)
    assert default is explicit is positional
    logging = StepLoggingConfig(log_interval=2)
    params = params.replace(step_logging=logging)
    assert base._get_evolution_kernel(params) is base._get_evolution_kernel(
        params, step_logging=logging
    )


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("logging", [None, StepLoggingConfig(log_interval=2)])
def test_prepared_run_does_not_trace_or_compile_and_preserves_seed(history, logging, caplog):
    events = []
    engine = TimingEngine(
        engine_params=base.AbstractEngineParams(pop_size=4, num_generations=3),
        trace_hook=lambda: events.append("trace"),
    )
    for seed in (44, 42, 43):
        original = engine.init_state(jax.random.key(seed))
        working = original.copy()
        with jax.log_compiles(True):
            prepared = engine.prepare_run(original, return_history=history, step_logging=logging)
            events.clear()
            caplog.clear()
            final, output, _ = engine.run(
                working,
                return_history=history,
                step_logging=logging,
                prepared_run=prepared,
                time_it=True,
            )
            assert events == []
            assert not any("Finished XLA compilation" in r.message for r in caplog.records)
        reference, reference_output, _ = engine.run(
            original.copy(), compile=False, return_history=history, step_logging=logging
        )
        np.testing.assert_allclose(final.best_fitness, reference.best_fitness)
        np.testing.assert_array_equal(
            jax.random.key_data(final.rng_key), jax.random.key_data(reference.rng_key)
        )
        assert int(final.generation) == 3
        if history:
            np.testing.assert_allclose(output.best_fitness, reference_output.best_fitness)
        else:
            assert output == ()


def test_prepared_run_checks_engine_and_configuration():
    engine = TimingEngine(engine_params=base.AbstractEngineParams(pop_size=4, num_generations=2))
    state = engine.init_state(jax.random.key(1))
    prepared = engine.prepare_run(state)
    other = TimingEngine(engine_params=engine.engine_params)
    for kwargs in (
        {"compile": False},
        {"return_history": False},
        {"step_logging": StepLoggingConfig(log_interval=1)},
    ):
        with pytest.raises(ValueError):
            engine.run(state.copy(), prepared_run=prepared, **kwargs)
    with pytest.raises(ValueError):
        other.run(state.copy(), prepared_run=prepared)


def test_new_input_signature_is_prepared_before_execution():
    events = []
    engine = TimingEngine(
        engine_params=base.AbstractEngineParams(pop_size=4, num_generations=2),
        trace_hook=lambda: events.append("trace"),
    )
    for key in (jax.random.key(0), jax.random.PRNGKey(0)):
        state = engine.init_state(key)
        prepared = engine.prepare_run(state)
        events.clear()
        engine.run(state, prepared_run=prepared, time_it=True)
        assert events == []


def test_changed_dtype_requires_preparation_and_cannot_silently_recompile():
    import jax.numpy as jnp

    engine = TimingEngine(engine_params=base.AbstractEngineParams(pop_size=4, num_generations=2))
    state = engine.init_state(jax.random.key(0))
    prepared = engine.prepare_run(state)
    changed = state.replace(
        population=state.population.replace(genes=state.population.genes.astype(jnp.float16))
    )
    with pytest.raises(RuntimeError, match="Evolution failed"):
        engine.run(changed.copy(), prepared_run=prepared)
    updated = engine.prepare_run(changed)
    engine.run(changed, prepared_run=updated, time_it=True)


def test_real_native_adapter_has_no_trace_inside_execution(monkeypatch):
    phase = "outside"
    events = []
    engine = TimingEngine(
        engine_params=base.AbstractEngineParams(pop_size=4, num_generations=3),
        trace_hook=lambda: events.append(phase),
    )
    population = engine.init_state(jax.random.key(0)).population
    adapter = map_adapter.MapElitesEngineAdapter(engine, 4, False, None, population)
    for seed in (44, 42, 43):
        ticks = iter(("prepare", "between", "execution", "after"))

        def clock():
            nonlocal phase
            phase = next(ticks)
            return time.perf_counter()

        with monkeypatch.context() as patch:
            patch.setattr(map_adapter, "time", SimpleNamespace(perf_counter=clock))
            result = adapter.run_once(jax.random.key(seed))
        assert "execution" not in events
        assert result["summary"]["final_generation"] == 3


def test_native_adapter_propagates_compilation_failure(monkeypatch):
    engine = TimingEngine(engine_params=base.AbstractEngineParams(pop_size=4, num_generations=2))
    population = engine.init_state(jax.random.key(0)).population
    adapter = map_adapter.MapElitesEngineAdapter(engine, 4, False, None, population)

    class BrokenKernel:
        def lower(self, *args):
            raise RuntimeError("preparation failed")

    monkeypatch.setattr(base, "_get_evolution_kernel", lambda *args, **kwargs: BrokenKernel())
    with pytest.raises(RuntimeError, match="preparation failed"):
        adapter.run_once(jax.random.key(0))


def test_genetic_prepared_path_keeps_best_genome_finalization(genetic_engine):
    state = genetic_engine.init_state(jax.random.key(42))
    prepared = genetic_engine.prepare_run(state)
    final, history, _ = genetic_engine.run(state.copy(), prepared_run=prepared, time_it=True)
    reference, reference_history, _ = genetic_engine.run(state.copy(), time_it=True)
    np.testing.assert_allclose(final.best_fitness, reference.best_fitness)
    np.testing.assert_allclose(history.best_fitness, reference_history.best_fitness)
    for actual, expected in zip(
        jax.tree.leaves(final.best_genome), jax.tree.leaves(reference.best_genome)
    ):
        np.testing.assert_allclose(actual, expected)


def test_explicit_named_sharding_is_prepared_without_measured_trace():
    events = []
    engine = TimingEngine(
        engine_params=base.AbstractEngineParams(pop_size=4, num_generations=2),
        trace_hook=lambda: events.append("trace"),
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("device",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = jax.device_put(engine.init_state(jax.random.key(42)), sharding)
    prepared = engine.prepare_run(state)
    events.clear()
    final, _, _ = engine.run(state, prepared_run=prepared, time_it=True)
    assert events == []
    assert int(final.generation) == 2
