"""Production adapters must prepare outside timing or declare it unavailable."""

import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import malthusjax.composer.adapters.base as universal
from malthusjax.composer.adapters.metrics import MetricSpec


def make_universal(events, python_loop=False):
    def step(framework, state, key, *unused):
        events.append("step")
        new = state + jax.random.uniform(key)
        return new, {"best_fitness": new}

    return universal.UniversalAdapterEngine(
        framework_obj=None,
        framework_params=None,
        init_fn=lambda *args: jnp.float32(0),
        step_fn=step,
        eval_mode="native",
        eval_translator=lambda x: x,
        metrics_catalog=[MetricSpec("best_fitness", "best_fitness")],
        pop_size=4,
        num_generations=2,
        use_python_loop=python_loop,
    )


def test_universal_invokes_compiled_object_and_waits_for_inputs(monkeypatch):
    events = []
    adapter = make_universal(events)

    class Pending:
        def block_until_ready(self):
            events.append("input_ready")

    adapter.init_fn = lambda *args: SimpleNamespace(pending=Pending())

    def compiled(*args):
        events.append("compiled_execute")
        return jnp.float32(1), {"best_fitness": jnp.ones(2)}

    class Kernel:
        def lower(self, *args):
            events.append("lower")
            return self

        def compile(self):
            events.append("compile")
            return compiled

        def __call__(self, *args):
            raise AssertionError("The unprepared JIT callable was invoked")

    monkeypatch.setattr(adapter, "_build_jit_loop", lambda: Kernel())
    result = adapter.run_once(jax.random.key(42))
    assert events.index("input_ready") < events.index("lower")
    assert events.index("compile") < events.index("compiled_execute")
    assert result["timing_metadata"]["phase"] == "prepared_first_execution"


@pytest.mark.parametrize("mode", ["compiled", "python", "unprepared"])
def test_universal_mode_is_truthful_and_keeps_optimization_result(mode):
    events = []
    adapter = make_universal(events, python_loop=mode == "python")
    result = adapter.run_once(jax.random.key(42), compile=mode != "unprepared")
    assert np.isfinite(result["summary"]["best_fitness"])
    assert len(result["history"]) == 2
    metadata = result["timing_metadata"]
    if mode == "compiled":
        assert "execution" in result["timings"]
        assert metadata["status"] == "valid"
    else:
        assert "execution" not in result["timings"]
        assert metadata["status"] == "unavailable"
        assert metadata["reason"]


def test_universal_first_and_reordered_seeds_do_not_trace_during_execution(monkeypatch):
    events = []
    adapter = make_universal(events)
    for seed in (44, 42, 43):
        calls = 0
        during = []

        def clock():
            nonlocal calls
            calls += 1
            if calls == 3:
                events.clear()
            if calls == 4:
                during.extend(events)
            return time.perf_counter()

        with monkeypatch.context() as patch:
            patch.setattr(universal, "time", SimpleNamespace(perf_counter=clock))
            measured = adapter.run_once(jax.random.key(seed))
        assert during == []
        reference = adapter.run_once(jax.random.key(seed), compile=False)
        assert measured["summary"]["best_fitness"] == reference["summary"]["best_fitness"]


def test_changed_upstream_run_configuration_rebuilds_during_preparation():
    adapter = make_universal([])
    adapter.run_once(jax.random.key(42))
    old_loop = adapter._build_jit_loop()
    adapter.num_generations = 3
    changed = adapter.run_once(jax.random.key(42))
    reference = make_universal([])
    reference.num_generations = 3
    expected = reference.run_once(jax.random.key(42))
    assert old_loop is not adapter._build_jit_loop()
    assert changed["history"] == expected["history"]
    assert changed["summary"] == expected["summary"]


def test_mo_adapter_prepares_scan_without_eager_throwaway_step(monkeypatch):
    import malthusjax.composer.mo_factory as module
    from malthusjax.core.genome.real_genome import RealGenomeConfig
    from malthusjax.engine.mo.mo_engine import MOEngine, MOEngineParams
    from malthusjax.operators.emitters.genetic import GeneticMutationEmitter
    from malthusjax.operators.mutation.real import GaussianMutation
    from tests.engine.mo.test_mo_engine import DummyMOConfig, DummyMOEvaluator

    events = []
    phase = "prepare"
    original = MOEngine.step

    def step(self, state):
        events.append(phase)
        return original(self, state)

    monkeypatch.setattr(MOEngine, "step", step)
    config = RealGenomeConfig(shape=(4,), bounds=(-1.0, 1.0))
    engine = MOEngine(
        engine_params=MOEngineParams(pop_size=4, num_generations=2),
        evaluator=DummyMOEvaluator(config=DummyMOConfig(), data=None),
        emitter=GeneticMutationEmitter(
            _batch_size=4, genome_config=config, mutation=GaussianMutation(mutation_rate=1.0)
        ),
    )
    adapter = module.MOEngineAdapter(engine, config)

    def clock():
        nonlocal phase
        # Native adapters use warmup-start/end followed by execution-start/end.
        phase = next(phases)
        return time.perf_counter()

    phases = iter(("prepare", "between", "execution", "after"))
    with monkeypatch.context() as patch:
        patch.setattr(module, "time", SimpleNamespace(perf_counter=clock))
        result = adapter.run_once(jax.random.key(42))
    assert "execution" not in events
    assert "warmup" in result["timings"]
    assert result["timing_metadata"]["phase"] == "prepared_first_execution"


def test_island_adapter_reports_unsupported_budget_timing_explicitly():
    from flax import struct

    from malthusjax.core.base import BasePopulation
    from malthusjax.engine.base import AbstractEvolutionState
    from malthusjax.engine.island_model.adapter import IslandEngineAdapter

    @struct.dataclass
    class ToyIslands:
        migration_interval: int = struct.field(pytree_node=False, default=2)

        def init_state(self, key):
            pop = BasePopulation(genes=jnp.ones((2, 4, 1)), fitness=jnp.ones((2, 4)))
            return AbstractEvolutionState(
                pop,
                jnp.ones((2, 1)),
                jnp.zeros(2, dtype=jnp.int32),
                jnp.ones(2),
                jax.random.split(key, 2),
            )

        def step(self, state):
            return state.replace(generation=state.generation + 2), jnp.float32(0)

    for generations in (4, 3):
        adapter = IslandEngineAdapter(ToyIslands(), generations=generations)
        result = adapter.run_once(jax.random.key(42))
        if generations == 4:
            assert result["timing_metadata"]["status"] == "valid"
            assert result["timings"]["warmup"] > 0
        else:
            assert "execution" not in result["timings"]
            assert result["timing_metadata"]["status"] == "unavailable"


def test_runner_persists_unavailable_timing_reason():
    from malthusjax.benchmarking.results import RunResult
    from malthusjax.benchmarking.runner import BenchmarkRunner

    adapter = make_universal([], python_loop=True)
    run = BenchmarkRunner(adapter, write_artifacts=False).run([42]).runs[0]
    restored = RunResult.from_dict(run.to_dict())
    assert restored.status == "success"
    assert restored.timing_metadata["status"] == "unavailable"
    assert "execution" not in restored.timings
    assert restored.timing_metadata["reason"]


def test_alternate_qd_catalog_path_prepares_actual_scan(monkeypatch):
    from flax import struct

    import malthusjax.engine.qd.adapter as module
    from malthusjax.engine.base import AbstractEngineParams, AbstractEvolutionState
    from tests.timing_helpers import TimingEngine

    @struct.dataclass
    class State(AbstractEvolutionState):
        repertoire: object

    phase = "prepare"
    events = []
    engine = TimingEngine(
        engine_params=AbstractEngineParams(pop_size=4, num_generations=2),
        trace_hook=lambda: events.append(phase),
    )
    initial = engine.init_state(jax.random.key(42))
    state = State(**vars(initial), repertoire=jnp.float32(0))
    monkeypatch.setattr(module, "_HAS_QDAX", True)
    monkeypatch.setattr(module, "compute_cvt_centroids", lambda **kwargs: jnp.zeros((2, 2)))
    adapter = module.MapElitesEngineAdapter(engine, None, generations=2, num_centroids=2)
    phases = iter(("prepare", "between", "execution", "after"))

    def clock():
        nonlocal phase
        phase = next(phases)
        return time.perf_counter()

    with monkeypatch.context() as patch:
        patch.setattr(module, "time", SimpleNamespace(perf_counter=clock))
        final, history, generations = adapter.run(state.copy())
    assert "execution" not in events
    reference, reference_history, _ = engine.run(state.copy())
    assert generations == 2
    np.testing.assert_allclose(final.best_fitness, reference.best_fitness)
    np.testing.assert_allclose(history.best_fitness, reference_history.best_fitness)
    assert adapter.last_timing_metadata["phase"] == "prepared_first_execution"
    assert "execution" in adapter.last_timings


@pytest.mark.parametrize("backend", ["evosax", "qdax", "tensorneat"])
def test_optional_upstream_factory_runs_the_prepared_universal_path(backend):
    pytest.importorskip(backend)
    if backend == "evosax":
        from malthusjax.composer.evosax_adapter import build_evosax_engine
        from tests.composer.test_evosax_adapter import make_bbob_evaluator

        adapter = build_evosax_engine(
            strategy_name="SimpleGA",
            evaluator=make_bbob_evaluator(num_dims=3),
            pop_size=4,
            generations=2,
            num_dims=3,
        )
    elif backend == "qdax":
        from tests.composer.test_qdax_adapter import TestQDaxAdapter

        adapter = TestQDaxAdapter().make_adapter()
    else:
        from tests.composer.test_tensorneat_adapter import TestTensorNEATAdapterRealIntegration

        adapter = TestTensorNEATAdapterRealIntegration().make_adapter()
    result = adapter.run_once(jax.random.PRNGKey(42))
    assert result["timing_metadata"]["status"] == "valid"
    assert result["timing_metadata"]["phase"] == "prepared_first_execution"
    assert np.isfinite(result["timings"]["execution"])


def test_genetic_finalization_does_not_compile_after_preparation(monkeypatch, caplog):
    import malthusjax.composer.engine_factory as module
    from malthusjax.composer.catalog import OperatorCatalog

    catalog = OperatorCatalog()
    adapter = module.build_engine(
        fitness_evaluator=catalog.get("sphere:dim=3"),
        selection_op=catalog.get("tournament:num_selections=6,tournament_size=2"),
        crossover_op=catalog.get("blend"),
        mutation_op=catalog.get("gaussian:mutation_rate=0.2"),
        pop_size=6,
        generations=3,
        genome_length=3,
    )
    calls = 0
    compilation = []

    def clock():
        nonlocal calls
        calls += 1
        if calls == 3:
            caplog.clear()
        elif calls == 4:
            compilation.extend(
                r.message for r in caplog.records if "Finished XLA compilation" in r.message
            )
        return time.perf_counter()

    with monkeypatch.context() as patch, jax.log_compiles(True):
        patch.setattr(module, "time", SimpleNamespace(perf_counter=clock))
        adapter.run_once(jax.random.key(42))
    assert compilation == []
