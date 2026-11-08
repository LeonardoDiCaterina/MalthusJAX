"""Reproduce timing-boundary defects without running optional GPU benchmarks.

Run from the repository: python3 docs/audit_remediation/timing_probe.py
This diagnoses the current implementation; it is not a throughput benchmark.
"""

import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import jax
import jax.numpy as jnp
from flax import struct

import malthusjax.composer.adapters.map_elites_adapter as map_adapter
import malthusjax.engine.base as engine_base
from malthusjax.composer.adapters.base import _block_all_until_ready
from malthusjax.core.base import BasePopulation
from malthusjax.engine.base import AbstractEngine, AbstractEngineParams, AbstractEvolutionState
from malthusjax.engine.qd.map_elites import QDGenerationOutput

events = []
phase = "outside"
seed = None


@struct.dataclass
class TraceEngine(AbstractEngine):
    def init_state(self, key, population, centroids=None):
        return AbstractEvolutionState(
            population=population,
            best_genome=population.genes[0],
            generation=jnp.int32(0),
            best_fitness=jnp.float32(0),
            rng_key=key,
        )

    def step(self, state):
        # Python executes here during JAX tracing, not on every compiled generation.
        events.append({"seed": seed, "phase": phase, "event": "trace_step"})
        next_state = state.replace(
            generation=state.generation + 1, best_fitness=state.best_fitness + 1
        )
        metrics = QDGenerationOutput(
            best_fitness=next_state.best_fitness,
            mean_fitness=next_state.best_fitness,
            std_fitness=jnp.float32(0),
            generation=next_state.generation,
            qd_score=next_state.best_fitness,
            coverage=jnp.float32(100),
        )
        return next_state, metrics


def collect():
    global phase, seed
    params = AbstractEngineParams(pop_size=4, num_generations=3)
    engine = TraceEngine(engine_params=params)
    population = BasePopulation(genes=jnp.ones((4, 2)), fitness=jnp.zeros(4))
    adapter = map_adapter.MapElitesEngineAdapter(
        engine, 4, False, None, initial_population=population
    )
    warm_fn = engine_base._get_evolution_kernel(
        params, compile_jit=True, unroll_num=params.unroll_num
    )
    run_fn = engine_base._get_evolution_kernel(
        params,
        compile_jit=True,
        unroll_num=params.unroll_num,
        return_history=True,
        step_logging=None,
    )
    result = {
        "environment": {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()]},
        "same_warmup_and_execution_function": warm_fn is run_fn,
    }
    compilation_events = []

    class CompileEvents(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if "_evolve_loop" in message and "Finished XLA compilation" in message:
                compilation_events.append({"seed": seed, "phase": phase, "message": message})

    measurements = []
    jax_logger = logging.getLogger("jax")
    handler = CompileEvents()
    for seed in (42, 43):
        clock_calls = 0

        def clock():
            nonlocal clock_calls
            global phase
            clock_calls += 1
            phase = {1: "warmup", 2: "between", 3: "execution", 4: "after"}[clock_calls]
            return time.perf_counter()

        jax_logger.addHandler(handler)
        try:
            with patch.object(jax_logger, "propagate", False), jax.log_compiles(True):
                with patch.object(map_adapter, "time", SimpleNamespace(perf_counter=clock)):
                    observed = adapter.run_once(jax.random.key(seed))
        finally:
            jax_logger.removeHandler(handler)
        measurements.append({"seed": seed, "timings": observed["timings"]})
    result["native_trace_events"] = list(events)
    result["native_xla_compilation_events"] = compilation_events
    result["toy_timings_not_hardware_benchmark"] = measurements

    # Force warmup to fail while providing a successful engine.run result.
    class BadLower:
        def lower(self, *args):
            raise RuntimeError("intentional warmup failure")

    dummy = engine.init_state(jax.random.key(0), population.copy())
    output = QDGenerationOutput(
        best_fitness=jnp.ones(3),
        mean_fitness=jnp.ones(3),
        std_fitness=jnp.zeros(3),
        generation=jnp.arange(1, 4),
        qd_score=jnp.ones(3),
        coverage=jnp.ones(3),
    )
    fake_engine = SimpleNamespace(
        engine_params=params,
        init_state=lambda *args: dummy,
        run=lambda *args, **kwargs: (dummy, output, None),
    )
    fake_adapter = map_adapter.MapElitesEngineAdapter(
        fake_engine, 4, False, None, initial_population=population
    )
    with patch.object(engine_base, "_get_evolution_kernel", return_value=BadLower()):
        try:
            response = fake_adapter.run_once(jax.random.key(1))
            result["warmup_failure_propagated"] = False
            result["warmup_failure_still_reports_execution"] = "execution" in response["timings"]
        except RuntimeError:
            result["warmup_failure_propagated"] = True

    # Inspect actual barrier coverage without assuming CPU behavior proves GPU completion.
    blocked = []

    class Pending:
        def __init__(self, name):
            self.name = name

        def block_until_ready(self):
            blocked.append(self.name)

    final = SimpleNamespace(best_fitness=Pending("best_fitness"), other=Pending("state_other"))
    history = {"history_metric": Pending("history")}
    with patch.object(
        engine_base, "_get_evolution_kernel", return_value=lambda *args: (final, history)
    ):
        engine.run(dummy, time_it=True)
    result["base_run_blocked_leaves"] = list(blocked)
    blocked.clear()
    _block_all_until_ready((final, history))
    result["universal_blocked_leaves"] = list(blocked)

    class BrokenPending:
        def block_until_ready(self):
            raise RuntimeError("intentional synchronization failure")

    try:
        _block_all_until_ready(SimpleNamespace(device_state=BrokenPending()))
        result["universal_struct_sync_failure_propagated"] = False
    except RuntimeError:
        result["universal_struct_sync_failure_propagated"] = True

    @jax.tree_util.register_pytree_node_class
    class SlotsState:
        __slots__ = ("pending",)

        def __init__(self, pending):
            self.pending = pending

        def tree_flatten(self):
            return (self.pending,), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(children[0])

    blocked.clear()
    _block_all_until_ready(SlotsState(Pending("slots_leaf")))
    result["universal_slots_pytree_blocked_leaves"] = list(blocked)
    return result


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
