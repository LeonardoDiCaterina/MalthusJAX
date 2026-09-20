"""Tests for Level 3 Engine logging and JIT callback bridge.

Validates that:
1. StepLoggingConfig dispatches device-to-host telemetry via jax.debug.callback.
2. NaN/Inf watchdog dispatches critical anomaly logs when non-finite fitness is detected.
3. When step_logging is None or inactive, callbacks are completely pruned from HLO at trace time.
4. Engine methods (debug_step, run, get_hlo_text) log to malthusjax logger hierarchy.
5. Custom logger channels are respected.
"""

import logging

import chex
import jax.numpy as jnp
import pytest

from malthusjax.core.base import BaseGenome
from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.core.fitness.composable.base import IdentityTransform, ScalarOutput
from malthusjax.core.fitness.composable.environments import BBOBEnv
from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.logger import StepLoggingConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.selection.elite_pool import ElitePoolSelection


@pytest.fixture
def test_engine():
    """Build a lightweight genetic engine for logging tests."""
    params = GeneticEngineParams(pop_size=16, elitism=1, num_generations=4)
    cfg = RealGenomeConfig(shape=(4,), bounds=(-5.0, 5.0))
    evaluator = OptimizationEvaluator(
        env=BBOBEnv.create(fn_name="sphere", num_dims=4),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False),
    )
    sel = ElitePoolSelection(num_selections=16, elite_k=4)
    cross = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
    mut = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)

    return GeneticEngine(
        engine_params=params,
        genome_config=cfg,
        evaluator=evaluator,
        selection=sel,
        crossover=cross,
        mutation=mut,
        enable_progress_bar=False,
    )


class NaNEvaluator(BaseEvaluator):
    """Evaluator that always outputs NaN to test anomaly watchdog."""

    def evaluate(self, genome: BaseGenome) -> chex.Numeric:
        return jnp.nan


def test_step_logging_emits_telemetry(test_engine, caplog):
    """StepLoggingConfig should dispatch telemetry every log_interval generations."""
    state = test_engine.init_state(42)
    step_cfg = StepLoggingConfig(log_interval=2, log_nan_watchdog=False)

    with caplog.at_level(logging.INFO, logger="malthusjax.engine.step"):
        final_state, history, _ = test_engine.run(state, step_logging=step_cfg)

    step_records = [r for r in caplog.records if r.name == "malthusjax.engine.step"]
    assert len(step_records) == 2, (
        f"Expected 2 step log records for 4 gens at interval 2, got {len(step_records)}"
    )
    assert "[Gen    2]" in step_records[0].message
    assert "Best Fitness:" in step_records[0].message
    assert "[Gen    4]" in step_records[1].message


def test_step_logging_configured_on_params(test_engine, caplog):
    """StepLoggingConfig on engine_params should be picked up automatically when run() has no explicit cfg."""
    params_with_logging = test_engine.engine_params.replace(
        step_logging=StepLoggingConfig(log_interval=1, log_nan_watchdog=False)
    )
    engine = test_engine.replace(engine_params=params_with_logging)
    state = engine.init_state(42)

    with caplog.at_level(logging.INFO, logger="malthusjax.engine.step"):
        engine.run(state)

    step_records = [r for r in caplog.records if r.name == "malthusjax.engine.step"]
    assert len(step_records) == 4, f"Expected 4 step logs, got {len(step_records)}"
    for g in [1, 2, 3, 4]:
        assert any(f"[Gen    {g}]" in r.message for r in step_records)


def test_nan_anomaly_watchdog_logs_critical(caplog):
    """NaN watchdog evaluates on-device and dispatches CRITICAL log via jax.debug.callback."""
    params = GeneticEngineParams(pop_size=16, elitism=1, num_generations=2)
    cfg = RealGenomeConfig(shape=(4,), bounds=(-5.0, 5.0))
    evaluator = NaNEvaluator(config=BaseEvaluatorConfig(maximize=False), data=None)

    engine = GeneticEngine(
        engine_params=params,
        genome_config=cfg,
        evaluator=evaluator,
        selection=ElitePoolSelection(num_selections=16, elite_k=4),
        crossover=SimulatedBinaryCrossover(num_offspring=2, eta=15.0),
        mutation=GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5),
        enable_progress_bar=False,
    )

    state = engine.init_state(42)
    step_cfg = StepLoggingConfig(log_interval=1, log_nan_watchdog=True)

    with caplog.at_level(logging.CRITICAL, logger="malthusjax.engine.anomaly"):
        engine.run(state, step_logging=step_cfg)

    anomaly_records = [r for r in caplog.records if r.name == "malthusjax.engine.anomaly"]
    assert len(anomaly_records) >= 1
    assert any("Non-finite best_fitness detected" in r.message for r in anomaly_records)
    assert any(r.levelno == logging.CRITICAL for r in anomaly_records)


def test_zero_overhead_when_step_logging_disabled(test_engine):
    """When step_logging=None, callbacks must be completely absent from lowered HLO."""
    state = test_engine.init_state(42)

    # 1. HLO with no step_logging
    hlo_disabled = test_engine.get_hlo_text(state, optimize=False, print_analysis=False)
    assert "custom-call" not in hlo_disabled and "callback" not in hlo_disabled

    # 2. HLO with active step_logging
    params_active = test_engine.engine_params.replace(
        step_logging=StepLoggingConfig(log_interval=1)
    )
    engine_active = test_engine.replace(engine_params=params_active)
    hlo_active = engine_active.get_hlo_text(state, optimize=False, print_analysis=False)
    assert (
        "custom-call" in hlo_active
        or "callback" in hlo_active
        or "xla.python.callback" in hlo_active
    )


def test_debug_step_logging(test_engine, caplog):
    """debug_step should log phase diagnostics to malthusjax.engine logger."""
    state = test_engine.init_state(42)

    with caplog.at_level(logging.DEBUG, logger="malthusjax.engine"):
        test_engine.debug_step(state)

    messages = [r.message for r in caplog.records]
    assert any("debug_step context:" in m for m in messages)
    assert any("phase 0 allocate entropy:" in m for m in messages)
    assert any("phase 1 selection:" in m for m in messages)
    assert any("debug_step population len:" in m for m in messages)


def test_run_verbose_and_timing_logging(test_engine, caplog):
    """run(verbose=True, time_it=True) should log info messages instead of stdout prints."""
    state = test_engine.init_state(42)

    with caplog.at_level(logging.INFO, logger="malthusjax.engine"):
        test_engine.run(state, verbose=True, time_it=True)

    messages = [r.message for r in caplog.records]
    assert any("Starting evolution:" in m for m in messages)
    assert any("Evolution completed in" in m for m in messages)


def test_get_hlo_text_logging(test_engine, caplog):
    """get_hlo_text(print_analysis=True) should log HLO analysis metrics."""
    state = test_engine.init_state(42)

    with caplog.at_level(logging.INFO, logger="malthusjax.engine"):
        test_engine.get_hlo_text(state, print_analysis=True)

    messages = [r.message for r in caplog.records]
    assert any("HLO Analysis:" in m for m in messages)


def test_custom_logger_channel(test_engine, caplog):
    """StepLoggingConfig can specify a custom logger channel."""
    custom_channel = "my_custom_tracker"
    state = test_engine.init_state(42)
    step_cfg = StepLoggingConfig(log_interval=2, log_nan_watchdog=False, logger_name=custom_channel)

    with caplog.at_level(logging.INFO, logger=custom_channel):
        test_engine.run(state, step_logging=step_cfg)

    custom_records = [r for r in caplog.records if r.name == custom_channel]
    assert len(custom_records) == 2
    assert "[Gen    2]" in custom_records[0].message
