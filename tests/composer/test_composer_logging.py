import logging
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from malthusjax.composer.adapters.base import UniversalAdapterEngine
from malthusjax.composer.composer import Composer
from malthusjax.core.logger import StepLoggingConfig, configure_logging, set_log_level


def test_composer_quick_run_with_step_logging(caplog, tmp_path):
    """Verify that quick_run with log_interval emits JIT telemetry and info messages."""
    composer = Composer.create_default()

    with caplog.at_level(logging.DEBUG, logger="malthusjax"):
        result = composer.quick_run(
            fitness="sphere:dim=4",
            pop_size=10,
            generations=6,
            seeds=(42,),
            output_dir=tmp_path / "exp_log",
            log_interval=2,
            log_level="DEBUG",
            serialize_history=False,
        )

    assert len(result.runs) == 1
    messages = [r.getMessage() for r in caplog.records]

    # Verify lifecycle messages
    assert any("Starting quick_run" in m for m in messages)
    assert any("Completed quick_run" in m for m in messages)

    # Verify JIT step telemetry callbacks fired every 2 generations (gen 2, 4, 6)
    step_logs = [m for m in messages if "Generation" in m or "Gen" in m or "generation" in m.lower()]
    assert len(step_logs) >= 3


def test_composer_from_toml_with_logging_section(caplog, tmp_path):
    """Verify that Composer.from_toml parses [logging] section and routes to pipelines."""
    toml_file = tmp_path / "experiment_logging.toml"
    toml_file.write_text(
        textwrap.dedent("""
        [experiment]
        name = "toml_logging_test"

        [logging]
        level = "DEBUG"
        interval = 2
        nan_watchdog = true

        [experiment.shared]
        fitness = "sphere:dim=4"
        pop_size = 10
        generations = 4
        seeds = [42]
        serialize_history = false

        [pipelines.pipeline_alpha]
        crossover = "uniform_real"
        mutation = "gaussian:mutation_rate=0.2,mutation_strength=0.1"
        """)
    )

    with caplog.at_level(logging.DEBUG, logger="malthusjax"):
        res = Composer.from_toml(toml_file)

    assert "pipeline_alpha" in res.pipelines
    messages = [r.getMessage() for r in caplog.records]
    assert any("Comparing" in m for m in messages)


def test_universal_adapter_engine_step_logging_telemetry(caplog):
    """Verify UniversalAdapterEngine JIT loop invokes _host_log_step via callback."""
    # Build minimal mock framework components
    def mock_init(fw_obj, key, params, init_pop):
        return jnp.zeros((10, 2))

    def mock_step(fw_obj, state, key, params, evaluator, eval_trans):
        # returns (next_state, metrics_dict)
        return state + 1.0, {"best_fitness": 0.42, "mean_fitness": 0.84}

    catalog_metric = MagicMock()
    catalog_metric.source = "best_fitness"
    catalog_metric.name = "best_fitness"
    catalog_metric.is_objective_value = True

    step_cfg = StepLoggingConfig(
        log_interval=2,
        log_nan_watchdog=True,
        logger_name="malthusjax.test_adapter",
    )

    adapter = UniversalAdapterEngine(
        framework_obj=MagicMock(),
        framework_params=MagicMock(),
        init_fn=mock_init,
        step_fn=mock_step,
        eval_mode="malthusjax",
        eval_translator=lambda x: x,
        metrics_catalog=[catalog_metric],
        pop_size=10,
        num_generations=6,
        step_logging=step_cfg,
    )

    with caplog.at_level(logging.INFO, logger="malthusjax.test_adapter"):
        res = adapter.run_once(jr.PRNGKey(123))

    assert "history" in res
    messages = [r.getMessage() for r in caplog.records if r.name == "malthusjax.test_adapter"]
    # Generations 2, 4, 6 should trigger step telemetry
    assert len(messages) == 3
    assert all("[Gen " in m for m in messages)


def test_universal_adapter_engine_zero_overhead_when_disabled():
    """Verify that UniversalAdapterEngine runs cleanly with step_logging=None."""
    def mock_init(fw_obj, key, params, init_pop):
        return jnp.zeros((5, 2))

    def mock_step(fw_obj, state, key, params, evaluator, eval_trans):
        return state, {"best_fitness": 1.0}

    catalog_metric = MagicMock()
    catalog_metric.source = "best_fitness"
    catalog_metric.name = "best_fitness"
    catalog_metric.is_objective_value = True

    adapter = UniversalAdapterEngine(
        framework_obj=MagicMock(),
        framework_params=MagicMock(),
        init_fn=mock_init,
        step_fn=mock_step,
        eval_mode="malthusjax",
        eval_translator=lambda x: x,
        metrics_catalog=[catalog_metric],
        pop_size=5,
        num_generations=4,
        step_logging=None,
    )

    res = adapter.run_once(jr.PRNGKey(42))
    assert res["summary"]["best_fitness"] == 1.0
