"""Unit and integration tests for the MalthusJAX Level 1 core logging subsystem."""

import json
import logging
from pathlib import Path

import pytest

from malthusjax.core.logger import (
    MalthusColorFormatter,
    MalthusJSONFormatter,
    StepLoggingConfig,
    _host_log_nan_anomaly,
    _host_log_step,
    _init_from_env,
    configure_logging,
    get_logger,
    set_log_level,
)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset malthusjax logger configuration after each test."""
    root = logging.getLogger("malthusjax")
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)


def test_logger_namespace():
    """Verify hierarchical logger naming conventions."""
    root_log = get_logger()
    assert root_log.name == "malthusjax"

    engine_log = get_logger("engine")
    assert engine_log.name == "malthusjax.engine"

    full_log = get_logger("malthusjax.operators.mutation")
    assert full_log.name == "malthusjax.operators.mutation"

    empty_log = get_logger("")
    assert empty_log.name == "malthusjax"


def test_default_silence_null_handler(capsys):
    """Verify library silence by default (NullHandler attaches, no stderr pollution)."""
    # Remove any stream handlers for isolation
    root = logging.getLogger("malthusjax")
    root.handlers = [h for h in root.handlers if isinstance(h, logging.NullHandler)]

    logger = get_logger("test_quiet")
    logger.info("This should not appear anywhere")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_set_log_level():
    """Test dynamic level changes across strings and ints."""
    root = logging.getLogger("malthusjax")

    set_log_level("DEBUG")
    assert root.level == logging.DEBUG

    set_log_level("warning")
    assert root.level == logging.WARNING

    set_log_level(logging.ERROR)
    assert root.level == logging.ERROR

    with pytest.raises(ValueError, match="Invalid log level"):
        set_log_level("NOT_A_VALID_LEVEL")


def test_color_formatter():
    """Verify ANSI coloring and plain text formatting options."""
    record = logging.LogRecord(
        name="malthusjax.engine",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Optimization step finished",
        args=(),
        exc_info=None,
    )

    # With colors enabled
    formatter_color = MalthusColorFormatter(use_color=True, show_timestamps=False)
    output_color = formatter_color.format(record)
    assert "\033[32m" in output_color  # Green color code
    assert "[engine]" in output_color
    assert "Optimization step finished" in output_color

    # Without colors
    formatter_plain = MalthusColorFormatter(use_color=False, show_timestamps=False)
    output_plain = formatter_plain.format(record)
    assert "\033[" not in output_plain
    assert "[INFO    ] [engine] Optimization step finished" in output_plain

    # With timestamps
    formatter_ts = MalthusColorFormatter(use_color=False, show_timestamps=True)
    output_ts = formatter_ts.format(record)
    assert "-" in output_ts[:10]  # e.g. YYYY-MM-DD


def test_json_formatter():
    """Verify structured newline-delimited JSON formatting."""
    record = logging.LogRecord(
        name="malthusjax.composer",
        level=logging.WARNING,
        pathname="test.py",
        lineno=42,
        msg="Fallback operator applied",
        args=(),
        exc_info=None,
    )
    record.generation = 50  # Custom attribute
    record.metric_val = 0.999

    formatter = MalthusJSONFormatter()
    raw_json = formatter.format(record)
    data = json.loads(raw_json)

    assert data["level"] == "WARNING"
    assert data["name"] == "malthusjax.composer"
    assert data["message"] == "Fallback operator applied"
    assert "timestamp" in data
    assert data["extra"]["generation"] == 50
    assert data["extra"]["metric_val"] == 0.999


def test_configure_logging_console(capsys):
    """Test configure_logging adding stream handler and printing output."""
    configure_logging(level="DEBUG", format_type="color")
    logger = get_logger("unit_test")

    logger.debug("Debug telemetry message")
    captured = capsys.readouterr()
    assert "Debug telemetry message" in captured.out


def test_configure_logging_file(tmp_path: Path):
    """Test configure_logging writing to file."""
    log_file = tmp_path / "logs" / "test_run.log"
    configure_logging(level="INFO", log_file=log_file)

    logger = get_logger("file_test")
    logger.info("Message written to disk")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Message written to disk" in content
    assert "file_test" in content


def test_step_logging_config():
    """Test StepLoggingConfig state and active status."""
    cfg_active = StepLoggingConfig(log_interval=10, log_nan_watchdog=True)
    assert cfg_active.is_active() is True
    assert cfg_active.log_interval == 10

    cfg_watchdog_only = StepLoggingConfig(log_interval=None, log_nan_watchdog=True)
    assert cfg_watchdog_only.is_active() is True

    cfg_inactive = StepLoggingConfig(log_interval=None, log_nan_watchdog=False)
    assert cfg_inactive.is_active() is False

    cfg_zero_interval = StepLoggingConfig(log_interval=0, log_nan_watchdog=False)
    assert cfg_zero_interval.is_active() is False


def test_host_log_step_callback(caplog):
    """Test host-side step callback prints expected format."""
    with caplog.at_level(logging.INFO):
        # Step with mean fitness
        _host_log_step(10, 42.1234, mean_fitness=50.6789, logger_name="malthusjax.engine.step")
        assert any(
            "[Gen   10] Best Fitness:    42.1234 | Mean Fitness:    50.6789" in record.message
            for record in caplog.records
        )

        # Step without mean fitness
        _host_log_step(20, 10.0, logger_name="malthusjax.engine.step")
        assert any(
            "[Gen   20] Best Fitness:    10.0000" in record.message for record in caplog.records
        )


def test_host_log_nan_anomaly_callback(caplog):
    """Test host-side NaN anomaly callback prints critical alert."""
    with caplog.at_level(logging.CRITICAL):
        _host_log_nan_anomaly(
            15, float("nan"), metric_name="fitness", logger_name="malthusjax.engine.anomaly"
        )
        assert any(
            "[CRITICAL] Non-finite fitness detected at generation 15: nan" in record.message
            for record in caplog.records
        )


def test_env_var_override(monkeypatch):
    """Test automatic configuration via MALTHUSJAX_LOG_LEVEL."""
    monkeypatch.setenv("MALTHUSJAX_LOG_LEVEL", "DEBUG")
    _init_from_env()
    root = logging.getLogger("malthusjax")
    assert root.level == logging.DEBUG
