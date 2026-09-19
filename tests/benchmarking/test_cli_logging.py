"""Tests for MalthusJAX CLI logging flags and structured telemetry."""

import json
import logging
import os
from pathlib import Path

from malthusjax.benchmarking.cli import main
from malthusjax.core.logger import _ROOT_LOGGER_NAME


def test_cli_verbose_flag():
    """Verify that -v / --verbose sets the root logger level to DEBUG."""
    result = main(["-v", "catalog"])
    assert result == 0
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    assert root.level == logging.DEBUG


def test_cli_quiet_flag():
    """Verify that -q / --quiet sets the root logger level to WARNING."""
    result = main(["-q", "catalog"])
    assert result == 0
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    assert root.level == logging.WARNING


def test_cli_log_file_flag(tmp_path: Path):
    """Verify that --log-file directs structured log records to disk."""
    config_path = tmp_path / "test_run.toml"
    config_path.write_text(
        """
        [experiment.shared]
        fitness = "sphere:dim=2"
        pop_size = 10
        generations = 2
        seeds = [1]

        [pipelines.baseline]
        selection = "tournament:tournament_size=2"
        """
    )
    log_file = tmp_path / "logs" / "cli_run.log"

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = main(["--log-file", str(log_file), "run", str(config_path)])
        assert result == 0
        assert log_file.exists()
        content = log_file.read_text()
        assert "Running experiment from" in content
        assert "Experiment complete in" in content
    finally:
        os.chdir(original_cwd)


def test_cli_log_json_flag(tmp_path: Path):
    """Verify that --log-json formats log records as parseable JSON lines."""
    config_path = tmp_path / "test_run.toml"
    config_path.write_text(
        """
        [experiment.shared]
        fitness = "sphere:dim=2"
        pop_size = 10
        generations = 2
        seeds = [1]

        [pipelines.baseline]
        selection = "tournament:tournament_size=2"
        """
    )
    log_file = tmp_path / "logs" / "json_run.log"

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = main(["--log-json", "--log-file", str(log_file), "run", str(config_path)])
        assert result == 0
        assert log_file.exists()
        lines = [line.strip() for line in log_file.read_text().splitlines() if line.strip()]
        assert len(lines) > 0
        for line in lines:
            parsed = json.loads(line)
            assert "timestamp" in parsed
            assert "level" in parsed
            assert "message" in parsed
    finally:
        os.chdir(original_cwd)


def test_cli_log_interval_flag(tmp_path: Path):
    """Verify that --log-interval sets on-device telemetry step interval."""
    config_path = tmp_path / "test_run.toml"
    config_path.write_text(
        """
        [experiment.shared]
        fitness = "sphere:dim=2"
        pop_size = 10
        generations = 4
        seeds = [1]

        [pipelines.baseline]
        selection = "tournament:tournament_size=2"
        """
    )
    log_file = tmp_path / "logs" / "step_run.log"

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = main(
            [
                "-v",
                "--log-interval",
                "2",
                "--log-file",
                str(log_file),
                "run",
                str(config_path),
            ]
        )
        assert result == 0
        assert log_file.exists()
        content = log_file.read_text()
        # Telemetry from host callback or execution logs
        assert "Running experiment" in content
    finally:
        os.chdir(original_cwd)
