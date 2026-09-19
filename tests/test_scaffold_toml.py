"""Unit tests for the MalthusJAX TOML scaffolding CLI tool."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import scripts.scaffold_toml as st
from malthusjax.benchmarking.config import BenchmarkConfig
from malthusjax.composer.config import load_experiment_config


@pytest.mark.parametrize("recipe", list(st.RECIPES.keys()))
def test_scaffold_all_recipes(recipe: str):
    """Ensure every registered recipe generates valid, compliant TOML that parses correctly."""
    info = st.RECIPES[recipe]
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        ret = st.main(["-r", recipe, "-o", str(tmp_path), "-n", f"test_{recipe}"])
        assert ret == 0
        assert tmp_path.exists()

        if info["type"] == "composer":
            res = load_experiment_config(str(tmp_path))
            assert res.meta["name"] == f"test_{recipe}"
            assert len(res.pipelines) > 0
            assert "logging" in res.meta
            assert res.meta["logging"]["level"] == "INFO"
        else:
            suite_cfg = BenchmarkConfig.from_toml(str(tmp_path))
            assert suite_cfg.suite.name == f"test_{recipe}"
            assert len(suite_cfg.pipelines) > 0
    finally:
        tmp_path.unlink(missing_ok=True)


def test_scaffold_no_logging_flag():
    """Verify that --no-logging flag omits the [logging] block."""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        ret = st.main(["-r", "single_run", "-o", str(tmp_path), "--no-logging"])
        assert ret == 0
        content = tmp_path.read_text()
        assert "[logging]" not in content

        res = load_experiment_config(str(tmp_path))
        assert "logging" not in res.meta
    finally:
        tmp_path.unlink(missing_ok=True)


def test_list_recipes(capsys):
    """Verify that --list-recipes displays available recipes."""
    ret = st.main(["--list-recipes"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Available MalthusJAX TOML Scaffolding Recipes:" in captured.out
    assert "single_run" in captured.out
    assert "benchmark_cartesian" in captured.out
