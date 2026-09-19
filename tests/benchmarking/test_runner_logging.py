"""Unit tests verifying unified logging integration across benchmarking and stats layers."""

from __future__ import annotations

import logging
from unittest.mock import patch

import chex
import pytest

from malthusjax.benchmarking.config import BenchmarkConfig
from malthusjax.benchmarking.runner import BenchmarkRunner, Engine
from malthusjax.benchmarking.sampling import generate_grid
from malthusjax.stats.regression_analyzer import OLSRegressionAnalyzer, RegressionSpec


class SimpleMockEngine(Engine):
    """Minimal engine returning valid result dict."""

    def run_once(self, key: chex.Array):
        return {
            "history": [{"generation": 0, "best_fitness": 1.0}],
            "summary": {"best_fitness": 1.0},
            "timings": {"warmup": 0.01, "execution": 0.05, "total": 0.06},
        }


def test_runner_emits_structured_logging(caplog):
    """Verify BenchmarkRunner emits DEBUG seed start and INFO seed completion traces."""
    runner = BenchmarkRunner(engine=SimpleMockEngine(), experiment_name="test_log_exp", write_artifacts=False)

    with caplog.at_level(logging.DEBUG, logger="malthusjax.benchmarking.runner"):
        result = runner.run(seeds=[42, 43])

    assert len(result.runs) == 2
    # Verify logger namespace and message contents
    records = [r for r in caplog.records if r.name == "malthusjax.benchmarking.runner"]
    assert len(records) >= 4  # 2 debug starts + 2 info completions

    debug_starts = [r for r in records if r.levelno == logging.DEBUG]
    info_dones = [r for r in records if r.levelno == logging.INFO]

    assert any("Seed 1/2 start (seed=42)" in r.message for r in debug_starts)
    assert any("Seed 2/2 start (seed=43)" in r.message for r in debug_starts)
    assert any("Seed 1/2 done status=success" in r.message for r in info_dones)
    assert any("Seed 2/2 done status=success" in r.message for r in info_dones)


def test_sampling_emits_debug_logging(caplog):
    """Verify generate_grid emits debug messages under malthusjax.benchmarking.sampling."""
    toml_dict = {
        "suite": {
            "name": "test_grid",
            "mode": "cartesian",
            "output_dir": "results/test",
            "num_seeds": 1,
        },
        "grid": {
            "functions": ["sphere"],
            "dims": [2],
            "pops": [32],
            "gens": [10],
        },
        "analysis": {"reference_pipeline": "base"},
        "pipelines": {"base": {}},
    }

    import tempfile
    import toml

    with tempfile.NamedTemporaryFile("w", suffix=".toml") as f:
        toml.dump(toml_dict, f)
        f.flush()
        config = BenchmarkConfig.from_toml(f.name)

    with caplog.at_level(logging.DEBUG, logger="malthusjax.benchmarking.sampling"):
        coords = generate_grid(config)

    assert len(coords) == 1
    sampling_records = [r for r in caplog.records if r.name == "malthusjax.benchmarking.sampling"]
    assert len(sampling_records) >= 1
    assert any("Generated 1 Cartesian coordinates" in r.message for r in sampling_records)


def test_regression_analyzer_emits_warning(caplog, tmp_path):
    """Verify OLSRegressionAnalyzer logs warnings to malthusjax.stats.regression on failure."""
    spec = RegressionSpec(dependent_vars=["runtime"])
    analyzer = OLSRegressionAnalyzer(spec)

    import pandas as pd
    mock_df = pd.DataFrame({
        "pipeline": ["target", "target", "ref", "ref"],
        "fn_name": ["sphere", "sphere", "sphere", "sphere"],
        "D": [10, 20, 10, 20],
        "P": [64, 64, 64, 64],
        "G": [100, 100, 100, 100],
        "seed": [1, 1, 1, 1],
        "runtime": [0.1, 0.2, 0.15, 0.25],
    })

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    # Mock fit_ols to simulate failure
    with patch("malthusjax.stats.regression_analyzer.fit_ols", side_effect=RuntimeError("Singular matrix")):
        with caplog.at_level(logging.WARNING, logger="malthusjax.stats.regression"):
            analyzer.analyze_suite(
                df_global=mock_df,
                ref_pipeline="ref",
                target_pipelines=["target"],
                analysis_dir=analysis_dir,
            )

    stats_records = [r for r in caplog.records if r.name == "malthusjax.stats.regression"]
    assert len(stats_records) >= 1
    assert any("Failed to run OLS" in r.message and "Singular matrix" in r.message for r in stats_records)
