import tempfile
import textwrap

import pytest

from malthusjax.benchmarking.config import (
    AnalysisConfig,
    BenchmarkConfig,
    CartesianGridConfig,
    LHSGridConfig,
    SuiteConfig,
)


def test_cartesian_config_from_toml():
    toml_content = textwrap.dedent("""
    [suite]
    name = "test_cartesian"
    mode = "cartesian"
    output_dir = "results/test"
    num_seeds = 3

    [grid]
    functions = ["sphere", "rastrigin"]
    dims = [2, 10]
    pops = [32, 64]
    gens = [100]

    [analysis]
    reference_pipeline = "base_ga"
    target_metrics = ["best_fitness"]

    [pipelines.base_ga]
    composer.engine.type = "SimpleGA"
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".toml") as f:
        f.write(toml_content)
        f.flush()

        config = BenchmarkConfig.from_toml(f.name)

        assert isinstance(config.suite, SuiteConfig)
        assert config.suite.mode == "cartesian"
        assert config.suite.num_seeds == 3

        assert isinstance(config.grid, CartesianGridConfig)
        assert config.grid.functions == ["sphere", "rastrigin"]
        assert config.grid.dims == [2, 10]

        assert isinstance(config.analysis, AnalysisConfig)
        assert config.analysis.reference_pipeline == "base_ga"
        assert config.analysis.target_metrics == ["best_fitness"]

        assert "base_ga" in config.pipelines

def test_lhs_config_from_toml():
    toml_content = textwrap.dedent("""
    [suite]
    name = "test_lhs"
    mode = "lhs"
    output_dir = "results/lhs"
    num_seeds = 1

    [grid]
    functions = ["rosenbrock"]
    dims_min = 2
    dims_max = 10
    pops_min = 10
    pops_max = 100
    gens_min = 50
    gens_max = 200
    num_samples = 5

    [pipelines.lhs_ga]
    composer.engine.type = "SimpleGA"
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".toml") as f:
        f.write(toml_content)
        f.flush()

        config = BenchmarkConfig.from_toml(f.name)
        assert config.suite.mode == "lhs"
        assert isinstance(config.grid, LHSGridConfig)
        assert config.grid.dims_min == 2
        assert config.grid.num_samples == 5

def test_config_invalid_mode():
    toml_content = textwrap.dedent("""
    [suite]
    name = "test_invalid"
    mode = "random"
    num_seeds = 1
    [grid]
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".toml") as f:
        f.write(toml_content)
        f.flush()
        with pytest.raises(ValueError, match="Unknown suite mode: random"):
            BenchmarkConfig.from_toml(f.name)

def test_config_missing_pipelines():
    toml_content = textwrap.dedent("""
    [suite]
    name = "test_cartesian"
    mode = "cartesian"
    output_dir = "results/test"
    num_seeds = 3

    [grid]
    functions = ["sphere"]
    dims = [2]
    pops = [32]
    gens = [100]
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".toml") as f:
        f.write(toml_content)
        f.flush()
        with pytest.raises(ValueError, match="No pipelines defined"):
            BenchmarkConfig.from_toml(f.name)
