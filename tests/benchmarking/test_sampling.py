import pytest

from malthusjax.benchmarking.config import (
    AnalysisConfig,
    BenchmarkConfig,
    CartesianGridConfig,
    LHSGridConfig,
    SuiteConfig,
)
from malthusjax.benchmarking.sampling import (
    generate_grid,
)


def test_generate_cartesian_grid():
    config = BenchmarkConfig(
        suite=SuiteConfig(name="test", mode="cartesian", output_dir=".", num_seeds=1),
        grid=CartesianGridConfig(functions=["sphere", "rastrigin"], dims=[10], pops=[100, 200], gens=[50]),
        analysis=AnalysisConfig(reference_pipeline="test"),
        pipelines={"test": {}}
    )

    grid = generate_grid(config)
    assert len(grid) == 4

    expected = [
        {"fn_name": "sphere", "D": 10, "P": 100, "G": 50},
        {"fn_name": "sphere", "D": 10, "P": 200, "G": 50},
        {"fn_name": "rastrigin", "D": 10, "P": 100, "G": 50},
        {"fn_name": "rastrigin", "D": 10, "P": 200, "G": 50},
    ]
    assert grid == expected

def test_generate_lhs_grid():
    config = BenchmarkConfig(
        suite=SuiteConfig(name="test", mode="lhs", output_dir=".", num_seeds=1),
        grid=LHSGridConfig(
            functions=["sphere", "rastrigin"],
            dims_min=2, dims_max=20,
            pops_min=10, pops_max=1000,
            gens_min=50, gens_max=200,
            num_samples=3
        ),
        analysis=AnalysisConfig(reference_pipeline="test"),
        pipelines={"test": {}}
    )

    grid = generate_grid(config)
    # 2 functions * 3 samples = 6 configurations
    assert len(grid) == 6

    # Check bounds
    for entry in grid:
        assert entry["fn_name"] in ["sphere", "rastrigin"]
        assert 2 <= entry["D"] <= 20
        assert 10 <= entry["P"] <= 1000
        assert 50 <= entry["G"] <= 200
        assert entry["lhs_id"].startswith("lhs")

def test_generate_grid_invalid_mode():
    config = BenchmarkConfig(
        suite=SuiteConfig(name="test", mode="invalid_mode", output_dir=".", num_seeds=1), # type: ignore
        grid=CartesianGridConfig(functions=["sphere"], dims=[10], pops=[100], gens=[50]),
        analysis=AnalysisConfig(reference_pipeline="test"),
        pipelines={"test": {}}
    )

    with pytest.raises(ValueError, match="Unknown mode: invalid_mode"):
        generate_grid(config)
