"""Targeted coverage tests for malthusjax.benchmarking.cli subcommands and edge cases."""

import json
from pathlib import Path
from unittest.mock import patch

from malthusjax.benchmarking.cli import main


def _create_mock_experiment_dir(
    base_dir: Path, exp_name: str, pipeline_names: list[str], num_seeds: int = 2
) -> Path:
    """Create a mock experiment results directory conforming to the disk schema."""
    exp_dir = base_dir / exp_name
    data_dir = exp_dir / "data"
    meta_dir = exp_dir / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    (meta_dir / "config_snapshot.toml").write_text("# snapshot\n")

    for p_name in pipeline_names:
        pipe_dir = data_dir / f"pipeline_{p_name}"
        pipe_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(1, num_seeds + 1):
            run_data = {
                "seed": seed,
                "status": "success",
                "metrics": {
                    "best_fitness": 0.05 * seed,
                    "mean_fitness": 0.1 * seed,
                    "duration_seconds": 0.5,
                },
                "history": [
                    {"generation": 1, "best_fitness": 0.1 * seed, "mean_fitness": 0.2 * seed},
                    {"generation": 2, "best_fitness": 0.05 * seed, "mean_fitness": 0.1 * seed},
                ],
                "artifacts": {},
                "duration_seconds": 0.5,
                "timings": {"step": 0.4},
                "error": None,
                "created_at": "2026-09-19T12:00:00",
            }
            (pipe_dir / f"seed_{seed}.json").write_text(json.dumps(run_data))
    return exp_dir


def test_cli_analyze_single_pipeline(tmp_path: Path):
    """Test handle_analyze when len(pipe_names) != 2 (standard mean/std tables, csv, md, tex)."""
    exp_dir = _create_mock_experiment_dir(tmp_path, "single_pipe_exp", ["baseline"])
    ret = main(["analyze", str(exp_dir)])
    assert ret == 0

    analysis_dir = exp_dir / "analysis"
    assert (analysis_dir / "baseline_summary.json").exists()
    assert (analysis_dir / "comparison_table.csv").exists()
    assert (analysis_dir / "comparison_table.md").exists()
    assert (analysis_dir / "comparison_table.tex").exists()


def test_cli_analyze_three_pipelines(tmp_path: Path):
    """Test handle_analyze when 3 pipelines are present."""
    exp_dir = _create_mock_experiment_dir(
        tmp_path, "three_pipe_exp", ["pipe_a", "pipe_b", "pipe_c"]
    )
    ret = main(["analyze", str(exp_dir)])
    assert ret == 0

    analysis_dir = exp_dir / "analysis"
    assert (analysis_dir / "pipe_a_summary.json").exists()
    assert (analysis_dir / "comparison_table.csv").exists()


def test_cli_analyze_missing_stats_module(tmp_path: Path, monkeypatch):
    """Test handle_analyze error branch when malthusjax.stats is missing for parity."""
    exp_dir = _create_mock_experiment_dir(tmp_path, "two_pipe_exp", ["pipe_a", "pipe_b"])

    import sys

    monkeypatch.setitem(sys.modules, "malthusjax.stats", None)

    ret = main(["analyze", str(exp_dir)])
    assert ret == 1


def test_cli_plot_subcommand(tmp_path: Path):
    """Test handle_plot generates convergence, boxplots, and timings plots."""
    exp_dir = _create_mock_experiment_dir(tmp_path, "plot_exp", ["pipe_a", "pipe_b"])
    ret = main(["plot", str(exp_dir)])
    assert ret == 0

    plot_dir = exp_dir / "plots"
    assert (plot_dir / "convergence.png").exists()
    assert (plot_dir / "fitness_distribution.png").exists()
    assert (plot_dir / "timings.png").exists()


def test_cli_report_subcommand(tmp_path: Path):
    """Test handle_report sequentially runs analyze and plot."""
    exp_dir = _create_mock_experiment_dir(
        tmp_path, "report_exp", ["pipe_a", "pipe_b"], num_seeds=10
    )
    ret = main(["report", str(exp_dir)])
    assert ret == 0

    assert (exp_dir / "analysis" / "parity_summary.json").exists()
    assert (exp_dir / "plots" / "convergence.png").exists()


def test_cli_aggregate_subcommand(tmp_path: Path):
    """Test handle_aggregate aggregates multiple experiments into a report suite."""
    exp1 = _create_mock_experiment_dir(tmp_path, "exp_sphere", ["pipe_a", "pipe_b"])
    exp2 = _create_mock_experiment_dir(tmp_path, "exp_rastrigin", ["pipe_a", "pipe_b"])
    out_dir = tmp_path / "aggregate_report"

    ret = main(["aggregate", "--out_dir", str(out_dir), str(exp1), str(exp2)])
    assert ret == 0

    assert (out_dir / "aggregate_summary.json").exists()
    assert (out_dir / "plots" / "convergence_grid.png").exists()
    assert (out_dir / "plots" / "fitness_distribution_grid.png").exists()
    assert (out_dir / "plots" / "timings_grid.png").exists()


def test_cli_aggregate_no_valid_dirs(tmp_path: Path):
    """Test handle_aggregate error branch when no valid experiment directories exist."""
    fake_dir = tmp_path / "nonexistent"
    fake_dir.mkdir()
    out_dir = tmp_path / "aggregate_fail"

    ret = main(["aggregate", "--out_dir", str(out_dir), str(fake_dir)])
    assert ret == 1


def test_cli_main_exception_handling():
    """Test main() exception handling when an unhandled exception occurs."""
    with patch(
        "malthusjax.benchmarking.cli.handle_catalog", side_effect=RuntimeError("Test crash")
    ):
        ret = main(["catalog"])
        assert ret == 1
