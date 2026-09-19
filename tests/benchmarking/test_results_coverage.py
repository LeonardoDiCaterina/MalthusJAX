"""Targeted coverage tests for malthusjax.benchmarking.results."""

import json
from pathlib import Path

import pytest

from malthusjax.benchmarking.results import (
    ComparisonResult,
    ExperimentResult,
    MetaComparison,
    RunResult,
)


def _make_sample_comparison() -> ComparisonResult:
    r1 = RunResult(
        seed=1,
        status="success",
        metrics={"best_fitness": 1.2, "mean_fitness": 2.5, "duration_seconds": 0.1},
        history=[
            {"generation": 0, "best_fitness": 5.0, "mean_fitness": 6.0},
            {"generation": 1, "best_fitness": 1.2, "mean_fitness": 2.5},
        ],
        artifacts={},
        duration_seconds=0.1,
    )
    r2 = RunResult(
        seed=2,
        status="success",
        metrics={"best_fitness": 0.8, "mean_fitness": 1.9, "duration_seconds": 0.2},
        history=[
            {"generation": 0, "best_fitness": 4.5, "mean_fitness": 5.5},
            {"generation": 1, "best_fitness": 0.8, "mean_fitness": 1.9},
        ],
        artifacts={},
        duration_seconds=0.2,
    )
    exp_a = ExperimentResult(name="pipe_a", runs=[r1, r2])

    r3 = RunResult(
        seed=1,
        status="success",
        metrics={"best_fitness": 0.5, "mean_fitness": 1.5, "duration_seconds": 0.15},
        history=[
            {"generation": 0, "best_fitness": 3.0, "mean_fitness": 4.0},
            {"generation": 1, "best_fitness": 0.5, "mean_fitness": 1.5},
        ],
        artifacts={},
        duration_seconds=0.15,
    )
    r4 = RunResult(
        seed=2,
        status="success",
        metrics={"best_fitness": 0.3, "mean_fitness": 1.1, "duration_seconds": 0.18},
        history=[
            {"generation": 0, "best_fitness": 2.8, "mean_fitness": 3.8},
            {"generation": 1, "best_fitness": 0.3, "mean_fitness": 1.1},
        ],
        artifacts={},
        duration_seconds=0.18,
    )
    exp_b = ExperimentResult(name="pipe_b", runs=[r3, r4])

    return ComparisonResult(
        pipelines={"pipe_a": exp_a, "pipe_b": exp_b},
        shared_config={"pop_size": 20},
        negate_map={"pipe_b": True},
    )


def test_run_result_serialization():
    run = RunResult(seed=42, status="success", metrics={"best_fitness": 0.1})
    d = run.to_dict()
    assert d["seed"] == 42
    s = run.to_json()
    run2 = RunResult.from_json(s)
    assert run2.seed == 42
    assert run2.metrics["best_fitness"] == 0.1
    assert run.summary["best_fitness"] == 0.1


def test_experiment_result_serialization_and_history():
    run1 = RunResult(
        seed=1, status="success", metrics={"best_fitness": 0.1}, history=[{"gen": 1, "val": 0.5}]
    )
    run2 = RunResult(
        seed=2, status="success", metrics={"best_fitness": 0.2}, history=[{"gen": 1, "val": 0.6}]
    )
    exp = ExperimentResult(name="test_exp", runs=[run1, run2])

    d = exp.to_dict()
    assert d["name"] == "test_exp"
    s = exp.to_json()
    exp2 = ExperimentResult.from_dict(json.loads(s))
    assert exp2.name == "test_exp"
    assert len(exp2.runs) == 2

    comb = exp.combined_history()
    assert len(comb) == 2
    assert comb[0]["seed"] == 1
    assert comb[1]["seed"] == 2

    summary = exp.aggregated_summary()
    assert "best_fitness" in summary
    assert "mean" in summary["best_fitness"]
    assert exp.canonical_summary is not None
    assert exp.gap_summary() is not None


def test_comparison_result_methods_and_plots(tmp_path: Path):
    comp = _make_sample_comparison()

    assert comp.names == ["pipe_a", "pipe_b"]

    # summary_table
    table = comp.summary_table()
    assert "pipe_a" in table
    assert "pipe_b" in table

    latex_table = comp.summary_table(latex=True)
    assert "\\begin{tabular}" in latex_table
    assert "\\hline" in latex_table

    # normalized_runs
    norm_a = comp.normalized_runs("pipe_a")
    assert len(norm_a) == 2
    norm_b = comp.normalized_runs("pipe_b")
    # pipe_b was negate_map=True, so fitness was flipped
    assert norm_b[0].metrics["best_fitness"] == -0.5

    with pytest.raises(KeyError):
        comp.normalized_runs("unknown_pipe")

    # final_metric_data
    final_data = comp.final_metric_data(metric_key="best_fitness")
    assert len(final_data["pipe_a"]) == 2

    # timing_data
    t_data = comp.timing_data()
    assert "pipe_a" in t_data

    # convergence_data
    c_data = comp.convergence_data()
    assert "pipe_a" in c_data

    # stats
    speedup = comp.statistical_speedup("pipe_a", "pipe_b")
    assert "mean_speedup" in speedup
    delta = comp.statistical_fitness_delta("pipe_a", "pipe_b")
    assert "mean_delta" in delta

    # plot_convergence single seed
    p1 = tmp_path / "conv_single.png"
    comp.plot_convergence(seed_index=1, save_path=p1)
    assert p1.exists()

    # plot_convergence multiple seeds
    p2 = tmp_path / "conv_multi.png"
    comp.plot_convergence(seed_index=[1, 2], save_path=p2)
    assert p2.exists()

    # plot_boxplots
    p3 = tmp_path / "boxplots.png"
    comp.plot_boxplots(metric_key="best_fitness", save_path=p3)
    assert p3.exists()

    # plot_timing_boxplot
    p4 = tmp_path / "timing_boxplots.png"
    comp.plot_timing_boxplot(save_path=p4)
    assert p4.exists()

    # plot_final_metric_boxplot
    p5 = tmp_path / "final_metric_boxplots.png"
    comp.plot_final_metric_boxplot(metric_key="best_fitness", save_path=p5)
    assert p5.exists()


def test_meta_comparison_grid_plots(tmp_path: Path):
    comp1 = _make_sample_comparison()
    comp2 = _make_sample_comparison()
    meta = MetaComparison({"exp1": comp1, "exp2": comp2})

    p1 = tmp_path / "grid_conv.png"
    meta.plot_convergence_grid(save_path=p1)
    assert p1.exists()

    p2 = tmp_path / "grid_box.png"
    meta.plot_boxplot_grid(metric_key="best_fitness", save_path=p2)
    assert p2.exists()

    st = meta.summary_table()
    assert "exp1" in st
    assert "exp2" in st

    # Empty MetaComparison
    empty_meta = MetaComparison({})
    assert empty_meta.plot_convergence_grid() is None
    assert empty_meta.plot_boxplot_grid() is None
