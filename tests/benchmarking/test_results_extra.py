import pytest
from malthusjax.benchmarking.results import ExperimentResult, RunResult, ComparisonResult

def test_experiment_to_json_from_json():
    r1 = RunResult(
        seed=0,
        status="success",
        metrics={"best_fitness": 1.0, "gap_to_optimum": 1.0},
        history=[{"gen": 0, "best": 1}],
        artifacts={},
    )
    r2 = RunResult(
        seed=1,
        status="success",
        metrics={"best_fitness": 0.5, "gap_to_optimum": 0.5},
        history=[{"gen": 0, "best": 0.5}],
        artifacts={},
    )

    exp = ExperimentResult(name="ex", runs=[r1, r2])
    
    json_str = exp.to_json()
    loaded_exp = ExperimentResult.from_dict(exp.to_dict())
    
    assert loaded_exp.name == exp.name
    assert len(loaded_exp.runs) == 2
    assert loaded_exp.runs[0].seed == r1.seed
    assert loaded_exp.runs[0].metrics == r1.metrics

def test_run_result_from_dict():
    r1 = RunResult(
        seed=0,
        status="success",
        metrics={"best_fitness": 1.0},
        history=[{"gen": 0, "best": 1}],
        artifacts={},
    )
    r1_dict = r1.to_dict()
    r1_new = RunResult.from_dict(r1_dict)
    assert r1_new.seed == r1.seed
    assert r1_new.status == r1.status
    assert r1_new.metrics == r1.metrics

def test_experiment_canonical_summary():
    r1 = RunResult(
        seed=0,
        status="success",
        metrics={"best_fitness": 1.0, "gap_to_optimum": 1.0},
        history=[{"gen": 0, "best": 1}],
        artifacts={},
    )
    exp = ExperimentResult(name="ex", runs=[r1])
    summary = exp.canonical_summary
    assert "best_fitness" in summary
    assert summary["best_fitness"] == 1.0

def test_comparison_statistical_methods():
    r1 = RunResult(
        seed=0,
        status="success",
        metrics={"best_fitness": 1.0},
        history=[{"gen": 0, "best": 1}],
        artifacts={},
        duration_seconds=1.0,
    )
    r2 = RunResult(
        seed=0,
        status="success",
        metrics={"best_fitness": 0.5},
        history=[{"gen": 0, "best": 0.5}],
        artifacts={},
        duration_seconds=2.0,
    )
    exp1 = ExperimentResult(name="ex1", runs=[r1])
    exp2 = ExperimentResult(name="ex2", runs=[r2])
    comp = ComparisonResult(pipelines={"pipe1": exp1, "pipe2": exp2})
    
    speedup = comp.statistical_speedup("pipe2", "pipe1")
    assert speedup["mean_speedup"] == 2.0
    
    delta = comp.statistical_fitness_delta("pipe2", "pipe1", metric_key="best_fitness")
    assert delta["mean_delta"] == 0.5
