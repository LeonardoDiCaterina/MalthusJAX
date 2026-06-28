import pytest

from malthusjax.benchmarking.results import ComparisonResult, ExperimentResult, RunResult


def test_summary_table_latex():
    # Setup simple ComparisonResult
    run1 = RunResult(seed=1, status="success", metrics={"best_fitness": 10.0, "mean_fitness": 12.0})
    exp1 = ExperimentResult(name="P1", runs=[run1])

    comp = ComparisonResult(pipelines={"P1": exp1}, shared_config={}, initial_population=None)

    latex = comp.summary_table(latex=True)
    assert "\\begin{tabular}" in latex
    assert "P1" in latex
    assert "10" in latex
    assert "\\end{tabular}" in latex
    assert "\\hline" in latex


def test_latex_escape():
    run1 = RunResult(seed=1, status="success", metrics={"score": 1.0})
    exp1 = ExperimentResult(name="P_1 & 2 %", runs=[run1])

    comp = ComparisonResult(pipelines={"P_1 & 2 %": exp1}, shared_config={})
    latex = comp.summary_table(latex=True)

    assert "P\\_1 \\& 2 \\%" in latex


def test_normalized_runs_error():
    comp = ComparisonResult(pipelines={}, shared_config={})
    with pytest.raises(KeyError, match="Unknown pipeline"):
        comp.normalized_runs("NonExistent")


def test_normalized_runs_maximization():
    run1 = RunResult(
        seed=1,
        status="success",
        metrics={"best_fitness": 10.0, "other": 5.0},
        history=[{"generation": 1, "best_fitness": 10.0, "other": 5.0}],
    )
    exp1 = ExperimentResult(name="P1", runs=[run1])

    # By default, negate_map gives False (minimize) which means sign=1.0
    comp = ComparisonResult(pipelines={"P1": exp1}, shared_config={}, negate_map={"P1": True})

    runs = comp.normalized_runs("P1")
    assert runs[0].metrics["best_fitness"] == -10.0
    assert runs[0].metrics["other"] == 5.0

    assert runs[0].history[0]["best_fitness"] == -10.0
    assert runs[0].history[0]["other"] == 5.0


def test_normalized_runs_minimization():
    run1 = RunResult(seed=1, status="success", metrics={"best_fitness": 10.0})
    exp1 = ExperimentResult(name="P1", runs=[run1])

    comp = ComparisonResult(pipelines={"P1": exp1}, shared_config={}, negate_map={"P1": False})

    runs = comp.normalized_runs("P1")
    assert runs[0].metrics["best_fitness"] == 10.0
