"""Contracts for extracting scientific evidence without GPU integrations."""

from types import SimpleNamespace

import numpy as np
import pytest

from malthusjax.benchmarking.empirical import (
    extract_control_runs,
    extract_qd_runs,
    prepare_output_paths,
    timing_comparison,
)
from tests.timing_records_helpers import record


def comparison(values=(-10.0, -2.0), execution=2.0):
    history = [{"best_fitness": v, "coverage": 50.0, "qd_score": -v} for v in values]
    run = SimpleNamespace(
        seed=42,
        status="success",
        error=None,
        history=history,
        timings={"execution": execution},
        metrics={},
        timing_record=record(execution, seed=42),
    )
    return SimpleNamespace(pipelines={"p": SimpleNamespace(runs=[run])})


def test_control_preserves_signed_observations_and_separate_best_so_far():
    result = extract_control_runs(comparison((-10.0, -2.0, -5.0)), "p", 2, 3, 4)
    assert result["mean"] == [-10.0, -2.0, -5.0]
    assert result["best_so_far_mean"] == [-10.0, -2.0, -2.0]
    assert result["per_seed"][0]["best_fitness"] == result["mean"]
    assert result["per_seed"][0]["execution_seconds"] == 2.0
    assert result["throughput"] == 12


@pytest.mark.parametrize("metric", ["best_fitness", "coverage", "qd_score"])
def test_missing_qd_metric_is_not_replaced_with_zero(metric):
    comp = comparison()
    del comp.pipelines["p"].runs[0].history[0][metric]
    with pytest.raises(ValueError, match=metric):
        extract_qd_runs(comp, "p", 2)


@pytest.mark.parametrize("execution", [None, 0, -1, np.nan, np.inf])
def test_invalid_execution_cannot_produce_a_measurement(execution):
    comp = comparison(execution=execution)
    with pytest.raises(ValueError, match="[Tt]iming|execution|duration"):
        extract_control_runs(comp, "p", 2, 2, 4)


def test_total_time_is_not_silently_used_as_execution_time():
    comp = comparison()
    comp.pipelines["p"].runs[0].timings = {"total": 10.0}
    comp.pipelines["p"].runs[0].timing_record = {}
    with pytest.raises(ValueError, match="[Tt]iming|execution|duration"):
        extract_control_runs(comp, "p", 2, 2, 4)


def test_unversioned_execution_metric_is_rejected():
    comp = comparison()
    run = comp.pipelines["p"].runs[0]
    run.timings = None
    run.metrics = {"time_execution": 3.0}
    run.timing_record = {}
    with pytest.raises(ValueError, match="legacy/unknown"):
        extract_control_runs(comp, "p", 1, 2, 1)


@pytest.mark.parametrize("problem", ["failed", "empty", "short", "nan", "duplicate"])
def test_bad_runs_cannot_be_aggregated(problem):
    comp = comparison()
    runs = comp.pipelines["p"].runs
    if problem == "failed":
        runs[0].status = "error"
    elif problem == "empty":
        runs.clear()
    elif problem == "short":
        runs[0].history.pop()
    elif problem == "nan":
        runs[0].history[0]["best_fitness"] = np.nan
    else:
        runs.append(runs[0])
    with pytest.raises(ValueError):
        extract_control_runs(comp, "p", 2, 2, 4)


def test_qd_preserves_backend_native_sign_and_instantaneous_curve():
    result = extract_qd_runs(comparison((-10.0, -2.0)), "p", 2)
    np.testing.assert_array_equal(result["fit_arr"], [[-10.0, -2.0]])
    np.testing.assert_array_equal(result["qd_arr"], [[10.0, 2.0]])
    np.testing.assert_array_equal(result["ms_arr"], [1000.0])
    assert result["per_seed"][0]["seed"] == 42


def test_slower_native_execution_remains_a_slowdown():
    result = timing_comparison([2.0, 4.0, 3.0], [1.0, 1.5, 2.0])
    assert result == {
        "ms_per_gen_malthusjax": 3.0,
        "ms_per_gen_qdax": 1.5,
        "speedup_ratio": 0.5,
        "timing_phase": "prepared_first_execution",
        "timing_estimator": "median_across_seeds",
        "timing_sample_counts": [3, 3],
    }


@pytest.mark.parametrize("samples", [[], [0.0], [np.nan], [-1.0]])
def test_invalid_timing_samples_are_rejected(samples):
    with pytest.raises(ValueError):
        timing_comparison(samples, [1.0])


def test_existing_outputs_are_preserved(tmp_path):
    output = tmp_path / "empirical_curves.json"
    output.write_bytes(b"existing evidence")
    with pytest.raises(FileExistsError):
        prepare_output_paths(output)
    assert output.read_bytes() == b"existing evidence"


def test_new_run_outputs_are_contained_in_the_requested_directory(tmp_path):
    output = tmp_path / "new-run" / "empirical_curves.json"
    tables, runs = prepare_output_paths(output)
    assert tables == output.parent / "tables"
    assert runs == output.parent / "runs"
    assert output.parent.is_dir()
    assert not output.exists()
