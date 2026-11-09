"""PT4: preserve observations and require declared comparable measurements."""

import csv
import json
from types import SimpleNamespace

import pytest

from malthusjax.benchmarking.io import read_summary_json, write_experiment_artifacts
from malthusjax.benchmarking.results import ComparisonResult, ExperimentResult, RunResult
from malthusjax.benchmarking.runner import BenchmarkRunner
from malthusjax.core.timing import execution_metadata
from malthusjax.core.timing_records import TimingPolicy, select_timing, timing_values
from tests.timing_records_helpers import record


def run(seed=0, seconds=1.0):
    return RunResult(
        seed=seed,
        status="success",
        metrics={"best_fitness": 2.0},
        timing_record=record(seconds, seed),
        timings={"execution": seconds},
    )


@pytest.mark.parametrize("seeds", [[0, 42, 43], [43, 0, 42], [42, 43, 0]])
def test_seed_order_keeps_every_observation(seeds):
    rows = [run(seed, seed + 1.0) for seed in seeds]
    values, _ = timing_values(rows)
    assert values == [seed + 1.0 for seed in seeds]
    comparison = ComparisonResult(pipelines={"p": ExperimentResult(name="p", runs=rows)})
    assert comparison.timing_data("execution")["p"] == values
    assert comparison.final_metric_data("execution")["p"] == values


@pytest.mark.parametrize(
    "mutation", ["legacy", "version", "phase", "scope", "mode", "environment", "policy", "invalid"]
)
def test_mixed_or_unknown_records_cannot_be_pooled(mutation):
    rows = [run(), run(1)]
    r = rows[1].timing_record
    if mutation == "legacy":
        rows[1].timing_record = {}
    elif mutation == "version":
        r["schema_version"] = 999
    elif mutation == "phase":
        r["target_phase"] = "steady_state"
    elif mutation == "invalid":
        r["trials"][0]["seconds"] = float("nan")
    else:
        key = {"mode": "backend_mode"}.get(mutation, mutation)
        r[key] = "different"
    with pytest.raises(ValueError):
        timing_values(rows)


@pytest.mark.parametrize("value", [None, -1, 0, float("inf"), float("nan")])
def test_invalid_raw_times_remain_visible_and_json_safe(value):
    r = record(value)
    assert r["status"] == "unavailable"
    assert r["trials"][0]["seconds"] is None
    assert r["trials"][0]["raw_value"] == repr(value)
    json.dumps(r, allow_nan=False)
    with pytest.raises(ValueError):
        select_timing({"timing_record": r})


def test_legacy_objective_success_survives_roundtrip():
    r = RunResult.from_dict(run().to_dict() | {"timing_record": None})
    assert r.status == "success"
    assert r.metrics["best_fitness"] == 2.0
    assert r.to_dict()["timing_record"]["status"] == "legacy/unknown"
    with pytest.raises(ValueError):
        select_timing(r)


def test_duplicate_seed_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        timing_values([run(), run()])


class Producer:
    def __init__(self, bad_repeat=False):
        self.calls = 0
        self.bad_repeat = bad_repeat

    def run_once(self, key):
        self.calls += 1
        seconds = [8.0, 4.0, 3.0, 1.0][(self.calls - 1) % 4]
        if self.bad_repeat and self.calls == 3:
            seconds = float("nan")
        return {
            "summary": {"best_fitness": 2},
            "history": [{"generation": 1, "best_fitness": 2}],
            "timings": {"execution": seconds},
            "timing_metadata": execution_metadata()
            | {
                "signature": "fixed",
                "phases": {"initialization": 1.0, "compilation": 1.0, "host_postprocessing": 1.0},
            },
        }


@pytest.mark.parametrize("bad_repeat", [False, True])
def test_runner_preserves_first_preconditioning_and_every_repeat(tmp_path, bad_repeat):
    producer = Producer(bad_repeat)
    policy = TimingPolicy(
        target_phase="prepared_execution_repeat", preconditioning=1, repetitions=2
    )
    result = BenchmarkRunner(producer, write_artifacts=False, timing_policy=policy).run([0])
    row = result.runs[0]
    assert row.status == "success"
    r = row.timing_record
    assert [t["phase"] for t in r["trials"]] == [
        "prepared_first_execution",
        "preconditioning",
        "prepared_execution_repeat",
        "prepared_execution_repeat",
    ]
    assert r["trials"][0]["seconds"] == 8
    assert len(r["trial_results"]) == 3
    if bad_repeat:
        with pytest.raises(ValueError):
            select_timing(row, "prepared_execution_repeat")
    else:
        assert select_timing(row, "prepared_execution_repeat")[0] == 2
        with pytest.raises(ValueError, match="phase mismatch"):
            select_timing(row)
    paths = write_experiment_artifacts(result, tmp_path)
    restored = read_summary_json(paths["summary_json"])
    assert restored.runs[0].timing_record == r
    rows = list(csv.DictReader(paths["timing_csv"].open()))
    assert len(rows) == 4
    assert [int(t["trial_id"]) for t in rows] == [0, 1, 2, 3]
    assert all(json.loads(t["timing_record"]) == r for t in rows)


def test_policy_must_be_declared_before_measurement():
    with pytest.raises(ValueError):
        TimingPolicy(repetitions=2)
    with pytest.raises(ValueError):
        TimingPolicy(target_phase="prepared_execution_repeat")


def test_phase_markers_and_signature_are_separate_from_execution(monkeypatch):
    import time

    import malthusjax.core.timing as timing

    ticks = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    waits = []
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(timing, "block_until_ready", lambda value, **kwargs: waits.append(value))
    capture = timing.TimingCapture()
    capture.initialized({"x": 1}, SimpleNamespace(num_generations=15))
    capture.prepared()
    capture.executed()
    metadata = capture.metadata()
    assert waits == [{"x": 1}]
    assert metadata["phases"] == {
        "initialization": 1.0,
        "compilation": 1.0,
        "host_postprocessing": 1.0,
    }
    assert metadata["signature"]["static_options"]["num_generations"] == "15"
    assert metadata["steady_state_validated"] is False


def test_missing_producer_phase_does_not_fail_objective():
    producer = Producer()
    producer.run_once = lambda key: {
        "summary": {"best_fitness": 3.0},
        "timings": {"execution": 1.0},
        "timing_metadata": execution_metadata(),
    }
    row = BenchmarkRunner(producer, write_artifacts=False).run([0]).runs[0]
    assert row.status == "success"
    assert row.metrics["best_fitness"] == 3
    assert row.timing_record["status"] == "unavailable"
    assert "missing" in row.timing_record["reason"]


def test_analyzer_rejects_wall_time_fallback_and_duplicate_rows(tmp_path):
    from scripts.benchmark_analyzer import parse_global_data

    path = tmp_path / "summary.json"
    row = {"seed": 0, "best_fitness": 2.0, "timings": {"total": 100.0}, "duration_seconds": 200.0}
    data = {"pipelines": {"p": {"per_seed": [row]}}}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="legacy/unknown"):
        parse_global_data(tmp_path)
    assert len(parse_global_data(tmp_path, ["best_fitness"])) == 1
    row["timing_record"] = record(3.0)
    path.write_text(json.dumps(data))
    assert parse_global_data(tmp_path).iloc[0]["execution_time"] == 3.0
    data["pipelines"]["p"]["per_seed"].append(row)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate"):
        parse_global_data(tmp_path)


def test_summary_table_uses_same_median_and_count_as_comparator():
    from malthusjax.core.timing_records import paired_timing_summary

    rows = [run(0, 9.0), run(42, 1.0), run(43, 2.0)]
    exp = ExperimentResult(name="p", runs=rows)
    table = ComparisonResult(pipelines={"p": exp}).summary_table()["p"]["time_execution"]
    summary = paired_timing_summary(rows, rows)
    assert table["mean"] == summary["left_median"] == 2.0
    assert table["count"] == summary["count"] == 3
    assert table["estimator"] == summary["estimator"]
