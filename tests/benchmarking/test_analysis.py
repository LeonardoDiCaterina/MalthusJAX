"""Unit tests for the benchmark analysis helpers."""

from __future__ import annotations

# import analysis without triggering malthusjax __init__ (evosax dependency)
import importlib.util
import json
from pathlib import Path
from pathlib import Path as _Path

import pytest

spec = importlib.util.spec_from_file_location(
    "malthusjax.benchmarking.analysis",
    _Path("src/malthusjax/benchmarking/analysis.py"),
)
assert spec is not None
analysis = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analysis)


@pytest.fixture(scope="module")
def sample_json_path() -> Path:
    # pick the first file under .benchmarks/*/*.json; tests will skip if none
    base = Path(".benchmarks")
    if not base.exists():
        pytest.skip("no benchmark directory present")
    files = list(base.rglob("*.json"))
    if not files:
        pytest.skip("no benchmark json files found")
    return files[0]


def test_load_and_records(sample_json_path: Path):
    data = analysis.load_benchmark_file(sample_json_path)
    assert isinstance(data, dict)
    assert "benchmarks" in data
    records = analysis.benchmarks_to_records(data)
    assert isinstance(records, list)
    assert records
    # every record must have group/name and mean
    for r in records:
        assert "group" in r
        assert "name" in r
        assert "mean" in r


def test_dataframe_conversion(sample_json_path: Path):
    # pandas conversion only runs if pandas is installed
    data = analysis.load_benchmark_file(sample_json_path)
    if analysis.pd is None:
        pytest.skip("pandas not available")
    df = analysis.to_dataframe(data)
    assert not df.empty
    assert "group" in df.columns
    assert "mean" in df.columns


def test_compute_kpis(sample_json_path: Path):
    data = analysis.load_benchmark_file(sample_json_path)
    kpis = analysis.compute_grouped_kpis(data)
    assert isinstance(kpis, dict)
    # verify that at least one group was summarised
    assert kpis
    for key, metrics in kpis.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert "mean" in metrics and "stddev" in metrics
        assert metrics["count"] >= 1


def test_benchmarks_to_records_and_grouping():
    data = {
        "benchmarks": [
            {
                "group": "group1",
                "name": "test_1",
                "stats": {"mean": 1.0, "stddev": 0.1},
                "extra_info": {"tags": ["fast"]},
            },
            {
                "group": "group1",
                "name": "test_1",
                "stats": {"mean": 1.2, "stddev": 0.15},
                "extra_info": {},
            },
        ]
    }

    records = analysis.benchmarks_to_records(data)
    assert len(records) == 2
    assert records[0]["group"] == "group1"
    assert records[0]["name"] == "test_1"
    assert records[0]["tags"] == ["fast"]

    kpis = analysis.compute_grouped_kpis(data)
    assert set(kpis.keys()) == {("group1", "test_1")}
    summary = kpis[("group1", "test_1")]
    assert summary["mean"] == pytest.approx(1.1)
    assert summary["count"] == 2


def test_to_dataframe_and_plot_group(tmp_path: Path):
    data = {
        "benchmarks": [
            {"group": "grp", "name": "a", "stats": {"mean": 0.5}},
            {"group": "grp", "name": "b", "stats": {"mean": 0.2}},
        ]
    }

    if analysis.pd is None:
        pytest.skip("pandas not available")

    df = analysis.to_dataframe(data)
    assert list(df["name"]) == ["a", "b"]
    assert list(df["mean"]) == [0.5, 0.2]

    matplotlib = pytest.importorskip("matplotlib")
    ax = analysis.plot_group("grp", data)
    assert ax.get_title() == "grp"
    assert len(ax.patches) == 2


def test_sample_usage_with_explicit_path(tmp_path: Path, capsys):
    benchmark_path = tmp_path / "benchmark.json"
    payload = {"benchmarks": [{"group": "grp", "name": "x", "stats": {"mean": 0.1}}]}
    benchmark_path.write_text(json.dumps(payload))

    analysis.sample_usage(benchmark_path)

    captured = capsys.readouterr()
    assert "loaded 1 entries" in captured.out
    assert "grp/x" in captured.out
