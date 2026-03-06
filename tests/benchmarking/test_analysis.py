"""Unit tests for the benchmark analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from malthusjax.benchmarking import analysis


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
