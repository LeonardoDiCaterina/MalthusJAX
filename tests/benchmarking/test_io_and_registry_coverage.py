"""Tests for io.py, registry.py, and analysis.py edge cases."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from malthusjax.benchmarking.analysis import compute_grouped_kpis, sample_usage, to_dataframe
from malthusjax.benchmarking.io import (
    DataLoader,
    write_histories_csv,
    write_summary_json,
)
from malthusjax.benchmarking.registry import DataRegistry
from malthusjax.benchmarking.results import ExperimentResult, RunResult


def test_data_loader_load_csv_errors(tmp_path: Path):
    missing = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        DataLoader.load_csv(missing)

    corrupt = tmp_path / "corrupt.csv"
    corrupt.write_text("invalid,data\nfoo,bar\n")
    with pytest.raises(ValueError, match="Failed to load CSV"):
        DataLoader.load_csv(corrupt)


def test_data_loader_load_npz_and_load_any(tmp_path: Path):
    missing = tmp_path / "non_existent.npz"
    with pytest.raises(FileNotFoundError):
        DataLoader.load_npz(missing)

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_text("not a valid npz archive")
    with pytest.raises(ValueError, match="Failed to load NPZ"):
        DataLoader.load_npz(corrupt)

    # Valid npz
    valid_npz = tmp_path / "valid.npz"
    np.savez(valid_npz, arr1=np.array([1.0, 2.0]), arr2=np.array([3.0, 4.0]))
    loaded = DataLoader.load_any(valid_npz)
    assert isinstance(loaded, dict)
    assert "arr1" in loaded


def test_data_loader_load_tsplib_and_load_any(tmp_path: Path):
    missing = tmp_path / "non_existent.tsp"
    with pytest.raises(FileNotFoundError):
        DataLoader.load_tsplib(missing)

    # Empty / no coords TSP
    empty_tsp = tmp_path / "empty.tsp"
    empty_tsp.write_text("NAME: empty\nTYPE: TSP\nEOF\n")
    with pytest.raises(ValueError, match="Could not parse valid coordinates"):
        DataLoader.load_tsplib(empty_tsp)

    # Valid TSP
    valid_tsp = tmp_path / "valid.tsp"
    valid_tsp.write_text("NAME: test\nTYPE: TSP\nNODE_COORD_SECTION\n1 0.0 0.0\n2 3.0 4.0\nEOF\n")
    mat = DataLoader.load_any(valid_tsp)
    assert mat.shape == (2, 2)
    assert float(mat[0, 1]) == pytest.approx(5.0)

    # Valid TXT routed to tsplib
    valid_txt = tmp_path / "valid.txt"
    valid_txt.write_text(valid_tsp.read_text())
    mat_txt = DataLoader.load_any(valid_txt)
    assert mat_txt.shape == (2, 2)

    # Unsupported format
    unsupported = tmp_path / "test.unsupported"
    unsupported.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        DataLoader.load_any(unsupported)


def test_write_summary_json_exception_cleanup(tmp_path: Path):
    target = tmp_path / "summary.json"
    exp = ExperimentResult(name="test_exp", runs=[])

    with patch("json.dump", side_effect=RuntimeError("disk failure")):
        with pytest.raises(RuntimeError):
            write_summary_json(exp, target)

    assert not target.exists()
    assert not target.with_suffix(".json.tmp").exists()


def test_write_histories_csv_exception_cleanup(tmp_path: Path):
    target = tmp_path / "histories.csv"
    run = RunResult(
        seed=1, status="success", metrics={"best_fitness": 1.0}, history=[{"gen": 1, "fit": 1.0}]
    )
    exp = ExperimentResult(name="test_exp", runs=[run])

    with patch("csv.DictWriter.writerows", side_effect=RuntimeError("write failed")):
        with pytest.raises(RuntimeError):
            write_histories_csv(exp, target)

    assert not target.exists()
    assert not target.with_suffix(".csv.tmp").exists()


def test_data_registry():
    reg = DataRegistry()
    reg.register("syn1", {"source": "synthetic", "dims": 5})
    resolved_syn = reg.resolve("syn1")
    assert resolved_syn["dims"] == 5

    with pytest.raises(KeyError, match="Data ID 'unknown' not found"):
        reg.resolve("unknown")

    # File source without path
    reg.register("file_no_path", {"source": "file"})
    with pytest.raises(ValueError, match="File source requires 'path'"):
        reg.resolve("file_no_path")

    # Unknown source
    reg.register("invalid_src", {"source": "sql_database"})
    with pytest.raises(ValueError, match="Unknown data source type 'sql_database'"):
        reg.resolve("invalid_src")


def test_analysis_coverage(tmp_path: Path):
    # to_dataframe without pandas
    with patch("malthusjax.benchmarking.analysis.pd", None):
        with pytest.raises(ImportError, match="pandas is required"):
            to_dataframe({})

    # compute_grouped_kpis with missing mean
    data = {
        "benchmarks": [
            {"group": "grp1", "name": "bench1", "stats": {"mean": None}},
            {"group": "grp2", "name": "bench2", "stats": {"mean": 10.0}},
        ]
    }
    kpis = compute_grouped_kpis(data)
    assert ("grp1", "bench1") not in kpis
    assert ("grp2", "bench2") in kpis
    assert kpis[("grp2", "bench2")]["mean"] == 10.0

    # sample_usage with missing path and no benchmark files found
    with patch("glob.glob", return_value=[]):
        with pytest.raises(FileNotFoundError, match="no benchmark files found"):
            sample_usage(None)

    # sample_usage with valid path
    sample_file = tmp_path / "bench.json"
    sample_file.write_text('{"benchmarks": [{"group": "g", "name": "n", "stats": {"mean": 2.5}}]}')
    sample_usage(sample_file)
