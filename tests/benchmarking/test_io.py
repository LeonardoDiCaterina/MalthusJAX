from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from malthusjax.benchmarking.io import (
    DataLoader,
    ensure_seed_folder,
    read_summary_json,
    write_experiment_artifacts,
    write_histories_csv,
    write_summary_json,
)
from malthusjax.benchmarking.results import ExperimentResult, RunResult


def test_json_roundtrip(tmp_path: Path) -> None:
    """Test JSON write/read round-trip."""
    r1 = RunResult(seed=0, status="success", metrics={"best": 1.0}, history=[{"gen": 0}])
    exp = ExperimentResult(name="test", runs=[r1])

    json_path = tmp_path / "summary.json"
    write_summary_json(exp, json_path)

    assert json_path.exists()
    exp2 = read_summary_json(json_path)
    assert exp2.name == exp.name
    assert len(exp2.runs) == 1
    assert exp2.runs[0].seed == 0


def test_csv_with_seed_column(tmp_path: Path) -> None:
    """Test CSV generation includes seed column."""
    r1 = RunResult(seed=1, status="success", metrics={}, history=[{"gen": 0, "best": 0.9}])
    r2 = RunResult(seed=2, status="success", metrics={}, history=[{"gen": 0, "best": 0.5}])
    exp = ExperimentResult(name="test", runs=[r1, r2])

    csv_path = tmp_path / "histories.csv"
    write_histories_csv(exp, csv_path)

    assert csv_path.exists()
    content = csv_path.read_text()
    assert "seed" in content
    assert "1" in content and "2" in content


def test_seed_folders(tmp_path: Path) -> None:
    """Test seed folder creation."""
    seed_dir = ensure_seed_folder(tmp_path, 42)
    assert seed_dir.exists()
    assert seed_dir.name == "seed_0042"


def test_full_artifacts_write(tmp_path: Path) -> None:
    """Test complete artifact writing."""
    r1 = RunResult(seed=10, status="success", metrics={"best": 1.0})
    exp = ExperimentResult(name="full_test", runs=[r1])

    paths = write_experiment_artifacts(exp, tmp_path)

    assert "summary_json" in paths
    assert "histories_csv" in paths
    assert "seed_0010" in paths

    # Verify files exist
    assert paths["summary_json"].exists()
    assert paths["histories_csv"].exists()
    assert paths["seed_0010"].is_dir()


def test_load_csv(tmp_path: Path) -> None:
    """Test loading a CSV file."""
    csv_path = tmp_path / "test.csv"
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.savetxt(csv_path, data, delimiter=",")

    loaded = DataLoader.load_csv(csv_path)
    assert isinstance(loaded, jnp.ndarray)
    np.testing.assert_allclose(loaded, data)


def test_load_npz(tmp_path: Path) -> None:
    """Test loading an NPZ file."""
    npz_path = tmp_path / "test.npz"
    data1 = np.array([1, 2, 3])
    data2 = np.array([4, 5, 6])
    np.savez(npz_path, arr1=data1, arr2=data2)

    loaded = DataLoader.load_npz(npz_path)
    assert isinstance(loaded, dict)
    assert "arr1" in loaded
    np.testing.assert_allclose(loaded["arr1"], data1)


def test_load_tsplib(tmp_path: Path) -> None:
    """Test loading a basic TSPLib file."""
    tsp_path = tmp_path / "test.tsp"

    content = (
        "NAME: test\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EUC_2D\n"
        "NODE_COORD_SECTION\n1 0.0 0.0\n2 3.0 0.0\n3 0.0 4.0\nEOF\n"
    )
    tsp_path.write_text(content)
    loaded = DataLoader.load_tsplib(tsp_path)
    assert loaded.shape == (3, 3)
    assert jnp.isclose(loaded[0, 1], 3.0)
    assert jnp.isclose(loaded[0, 2], 4.0)
    assert jnp.isclose(loaded[1, 2], 5.0)


def test_load_any(tmp_path: Path) -> None:
    """Test auto-detection in load_any."""
    csv_path = tmp_path / "test.csv"
    np.savetxt(csv_path, [[1.0]], delimiter=",")
    assert isinstance(DataLoader.load_any(csv_path), jnp.ndarray)


def test_missing_file_error() -> None:
    """Test error handling for missing files."""
    with pytest.raises(FileNotFoundError):
        DataLoader.load_csv("nonexistent.csv")
