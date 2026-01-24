from malthusjax.benchmarking.io import (
    ensure_seed_folder,
    read_summary_json,
    write_experiment_artifacts,
    write_histories_csv,
    write_summary_json,
)
from malthusjax.benchmarking.results import ExperimentResult, RunResult


def test_json_roundtrip(tmp_path):
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


def test_csv_with_seed_column(tmp_path):
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


def test_seed_folders(tmp_path):
    """Test seed folder creation."""
    seed_dir = ensure_seed_folder(tmp_path, 42)
    assert seed_dir.exists()
    assert seed_dir.name == "seed_0042"


def test_full_artifacts_write(tmp_path):
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
