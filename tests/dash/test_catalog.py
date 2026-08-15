import json

import pytest

from malthusjax.dash.catalog import DataCatalog


def test_catalog_empty_raises():
    cat = DataCatalog()
    with pytest.raises(ValueError, match=r"Call load\(\) first"):
        _ = cat.data


def test_catalog_load_empty_dir(tmp_path):
    cat = DataCatalog()
    cat.add_source("test", tmp_path)
    cat.load()
    assert cat.data.empty
    assert "pipeline" in cat.data.columns


def test_catalog_load_valid_json(tmp_path):
    # Mock a benchmark JSON artifact
    mock_data = {
        "experiment": "test_exp",
        "config": {"fn_name": "Sphere", "D": 10, "P": 50, "G": 100},
        "pipelines": {
            "PipelineA": [
                {"seed": 1, "best_fitness": 0.01, "duration_seconds": 1.5},
                {"seed": 2, "best_fitness": 0.02, "duration_seconds": 1.6},
            ]
        },
    }

    file_path = tmp_path / "benchmark_results.json"
    with open(file_path, "w") as f:
        json.dump(mock_data, f)

    cat = DataCatalog()
    cat.add_source("src1", tmp_path)
    cat.load()

    df = cat.data
    assert not df.empty
    assert len(df) == 2
    assert (df["pipeline"] == "PipelineA").all()
    assert (df["fn_name"] == "Sphere").all()
    assert (df["D"] == 10).all()
    assert (df["seed"] == [1, 2]).all()
    assert (df["source"] == "src1").all()
