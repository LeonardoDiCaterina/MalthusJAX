import pytest

from malthusjax.composer.config import (
    infer_genome_length,
    load_config,
    load_experiment_config,
    normalize_seeds,
)


def test_load_config(tmp_path):
    toml_content = """
    [pipelines.test_pipeline]
    strategy = "ga"
    pop_size = 100
    """
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(toml_content)

    cfg = load_config(str(toml_file), "test_pipeline")
    assert cfg["strategy"] == "ga"
    assert cfg["pop_size"] == 100


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.toml", "test")


def test_load_config_missing_pipeline(tmp_path):
    toml_content = """
    [pipelines.other_pipeline]
    strategy = "ga"
    """
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(toml_content)

    with pytest.raises(KeyError, match="Pipeline 'test_pipeline' not found"):
        load_config(str(toml_file), "test_pipeline")


def test_load_experiment_config(tmp_path):
    toml_content = """
    [experiment]
    name = "test_exp"
    output_dir = "results/"

    [experiment.shared]
    pop_size = 50
    bounds = [-1.0, 1.0]
    seeds = [1, 2, 3]

    [pipelines.pipeline_a]
    strategy = "ga"
    
    [pipelines.pipeline_b]
    strategy = "es"
    pop_size = 100
    """
    toml_file = tmp_path / "exp.toml"
    toml_file.write_text(toml_content)

    result = load_experiment_config(str(toml_file))

    # Check metadata
    assert result.meta["name"] == "test_exp"
    assert result.meta["output_dir"] == "results/"

    # Check pipelines resolving shared values
    assert "pipeline_a" in result.pipelines
    assert "pipeline_b" in result.pipelines

    # pipeline_a inherits pop_size from shared
    assert result.pipelines["pipeline_a"]["pop_size"] == 50
    assert result.pipelines["pipeline_a"]["bounds"] == (-1.0, 1.0)
    assert result.pipelines["pipeline_a"]["strategy"] == "ga"
    assert result.pipelines["pipeline_a"]["seeds"] == (1, 2, 3)

    # pipeline_b overrides pop_size
    assert result.pipelines["pipeline_b"]["pop_size"] == 100
    assert result.pipelines["pipeline_b"]["bounds"] == (-1.0, 1.0)
    assert result.pipelines["pipeline_b"]["strategy"] == "es"
    assert result.pipelines["pipeline_b"]["seeds"] == (1, 2, 3)

    # Check unpack iteration works
    meta, pipelines = result
    assert meta == result.meta
    assert pipelines == result.pipelines


def test_load_experiment_config_subset(tmp_path):
    toml_content = """
    [experiment]
    name = "test"
    [pipelines.p1]
    [pipelines.p2]
    """
    toml_file = tmp_path / "exp.toml"
    toml_file.write_text(toml_content)

    result = load_experiment_config(str(toml_file), pipelines=["p2"])
    assert "p2" in result.pipelines
    assert "p1" not in result.pipelines


def test_load_experiment_config_missing_subset(tmp_path):
    toml_content = """
    [pipelines.p1]
    """
    toml_file = tmp_path / "exp.toml"
    toml_file.write_text(toml_content)

    with pytest.raises(KeyError, match="Pipelines not found"):
        load_experiment_config(str(toml_file), pipelines=["missing"])


def test_load_experiment_config_no_pipelines(tmp_path):
    toml_content = """
    [experiment]
    name = "test"
    """
    toml_file = tmp_path / "exp.toml"
    toml_file.write_text(toml_content)

    with pytest.raises(KeyError, match="No \\\\[pipelines.\\*\\\\] sections found"):
        load_experiment_config(str(toml_file))


def test_parse_data_section(tmp_path):
    toml_content = """
    [data.dataset1]
    path = "data.csv"
    
    [pipelines.p1]
    """
    toml_file = tmp_path / "exp.toml"
    toml_file.write_text(toml_content)

    result = load_experiment_config(str(toml_file))
    assert "dataset1" in result.data_registry
    assert result.data_registry["dataset1"]["path"] == "data.csv"


def test_normalize_seeds():
    assert normalize_seeds(5) == (1, 2, 3, 4, 5)
    assert normalize_seeds([42, 43]) == (42, 43)

    with pytest.raises(ValueError, match="seeds must be > 0"):
        normalize_seeds(0)

    with pytest.raises(ValueError, match="seeds must not be empty"):
        normalize_seeds([])


def test_infer_genome_length():
    # Explicit kwarg takes precedence
    assert infer_genome_length({"genome_length": 25, "fitness": "sphere:dim=15"}) == 25

    # Then parse from fitness spec
    assert infer_genome_length({"fitness": "sphere:dim=12"}) == 12
    assert infer_genome_length({"fitness": "sphere:num_dims=8"}) == 8

    # Fallback default
    assert infer_genome_length({"fitness": "sphere"}) == 10
