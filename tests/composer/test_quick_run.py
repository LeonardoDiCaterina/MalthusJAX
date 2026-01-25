from malthusjax.composer.composer import Composer


def test_quick_run_basic():
    """Test basic quick_run functionality."""
    composer = Composer.create_default()

    result = composer.quick_run(
        seeds=[1, 2],
        experiment_name="test_quick",
        output_dir=None,
        generations=2,
    )

    assert result.name == "test_quick"
    assert len(result.runs) == 2
    assert all(run.status == "success" for run in result.runs)

    agg = result.aggregated_summary()
    assert "best_fitness" in agg


def test_quick_run_with_output(tmp_path):
    """Test quick_run writes artifacts correctly."""
    composer = Composer()

    result = composer.quick_run(
        seeds=[42],
        experiment_name="output_test",
        output_dir=tmp_path / "test_output",
        generations=1,
    )

    output_dir = tmp_path / "test_output"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "histories_combined.csv").exists()
    assert (output_dir / "seed_0042").is_dir()

    assert "artifact_paths" in result.metadata


def test_quick_run_default_output_dir(tmp_path):
    """Test default output directory creation."""
    import os

    original_cwd = os.getcwd()

    try:
        os.chdir(tmp_path)

        composer = Composer()
        result = composer.quick_run(
            seeds=[1],
            experiment_name="default_dir_test",
            generations=1,
        )

        expected_dir = tmp_path / "results" / "default_dir_test"
        assert expected_dir.exists()
        assert (expected_dir / "summary.json").exists()
        from malthusjax.benchmarking.results import ExperimentResult

        assert isinstance(result, ExperimentResult)

    finally:
        os.chdir(original_cwd)


def test_composer_create_default():
    """Test default composer creation."""
    composer = Composer.create_default()
    assert composer.config["version"] == "0.1"
