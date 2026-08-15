"""Tests for the unified mjax benchmarking CLI."""

from pathlib import Path

from malthusjax.benchmarking.cli import main


def test_cli_help():
    """Test CLI help doesn't crash."""
    try:
        main(["--help"])
    except SystemExit as e:
        # argparse calls sys.exit(0) for --help
        assert e.code == 0


def test_cli_catalog(capsys):
    """Test that the catalog subcommand lists available operators."""
    result = main(["catalog"])
    assert result == 0
    captured = capsys.readouterr()
    assert "MalthusJAX Operator Catalog" in captured.out


def test_cli_run(tmp_path: Path):
    """Test that the run subcommand executes and saves data correctly."""
    config_path = tmp_path / "test_run.toml"
    config_path.write_text(
        """
        [experiment.shared]
        fitness = "sphere:dim=2"
        pop_size = 10
        generations = 2
        seeds = [1]

        [pipelines.baseline]
        selection = "tournament:tournament_size=2"
        """
    )

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = main(["run", str(config_path)])
        assert result == 0

        # Check output structure
        out_dir = Path("results") / "test_run"
        assert out_dir.exists()
        assert (out_dir / "metadata" / "config_snapshot.toml").exists()
        assert (out_dir / "data" / "pipeline_baseline" / "seed_1.json").exists()
    finally:
        os.chdir(original_cwd)


def test_cli_parity_and_analyze(tmp_path: Path):
    """Test that parity execution and subsequent offline analysis work."""
    config_path = tmp_path / "test_parity.toml"
    config_path.write_text(
        """
        [experiment.shared]
        fitness = "sphere:dim=2"
        pop_size = 10
        generations = 2
        seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        [pipelines.malthusjax]
        selection = "tournament:tournament_size=2"

        [pipelines.evosax]
        selection = "tournament:tournament_size=2"
        """
    )

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # 1. Run parity execution
        result = main(["parity", str(config_path)])
        assert result == 0

        out_dir = Path("results") / "test_parity"
        assert (out_dir / "data" / "pipeline_malthusjax").exists()
        assert (out_dir / "data" / "pipeline_evosax").exists()

        # 2. Run offline analysis
        analyze_result = main(["analyze", str(out_dir)])
        assert analyze_result == 0

        # Check that parity JSON and MD were generated
        analysis_dir = out_dir / "analysis"
        assert (analysis_dir / "parity_summary.json").exists()
        assert (analysis_dir / "parity_summary.md").exists()

    finally:
        os.chdir(original_cwd)
