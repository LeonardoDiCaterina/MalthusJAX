from malthusjax.benchmarking.cli import main


def test_cli_basic():
    """Test CLI with minimal args."""
    result_code = main(["--quiet", "--seeds", "1", "--generations", "1"])
    assert result_code == 0


def test_cli_with_output_dir(tmp_path):
    """Test CLI writes to specified directory."""
    output_dir = tmp_path / "cli_test"

    result_code = main(
        [
            "--quiet",
            "--seeds",
            "42",
            "--name",
            "cli_test_exp",
            "--output-dir",
            str(output_dir),
            "--generations",
            "1",
        ]
    )

    assert result_code == 0
    assert (output_dir / "summary.json").exists()


def test_cli_help():
    """Test CLI help doesn't crash."""
    try:
        main(["--help"])
    except SystemExit as e:
        # argparse calls sys.exit(0) for --help
        assert e.code == 0
