"""Tests for MalthusDash CLI."""

import pytest

from malthusjax.dash.cli import main


def test_dash_cli_help(capsys):
    """Test dash CLI help display."""
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    captured = capsys.readouterr()
    assert "MalthusDash" in captured.out


def test_dash_cli_no_args(capsys):
    """Test dash CLI with no arguments prints help."""
    main([])
    captured = capsys.readouterr()
    assert "MalthusDash" in captured.out
