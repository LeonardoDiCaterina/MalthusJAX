"""Tests for Malthus-GP (mgp) CLI commands."""

from pathlib import Path
import numpy as np

from malthus_gp.cli import main as mgp_main
from malthusjax.benchmarking.cli import main as mjax_main


def test_cli_fit_and_formula(tmp_path: Path) -> None:
    """Tests 'mgp fit' and 'mgp formula' subcommands."""
    model_file = tmp_path / "model.toml"
    ret = mgp_main(
        [
            "fit",
            "--estimator",
            "cgp",
            "--data",
            "synthetic",
            "--generations",
            "10",
            "--pop-size",
            "5",
            "--n-cols",
            "8",
            "--output",
            str(model_file),
        ]
    )
    assert ret == 0
    assert model_file.is_file()

    # Test formula
    ret_formula = mgp_main(["formula", "--model", str(model_file)])
    assert ret_formula == 0


def test_cli_predict(tmp_path: Path) -> None:
    """Tests 'mgp predict' subcommand using saved model TOML."""
    model_file = tmp_path / "model.toml"
    preds_file = tmp_path / "predictions.csv"

    # Fit first
    mgp_main(
        [
            "fit",
            "--estimator",
            "cgp",
            "--data",
            "synthetic",
            "--generations",
            "10",
            "--output",
            str(model_file),
        ]
    )

    # Predict
    ret_pred = mgp_main(
        [
            "predict",
            "--model",
            str(model_file),
            "--data",
            "synthetic",
            "--output",
            str(preds_file),
        ]
    )
    assert ret_pred == 0
    assert preds_file.is_file()
    data = np.loadtxt(preds_file, delimiter=",")
    assert len(data) > 0


def test_cli_export_toml(tmp_path: Path) -> None:
    """Tests 'mgp export-toml' subcommand."""
    template_file = tmp_path / "config.toml"
    ret = mgp_main(["export-toml", "--estimator", "dcgp", "--output", str(template_file)])
    assert ret == 0
    assert template_file.is_file()
    content = template_file.read_text(encoding="utf-8")
    assert "DCGPRegressor" in content


def test_mjax_gp_forwarding(tmp_path: Path) -> None:
    """Tests that 'mjax gp ...' forwards seamlessly to mgp."""
    model_file = tmp_path / "mjax_gp_model.toml"
    ret = mjax_main(
        [
            "gp",
            "fit",
            "--estimator",
            "cgp",
            "--generations",
            "5",
            "-o",
            str(model_file),
        ]
    )
    assert ret == 0
    assert model_file.is_file()
