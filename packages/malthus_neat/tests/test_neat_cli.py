"""Tests for mneat standalone CLI and mjax neat command forwarding."""

from __future__ import annotations

import tempfile
from pathlib import Path


from malthus_neat.cli import main as mneat_main
from malthusjax.benchmarking.cli import main as mjax_main


def test_mneat_cli_export_toml() -> None:
    """Tests mneat export-toml subcommand."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_p = Path(tmpdir) / "template.toml"
        ret = mneat_main(["export-toml", "--estimator", "regressor", "--output", str(out_p)])
        assert ret == 0
        assert out_p.exists()
        content = out_p.read_text(encoding="utf-8")
        assert "NEATRegressor" in content


def test_mneat_cli_fit_predict_visualize() -> None:
    """Tests mneat fit, predict, and visualize subcommands end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_p = Path(tmpdir) / "model.toml"
        preds_p = Path(tmpdir) / "preds.csv"
        vis_p = Path(tmpdir) / "vis.svg"
        test_preds_p = Path(tmpdir) / "test_preds.csv"

        # 1. Fit
        ret_fit = mneat_main(
            [
                "fit",
                "--estimator",
                "regressor",
                "--data",
                "synthetic",
                "--pop-size",
                "6",
                "--generations",
                "3",
                "--output",
                str(model_p),
                "--predict-out",
                str(preds_p),
                "--visualize-out",
                str(vis_p),
            ]
        )
        assert ret_fit == 0
        assert model_p.exists()
        assert preds_p.exists()
        assert vis_p.exists()

        # 2. Predict
        ret_pred = mneat_main(
            [
                "predict",
                "--model",
                str(model_p),
                "--data",
                str(preds_p),
                "--output",
                str(test_preds_p),
            ]
        )
        assert ret_pred == 0
        assert test_preds_p.exists()

        # 3. Visualize DOT
        dot_p = Path(tmpdir) / "model.dot"
        ret_vis = mneat_main(
            [
                "visualize",
                "--model",
                str(model_p),
                "--output",
                str(dot_p),
                "--format",
                "dot",
            ]
        )
        assert ret_vis == 0
        assert dot_p.exists()
        assert "digraph NEAT" in dot_p.read_text(encoding="utf-8")


def test_mjax_cli_forwarding() -> None:
    """Tests mjax neat forwarding to mneat."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_p = Path(tmpdir) / "mjax_neat_template.toml"
        ret = mjax_main(
            ["neat", "export-toml", "--estimator", "classifier", "--output", str(out_p)]
        )
        assert ret == 0
        assert out_p.exists()
        assert "NEATClassifier" in out_p.read_text(encoding="utf-8")
