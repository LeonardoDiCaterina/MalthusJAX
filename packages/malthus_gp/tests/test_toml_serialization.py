"""Tests for Malthus-GP TOML serialization and deserialization."""

from pathlib import Path
import numpy as np
import pytest

from malthus_gp import (
    BaseGPEstimator,
    CGPRegressor,
    DCGPRegressor,
)


def test_toml_serialization_unfitted_template(tmp_path: Path) -> None:
    """Tests exporting template TOML before fitting."""
    reg = CGPRegressor(n_rows=2, n_cols=25, pop_size=6, max_generations=100)
    toml_str = reg.to_toml()
    assert "[estimator]" in toml_str
    assert 'class = "CGPRegressor"' in toml_str
    assert "[params]" in toml_str
    assert "max_generations = 100" in toml_str

    out_file = tmp_path / "template.toml"
    reg.to_toml(out_file)
    assert out_file.is_file()

    loaded = BaseGPEstimator.from_toml(out_file)
    assert isinstance(loaded, CGPRegressor)
    assert loaded.max_generations == 100
    assert loaded.n_cols == 25


def test_toml_fitted_cgp_roundtrip(tmp_path: Path) -> None:
    """Tests fitted CGPRegressor roundtrip serialization with exact prediction preservation."""
    np.random.seed(42)
    X = np.linspace(-1.0, 1.0, 25).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 0.5

    reg = CGPRegressor(n_rows=1, n_cols=10, pop_size=5, max_generations=25, random_state=42)
    reg.fit(X, y)
    orig_preds = reg.predict(X)
    orig_formula = reg.formula()

    out_file = tmp_path / "fitted_cgp.toml"
    reg.to_toml(out_file)

    loaded = CGPRegressor.from_toml(out_file)
    assert hasattr(loaded, "best_genome_")
    assert hasattr(loaded, "best_fitness_")
    assert loaded.best_fitness_ == pytest.approx(reg.best_fitness_, rel=1e-5)

    loaded_preds = loaded.predict(X)
    np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5, atol=1e-5)
    assert loaded.formula() == orig_formula


def test_toml_fitted_dcgp_roundtrip(tmp_path: Path) -> None:
    """Tests fitted DCGPRegressor roundtrip serialization with continuous weight recovery."""
    np.random.seed(42)
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = np.tanh(X[:, 0])

    dcgp = DCGPRegressor(
        n_rows=1,
        n_cols=6,
        pop_size=5,
        max_generations=15,
        lsmf_epochs=2,
        random_state=42,
    )
    dcgp.fit(X, y)
    orig_preds = dcgp.predict(X)

    out_file = tmp_path / "fitted_dcgp.toml"
    dcgp.to_toml(out_file)

    loaded = BaseGPEstimator.from_toml(out_file)
    assert isinstance(loaded, DCGPRegressor)
    loaded_preds = loaded.predict(X)
    np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5, atol=1e-5)
