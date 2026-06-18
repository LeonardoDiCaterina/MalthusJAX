import numpy as np
import pytest

from malthusjax.stats.core import RegressionDataset
from malthusjax.stats.regression import fit_ols


def test_fit_ols_basic():
    # y = 2*x1 - 3*x2 + 1
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, 100)
    x2 = rng.normal(0, 1, 100)
    y = 2*x1 - 3*x2 + 1 + rng.normal(0, 0.1, 100)
    
    dataset = RegressionDataset(
        y=y,
        X={"x1": x1, "x2": x2},
        label="test_ols"
    )
    
    res = fit_ols(dataset)
    assert res.label == "test_ols"
    assert "x1" in res.coefficients
    assert "x2" in res.coefficients
    assert "const" in res.coefficients
    
    assert np.isclose(res.coefficients["x1"], 2.0, atol=0.1)
    assert np.isclose(res.coefficients["x2"], -3.0, atol=0.1)
    assert np.isclose(res.coefficients["const"], 1.0, atol=0.1)
    
    assert res.r_squared > 0.9
    
    # Check robust SEs
    assert "HC0" in res.robust_p_values["x1"]
    assert "HC1" in res.robust_p_values["x1"]
    assert "HC3" in res.robust_p_values["x1"]

def test_fit_ols_empty_X():
    dataset = RegressionDataset(y=np.array([1, 2]), X={})
    with pytest.raises(ValueError, match="cannot be empty"):
        fit_ols(dataset)

def test_fit_ols_empty_y():
    dataset = RegressionDataset(y=np.array([]), X={"x1": np.array([])})
    with pytest.raises(ValueError, match="cannot be empty"):
        fit_ols(dataset)

def test_fit_ols_shape_mismatch():
    dataset = RegressionDataset(y=np.array([1, 2, 3]), X={"x1": np.array([1, 2])})
    with pytest.raises(ValueError, match="does not match target shape"):
        fit_ols(dataset)
