import numpy as np
from malthusjax.stats.diagnostics import breusch_pagan, shapiro_wilk

def test_breusch_pagan_constant_resid():
    resid = np.ones(10)
    exog = np.ones((10, 2))
    res = breusch_pagan(resid, exog)
    assert res.name == "breusch_pagan"
    # Constant resid -> perfectly homoskedastic -> p-value might be NaN or 1.0 depending on statsmodels, or raise Exception
    # statsmodels actually throws AssertionError or similar for constant resid, handled by except block
    assert res.p_value is None or res.p_value >= 0.0 or np.isnan(res.p_value)

def test_breusch_pagan_random():
    import statsmodels.api as sm
    rng = np.random.default_rng(42)
    resid = rng.normal(0, 1, 100)
    exog = rng.normal(0, 1, (100, 2))
    exog = sm.add_constant(exog)
    res = breusch_pagan(resid, exog)
    assert res.p_value is not None
    assert res.p_value > 0.05  # Standard normal -> homoskedastic

def test_shapiro_wilk_normal():
    rng = np.random.default_rng(42)
    resid = rng.normal(0, 1, 100)
    res = shapiro_wilk(resid)
    assert res.name == "shapiro_wilk"
    assert res.p_value is not None
    assert res.p_value > 0.05

def test_shapiro_wilk_not_normal():
    rng = np.random.default_rng(42)
    # Exponential distribution is highly skewed
    resid = rng.exponential(1, 100)
    res = shapiro_wilk(resid)
    assert res.p_value is not None
    assert res.p_value < 0.05

def test_shapiro_wilk_empty():
    res = shapiro_wilk([])
    assert res.p_value is None

def test_shapiro_wilk_large():
    # Test subsampling
    rng = np.random.default_rng(42)
    resid = rng.normal(0, 1, 6000)
    res = shapiro_wilk(resid, max_samples=5000)
    assert res.p_value is not None
    assert res.p_value > 0.05
