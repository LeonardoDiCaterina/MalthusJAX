import numpy as np
import pandas as pd
import pytest
from malthusjax.stats.dataset_builder import synthesize_regression_dataset, build_design_matrix
from malthusjax.stats.regression import fit_ols
from malthusjax.stats.core import RegressionDataset

def test_duplicates_cannot_expand_join():
    row=dict(fn_name="sphere",seed=1,D=2,P=3,G=4)
    frame=pd.DataFrame([row|dict(pipeline=p) for p in ("a","b") for _ in range(2)])
    with pytest.raises(ValueError, match="identity"):
        synthesize_regression_dataset(frame,"a","b")
    frame["replicate_id"]=[0,1,0,1]
    assert len(synthesize_regression_dataset(frame,"a","b")) == 4

def test_gap_transform_is_declared_and_offset_invariant():
    df=pd.DataFrame(dict(D=[2,3],P=[4,5],G=[6,7],is_treatment=[0,1],fitness=[12.,14.]))
    y,_=build_design_matrix(df,"fitness",transform="log_gap",optimum=10.,epsilon=.01)
    np.testing.assert_allclose(y,np.log([2.01,4.01]))
    raw,_=build_design_matrix(df,"fitness")
    np.testing.assert_array_equal(raw,[12.,14.])

def test_fit_reports_actual_adjusted_r2_and_selected_covariance():
    import statsmodels.api as sm
    x=np.arange(20.)
    y=2*x+np.sin(x)
    result=fit_ols(RegressionDataset(y,{"x":x},covariance="HC3"))
    reference=sm.OLS(y,sm.add_constant(x)).fit(cov_type="HC3")
    assert result.adjusted_r_squared == pytest.approx(reference.rsquared_adj)
    assert result.adjusted_r_squared != result.r_squared
    np.testing.assert_allclose(list(result.standard_errors.values()), reference.bse)
